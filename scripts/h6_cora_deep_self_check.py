"""Run independent C3 artifact verification and adversarial integrity checks.

The checks operate on temporary hard-linked copies of one completed run.  A
hard link keeps large local checkpoints cheap to inspect; every file mutated by
an attack is copied away from the link before it is written.  The completed
run is therefore evidence-only while each attack exercises the public
``h6_cora_sft.py verify`` entry point in a separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _private_copy(path: Path) -> None:
    """Break a hard link before mutating one attack file."""

    temporary = path.with_name(f".{path.name}.private-{os.getpid()}.tmp")
    shutil.copy2(path, temporary)
    os.replace(temporary, path)


def _set_payload_digest(payload: dict[str, Any], field: str, *, rows_only: bool = False) -> None:
    if rows_only:
        payload[field] = _sha(payload.get("rows", []))
        return
    check = dict(payload)
    check.pop(field, None)
    payload[field] = _sha(check)


def _set_resource_digests(payload: dict[str, Any]) -> None:
    digest_payload = {
        key: value for key, value in payload.items() if key not in {"content_sha256", "ledger_sha256"}
    }
    payload["content_sha256"] = _sha(digest_payload)
    payload["ledger_sha256"] = _sha({**digest_payload, "content_sha256": payload["content_sha256"]})


def _copy_run(source: Path) -> tuple[Path, Path]:
    parent = Path(tempfile.mkdtemp(prefix=f"{source.name}.deep-self-check-", dir=str(source.parent)))
    target = parent / source.name
    try:
        shutil.copytree(source, target, copy_function=os.link)
    except OSError:
        # The normal path is hard links.  Keep a functional fallback for a
        # filesystem that does not support them; verification remains the same.
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target, copy_function=shutil.copy2)
    return parent, target


def _verify_process(run_dir: Path, config: dict[str, Any], script: Path) -> dict[str, Any]:
    output_root = run_dir.parent
    command = [
        sys.executable,
        str(script),
        "verify",
        "--run-id",
        str(config["run_id"]),
        "--output-root",
        str(output_root),
        "--release-root",
        str(config["release_root"]),
        "--checkpoint",
        str(config["checkpoint_path"]),
        "--hydra-config",
        str(config["hydra_config_path"]),
        "--internvl-root",
        str(config["internvl_path"]),
        "--reconstruction-manifest",
        str(config["reconstruction_manifest"]),
        "--supplement-manifest",
        str(config["supplement_manifest"]),
        "--device",
        "cpu",
        "--updates",
        str(config["updates"]),
        "--accumulation",
        str(config["gradient_accumulation"]),
        "--max-hours",
        str(config["max_hours"]),
    ]
    environment = dict(os.environ)
    environment["PYTHONWARNINGS"] = "ignore"
    result = subprocess.run(
        command,
        cwd=str(script.parent.parent),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    verification_path = run_dir / "verify.json"
    verification = _read(verification_path) if verification_path.is_file() else {}
    return {
        "returncode": int(result.returncode),
        "status": verification.get("status"),
        "errors": list(verification.get("errors", [])),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _attack(
    source: Path,
    config: dict[str, Any],
    script: Path,
    name: str,
    mutate: Callable[[Path], None],
) -> dict[str, Any]:
    temporary_parent, temporary_run = _copy_run(source)
    try:
        mutate(temporary_run)
        result = _verify_process(temporary_run, config, script)
        return {
            "name": name,
            "rejected": result["returncode"] != 0 and result["status"] == "FAILED" and bool(result["errors"]),
            "returncode": result["returncode"],
            "status": result["status"],
            "errors": result["errors"],
        }
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _mutate_config(run: Path) -> None:
    path = run / "run_config.json"
    payload = _read(path)
    payload["seed"] = int(payload.get("seed", 17)) + 1
    _private_copy(path)
    _set_payload_digest(payload, "content_sha256")
    _write(path, payload)


def _mutate_manifest(run: Path) -> None:
    path = run / "manifest.json"
    payload = _read(path)
    validation = [row for row in payload["samples"] if row.get("split") == "validation"]
    if not validation:
        raise RuntimeError("deep_check_no_validation_sample")
    validation[0]["route_mask"][0] = not bool(validation[0]["route_mask"][0])
    _private_copy(path)
    _set_payload_digest(payload, "manifest_sha256")
    _write(path, payload)


def _mutate_prediction(run: Path) -> None:
    path = run / "m1_predictions.json"
    payload = _read(path)
    payload["rows"][0]["route"][0][0] = float(payload["rows"][0]["route"][0][0]) + 0.5
    _private_copy(path)
    _set_payload_digest(payload, "content_sha256", rows_only=True)
    _write(path, payload)


def _mutate_duplicate_root(run: Path) -> None:
    manifest_path = run / "manifest.json"
    manifest = _read(manifest_path)
    manifest["samples"][1]["root_id"] = manifest["samples"][0]["root_id"]
    _private_copy(manifest_path)
    _set_payload_digest(manifest, "manifest_sha256")
    _write(manifest_path, manifest)


def _mutate_resource_monitor(run: Path) -> None:
    path = run / "resource-ledger.json"
    payload = _read(path)
    train_monitor = payload["gpu_monitoring"]["phases"]["train"]
    if isinstance(train_monitor, list):
        train_monitor = train_monitor[0]
    train_monitor["path"] = "/tmp/c3-deep-self-check-missing-gpu-monitor.jsonl"
    _private_copy(path)
    _set_resource_digests(payload)
    _write(path, payload)


def _mutate_failure_record(run: Path) -> None:
    path = run / "m1_prediction_failures.json"
    payload = _read(path)
    if payload.get("rows"):
        payload["rows"] = payload["rows"][:-1]
    else:
        prediction = _read(run / "m1_predictions.json")["rows"][0]
        prediction = dict(prediction)
        prediction["prediction_exception"] = "deep_self_check_injected_failure"
        payload["rows"] = [prediction]
    payload["row_count"] = len(payload["rows"])
    payload["status"] = "FAILURES_RECORDED" if payload["rows"] else "NO_FAILURES"
    _private_copy(path)
    _set_payload_digest(payload, "content_sha256")
    _write(path, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    script = (run_dir.parents[3] / "scripts" / "h6_cora_sft.py").resolve()
    if not script.is_file():
        script = (Path(__file__).resolve().parent / "h6_cora_sft.py").resolve()
    config = _read(run_dir / "run_config.json")
    base = _verify_process(run_dir, config, script)
    attacks = [
        ("config_seed", _mutate_config),
        ("manifest_mask", _mutate_manifest),
        ("prediction_value", _mutate_prediction),
        ("duplicate_root", _mutate_duplicate_root),
        ("gpu_monitor_path", _mutate_resource_monitor),
        ("failure_record", _mutate_failure_record),
    ]
    attack_results = [_attack(run_dir, config, script, name, mutate) for name, mutate in attacks]
    payload: dict[str, Any] = {
        "schema_version": "safedrive.c3.deep_self_check.v1",
        "run_id": str(config["run_id"]),
        "run_dir": str(run_dir),
        "base_verify": {
            "passed": base["returncode"] == 0 and base["status"] == "VERIFIED" and not base["errors"],
            "returncode": base["returncode"],
            "status": base["status"],
            "errors": base["errors"],
        },
        "attacks": attack_results,
        "attack_count": len(attack_results),
        "all_attacks_rejected": all(item["rejected"] for item in attack_results),
    }
    payload["status"] = "PASSED" if payload["base_verify"]["passed"] and payload["all_attacks_rejected"] else "FAILED"
    payload["content_sha256"] = _sha(payload)
    _write(run_dir / "deep-self-check.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
