#!/usr/bin/env python3
"""Run the bounded C2 admission → materialize → smoke → pilot → development sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h2.contracts import stable_sha256  # noqa: E402
from data_pipeline.h6.cora.config import CORA_C2_CONFIG  # noqa: E402


MAPS = tuple(str(item) for item in CORA_C2_CONFIG["maps"])


def _run(arguments: Sequence[str], *, require_success: bool = True) -> dict[str, Any]:
    completed = subprocess.run(
        list(arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = (completed.stdout or "").strip().splitlines()
    payload: dict[str, Any] = {
        "command": list(arguments),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if lines:
        try:
            payload["result"] = json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    if require_success and completed.returncode != 0:
        raise RuntimeError(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return payload


def _python(*arguments: str, require_success: bool = True) -> dict[str, Any]:
    return _run((sys.executable, *arguments), require_success=require_success)


def _restart(map_name: str) -> dict[str, Any]:
    return _python(
        "scripts/sdf.py",
        "sim",
        "restart",
        "--map",
        map_name,
        "--rhi",
        "dx12",
        "--startup-timeout",
        "180",
        "--shutdown-timeout",
        "30",
        "--json",
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _admission(trace: dict[str, Any]) -> None:
    trace["steps"].append(_python("scripts/sdf.py", "doctor"))
    trace["steps"].append(
        _run(
            (
                sys.executable,
                "-c",
                "import json,torch; print(json.dumps({'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))",
            )
        )
    )
    trace["steps"].append(_run(("df", "-B1", str(ROOT))))
    preflight = _python("scripts/sdf.py", "sim", "preflight", "--json", require_success=False)
    trace["steps"].append(preflight)
    result = preflight.get("result") or {}
    if preflight["exit_code"] == 0 and result.get("status") == "READY":
        return
    if result.get("status") == "RETRYABLE_FAILURE":
        trace["steps"].append(
            _python(
                "scripts/sdf.py",
                "sim",
                "ensure",
                "--map",
                "Town01",
                "--rhi",
                "dx12",
                "--startup-timeout",
                "180",
                "--json",
            )
        )
        retry = _python("scripts/sdf.py", "sim", "preflight", "--json")
        trace["steps"].append(retry)
        if (retry.get("result") or {}).get("status") == "READY":
            return
    raise RuntimeError(f"CORA_ADMISSION_NOT_READY:{result}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=str(CORA_C2_CONFIG["dataset_id"]))
    args = parser.parse_args()
    if args.dataset_id != str(CORA_C2_CONFIG["dataset_id"]):
        raise ValueError("cora_run_dataset_not_frozen")
    evidence_dir = ROOT / "docs" / "runtime-evidence" / "h6" / args.dataset_id
    trace: dict[str, Any] = {
        "schema_version": "safedrive.cora.orchestrator.v1",
        "dataset_id": args.dataset_id,
        "started_wall_time_s": time.time(),
        "steps": [],
    }
    try:
        _admission(trace)
        for map_name in MAPS:
            trace["steps"].append(_restart(map_name))
            trace["steps"].append(
                _python("scripts/h6_cora_collect.py", "--dataset-id", args.dataset_id, "materialize-map", "--map", map_name)
            )
        trace["steps"].append(
            _python("scripts/h6_cora_collect.py", "--dataset-id", args.dataset_id, "freeze-manifest")
        )
        for scope in ("smoke", "pilot"):
            for map_name in MAPS:
                trace["steps"].append(_restart(map_name))
                trace["steps"].append(
                    _python(
                        "scripts/h6_cora_collect.py",
                        "--dataset-id",
                        args.dataset_id,
                        "collect-map",
                        "--map",
                        map_name,
                        "--scope",
                        scope,
                    )
                )
        pilot = _python("scripts/h6_cora_audit.py", "--dataset-id", args.dataset_id, "--scope", "pilot")
        trace["steps"].append(pilot)
        if bool((pilot.get("result") or {}).get("passed")):
            for map_name in MAPS:
                trace["steps"].append(_restart(map_name))
                trace["steps"].append(
                    _python(
                        "scripts/h6_cora_collect.py",
                        "--dataset-id",
                        args.dataset_id,
                        "collect-map",
                        "--map",
                        map_name,
                        "--scope",
                        "development",
                    )
                )
            trace["steps"].append(
                _python("scripts/h6_cora_audit.py", "--dataset-id", args.dataset_id, "--scope", "development")
            )
        else:
            trace["pilot_gate_stopped_expansion"] = True
        trace["steps"].append(_python("scripts/h6_cora_finalize.py", "--dataset-id", args.dataset_id))
        trace["ok"] = True
    except BaseException as exc:
        trace["ok"] = False
        trace["error"] = f"{type(exc).__name__}:{exc}"
    trace["ended_wall_time_s"] = time.time()
    trace["orchestrator_sha256"] = stable_sha256(trace)
    output = evidence_dir / "orchestrator.json"
    _atomic_json(output, trace)
    print(
        json.dumps(
            {
                "ok": trace["ok"],
                "dataset_id": args.dataset_id,
                "evidence": str(output),
                "error": trace.get("error"),
                "orchestrator_sha256": trace["orchestrator_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if trace["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
