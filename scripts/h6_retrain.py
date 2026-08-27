#!/usr/bin/env python3
"""Collect seed-disjoint H6 development outcomes and retrain World v3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.h5_run import _ensure_map  # noqa: E402


def _run(
    arguments: list[str], *, accept_calibration_failure: bool = False, allow_failure: bool = False
) -> dict:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout or ""
    lines = stdout.strip().splitlines()
    result = None
    if stdout.strip():
        try:
            result = json.loads(stdout.strip())
        except json.JSONDecodeError:
            for line in reversed(lines):
                try:
                    result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    payload = {
        "command": [sys.executable, *arguments],
        "exit_code": completed.returncode,
        "result": result,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0 and not accept_calibration_failure and not allow_failure:
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return payload


def _preflight_with_single_recovery() -> list[dict]:
    """Apply the one-ensure/one-recheck CARLA hard-gate policy."""

    steps: list[dict] = []
    command = ["scripts/sdf.py", "sim", "preflight", "--json"]
    first = _run(command, allow_failure=True)
    steps.append(first)
    status = str((first.get("result") or {}).get("status") or "")
    if first.get("exit_code") == 0 and status == "READY":
        return steps
    if status != "RETRYABLE_FAILURE":
        raise RuntimeError(json.dumps({"preflight": first}, sort_keys=True))
    ensure = _run(
        ["scripts/sdf.py", "sim", "ensure", "--json"],
        allow_failure=True,
    )
    steps.append(ensure)
    second = _run(command, allow_failure=True)
    steps.append(second)
    second_status = str((second.get("result") or {}).get("status") or "")
    if second.get("exit_code") != 0 or second_status != "READY":
        raise RuntimeError(
            json.dumps(
                {"preflight": first, "ensure": ensure, "recheck": second},
                sort_keys=True,
            )
        )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--base-world-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--old-data-root", action="append", type=Path, default=[])
    parser.add_argument("--old-split-manifest", type=Path)
    parser.add_argument("--contract", choices=("vla90", "vla75-v2"), default="vla90")
    parser.add_argument("--formal-lineage", choices=("a", "b", "c"), default=None)
    args = parser.parse_args()
    if args.contract == "vla75-v2" and not args.dataset_id.startswith("h6-vla75-"):
        raise SystemExit("h6_vla75_dataset_id_required")
    if args.contract == "vla75-v2" and not args.formal_lineage:
        # Development retraining may use no formal seed; lineage is required
        # only when a formal collector/readiness gate is invoked.
        args.formal_lineage = "a"
    if bool(args.old_data_root) != bool(args.old_split_manifest):
        raise SystemExit("old roots and old split manifest must be supplied together")

    trace = {
        "schema_version": "safedrive.h6.retrain_trace.v1",
        "dataset_id": args.dataset_id,
        "scope": args.scope,
        "started_wall_time_s": time.time(),
        "steps": [],
    }
    try:
        trace["steps"].extend(_preflight_with_single_recovery())
        for map_name in ("Town01", "Town03", "Town05"):
            trace["steps"].append(_ensure_map(map_name))
            trace["steps"].append(
                _run(
                    [
                        "scripts/h6_collect.py",
                        "--dataset-id",
                        args.dataset_id,
                        "--map",
                        map_name,
                        "--scope",
                        args.scope,
                        "--matrix",
                        "training",
                        "--world-v3-summary",
                        str(args.base_world_summary),
                        "--development-exploration",
                        "--contract",
                        args.contract,
                    ]
                    + (["--formal-lineage", args.formal_lineage] if args.formal_lineage else [])
                )
            )
        train = [
            "scripts/train_world_v3.py",
            "--h6-root",
            str(ROOT / "generated" / "h5" / args.dataset_id),
            "--output-dir",
            str(args.output_dir),
            "--seeds",
            args.seeds,
            "--device",
            "cuda",
            "--max-epochs",
            str(args.max_epochs),
            "--patience",
            str(args.patience),
            "--contract",
            args.contract,
        ]
        for root in args.old_data_root:
            train.extend(("--data-root", str(root)))
        if args.old_split_manifest is not None:
            train.extend(("--split-manifest", str(args.old_split_manifest)))
        training_step = _run(train, accept_calibration_failure=True)
        trace["steps"].append(training_step)
        summary = args.output_dir / "training-summary.json"
        if not summary.is_file():
            raise RuntimeError("world_v3_training_summary_missing")
        readiness_arguments = [
            "scripts/h6_readiness.py",
            "--world-v3-summary",
            str(summary),
            "--scope",
            args.scope,
            "--contract",
            args.contract,
        ]
        if args.contract == "vla75-v2":
            # v2 readiness always needs an explicit lineage, even for the
            # development-only audit.  This keeps the config/matrix/hash
            # namespace unambiguous without requiring a formal run-lock yet.
            readiness_arguments.extend(("--formal-lineage", args.formal_lineage))
        readiness = _run(
            readiness_arguments,
            accept_calibration_failure=True,
        )
        trace["steps"].append(readiness)
        trace["ok"] = training_step["exit_code"] == 0 and readiness["exit_code"] == 0
        if not trace["ok"]:
            trace["error"] = "world_v3_calibration_or_readiness_failed"
    except BaseException as exc:
        trace["ok"] = False
        trace["error"] = f"{type(exc).__name__}:{exc}"
    trace["ended_wall_time_s"] = time.time()
    output = ROOT / "docs" / "runtime-evidence" / "h6" / args.dataset_id / "retrain.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(trace, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": trace["ok"],
                "dataset_id": args.dataset_id,
                "evidence": str(output),
                "summary": str(args.output_dir / "training-summary.json"),
                "error": trace.get("error"),
            },
            sort_keys=True,
        )
    )
    return 0 if trace["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
