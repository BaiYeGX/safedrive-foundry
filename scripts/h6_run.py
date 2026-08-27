#!/usr/bin/env python3
"""Orchestrate fresh H6 pilot/full collection and v1/v2 acceptance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))
MAPS = ("Town01", "Town03", "Town05")

from scripts.h5_run import _ensure_map  # noqa: E402
from data_pipeline.h6.lineage import (  # noqa: E402
    all_formal_lineages_failed,
    assert_formal_lineage_available,
    frozen_run_lock_identity,
    record_formal_lineage_result,
)
from data_pipeline.h6.run_lock import verify_run_lock  # noqa: E402


def _run(arguments, *, allow_failure: bool = False):
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
            # ``sdf ... --json`` emits pretty-printed JSON while the H6
            # helpers emit a compact final line.  Parse the complete stream
            # first, then fall back to the last JSON line for mixed output.
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
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return payload


def _preflight_with_single_recovery() -> list[dict]:
    """Run the bounded environment gate required before any collection.

    A retryable server-not-ready result gets exactly one ``sim ensure`` and
    exactly one read-only preflight recheck.  Permission/version/tick-owner
    and other non-retryable states stop immediately; this helper never loops
    or turns a failed CARLA check into a synthetic success.
    """

    steps: list[dict] = []
    command = ["scripts/sdf.py", "sim", "preflight", "--json"]
    first = _run(command, allow_failure=True)
    steps.append(first)
    status = str((first.get("result") or {}).get("status") or "")
    if first.get("exit_code") == 0 and status == "READY":
        return steps
    if status != "RETRYABLE_FAILURE":
        raise RuntimeError(json.dumps({"preflight": first}, sort_keys=True))
    ensure = _run(["scripts/sdf.py", "sim", "ensure", "--json"], allow_failure=True)
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
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--world-v3-summary", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--contract", choices=("vla90", "vla75-v2"), default="vla90")
    parser.add_argument("--formal-lineage", choices=("a", "b", "c"), default=None)
    parser.add_argument("--run-lock", type=Path, default=None)
    args = parser.parse_args()
    if args.pilot and args.full:
        raise ValueError("h6_vla75_pilot_and_full_are_separate_runs")
    if args.contract == "vla75-v2" and not args.formal_lineage:
        raise ValueError("h6_vla75_formal_lineage_required")
    if args.contract == "vla75-v2" and args.run_lock is None:
        raise ValueError("h6_vla75_run_lock_required")
    if args.contract == "vla75-v2" and args.run_lock is not None:
        lock = json.loads(args.run_lock.read_text(encoding="utf-8"))
        verification = verify_run_lock(lock, root=ROOT)
        if not verification["valid"]:
            raise ValueError(f"run_lock_invalid:{verification['failures']}")
        if str(lock.get("lineage_id")) != str(args.formal_lineage):
            raise ValueError("h6_vla75_run_lock_lineage_mismatch")
        if args.dataset_id is not None and str(lock.get("dataset_id")) != str(args.dataset_id):
            raise ValueError("h6_vla75_run_lock_dataset_mismatch")
        expected_pairs = 108 if args.full else 12 if args.pilot else None
        if expected_pairs is not None and int(lock.get("matrix_pairs", 0)) != expected_pairs:
            raise ValueError("h6_vla75_run_lock_matrix_scope_mismatch")
        if args.pilot or args.full:
            assert_formal_lineage_available(
                ROOT,
                args.formal_lineage,
                scope="full" if args.full else "pilot",
                run_lock_sha256=str(lock.get("lock_sha256") or ""),
                run_lock_identity=frozen_run_lock_identity(lock),
            )
    dataset_id = args.dataset_id or time.strftime(
        "h6-vla75-%Y%m%dT%H%M%SZ" if args.contract == "vla75-v2" else "h6-vla90-%Y%m%dT%H%M%SZ",
        time.gmtime(),
    )
    trace = {"dataset_id": dataset_id, "started_wall_time_s": time.time(), "steps": []}
    try:
        # The CARLA hard gate is evaluated before readiness/model work so an
        # unavailable server cannot be mistaken for a failed/partial H6 run.
        trace["steps"].extend(_preflight_with_single_recovery())
        scopes = []
        if args.pilot:
            scopes.append("pilot")
        if args.full:
            scopes.append("full")
        readiness_scope = "full" if "full" in scopes or not scopes else "pilot"
        trace["steps"].append(
            _run(
                [
                    "scripts/h6_readiness.py",
                    "--world-v3-summary",
                    str(args.world_v3_summary),
                    "--scope",
                    readiness_scope,
                    "--contract",
                    args.contract,
                ]
                + (["--formal-lineage", args.formal_lineage] if args.formal_lineage else [])
                + (["--run-lock", str(args.run_lock)] if args.run_lock else [])
                + (["--require-run-lock"] if args.contract == "vla75-v2" else [])
            )
        )
        for scope in scopes:
            for map_name in MAPS:
                trace["steps"].append(_ensure_map(map_name))
                trace["steps"].append(
                    _run(
                        [
                            "scripts/h6_collect.py",
                            "--dataset-id",
                            dataset_id,
                            "--map",
                            map_name,
                            "--scope",
                            scope,
                            "--world-v3-summary",
                            str(args.world_v3_summary),
                            "--contract",
                            args.contract,
                        ]
                        + (["--formal-lineage", args.formal_lineage] if args.formal_lineage else [])
                        + (["--run-lock", str(args.run_lock)] if args.run_lock else [])
                    )
                )
            acceptance_step = _run(
                [
                    "scripts/h6_acceptance.py",
                    "--dataset-id",
                    dataset_id,
                    "--scope",
                    scope,
                    "--contract",
                    args.contract,
                ]
                + (["--formal-lineage", args.formal_lineage] if args.formal_lineage else [])
                + (["--run-lock", str(args.run_lock)] if args.run_lock else []),
                allow_failure=args.contract == "vla75-v2",
            )
            trace["steps"].append(acceptance_step)
            if args.contract == "vla75-v2":
                acceptance_result = acceptance_step.get("result")
                if not isinstance(acceptance_result, dict) or "ok" not in acceptance_result:
                    raise RuntimeError("h6_vla75_acceptance_result_missing")
                lineage_state = record_formal_lineage_result(
                    ROOT,
                    args.formal_lineage,
                    scope=scope,
                    passed=bool(acceptance_result.get("ok")),
                    dataset_id=dataset_id,
                    run_lock_sha256=str((lock if args.run_lock else {}).get("lock_sha256") or ""),
                    run_lock_identity=frozen_run_lock_identity(lock if args.run_lock else None),
                    evidence_path=acceptance_result.get("evidence"),
                    evidence_sha256=acceptance_result.get("evidence_sha256"),
                    gate_result=acceptance_result,
                )
                trace["lineage_state"] = lineage_state
                if not bool(acceptance_result.get("ok")):
                    trace["ok"] = False
                    trace["error"] = "h6_vla75_lineage_gate_failed"
                    trace["all_formal_lineages_failed"] = all_formal_lineages_failed(ROOT)
                    break
        if args.accept and not scopes:
            acceptance_step = _run(
                [
                    "scripts/h6_acceptance.py",
                    "--dataset-id",
                    dataset_id,
                    "--contract",
                    args.contract,
                ]
                + (["--formal-lineage", args.formal_lineage] if args.formal_lineage else [])
                + (["--run-lock", str(args.run_lock)] if args.run_lock else []),
                allow_failure=args.contract == "vla75-v2",
            )
            trace["steps"].append(acceptance_step)
            if args.contract == "vla75-v2":
                acceptance_result = acceptance_step.get("result")
                if not isinstance(acceptance_result, dict) or "ok" not in acceptance_result:
                    raise RuntimeError("h6_vla75_acceptance_result_missing")
                if args.run_lock is None:
                    raise RuntimeError("h6_vla75_run_lock_required")
                lock_scope = "full" if int(lock.get("matrix_pairs", 0)) == 108 else "pilot"
                assert_formal_lineage_available(
                    ROOT,
                    args.formal_lineage,
                    scope=lock_scope,
                    run_lock_sha256=str(lock.get("lock_sha256") or ""),
                    run_lock_identity=frozen_run_lock_identity(lock),
                )
                lineage_state = record_formal_lineage_result(
                    ROOT,
                    args.formal_lineage,
                    scope=lock_scope,
                    passed=bool(acceptance_result.get("ok")),
                    dataset_id=dataset_id,
                    run_lock_sha256=str(lock.get("lock_sha256") or ""),
                    run_lock_identity=frozen_run_lock_identity(lock),
                    evidence_path=acceptance_result.get("evidence"),
                    evidence_sha256=acceptance_result.get("evidence_sha256"),
                    gate_result=acceptance_result,
                )
                trace["lineage_state"] = lineage_state
                if not bool(acceptance_result.get("ok")):
                    trace["ok"] = False
                    trace["error"] = "h6_vla75_lineage_gate_failed"
                    trace["all_formal_lineages_failed"] = all_formal_lineages_failed(ROOT)
                else:
                    trace["ok"] = True
            else:
                trace["ok"] = True
        elif "ok" not in trace:
            trace["ok"] = True
    except BaseException as exc:
        trace["ok"] = False
        trace["error"] = f"{type(exc).__name__}:{exc}"
    trace["ended_wall_time_s"] = time.time()
    output = ROOT / "docs/runtime-evidence/h6" / dataset_id / "orchestrator.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(trace, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": trace["ok"], "dataset_id": dataset_id, "evidence": str(output), "error": trace.get("error")}, sort_keys=True))
    return 0 if trace["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
