#!/usr/bin/env python3
"""Collect paired Classic-off vs World-v3 VLA-primary CARLA runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h6.config import (  # noqa: E402
    H6_VLA90_CONFIG_SHA256,
    h6_vla75_config_sha256,
)
from data_pipeline.h6.matrix import (  # noqa: E402
    load_h6_matrix,
    load_h6_training_matrix,
    load_h6_vla75_matrix,
    materialize_h6_scenario,
)
from data_pipeline.h6.runtime import WorldV3Scorer, WorldVLA75Scorer  # noqa: E402
from data_pipeline.h6.run_lock import (  # noqa: E402
    verify_run_lock,
    verify_summary_checkpoints_against_lock,
)
from scripts.h5_collect import collect_map  # noqa: E402


def _load_world_v3(
    summary_path: Path,
    *,
    development_exploration: bool,
    contract: str = "vla90",
    run_lock: dict | None = None,
) -> tuple[WorldV3Scorer, dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = dict(summary.get("calibration") or {})
    if not calibration.get("passed") and not development_exploration:
        raise ValueError("world_v3_dev_calibration_not_passed")
    checkpoints = []
    for item in summary.get("models", ()):
        checkpoint = Path(item["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        checkpoints.append(checkpoint)
    if not checkpoints or not all(path.is_file() for path in checkpoints):
        raise FileNotFoundError("world_v3_checkpoint_missing")
    router_calibration = dict(summary.get("router_calibration") or {})
    if contract == "vla75-v2" and not development_exploration and not router_calibration:
        raise ValueError("vla75_router_calibration_missing")
    if run_lock is not None:
        if contract == "vla75-v2" and not development_exploration:
            checkpoint_binding = verify_summary_checkpoints_against_lock(
                summary.get("models"), run_lock, root=ROOT
            )
            if not checkpoint_binding["valid"]:
                raise ValueError(
                    f"h6_vla75_run_lock_summary_mismatch:{checkpoint_binding['failures']}"
                )
        locked_calibration = run_lock.get("calibration")
        if isinstance(locked_calibration, dict):
            locked_deployment = locked_calibration.get("deployment")
            if contract == "vla75-v2" and not development_exploration and not isinstance(
                locked_deployment, dict
            ):
                raise ValueError("h6_vla75_run_lock_deployment_calibration_missing")
            if isinstance(locked_deployment, dict):
                # A formal run must use exactly the deployment thresholds and
                # temperatures that were hashed into its immutable lock.
                summary_deployment = dict(calibration)
                if stable_sha256(summary_deployment) != stable_sha256(locked_deployment):
                    raise ValueError("h6_vla75_run_lock_calibration_mismatch")
                calibration = dict(locked_deployment)
            locked_router = locked_calibration.get("router")
            if contract == "vla75-v2" and not development_exploration and not isinstance(
                locked_router, dict
            ):
                raise ValueError("h6_vla75_run_lock_router_calibration_missing")
            if isinstance(locked_router, dict):
                if router_calibration and stable_sha256(router_calibration) != stable_sha256(locked_router):
                    raise ValueError("h6_vla75_run_lock_router_calibration_mismatch")
                router_calibration = dict(locked_router)
            locked_temperatures = locked_calibration.get("temperatures")
            if contract == "vla75-v2" and not development_exploration and not isinstance(
                locked_temperatures, dict
            ):
                raise ValueError("h6_vla75_run_lock_temperature_calibration_missing")
            if isinstance(locked_temperatures, dict):
                summary_temperatures = dict(summary.get("temperature_calibration") or {})
                if stable_sha256(summary_temperatures) != stable_sha256(locked_temperatures):
                    raise ValueError("h6_vla75_run_lock_temperature_calibration_mismatch")
    runtime_calibration = (
        calibration
        if calibration.get("passed")
        else {"trust_threshold": 0.0, "risk_ceiling": 1.0}
    )
    scorer_type = WorldVLA75Scorer if contract == "vla75-v2" else WorldV3Scorer
    try:
        scorer = scorer_type.from_checkpoints(
            checkpoints,
            device="cuda",
            calibration=runtime_calibration,
        )
    except ValueError as exc:
        # A development exploration may intentionally start from the last
        # v1/v3 summary to collect fresh seed-89/97 closed-loop outcomes
        # before the first v75 checkpoint exists.  Keep this compatibility
        # escape hatch strictly development-only; formal v75 collection must
        # load the independent 14-output schema and therefore re-raises.
        if not (
            contract == "vla75-v2"
            and development_exploration
            and str(exc) == "world_vla75_checkpoint_schema_mismatch"
        ):
            raise
        scorer = WorldV3Scorer.from_checkpoints(
            checkpoints,
            device="cuda",
            calibration=runtime_calibration,
        )
    if development_exploration:
        # Explicitly nonformal behavior used only to collect VLA outcomes for
        # retraining. Formal calibration/acceptance rejects this config hash.
        scorer.development_force_vla = True
    scorer.router_calibration = router_calibration
    return scorer, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--world-v3-summary", type=Path, required=True)
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--development-exploration", action="store_true")
    parser.add_argument(
        "--contract",
        choices=("vla90", "vla75-v2"),
        default="vla90",
        help="v1 compatibility or the new VLA75 v2 contract",
    )
    parser.add_argument("--run-lock", type=Path, default=None)
    parser.add_argument(
        "--formal-lineage",
        choices=("a", "b", "c"),
        default=None,
        help="required for vla75-v2 formal collection; never used for training",
    )
    parser.add_argument(
        "--matrix",
        choices=("training", "acceptance"),
        default="acceptance",
        help="training is development-only; acceptance is held out for formal evidence",
    )
    args = parser.parse_args()
    if args.contract == "vla75-v2" and args.matrix != "training" and not args.formal_lineage:
        raise ValueError("h6_vla75_formal_lineage_required")
    if args.contract == "vla75-v2" and not args.dataset_id.startswith("h6-vla75-"):
        raise ValueError("h6_vla75_dataset_id_required")
    run_lock = None
    if args.run_lock is not None:
        run_lock = json.loads(args.run_lock.read_text(encoding="utf-8"))
        verification = verify_run_lock(run_lock, root=ROOT)
        if not verification["valid"]:
            raise ValueError(f"run_lock_invalid:{verification['failures']}")
        if args.contract == "vla75-v2":
            if str(run_lock.get("dataset_id")) != str(args.dataset_id):
                raise ValueError("h6_vla75_run_lock_dataset_mismatch")
            if args.formal_lineage and str(run_lock.get("lineage_id")) != str(args.formal_lineage):
                raise ValueError("h6_vla75_run_lock_lineage_mismatch")
            expected_pairs = 108 if args.scope == "full" else 12
            if int(run_lock.get("matrix_pairs", 0)) != expected_pairs:
                raise ValueError("h6_vla75_run_lock_matrix_scope_mismatch")
    if args.contract == "vla75-v2" and args.matrix != "training" and run_lock is None:
        raise ValueError("h6_vla75_run_lock_required")
    scorer, summary = _load_world_v3(
        args.world_v3_summary,
        development_exploration=args.development_exploration,
        contract=args.contract,
        run_lock=run_lock,
    )
    if args.matrix == "training" and not args.development_exploration:
        raise ValueError("h6_training_matrix_is_development_only")
    base_config_sha = (
        h6_vla75_config_sha256(args.formal_lineage or "a")
        if args.contract == "vla75-v2"
        else H6_VLA90_CONFIG_SHA256
    )
    config_sha = (
        stable_sha256(
            {
                "base": base_config_sha,
                "development_exploration": True,
                "world_summary": summary.get("evidence_sha256"),
                "matrix": args.matrix,
                "contract": args.contract,
                "formal_lineage": args.formal_lineage,
            }
        )
        if args.development_exploration
        else base_config_sha
    )
    if args.contract == "vla75-v2":
        scenarios = (
            load_h6_training_matrix(full=args.scope == "full")
            if args.matrix == "training"
            else load_h6_vla75_matrix(args.formal_lineage, full=args.scope == "full")
        )
    else:
        scenarios = (
            load_h6_training_matrix(full=args.scope == "full")
            if args.matrix == "training"
            else load_h6_matrix(full=args.scope == "full")
        )
    result = collect_map(
        args.dataset_id,
        args.map,
        args.scope,
        pair_id=args.pair_id,
        scorer_override=scorer,
        arms_override=("off", "on"),
        experiment_config_sha256=config_sha,
        evidence_root=ROOT / "docs" / "runtime-evidence" / "h6",
        scenarios_override=scenarios,
        scenario_materializer=materialize_h6_scenario,
        classic_only_off=True,
        run_lock_sha256=None if run_lock is None else str(run_lock.get("lock_sha256")),
        run_schema_version=(
            "safedrive.h6.vla75.run.v2"
            if args.contract == "vla75-v2"
            else "safedrive.h5.closed_loop.v1"
        ),
        worktree_identity_override=(
            None
            if run_lock is None
            else dict(run_lock.get("worktree") or {})
        ),
        router_config=(
            getattr(scorer, "router_calibration", None)
            if args.contract == "vla75-v2"
            else None
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
