#!/usr/bin/env python3
"""R2 / G4A paired outcome orchestrator.

Subcommands:
  registry-dry-run  — spawn/script/cleanup validation (no VLA, no oracle)
  run-pair          — anchor + branch 0/1 for one scenario/seed (live)
  run-set           — all 12 pairs
  aggregate         — offline oracle table from evidence dir
  snap-waypoints    — authoring helper: project registry poses to roads

Does not start R3/World. Branch mode never re-forwards VLA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "simlingo-main"))

from driving_vla.evaluation.comparability import (  # noqa: E402
    evaluate_pair_comparability,
)
from driving_vla.evaluation.fixture_runtime import (  # noqa: E402
    FixtureError,
    apply_weather,
    cleanup_session,
    compare_measured,
    connect_world,
    measure_initial_state,
    nearest_driving_waypoint,
    open_fixture_session,
    restore_async,
    step_fixture,
)
from driving_vla.evaluation.oracle import (  # noqa: E402
    aggregate_oracle_table,
    evaluate_pair_oracle,
)
from driving_vla.evaluation.outcome_metrics import (  # noqa: E402
    aggregate_branch_outcome,
)
from driving_vla.evaluation.paired_contract import (  # noqa: E402
    content_hash,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    freeze_registry,
    load_scenario_registry,
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_snap_waypoints(args: argparse.Namespace) -> int:
    """Authoring aid: print nearest driving waypoints for each fixture pose."""
    reg = load_scenario_registry(args.registry)
    client, world = connect_world(
        host=args.host,
        port=args.port,
        map_name="Town03",
        sim_dt_s=0.05,
        sync=True,
    )
    try:
        rows = []
        for fx in reg.fixtures:
            ego_wp = nearest_driving_waypoint(
                world, fx.ego.transform.x, fx.ego.transform.y, fx.ego.transform.z
            )
            actors = []
            for a in fx.actors:
                actors.append(
                    {
                        "name": a.name,
                        "requested": a.transform.raw_dict(),
                        "nearest": nearest_driving_waypoint(
                            world, a.transform.x, a.transform.y, a.transform.z
                        ),
                    }
                )
            rows.append(
                {
                    "scenario_id": fx.scenario_id,
                    "seed_id": fx.seed_id,
                    "ego_requested": fx.ego.transform.raw_dict(),
                    "ego_nearest": ego_wp,
                    "actors": actors,
                }
            )
            print(
                f"{fx.scenario_id}/{fx.seed_id} ego "
                f"req=({fx.ego.transform.x:.1f},{fx.ego.transform.y:.1f}) "
                f"wp=({ego_wp['x']:.2f},{ego_wp['y']:.2f}) yaw={ego_wp['yaw_deg']:.1f}",
                flush=True,
            )
        out = Path(args.out) if args.out else Path("docs/runtime-evidence/r2-g4a-paired-pilot/registry/waypoint_snap.json")
        _write_json(out, {"rows": rows})
        print(f"wrote {out}", flush=True)
        return 0
    finally:
        restore_async(world)


def cmd_registry_dry_run(args: argparse.Namespace) -> int:
    reg = load_scenario_registry(args.registry)
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)

    client, world = connect_world(
        host=args.host,
        port=args.port,
        map_name="Town03",
        sim_dt_s=0.05,
        sync=True,
    )
    report: dict[str, Any] = {
        "schema": "safedrive.g4a.registry_dry_run.v1",
        "registry_sha256": reg.compute_registry_sha256(),
        "started_wall_time": time.time(),
        "pairs": [],
        "ok": True,
        "n_ok": 0,
        "n_fail": 0,
    }
    try:
        for fx in reg.fixtures:
            pair_key = f"{fx.scenario_id}/{fx.seed_id}"
            entry: dict[str, Any] = {
                "scenario_id": fx.scenario_id,
                "seed_id": fx.seed_id,
                "family": fx.family,
                "requested_initial_state_hash": fx.requested_initial_state_hash(),
                "attempts": [],
                "ok": False,
            }
            print(f"[dry-run] {pair_key} ...", flush=True)
            apply_weather(world, fx.weather)
            measured_list = []
            try:
                for attempt in range(2):
                    session = open_fixture_session(client, world, fx, settle_ticks=5)
                    try:
                        measured = measure_initial_state(session)
                        # short scripted motion check (no VLA)
                        n_ticks = int(round(2.5 / fx.sim_dt_s))
                        ticks = step_fixture(session, n_ticks=min(20, n_ticks), sim_dt_s=fx.sim_dt_s)
                        measured_list.append(measured)
                        entry["attempts"].append(
                            {
                                "attempt": attempt,
                                "measured_hash": measured.measured_hash(),
                                "simulation_frame": measured.simulation_frame,
                                "n_script_ticks": len(ticks),
                                "ego_xy0": [
                                    measured.ego().transform.x,
                                    measured.ego().transform.y,
                                ],
                                "script_phases": dict(measured.actor_script_phase),
                            }
                        )
                    finally:
                        cleanup_session(session)

                mismatches = compare_measured(measured_list[0], measured_list[1])
                entry["rebuild_mismatches"] = mismatches
                entry["measured_hashes"] = [m.measured_hash() for m in measured_list]
                # Dry-run: positions after settle may drift slightly with physics;
                # use slightly looser pos for rebuild check of unscripted hold, but
                # report hard gate status separately.
                hard = compare_measured(
                    measured_list[0],
                    measured_list[1],
                    pos_tol_m=0.05,
                    yaw_tol_deg=0.5,
                    vel_tol_mps=0.5,
                )
                entry["rebuild_ok"] = len(hard) == 0
                entry["ok"] = entry["rebuild_ok"]
                if entry["ok"]:
                    report["n_ok"] += 1
                    print(f"  OK rebuild hashes={entry['measured_hashes']}", flush=True)
                else:
                    report["n_fail"] += 1
                    report["ok"] = False
                    print(f"  FAIL rebuild {hard}", flush=True)
            except FixtureError as exc:
                report["n_fail"] += 1
                report["ok"] = False
                entry["ok"] = False
                entry["error"] = str(exc)
                entry["error_type"] = "FixtureError"
                print(f"  FAIL {exc}", flush=True)
            except Exception as exc:  # noqa: BLE001
                report["n_fail"] += 1
                report["ok"] = False
                entry["ok"] = False
                entry["error"] = str(exc)
                entry["error_type"] = type(exc).__name__
                entry["traceback"] = traceback.format_exc()
                print(f"  ERROR {exc}", flush=True)
            report["pairs"].append(entry)

        report["ended_wall_time"] = time.time()
        _write_json(evidence / "dry_run_report.json", report)

        if report["ok"] and report["n_ok"] == 12:
            frozen = freeze_registry(reg)
            manifest = frozen.freeze_manifest()
            # copy registry into evidence
            reg_copy = evidence / "scenario_registry_v1.toml"
            reg_copy.write_bytes(Path(args.registry).read_bytes())
            _write_json(evidence / "registry_manifest.json", manifest)
            print(
                f"REGISTRY FROZEN sha256={manifest['registry_sha256']} pairs=12",
                flush=True,
            )
            return 0

        print(
            f"DRY-RUN NOT FROZEN ok={report['n_ok']}/12 fail={report['n_fail']}",
            flush=True,
        )
        return 2
    finally:
        restore_async(world)


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Aggregate exactly the 12 pair+attempt slots from run_set_manifest/report.

    Fail-closed if not exactly 12 unique pair_id+attempt_id rows. Does not scan
    the whole tree or double-count ledger + files.
    """
    from driving_vla.evaluation.oracle import PairOracleResult
    from driving_vla.evaluation.runner_contract import (
        RunnerContractError,
        aggregate_from_run_set_spec,
        slots_from_run_set_report_or_manifest,
    )

    evidence = Path(args.evidence_dir)
    pairs_dir = evidence / "pairs"
    man_path = Path(getattr(args, "run_set_manifest", "") or evidence / "run_set_manifest.json")
    rep_path = Path(getattr(args, "run_set_report", "") or evidence / "run_set_report.json")

    run_set_manifest = None
    run_set_report = None
    if rep_path.is_file():
        run_set_report = json.loads(rep_path.read_text(encoding="utf-8"))
    if man_path.is_file():
        run_set_manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if run_set_report is None and run_set_manifest is None:
        print(
            "aggregate fail-closed: need run_set_report.json or run_set_manifest.json",
            flush=True,
        )
        return 1

    try:
        slots = slots_from_run_set_report_or_manifest(
            run_set_manifest=run_set_manifest,
            run_set_report=run_set_report,
        )
        results = aggregate_from_run_set_spec(
            pairs_root=pairs_dir,
            slots=slots,
            require_n=12,
        )
    except RunnerContractError as exc:
        print(f"aggregate fail-closed: {exc}", flush=True)
        return 1

    objs = []
    for r in results:
        objs.append(
            PairOracleResult(
                pair_id=str(r["pair_id"]),
                scenario_id=str(r.get("scenario_id", "")),
                seed_id=str(r.get("seed_id", "")),
                family=str(r.get("family", "")),
                comparable=bool(r.get("comparable", False)),
                top1_candidate_id=str(
                    r.get("top1_candidate_id") or "UNKNOWN_CANDIDATE"
                ),
                top1_candidate_index=int(r.get("top1_candidate_index", 0) or 0),
                oracle_candidate_id=r.get("oracle_candidate_id"),
                oracle_candidate_index=r.get("oracle_candidate_index"),
                oracle_decision_level=r.get("oracle_decision_level"),
                decision_reason=str(r.get("decision_reason", "")),
                pair_label=str(r.get("pair_label", "INCOMPARABLE")),
                both_bad=bool(r.get("both_bad", False)),
                outcome_delta=dict(r.get("outcome_delta") or {}),
                failure_reasons=tuple(r.get("failure_reasons") or ()),
                relative_winner_if_both_bad=r.get("relative_winner_if_both_bad"),
            )
        )
    table = aggregate_oracle_table(objs)
    # Explicit denominator includes failed/incomparable
    table["n_rows"] = len(objs)
    table["n_comparable"] = sum(1 for o in objs if o.comparable)
    table["n_incomparable_or_failed"] = sum(1 for o in objs if not o.comparable)
    table["denominator"] = 12
    table["source"] = "run_set_manifest_or_report_slots_only"
    _write_json(evidence / "oracle_table.json", table)
    _write_json(evidence / "pilot_summary.json", table["pilot"])
    _write_json(evidence / "aggregate_rows.json", {"rows": results, "n": len(results)})
    print(json.dumps(table["pilot"], indent=2), flush=True)
    print(
        json.dumps(
            {
                "n_rows": 12,
                "n_comparable": table["n_comparable"],
                "n_incomparable_or_failed": table["n_incomparable_or_failed"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def cmd_run_pair(args: argparse.Namespace) -> int:
    """R2-C/D live: one anchor forward + branch 0/1 cold rebuilds."""
    from driving_vla.evaluation.paired_live import run_pair

    order = None
    if getattr(args, "branch_order", ""):
        parts = [int(x) for x in str(args.branch_order).split(",")]
        if len(parts) != 2 or set(parts) != {0, 1}:
            print("branch-order must be like 0,1 or 1,0", flush=True)
            return 2
        order = (parts[0], parts[1])
    try:
        manifest = run_pair(
            registry_path=args.registry,
            scenario_id=args.scenario_id,
            seed_id=args.seed_id,
            evidence_root=Path(args.evidence_dir),
            host=args.host,
            port=args.port,
            branch_order=order,
            device=str(getattr(args, "device", "cuda")),
            registry_manifest_path=getattr(args, "registry_manifest", None) or None,
        )
        label = None
        if isinstance(manifest.get("oracle"), dict):
            label = manifest["oracle"].get("pair_label")
        print(
            json.dumps(
                {
                    "status": "OK",
                    "pair_id": manifest.get("pair_id"),
                    "comparable": manifest.get("comparable"),
                    "label": label,
                    "idempotent_read": bool(manifest.get("idempotent_read", False)),
                    "attempt_id": manifest.get("attempt_id"),
                },
                indent=2,
            ),
            flush=True,
        )
        if manifest.get("idempotent_read"):
            return 0
        return 0 if manifest.get("comparable") else 4
    except Exception as exc:
        print(f"run-pair FAILED: {exc}", flush=True)
        traceback.print_exc()
        return 1


def _default_registry_manifest() -> Path:
    return (
        ROOT
        / "docs"
        / "runtime-evidence"
        / "r2-g4a-paired-pilot"
        / "registry"
        / "registry_manifest.json"
    )


def cmd_run_set(args: argparse.Namespace) -> int:
    """R2-D orchestrator: immutable manifest → fixed-order run_pair → full report.

    continue_policy is fixed at manifest freeze time (continue_all | stop_on_fail),
    never chosen from outcomes mid-run. ``--plan-only`` freezes manifest only.
    """
    from driving_vla.evaluation.paired_live import (
        EXECUTOR_CONFIG_HASH,
        _identity_hashes_without_carla,
        bind_spatial_model_retimer_hash,
        build_spatial_run_identity,
        run_pair,
    )
    from driving_vla.evaluation.runner_contract import (
        CONTINUE_POLICY_CONTINUE_ALL,
        CONTINUE_POLICY_STOP_ON_FAIL,
        RETRY_POLICY_NO_AUTO_RETRY,
        build_run_set_manifest,
        ensure_run_set_manifest,
        execute_run_set_orchestration,
        normalize_continue_policy,
        require_frozen_registry,
        run_set_exit_code,
        write_json,
        write_json_atomic,
    )

    evidence = Path(args.evidence_dir)
    pairs_root = evidence / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    rs_man_path = evidence / "run_set_manifest.json"
    report_path = evidence / "run_set_report.json"
    checkpoint_path = evidence / "run_set_checkpoint.json"

    manifest_path = getattr(args, "registry_manifest", None) or None
    if not manifest_path:
        default_man = _default_registry_manifest()
        if default_man.is_file():
            manifest_path = str(default_man)

    # Pre-registered continue policy (CLI maps to fixed enum; not outcome-driven).
    if bool(getattr(args, "stop_on_fail", False)):
        continue_policy = CONTINUE_POLICY_STOP_ON_FAIL
    else:
        continue_policy = normalize_continue_policy(
            getattr(args, "continue_policy", None) or CONTINUE_POLICY_CONTINUE_ALL
        )
    # R2-D freezes no_auto_retry.
    retry_policy = RETRY_POLICY_NO_AUTO_RETRY
    spatial_k2 = bool(getattr(args, "spatial_k2", False))
    spatial_head_ckpt = str(getattr(args, "spatial_head_ckpt", "") or "")
    spatial_identity: dict[str, str] = {}

    try:
        reg, freeze_audit, man_path = require_frozen_registry(
            args.registry,
            manifest_path=manifest_path,
            repo_root=ROOT,
        )
        model_checkpoint_hash, retimer_hash, model_retimer_hash, _cfg = (
            _identity_hashes_without_carla(device=str(getattr(args, "device", "cuda")))
        )
        if spatial_k2:
            if not spatial_head_ckpt:
                raise RuntimeError("--spatial-k2 requires --spatial-head-ckpt")
            from driving_vla.model.checkpoint_contract import (
                require_checkpoint_blind_registry,
                require_checkpoint_for_use,
            )

            require_checkpoint_for_use(spatial_head_ckpt, "r2k_pilot")
            require_checkpoint_blind_registry(
                spatial_head_ckpt,
                str(freeze_audit["registry_sha256"]),
            )
            spatial_identity = build_spatial_run_identity(spatial_head_ckpt)
            model_retimer_hash = bind_spatial_model_retimer_hash(
                model_retimer_hash, spatial_identity
            )
        from driving_vla.evaluation.scenario_registry import REGISTRY_SCHEMA

        # Stable identity for resume validation. Planned slots are never
        # recomputed when run_set_manifest.json already exists.
        identity = {
            "registry_sha256": str(
                freeze_audit.get("registry_sha256") or reg.compute_registry_sha256()
            ),
            "registry_schema_version": REGISTRY_SCHEMA,
            "model_retimer_hash": model_retimer_hash,
            "model_checkpoint_hash": model_checkpoint_hash,
            "retimer_hash": retimer_hash,
            "executor_config_hash": EXECUTOR_CONFIG_HASH,
            "continue_policy": continue_policy,
            "retry_policy": retry_policy,
            "n_pairs": 12,
            **spatial_identity,
        }

        def _build_new_manifest() -> dict:
            # Evidence scan + planned IDs ONLY on first exclusive create.
            return build_run_set_manifest(
                registry=reg,
                freeze_audit=freeze_audit,
                pairs_root=pairs_root,
                model_retimer_hash=model_retimer_hash,
                executor_config_hash=EXECUTOR_CONFIG_HASH,
                continue_policy=continue_policy,
                retry_policy=retry_policy,
                model_checkpoint_hash=model_checkpoint_hash,
                retimer_hash=retimer_hash,
                registry_path=str(Path(args.registry).as_posix()),
                registry_manifest_path=str(Path(man_path).as_posix()),
                **spatial_identity,
            )

        # Existing manifest → self-hash + stable identity only (no Evidence rescan).
        # Missing → build_fn scans once and exclusive-creates.
        run_set_manifest, man_mode = ensure_run_set_manifest(
            rs_man_path,
            identity if rs_man_path.is_file() else None,
            build_fn=None if rs_man_path.is_file() else _build_new_manifest,
        )
    except Exception as exc:
        print(f"run-set FAILED preflight/manifest: {exc}", flush=True)
        traceback.print_exc()
        return 1

    print(
        f"[run-set] run_set_manifest {man_mode} "
        f"hash={run_set_manifest['manifest_content_hash'][:16]}… "
        f"continue_policy={run_set_manifest['continue_policy']} "
        f"retry_policy={run_set_manifest['retry_policy']}",
        flush=True,
    )

    if bool(getattr(args, "plan_only", False)):
        plan_out = evidence / "run_set_plan.json"
        write_json(
            plan_out,
            {
                "mode": "plan_only",
                "manifest_mode": man_mode,
                "run_set_manifest_path": str(rs_man_path.as_posix()),
                "manifest_content_hash": run_set_manifest["manifest_content_hash"],
                "n_planned": 12,
                "pairs": run_set_manifest["pairs"],
                "continue_policy": run_set_manifest["continue_policy"],
                "retry_policy": run_set_manifest["retry_policy"],
            },
        )
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "n_planned": 12,
                    "manifest_mode": man_mode,
                    "manifest_path": str(rs_man_path.as_posix()),
                    "first": run_set_manifest["pairs"][0],
                    "last": run_set_manifest["pairs"][-1],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    device = str(getattr(args, "device", "cuda"))

    # Lazy shared SimLingo load: only on first real pair execution.
    # Offline recovery tests mock run_pair and must not require torch/SimLingo.
    shared_policy = None
    shared_identity = (
        str(run_set_manifest.get("model_checkpoint_hash") or model_checkpoint_hash),
        str(run_set_manifest.get("retimer_hash") or retimer_hash),
        str(run_set_manifest.get("model_retimer_hash") or model_retimer_hash),
    )
    _policy_load_error: Exception | None = None
    _policy_load_failed = False

    def _ensure_shared_policy() -> Any:
        nonlocal shared_policy, _policy_load_error, _policy_load_failed
        if shared_policy is not None:
            return shared_policy
        if _policy_load_failed and _policy_load_error is not None:
            # Cache failure: avoid N× traceback on offline multi-pair resume.
            raise _policy_load_error
        try:
            import torch
            from driving_vla.model.neural_policy import NeuralV1Policy, NeuralV2Policy
            from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for run-set")
            print("[run-set] loading shared SimLingo policy once…", flush=True)
            _rt = SimLingoNeuralRuntime(device=device)
            _load = _rt.load()
            if not _load.ok:
                raise RuntimeError(f"SimLingo load failed: {_load.error}")
            if spatial_k2:
                if not spatial_head_ckpt:
                    raise RuntimeError("--spatial-k2 requires --spatial-head-ckpt")
                # Formal R2-K: require formal-OK checkpoint contract
                from driving_vla.model.checkpoint_contract import (
                    require_checkpoint_for_use,
                )

                require_checkpoint_for_use(spatial_head_ckpt, "r2k_pilot")
                shared_policy = NeuralV2Policy(
                    runtime=_rt,
                    keep_on_gpu=True,
                    spatial_head_checkpoint=spatial_head_ckpt,
                    device=device,
                    require_driving_feature=True,
                    checkpoint_use="r2k_pilot",
                )
                print(
                    f"[run-set] shared NeuralV2Policy ready ckpt={spatial_head_ckpt}",
                    flush=True,
                )
            else:
                shared_policy = NeuralV1Policy(runtime=_rt, keep_on_gpu=True)
                print("[run-set] shared NeuralV1Policy ready", flush=True)
            shared_policy.ensure_loaded()
            return shared_policy
        except Exception as exc:
            _policy_load_failed = True
            _policy_load_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
            print(f"run-set FAILED shared policy load: {exc}", flush=True)
            traceback.print_exc()
            raise

    def _ensure_carla_after_rpc_fault(reason: str) -> None:
        """If CARLA died mid-pair, restart once so later pairs can still run."""
        print(f"[run-set] CARLA recovery after: {reason[:160]}", flush=True)
        try:
            import subprocess

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "sdf.py"),
                "sim",
                "ensure",
                "--map",
                "Town03",
                "--rhi",
                "dx12",
                "--startup-timeout",
                "180",
                "--json",
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=False, timeout=240)
        except Exception as exc:  # noqa: BLE001
            print(f"[run-set] ensure failed: {exc}", flush=True)

    def _run_pair_fn(
        *,
        scenario_id: str,
        seed_id: str,
        branch_order: tuple[int, ...],
        force_attempt_id: int,
        retry_policy: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        load_exc: Exception | None = None
        policy = None
        try:
            policy = _ensure_shared_policy()
        except Exception as exc:  # noqa: BLE001
            # Offline unit tests mock run_pair and never need SimLingo/torch.
            # Live path still fails closed if run_pair also cannot proceed.
            load_exc = exc
        try:
            out = dict(
                run_pair(
                    registry_path=args.registry,
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    evidence_root=pairs_root,
                    host=args.host,
                    port=args.port,
                    branch_order=tuple(branch_order),  # type: ignore[arg-type]
                    device=device,
                    registry_manifest_path=manifest_path,
                    repo_root=ROOT,
                    force_attempt_id=int(force_attempt_id),
                    retry_policy=str(retry_policy),
                    shared_policy=policy,
                    shared_identity=shared_identity,
                    carla_timeout_s=60.0,
                    spatial_k2=spatial_k2,
                    spatial_head_checkpoint=spatial_head_ckpt or None,
                    spatial_run_identity=spatial_identity or None,
                )
            )
        except Exception as exc:
            if load_exc is not None and policy is None:
                raise load_exc from exc
            err = str(exc).lower()
            if "time-out" in err or "timeout" in err or "connect" in err:
                _ensure_carla_after_rpc_fault(str(exc))
            raise
        # run_pair may return FAILED without raising
        err2 = str(out.get("error") or out.get("failure") or "").lower()
        if str(out.get("status")) == "FAILED" and (
            "time-out" in err2 or "timeout" in err2 or "connect" in err2
        ):
            _ensure_carla_after_rpc_fault(err2)
        return out

    try:
        report = execute_run_set_orchestration(
            run_set_manifest=run_set_manifest,
            run_pair_fn=_run_pair_fn,
            pairs_root=pairs_root,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
        )
    except Exception as exc:
        print(f"run-set FAILED orchestration: {exc}", flush=True)
        traceback.print_exc()
        return 1

    report["run_set_manifest_path"] = str(rs_man_path.as_posix())
    report["mode"] = "live"
    write_json_atomic(report_path, report)

    if not bool(getattr(args, "no_aggregate", False)):
        agg_ns = argparse.Namespace(
            evidence_dir=str(evidence),
            run_set_manifest=str(rs_man_path),
            run_set_report=str(report_path),
        )
        agg_code = cmd_aggregate(agg_ns)
        report["aggregate_exit_code"] = int(agg_code)
        write_json_atomic(report_path, report)

    summary = report["summary"]
    code = run_set_exit_code(
        summary, min_comparable=int(getattr(args, "min_comparable", 10))
    )
    print(
        json.dumps(
            {
                "status": "OK" if code == 0 else ("PILOT_GATE" if code == 4 else "FAILED"),
                "exit_code": code,
                "summary": summary,
                "report_path": str(report_path.as_posix()),
                "checkpoint_path": str(checkpoint_path.as_posix()),
                "n_pair_results": len(report["pair_results"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R2 G4A paired oracle orchestrator")
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="path to scenario_registry_v1.toml",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("registry-dry-run", help="fixture spawn/script/rebuild validation")
    d.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot/registry",
    )
    d.set_defaults(func=cmd_registry_dry_run)

    s = sub.add_parser("snap-waypoints", help="authoring: nearest road waypoints")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_snap_waypoints)

    r = sub.add_parser("run-pair", help="single pair anchor+branches")
    r.add_argument("--scenario-id", required=True)
    r.add_argument("--seed-id", required=True)
    r.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot/pairs",
    )
    r.add_argument("--branch-order", default="", help="e.g. 0,1 or 1,0")
    r.add_argument("--device", default="cuda")
    r.add_argument(
        "--registry-manifest",
        default="",
        help="path to frozen registry_manifest.json (required layout under evidence/registry)",
    )
    r.set_defaults(func=cmd_run_pair)

    rs = sub.add_parser("run-set", help="all 12 pairs (registry order + counterbalance)")
    rs.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot",
    )
    rs.add_argument(
        "--registry-manifest",
        default="",
        help="frozen registry_manifest.json (default: evidence registry path)",
    )
    rs.add_argument("--device", default="cuda")
    rs.add_argument(
        "--plan-only",
        action="store_true",
        help="validate frozen registry and write plan JSON only (no CARLA/VLA)",
    )
    rs.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="pre-registered continue_policy=stop_on_fail (not outcome-driven)",
    )
    rs.add_argument(
        "--continue-policy",
        default="",
        help="continue_all (default) or stop_on_fail; frozen into run_set_manifest",
    )
    rs.add_argument(
        "--min-comparable",
        type=int,
        default=10,
        help="pilot gate: minimum comparable pairs for exit 0 (default 10)",
    )
    rs.add_argument(
        "--no-aggregate",
        action="store_true",
        help="skip oracle aggregate after run-set",
    )
    rs.add_argument(
        "--spatial-k2",
        action="store_true",
        help="R2-X: use NeuralV2Policy dual residual + Guard V2 anchor",
    )
    rs.add_argument(
        "--spatial-head-ckpt",
        default="",
        help="path to formal-OK spatial head checkpoint (required with --spatial-k2)",
    )
    rs.set_defaults(func=cmd_run_set)

    a = sub.add_parser("aggregate", help="offline oracle aggregation from run-set slots")
    a.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot",
    )
    a.add_argument(
        "--run-set-manifest",
        default="",
        help="path to run_set_manifest.json (default: <evidence>/run_set_manifest.json)",
    )
    a.add_argument(
        "--run-set-report",
        default="",
        help="path to run_set_report.json (default: <evidence>/run_set_report.json)",
    )
    a.set_defaults(func=cmd_aggregate)

    fr = sub.add_parser(
        "freeze-repeat-plan",
        help="R2-E: exclusive-create immutable repeat_audit_plan BEFORE outcomes",
    )
    fr.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot",
    )
    fr.add_argument("--registry-manifest", default="")
    fr.add_argument("--device", default="cuda")
    fr.set_defaults(func=cmd_freeze_repeat_plan)

    ra = sub.add_parser(
        "run-repeat-audit",
        help="R2-E: run frozen 2-pair repeats (independent attempts)",
    )
    ra.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot",
    )
    ra.add_argument("--registry-manifest", default="")
    ra.add_argument("--device", default="cuda")
    ra.add_argument("--host", default="127.0.0.1")
    ra.add_argument("--port", type=int, default=2000)
    ra.set_defaults(func=cmd_run_repeat_audit)

    cl = sub.add_parser(
        "close-r2",
        help="R2-E: validate D + write oracle/pilot/repeat/closure reports",
    )
    cl.add_argument(
        "--evidence-dir",
        default="docs/runtime-evidence/r2-g4a-paired-pilot",
    )
    cl.add_argument("--registry-manifest", default="")
    cl.add_argument("--min-comparable", type=int, default=10)
    cl.add_argument(
        "--run-repeats",
        action="store_true",
        help="if comparable>=min, execute frozen repeats before closing",
    )
    cl.add_argument("--device", default="cuda")
    cl.add_argument("--host", default="127.0.0.1")
    cl.add_argument("--port", type=int, default=2000)
    cl.set_defaults(func=cmd_close_r2)
    return p


def cmd_freeze_repeat_plan(args: argparse.Namespace) -> int:
    """Exclusive-create or reuse immutable repeat_audit_plan (no outcomes used)."""
    from driving_vla.evaluation.paired_live import (
        EXECUTOR_CONFIG_HASH,
        _identity_hashes_without_carla,
    )
    from driving_vla.evaluation.runner_contract import (
        REPEAT_AUDIT_SEED,
        build_repeat_audit_plan,
        ensure_repeat_audit_plan,
        require_frozen_registry,
    )

    evidence = Path(args.evidence_dir)
    pairs_root = evidence / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)
    plan_path = evidence / "repeat_audit_plan.json"
    manifest_path = getattr(args, "registry_manifest", None) or None
    if not manifest_path:
        default_man = _default_registry_manifest()
        if default_man.is_file():
            manifest_path = str(default_man)
    try:
        reg, freeze_audit, man_path = require_frozen_registry(
            args.registry,
            manifest_path=manifest_path,
            repo_root=ROOT,
        )
        model_checkpoint_hash, retimer_hash, model_retimer_hash, _cfg = (
            _identity_hashes_without_carla(device=str(getattr(args, "device", "cuda")))
        )
        identity = {
            "registry_sha256": str(
                freeze_audit.get("registry_sha256") or reg.compute_registry_sha256()
            ),
            "model_retimer_hash": model_retimer_hash,
            "model_checkpoint_hash": model_checkpoint_hash,
            "retimer_hash": retimer_hash,
            "executor_config_hash": EXECUTOR_CONFIG_HASH,
            "audit_seed": REPEAT_AUDIT_SEED,
        }

        def _build() -> dict:
            return build_repeat_audit_plan(
                registry=reg,
                freeze_audit=freeze_audit,
                pairs_root=pairs_root,
                model_retimer_hash=model_retimer_hash,
                executor_config_hash=EXECUTOR_CONFIG_HASH,
                model_checkpoint_hash=model_checkpoint_hash,
                retimer_hash=retimer_hash,
            )

        plan, mode = ensure_repeat_audit_plan(
            plan_path,
            identity if plan_path.is_file() else None,
            build_fn=None if plan_path.is_file() else _build,
        )
    except Exception as exc:
        print(f"freeze-repeat-plan FAILED: {exc}", flush=True)
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {
                "status": "OK",
                "mode": mode,
                "plan_path": str(plan_path.as_posix()),
                "plan_content_hash": plan.get("plan_content_hash"),
                "selected_indices": plan.get("selected_indices"),
                "selected_families": plan.get("selected_families"),
                "pairs": [
                    {
                        "index": p["index"],
                        "scenario_id": p["scenario_id"],
                        "seed_id": p["seed_id"],
                        "family": p["family"],
                        "pair_id": p["pair_id"],
                        "branch_order": p["branch_order"],
                        "d_planned_attempt_id": p["d_planned_attempt_id"],
                        "repeat_attempt_id": p["repeat_attempt_id"],
                    }
                    for p in plan.get("pairs") or []
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def cmd_run_repeat_audit(args: argparse.Namespace) -> int:
    """Execute frozen repeat pairs at independent attempt ids (not D denominator)."""
    from driving_vla.evaluation.paired_live import (
        EXECUTOR_CONFIG_HASH,
        _identity_hashes_without_carla,
        run_pair,
    )
    from driving_vla.evaluation.runner_contract import (
        REPEAT_AUDIT_SEED,
        compare_repeat_label_consistency,
        ensure_repeat_audit_plan,
        load_pair_oracle_label,
        require_frozen_registry,
        write_json_atomic,
    )

    evidence = Path(args.evidence_dir)
    pairs_root = evidence / "pairs"
    plan_path = evidence / "repeat_audit_plan.json"
    report_path = evidence / "repeat_audit_report.json"
    if not plan_path.is_file():
        print("run-repeat-audit FAILED: missing repeat_audit_plan.json", flush=True)
        return 1
    manifest_path = getattr(args, "registry_manifest", None) or None
    if not manifest_path:
        default_man = _default_registry_manifest()
        if default_man.is_file():
            manifest_path = str(default_man)
    try:
        reg, freeze_audit, _man = require_frozen_registry(
            args.registry,
            manifest_path=manifest_path,
            repo_root=ROOT,
        )
        model_checkpoint_hash, retimer_hash, model_retimer_hash, _cfg = (
            _identity_hashes_without_carla(device=str(getattr(args, "device", "cuda")))
        )
        identity = {
            "registry_sha256": str(
                freeze_audit.get("registry_sha256") or reg.compute_registry_sha256()
            ),
            "model_retimer_hash": model_retimer_hash,
            "model_checkpoint_hash": model_checkpoint_hash,
            "retimer_hash": retimer_hash,
            "executor_config_hash": EXECUTOR_CONFIG_HASH,
            "audit_seed": REPEAT_AUDIT_SEED,
        }
        plan, mode = ensure_repeat_audit_plan(plan_path, identity)
        if mode != "reused":
            raise RuntimeError("repeat plan must already exist before run-repeat-audit")
    except Exception as exc:
        print(f"run-repeat-audit FAILED preflight: {exc}", flush=True)
        traceback.print_exc()
        return 1

    device = str(getattr(args, "device", "cuda"))
    host = str(getattr(args, "host", "127.0.0.1"))
    port = int(getattr(args, "port", 2000))
    rows: list[dict[str, Any]] = []
    for slot in plan.get("pairs") or []:
        pair_id = str(slot["pair_id"])
        d_aid = int(slot["d_planned_attempt_id"])
        r_aid = int(slot["repeat_attempt_id"])
        branch_order = tuple(int(x) for x in slot["branch_order"])
        print(
            f"[repeat] {slot['scenario_id']}/{slot['seed_id']} "
            f"pair={pair_id} d_attempt={d_aid} repeat_attempt={r_aid} "
            f"branch_order={branch_order}",
            flush=True,
        )
        original = load_pair_oracle_label(
            pairs_root, pair_id=pair_id, attempt_id=d_aid
        )
        try:
            man = dict(
                run_pair(
                    registry_path=args.registry,
                    scenario_id=str(slot["scenario_id"]),
                    seed_id=str(slot["seed_id"]),
                    evidence_root=pairs_root,
                    host=host,
                    port=port,
                    branch_order=branch_order,  # type: ignore[arg-type]
                    device=device,
                    registry_manifest_path=manifest_path,
                    repo_root=ROOT,
                    force_attempt_id=r_aid,
                    retry_policy="no_auto_retry",
                )
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            man = {"status": "FAILED", "error": str(exc), "attempt_id": r_aid}
            err = str(exc)
            traceback.print_exc()
        repeat = load_pair_oracle_label(pairs_root, pair_id=pair_id, attempt_id=r_aid)
        if not repeat.get("found") and man:
            # fall back to returned manifest fields
            repeat = {
                "pair_id": pair_id,
                "attempt_id": r_aid,
                "found": True,
                "status": man.get("status"),
                "comparable": bool(man.get("comparable", False)),
                "pair_label": (man.get("oracle") or {}).get("pair_label")
                or man.get("pair_label"),
                "oracle_candidate_index": (man.get("oracle") or {}).get(
                    "oracle_candidate_index"
                ),
                "both_bad": bool((man.get("oracle") or {}).get("both_bad", False)),
                "error": man.get("error") or err,
            }
        cmp = compare_repeat_label_consistency(original=original, repeat=repeat)
        rows.append(
            {
                "index": int(slot["index"]),
                "scenario_id": slot["scenario_id"],
                "seed_id": slot["seed_id"],
                "family": slot["family"],
                "pair_id": pair_id,
                "branch_order": list(branch_order),
                "d_planned_attempt_id": d_aid,
                "repeat_attempt_id": r_aid,
                "original": {
                    k: original.get(k)
                    for k in (
                        "status",
                        "comparable",
                        "pair_label",
                        "oracle_candidate_index",
                        "both_bad",
                        "error",
                    )
                },
                "repeat": {
                    k: repeat.get(k)
                    for k in (
                        "status",
                        "comparable",
                        "pair_label",
                        "oracle_candidate_index",
                        "both_bad",
                        "error",
                    )
                },
                "comparison": cmp,
                "run_pair_status": man.get("status"),
                "run_pair_error": man.get("error") or err,
                "in_d_denominator": False,
            }
        )

    n_consistent = sum(1 for r in rows if r["comparison"]["label_consistent"])
    report = {
        "schema_version": "safedrive.g4a.repeat_audit_report.v1",
        "plan_content_hash": plan.get("plan_content_hash"),
        "registry_sha256": plan.get("registry_sha256"),
        "n_planned": len(plan.get("pairs") or []),
        "n_executed": len(rows),
        "n_label_consistent": n_consistent,
        "all_labels_consistent": bool(rows) and n_consistent == len(rows),
        "pairs": rows,
        "note": "repeat attempts excluded from D 12-pair denominator",
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["all_labels_consistent"] else 4


def cmd_close_r2(args: argparse.Namespace) -> int:
    """Validate D evidence, optional repeats, write closure reports."""
    from driving_vla.evaluation.oracle import PairOracleResult, aggregate_oracle_table
    from driving_vla.evaluation.runner_contract import (
        RunnerContractError,
        aggregate_from_run_set_spec,
        classify_r2_closure,
        compare_repeat_label_consistency,
        ensure_repeat_audit_plan,
        load_pair_oracle_label,
        load_run_set_checkpoint,
        slots_from_run_set_report_or_manifest,
        validate_report_against_manifest,
        validate_run_set_manifest_self_integrity,
        write_json,
        write_json_atomic,
    )

    evidence = Path(args.evidence_dir)
    pairs_root = evidence / "pairs"
    man_path = evidence / "run_set_manifest.json"
    rep_path = evidence / "run_set_report.json"
    ckpt_path = evidence / "run_set_checkpoint.json"
    plan_path = evidence / "repeat_audit_plan.json"
    min_comp = int(getattr(args, "min_comparable", 10))

    if not man_path.is_file():
        print("close-r2 FAILED: missing run_set_manifest.json", flush=True)
        return 1
    run_set_manifest = json.loads(man_path.read_text(encoding="utf-8"))
    try:
        validate_run_set_manifest_self_integrity(run_set_manifest)
    except RunnerContractError as exc:
        print(f"close-r2 FAILED manifest: {exc}", flush=True)
        return 1

    run_set_report = None
    if rep_path.is_file():
        run_set_report = json.loads(rep_path.read_text(encoding="utf-8"))
        try:
            validate_report_against_manifest(run_set_report, run_set_manifest)
        except RunnerContractError as exc:
            print(f"close-r2 FAILED report/manifest: {exc}", flush=True)
            return 1
    if ckpt_path.is_file():
        try:
            load_run_set_checkpoint(ckpt_path, run_set_manifest=run_set_manifest)
        except RunnerContractError as exc:
            print(f"close-r2 FAILED checkpoint: {exc}", flush=True)
            return 1

    try:
        slots = slots_from_run_set_report_or_manifest(
            run_set_manifest=run_set_manifest,
            run_set_report=run_set_report,
        )
        results = aggregate_from_run_set_spec(
            pairs_root=pairs_root, slots=slots, require_n=12
        )
    except RunnerContractError as exc:
        print(f"close-r2 FAILED aggregate: {exc}", flush=True)
        return 1

    objs = []
    for r in results:
        objs.append(
            PairOracleResult(
                pair_id=str(r["pair_id"]),
                scenario_id=str(r.get("scenario_id", "")),
                seed_id=str(r.get("seed_id", "")),
                family=str(r.get("family", "")),
                comparable=bool(r.get("comparable", False)),
                top1_candidate_id=str(r.get("top1_candidate_id", "v1_nominal")),
                top1_candidate_index=int(r.get("top1_candidate_index", 0) or 0),
                oracle_candidate_id=r.get("oracle_candidate_id"),
                oracle_candidate_index=r.get("oracle_candidate_index"),
                oracle_decision_level=r.get("oracle_decision_level"),
                decision_reason=str(r.get("decision_reason", "")),
                pair_label=str(r.get("pair_label", "INCOMPARABLE")),
                both_bad=bool(r.get("both_bad", False)),
                outcome_delta=dict(r.get("outcome_delta") or {}),
                failure_reasons=tuple(r.get("failure_reasons") or ()),
                relative_winner_if_both_bad=r.get("relative_winner_if_both_bad"),
            )
        )
    table = aggregate_oracle_table(objs)
    table["n_rows"] = len(objs)
    table["n_comparable"] = sum(1 for o in objs if o.comparable)
    table["n_incomparable_or_failed"] = sum(1 for o in objs if not o.comparable)
    table["denominator"] = 12
    table["source"] = "run_set_manifest_or_report_slots_only"
    write_json(evidence / "oracle_table.json", table)
    write_json(evidence / "pilot_summary.json", table["pilot"])
    write_json(evidence / "aggregate_rows.json", {"rows": results, "n": len(results)})

    n_comp = int(table["n_comparable"])
    pilot_label = str(table["pilot"].get("label") or "PILOT_INCONCLUSIVE")

    # resource / failure rollup from attempt artifacts
    resource_rollup = _rollup_d_metrics(pairs_root, slots, results)

    dominant_fail = resource_rollup.get("dominant_failure_class")
    repeat_report = None
    repeat_consistent: bool | None = None
    n_repeat_done = 0

    if n_comp >= min_comp and bool(getattr(args, "run_repeats", False)):
        # execute repeats via same process
        ns = argparse.Namespace(
            registry=args.registry,
            registry_manifest=getattr(args, "registry_manifest", "") or "",
            evidence_dir=str(evidence),
            device=str(getattr(args, "device", "cuda")),
            host=str(getattr(args, "host", "127.0.0.1")),
            port=int(getattr(args, "port", 2000)),
        )
        rr_code = cmd_run_repeat_audit(ns)
        print(f"[close-r2] run-repeat-audit exit={rr_code}", flush=True)

    rep_audit_path = evidence / "repeat_audit_report.json"
    if rep_audit_path.is_file():
        repeat_report = json.loads(rep_audit_path.read_text(encoding="utf-8"))
        n_repeat_done = int(repeat_report.get("n_executed") or 0)
        if "all_labels_consistent" in repeat_report:
            repeat_consistent = bool(repeat_report["all_labels_consistent"])
    elif plan_path.is_file() and n_comp >= min_comp:
        # Offline compare if repeats already on disk
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        try:
            ensure_repeat_audit_plan(plan_path, None)  # self-validate reuse
        except Exception:
            pass
        offline_rows = []
        for slot in plan.get("pairs") or []:
            o = load_pair_oracle_label(
                pairs_root,
                pair_id=str(slot["pair_id"]),
                attempt_id=int(slot["d_planned_attempt_id"]),
            )
            r = load_pair_oracle_label(
                pairs_root,
                pair_id=str(slot["pair_id"]),
                attempt_id=int(slot["repeat_attempt_id"]),
            )
            cmp = compare_repeat_label_consistency(original=o, repeat=r)
            offline_rows.append(
                {
                    "index": slot["index"],
                    "pair_id": slot["pair_id"],
                    "comparison": cmp,
                    "original": o,
                    "repeat": r,
                }
            )
        if offline_rows and all(r["repeat"].get("found") for r in offline_rows):
            n_repeat_done = len(offline_rows)
            repeat_consistent = all(
                r["comparison"]["label_consistent"] for r in offline_rows
            )
            repeat_report = {
                "schema_version": "safedrive.g4a.repeat_audit_report.v1",
                "plan_content_hash": plan.get("plan_content_hash"),
                "n_planned": len(offline_rows),
                "n_executed": n_repeat_done,
                "n_label_consistent": sum(
                    1 for r in offline_rows if r["comparison"]["label_consistent"]
                ),
                "all_labels_consistent": bool(repeat_consistent),
                "pairs": offline_rows,
                "source": "offline_disk_compare",
            }
            write_json_atomic(rep_audit_path, repeat_report)

    n_repeat_planned = 2
    if plan_path.is_file():
        try:
            p = json.loads(plan_path.read_text(encoding="utf-8"))
            n_repeat_planned = int(p.get("n_repeat_pairs") or 2)
        except Exception:
            pass

    if n_comp < min_comp:
        # skip repeats; classify inconclusive/blocked/repair
        closure = classify_r2_closure(
            n_comparable=n_comp,
            min_comparable=min_comp,
            repeat_labels_all_consistent=None,
            n_repeat_done=0,
            n_repeat_planned=n_repeat_planned,
            pilot_label=pilot_label,
            dominant_failure_class=dominant_fail,
        )
        # Override: when below gate, use PILOT_INCONCLUSIVE as r2 experimental status
        # but keep classify mapping for REPAIR/BLOCKED
        if closure["r2_status"] not in {
            "BLOCKED_EXTERNAL",
            "REPAIR_REQUIRED",
            "PILOT_INCONCLUSIVE",
        }:
            closure["r2_status"] = "PILOT_INCONCLUSIVE"
            closure["completed_with_limits"] = False
    else:
        closure = classify_r2_closure(
            n_comparable=n_comp,
            min_comparable=min_comp,
            repeat_labels_all_consistent=repeat_consistent,
            n_repeat_done=n_repeat_done,
            n_repeat_planned=n_repeat_planned,
            pilot_label=pilot_label,
            dominant_failure_class=dominant_fail,
        )

    pair_status_table = []
    for r in results:
        pair_status_table.append(
            {
                "pair_id": r.get("pair_id"),
                "scenario_id": r.get("scenario_id"),
                "seed_id": r.get("seed_id"),
                "family": r.get("family"),
                "attempt_id": r.get("attempt_id"),
                "status": r.get("status"),
                "comparable": r.get("comparable"),
                "pair_label": r.get("pair_label"),
                "error": r.get("error"),
            }
        )

    closure_report = {
        "schema_version": "safedrive.g4a.r2_closure_report.v1",
        "r2_status": closure["r2_status"],
        "completed_with_limits": closure["completed_with_limits"],
        "pilot_label": closure["pilot_label"],
        "closure_reasons": closure["reasons"],
        "run_set_manifest_content_hash": run_set_manifest.get("manifest_content_hash"),
        "registry_sha256": run_set_manifest.get("registry_sha256"),
        "model_retimer_hash": run_set_manifest.get("model_retimer_hash"),
        "executor_config_hash": run_set_manifest.get("executor_config_hash"),
        "denominator": 12,
        "n_comparable": n_comp,
        "n_incomparable_or_failed": table["n_incomparable_or_failed"],
        "pilot": table["pilot"],
        "counts": table.get("counts"),
        "pair_status_table": pair_status_table,
        "repeat_audit": {
            "plan_path": str(plan_path.as_posix()) if plan_path.is_file() else None,
            "report": repeat_report,
            "all_labels_consistent": repeat_consistent,
            "n_repeat_done": n_repeat_done,
            "n_repeat_planned": n_repeat_planned,
            "in_d_denominator": False,
        },
        "resources": resource_rollup,
        "limits": [
            "6 scenarios × 2 seeds pilot only; not statistically significant",
            "CARLA SIL only",
            "R1 candidates differ only in longitudinal timing",
            "actor scripts fixed non-interactive",
            "oracle uses privileged future; offline only",
            "probability untrained/uncalibrated",
            "World not implemented",
            "2.5s primary horizon only",
        ],
        "forbidden_next": [
            "do_not_start_R3",
            "do_not_create_world_dataset",
            "do_not_train_or_attach_world",
        ],
    }
    write_json(evidence / "r2_closure_report.json", closure_report)
    write_json(evidence / "pilot_summary.json", {
        **table["pilot"],
        "r2_status": closure["r2_status"],
        "closure_reasons": closure["reasons"],
    })

    print(json.dumps(closure_report, indent=2, default=str), flush=True)
    if closure["completed_with_limits"]:
        return 0
    if closure["r2_status"] in {"PILOT_INCONCLUSIVE", "REPAIR_REQUIRED", "BLOCKED_EXTERNAL"}:
        return 4
    return 1


# Canonical report keys → actual BranchOutcomeMetrics / branch_summary.metrics fields.
# Real Evidence uses BranchOutcomeMetrics.to_dict() names (not short aliases).
BRANCH_METRIC_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "collision_episode_count": ("collision_episode_count", "had_collision", "collision"),
    "offroad_fraction": ("offroad_fraction",),
    "minimum_ttc_s": ("minimum_ttc_s", "min_ttc_s"),
    "minimum_actor_clearance_m": ("minimum_actor_clearance_m", "min_clearance_m"),
    "route_progress_delta_m": ("route_progress_delta_m", "progress_m", "route_progress_m"),
    "jerk_abs_p95": ("jerk_abs_p95", "jerk_p95"),
}


def extract_branch_metric_value(
    metrics: Mapping[str, Any], report_key: str
) -> float | None:
    """Read one outcome metric using real Evidence field names (+ legacy aliases)."""
    aliases = BRANCH_METRIC_FIELD_MAP.get(report_key, (report_key,))
    for name in aliases:
        if name not in metrics or metrics[name] is None:
            continue
        try:
            return float(metrics[name])
        except (TypeError, ValueError):
            continue
    return None


def classify_dominant_failure_class(
    failures: Sequence[Mapping[str, Any]],
) -> str | None:
    """Map failure text to carla | fixture | runner (timeout variants included)."""
    if not failures:
        return None
    text = " ".join(
        str(f.get("error") or f.get("failure") or "")
        + " "
        + " ".join(str(x) for x in (f.get("failure_reasons") or []))
        for f in failures
    ).lower()
    # Match both "timeout" and CARLA's "time-out of Nms while waiting for the simulator"
    if any(
        k in text
        for k in (
            "carla",
            "rpc",
            "timeout",
            "time-out",
            "simulator",
            "server",
            "tick",
            "sync",
            "connect_world",
        )
    ):
        return "carla"
    if any(k in text for k in ("spawn", "fixture", "waypoint")):
        return "fixture"
    return "runner"


def _rollup_d_metrics(
    pairs_root: Path,
    slots: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect latency/MPC/resource/failure signals from attempt artifacts."""
    latencies: list[float] = []
    vram: list[float] = []
    mpc_solved = mpc_timeout = mpc_fallback = 0
    cleanup_fail = 0
    failures: list[dict[str, Any]] = []
    metric_keys = tuple(BRANCH_METRIC_FIELD_MAP.keys())
    metrics_agg: dict[str, list[float]] = {k: [] for k in metric_keys}
    n_branch_summaries = 0

    for slot, row in zip(slots, results):
        pair_id = str(slot.get("pair_id") or row.get("pair_id"))
        aid = int(row.get("attempt_id", slot.get("planned_attempt_id", 0)))
        adir = pairs_root / pair_id / f"attempt_{aid}"
        if not adir.is_dir() and aid == 0:
            adir = pairs_root / pair_id
        if not row.get("comparable", False) or row.get("status") == "FAILED":
            err = row.get("error") or row.get("failure")
            reasons = row.get("failure_reasons")
            if not reasons and err:
                reasons = [str(err)]
            failures.append(
                {
                    "pair_id": pair_id,
                    "attempt_id": aid,
                    "status": row.get("status"),
                    "error": err,
                    "pair_label": row.get("pair_label"),
                    "failure_reasons": reasons,
                }
            )
        # anchor latency / VRAM
        p = adir / "anchor" / "run_config.json"
        if p.is_file():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
                if cfg.get("latency_ms") is not None:
                    latencies.append(float(cfg["latency_ms"]))
                if cfg.get("peak_vram_mb") is not None:
                    vram.append(float(cfg["peak_vram_mb"]))
            except Exception:
                pass
        for br in ("branch-0", "branch-1"):
            sp = adir / br / "branch_summary.json"
            if not sp.is_file():
                continue
            try:
                sm = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                continue
            n_branch_summaries += 1
            mpc_solved += int(sm.get("mpc_solved") or 0)
            mpc_timeout += int(sm.get("mpc_timeout") or 0)
            mpc_fallback += int(sm.get("mpc_fallback") or 0)
            if sm.get("cleanup_ok") is False:
                cleanup_fail += 1
            m = sm.get("metrics") or {}
            if not isinstance(m, Mapping):
                continue
            for k in metric_keys:
                val = extract_branch_metric_value(m, k)
                if val is not None:
                    metrics_agg[k].append(val)

    def _pct(vals: list[float], p: float) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        if len(s) == 1:
            return s[0]
        k = (len(s) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    dominant = classify_dominant_failure_class(failures)

    return {
        "forward_latency_ms": {
            "n": len(latencies),
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
            "p99": _pct(latencies, 99),
            "samples": latencies[:24],
        },
        "peak_vram_mb": {
            "n": len(vram),
            "max": max(vram) if vram else None,
            "p50": _pct(vram, 50),
            "samples": vram[:24],
        },
        "mpc": {
            "solved_ticks": mpc_solved,
            "timeout_ticks": mpc_timeout,
            "fallback_ticks": mpc_fallback,
        },
        "cleanup_fail_branches": cleanup_fail,
        "n_branch_summaries": n_branch_summaries,
        "outcome_metric_samples": {
            k: {
                "n": len(v),
                "min": min(v) if v else None,
                "max": max(v) if v else None,
                "p50": _pct(v, 50),
            }
            for k, v in metrics_agg.items()
        },
        "metric_field_map": {
            k: list(v) for k, v in BRANCH_METRIC_FIELD_MAP.items()
        },
        "failures": failures,
        "dominant_failure_class": dominant,
        "n_failed_or_incomparable_rows": len(failures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
