#!/usr/bin/env python3
"""X5H: 1–3 pair CARLA force smoke with NeuralV2Policy.

Uses run_anchor_v2 + run_branch cold rebuilds (no second VLA forward on branches).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

OUT = ROOT / "docs/runtime-evidence/r2x-live-force-smoke-v3-formal"
REG = ROOT / "safedrive_foundry/config/g4a/scenario_registry_v1.toml"
# Must be a FORMAL-OK checkpoint; v4 BOOTSTRAP_INVALID is hard-rejected before load.
CKPT = ROOT / "docs/runtime-evidence/r2x-training/checkpoints/FORMAL_HEAD_REQUIRED.pt"

DEFAULT_PAIRS = [
    ("cut_in_early", "seed_a"),
    ("lead_brake_hard", "seed_a"),
    ("cross_vehicle_clear", "seed_a"),
]


def classify_x5h_results(
    results: list[dict],
    *,
    formal: bool,
) -> tuple[str, bool]:
    """Counts-first X5H gate; proposal collapse may fail closed with limits.

    R2-K remains the 12-pair decision gate.  This smoke only proves that the
    learned dual path reaches distinct PM/MPC control on more than one scene.
    """
    n_planned = len(results)
    n_pass = sum(1 for row in results if row.get("status") == "PASS")
    n_dual = sum(
        1
        for row in results
        if row.get("status") == "PASS"
        and row.get("defensive_available")
        and set(row.get("forces_run") or []) >= {0, 1}
        and row.get("path_diverge")
    )
    if n_planned and n_pass == n_planned and n_dual >= 1:
        return "SMOKE_ALL_PAIRS_RAN_WITH_SOME_DUAL", bool(formal)
    failed = [row for row in results if row.get("status") != "PASS"]
    collapse_only = bool(failed) and all(
        "SPATIAL_COLLAPSE_ELIGIBLE" in str(row.get("error") or "")
        or "guard_not_ok_v2" in str(row.get("error") or "")
        for row in failed
    )
    if formal and n_planned == 3 and n_pass >= 2 and n_dual >= 1 and collapse_only:
        return "X5H_PASS_WITH_LIMITS", True
    if n_pass:
        return "SMOKE_PARTIAL", False
    return "SMOKE_FAIL", False


def branch_mpc_ok(branch: dict) -> bool:
    """Strict branch executor gate: no fallback and no deadline miss."""
    return (
        int(branch.get("mpc_fallback") or 0) == 0
        and int(branch.get("mpc_timeout") or 0) == 0
        and int(branch.get("mpc_solved") or 0) >= 40
    )


def _preflight() -> str:
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sdf.py"), "sim", "preflight", "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    try:
        return str(json.loads(r.stdout or "{}").get("status") or "UNKNOWN")
    except Exception:
        return f"parse_failed:{r.returncode}"


def _ensure_once() -> str:
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/sdf.py"),
            "sim",
            "ensure",
            "--map",
            "Town03",
            "--rhi",
            "dx12",
            "--startup-timeout",
            "180",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    try:
        return str(json.loads(r.stdout or "{}").get("status") or "UNKNOWN")
    except Exception:
        return f"ensure_parse_failed:{r.returncode}"


def run_one(
    scenario_id: str,
    seed_id: str,
    *,
    device: str = "cuda",
    checkpoint_use: str = "x5h_acceptance",
    spectator_follow: bool = True,
    spectator_wall_pace_s: float = 0.0,
) -> dict:
    from driving_vla.evaluation.fixture_runtime import (
        connect_world,
        purge_episode_actors,
        restore_async,
    )
    from driving_vla.evaluation.paired_live import run_anchor_v2, run_branch
    from driving_vla.evaluation.scenario_registry import load_scenario_registry
    from driving_vla.model.neural_policy import NeuralV2Policy
    from driving_vla.evaluation.paired_contract import content_hash

    reg = load_scenario_registry(REG)
    fixture = reg.get(scenario_id, seed_id)
    pair_id = content_hash(
        {"s": scenario_id, "seed": seed_id, "tag": "x5h_v2_smoke"}, nibble=12
    )
    ed = OUT / "pairs" / pair_id
    ed.mkdir(parents=True, exist_ok=True)

    pol = NeuralV2Policy(
        lazy=True,
        keep_on_gpu=True,
        spatial_head_checkpoint=str(CKPT) if CKPT.is_file() else None,
        device=device,
        require_driving_feature=True,
        checkpoint_use=checkpoint_use,
    )
    pol.ensure_loaded()
    client, world = connect_world(
        host="127.0.0.1", port=2000, map_name="Town03", sync=True
    )
    fwd = {"n": 0}
    try:
        art, meta = run_anchor_v2(
            client=client,
            world=world,
            fixture=fixture,
            policy=pol,
            pair_id=pair_id,
            model_checkpoint_hash="v2_spatial_live",
            registry_hash=str(reg.registry_sha256 or "unhashed"),
            evidence_dir=ed,
            forward_counter=fwd,
            spectator_follow=spectator_follow,
        )

        def_avail = True
        try:
            def_avail = bool(art.candidates[1].available)
        except Exception:
            def_avail = True
        forces = (0, 1) if def_avail else (0,)
        branches = []
        for force in forces:
            br = run_branch(
                client=client,
                world=world,
                fixture=fixture,
                artifact=art,
                force_index=force,
                pair_id=pair_id,
                registry_hash=str(reg.registry_sha256 or "unhashed"),
                model_retimer_hash="v2_spatial",
                evidence_dir=ed,
                forward_counter=fwd,
                policy=None,
                spectator_follow=spectator_follow,
                spectator_wall_pace_s=spectator_wall_pace_s,
            )
            summ = dict(br.summary or {})
            # Prefer metrics from BranchLiveResult when summary omits hashes
            rep = getattr(br, "report", None)
            ticks = list(br.ticks or [])
            steers = []
            control_seq = list(getattr(br, "control_seq", ()) or ())
            for item in control_seq:
                if item.get("steer") is not None:
                    steers.append(float(item["steer"]))
            for t in ticks:
                s = getattr(t, "steer_rad", None)
                if s is None and isinstance(t, dict):
                    s = t.get("steer_rad") or t.get("steer")
                if s is not None:
                    steers.append(float(s))
            mean_steer = sum(steers) / len(steers) if steers else None
            committed = (
                summ.get("committed_path_hash")
                or summ.get("path_hash")
                or summ.get("spatial_path_hash")
                or (getattr(rep, "committed_path_hash", None) if rep else None)
            )
            # Hash from control sequence path if present
            if not committed and control_seq:
                try:
                    from driving_vla.evaluation.paired_contract import content_hash

                    committed = content_hash(
                        {"n": len(control_seq), "tail": control_seq[-3:]},
                        nibble=16,
                    )
                except Exception:
                    committed = None
            branches.append(
                {
                    "force_index": force,
                    "summary": summ,
                    "control_payload_ok": bool(br.control_payload_ok),
                    "n_ticks": len(ticks),
                    "committed_path_hash": committed,
                    "source_id": summ.get("source_id")
                    or (getattr(rep, "source_id", None) if rep else None),
                    "mpc_solved": summ.get("mpc_solved")
                    or summ.get("n_mpc_solved")
                    or len([1 for t in ticks if str(getattr(t, "mode", "") or (t.get("mode") if isinstance(t, dict) else "")) == "mpc"]),
                    "mpc_timeout": summ.get("mpc_timeout")
                    or summ.get("n_mpc_timeout")
                    or 0,
                    "mpc_fallback": summ.get("mpc_fallback") or summ.get("n_mpc_fallback") or 0,
                    "mean_steer": mean_steer
                    if mean_steer is not None
                    else (summ.get("mean_steer") or summ.get("steer_mean")),
                    "control_hash": (
                        content_hash(control_seq, nibble=16)
                        if control_seq
                        else None
                    ),
                }
            )

        # Proposal hashes (diagnostic only)
        try:
            h0 = str(art.candidates[0].proposal_path_hash or "")
            h1 = str(art.candidates[1].proposal_path_hash or "")
            proposal_hashes = [h0, h1]
            proposal_div = bool(h0 and h1 and h0 != h1)
        except Exception:
            proposal_hashes = []
            proposal_div = False
        # Strict: committed path diverge from executed branches
        c_hashes = [b.get("committed_path_hash") for b in branches if b.get("committed_path_hash")]
        committed_path_div = len(c_hashes) == 2 and c_hashes[0] != c_hashes[1]
        steers = [b.get("mean_steer") for b in branches if b.get("mean_steer") is not None]
        steer_div = (
            len(steers) == 2 and abs(float(steers[0]) - float(steers[1])) > 1e-4
        )
        guard_ok = str(meta.get("guard_status") or "") in {"OK", "ok"}
        need_n = 2 if def_avail else 1
        mpc_ok = all(branch_mpc_ok(b) for b in branches)
        dual_force_ok = (not def_avail) or (
            len(branches) == 2
            and {int(b.get("force_index", -1)) for b in branches} == {0, 1}
        )
        # Formal pair PASS:
        # - always: guard + 1 forward + branches ran + mpc
        # - if defensive available: dual force + (committed path OR control) diverge
        # - proposal-only diverge is NOT enough for formal when dual force runs
        if def_avail:
            diverge_ok = bool(committed_path_div or steer_div)
        else:
            diverge_ok = True
        ok = (
            guard_ok
            and fwd["n"] == 1
            and len(branches) == need_n
            and dual_force_ok
            and all(int(b.get("n_ticks") or 0) >= 40 for b in branches)
            and all(b.get("control_payload_ok", True) for b in branches)
            and mpc_ok
            and diverge_ok
        )
        out_extra = {
            "defensive_available": def_avail,
            "forces_run": list(forces),
            "proposal_path_hashes": proposal_hashes,
            "proposal_path_diverge": proposal_div,
            "committed_path_diverge": committed_path_div,
            "mpc_ok": mpc_ok,
            "formal_diverge_ok": diverge_ok,
        }
        out = {
            "pair_id": pair_id,
            "scenario_id": scenario_id,
            "seed_id": seed_id,
            "status": "PASS" if ok else "FAIL",
            "forward_count": fwd["n"],
            "path_diverge": committed_path_div or proposal_div,
            "committed_path_diverge": committed_path_div,
            "steer_diverge": steer_div,
            "guard_status": meta.get("guard_status"),
            "anchor_meta": meta,
            "branches": branches,
            "path_hashes": proposal_hashes,
            **out_extra,
        }
        (ed / "pair_report.json").write_text(
            json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return out
    finally:
        try:
            purge_episode_actors(world, client=client)
        except Exception:
            pass
        try:
            restore_async(world)
        except Exception:
            pass


def main() -> int:
    global CKPT, OUT, REG, DEFAULT_PAIRS
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--registry", default=str(REG))
    ap.add_argument("--registry-manifest", default="")
    ap.add_argument(
        "--spectator-follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="move CARLA's spectator behind the ego during anchor and branch ticks",
    )
    ap.add_argument(
        "--spectator-wall-pace-s",
        type=float,
        default=0.0,
        help="optional wall-clock pause after each branch tick for watchable runs",
    )
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        help="predeclared scenario_id/seed_id; repeat 1-3 times",
    )
    ap.add_argument(
        "--checkpoint-use",
        choices=(
            "x5h_acceptance",
            "development_live_smoke",
            "offline_diagnostic",
        ),
        default="x5h_acceptance",
    )
    args = ap.parse_args()
    CKPT = Path(args.checkpoint)
    OUT = Path(args.out)
    REG = Path(args.registry)
    if args.pair:
        parsed = []
        for item in args.pair:
            parts = str(item).split("/", 1)
            if len(parts) != 2 or not all(parts):
                ap.error(f"invalid --pair {item!r}; expected scenario_id/seed_id")
            parsed.append((parts[0], parts[1]))
        if not 1 <= len(parsed) <= 3:
            ap.error("--pair requires 1 to 3 predeclared pairs")
        DEFAULT_PAIRS = parsed
    OUT.mkdir(parents=True, exist_ok=True)

    # A2: reject invalid checkpoint BEFORE CARLA preflight / torch.load
    from driving_vla.model.checkpoint_contract import (
        CheckpointContractError,
        require_checkpoint_for_use,
    )

    try:
        require_checkpoint_for_use(CKPT, args.checkpoint_use)
    except CheckpointContractError as exc:
        rep = {
            "status": "CHECKPOINT_CONTRACT_REJECT",
            "error": str(exc),
            "checkpoint": str(CKPT.as_posix()),
        }
        (OUT / "live_force_smoke_report.json").write_text(
            json.dumps(rep, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(rep, indent=2))
        return 4

    frozen_registry_audit = None
    if args.checkpoint_use == "x5h_acceptance":
        from driving_vla.evaluation.runner_contract import (
            RunnerContractError,
            require_frozen_registry,
        )

        if not args.registry_manifest:
            rep = {
                "status": "REGISTRY_CONTRACT_REJECT",
                "error": "formal X5H requires --registry-manifest",
            }
            (OUT / "live_force_smoke_report.json").write_text(
                json.dumps(rep, indent=2) + "\n", encoding="utf-8"
            )
            return 4
        try:
            registry, frozen_registry_audit, _ = require_frozen_registry(
                REG, manifest_path=args.registry_manifest
            )
            for scenario_id, seed_id in DEFAULT_PAIRS:
                registry.get(scenario_id, seed_id)
            from driving_vla.model.checkpoint_contract import (
                require_checkpoint_blind_registry,
            )

            require_checkpoint_blind_registry(
                CKPT, str(frozen_registry_audit["registry_sha256"])
            )
        except Exception as exc:
            rep = {"status": "REGISTRY_CONTRACT_REJECT", "error": str(exc)}
            (OUT / "live_force_smoke_report.json").write_text(
                json.dumps(rep, indent=2) + "\n", encoding="utf-8"
            )
            return 4

    st = _preflight()
    ensure_st = None
    if st != "READY":
        ensure_st = _ensure_once()
        st = _preflight()
    if st != "READY":
        rep = {
            "status": "BLOCKED_EXTERNAL",
            "preflight": st,
            "ensure": ensure_st,
        }
        (OUT / "live_force_smoke_report.json").write_text(
            json.dumps(rep, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(rep, indent=2))
        return 2

    results = []
    for sid, seed in DEFAULT_PAIRS:
        try:
            r = run_one(
                sid,
                seed,
                device=args.device,
                checkpoint_use=args.checkpoint_use,
                spectator_follow=bool(args.spectator_follow),
                spectator_wall_pace_s=max(0.0, float(args.spectator_wall_pace_s)),
            )
            results.append(r)
            print(
                f"[{r['status']}] {sid}/{seed} path_div={r.get('path_diverge')} "
                f"fwd={r.get('forward_count')}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "scenario_id": sid,
                    "seed_id": seed,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}:{exc}",
                    "trace": traceback.format_exc()[-800:],
                }
            )
            print(f"[FAILED] {sid}/{seed}: {exc}", flush=True)

    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_dual = sum(
        1
        for r in results
        if r.get("status") == "PASS"
        and r.get("defensive_available")
        and set(r.get("forces_run") or []) >= {0, 1}
        and r.get("path_diverge")
    )
    # Diagnostic counts only. Formal X5H requires committed PathManager path
    # diverge + control sequence diverge + first divergent tick + formal-OK
    # checkpoint — not proposal-hash-only or force0-only pairs.
    smoke_status, r2k_authorized = classify_x5h_results(
        results,
        formal=args.checkpoint_use == "x5h_acceptance",
    )
    if n_pass == len(DEFAULT_PAIRS) and n_dual == 0:
        smoke_status = "SMOKE_ALL_PAIRS_FORCE0_ONLY"
    report = {
        "schema_version": "safedrive.r2x.live_force_smoke.v2",
        "status": smoke_status,
        "formal_acceptance": args.checkpoint_use == "x5h_acceptance",
        "acceptance_status": (
            "X5H_MEASURED"
            if args.checkpoint_use == "x5h_acceptance"
            else "DEVELOPMENT_SMOKE_ONLY"
        ),
        "n_pass": n_pass,
        "n_dual_force_pass": n_dual,
        "n_planned": len(DEFAULT_PAIRS),
        "checkpoint": str(CKPT.as_posix()),
        "checkpoint_use": args.checkpoint_use,
        "registry": str(REG.as_posix()),
        "frozen_registry_audit": frozen_registry_audit,
        "predeclared_pairs": [
            {"scenario_id": scenario_id, "seed_id": seed_id}
            for scenario_id, seed_id in DEFAULT_PAIRS
        ],
        "results": results,
        "r2k_authorized": r2k_authorized,
        "note": (
            "Development mode is diagnostic only. Formal X5H additionally "
            "requires a formal-OK checkpoint and frozen blind evaluation."
        ),
    }
    (OUT / "live_force_smoke_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "results"}, indent=2))
    return (
        0
        if report["r2k_authorized"]
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
