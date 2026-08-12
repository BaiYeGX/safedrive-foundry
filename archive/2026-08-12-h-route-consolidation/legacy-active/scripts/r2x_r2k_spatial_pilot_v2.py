#!/usr/bin/env python3
"""R2-K Spatial K2 pilot: 12-pair force exam on frozen registry (new Evidence).

Does not overwrite longitudinal r2-g4a-paired-pilot.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "scripts"))

# Do not overwrite frozen interrupted pilot-v2; formal run writes pilot-v3.
OUT = ROOT / "docs/runtime-evidence/r2-spatial-k2-pilot-v3"
REG = ROOT / "safedrive_foundry/config/g4a/scenario_registry_v1.toml"


def main() -> int:
    import r2x_live_force_smoke_v2 as smoke
    from driving_vla.evaluation.scenario_registry import load_scenario_registry
    from driving_vla.model.checkpoint_contract import (
        CheckpointContractError,
        require_checkpoint_for_use,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    # A2: fail-closed before CARLA / torch.load
    try:
        require_checkpoint_for_use(smoke.CKPT, "r2k_pilot")
    except CheckpointContractError as exc:
        rep = {
            "status": "CHECKPOINT_CONTRACT_REJECT",
            "error": str(exc),
            "checkpoint": str(smoke.CKPT.as_posix()),
            "r2k_authorized": False,
        }
        (OUT / "r2k_closure_report.json").write_text(
            json.dumps(rep, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(rep, indent=2))
        return 4
    reg = load_scenario_registry(REG)
    pairs = [(f.scenario_id, f.seed_id) for f in reg.fixtures]
    # freeze pointer
    freeze = {
        "schema_version": "safedrive.r2k.spatial.v2.freeze",
        "registry": str(REG.as_posix()),
        "registry_sha256": reg.registry_sha256,
        "n_pairs": len(pairs),
        "pairs": [{"scenario_id": a, "seed_id": b} for a, b in pairs],
        "checkpoint": str(smoke.CKPT.as_posix()),
        "note": "exam only; not train data",
    }
    (OUT / "freeze_manifest.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    # redirect smoke OUT
    smoke.OUT = OUT / "live_pairs"
    smoke.OUT.mkdir(parents=True, exist_ok=True)
    smoke.DEFAULT_PAIRS = pairs

    st = smoke._preflight()
    if st != "READY":
        smoke._ensure_once()
        st = smoke._preflight()
    if st != "READY":
        rep = {"status": "BLOCKED_EXTERNAL", "preflight": st}
        (OUT / "r2k_closure_report.json").write_text(
            json.dumps(rep, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(rep, indent=2))
        return 2

    results = []
    for sid, seed in pairs:
        attempt = 0
        while attempt < 2:
            attempt += 1
            try:
                r = smoke.run_one(sid, seed, device="cuda")
                results.append(r)
                print(
                    f"[{r.get('status')}] {sid}/{seed} def={r.get('defensive_available')} "
                    f"path_div={r.get('path_diverge')}",
                    flush=True,
                )
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}:{exc}"
                conn_fail = any(
                    x in err.lower()
                    for x in (
                        "time-out",
                        "timeout",
                        "connect_world",
                        "rpc",
                        "simulator",
                    )
                )
                if conn_fail and attempt < 2:
                    print(
                        f"[RETRY] {sid}/{seed} after CARLA error; ensure…",
                        flush=True,
                    )
                    smoke._ensure_once()
                    st2 = smoke._preflight()
                    if st2 != "READY":
                        results.append(
                            {
                                "scenario_id": sid,
                                "seed_id": seed,
                                "status": "FAILED",
                                "error": f"ensure_after_fail:{st2}:{err}",
                            }
                        )
                        print(f"[FAILED] {sid}/{seed}: ensure {st2}", flush=True)
                        break
                    continue
                results.append(
                    {
                        "scenario_id": sid,
                        "seed_id": seed,
                        "status": "FAILED",
                        "error": err,
                        "trace": traceback.format_exc()[-600:],
                    }
                )
                print(f"[FAILED] {sid}/{seed}: {exc}", flush=True)
                break

    n = len(results)
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_dual = sum(
        1
        for r in results
        if r.get("status") == "PASS"
        and r.get("defensive_available")
        and (r.get("path_diverge") or r.get("path_div"))
        and set(r.get("forces_run") or []) >= {0, 1}
    )
    n_fail = sum(1 for r in results if r.get("status") not in {"PASS"})
    # This script is a force-smoke loop, NOT a formal R2-K paired pilot.
    # Do not emit frozen pilot labels (ENTER_WORLD / WEAK_SELECTION_SPACE /
    # NO_SELECTION_SPACE / IMPROVE_VLA / PILOT_INCONCLUSIVE) — those require
    # runner_contract + Oracle + comparability + repeats.
    closure = {
        "schema_version": "safedrive.r2k.spatial.smoke_loop.v1",
        "status": "SMOKE_LOOP_DIAGNOSTIC_ONLY",
        "formal_r2k": False,
        "pilot_label": None,
        "pilot_label_note": (
            "no_formal_pilot_label: smoke_loop_lacks_comparability_oracle_"
            "decisive_wins_repeat_runner_contract"
        ),
        "n_planned": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_dual_force_pass": n_dual,
        "registry_sha256": reg.registry_sha256,
        "evidence": str(OUT.as_posix()),
        "longitudinal_r2_readonly": "docs/runtime-evidence/r2-g4a-paired-pilot",
        "r3_auto_start": False,
        "results": results,
    }
    (OUT / "r2k_closure_report.json").write_text(
        json.dumps(closure, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: closure[k] for k in closure if k != "results"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
