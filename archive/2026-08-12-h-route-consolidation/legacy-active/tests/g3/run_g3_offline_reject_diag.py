#!/usr/bin/env python3
"""Offline diagnosis: route/V0-like candidates through SafetyKernel (no CARLA).

Writes reject reasons / decision kinds so live wiring can be fixed without bypass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle, arrays_to_candidate_set  # noqa: E402
from driving_vla.baselines.route_ego import RouteEgoBaseline  # noqa: E402
from driving_vla.runtime.mode import RuntimeMode, availability_for_mode, filter_candidates_for_mode  # noqa: E402
from safety_kernel import SafetyKernel, load_safety_config  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
)


def main() -> int:
    out = ROOT / "docs/runtime-evidence/g3-05/offline_reject_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    route = tuple((float(i) * 2.0, 0.0) for i in range(40))
    obs = ObservationBundle(
        run_id="diag",
        frame_id="f0",
        scenario_id="offline_reject",
        simulation_time_s=10.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=4.0,
        route_xy=route,
    )
    bl = RouteEgoBaseline()
    arrs = bl.predict(obs)
    cset = arrays_to_candidate_set(
        arrs, obs, model_id=bl.model_id, source=CandidateSource.VLA_FAST, valid_for_s=0.40
    )
    cset = filter_candidates_for_mode(cset, RuntimeMode.VLA_SAFETY)

    snap = ObservableSnapshot(
        run_id="diag",
        frame_id="f0",
        scenario_id="offline_reject",
        simulation_time_s=10.0,
        wall_time_s=0.0,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=4.0,
        observed_time_s=10.0,
        freshness_s=0.0,
        corridor_centerline=route[:40],
        corridor_half_width_m=2.5,
        privilege=ObservationPrivilege.OBSERVABLE,
    )
    kernel = SafetyKernel(load_safety_config())
    res = kernel.tick(
        snap,
        cset,
        now_s=10.0,
        availability=availability_for_mode(RuntimeMode.VLA_SAFETY, vla_ok=True),
    )
    d = res.decision
    report = {
        "source": "route_ego_baseline_as_vla_fast",
        "n_candidates": len(cset.candidates),
        "last_t": cset.candidates[0].points[-1].t if cset.candidates else None,
        "horizon_span": cset.candidates[0].horizon_s if cset.candidates else None,
        "decision_kind": str(getattr(d.decision_kind, "value", d.decision_kind)),
        "executed_trajectory_id": d.executed_trajectory_id,
        "reject_reasons": list(d.reject_reasons),
        "state_after": str(getattr(d.state_after, "value", d.state_after)),
        "valid_for_s": 0.40,
        "notes": [
            "Horizon contract last_t=2.5; Safety min_horizon_s=2.0 uses span≈2.25",
            "If decision is not ACCEPT/QP, inspect reject_reasons before live",
        ],
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
