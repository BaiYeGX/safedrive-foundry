#!/usr/bin/env python3
"""G3-02 offline baseline report: ADE/FDE vs route expert + Safety accept rate."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.adapter.policy_adapter import ObservationBundle, arrays_to_candidate_set  # noqa: E402
from driving_vla.baselines.multi_k2 import MultiCandidateK2Interface  # noqa: E402
from driving_vla.baselines.route_ego import RouteEgoBaseline  # noqa: E402
from driving_vla.baselines.vision_k1 import VisionK1Baseline  # noqa: E402
from safety_kernel import SafetyKernel, load_safety_config  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservableSnapshot,
    ObservationPrivilege,
)


def _ade_fde(pred, expert) -> tuple[float, float]:
    n = min(len(pred), len(expert))
    if n == 0:
        return float("nan"), float("nan")
    errs = []
    for i in range(n):
        dx = pred[i][0] - expert[i][0]
        dy = pred[i][1] - expert[i][1]
        errs.append(math.hypot(dx, dy))
    return sum(errs) / n, errs[-1]


def main() -> int:
    out = ROOT / "docs/architecture/evidence/g3-02/baseline_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    models = {
        "route_ego": (RouteEgoBaseline(), CandidateSource.CLASSIC),
        "vision_k1": (VisionK1Baseline(), CandidateSource.VLA_FAST),
        "multi_k2": (MultiCandidateK2Interface(), CandidateSource.VLA_FAST),
    }
    kernel = SafetyKernel(load_safety_config())
    rows = []
    for name, (model, src) in models.items():
        ades, fdes = [], []
        accept = 0
        n = 8
        t0 = time.perf_counter()
        for i in range(n):
            route = tuple((float(j) * 2.0, 0.1 * math.sin(j * 0.2)) for j in range(40))
            obs = ObservationBundle(
                run_id="g302",
                frame_id=f"f{i}",
                scenario_id="split_fixed",
                simulation_time_s=float(i),
                ego_x=0.0,
                ego_y=0.0,
                ego_yaw=0.0,
                ego_v=4.0 + 0.2 * i,
                route_xy=route,
                front_rgb=None,
                ego_history=tuple((0.1 * k, 0.0, 0.0, 3.5 + 0.1 * k) for k in range(4)),
                meta={"image_scalar_history": [0.4, 0.45, 0.5]},
            )
            expert = RouteEgoBaseline().predict(obs)[0]
            arrs = model.predict(obs)
            ade, fde = _ade_fde(arrs[0].points_xy_yaw_v_a_kappa, expert.points_xy_yaw_v_a_kappa)
            ades.append(ade)
            fdes.append(fde)
            cset = arrays_to_candidate_set(arrs, obs, model_id=model.model_id, source=src)
            snap = ObservableSnapshot(
                run_id="g302",
                frame_id=f"f{i}",
                scenario_id="split_fixed",
                simulation_time_s=float(i),
                wall_time_s=0.0,
                ego_x=0.0,
                ego_y=0.0,
                ego_yaw=0.0,
                ego_v=obs.ego_v,
                observed_time_s=float(i),
                corridor_centerline=route[:40],
                corridor_half_width_m=2.5,
                privilege=ObservationPrivilege.OBSERVABLE,
            )
            res = kernel.tick(snap, cset, now_s=float(i))
            kind = str(getattr(res.decision.decision_kind, "value", res.decision.decision_kind))
            if kind in {"ACCEPT", "QP", "RATO"}:
                accept += 1
        elapsed = time.perf_counter() - t0
        budget = getattr(model, "encoder_budget", {"params_m": 0.0, "kind": "n/a"})
        rows.append(
            {
                "model": name,
                "model_id": model.model_id,
                "k": getattr(model, "k", len(arrs)),
                "n": n,
                "ade_mean": sum(ades) / len(ades),
                "fde_mean": sum(fdes) / len(fdes),
                "safety_accept_rate": accept / n,
                "elapsed_s": elapsed,
                "encoder_budget": budget,
            }
        )

    report = {
        "schema": "g3_02_baseline_report_v1",
        "split": "fixed_synthetic_offline",
        "metrics": rows,
        "config_hash": hashlib.sha256(
            json.dumps(rows, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "notes": [
            "Expert = RouteEgo on same obs (relative isolation of vision/K2 heads)",
            "Not a live CARLA score; open-loop offline only",
        ],
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
