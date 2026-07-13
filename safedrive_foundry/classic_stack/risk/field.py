"""Versioned RiskField from CV/CTRV/IDM-style uncertainty envelopes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RiskSample:
    t: float
    s: float
    d: float
    collision_prob: float
    ttc_s: float
    thw_s: float
    clearance_m: float
    road_margin_m: float
    uncertainty: float
    trackability: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class RiskField:
    schema_version: str
    prediction_model: str
    samples: list[RiskSample]
    oracle_upper_bound: float | None = None
    observable_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prediction_model": self.prediction_model,
            "samples": [s.to_dict() for s in self.samples],
            "oracle_upper_bound": self.oracle_upper_bound,
            "observable_score": self.observable_score,
        }


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-6:
        return default
    return num / den


def evaluate_risk_field(
    *,
    ego_v: float,
    actors: Sequence[dict[str, float]],
    road_half_width_m: float = 3.5,
    lateral_offset_m: float = 0.0,
    prediction_model: str = "cv_ctrv_idm",
    uncertainty_scale: float = 1.0,
    horizon_s: float = 4.0,
    dt_s: float = 0.2,
    track: str = "observable",  # observable|oracle — input path separation
    _compute_oracle: bool = True,
) -> RiskField:
    """Return non-negative finite risk scores.

    - gap<=0 (already past actor) → high collision risk, not divergent TTC.
    - Oracle uses reduced uncertainty; Observable uses reported scale.
    """

    samples: list[RiskSample] = []
    # Track separation: oracle uses lower uncertainty internally
    unc_scale = uncertainty_scale
    if track == "oracle":
        unc_scale = max(0.1, uncertainty_scale * 0.3)

    t = 0.0
    while t <= horizon_s + 1e-9:
        best_gap = 50.0
        best_ttc = 30.0
        for a in actors:
            gap = float(a.get("s", 20.0)) - max(0.0, ego_v) * t
            rel_v = max(0.0, ego_v) - float(a.get("v", 0.0))
            if gap <= 0.0:
                # already reached / passed actor station
                ttc = 0.0
                gap = min(gap, 0.0)
            elif rel_v > 1e-3:
                ttc = gap / rel_v
            else:
                ttc = 30.0
            best_gap = min(best_gap, gap)
            best_ttc = min(best_ttc, max(0.0, ttc))

        clearance = max(0.0, best_gap)
        thw = _safe_div(clearance, max(ego_v, 0.1), default=30.0)
        thw = max(0.0, min(thw, 30.0))
        road_margin = max(0.0, road_half_width_m - abs(lateral_offset_m))
        unc = max(0.0, unc_scale * (0.2 + 0.1 * t))
        # Bound each term to keep total finite and non-negative
        coll = 1.0 / (1.0 + max(best_ttc, 1e-3))
        if best_gap <= 0.0:
            coll = 1.0
        trackability = 1.0 / (1.0 + abs(lateral_offset_m) + unc)
        total = (
            2.0 * coll
            + 1.0 / (1.0 + best_ttc)
            + 0.5 / (1.0 + thw)
            + 0.8 / (1.0 + clearance)
            + 0.5 / (1.0 + road_margin)
            + 0.6 * unc
            + 0.3 * (1.0 - trackability)
        )
        total = max(0.0, min(total, 1e6))
        samples.append(
            RiskSample(
                t=t,
                s=max(0.0, ego_v) * t,
                d=lateral_offset_m,
                collision_prob=coll,
                ttc_s=best_ttc,
                thw_s=thw,
                clearance_m=clearance,
                road_margin_m=road_margin,
                uncertainty=unc,
                trackability=trackability,
                total=total,
            )
        )
        t += dt_s

    obs = sum(s.total for s in samples) / max(1, len(samples))
    obs = max(0.0, min(obs, 1e6))
    oracle = None
    if _compute_oracle and track == "observable":
        oracle_field = evaluate_risk_field(
            ego_v=ego_v,
            actors=actors,
            road_half_width_m=road_half_width_m,
            lateral_offset_m=lateral_offset_m,
            prediction_model=prediction_model,
            uncertainty_scale=uncertainty_scale,
            horizon_s=horizon_s,
            dt_s=dt_s,
            track="oracle",
            _compute_oracle=False,
        )
        oracle = oracle_field.observable_score
    return RiskField(
        schema_version="safedrive.risk.v1",
        prediction_model=prediction_model,
        samples=samples,
        oracle_upper_bound=oracle,
        observable_score=obs,
    )


def monotonicity_ok(
    actors: Sequence[dict[str, float]],
    *,
    ego_v: float = 8.0,
    scales: Sequence[float] = (0.5, 1.0, 2.0),
) -> bool:
    """Safety margin proxy should worsen (risk total rise) as uncertainty rises."""

    scores = []
    for sc in scales:
        field = evaluate_risk_field(ego_v=ego_v, actors=actors, uncertainty_scale=sc)
        scores.append(field.observable_score or 0.0)
    return all(scores[i] <= scores[i + 1] + 1e-6 for i in range(len(scores) - 1))
