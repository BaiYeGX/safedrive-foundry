"""Versioned R2 Oracle V2 with minimum-intervention clearance semantics.

The frozen historical Oracle remains in ``oracle.py``.  V2 saturates clearance
once both candidates are safely separated, preventing an unnecessarily large
detour from winning solely because 10 m clearance exceeds 5 m clearance.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from driving_vla.evaluation.oracle import (
    BOTH_BAD_OFFROAD,
    JERK_P95_EPS,
    LABEL_BOTH_BAD,
    LABEL_CANDIDATE_1_BEST,
    LABEL_INCOMPARABLE,
    LABEL_TIE,
    LABEL_TOP1_BEST,
    LEVEL_CLEARANCE,
    LEVEL_COLLISION,
    LEVEL_COMFORT,
    LEVEL_NEAR_CONFLICT,
    LEVEL_OFFROAD,
    LEVEL_PROGRESS,
    LEVEL_TIE_TOP1,
    OFFROAD_FRACTION_EPS,
    PROGRESS_EPS_M,
    PairOracleResult,
)
from driving_vla.evaluation.outcome_metrics import (
    BranchOutcomeMetrics,
    ttc_risk_bucket,
)

ORACLE_SCHEMA_V2 = "safedrive.r2_pair_oracle.v2"
SAFE_CLEARANCE_SATURATION_M = 2.0
CLEARANCE_DEFICIT_EPS_M = 0.50


def _has_collision(metrics: BranchOutcomeMetrics) -> bool:
    return (
        int(metrics.collision_episode_count) > 0
        or metrics.first_collision_time_s is not None
    )


def _both_bad(
    first: BranchOutcomeMetrics,
    second: BranchOutcomeMetrics,
) -> bool:
    return bool(
        (_has_collision(first) and _has_collision(second))
        or (
            first.offroad_fraction >= BOTH_BAD_OFFROAD
            and second.offroad_fraction >= BOTH_BAD_OFFROAD
        )
        or (
            not first.completed_primary_horizon
            and not second.completed_primary_horizon
        )
    )


def saturated_clearance(clearance_m: float | None) -> float | None:
    if clearance_m is None:
        return None
    return min(float(clearance_m), SAFE_CLEARANCE_SATURATION_M)


def minimum_intervention_cost(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    """Teacher-side lexicographic cost; lower is better.

    Privileged metrics may be used only to choose labels offline.
    """
    collision = bool(metrics.get("collision", False))
    offroad = float(metrics.get("offroad_fraction", 0.0))
    ttc = metrics.get("ttc_s")
    clearance = metrics.get("clearance_m")
    progress = float(metrics.get("progress_m", 0.0))
    jerk = float(metrics.get("jerk_p95", metrics.get("comfort_cost", 0.0)))
    effective_clearance = saturated_clearance(
        None if clearance is None else float(clearance)
    )
    deficit = (
        SAFE_CLEARANCE_SATURATION_M
        if effective_clearance is None
        else SAFE_CLEARANCE_SATURATION_M - effective_clearance
    )
    return (
        1 if collision else 0,
        round(offroad, 6),
        -ttc_risk_bucket(None if ttc is None else float(ttc)),
        round(deficit, 6),
        -round(progress, 6),
        round(jerk, 6),
    )


def _compare(
    first: BranchOutcomeMetrics,
    second: BranchOutcomeMetrics,
) -> tuple[int, str, str]:
    collision0, collision1 = _has_collision(first), _has_collision(second)
    if collision0 != collision1:
        return (
            (1 if collision0 else 0),
            LEVEL_COLLISION,
            "collision_presence",
        )
    if collision0 and collision1:
        time0, time1 = first.first_collision_time_s, second.first_collision_time_s
        if time0 is not None and time1 is not None and abs(time0 - time1) > 1.0e-9:
            return (
                0 if time0 > time1 else 1,
                LEVEL_COLLISION,
                f"first_collision_time {time0:.3f} vs {time1:.3f}",
            )

    if abs(first.offroad_fraction - second.offroad_fraction) > OFFROAD_FRACTION_EPS:
        return (
            0 if first.offroad_fraction < second.offroad_fraction else 1,
            LEVEL_OFFROAD,
            (
                f"offroad_fraction {first.offroad_fraction:.4f} "
                f"vs {second.offroad_fraction:.4f}"
            ),
        )

    bucket0 = ttc_risk_bucket(first.minimum_ttc_s)
    bucket1 = ttc_risk_bucket(second.minimum_ttc_s)
    if bucket0 != bucket1:
        return (
            0 if bucket0 > bucket1 else 1,
            LEVEL_NEAR_CONFLICT,
            f"ttc_bucket {bucket0} vs {bucket1}",
        )

    clearance0 = saturated_clearance(first.minimum_actor_clearance_m)
    clearance1 = saturated_clearance(second.minimum_actor_clearance_m)
    if (
        clearance0 is not None
        and clearance1 is not None
        and abs(clearance0 - clearance1) > CLEARANCE_DEFICIT_EPS_M
    ):
        return (
            0 if clearance0 > clearance1 else 1,
            LEVEL_CLEARANCE,
            (
                "saturated_clearance "
                f"{clearance0:.3f} vs {clearance1:.3f}; "
                "raw "
                f"{first.minimum_actor_clearance_m:.3f} vs "
                f"{second.minimum_actor_clearance_m:.3f}"
            ),
        )

    if (
        abs(first.route_progress_delta_m - second.route_progress_delta_m)
        > PROGRESS_EPS_M
    ):
        return (
            0
            if first.route_progress_delta_m > second.route_progress_delta_m
            else 1,
            LEVEL_PROGRESS,
            (
                f"progress {first.route_progress_delta_m:.3f} "
                f"vs {second.route_progress_delta_m:.3f}"
            ),
        )

    if abs(first.jerk_abs_p95 - second.jerk_abs_p95) > JERK_P95_EPS:
        return (
            0 if first.jerk_abs_p95 < second.jerk_abs_p95 else 1,
            LEVEL_COMFORT,
            f"jerk_p95 {first.jerk_abs_p95:.3f} vs {second.jerk_abs_p95:.3f}",
        )
    return -1, LEVEL_TIE_TOP1, "exact_tie_return_top1"


def evaluate_pair_oracle_v2(
    *,
    pair_id: str,
    scenario_id: str,
    seed_id: str,
    family: str,
    comparable: bool,
    top1_index: int,
    metrics0: BranchOutcomeMetrics | None,
    metrics1: BranchOutcomeMetrics | None,
    incomparable_reasons: Sequence[str] = (),
    candidate_ids: tuple[str, str] = (
        "v3_nominal_progress",
        "v3_alternative",
    ),
) -> PairOracleResult:
    if int(top1_index) not in (0, 1):
        raise ValueError("top1_index must be 0 or 1")
    top1_index = int(top1_index)
    if not comparable or metrics0 is None or metrics1 is None:
        return PairOracleResult(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            family=family,
            comparable=False,
            top1_candidate_id=candidate_ids[top1_index],
            top1_candidate_index=top1_index,
            oracle_candidate_id=None,
            oracle_candidate_index=None,
            oracle_decision_level=None,
            decision_reason="incomparable",
            pair_label=LABEL_INCOMPARABLE,
            both_bad=False,
            outcome_delta={"oracle_schema": ORACLE_SCHEMA_V2},
            failure_reasons=tuple(incomparable_reasons),
        )

    winner, level, reason = _compare(metrics0, metrics1)
    both_bad = _both_bad(metrics0, metrics1)
    oracle_index = top1_index if winner < 0 else winner
    if winner < 0:
        pair_label = LABEL_TIE
    elif oracle_index == 1:
        pair_label = LABEL_CANDIDATE_1_BEST
    else:
        pair_label = LABEL_TOP1_BEST
    relative = None
    if both_bad:
        pair_label = LABEL_BOTH_BAD
        relative = (
            metrics0.candidate_id
            if oracle_index == 0
            else metrics1.candidate_id
        )
    ids = (metrics0.candidate_id, metrics1.candidate_id)
    return PairOracleResult(
        pair_id=pair_id,
        scenario_id=scenario_id,
        seed_id=seed_id,
        family=family,
        comparable=True,
        top1_candidate_id=ids[top1_index],
        top1_candidate_index=top1_index,
        oracle_candidate_id=ids[oracle_index],
        oracle_candidate_index=oracle_index,
        oracle_decision_level=level,
        decision_reason=reason,
        pair_label=pair_label,
        both_bad=both_bad,
        outcome_delta={
            "oracle_schema": ORACLE_SCHEMA_V2,
            "safe_clearance_saturation_m": SAFE_CLEARANCE_SATURATION_M,
            "raw_clearance_m": [
                metrics0.minimum_actor_clearance_m,
                metrics1.minimum_actor_clearance_m,
            ],
            "saturated_clearance_m": [
                saturated_clearance(metrics0.minimum_actor_clearance_m),
                saturated_clearance(metrics1.minimum_actor_clearance_m),
            ],
            "route_progress_delta_m": [
                metrics0.route_progress_delta_m,
                metrics1.route_progress_delta_m,
            ],
            "winner_index": oracle_index,
            "lex_level": level,
        },
        failure_reasons=(),
        relative_winner_if_both_bad=relative,
    )


__all__ = [
    "CLEARANCE_DEFICIT_EPS_M",
    "ORACLE_SCHEMA_V2",
    "SAFE_CLEARANCE_SATURATION_M",
    "evaluate_pair_oracle_v2",
    "minimum_intervention_cost",
    "saturated_clearance",
]
