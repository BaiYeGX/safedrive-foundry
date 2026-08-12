"""Offline lexicographic oracle best-of-K (R2 §8–9).

Does not read R1 probability priors. Must not be imported by runtime control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from driving_vla.evaluation.outcome_metrics import BranchOutcomeMetrics, ttc_risk_bucket

# Pre-registered thresholds (R2 §8.2) — frozen before seeing pilot outcomes.
OFFROAD_FRACTION_EPS = 0.02
CLEARANCE_EPS_M = 0.50
PROGRESS_EPS_M = 0.50
JERK_P95_EPS = 1.0
BOTH_BAD_OFFROAD = 0.10

LABEL_TOP1_BEST = "TOP1_BEST"
LABEL_CANDIDATE_1_BEST = "CANDIDATE_1_BEST"
LABEL_TIE = "TIE"
LABEL_BOTH_BAD = "BOTH_BAD"
LABEL_INCOMPARABLE = "INCOMPARABLE"

PILOT_IMPROVE_VLA = "IMPROVE_VLA"
PILOT_ENTER_WORLD = "ENTER_WORLD"
PILOT_WEAK = "WEAK_SELECTION_SPACE"
PILOT_NONE = "NO_SELECTION_SPACE"
PILOT_INCONCLUSIVE = "PILOT_INCONCLUSIVE"

# Decision level names for audit
LEVEL_COLLISION = "collision"
LEVEL_OFFROAD = "offroad"
LEVEL_NEAR_CONFLICT = "near_conflict"
LEVEL_CLEARANCE = "clearance"
LEVEL_PROGRESS = "progress"
LEVEL_COMFORT = "comfort"
LEVEL_TIE_TOP1 = "tie_top1"


@dataclass(frozen=True)
class PairOracleResult:
    pair_id: str
    scenario_id: str
    seed_id: str
    family: str
    comparable: bool
    top1_candidate_id: str
    top1_candidate_index: int
    oracle_candidate_id: str | None
    oracle_candidate_index: int | None
    oracle_decision_level: str | None
    decision_reason: str
    pair_label: str
    both_bad: bool
    outcome_delta: Mapping[str, Any]
    failure_reasons: tuple[str, ...] = ()
    relative_winner_if_both_bad: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "seed_id": self.seed_id,
            "family": self.family,
            "comparable": self.comparable,
            "top1_candidate_id": self.top1_candidate_id,
            "top1_candidate_index": int(self.top1_candidate_index),
            "oracle_candidate_id": self.oracle_candidate_id,
            "oracle_candidate_index": self.oracle_candidate_index,
            "oracle_decision_level": self.oracle_decision_level,
            "decision_reason": self.decision_reason,
            "pair_label": self.pair_label,
            "both_bad": bool(self.both_bad),
            "outcome_delta": dict(self.outcome_delta),
            "failure_reasons": list(self.failure_reasons),
            "relative_winner_if_both_bad": self.relative_winner_if_both_bad,
        }


@dataclass(frozen=True)
class PilotSummary:
    label: str
    n_pairs: int
    n_comparable: int
    comparable_rate: float
    n_top1_best: int
    n_candidate1_best: int
    n_tie: int
    n_both_bad: int
    n_incomparable: int
    candidate1_win_families: tuple[str, ...]
    decisive_comparable: int
    reasons: tuple[str, ...]
    counts_by_family: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_pairs": int(self.n_pairs),
            "n_comparable": int(self.n_comparable),
            "comparable_rate": float(self.comparable_rate),
            "n_top1_best": int(self.n_top1_best),
            "n_candidate1_best": int(self.n_candidate1_best),
            "n_tie": int(self.n_tie),
            "n_both_bad": int(self.n_both_bad),
            "n_incomparable": int(self.n_incomparable),
            "candidate1_win_families": list(self.candidate1_win_families),
            "decisive_comparable": int(self.decisive_comparable),
            "reasons": list(self.reasons),
            "counts_by_family": {k: dict(v) for k, v in self.counts_by_family.items()},
        }


def _has_collision(m: BranchOutcomeMetrics) -> bool:
    return int(m.collision_episode_count) > 0 or m.first_collision_time_s is not None


def is_both_bad(m0: BranchOutcomeMetrics, m1: BranchOutcomeMetrics) -> bool:
    if _has_collision(m0) and _has_collision(m1):
        return True
    if m0.offroad_fraction >= BOTH_BAD_OFFROAD and m1.offroad_fraction >= BOTH_BAD_OFFROAD:
        return True
    incomplete0 = not m0.completed_primary_horizon
    incomplete1 = not m1.completed_primary_horizon
    if incomplete0 and incomplete1:
        return True
    return False


def _compare_lexicographic(
    m0: BranchOutcomeMetrics,
    m1: BranchOutcomeMetrics,
) -> tuple[int, str, str]:
    """Return (winner_index, level, reason). winner_index is 0, 1, or -1 for tie."""

    # 1. collision
    c0, c1 = _has_collision(m0), _has_collision(m1)
    if c0 != c1:
        winner = 1 if c0 and not c1 else 0
        return winner, LEVEL_COLLISION, "collision_presence"
    if c0 and c1:
        t0 = m0.first_collision_time_s
        t1 = m1.first_collision_time_s
        # later collision is better; None should not happen if has collision
        if t0 is not None and t1 is not None and abs(t0 - t1) > 1e-9:
            winner = 0 if t0 > t1 else 1
            return winner, LEVEL_COLLISION, f"first_collision_time {t0:.3f} vs {t1:.3f}"

    # 2. off-road
    if abs(m0.offroad_fraction - m1.offroad_fraction) > OFFROAD_FRACTION_EPS:
        winner = 0 if m0.offroad_fraction < m1.offroad_fraction else 1
        return (
            winner,
            LEVEL_OFFROAD,
            f"offroad_fraction {m0.offroad_fraction:.4f} vs {m1.offroad_fraction:.4f}",
        )

    # 3. near conflict TTC buckets (lower bucket worse)
    b0 = ttc_risk_bucket(m0.minimum_ttc_s)
    b1 = ttc_risk_bucket(m1.minimum_ttc_s)
    if b0 != b1:
        winner = 0 if b0 > b1 else 1
        return winner, LEVEL_NEAR_CONFLICT, f"ttc_bucket {b0} vs {b1}"

    # 4. minimum clearance (larger better); None treated as missing → no decisive
    cl0, cl1 = m0.minimum_actor_clearance_m, m1.minimum_actor_clearance_m
    if cl0 is not None and cl1 is not None and abs(cl0 - cl1) > CLEARANCE_EPS_M:
        winner = 0 if cl0 > cl1 else 1
        return winner, LEVEL_CLEARANCE, f"clearance {cl0:.3f} vs {cl1:.3f}"

    # 5. route progress
    if abs(m0.route_progress_delta_m - m1.route_progress_delta_m) > PROGRESS_EPS_M:
        winner = 0 if m0.route_progress_delta_m > m1.route_progress_delta_m else 1
        return (
            winner,
            LEVEL_PROGRESS,
            f"progress {m0.route_progress_delta_m:.3f} vs {m1.route_progress_delta_m:.3f}",
        )

    # 6. comfort (jerk P95 lower better)
    if abs(m0.jerk_abs_p95 - m1.jerk_abs_p95) > JERK_P95_EPS:
        winner = 0 if m0.jerk_abs_p95 < m1.jerk_abs_p95 else 1
        return winner, LEVEL_COMFORT, f"jerk_p95 {m0.jerk_abs_p95:.3f} vs {m1.jerk_abs_p95:.3f}"

    # 7. exact tie → deterministic top-1 (index 0)
    return -1, LEVEL_TIE_TOP1, "exact_tie_return_top1"


def evaluate_pair_oracle(
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
) -> PairOracleResult:
    """Produce one pair oracle row. Probability is never consulted."""
    top1_index = int(top1_index)
    if top1_index not in (0, 1):
        raise ValueError("top1_index must be 0 or 1")

    if not comparable or metrics0 is None or metrics1 is None:
        return PairOracleResult(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            family=family,
            comparable=False,
            top1_candidate_id="v1_nominal" if top1_index == 0 else "v1_conservative",
            top1_candidate_index=top1_index,
            oracle_candidate_id=None,
            oracle_candidate_index=None,
            oracle_decision_level=None,
            decision_reason="incomparable",
            pair_label=LABEL_INCOMPARABLE,
            both_bad=False,
            outcome_delta={},
            failure_reasons=tuple(incomparable_reasons),
        )

    both = is_both_bad(metrics0, metrics1)
    winner, level, reason = _compare_lexicographic(metrics0, metrics1)
    if winner < 0:
        # tie: oracle returns top1 for execution semantics, label remains TIE
        oracle_idx = top1_index
        pair_label = LABEL_TIE
        level = LEVEL_TIE_TOP1
    else:
        oracle_idx = winner
        if winner == top1_index:
            pair_label = LABEL_TOP1_BEST
        else:
            pair_label = LABEL_CANDIDATE_1_BEST if winner == 1 else LABEL_TOP1_BEST
            # If top1 is not 0, map carefully: CANDIDATE_1_BEST means index 1 wins
            if winner == 1:
                pair_label = LABEL_CANDIDATE_1_BEST
            else:
                pair_label = LABEL_TOP1_BEST if top1_index == 0 else LABEL_TOP1_BEST

    # BOTH_BAD is a co-label; primary label becomes BOTH_BAD only as pair_label when both bad?
    # Spec: BOTH_BAD is a pair label; still may have relative oracle winner.
    relative = None
    if both:
        relative = metrics0.candidate_id if oracle_idx == 0 else metrics1.candidate_id
        # Keep relative winner but surface BOTH_BAD as pair_label
        pair_label = LABEL_BOTH_BAD

    id0 = metrics0.candidate_id
    id1 = metrics1.candidate_id
    top1_id = id0 if top1_index == 0 else id1
    oracle_id = id0 if oracle_idx == 0 else id1

    delta = {
        "collision_0": _has_collision(metrics0),
        "collision_1": _has_collision(metrics1),
        "first_collision_time_s": [metrics0.first_collision_time_s, metrics1.first_collision_time_s],
        "offroad_fraction": [metrics0.offroad_fraction, metrics1.offroad_fraction],
        "minimum_ttc_s": [metrics0.minimum_ttc_s, metrics1.minimum_ttc_s],
        "ttc_bucket": [ttc_risk_bucket(metrics0.minimum_ttc_s), ttc_risk_bucket(metrics1.minimum_ttc_s)],
        "minimum_actor_clearance_m": [
            metrics0.minimum_actor_clearance_m,
            metrics1.minimum_actor_clearance_m,
        ],
        "route_progress_delta_m": [
            metrics0.route_progress_delta_m,
            metrics1.route_progress_delta_m,
        ],
        "jerk_abs_p95": [metrics0.jerk_abs_p95, metrics1.jerk_abs_p95],
        "winner_index": oracle_idx,
        "lex_level": level,
    }

    return PairOracleResult(
        pair_id=pair_id,
        scenario_id=scenario_id,
        seed_id=seed_id,
        family=family,
        comparable=True,
        top1_candidate_id=top1_id,
        top1_candidate_index=top1_index,
        oracle_candidate_id=oracle_id,
        oracle_candidate_index=oracle_idx,
        oracle_decision_level=level,
        decision_reason=reason,
        pair_label=pair_label,
        both_bad=both,
        outcome_delta=delta,
        failure_reasons=(),
        relative_winner_if_both_bad=relative,
    )


def assign_pilot_label(results: Sequence[PairOracleResult]) -> PilotSummary:
    """Assign one primary pilot label (R2 §9)."""
    n = len(results)
    comparable = [r for r in results if r.comparable]
    n_comp = len(comparable)
    rate = (n_comp / n) if n else 0.0

    n_top1 = sum(1 for r in comparable if r.pair_label == LABEL_TOP1_BEST)
    n_c1 = sum(1 for r in comparable if r.pair_label == LABEL_CANDIDATE_1_BEST)
    # BOTH_BAD rows: also count relative wins for candidate1 if relative winner is index 1
    n_tie = sum(1 for r in comparable if r.pair_label == LABEL_TIE)
    n_both = sum(1 for r in comparable if r.both_bad or r.pair_label == LABEL_BOTH_BAD)
    n_incomp = sum(1 for r in results if not r.comparable)

    # Candidate-1 wins: label CANDIDATE_1_BEST, or BOTH_BAD with relative winner index 1
    c1_win_rows = [
        r
        for r in comparable
        if r.pair_label == LABEL_CANDIDATE_1_BEST
        or (
            r.both_bad
            and r.oracle_candidate_index == 1
            and r.pair_label == LABEL_BOTH_BAD
        )
    ]
    # Spec §9 uses "candidate 1 at least in 2 pair 获胜" — count decisive oracle winner index==1
    c1_oracle_wins = [r for r in comparable if r.oracle_candidate_index == 1 and r.pair_label != LABEL_TIE]
    # For pilot, candidate1 win count prefers explicit CANDIDATE_1_BEST; also count
    # oracle_candidate_index==1 when not tie/both-bad-only? Use oracle index for ENTER_WORLD.
    c1_wins = [r for r in comparable if r.oracle_candidate_index == 1 and not (
        r.pair_label == LABEL_TIE
    )]
    # Exclude pure TOP1 when oracle returns top1; include BOTH_BAD relative
    c1_wins = [
        r
        for r in comparable
        if r.oracle_candidate_index == 1 and r.pair_label in {LABEL_CANDIDATE_1_BEST, LABEL_BOTH_BAD}
    ]
    # Also when top1 is 0 and oracle picks 1 without both_bad → CANDIDATE_1_BEST already
    c1_wins = [r for r in comparable if r.oracle_candidate_index == 1 and r.pair_label != LABEL_TIE]

    families_c1 = tuple(sorted({r.family for r in c1_wins}))
    decisive = [
        r
        for r in comparable
        if r.pair_label in {LABEL_TOP1_BEST, LABEL_CANDIDATE_1_BEST, LABEL_BOTH_BAD}
        and r.pair_label != LABEL_TIE
        and r.oracle_decision_level != LEVEL_TIE_TOP1
    ]
    # Decisive = non-tie comparable
    decisive = [r for r in comparable if r.pair_label != LABEL_TIE]
    n_decisive = len(decisive)

    by_family: dict[str, dict[str, int]] = {}
    for r in results:
        fam = r.family or "unknown"
        by_family.setdefault(fam, {})
        by_family[fam][r.pair_label] = by_family[fam].get(r.pair_label, 0) + 1

    reasons: list[str] = []
    label = PILOT_INCONCLUSIVE

    # 1. IMPROVE_VLA
    both_rate = (n_both / n_comp) if n_comp else 0.0
    if n_comp > 0 and both_rate >= 0.50:
        label = PILOT_IMPROVE_VLA
        reasons.append(f"both_bad_rate={both_rate:.2f}>=0.50")
    elif n > 0 and n_comp / n < 0.50:
        # anchor collapse / guard reject making executable pairs scarce
        label = PILOT_IMPROVE_VLA
        reasons.append(f"comparable_rate={rate:.2f}<0.50")
    # 2. ENTER_WORLD
    elif (
        n_decisive >= 4
        and len(c1_wins) >= 2
        and len(families_c1) >= 2
    ):
        label = PILOT_ENTER_WORLD
        reasons.append(
            f"decisive={n_decisive}, c1_wins={len(c1_wins)}, families={list(families_c1)}"
        )
    # 3. WEAK_SELECTION_SPACE
    elif 1 <= len(c1_wins) <= 3 or (
        len(c1_wins) >= 1 and (len(families_c1) <= 1)
    ):
        label = PILOT_WEAK
        reasons.append(
            f"c1_wins={len(c1_wins)}, families={list(families_c1)}, decisive={n_decisive}"
        )
    # 4. NO_SELECTION_SPACE
    elif (
        rate >= 0.80
        and len(c1_wins) == 0
        and n_comp > 0
        and ((n_top1 + n_tie) / n_comp) >= 0.80
    ):
        label = PILOT_NONE
        reasons.append("no_candidate1_wins_and_top1_or_tie_dominant")
    else:
        label = PILOT_INCONCLUSIVE
        reasons.append("no_primary_rule_matched")

    return PilotSummary(
        label=label,
        n_pairs=n,
        n_comparable=n_comp,
        comparable_rate=rate,
        n_top1_best=n_top1,
        n_candidate1_best=n_c1,
        n_tie=n_tie,
        n_both_bad=n_both,
        n_incomparable=n_incomp,
        candidate1_win_families=families_c1,
        decisive_comparable=n_decisive,
        reasons=tuple(reasons),
        counts_by_family=by_family,
    )


def aggregate_oracle_table(results: Sequence[PairOracleResult]) -> dict[str, Any]:
    """Counts-first aggregation; no fake high-precision percentages on n=12."""
    pilot = assign_pilot_label(results)
    return {
        "pairs": [r.to_dict() for r in results],
        "pilot": pilot.to_dict(),
        "counts": {
            "comparable": pilot.n_comparable,
            "incomparable": pilot.n_incomparable,
            "top1_best": pilot.n_top1_best,
            "candidate1_best": pilot.n_candidate1_best,
            "tie": pilot.n_tie,
            "both_bad": pilot.n_both_bad,
        },
        "note": "R2 pilot counts only; not statistically significant safety claims.",
    }
