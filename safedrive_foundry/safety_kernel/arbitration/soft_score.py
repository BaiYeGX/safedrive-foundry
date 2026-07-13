"""Deterministic soft scoring of hard-prefiltered candidates (never overrides hard safety)."""

from __future__ import annotations

import math
from typing import Sequence

from safety_kernel.arbitration.types import SoftScore
from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    CandidateSource,
    ConstraintMargin,
    ObservableSnapshot,
    PolicyCandidate,
)
from safety_kernel.validator.checks import run_full_checks


def _path_progress(cand: PolicyCandidate) -> float:
    if len(cand.points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(cand.points)):
        p0, p1 = cand.points[i - 1], cand.points[i]
        total += math.hypot(p1.x - p0.x, p1.y - p0.y)
    return total


def _comfort_jerk_rms(cand: PolicyCandidate) -> float:
    if not cand.points:
        return 0.0
    return math.sqrt(sum(p.jerk * p.jerk for p in cand.points) / len(cand.points))


def _min_hard_margin(margins: Sequence[ConstraintMargin]) -> float:
    hard = [m.margin for m in margins if m.hard]
    if not hard:
        return 0.0
    return float(min(hard))


def score_candidate(
    cand: PolicyCandidate,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
    *,
    now_s: float | None = None,
    margins: Sequence[ConstraintMargin] | None = None,
) -> SoftScore:
    """Higher total is better. Pure function of cand/obs/cfg (deterministic)."""
    arb = cfg.arbitration
    now = obs.simulation_time_s if now_s is None else now_s
    ms = list(margins) if margins is not None else run_full_checks(cand, obs, cfg, now_s=now)

    progress_m = _path_progress(cand)
    # Normalize progress to ~[0,1] over a 30 m reference horizon.
    progress = max(0.0, min(1.5, progress_m / 30.0))
    jerk = _comfort_jerk_rms(cand)
    comfort = max(0.0, 1.0 - jerk / max(cfg.max_jerk_mps3, 1e-3))
    margin_raw = _min_hard_margin(ms)
    # Soft-map margin: saturates around ±2 m.
    margin = max(-1.0, min(1.0, margin_raw / 2.0))
    probability = float(max(0.0, min(1.0, cand.probability)))
    uncertainty_term = 1.0 - float(max(0.0, min(1.0, cand.uncertainty)))

    if cand.source is CandidateSource.CLASSIC:
        source_bonus = arb.classic_source_bonus
    elif cand.source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW}:
        source_bonus = arb.vla_source_bonus
    else:
        source_bonus = 0.0
    if str(cand.dynamics_meta.get("ranked_by", "")).startswith("world"):
        source_bonus += arb.world_ranked_bonus

    total = (
        arb.w_progress * progress
        + arb.w_comfort * comfort
        + arb.w_margin * margin
        + arb.w_probability * probability
        + arb.w_uncertainty * uncertainty_term
        + source_bonus
    )
    return SoftScore(
        candidate_id=cand.candidate_id,
        source=cand.source.value,
        total=float(total),
        progress=float(progress),
        comfort=float(comfort),
        margin=float(margin),
        probability=probability,
        uncertainty_term=float(uncertainty_term),
        source_bonus=float(source_bonus),
        extras={"progress_m": progress_m, "jerk_rms": jerk, "min_hard_margin": margin_raw},
    )


def rank_candidates(
    candidates: Sequence[PolicyCandidate],
    scores: Sequence[SoftScore],
) -> list[PolicyCandidate]:
    """Deterministic rank: soft total desc, classic preference, candidate_id asc."""
    by_id = {s.candidate_id: s for s in scores}
    def key(c: PolicyCandidate) -> tuple:
        s = by_id.get(c.candidate_id)
        total = s.total if s is not None else -1e9
        classic = 0 if c.source is CandidateSource.CLASSIC else 1
        return (-total, classic, c.candidate_id)

    return sorted(candidates, key=key)
