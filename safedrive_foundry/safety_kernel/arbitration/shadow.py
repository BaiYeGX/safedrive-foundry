"""Classic Shadow: advisory compare-only path (no control / no tick ownership)."""

from __future__ import annotations

from safety_kernel.arbitration.soft_score import score_candidate
from safety_kernel.arbitration.types import ShadowResult
from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    CandidateSource,
    ObservableSnapshot,
    PolicyCandidate,
)


def run_classic_shadow(
    *,
    classic: PolicyCandidate | None,
    executed: PolicyCandidate | None,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
    now_s: float | None = None,
) -> ShadowResult:
    if not cfg.arbitration.shadow_enabled:
        return ShadowResult(
            enabled=False,
            classic_candidate_id=None,
            executed_candidate_id=None,
            classic_soft_score=None,
            executed_soft_score=None,
            score_delta=None,
            would_prefer_classic=False,
            message="shadow_disabled",
        )
    if classic is None:
        return ShadowResult(
            enabled=True,
            classic_candidate_id=None,
            executed_candidate_id=None if executed is None else executed.candidate_id,
            classic_soft_score=None,
            executed_soft_score=None,
            score_delta=None,
            would_prefer_classic=False,
            message="no_classic_shadow_seed",
        )
    # Shadow seed must be classic-sourced (or explicitly tagged).
    if classic.source not in {CandidateSource.CLASSIC, CandidateSource.SHADOW}:
        return ShadowResult(
            enabled=True,
            classic_candidate_id=classic.candidate_id,
            executed_candidate_id=None if executed is None else executed.candidate_id,
            classic_soft_score=None,
            executed_soft_score=None,
            score_delta=None,
            would_prefer_classic=False,
            message="shadow_seed_not_classic",
        )

    c_score = score_candidate(classic, obs, cfg, now_s=now_s)
    e_score = None if executed is None else score_candidate(executed, obs, cfg, now_s=now_s)
    e_total = None if e_score is None else e_score.total
    delta = None if e_total is None else float(c_score.total - e_total)
    prefer = bool(delta is not None and delta > 1e-9)
    return ShadowResult(
        enabled=True,
        classic_candidate_id=classic.candidate_id,
        executed_candidate_id=None if executed is None else executed.candidate_id,
        classic_soft_score=c_score.total,
        executed_soft_score=e_total,
        score_delta=delta,
        would_prefer_classic=prefer,
        claims_control=False,
        claims_tick_ownership=False,
        message="shadow_compare_only",
    )
