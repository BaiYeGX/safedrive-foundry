"""G2-04 arbitration: prefilter → soft score → rank → final → repair → fallback → shadow."""

from __future__ import annotations

from safety_kernel.arbitration.degradation import degrade_candidate_set
from safety_kernel.arbitration.pipeline import ArbitrationPipeline, PipelineTickResult
from safety_kernel.arbitration.shadow import run_classic_shadow
from safety_kernel.arbitration.soft_score import rank_candidates, score_candidate
from safety_kernel.arbitration.types import (
    ArbitrationRecord,
    CandidateAudit,
    DegradationReason,
    PipelineStage,
    ShadowResult,
    SoftScore,
)

__all__ = [
    "ArbitrationPipeline",
    "ArbitrationRecord",
    "CandidateAudit",
    "DegradationReason",
    "PipelineStage",
    "PipelineTickResult",
    "ShadowResult",
    "SoftScore",
    "degrade_candidate_set",
    "rank_candidates",
    "run_classic_shadow",
    "score_candidate",
]
