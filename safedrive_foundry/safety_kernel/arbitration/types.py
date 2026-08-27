"""Arbitration / shadow / degradation contracts (G2-04)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from safety_kernel.contracts.types import (
    CandidateSource,
    DecisionKind,
    FallbackRequest,
    PolicyCandidate,
    SafetyMode,
)


class DegradationReason(str, Enum):
    NONE = "none"
    SOURCE_UNAVAILABLE = "source_unavailable"
    TIMEOUT = "timeout"
    OOD = "ood"
    OVERCONFIDENT = "overconfident"
    SOFT_STALE = "soft_stale"
    CANDIDATE_UNAVAILABLE = "candidate_unavailable"


class PipelineStage(str, Enum):
    STATE = "state"
    DEGRADE = "degrade"
    PREFILTER = "prefilter"
    SOFT_SCORE = "soft_score"
    ARBITRATE = "arbitrate"
    FINAL = "final"
    REPAIR = "repair"
    FALLBACK = "fallback"
    SHADOW = "shadow"
    DONE = "done"


@dataclass(frozen=True)
class SoftScore:
    candidate_id: str
    source: str
    total: float
    progress: float
    comfort: float
    margin: float
    probability: float
    uncertainty_term: float
    source_bonus: float
    extras: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "total": self.total,
            "progress": self.progress,
            "comfort": self.comfort,
            "margin": self.margin,
            "probability": self.probability,
            "uncertainty_term": self.uncertainty_term,
            "source_bonus": self.source_bonus,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    source: str
    version_hint: str
    prefilter_ok: bool
    final_ok: bool | None
    soft_score: SoftScore | None
    reject_reasons: tuple[str, ...]
    degradation: DegradationReason
    repaired: bool
    selected: bool
    repair_attempted: bool = False
    repair_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "version_hint": self.version_hint,
            "prefilter_ok": self.prefilter_ok,
            "final_ok": self.final_ok,
            "soft_score": None if self.soft_score is None else self.soft_score.to_dict(),
            "reject_reasons": list(self.reject_reasons),
            "degradation": self.degradation.value,
            "repaired": self.repaired,
            "selected": self.selected,
            "repair_attempted": self.repair_attempted,
            "repair_success": self.repair_success,
        }


@dataclass(frozen=True)
class ShadowResult:
    """Classic Shadow: compare-only; never claims control or tick ownership."""

    enabled: bool
    classic_candidate_id: str | None
    executed_candidate_id: str | None
    classic_soft_score: float | None
    executed_soft_score: float | None
    score_delta: float | None
    would_prefer_classic: bool
    claims_control: bool = False
    claims_tick_ownership: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "classic_candidate_id": self.classic_candidate_id,
            "executed_candidate_id": self.executed_candidate_id,
            "classic_soft_score": self.classic_soft_score,
            "executed_soft_score": self.executed_soft_score,
            "score_delta": self.score_delta,
            "would_prefer_classic": self.would_prefer_classic,
            "claims_control": self.claims_control,
            "claims_tick_ownership": self.claims_tick_ownership,
            "message": self.message,
        }


@dataclass(frozen=True)
class ArbitrationRecord:
    """Full audit of one arbitration pipeline tick."""

    run_id: str
    frame_id: str
    stages: tuple[str, ...]
    ranked_ids: tuple[str, ...]
    selected_id: str | None
    decision_kind: str
    mode_before: str
    mode_after: str
    latency_ms: float
    arbitration_latency_ms: float
    fallback: FallbackRequest | None
    shadow: ShadowResult | None
    audits: tuple[CandidateAudit, ...]
    learning_required: bool
    emergency_locked: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "frame_id": self.frame_id,
            "stages": list(self.stages),
            "ranked_ids": list(self.ranked_ids),
            "selected_id": self.selected_id,
            "decision_kind": self.decision_kind,
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "latency_ms": self.latency_ms,
            "arbitration_latency_ms": self.arbitration_latency_ms,
            "fallback": None if self.fallback is None else self.fallback.to_dict(),
            "shadow": None if self.shadow is None else self.shadow.to_dict(),
            "audits": [a.to_dict() for a in self.audits],
            "learning_required": self.learning_required,
            "emergency_locked": self.emergency_locked,
            "notes": list(self.notes),
        }


def source_version_hint(cand: PolicyCandidate) -> str:
    meta = dict(cand.dynamics_meta) if cand.dynamics_meta else {}
    if "model_id" in meta:
        return str(meta["model_id"])
    if "version" in meta:
        return str(meta["version"])
    return cand.source.value


def source_rank_key(source: CandidateSource) -> int:
    """Lower is preferred for deterministic tie-breaks after soft score."""
    order = {
        CandidateSource.CLASSIC: 0,
        CandidateSource.SHADOW: 1,
        CandidateSource.VLA_FAST: 2,
        CandidateSource.VLA_SLOW: 3,
        CandidateSource.SYNTHETIC: 4,
        CandidateSource.REPLAY: 5,
        CandidateSource.UNKNOWN: 6,
    }
    return order.get(source, 9)
