"""Frozen H1 routing over Guard-pass candidates only."""

from __future__ import annotations

import math

from driving_vla.hybrid.contracts import (
    CandidateDifference,
    HybridCandidateSet,
    RoutingResult,
    SelectionSpace,
    WorldDisposition,
)
from safety_kernel.arbitration.soft_score import rank_candidates, score_candidate
from safety_kernel.config import SafetyKernelConfig, load_safety_config
from safety_kernel.contracts.types import PolicyCandidateSet


MAX_DUPLICATE_POSITION_DELTA_M = 0.5
MAX_DUPLICATE_RMS_SPEED_DELTA_MPS = 0.5


class FrozenH1Router:
    """Implement 0/1/2 PASS behavior while World is absent in H1."""

    selector_name = "safety_soft_score_frozen"

    def __init__(self, config: SafetyKernelConfig | None = None) -> None:
        self.config = config or load_safety_config()

    def route(self, candidate_set: HybridCandidateSet) -> RoutingResult:
        passed = sorted(
            (
                item
                for item in candidate_set.candidates
                if item.guard is not None and item.guard.passed
            ),
            key=lambda item: item.candidate.candidate_id,
        )
        pass_ids = tuple(item.candidate.candidate_id for item in passed)
        rejected_ids = tuple(
            sorted(
                item.candidate.candidate_id
                for item in candidate_set.candidates
                if item.guard is None or not item.guard.passed
            )
        )
        if not passed:
            return RoutingResult(
                pass_candidate_ids=(),
                rejected_candidate_ids=rejected_ids,
                selected_candidate_id=None,
                selection_space=SelectionSpace.ZERO_PASS,
                world=WorldDisposition.DEFERRED_NOT_IMPLEMENTED,
                selector="none",
                reason="zero_guard_pass_safety_fallback",
                difference=None,
            )
        if len(passed) == 1:
            return RoutingResult(
                pass_candidate_ids=pass_ids,
                rejected_candidate_ids=rejected_ids,
                selected_candidate_id=pass_ids[0],
                selection_space=SelectionSpace.SINGLE_PASS,
                world=WorldDisposition.DEFERRED_NOT_IMPLEMENTED,
                selector="direct_single_pass",
                reason="single_guard_pass",
                difference=None,
            )

        left, right = passed[0].candidate, passed[1].candidate
        position = [
            math.hypot(lp.x - rp.x, lp.y - rp.y)
            for lp, rp in zip(left.points, right.points)
        ]
        speed_sq = [(lp.v - rp.v) ** 2 for lp, rp in zip(left.points, right.points)]
        difference = CandidateDifference(
            max_position_delta_m=max(position, default=0.0),
            rms_speed_delta_mps=math.sqrt(sum(speed_sq) / max(1, len(speed_sq))),
        )
        selection_space = (
            SelectionSpace.NO_SELECTION_SPACE
            if difference.max_position_delta_m <= MAX_DUPLICATE_POSITION_DELTA_M
            and difference.rms_speed_delta_mps <= MAX_DUPLICATE_RMS_SPEED_DELTA_MPS
            else SelectionSpace.DISTINCT
        )
        scores = [
            score_candidate(item.candidate, candidate_set.anchor.safety_snapshot, self.config)
            for item in passed
        ]
        ranked = rank_candidates([item.candidate for item in passed], scores)
        selected = ranked[0]
        return RoutingResult(
            pass_candidate_ids=pass_ids,
            rejected_candidate_ids=rejected_ids,
            selected_candidate_id=selected.candidate_id,
            selection_space=selection_space,
            world=WorldDisposition.DEFERRED_NOT_IMPLEMENTED,
            selector=self.selector_name,
            reason=(
                "near_duplicate_frozen_selector"
                if selection_space is SelectionSpace.NO_SELECTION_SPACE
                else "world_absent_frozen_selector"
            ),
            difference=difference,
            scores={score.candidate_id: score.total for score in scores},
        )

    @staticmethod
    def safety_input(
        candidate_set: HybridCandidateSet, routing: RoutingResult
    ) -> PolicyCandidateSet:
        if routing.selected_candidate_id is None:
            return candidate_set.to_policy_candidate_set(())
        selected = tuple(
            item.candidate
            for item in candidate_set.candidates
            if item.candidate.candidate_id == routing.selected_candidate_id
        )
        if len(selected) != 1:
            raise RuntimeError("selected_candidate_not_resolvable")
        return candidate_set.to_policy_candidate_set(selected)


__all__ = [
    "FrozenH1Router",
    "MAX_DUPLICATE_POSITION_DELTA_M",
    "MAX_DUPLICATE_RMS_SPEED_DELTA_MPS",
]
