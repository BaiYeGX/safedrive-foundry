"""H5 World-on/off selector with risk gate and hysteresis.

H5 must not let the World ping-pong between Expert and VLA at 20Hz.  This router
wraps the normalized H4 scorer with:

- risk-gated defer (already inside NormalizedWorldScorer);
- minimum hold ticks;
- hysteresis margin for switching;
- fallback to FrozenH1Router on defer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from driving_vla.hybrid.contracts import (
    HybridCandidateSet,
    RoutingResult,
    WorldDisposition,
)
from driving_vla.hybrid.router import FrozenH1Router

from data_pipeline.h4.runtime import NormalizedWorldScorer


class H5WorldRouter:
    """Stateful World router with hysteresis for closed-loop use."""

    selector_name = "h5_world_v1"

    def __init__(
        self,
        scorer: NormalizedWorldScorer,
        fallback: FrozenH1Router | None = None,
        *,
        min_hold_ticks: int = 10,
        hysteresis_margin: float = 0.05,
    ) -> None:
        if min_hold_ticks < 1:
            raise ValueError("min_hold_ticks_must_be_positive")
        self.scorer = scorer
        self.fallback = fallback or FrozenH1Router()
        self.min_hold_ticks = int(min_hold_ticks)
        self.hysteresis_margin = float(hysteresis_margin)
        self._last_selected_id: str | None = None
        self._hold_count = 0

    def reset(self) -> None:
        self._last_selected_id = None
        self._hold_count = 0

    def _defer(self, candidate_set: HybridCandidateSet, reason: str) -> RoutingResult:
        baseline = self.fallback.route(candidate_set)
        # A defer means the non-learning selector is authoritative.  Reset the
        # World hold state so a later World decision starts clean.
        self.reset()
        return replace(
            baseline,
            world=WorldDisposition.DEFERRED_LOW_CONFIDENCE,
            selector=baseline.selector,
            reason=f"h5_defer:{reason}",
        )

    def route(
        self,
        candidate_set: HybridCandidateSet,
        features: Mapping[str, tuple[Sequence[float], Sequence[Sequence[float]]]] | None = None,
    ) -> RoutingResult:
        passed = [
            item for item in candidate_set.candidates
            if item.guard is not None and item.guard.passed
        ]
        if len(passed) < 2:
            result = self.fallback.route(candidate_set)
            self.reset()
            return replace(
                result,
                world=WorldDisposition.DEFERRED_NOT_APPLICABLE,
                reason="h5_not_applicable_single_or_zero_pass",
            )

        baseline = self.fallback.route(candidate_set)
        if baseline.selection_space.value == "NO_SELECTION_SPACE":
            return self._defer(candidate_set, "no_selection_space")
        if features is None:
            return self._defer(candidate_set, "feature_missing")

        ordered = sorted(passed, key=lambda item: item.candidate.candidate_id)
        payloads = []
        for item in ordered:
            key = item.candidate.candidate_id
            if key not in features:
                return self._defer(candidate_set, "feature_missing")
            context, candidate = features[key]
            payloads.append((key, context, candidate))

        try:
            score = self.scorer.score_pair(payloads[0], payloads[1])
        except (TypeError, ValueError, RuntimeError):
            return self._defer(candidate_set, "invalid_input")

        if score.disposition != "ranked" or score.selected_candidate_key is None:
            return self._defer(candidate_set, score.defer_reason or "low_confidence")

        proposed = score.selected_candidate_key
        passed_ids = {item.candidate.candidate_id for item in passed}
        if proposed not in passed_ids:
            return self._defer(candidate_set, "selected_candidate_not_in_passed")

        margin = abs(score.predictions[0].utility - score.predictions[1].utility)
        # Hysteresis: keep the current World selection unless the candidate is
        # no longer passed, the hold period has elapsed, and the margin is large
        # enough to justify switching.
        if (
            self._last_selected_id is not None
            and self._last_selected_id in passed_ids
            and self._last_selected_id != proposed
            and (self._hold_count < self.min_hold_ticks or margin < self.hysteresis_margin)
        ):
            selected = self._last_selected_id
            self._hold_count += 1
            reason = "h5_world_hold_hysteresis"
        else:
            selected = proposed
            self._last_selected_id = selected
            self._hold_count = 1
            reason = "h5_world_ranked"

        return RoutingResult(
            pass_candidate_ids=baseline.pass_candidate_ids,
            rejected_candidate_ids=baseline.rejected_candidate_ids,
            selected_candidate_id=selected,
            selection_space=baseline.selection_space,
            world=WorldDisposition.RANKED,
            selector=self.selector_name,
            reason=reason,
            difference=baseline.difference,
            scores={item.candidate_key: item.utility for item in score.predictions},
        )


__all__ = ["H5WorldRouter"]
