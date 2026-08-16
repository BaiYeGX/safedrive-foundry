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
    requires_features = True

    def __init__(
        self,
        scorer: NormalizedWorldScorer,
        fallback: FrozenH1Router | None = None,
        *,
        min_hold_ticks: int = 10,
        hysteresis_margin: float = 0.05,
        emergency_switch_margin: float = 1.5,
        single_pass_grace_ticks: int = 3,
        force_defer: bool = False,
        scorer_deadline_ms: float = 50.0,
    ) -> None:
        if min_hold_ticks < 1:
            raise ValueError("min_hold_ticks_must_be_positive")
        self.scorer = scorer
        self.fallback = fallback or FrozenH1Router()
        self.min_hold_ticks = int(min_hold_ticks)
        self.hysteresis_margin = float(hysteresis_margin)
        self.emergency_switch_margin = float(emergency_switch_margin)
        self.single_pass_grace_ticks = int(single_pass_grace_ticks)
        self.force_defer = bool(force_defer)
        self.scorer_deadline_ms = float(scorer_deadline_ms)
        if self.scorer_deadline_ms <= 0.0:
            raise ValueError("scorer_deadline_ms_must_be_positive")
        if self.emergency_switch_margin < self.hysteresis_margin:
            raise ValueError("emergency_switch_margin_must_be_ge_hysteresis_margin")
        if self.single_pass_grace_ticks < 0:
            raise ValueError("single_pass_grace_ticks_must_be_nonnegative")
        self._last_selected_id: str | None = None
        self._last_selected_source: str | None = None
        self._hold_count = 0
        self._history: list[dict] = []
        self._switch_count = 0
        self._defer_count = 0
        self._single_pass_count = 0
        self._last_score = None

    @property
    def last_score(self):
        return self._last_score

    def reset(self) -> None:
        self._last_selected_id = None
        self._last_selected_source = None
        self._hold_count = 0
        self._history.clear()
        self._switch_count = 0
        self._defer_count = 0
        self._single_pass_count = 0

    def _clear_hold(self) -> None:
        # Keep cumulative metrics/history; only forget the current hold/source.
        self._last_selected_id = None
        self._last_selected_source = None
        self._hold_count = 0
        self._single_pass_count = 0

    def metrics(self) -> dict:
        return {
            "decisions": len(self._history),
            "switch_count": self._switch_count,
            "defer_count": self._defer_count,
            "current_hold_ticks": self._hold_count,
            "history": list(self._history),
        }

    def _defer(self, candidate_set: HybridCandidateSet, reason: str) -> RoutingResult:
        baseline = self.fallback.route(candidate_set)
        # A defer means the non-learning selector is authoritative.  Reset the
        # World hold state so a later World decision starts clean.
        self._defer_count += 1
        self._history.append({"type": "defer", "reason": reason})
        self._last_selected_id = None
        self._last_selected_source = None
        self._hold_count = 0
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
        self._last_score = None
        passed = [
            item for item in candidate_set.candidates
            if item.guard is not None and item.guard.passed
        ]
        if len(passed) < 2:
            result = self.fallback.route(candidate_set)
            # A single-frame Guard glitch must not erase the hysteresis state.
            # Only after the configured grace period without a two-candidate
            # selection do we reset to a fresh World session.
            self._single_pass_count += 1
            if self._single_pass_count >= self.single_pass_grace_ticks:
                self._clear_hold()
            return replace(
                result,
                world=WorldDisposition.DEFERRED_NOT_APPLICABLE,
                reason="h5_not_applicable_single_or_zero_pass",
            )

        self._single_pass_count = 0
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

        self._last_score = score

        if float(getattr(score, "latency_ms", 0.0)) > self.scorer_deadline_ms:
            return self._defer(candidate_set, "deadline")

        if self.force_defer:
            return self._defer(candidate_set, "forced_defer")

        if score.disposition != "ranked" or score.selected_candidate_key is None:
            return self._defer(candidate_set, score.defer_reason or "low_confidence")

        proposed = score.selected_candidate_key
        passed_ids = {item.candidate.candidate_id for item in passed}
        if proposed not in passed_ids:
            return self._defer(candidate_set, "selected_candidate_not_in_passed")
        source_by_id = {
            item.candidate.candidate_id: str(item.provenance.source.value)
            for item in passed
        }
        proposed_source = source_by_id.get(proposed, "unknown")
        passed_sources = set(source_by_id.values())

        margin = abs(score.predictions[0].utility - score.predictions[1].utility)
        previous_selected = self._last_selected_id
        previous_source = self._last_selected_source
        # Hysteresis: keep the current World selection unless the candidate is
        # no longer passed, the margin is below the switch floor, or the hold
        # period has not elapsed.  A very large margin is treated as an
        # emergency switch and may break the hold early.
        keep_current = (
            previous_source is not None
            and previous_source in passed_sources
            and previous_source != proposed_source
            and (
                margin < self.hysteresis_margin
                or (
                    self._hold_count < self.min_hold_ticks
                    and margin < self.emergency_switch_margin
                )
            )
        )
        if keep_current:
            # Select this tick's candidate for the same source, never a stale id.
            selected = next(
                (item.candidate.candidate_id for item in passed if str(item.provenance.source.value) == previous_source),
                proposed,
            )
            self._last_selected_id = selected
            self._hold_count += 1
            reason = "h5_world_hold_hysteresis"
        else:
            selected = proposed
            self._last_selected_id = selected
            self._last_selected_source = proposed_source
            self._hold_count = 1
            reason = "h5_world_ranked"

        if self._last_selected_source != previous_source and previous_source is not None and self._history and self._history[-1].get("type") == "ranked":
            self._switch_count += 1
        self._history.append({
            "type": "ranked",
            "reason": reason,
            "selected": selected,
            "selected_source": self._last_selected_source,
            "proposed": proposed,
            "proposed_source": proposed_source,
            "margin": float(margin),
            "hold_count": self._hold_count,
        })

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


    @staticmethod
    def safety_input(
        candidate_set: HybridCandidateSet, routing: RoutingResult
    ):
        from driving_vla.hybrid.contracts import PolicyCandidateSet  # type: ignore
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


__all__ = ["H5WorldRouter"]
