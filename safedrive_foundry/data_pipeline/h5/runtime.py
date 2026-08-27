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
    supports_event_break = True

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
        ema_alpha: float | None = None,
        vla75_mode: bool | None = None,
    ) -> None:
        if min_hold_ticks < 1:
            raise ValueError("min_hold_ticks_must_be_positive")
        self.scorer = scorer
        if hasattr(scorer, "vla_trust_threshold"):
            self.selector_name = "world_v3_vla_primary"
        self.fallback = fallback or FrozenH1Router()
        self.min_hold_ticks = int(min_hold_ticks)
        self.hysteresis_margin = float(hysteresis_margin)
        self.emergency_switch_margin = float(emergency_switch_margin)
        self.single_pass_grace_ticks = int(single_pass_grace_ticks)
        self.force_defer = bool(force_defer)
        self.scorer_deadline_ms = float(scorer_deadline_ms)
        self.ema_alpha = None if ema_alpha is None else float(ema_alpha)
        if self.ema_alpha is not None and not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha_must_be_in_(0,1]")
        self.vla75_mode = bool(
            hasattr(scorer, "schema_version")
            and str(getattr(scorer, "schema_version", "")).endswith("vla75.pair_exec.v1")
            if vla75_mode is None
            else vla75_mode
        )
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
        self._ema_scores: dict[str, float] = {}
        self._last_eligible_ids: frozenset[str] | None = None

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
        self._ema_scores.clear()
        self._last_eligible_ids = None

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
            "ping_pong": self._ping_pong(),
        }

    def _ping_pong(self, window_ticks: int = 10) -> bool:
        sources = [
            str(item.get("selected_source", ""))
            for item in self._history
            if item.get("type") == "ranked"
        ]
        for start, source in enumerate(sources):
            if source not in {"vla", "vla_fast", "vla_slow", "expert", "classic"}:
                continue
            normalized = "vla" if source.startswith("vla") else "expert"
            opposite = False
            for value in sources[start + 1 : start + int(window_ticks) + 1]:
                current = "vla" if value.startswith("vla") else "expert" if value in {"expert", "classic"} else value
                if current in {"vla", "expert"} and current != normalized:
                    opposite = True
                if opposite and current == normalized:
                    return True
        return False

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

    @staticmethod
    def _source_by_id(candidate_set: HybridCandidateSet) -> dict[str, str]:
        return {
            item.candidate.candidate_id: str(item.provenance.source.value)
            for item in candidate_set.candidates
        }

    def route(
        self,
        candidate_set: HybridCandidateSet,
        features: Mapping[str, tuple[Sequence[float], Sequence[Sequence[float]]]] | None = None,
        *,
        event_break: bool = False,
    ) -> RoutingResult:
        self._last_score = None
        passed = [
            item for item in candidate_set.candidates
            if item.guard is not None and item.guard.passed
        ]
        eligible_ids = frozenset(str(item.candidate.candidate_id) for item in passed)
        eligible_changed = (
            self.vla75_mode
            and self._last_eligible_ids is not None
            and eligible_ids != self._last_eligible_ids
        )
        self._last_eligible_ids = eligible_ids
        if len(passed) < 2:
            result = self.fallback.route(candidate_set)
            if eligible_changed:
                # A Guard eligibility change is an explicit temporal event;
                # do not carry a prior source hold through a newly missing or
                # newly admitted candidate.
                self._clear_hold()
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
        # H1's duplicate threshold was designed for a source-biased frozen
        # soft-score fallback.  World v3 predicts outcome, trust and risk even
        # when the two short-horizon trajectories are geometrically close, so
        # suppressing the model here silently prevents it from ever building a
        # VLA-primary policy at low speed.  Keep the selection-space label for
        # audit, but only preserve the historical defer for pre-v3 scorers.
        is_world_v3 = hasattr(self.scorer, "vla_trust_threshold")
        if baseline.selection_space.value == "NO_SELECTION_SPACE" and not is_world_v3:
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

        source_by_id = self._source_by_id(candidate_set)
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

        predictions = tuple(getattr(score, "predictions", ()))
        prediction_by_id = {
            str(getattr(item, "candidate_key", "")): item for item in predictions
        }
        raw_order = tuple(
            str(item)
            for item in (
                getattr(score, "raw_preference_order", ())
                or getattr(score, "preference_order", ())
                or ()
            )
        )
        if not raw_order and predictions:
            raw_order = tuple(
                item.candidate_key
                for item in sorted(
                    predictions,
                    key=lambda item: (
                        float(getattr(item, "deployment_score", getattr(item, "utility", 0.0))),
                        str(getattr(item, "candidate_key", "")),
                    ),
                    reverse=True,
                )
            )
        raw_preferred = raw_order[0] if raw_order else score.selected_candidate_key
        raw_source = source_by_id.get(raw_preferred, "unknown")
        raw_gate_reasons: dict[str, bool] = {}
        risk_breach = False
        if self.vla75_mode:
            vla_id = next((key for key, source in source_by_id.items() if source == "vla"), None)
            expert_id = next((key for key, source in source_by_id.items() if source == "expert"), None)
            vla_prediction = prediction_by_id.get(vla_id)
            expert_prediction = prediction_by_id.get(expert_id)
            if vla_prediction is not None and expert_prediction is not None:
                vla_risk = float(getattr(vla_prediction, "unsafe_probability", math.inf))
                risk_ceiling = float(getattr(score, "risk_ceiling", math.inf))
                risk_breach = not math.isfinite(vla_risk) or vla_risk > risk_ceiling + 1e-12
                raw_gate_reasons = {
                    "score": float(getattr(vla_prediction, "deployment_score", -math.inf))
                    >= float(getattr(expert_prediction, "deployment_score", math.inf)),
                    "pair_preference": float(getattr(vla_prediction, "preference_utility", -math.inf))
                    >= float(getattr(expert_prediction, "preference_utility", math.inf)),
                    "trust": float(getattr(vla_prediction, "trust_probability", 0.0))
                    >= float(getattr(score, "trust_threshold", math.inf)),
                    "risk": float(getattr(vla_prediction, "unsafe_probability", math.inf))
                    <= float(getattr(score, "risk_ceiling", -math.inf)),
                    "pair_completeness": True,
                }
            else:
                raw_gate_reasons = {
                    "score": False,
                    "pair_preference": False,
                    "trust": False,
                    "risk": False,
                    "pair_completeness": False,
                }
        temporal_event_break = bool(
            event_break or (eligible_changed if self.vla75_mode else False) or risk_breach
        )

        # World v3 separates an objective score from a calibrated trust score.
        # If the VLA candidate meets both dev-locked gates, it is the primary
        # proposal; otherwise the objective ranking above remains authoritative.
        vla_primary = False
        trust_threshold = getattr(self.scorer, "vla_trust_threshold", None)
        risk_ceiling = getattr(self.scorer, "vla_risk_ceiling", None)
        if trust_threshold is not None and risk_ceiling is not None:
            vla_id = next(
                (candidate_id for candidate_id, source in source_by_id.items() if source == "vla"),
                None,
            )
            prediction_by_id = {
                item.candidate_key: item for item in getattr(score, "predictions", ())
            }
            vla_prediction = prediction_by_id.get(vla_id)
            expert_predictions = [
                prediction_by_id.get(candidate_id)
                for candidate_id, source in source_by_id.items()
                if source == "expert"
            ]
            expert_prediction = next(
                (item for item in expert_predictions if item is not None), None
            )
            development_force_vla = bool(
                getattr(self.scorer, "development_force_vla", False)
            )
            if (
                vla_id is not None
                and vla_prediction is not None
                and float(getattr(vla_prediction, "trust_probability")) >= float(trust_threshold)
                and float(getattr(vla_prediction, "unsafe_probability")) <= float(risk_ceiling)
                and (
                    development_force_vla
                    or expert_prediction is None
                    or float(getattr(vla_prediction, "utility"))
                    >= float(getattr(expert_prediction, "utility"))
                )
                and (
                    not self.vla75_mode
                    or expert_prediction is None
                    or float(getattr(vla_prediction, "preference_utility", -math.inf))
                    >= float(getattr(expert_prediction, "preference_utility", math.inf))
                )
            ):
                proposed = vla_id
                proposed_source = "vla"
                vla_primary = True

        margin = abs(score.predictions[0].utility - score.predictions[1].utility)
        if self.ema_alpha is not None:
            smoothed: dict[str, float] = {}
            for prediction in score.predictions:
                key = str(prediction.candidate_key)
                value = float(
                    getattr(
                        prediction,
                        "preference_utility",
                        prediction.utility,
                    )
                    if self.vla75_mode
                    else prediction.utility
                )
                previous = self._ema_scores.get(key, value)
                smoothed[key] = self.ema_alpha * value + (1.0 - self.ema_alpha) * previous
            self._ema_scores = smoothed
            if not temporal_event_break:
                margin = abs(smoothed.get(score.predictions[0].candidate_key, 0.0) - smoothed.get(score.predictions[1].candidate_key, 0.0))
            else:
                self._clear_hold()
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
            and not temporal_event_break
            and (not vla_primary or self.vla75_mode)
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
            reason = (
                "h5_world_event_break"
                if temporal_event_break
                else
                "development_forced_vla"
                if vla_primary and bool(getattr(self.scorer, "development_force_vla", False))
                else "world_v3_vla_trust"
                if vla_primary
                else "h5_world_ranked"
            )

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
            "vla_primary": vla_primary,
            "event_break": temporal_event_break,
            "risk_breach": risk_breach,
            "eligible_changed": eligible_changed,
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
            review_candidate_ids=tuple(
                item.candidate.candidate_id
                for item in passed
                if bool(getattr(item.guard, "needs_review", False))
            ),
            preference_order=(selected,) + tuple(
                item.candidate.candidate_id
                for item in passed
                if item.candidate.candidate_id != selected
            ),
            raw_preferred_candidate_id=raw_preferred,
            raw_preferred_source=raw_source,
            raw_gate_reasons=raw_gate_reasons,
            stabilized_preferred_candidate_id=selected,
            stabilized_preferred_source=source_by_id.get(selected, proposed_source),
        )


    @staticmethod
    def safety_input(
        candidate_set: HybridCandidateSet, routing: RoutingResult
    ):
        if routing.selected_candidate_id is None:
            return candidate_set.to_policy_candidate_set(())
        eligible = {
            item.candidate.candidate_id: item.candidate
            for item in candidate_set.candidates
            if item.guard is not None and item.guard.passed
        }
        order = routing.preference_order or (routing.selected_candidate_id,)
        ordered_ids = tuple(candidate_id for candidate_id in order if candidate_id in eligible)
        ordered_ids += tuple(
            candidate_id for candidate_id in sorted(eligible) if candidate_id not in ordered_ids
        )
        if not ordered_ids or ordered_ids[0] != routing.selected_candidate_id:
            raise RuntimeError("selected_candidate_not_resolvable")
        return candidate_set.to_policy_candidate_set(
            tuple(eligible[candidate_id] for candidate_id in ordered_ids),
            preference_order=ordered_ids,
        )


__all__ = ["H5WorldRouter"]
