"""Single source-scoped temporal selector shared by offline and live H6."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


TRACE_SCHEMA = "safedrive.h6.temporal_selector.trace.v1"
SOURCES = ("expert", "vla")


def normalize_source(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"classic", "classic_expert", "expert"}:
        return "expert"
    if text in {"vla", "vla_fast", "vla_slow"}:
        return "vla"
    if text in {"mrm", "minimal_risk", "minimal_risk_brake"}:
        return "mrm"
    return text


@dataclass(frozen=True)
class TemporalSelectorConfig:
    ema_alpha: float = 0.50
    hold_ticks: int = 10
    hysteresis: float = 0.10
    emergency_switch_margin: float = 1.50
    ping_pong_window_ticks: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha_must_be_in_(0,1]")
        if int(self.hold_ticks) < 0:
            raise ValueError("hold_ticks_must_be_nonnegative")
        if float(self.hysteresis) < 0.0:
            raise ValueError("hysteresis_must_be_nonnegative")
        if float(self.emergency_switch_margin) < float(self.hysteresis):
            raise ValueError("emergency_margin_must_be_ge_hysteresis")
        if int(self.ping_pong_window_ticks) < 1:
            raise ValueError("ping_pong_window_must_be_positive")


@dataclass(frozen=True)
class TemporalSelectorDecision:
    scope_key: str
    disposition: str
    selected_source: str | None
    selected_candidate_id: str | None
    raw_preferred_source: str | None
    raw_scores: Mapping[str, float]
    ema_scores: Mapping[str, float]
    margin: float | None
    hold_age: int
    switch_count: int
    ping_pong: bool
    reason: str
    learned_defer_reason: str | None
    trace_schema: str = TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalSelectorCore:
    """Pure deterministic selector whose state contains sources, never IDs."""

    def __init__(self, config: TemporalSelectorConfig | None = None) -> None:
        self.config = config or TemporalSelectorConfig()
        self.reset()

    def reset(self) -> None:
        self._scope_key: str | None = None
        self._ema: dict[str, float] = {}
        self._held_source: str | None = None
        self._hold_age = 0
        self._switch_count = 0
        self._history: list[str] = []
        self._trace: list[dict[str, Any]] = []

    def _reset_scope(self, scope_key: str) -> None:
        self._scope_key = scope_key
        self._ema = {}
        self._held_source = None
        self._hold_age = 0
        self._switch_count = 0
        self._history = []
        self._trace = []

    def _ping_pong(self) -> bool:
        window = int(self.config.ping_pong_window_ticks)
        for start, source in enumerate(self._history):
            opposite = False
            for current in self._history[start + 1 : start + window + 1]:
                if current != source:
                    opposite = True
                if opposite and current == source:
                    return True
        return False

    def metrics(self) -> dict[str, Any]:
        return {
            "scope_key": self._scope_key,
            "held_source": self._held_source,
            "hold_age": self._hold_age,
            "switches": self._switch_count,
            "ping_pong": self._ping_pong(),
            "history": list(self._history),
            "trace": list(self._trace),
        }

    def _record(
        self,
        *,
        scope_key: str,
        disposition: str,
        selected_source: str | None,
        selected_candidate_id: str | None,
        raw_preferred_source: str | None,
        raw_scores: Mapping[str, float],
        margin: float | None,
        reason: str,
        learned_defer_reason: str | None,
    ) -> TemporalSelectorDecision:
        decision = TemporalSelectorDecision(
            scope_key=scope_key,
            disposition=disposition,
            selected_source=selected_source,
            selected_candidate_id=selected_candidate_id,
            raw_preferred_source=raw_preferred_source,
            raw_scores=dict(raw_scores),
            ema_scores=dict(self._ema),
            margin=margin,
            hold_age=self._hold_age,
            switch_count=self._switch_count,
            ping_pong=self._ping_pong(),
            reason=reason,
            learned_defer_reason=learned_defer_reason,
        )
        self._trace.append(decision.to_dict())
        return decision

    def _choose_fallback(
        self,
        eligible: set[str],
        unsafe: set[str],
    ) -> tuple[str | None, str]:
        safe = eligible - unsafe
        if self._held_source in safe:
            return self._held_source, "defer_held_source"
        if "expert" in safe:
            return "expert", "defer_frozen_expert"
        if "vla" in safe:
            return "vla", "defer_frozen_vla"
        return None, "defer_no_eligible_mrm"

    def step(
        self,
        *,
        scope_key: str,
        source_scores: Mapping[str, float],
        fresh_candidate_ids: Mapping[str, str],
        eligible_sources: Sequence[str] | set[str] | frozenset[str],
        raw_preferred_source: str | None,
        learned_defer_reason: str | None = None,
        event_break: bool = False,
        emergency_sources: Sequence[str] | set[str] | frozenset[str] = (),
        unsafe_sources: Sequence[str] | set[str] | frozenset[str] = (),
    ) -> TemporalSelectorDecision:
        scope = str(scope_key)
        if not scope:
            raise ValueError("temporal_scope_key_required")
        if self._scope_key != scope:
            self._reset_scope(scope)

        ids = {
            normalize_source(source): str(candidate_id)
            for source, candidate_id in fresh_candidate_ids.items()
            if normalize_source(source) in SOURCES and str(candidate_id)
        }
        eligible = {
            normalize_source(source)
            for source in eligible_sources
            if normalize_source(source) in SOURCES
        }
        if any(source not in ids for source in eligible):
            raise ValueError("temporal_eligible_source_missing_fresh_candidate")
        raw_source = normalize_source(raw_preferred_source)
        if raw_source not in SOURCES:
            raw_source = None
        unsafe = {
            normalize_source(source)
            for source in tuple(emergency_sources) + tuple(unsafe_sources)
            if normalize_source(source) in SOURCES
        }
        scores: dict[str, float] = {}
        for source, value in source_scores.items():
            normalized = normalize_source(source)
            if normalized not in SOURCES:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("temporal_scores_must_be_finite")
            scores[normalized] = number
            previous = self._ema.get(normalized, number)
            self._ema[normalized] = (
                float(self.config.ema_alpha) * number
                + (1.0 - float(self.config.ema_alpha)) * previous
            )

        defer_requested = learned_defer_reason is not None or len(eligible) < 2
        if defer_requested:
            selected, reason = self._choose_fallback(eligible, unsafe)
            previous = self._held_source
            if selected is None:
                self._held_source = None
                self._hold_age = 0
            else:
                if previous is not None and selected != previous:
                    self._switch_count += 1
                self._held_source = selected
                self._hold_age = self._hold_age + 1 if selected == previous else 1
                self._history.append(selected)
            return self._record(
                scope_key=scope,
                disposition=(
                    "DEFER_SINGLE_CANDIDATE" if len(eligible) < 2 else "DEFER_AMBIGUOUS"
                ),
                selected_source=selected,
                selected_candidate_id=None if selected is None else ids[selected],
                raw_preferred_source=raw_source,
                raw_scores=scores,
                margin=None,
                reason=reason,
                learned_defer_reason=learned_defer_reason,
            )

        safe = eligible - unsafe
        if not safe:
            selected, reason = self._choose_fallback(eligible, unsafe)
            self._held_source = None
            self._hold_age = 0
            return self._record(
                scope_key=scope,
                disposition="DEFER_AMBIGUOUS",
                selected_source=selected,
                selected_candidate_id=None,
                raw_preferred_source=raw_source,
                raw_scores=scores,
                margin=None,
                reason=reason,
                learned_defer_reason="all_eligible_sources_unsafe",
            )
        if raw_source not in safe:
            raw_source = max(safe, key=lambda source: (self._ema.get(source, -math.inf), source))
        if raw_source not in scores or any(source not in scores for source in safe):
            raise ValueError("temporal_ranked_scores_incomplete")
        proposed = max(safe, key=lambda source: (self._ema[source], source == raw_source, source))
        previous = self._held_source
        margin = None if previous is None else float(self._ema[proposed] - self._ema.get(previous, -math.inf))

        disposition = "CHOOSE"
        reason = "choose_initial"
        selected = proposed
        if previous is None:
            pass
        elif previous not in eligible:
            disposition = "SWITCH"
            reason = "switch_held_ineligible"
        elif previous in unsafe:
            disposition = "SWITCH"
            reason = "switch_emergency_risk"
        elif previous == proposed:
            disposition = "CHOOSE"
            reason = "choose_continue"
        elif bool(event_break):
            disposition = "SWITCH"
            reason = "switch_event_break"
        elif margin is not None and margin >= float(self.config.emergency_switch_margin):
            disposition = "SWITCH"
            reason = "switch_emergency_margin"
        elif self._hold_age < int(self.config.hold_ticks):
            disposition = "HOLD"
            selected = previous
            reason = "hold_minimum"
        elif margin is not None and margin <= float(self.config.hysteresis):
            disposition = "HOLD"
            selected = previous
            reason = "hold_hysteresis"
        else:
            disposition = "SWITCH"
            reason = "switch_margin"

        if selected is None or selected not in eligible or selected in unsafe:
            raise RuntimeError("temporal_selected_source_not_safe_eligible")
        if previous is not None and selected != previous:
            self._switch_count += 1
        self._held_source = selected
        self._hold_age = self._hold_age + 1 if selected == previous else 1
        self._history.append(selected)
        return self._record(
            scope_key=scope,
            disposition=disposition,
            selected_source=selected,
            selected_candidate_id=ids[selected],
            raw_preferred_source=raw_source,
            raw_scores=scores,
            margin=margin,
            reason=reason,
            learned_defer_reason=learned_defer_reason,
        )


__all__ = [
    "TRACE_SCHEMA",
    "TemporalSelectorConfig",
    "TemporalSelectorCore",
    "TemporalSelectorDecision",
    "normalize_source",
]
