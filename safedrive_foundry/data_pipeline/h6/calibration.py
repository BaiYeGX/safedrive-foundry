"""Dev-only calibration for the VLA-primary deployment threshold."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from data_pipeline.h6.contracts import WorldV3Prediction


@dataclass(frozen=True)
class CalibrationRow:
    pair_id: str
    vla_prediction: WorldV3Prediction
    expert_prediction: WorldV3Prediction
    vla_unsafe: bool
    expert_unsafe: bool
    vla_progress_m: float
    expert_progress_m: float
    raw_preference: bool | None = None
    executable: bool | None = None
    applied_source: str | None = None
    phase: str = "unknown"
    group_key: str = "unknown"
    # Sequence/tick identity is required for temporal calibration.  The
    # fields are optional so v1/offline callers can continue constructing the
    # original row shape; v75 training populates them from the real closed
    # loop arm and tick rather than from an inferred policy summary.
    sequence_id: str | None = None
    tick: int | None = None
    event_break: bool = False
    risk_breach: bool = False
    eligible_changed: bool = False


@dataclass(frozen=True)
class VLADeploymentCalibration:
    passed: bool
    trust_threshold: float
    risk_ceiling: float
    vla_coverage: float
    unsafe_rate: float
    expert_unsafe_rate: float
    unsafe_delta: float
    mean_progress_delta_m: float
    rows: int
    policy_rows: int
    target_vla_coverage: float
    max_unsafe_delta: float
    min_trust_threshold: float
    min_mean_progress_delta_m: float
    reason: str

    def to_dict(self):
        return asdict(self)


def calibrate_vla_deployment(
    rows: Sequence[CalibrationRow],
    *,
    policy_rows: Sequence[CalibrationRow] | None = None,
    target_vla_coverage: float = 0.90,
    max_unsafe_delta: float = 0.01,
    min_trust_threshold: float = 0.50,
    min_mean_progress_delta_m: float = 0.0,
    max_risk_ceiling: float = 1.0,
) -> VLADeploymentCalibration:
    if not rows:
        raise ValueError("world_v3_calibration_rows_required")
    policy = tuple(rows if policy_rows is None else policy_rows)
    if not policy:
        raise ValueError("world_v3_policy_calibration_rows_required")
    if not 0.0 <= float(max_risk_ceiling) <= 1.0:
        raise ValueError("max_risk_ceiling_out_of_range")
    candidates = []
    thresholds = sorted(
        {min_trust_threshold, 1.0, *(round(row.vla_prediction.trust_probability, 6) for row in rows)},
        reverse=True,
    )
    risk_ceilings = sorted(
        {
            float(max_risk_ceiling),
            *(
                value
                for value in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0)
                if value <= float(max_risk_ceiling) + 1e-12
            ),
        }
    )
    expert_unsafe_rate = sum(int(row.expert_unsafe) for row in policy) / len(policy)
    for threshold in thresholds:
        if threshold + 1e-12 < min_trust_threshold:
            continue
        for risk_ceiling in risk_ceilings:
            use_vla = [
                row.vla_prediction.trust_probability + 1e-12 >= threshold
                and row.vla_prediction.unsafe_probability <= risk_ceiling + 1e-12
                and row.vla_prediction.deployment_score + 1e-12
                >= row.expert_prediction.deployment_score
                and (
                    row.raw_preference is None
                    or bool(row.raw_preference)
                )
                for row in rows
            ]
            coverage = sum(use_vla) / len(rows)
            policy_use_vla = [
                row.vla_prediction.trust_probability + 1e-12 >= threshold
                and row.vla_prediction.unsafe_probability <= risk_ceiling + 1e-12
                and row.vla_prediction.deployment_score + 1e-12
                >= row.expert_prediction.deployment_score
                and (
                    row.raw_preference is None
                    or bool(row.raw_preference)
                )
                for row in policy
            ]
            unsafe = sum(
                int(row.vla_unsafe if selected else row.expert_unsafe)
                for row, selected in zip(policy, policy_use_vla)
            ) / len(policy)
            progress = sum(
                (row.vla_progress_m if selected else row.expert_progress_m)
                - row.expert_progress_m
                for row, selected in zip(policy, policy_use_vla)
            ) / len(policy)
            delta = unsafe - expert_unsafe_rate
            if (
                coverage + 1e-12 >= target_vla_coverage
                and delta <= max_unsafe_delta + 1e-12
                and progress + 1e-12 >= min_mean_progress_delta_m
            ):
                candidates.append((threshold, -risk_ceiling, coverage, progress, unsafe, delta))
    if not candidates:
        return VLADeploymentCalibration(
            passed=False,
            trust_threshold=1.0,
            risk_ceiling=0.0,
            vla_coverage=0.0,
            unsafe_rate=expert_unsafe_rate,
            expert_unsafe_rate=expert_unsafe_rate,
            unsafe_delta=0.0,
            mean_progress_delta_m=0.0,
            rows=len(rows),
            policy_rows=len(policy),
            target_vla_coverage=target_vla_coverage,
            max_unsafe_delta=max_unsafe_delta,
            min_trust_threshold=min_trust_threshold,
            min_mean_progress_delta_m=min_mean_progress_delta_m,
            reason="no_dev_threshold_meets_coverage_safety_progress_and_high_trust",
        )
    # Prefer the strictest trust threshold, then the lowest risk ceiling,
    # followed by coverage and progress.  This avoids choosing 0.0 merely to
    # inflate VLA use.
    threshold, negative_risk, coverage, progress, unsafe, delta = max(candidates)
    return VLADeploymentCalibration(
        passed=True,
        trust_threshold=float(threshold),
        risk_ceiling=float(-negative_risk),
        vla_coverage=float(coverage),
        unsafe_rate=float(unsafe),
        expert_unsafe_rate=float(expert_unsafe_rate),
        unsafe_delta=float(delta),
        mean_progress_delta_m=float(progress),
        rows=len(rows),
        policy_rows=len(policy),
        target_vla_coverage=target_vla_coverage,
        max_unsafe_delta=max_unsafe_delta,
        min_trust_threshold=min_trust_threshold,
        min_mean_progress_delta_m=min_mean_progress_delta_m,
        reason="dev_calibrated",
    )


@dataclass(frozen=True)
class TemperatureCalibration:
    """Per-head temperature scaling parameters for development calibration."""

    temperatures: Mapping[str, float]
    bounds: tuple[float, float] = (0.05, 10.0)
    rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperatures": {
                str(key): float(value) for key, value in self.temperatures.items()
            },
            "bounds": list(self.bounds),
            "rows": self.rows,
        }


def temperature_scale_logit(logit: float, temperature: float) -> float:
    """Apply bounded temperature scaling to one logit."""

    if not math.isfinite(float(logit)) or not math.isfinite(float(temperature)):
        raise ValueError("temperature_scaling_requires_finite_values")
    if temperature <= 0.0:
        raise ValueError("temperature_must_be_positive")
    return float(logit) / float(temperature)


def fit_temperature(
    logits: Sequence[float],
    labels: Sequence[bool | float],
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
    grid_size: int = 161,
) -> float:
    """Fit a scalar temperature by deterministic held-out NLL grid search."""

    if len(logits) != len(labels) or not logits:
        raise ValueError("temperature_fit_rows_required")
    low, high = (float(bounds[0]), float(bounds[1]))
    if not 0.0 < low <= high:
        raise ValueError("temperature_bounds_invalid")
    count = max(2, int(grid_size))
    log_low, log_high = math.log(low), math.log(high)
    candidates = [
        math.exp(log_low + (log_high - log_low) * index / (count - 1))
        for index in range(count)
    ]
    best = (float("inf"), high)
    for temperature in candidates:
        nll = 0.0
        for raw, label in zip(logits, labels):
            scaled = max(-40.0, min(40.0, temperature_scale_logit(float(raw), temperature)))
            target = 1.0 if bool(label) else 0.0
            nll += max(scaled, 0.0) - scaled * target + math.log1p(math.exp(-abs(scaled)))
        nll /= len(logits)
        if nll < best[0] - 1e-12 or (
            abs(nll - best[0]) <= 1e-12 and temperature < best[1]
        ):
            best = (nll, temperature)
    return float(best[1])


def calibrate_temperature_heads(
    rows: Sequence[Mapping[str, Any]],
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> TemperatureCalibration:
    """Fit collision/red/offroad/repair/trust temperatures independently."""

    heads = ("collision", "red", "offroad", "repair", "trust")
    label_keys = {
        "collision": "collision",
        "red": "red_light_violation",
        "offroad": "offroad",
        "repair": "repair_success",
        "trust": "trust",
    }
    temperatures: dict[str, float] = {}
    for head in heads:
        logits: list[float] = []
        labels: list[bool] = []
        for row in rows:
            logit_key = head if head in row else f"{head}_logit"
            label_key = label_keys[head]
            if label_key not in row:
                # Accept the compact names used by runtime audit rows as well
                # as the explicit H6 field names.  This function is only a
                # dev calibration helper; no alternate label is introduced.
                label_key = {
                    "collision": "collision",
                    "red": "red",
                    "offroad": "offroad",
                    "repair": "repair",
                    "trust": "trust",
                }[head]
            if logit_key not in row or label_key not in row:
                continue
            if row[label_key] is None:
                continue
            logits.append(float(row[logit_key]))
            labels.append(bool(row[label_key]))
        temperatures[head] = fit_temperature(logits, labels, bounds=bounds) if logits else 1.0
    return TemperatureCalibration(temperatures=temperatures, bounds=bounds, rows=len(rows))


@dataclass(frozen=True)
class VLA75RouterCalibration:
    passed: bool
    ema_alpha: float
    hold_ticks: int
    hysteresis: float
    switches: int
    vla_coverage: float
    unsafe_delta: float
    reason: str
    ping_pong: bool = False
    rows: int = 0
    sequences: int = 0
    observed_vla_coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_vla75_router_config(
    rows: Sequence[Mapping[str, Any] | CalibrationRow],
    *,
    alpha_grid: Sequence[float] = (0.25, 0.50, 0.75),
    hold_grid: Sequence[int] = (6, 10, 14),
    hysteresis_grid: Sequence[float] = (0.05, 0.10, 0.20),
    max_unsafe_delta: float = 0.01,
    target_actual_vla_coverage: float = 0.75,
    max_switches_per_30s: int = 2,
    ping_pong_window_ticks: int = 10,
) -> VLA75RouterCalibration:
    """Select temporal parameters by replaying the frozen router contract.

    The old implementation copied the already observed ``applied_source``
    sequence into every grid candidate.  That made alpha/hold/hysteresis
    dead configuration and could report a zero-switch router even when the
    raw World preference oscillated every tick.  This implementation keeps
    source identity as an offline label, reconstructs the raw pair proposal
    from the two candidate predictions, and then replays EMA/hold/hysteresis
    independently for every physical sequence.  No formal or test data is
    accepted here; the caller is responsible for passing development rows.
    """

    if not rows:
        raise ValueError("vla75_router_calibration_rows_required")
    if not alpha_grid or not hold_grid or not hysteresis_grid:
        raise ValueError("vla75_router_calibration_grid_required")
    if int(max_switches_per_30s) < 0 or int(ping_pong_window_ticks) < 1:
        raise ValueError("vla75_router_calibration_stability_bounds")

    def get(row: Mapping[str, Any] | CalibrationRow, key: str, default: Any = None) -> Any:
        if isinstance(row, Mapping):
            return row.get(key, default)
        return getattr(row, key, default)

    def source(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in {"vla", "vla_fast", "vla_slow"}:
            return "vla"
        if text in {"expert", "classic", "classic_expert"}:
            return "expert"
        if text in {"mrm", "minimal_risk", "minimal_risk_brake"}:
            return "mrm"
        return text

    def prediction(row: Mapping[str, Any] | CalibrationRow, name: str) -> Any:
        value = get(row, f"{name}_prediction")
        return value

    def pget(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    def number(value: Any, default: float) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return float(default)
        return converted if math.isfinite(converted) else float(default)

    def prediction_value(
        row: Mapping[str, Any] | CalibrationRow,
        name: str,
        field: str,
        fallback_field: str,
        default: float,
    ) -> float:
        value = prediction(row, name)
        observed = pget(value, field)
        if observed is None:
            observed = get(row, f"{name}_{fallback_field}")
        if observed is None and field == "deployment_score":
            observed = pget(value, "utility")
        return number(observed, default)

    def candidate_metrics(row: Mapping[str, Any] | CalibrationRow) -> dict[str, float | bool]:
        observed_source = source(get(row, "applied_source") or get(row, "source"))
        # Missing prediction fields are supported only for legacy calibration
        # helpers.  In that case the observed source is the honest fallback;
        # v75 training rows always carry both source-blind predictions.
        vla_default = 1.0 if observed_source == "vla" else 0.0
        expert_default = 1.0 if observed_source == "expert" else 0.0
        vla_score = prediction_value(row, "vla", "deployment_score", "score", vla_default)
        expert_score = prediction_value(row, "expert", "deployment_score", "score", expert_default)
        vla_pref = prediction_value(row, "vla", "preference_utility", "preference", vla_score)
        expert_pref = prediction_value(row, "expert", "preference_utility", "preference", expert_score)
        vla_trust = prediction_value(row, "vla", "trust_probability", "trust", 1.0)
        vla_risk = prediction_value(row, "vla", "unsafe_probability", "risk", 0.0)
        expert_trust = prediction_value(row, "expert", "trust_probability", "trust", 1.0)
        expert_risk = prediction_value(row, "expert", "unsafe_probability", "risk", 0.0)
        raw_override = get(row, "raw_preference")
        threshold = number(get(row, "trust_threshold"), 0.50)
        ceiling = number(get(row, "risk_ceiling"), 0.20)
        pair_complete = all(
            math.isfinite(value)
            for value in (
                vla_score,
                expert_score,
                vla_pref,
                expert_pref,
                vla_trust,
                vla_risk,
                expert_trust,
                expert_risk,
            )
        )
        gate = (
            pair_complete
            and vla_score + 1e-12 >= expert_score
            and vla_pref + 1e-12 >= expert_pref
            and vla_trust + 1e-12 >= threshold
            and vla_risk <= ceiling + 1e-12
            and (raw_override is None or bool(raw_override))
        )
        # The raw *proposal* follows the same deployment-score ordering as
        # formal acceptance.  The complete gate is tracked separately and is
        # used to signal a risk/event break; it must not be replaced with an
        # EMA or an already-selected/applied source.
        if vla_score > expert_score + 1e-12:
            raw_source = "vla"
        elif expert_score > vla_score + 1e-12:
            raw_source = "expert"
        else:
            raw_source = "vla" if vla_pref >= expert_pref else "expert"
        return {
            "vla_score": vla_score,
            "expert_score": expert_score,
            "vla_pref": vla_pref,
            "expert_pref": expert_pref,
            "vla_trust": vla_trust,
            "vla_risk": vla_risk,
            "expert_trust": expert_trust,
            "expert_risk": expert_risk,
            "raw_source": raw_source,
            "vla_gate": gate,
            "risk_breach": (not math.isfinite(vla_risk)) or vla_risk > ceiling + 1e-12,
        }

    prepared = [(index, row, candidate_metrics(row)) for index, row in enumerate(rows)]

    def sequence_key(index: int, row: Mapping[str, Any] | CalibrationRow) -> str:
        explicit = get(row, "sequence_id")
        if explicit is not None and str(explicit):
            value = str(explicit)
            arm = get(row, "arm")
            return f"{value}:{arm}" if arm is not None else value
        # If tick metadata exists, pair_id is the physical sequence.  Rows
        # without temporal metadata remain one-row sequences, preserving the
        # old policy-calibration behavior without manufacturing adjacency.
        tick = get(row, "tick")
        pair_id = str(get(row, "pair_id") or f"row-{index}")
        return pair_id if tick is not None else f"__legacy__:{index}"

    grouped: dict[str, list[tuple[int, Any, dict[str, float | bool]]]] = {}
    for index, row, metrics in prepared:
        grouped.setdefault(sequence_key(index, row), []).append((index, row, metrics))
    for sequence in grouped.values():
        sequence.sort(
            key=lambda item: (
                int(get(item[1], "tick")) if get(item[1], "tick") is not None else item[0],
                item[0],
            )
        )

    def replay(alpha: float, hold_ticks: int, hysteresis: float):
        selected_by_index: dict[int, str] = {}
        all_selected: list[str] = []
        total_switches = 0
        max_sequence_switches = 0
        any_ping_pong = False
        for sequence in grouped.values():
            ema: dict[str, float] = {}
            previous_source: str | None = None
            hold_count = 0
            history: list[str] = []
            switches = 0
            previous_row = None
            for index, row, metrics in sequence:
                values = {
                    "vla": float(metrics["vla_pref"]),
                    "expert": float(metrics["expert_pref"]),
                }
                ema = {
                    key: float(alpha) * value + (1.0 - float(alpha)) * ema.get(key, value)
                    for key, value in values.items()
                }
                proposed = str(metrics["raw_source"])
                event_break = bool(
                    get(row, "event_break", False)
                    or get(row, "eligible_changed", False)
                    or get(row, "risk_breach", False)
                    or bool(metrics["risk_breach"])
                )
                if previous_row is not None:
                    previous_tick = get(previous_row, "tick")
                    current_tick = get(row, "tick")
                    if previous_tick is not None and current_tick is not None:
                        try:
                            if int(current_tick) != int(previous_tick) + 1:
                                event_break = True
                        except (TypeError, ValueError):
                            event_break = True
                    if str(get(previous_row, "phase", "unknown")) != str(get(row, "phase", "unknown")):
                        event_break = True
                    previous_hazard = (
                        bool(get(previous_row, "vla_unsafe", False)),
                        bool(get(previous_row, "expert_unsafe", False)),
                    )
                    current_hazard = (
                        bool(get(row, "vla_unsafe", False)),
                        bool(get(row, "expert_unsafe", False)),
                    )
                    if previous_hazard != current_hazard:
                        event_break = True
                    previous_exec = get(previous_row, "executable")
                    current_exec = get(row, "executable")
                    if previous_exec != current_exec:
                        event_break = True
                    # Trust transitions are evaluated from the frozen
                    # prediction rather than an arbitrary applied source.
                    previous_metrics = candidate_metrics(previous_row)
                    previous_trust = bool(float(previous_metrics["vla_trust"]) >= 0.50)
                    current_trust = bool(float(metrics["vla_trust"]) >= 0.50)
                    if previous_trust != current_trust:
                        event_break = True
                ordered = sorted(ema.values(), reverse=True)
                margin = float(ordered[0] - ordered[1]) if len(ordered) >= 2 else 0.0
                keep = (
                    not event_break
                    and previous_source in {"vla", "expert"}
                    and proposed != previous_source
                    and hold_count < int(hold_ticks)
                    and (
                        margin < float(hysteresis)
                        or margin < 1.5
                    )
                )
                selected = previous_source if keep else proposed
                if previous_source is not None and selected != previous_source:
                    switches += 1
                hold_count = hold_count + 1 if keep else 1
                previous_source = selected
                history.append(selected)
                selected_by_index[index] = selected
                all_selected.append(selected)
                previous_row = row
            total_switches += switches
            max_sequence_switches = max(max_sequence_switches, switches)
            for start, first_source in enumerate(history):
                if first_source not in {"vla", "expert"}:
                    continue
                opposite = False
                for value in history[start + 1 : start + int(ping_pong_window_ticks) + 1]:
                    if value in {"vla", "expert"} and value != first_source:
                        opposite = True
                    if opposite and value == first_source:
                        any_ping_pong = True
                        break
                if any_ping_pong:
                    break
        return selected_by_index, all_selected, total_switches, max_sequence_switches, any_ping_pong

    observed_sources = [source(get(row, "applied_source") or get(row, "source")) for row in rows]
    observed_vla_coverage = observed_sources.count("vla") / len(observed_sources)
    baseline_unsafe = sum(bool(get(row, "expert_unsafe", False)) for row in rows) / len(rows)
    best: tuple[tuple[Any, ...], VLA75RouterCalibration] | None = None
    for alpha, hold, hysteresis in product(alpha_grid, hold_grid, hysteresis_grid):
        try:
            alpha_value = float(alpha)
            hold_value = int(hold)
            hysteresis_value = float(hysteresis)
        except (TypeError, ValueError):
            continue
        if not 0.0 < alpha_value <= 1.0 or hold_value < 0 or hysteresis_value < 0.0:
            continue
        selected, selected_sources, switches, max_sequence_switches, ping_pong = replay(
            alpha_value, hold_value, hysteresis_value
        )
        selected_unsafe = sum(
            bool(get(rows[index], "vla_unsafe", False))
            if selected.get(index) == "vla"
            else bool(get(rows[index], "expert_unsafe", False))
            for index in range(len(rows))
        ) / len(rows)
        delta = selected_unsafe - baseline_unsafe
        coverage = selected_sources.count("vla") / len(rows)
        safety_ok = delta <= float(max_unsafe_delta) + 1e-12
        coverage_ok = coverage + 1e-12 >= float(target_actual_vla_coverage)
        stability_ok = (
            not ping_pong and max_sequence_switches <= int(max_switches_per_30s)
        )
        passed = safety_ok and coverage_ok and stability_ok
        candidate = VLA75RouterCalibration(
            passed=passed,
            ema_alpha=alpha_value,
            hold_ticks=hold_value,
            hysteresis=hysteresis_value,
            switches=int(switches),
            vla_coverage=float(coverage),
            unsafe_delta=float(delta),
            reason="dev_grid_selected" if passed else "no_grid_candidate_meets_gates",
            ping_pong=bool(ping_pong),
            rows=len(rows),
            sequences=len(grouped),
            observed_vla_coverage=float(observed_vla_coverage),
        )
        # Gates are lexicographic: safety and coverage first, then temporal
        # stability/minimal switching, followed by deterministic parameters.
        key = (
            int(passed),
            int(safety_ok),
            int(coverage_ok),
            int(stability_ok),
            int(not ping_pong),
            -int(max_sequence_switches),
            -int(switches),
            float(coverage),
            -float(delta),
            -int(hold_value),
            -float(hysteresis_value),
            float(alpha_value),
        )
        if best is None or key > best[0]:
            best = (key, candidate)
    if best is None:
        raise ValueError("vla75_router_calibration_grid_invalid")
    return best[1]


__all__ = [
    "CalibrationRow",
    "VLADeploymentCalibration",
    "calibrate_vla_deployment",
    "TemperatureCalibration",
    "VLA75RouterCalibration",
    "calibrate_temperature_heads",
    "fit_temperature",
    "select_vla75_router_config",
    "temperature_scale_logit",
]
