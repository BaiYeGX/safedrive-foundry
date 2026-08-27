"""Hard acceptance gate for actual VLA-primary closed-loop execution."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_pipeline.h6.config import (
    H6_VLA90_CONFIG_SHA256,
    H6_VLA75_FORMAL_LINEAGES,
    h6_vla75_config_sha256,
)
from data_pipeline.h6.run_lock import verify_run_lock


def _dict(run):
    return run.to_dict() if hasattr(run, "to_dict") else dict(run)


def _unsafe(run: Mapping[str, Any]) -> bool:
    return (
        int(run.get("collision_count", 0)) > 0
        or bool(run.get("red_light_violation", False))
        or float(run.get("off_corridor_duration_s", 0.0)) > 0.25
    )


def _p99(values: Sequence[float]) -> float:
    if not values:
        return float("inf")
    ordered = sorted(float(item) for item in values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))]


def _progress_ci(deltas: Sequence[float], *, rounds: int = 10000, seed: int = 90):
    if not deltas:
        return {"n": 0, "mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)
    )
    return {
        "n": n,
        "mean": sum(deltas) / n,
        "lower_95": means[max(0, int(0.025 * rounds) - 1)],
        "upper_95": means[min(rounds - 1, int(0.975 * rounds) - 1)],
    }


def _ping_pong(sources: Sequence[str], window_ticks: int = 10) -> bool:
    for start, source in enumerate(sources):
        if source not in {"vla", "expert"}:
            continue
        opposite_seen = False
        for end in range(start + 1, min(len(sources), start + window_ticks + 1)):
            if sources[end] in {"vla", "expert"} and sources[end] != source:
                opposite_seen = True
            if opposite_seen and sources[end] == source:
                return True
    return False


def _max_switches_in_window(
    sources: Sequence[str],
    decisions: Sequence[Mapping[str, Any]],
    *,
    window_s: float = 30.0,
) -> int:
    """Return the worst source-switch count in any fixed-time window.

    The contract is a rate bound, not an average over all paired runs.  A
    long run with one oscillatory 30-second segment must fail even if quiet
    segments elsewhere dilute the aggregate rate.  Simulation timestamps are
    preferred; fixed 20 Hz ticks are a deterministic fallback for legacy
    records that only carry tick indices.
    """

    if len(sources) < 2:
        return 0
    switch_times: list[float] = []
    previous = str(sources[0])
    for index in range(1, len(sources)):
        current = str(sources[index])
        if current == previous:
            continue
        raw_time = decisions[index].get("simulation_time_s") if index < len(decisions) else None
        try:
            timestamp = float(raw_time)
        except (TypeError, ValueError):
            timestamp = float(index) * 0.05
        if not math.isfinite(timestamp):
            timestamp = float(index) * 0.05
        switch_times.append(timestamp)
        previous = current
    if not switch_times:
        return 0
    bound = max(0.0, float(window_s))
    return max(
        sum(1 for value in switch_times if start <= value <= start + bound + 1e-12)
        for start in switch_times
    )


def _v2_source(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"vla", "vla_fast", "vla_slow"}:
        return "vla"
    if text in {"expert", "classic", "classic_expert"}:
        return "expert"
    if text in {"mrm", "minimal_risk", "emergency", "hold", "none"}:
        return "mrm"
    return "unknown"


def _v2_prediction_pair(
    score: Mapping[str, Any],
    *,
    candidate_sources: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    raw = score.get("raw_predictions")
    if raw is None:
        raw = score.get("predictions")
    rows = list(raw or ())
    vla = None
    expert = None
    seen_keys: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        source = _v2_source(row.get("source"))
        key = str(row.get("candidate_key", row.get("candidate_id", "")))
        if not key or key in seen_keys:
            # Duplicate/anonymous rows cannot prove that two independent
            # candidates were scored.  Leave one side incomplete so the
            # caller records a denominator-preserving provenance failure.
            continue
        seen_keys.add(key)
        if source == "unknown" and candidate_sources is not None:
            manifest = candidate_sources.get(key)
            if isinstance(manifest, Mapping):
                provenance = manifest.get("provenance")
                source = _v2_source(
                    manifest.get("source")
                    or (provenance.get("source") if isinstance(provenance, Mapping) else None)
                )
        if source == "unknown":
            suffix = key.rsplit(":", 1)[-1]
            source = _v2_source(suffix)
            if source == "unknown":
                source = _v2_source(suffix.rsplit("_", 1)[0])
        if source == "vla" and vla is None:
            vla = row
        elif source == "expert" and expert is None:
            expert = row
    return vla, expert


def _v2_raw_gate(
    decision: Mapping[str, Any],
    *,
    target_world_vla_preference: float,
    max_vla_risk: float = 0.20,
) -> tuple[bool, dict[str, bool], str | None, str | None]:
    """Evaluate only the raw, complete World pair on one decision tick."""

    score = decision.get("world_score")
    score = score if isinstance(score, Mapping) else {}
    candidate_manifest = decision.get("candidates")
    candidate_manifest = candidate_manifest if isinstance(candidate_manifest, Mapping) else None
    vla, expert = _v2_prediction_pair(score, candidate_sources=candidate_manifest)
    reasons = {
        "score": False,
        "pair_preference": False,
        "trust": False,
        "risk": False,
        "pair_completeness": vla is not None and expert is not None,
    }
    raw_preferred_id = None
    raw_preferred_source = None
    if vla is None or expert is None:
        return False, reasons, raw_preferred_id, raw_preferred_source
    try:
        vla_score = float(vla["deployment_score"])
        expert_score = float(expert["deployment_score"])
        vla_pref = float(vla["preference_utility"])
        expert_pref = float(expert["preference_utility"])
        # Every field consumed by the formal gate must be present and finite.
        # A pair with two candidate ids but a missing head is not a complete
        # pair score; it remains in the all-tick denominator and is also a
        # provenance failure.
        for field in (
            "deployment_score",
            "preference_utility",
            "unsafe_probability",
        ):
            if field not in vla or field not in expert:
                raise KeyError(field)
        strict_output = score.get("schema_version") == "safedrive.world.vla75.pair_exec.v1"
        if strict_output:
            raw_rows = score.get("raw_predictions")
            if raw_rows is None:
                raw_rows = score.get("predictions")
            if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)) or len(raw_rows) != 2:
                raise KeyError("pair_cardinality")
            required_fields = (
                "candidate_key",
                "objective_utility",
                "progress_mean_m",
                "progress_logvar",
                "completion_logit",
                "collision_logit",
                "red_light_logit",
                "offroad_logit",
                "jerk_mean_log1p",
                "acceleration_mean_mps2",
                "lateral_acceleration_mean_mps2",
                "repair_success_logit",
                "preference_utility",
                "trust_logit",
                "trust_probability",
                "executable_logit",
                "executable_probability",
                "deployment_score",
                "ensemble_std",
            )
            for row in (vla, expert):
                if any(field not in row for field in required_fields):
                    raise KeyError("prediction_head")
                finite_fields = set(required_fields) | {
                    "unsafe_probability",
                    "trust_probability",
                    "trust_logit",
                    "executable_probability",
                    "executable_logit",
                }
                for field in finite_fields:
                    if field in row and not math.isfinite(float(row[field])):
                        raise ValueError(f"nonfinite_prediction_head:{field}")
        trust_value = vla.get("trust_probability")
        if trust_value is None:
            trust_logit = vla.get("trust_logit")
            if trust_logit is None:
                raise KeyError("trust")
            trust_value = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(trust_logit)))))
        risk_value = vla.get("unsafe_probability")
        if risk_value is None:
            raise KeyError("unsafe_probability")
        vla_trust = float(trust_value)
        vla_risk = float(risk_value)
        if strict_output:
            expert_trust = expert.get("trust_probability")
            if expert_trust is None and expert.get("trust_logit") is not None:
                expert_trust = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(expert["trust_logit"])))) )
            expert_risk = expert.get("unsafe_probability")
            if expert_trust is None or expert_risk is None:
                raise KeyError("expert_trust_or_risk")
        trust_threshold = float(score["trust_threshold"])
        risk_ceiling = float(score["risk_ceiling"])
        values = (
            vla_score,
            expert_score,
            vla_pref,
            expert_pref,
            vla_trust,
            vla_risk,
            trust_threshold,
            risk_ceiling,
        )
        if strict_output:
            values += (float(expert_trust), float(expert_risk))
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons["pair_completeness"] = False
        return False, reasons, raw_preferred_id, raw_preferred_source
    if not all(math.isfinite(value) for value in values):
        return False, reasons, raw_preferred_id, raw_preferred_source
    reasons["score"] = vla_score + 1e-12 >= expert_score
    reasons["pair_preference"] = vla_pref + 1e-12 >= expert_pref
    reasons["trust"] = vla_trust + 1e-12 >= trust_threshold
    reasons["risk"] = (
        risk_ceiling <= float(max_vla_risk) + 1e-12
        and vla_risk <= min(risk_ceiling, float(max_vla_risk)) + 1e-12
    )
    # The raw selector is derived from the two score heads, never from EMA,
    # hold, selected-source or a force-VLA flag.  An explicitly persisted order
    # is checked by the caller as provenance, but cannot change this value.
    derived = max(
        (vla, expert),
        key=lambda row: (
            float(row["deployment_score"]),
            float(row["preference_utility"]),
            str(row.get("candidate_key", "")),
        ),
    )
    raw_preferred_id = str(derived.get("candidate_key"))
    raw_preferred_source = _v2_source(
        derived.get("source") or raw_preferred_id.rsplit(":", 1)[-1]
    )
    return all(reasons.values()), reasons, raw_preferred_id, raw_preferred_source


def _v2_applied_source(
    decision: Mapping[str, Any],
) -> tuple[str, str | None, str | None, bool]:
    """Resolve post-control source; return source, id, reason, valid."""

    applied_id = (
        decision.get("applied_candidate_id")
        or decision.get("applied_id")
        or decision.get("applied_executed_id")
    )
    applied_source = _v2_source(
        decision.get("applied_source") or decision.get("applied_candidate_source")
    )
    mode = str(decision.get("applied_mode") or "").upper()
    candidates = decision.get("candidates")
    candidate = candidates.get(str(applied_id)) if isinstance(candidates, Mapping) and applied_id is not None else None
    candidate_source_value = None
    if isinstance(candidate, Mapping):
        candidate_source_value = candidate.get("source")
        if candidate_source_value is None and isinstance(candidate.get("provenance"), Mapping):
            candidate_source_value = candidate["provenance"].get("source")
    candidate_source = _v2_source(candidate_source_value)
    if applied_source == "unknown" and candidate_source != "unknown":
        applied_source = candidate_source
    if applied_id is None and applied_source == "unknown":
        # An explicit non-tracking Safety decision is a valid MRM outcome.  A
        # missing provenance field is not silently treated as a VLA success.
        kind = str(decision.get("safety_decision_kind") or "").upper()
        if kind in {"EMERGENCY", "MINIMAL_RISK", "HARD_REJECT", "HOLD_NO_EXEC"} or mode in {
            "EMERGENCY_BRAKE", "MINIMAL_RISK_BRAKE", "HOLD_NO_EXEC"
        }:
            return "mrm", None, None, True
        return "mrm", None, "missing_applied_binding", False
    if applied_source not in {"vla", "expert", "mrm"}:
        return "mrm", str(applied_id) if applied_id is not None else None, "unknown_applied_source", False
    if applied_source == "mrm":
        if applied_id is not None:
            return "mrm", str(applied_id), "mrm_with_candidate_id", False
        if mode == "TRACK_APPROVED":
            return "mrm", None, "track_approved_without_candidate", False
        return "mrm", None, None, True
    if not applied_id:
        return "mrm", None, "missing_applied_id", False
    if mode != "TRACK_APPROVED":
        return "mrm", str(applied_id), "candidate_not_track_approved", False
    if not isinstance(candidates, Mapping):
        return "mrm", str(applied_id), "candidate_manifest_missing", False
    if isinstance(candidates, Mapping) and str(applied_id) not in candidates:
        repair = decision.get("repair")
        if (
            isinstance(repair, Mapping)
            and bool(repair.get("success"))
            and repair.get("post_repair_id") == applied_id
            and repair.get("pre_repair_id")
            and repair.get("method", repair.get("mode"))
            and repair.get("final_validation") is True
        ):
            # A bounded Safety repair is allowed to mint a new trajectory id;
            # its pre-repair candidate/source remains the provenance anchor.
            repaired_source = _v2_source(
                decision.get("safety_executed_source")
                or decision.get("selected_candidate_source")
                or decision.get("selected_source")
            )
            if repaired_source in {"vla", "expert"}:
                return repaired_source, str(applied_id), None, True
        return "mrm", str(applied_id), "orphan_applied_id", False
    if candidate_source == "unknown":
        return "mrm", str(applied_id), "applied_candidate_source_missing", False
    if candidate_source != applied_source:
        return "mrm", str(applied_id), "applied_source_id_mismatch", False
    return applied_source, str(applied_id), None, True


def _v2_phase(run: Mapping[str, Any], tick: int, total: int) -> str:
    decision = list(run.get("decisions", ()))[tick] if tick < len(run.get("decisions", ())) else {}
    explicit = str(decision.get("phase") or run.get("phase") or "").strip()
    if explicit:
        return explicit
    boundaries = run.get("phase_boundaries")
    if not isinstance(boundaries, Mapping):
        scenario = run.get("scenario")
        boundaries = scenario.get("phase_boundaries") if isinstance(scenario, Mapping) else None
    if isinstance(boundaries, Mapping) and boundaries:
        try:
            event_tick = min(int(value) for value in boundaries.values())
        except (TypeError, ValueError):
            event_tick = None
        if event_tick is not None:
            if int(tick) < event_tick:
                return "pre_event"
            if int(tick) == event_tick:
                return "intervention"
            return "recovery"
    third = max(1, int(total) // 3)
    return "early" if tick < third else "late" if tick >= 2 * third else "middle"


def evaluate_vla75_gate(
    runs: Sequence[Mapping[str, Any]],
    *,
    target_arm: str = "on",
    baseline_arm: str = "off",
    target_world_vla_preference: float = 0.90,
    target_actual_vla_coverage: float = 0.75,
    max_classic_mrm_share: float = 0.25,
    max_unsafe_delta: float = 0.01,
    forbid_target_only_unsafe: bool = True,
    scorer_deadline_ms: float = 50.0,
    max_switches_per_30s: int = 2,
    ping_pong_window_ticks: int = 10,
    lineage_id: str | None = None,
    expected_config_sha256: str | None = None,
    max_world_incremental_gib: float = 1.5,
    max_vla_risk: float = 0.20,
    max_whole_gpu_peak_gib: float | None = 14.5,
    require_spectator_follow: bool = True,
    run_lock: Mapping[str, Any] | None = None,
    run_lock_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the immutable H6 VLA75 v2 contract.

    Raw World coverage uses every decision tick as denominator.  Applied VLA
    coverage is computed only from the post-control binding, so selected or
    Safety-executed candidates, EMA/hold, and development force-VLA cannot
    inflate either formal metric.
    """

    if not 0.0 <= float(target_world_vla_preference) <= 1.0:
        raise ValueError("target_world_vla_preference_out_of_range")
    if not 0.0 <= float(target_actual_vla_coverage) <= 1.0:
        raise ValueError("target_actual_vla_coverage_out_of_range")
    if not 0.0 <= float(max_classic_mrm_share) <= 1.0:
        raise ValueError("max_classic_mrm_share_out_of_range")
    if float(max_unsafe_delta) < 0.0:
        raise ValueError("max_unsafe_delta_must_be_nonnegative")
    if float(scorer_deadline_ms) <= 0.0:
        raise ValueError("scorer_deadline_ms_must_be_positive")
    if int(max_switches_per_30s) < 0 or int(ping_pong_window_ticks) < 1:
        raise ValueError("stability_contract_invalid")
    if not 0.0 <= float(max_vla_risk) <= 1.0:
        raise ValueError("max_vla_risk_out_of_range")
    if max_whole_gpu_peak_gib is not None and float(max_whole_gpu_peak_gib) <= 0.0:
        raise ValueError("max_whole_gpu_peak_gib_must_be_positive")
    rows = [_dict(run) for run in runs]
    target = [run for run in rows if run.get("arm") == target_arm]
    baseline_rows = [run for run in rows if run.get("arm") == baseline_arm]
    baseline = {str(run.get("pair_id")): run for run in baseline_rows}
    failures: set[str] = set()
    provenance: list[str] = []
    raw_reason_counts: dict[str, int] = {name: 0 for name in ("score", "pair_preference", "trust", "risk", "pair_completeness")}
    raw_group: dict[str, dict[str, int]] = {}
    transition_matrix: dict[str, int] = {}
    transition_groups: dict[str, dict[str, int]] = {}
    actual_sources: list[str] = []
    total_switches = 0
    max_window_switches = 0
    total_sim_s = 0.0
    scorer_latencies: list[float] = []
    deadline_misses = 0
    raw_high = 0
    raw_pair_scored = 0
    applied_counts = {"vla": 0, "expert": 0, "mrm": 0}
    ping_pong_runs: list[str] = []
    invalid_applied: list[str] = []
    guard_verdict_counts = {"PASS": 0, "REVIEW": 0, "REJECT": 0, "UNKNOWN": 0}
    world_defer_count = 0
    safety_fallback_count = 0
    repair_method_counts: dict[str, int] = {}
    observed_max_world_incremental_gib = 0.0
    observed_max_whole_gpu_peak_gib = 0.0

    matrix_report: dict[str, Any] | None = None
    target_pair_counts: dict[str, int] = {}
    for run in target:
        key = str(run.get("pair_id"))
        target_pair_counts[key] = target_pair_counts.get(key, 0) + 1
    baseline_pair_counts: dict[str, int] = {}
    for run in baseline_rows:
        key = str(run.get("pair_id"))
        baseline_pair_counts[key] = baseline_pair_counts.get(key, 0) + 1
    duplicate_target_pairs = sorted(key for key, count in target_pair_counts.items() if count != 1)
    duplicate_baseline_pairs = sorted(key for key, count in baseline_pair_counts.items() if count != 1)
    if duplicate_target_pairs or duplicate_baseline_pairs:
        failures.add("matrix_coverage")
        provenance.append("duplicate_matrix_runs")
    if run_lock is not None:
        expected_pair_ids = {
            str(item) for item in (run_lock.get("matrix_pair_ids") or ())
        }
        observed_target_ids = {str(run.get("pair_id")) for run in target}
        observed_baseline_ids = {str(run.get("pair_id")) for run in baseline.values()}
        missing_target = sorted(expected_pair_ids - observed_target_ids)
        missing_baseline = sorted(expected_pair_ids - observed_baseline_ids)
        unexpected = sorted(
            (observed_target_ids | observed_baseline_ids) - expected_pair_ids
        )
        matrix_report = {
            "expected_pairs": len(expected_pair_ids),
            "observed_target_pairs": len(observed_target_ids),
            "observed_baseline_pairs": len(observed_baseline_ids),
            "missing_target": missing_target,
            "missing_baseline": missing_baseline,
            "unexpected": unexpected,
            "duplicate_target_pairs": duplicate_target_pairs,
            "duplicate_baseline_pairs": duplicate_baseline_pairs,
        }
        if missing_target or missing_baseline or unexpected:
            failures.add("matrix_coverage")

    expected_hash = expected_config_sha256
    expected_run_lock_hash = (
        None if run_lock is None else str(run_lock.get("lock_sha256") or "")
    )
    expected_model_hash = (
        None
        if run_lock is None
        else str(run_lock.get("model_hash") or "") or None
    )
    expected_feature_schema = (
        None
        if run_lock is None
        else str(run_lock.get("feature_schema") or "") or None
    )
    lock_calibration = run_lock.get("calibration") if isinstance(run_lock, Mapping) else None
    lock_deployment = (
        lock_calibration.get("deployment")
        if isinstance(lock_calibration, Mapping)
        and isinstance(lock_calibration.get("deployment"), Mapping)
        else lock_calibration
        if isinstance(lock_calibration, Mapping)
        else None
    )
    def _locked_float(name: str) -> float | None:
        if not isinstance(lock_deployment, Mapping) or lock_deployment.get(name) is None:
            return None
        try:
            value = float(lock_deployment[name])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    expected_trust_threshold = _locked_float("trust_threshold")
    expected_risk_ceiling = _locked_float("risk_ceiling")
    expected_router_calibration = (
        lock_calibration.get("router")
        if isinstance(lock_calibration, Mapping)
        and isinstance(lock_calibration.get("router"), Mapping)
        else None
    )
    lock_worktree = run_lock.get("worktree") if isinstance(run_lock, Mapping) else None
    expected_worktree_hash = (
        str(lock_worktree.get("full_worktree_hash") or "") or None
        if isinstance(lock_worktree, Mapping)
        else None
    )
    if lineage_id is not None and run_lock is None:
        # A lineage identifies a formal run, and formal runs are only valid
        # when the immutable config/matrix/checkpoint lock is supplied.  The
        # lock remains optional for generic v2 unit fixtures that omit a
        # lineage, preserving a useful offline test surface.
        provenance.append("run_lock_missing")
    if run_lock is not None and not expected_run_lock_hash:
        provenance.append("run_lock_hash_missing")
    if run_lock is not None:
        lock_verification = verify_run_lock(
            run_lock,
            root=run_lock_root or Path(__file__).resolve().parents[3],
        )
        if not lock_verification["valid"]:
            provenance.extend(f"run_lock:{item}" for item in lock_verification["failures"])
    if lineage_id is not None:
        try:
            lineage = str(lineage_id).lower()
            expected_hash = h6_vla75_config_sha256(lineage)
        except ValueError:
            provenance.append(f"unknown_lineage:{lineage_id}")
    if run_lock is not None and lineage_id is not None:
        if str(run_lock.get("lineage_id", "")).lower() != str(lineage_id).lower():
            provenance.append("run_lock:lineage_argument_mismatch")
    expected_manifest_kind = (
        None
        if lineage_id is None
        else f"h6_vla75_fresh_lineage_{str(lineage_id).lower()}"
    )
    strict_provenance = lineage_id is not None or run_lock is not None
    required_decision_fields = (
        "raw_preferred_candidate_id",
        "raw_preferred_source",
        "raw_gate_reasons",
        "stabilized_preferred_candidate_id",
        "stabilized_preferred_source",
        "selected_candidate_id",
        "selected_candidate_source",
        "safety_executed_candidate_id",
        "safety_executed_source",
        "applied_candidate_id",
        "applied_source",
        "applied_candidate_source",
        "repair_input_id",
        "repair_output_id",
        "repair_method",
        "repair_success",
        "repair_final_validation",
        "candidate_hashes",
        "model_hash",
        "feature_schema",
        "worktree_hash",
    )
    for run in target:
        pair_id = str(run.get("pair_id"))
        if expected_hash is None:
            expected_hash = str(run.get("config_sha256") or "") or None
        if not run.get("config_sha256"):
            provenance.append(f"missing_config_hash:{pair_id}")
        elif expected_hash is not None and run.get("config_sha256") != expected_hash:
            provenance.append(f"config_hash:{pair_id}")
        if expected_run_lock_hash is not None:
            observed_lock = str(run.get("run_lock_sha256") or "")
            if observed_lock != expected_run_lock_hash:
                provenance.append(f"run_lock:{pair_id}")
        if str(run.get("schema_version", "")).startswith("safedrive.vla90"):
            provenance.append(f"v1_run_in_v2_gate:{pair_id}")
        if run.get("schema_version") != "safedrive.h6.vla75.run.v2":
            provenance.append(f"vla75_run_schema:{pair_id}")
        if expected_router_calibration is not None:
            observed_router = run.get("router_calibration")
            if not isinstance(observed_router, Mapping) or dict(observed_router) != dict(expected_router_calibration):
                provenance.append(f"router_calibration:{pair_id}")
        if expected_manifest_kind is not None and run.get("manifest_kind") != expected_manifest_kind:
            provenance.append(f"manifest_lineage:{pair_id}")
        if not bool(run.get("ok")):
            provenance.append(f"run_not_ok:{pair_id}")
        if require_spectator_follow and (
            int(run.get("spectator_follow_updates", 0)) <= 0
            or run.get("spectator_follow_error") not in (None, "")
        ):
            provenance.append(f"spectator_follow:{pair_id}")
        decisions = list(run.get("decisions", ()))
        ticks = int(run.get("ticks_executed", 0))
        if len(decisions) != ticks:
            provenance.append(f"decision_tick_mismatch:{pair_id}")
        try:
            run_duration_s = float(run.get("simulation_duration_s", 0.05 * len(decisions)))
        except (TypeError, ValueError):
            run_duration_s = float("nan")
            provenance.append(f"simulation_duration:{pair_id}")
        if math.isfinite(run_duration_s) and run_duration_s >= 0.0:
            total_sim_s += run_duration_s
        else:
            provenance.append(f"simulation_duration:{pair_id}")
        run_sources: list[str] = []
        for tick, decision in enumerate(decisions):
            if not isinstance(decision, Mapping):
                decision = {}
                provenance.append(f"malformed_decision:{pair_id}:{tick}")
            if strict_provenance:
                for field in required_decision_fields:
                    if field not in decision:
                        provenance.append(f"decision_field_missing:{pair_id}:{tick}:{field}")
                raw_reasons_payload = decision.get("raw_gate_reasons")
                if not isinstance(raw_reasons_payload, Mapping):
                    provenance.append(f"raw_gate_reasons_missing:{pair_id}:{tick}")
                else:
                    for reason_name in raw_reason_counts:
                        if reason_name not in raw_reasons_payload:
                            provenance.append(
                                f"raw_gate_reason_missing:{pair_id}:{tick}:{reason_name}"
                            )
            decision_world = decision.get("world_score")
            decision_world = decision_world if isinstance(decision_world, Mapping) else {}
            if expected_trust_threshold is not None and decision_world:
                try:
                    if abs(float(decision_world.get("trust_threshold")) - expected_trust_threshold) > 1e-12:
                        provenance.append(f"trust_threshold_lock:{pair_id}:{tick}")
                except (TypeError, ValueError):
                    provenance.append(f"trust_threshold_lock:{pair_id}:{tick}")
            if expected_risk_ceiling is not None and decision_world:
                try:
                    if abs(float(decision_world.get("risk_ceiling")) - expected_risk_ceiling) > 1e-12:
                        provenance.append(f"risk_ceiling_lock:{pair_id}:{tick}")
                except (TypeError, ValueError):
                    provenance.append(f"risk_ceiling_lock:{pair_id}:{tick}")
            decision_worktree = decision.get("worktree")
            decision_worktree = decision_worktree if isinstance(decision_worktree, Mapping) else {}
            run_worktree = run.get("worktree")
            run_worktree = run_worktree if isinstance(run_worktree, Mapping) else {}
            if lineage_id is not None and isinstance(lock_worktree, Mapping) and lock_worktree:
                for identity_key in (
                    "head",
                    "branch",
                    "full_worktree_hash",
                    "untracked_manifest_sha256",
                ):
                    expected_identity = lock_worktree.get(identity_key)
                    if expected_identity is None:
                        continue
                    if run_worktree.get(identity_key) != expected_identity:
                        provenance.append(
                            f"worktree_identity_lock:{pair_id}:{tick}:{identity_key}"
                        )
            raw_ok, reasons, raw_id, raw_source = _v2_raw_gate(
                decision,
                target_world_vla_preference=target_world_vla_preference,
                max_vla_risk=max_vla_risk,
            )
            scored_ids: set[str] = set()
            eligible_ids: set[str] = set()
            if strict_provenance and decision_world:
                guard_payload = decision.get("guard")
                guard_payload = guard_payload if isinstance(guard_payload, Mapping) else {}
                scored_payload = decision_world.get("raw_predictions")
                if scored_payload is None:
                    scored_payload = decision_world.get("predictions")
                scored_ids = {
                    str(item.get("candidate_key", item.get("candidate_id", "")))
                    for item in (scored_payload or ())
                    if isinstance(item, Mapping)
                }
                eligible_ids = {
                    str(candidate_id)
                    for candidate_id, value in guard_payload.items()
                    if isinstance(value, Mapping)
                    and str(value.get("verdict", "")).upper() in {"PASS", "REVIEW"}
                }
                if scored_ids and (not eligible_ids or not scored_ids.issubset(eligible_ids)):
                    provenance.append(f"world_guard_eligibility:{pair_id}:{tick}")
            for name, ok in reasons.items():
                if not ok:
                    raw_reason_counts[name] += 1
            if reasons.get("pair_completeness"):
                raw_pair_scored += 1
            if raw_ok:
                raw_high += 1
            elif not reasons.get("pair_completeness"):
                # A missing/incomplete pair is both a metric miss and a
                # provenance failure.  Ordinary score/trust/risk misses are
                # reported through the raw reason histogram only; they are
                # legitimate negative Evidence, not malformed Evidence.
                provenance.append(f"raw_pair_incomplete:{pair_id}:{tick}")
            # Persisted raw fields are an audit trail, never an input to the
            # metric.  If a collector rewrites them after EMA/hold/selected
            # routing, retain the score-derived result above and fail the
            # provenance contract instead of allowing the rewrite to pass.
            persisted_raw_id = decision.get("raw_preferred_candidate_id")
            if strict_provenance and reasons.get("pair_completeness"):
                # A complete raw pair must have a concrete, score-derived
                # provenance anchor.  Merely including the JSON keys with
                # null values is not enough to prove that the collector kept
                # raw and stabilized routing separate.
                if persisted_raw_id is None or decision.get("raw_preferred_source") is None:
                    provenance.append(f"raw_preferred_binding_missing:{pair_id}:{tick}")
                if raw_id is not None and scored_ids and str(raw_id) not in scored_ids:
                    provenance.append(f"raw_preferred_id_orphan:{pair_id}:{tick}")
                candidates_manifest = decision.get("candidates")
                if not isinstance(candidates_manifest, Mapping):
                    provenance.append(f"candidate_manifest_missing:{pair_id}:{tick}")
                else:
                    manifest_sources = {
                        _v2_source(
                            value.get("source")
                            or (
                                value.get("provenance", {}).get("source")
                                if isinstance(value.get("provenance"), Mapping)
                                else None
                            )
                        )
                        for value in candidates_manifest.values()
                        if isinstance(value, Mapping)
                    }
                    if not {"vla", "expert"}.issubset(manifest_sources):
                        provenance.append(f"candidate_manifest_sources:{pair_id}:{tick}")
                if raw_id is not None and isinstance(candidates_manifest, Mapping):
                    manifest = candidates_manifest.get(str(raw_id))
                    if isinstance(manifest, Mapping):
                        manifest_provenance = manifest.get("provenance")
                        manifest_source = _v2_source(
                            manifest.get("source")
                            or (
                                manifest_provenance.get("source")
                                if isinstance(manifest_provenance, Mapping)
                                else None
                            )
                        )
                        if manifest_source != "unknown" and _v2_source(raw_source) != manifest_source:
                            provenance.append(f"raw_preferred_manifest_source:{pair_id}:{tick}")
            if persisted_raw_id is not None and raw_id is not None and str(persisted_raw_id) != str(raw_id):
                provenance.append(f"raw_preferred_id_mismatch:{pair_id}:{tick}")
            persisted_raw_source = decision.get("raw_preferred_source")
            if (
                persisted_raw_source is not None
                and raw_source is not None
                and _v2_source(persisted_raw_source) != _v2_source(raw_source)
            ):
                provenance.append(f"raw_preferred_source_mismatch:{pair_id}:{tick}")
            persisted_reasons = decision.get("raw_gate_reasons")
            if isinstance(persisted_reasons, Mapping):
                for name, expected in reasons.items():
                    if name in persisted_reasons and bool(persisted_reasons[name]) != bool(expected):
                        provenance.append(f"raw_gate_reason_mismatch:{pair_id}:{tick}:{name}")
            if strict_provenance and decision_world:
                disposition = str(decision_world.get("disposition") or "")
                stabilized_id = decision.get("stabilized_preferred_candidate_id")
                selected_id = decision.get("selected_candidate_id")
                if disposition == "ranked" and (
                    stabilized_id is None
                    or decision.get("stabilized_preferred_source") is None
                ):
                    provenance.append(f"stabilized_binding_missing:{pair_id}:{tick}")
                if stabilized_id is not None and eligible_ids and str(stabilized_id) not in eligible_ids:
                    provenance.append(f"stabilized_candidate_ineligible:{pair_id}:{tick}")
                if selected_id is not None and eligible_ids and str(selected_id) not in eligible_ids:
                    provenance.append(f"selected_candidate_ineligible:{pair_id}:{tick}")
            phase = _v2_phase(run, tick, max(1, len(decisions)))
            scenario = dict(run.get("scenario") or {})
            group = "|".join(
                str(scenario.get(key, "unknown")) for key in ("map_name", "family", "weather")
            ) + f"|{phase}"
            slot = raw_group.setdefault(group, {"ticks": 0, "raw_high": 0, **{name: 0 for name in raw_reason_counts}})
            slot["ticks"] += 1
            slot["raw_high"] += int(raw_ok)
            for name, ok in reasons.items():
                slot[name] += int(ok)

            source, applied_id, applied_reason, applied_valid = _v2_applied_source(decision)
            if not applied_valid:
                invalid_applied.append(f"{pair_id}:{tick}:{applied_reason}")
                provenance.append(f"applied_binding:{pair_id}:{tick}:{applied_reason}")
            applied_counts[source] += 1
            actual_sources.append(source)
            run_sources.append(source)
            guard_payload = decision.get("guard")
            guard_payload = guard_payload if isinstance(guard_payload, Mapping) else {}
            for guard in guard_payload.values():
                if not isinstance(guard, Mapping):
                    guard = {}
                verdict = str(guard.get("verdict") or "UNKNOWN").upper()
                guard_verdict_counts[verdict if verdict in guard_verdict_counts else "UNKNOWN"] += 1
            world_score_payload = decision_world
            if str(world_score_payload.get("disposition") or "").startswith("defer"):
                world_defer_count += 1
            if str(decision.get("applied_mode") or "").upper() != "TRACK_APPROVED":
                safety_fallback_count += 1
            selected_source = _v2_source(
                decision.get("selected_candidate_source")
                or decision.get("selected_source")
            )
            safety_source = _v2_source(
                decision.get("safety_executed_source")
                or decision.get("executed_source")
            )
            if selected_source == "unknown":
                routing_payload = decision.get("routing")
                routing_payload = routing_payload if isinstance(routing_payload, Mapping) else {}
                selected_id = decision.get("selected_candidate_id") or routing_payload.get("selected_candidate_id")
                selected_source = _v2_source(str(selected_id).rsplit(":", 1)[-1]) if selected_id else "unknown"
            if safety_source == "unknown":
                executed_id = decision.get("safety_executed_candidate_id") or decision.get("executed_candidate_id")
                safety_source = _v2_source(str(executed_id).rsplit(":", 1)[-1]) if executed_id else "mrm"
            safety_executed_id = (
                decision.get("safety_executed_candidate_id")
                or decision.get("safety_executed_id")
                or decision.get("executed_candidate_id")
            )
            # ``applied_candidate_id`` is written after the control adapter
            # returns.  It normally equals Safety's executed id; the only
            # legitimate difference is a bounded, successfully revalidated
            # repair whose post-repair id is the applied id.
            if source in {"vla", "expert"} and not safety_executed_id:
                provenance.append(f"safety_executed_binding_missing:{pair_id}:{tick}")
            if (
                applied_id is not None
                and safety_executed_id is not None
                and str(applied_id) != str(safety_executed_id)
            ):
                repair = decision.get("repair")
                repaired_binding = bool(
                    isinstance(repair, Mapping)
                    and repair.get("success") is True
                    and repair.get("pre_repair_id")
                    and repair.get("post_repair_id") == applied_id
                    and repair.get("final_validation") is True
                )
                if not repaired_binding:
                    provenance.append(f"executed_applied_id_mismatch:{pair_id}:{tick}")
            transition = f"{selected_source}->{safety_source}->{source}"
            transition_matrix[transition] = transition_matrix.get(transition, 0) + 1
            transition_groups.setdefault(group, {})[transition] = (
                transition_groups.setdefault(group, {}).get(transition, 0) + 1
            )
            if source in {"vla", "expert"} and safety_source in {"vla", "expert"} and source != safety_source:
                provenance.append(f"safety_applied_source_mismatch:{pair_id}:{tick}")
            if decision.get("executed_source") and _v2_source(decision.get("executed_source")) not in {source, "mrm"}:
                provenance.append(f"legacy_executed_source_mismatch:{pair_id}:{tick}")
            repair = decision.get("repair")
            if strict_provenance:
                flat_repair = {
                    "pre_repair_id": decision.get("repair_input_id"),
                    "post_repair_id": decision.get("repair_output_id"),
                    "method": decision.get("repair_method"),
                    "success": decision.get("repair_success"),
                    "final_validation": decision.get("repair_final_validation"),
                }
                if isinstance(repair, Mapping):
                    for field in ("pre_repair_id", "post_repair_id", "success", "final_validation"):
                        if field in repair and flat_repair[field] != repair.get(field):
                            provenance.append(f"repair_field_mismatch:{pair_id}:{tick}:{field}")
                    repair_method_value = repair.get("method", repair.get("mode"))
                    if repair_method_value is not None and flat_repair["method"] != repair_method_value:
                        provenance.append(f"repair_field_mismatch:{pair_id}:{tick}:method")
                elif any(value is not None for value in flat_repair.values()):
                    provenance.append(f"repair_payload_missing:{pair_id}:{tick}")
            if isinstance(repair, Mapping) and bool(repair.get("success")):
                method = str(repair.get("method") or repair.get("mode") or "")
                if not repair.get("pre_repair_id") or not repair.get("post_repair_id") or not method:
                    provenance.append(f"repair_binding_missing:{pair_id}:{tick}")
                repair_method_counts[method] = repair_method_counts.get(method, 0) + 1
                final_validation = repair.get("final_validation")
                if final_validation is not True:
                    provenance.append(f"repair_not_revalidated:{pair_id}:{tick}")
            latency = decision.get("scorer_latency_ms")
            if latency is not None:
                try:
                    value = float(latency)
                    if math.isfinite(value):
                        scorer_latencies.append(value)
                    else:
                        provenance.append(f"nonfinite_scorer_latency:{pair_id}:{tick}")
                except (TypeError, ValueError):
                    provenance.append(f"malformed_scorer_latency:{pair_id}:{tick}")
            else:
                provenance.append(f"missing_scorer_latency:{pair_id}:{tick}")
            deadline_misses += int(bool(decision.get("deadline_miss")))
            raw_hashes = decision.get("candidate_hashes") or decision.get("candidate_sha256")
            strict_hashes = lineage_id is not None or run_lock is not None
            if (
                not isinstance(raw_hashes, Mapping)
                or len(raw_hashes) < 2
                or (
                    strict_hashes
                    and any(
                        not str(key)
                        or not isinstance(value, str)
                        or len(value) != 64
                        or any(character not in "0123456789abcdefABCDEF" for character in value)
                        for key, value in raw_hashes.items()
                    )
                )
            ):
                provenance.append(f"candidate_hashes:{pair_id}:{tick}")
            if strict_hashes and isinstance(raw_hashes, Mapping):
                candidate_manifest = decision.get("candidates")
                if isinstance(candidate_manifest, Mapping) and candidate_manifest:
                    if {str(key) for key in raw_hashes} != {str(key) for key in candidate_manifest}:
                        provenance.append(f"candidate_hash_ids:{pair_id}:{tick}")
            if not decision.get("model_hash") and not decision_world.get("model_hash"):
                provenance.append(f"model_hash:{pair_id}:{tick}")
            elif lineage_id is not None:
                model_value = decision.get("model_hash") or decision_world.get("model_hash")
                world_model_value = decision_world.get("model_hash")
                if decision.get("model_hash") and world_model_value and decision.get("model_hash") != world_model_value:
                    provenance.append(f"model_hash_decision_world_mismatch:{pair_id}:{tick}")
                if not isinstance(model_value, str) or len(model_value) != 64 or any(
                    character not in "0123456789abcdefABCDEF" for character in model_value
                ):
                    provenance.append(f"model_hash:{pair_id}:{tick}")
                if expected_model_hash is not None and model_value != expected_model_hash:
                    provenance.append(f"model_hash_lock:{pair_id}:{tick}")
            if not decision.get("feature_schema") and not decision_world.get("feature_schema"):
                provenance.append(f"feature_schema:{pair_id}:{tick}")
            elif expected_feature_schema is not None:
                feature_value = decision.get("feature_schema") or decision_world.get("feature_schema")
                world_feature_value = decision_world.get("feature_schema")
                if decision.get("feature_schema") and world_feature_value and decision.get("feature_schema") != world_feature_value:
                    provenance.append(f"feature_schema_decision_world_mismatch:{pair_id}:{tick}")
                if feature_value != expected_feature_schema:
                    provenance.append(f"feature_schema_lock:{pair_id}:{tick}")
            if not decision.get("worktree_hash") and not run.get("worktree_hash") and not (
                run_worktree.get("worktree_diff_sha256") or run_worktree.get("full_worktree_hash")
            ):
                provenance.append(f"worktree_hash:{pair_id}:{tick}")
            elif lineage_id is not None:
                worktree_value = (
                    decision.get("worktree_hash")
                    or run.get("worktree_hash")
                    or decision_worktree.get("full_worktree_hash")
                    or run_worktree.get("full_worktree_hash")
                    or run_worktree.get("worktree_diff_sha256")
                )
                if not isinstance(worktree_value, str) or len(worktree_value) != 64 or any(
                    character not in "0123456789abcdefABCDEF" for character in worktree_value
                ):
                    provenance.append(f"worktree_hash:{pair_id}:{tick}")
                if expected_worktree_hash is not None and worktree_value != expected_worktree_hash:
                    provenance.append(f"worktree_hash_lock:{pair_id}:{tick}")
                decision_worktree_value = decision.get("worktree_hash")
                run_worktree_value = run.get("worktree_hash")
                if decision_worktree_value and run_worktree_value and decision_worktree_value != run_worktree_value:
                    provenance.append(f"worktree_hash_decision_run_mismatch:{pair_id}:{tick}")
        if _ping_pong(run_sources, ping_pong_window_ticks):
            ping_pong_runs.append(pair_id)
        # Use the post-control sequence for the stability claim.  The stored
        # aggregate is accepted only as a cross-check, never as a substitute.
        computed_switches = sum(
            run_sources[i] != run_sources[i - 1]
            for i in range(1, len(run_sources))
        )
        reported_switches = run.get("actual_switch_count")
        if reported_switches is not None and int(reported_switches) != computed_switches:
            provenance.append(f"switch_count_mismatch:{pair_id}")
        total_switches += computed_switches
        window_switches = _max_switches_in_window(run_sources, decisions)
        max_window_switches = max(max_window_switches, window_switches)
        if window_switches > int(max_switches_per_30s):
            failures.add("switch_rate")
        if lineage_id is not None or run_lock is not None:
            incremental = run.get("world_incremental_gpu_gib")
            if incremental is None:
                incremental = run.get("world_incremental_gpu_gb")
            if incremental is None:
                provenance.append(f"world_incremental_gpu_missing:{pair_id}")
                failures.add("resource_gpu")
            try:
                incremental_value = float(incremental)
            except (TypeError, ValueError):
                incremental_value = float("nan")
            if not math.isfinite(incremental_value):
                provenance.append(f"world_incremental_gpu_missing:{pair_id}")
                failures.add("resource_gpu")
            elif incremental_value > max_world_incremental_gib + 1e-12:
                failures.add("resource_gpu")
            else:
                observed_max_world_incremental_gib = max(
                    observed_max_world_incremental_gib, incremental_value
                )
            whole = run.get("whole_gpu_peak_gb")
            if max_whole_gpu_peak_gib is not None and whole is None:
                provenance.append(f"whole_gpu_peak_missing:{pair_id}")
                failures.add("resource_gpu")
            if max_whole_gpu_peak_gib is not None:
                try:
                    whole_value = float(whole)
                except (TypeError, ValueError):
                    whole_value = float("nan")
                if not math.isfinite(whole_value):
                    provenance.append(f"whole_gpu_peak_missing:{pair_id}")
                    failures.add("resource_gpu")
                elif whole_value > max_whole_gpu_peak_gib + 1e-12:
                    failures.add("resource_gpu")
                else:
                    observed_max_whole_gpu_peak_gib = max(
                        observed_max_whole_gpu_peak_gib, whole_value
                    )

    total_decisions = len(actual_sources)
    raw_coverage = raw_high / total_decisions if total_decisions else 0.0
    actual_coverage = applied_counts["vla"] / total_decisions if total_decisions else 0.0
    classic_mrm_share = (applied_counts["expert"] + applied_counts["mrm"]) / total_decisions if total_decisions else 1.0
    if not target:
        failures.add("no_target_runs")
    if raw_coverage + 1e-12 < target_world_vla_preference:
        failures.add("world_vla_preference")
    if actual_coverage + 1e-12 < target_actual_vla_coverage:
        failures.add("actual_vla_coverage")
    if classic_mrm_share > max_classic_mrm_share + 1e-12:
        failures.add("classic_mrm_share")

    comparable = [
        (run, baseline[str(run.get("pair_id"))])
        for run in target
        if str(run.get("pair_id")) in baseline
    ]
    if len(comparable) != len(target):
        failures.add("missing_paired_baseline")
    target_only_unsafe: list[str] = []
    target_unsafe = 0
    baseline_unsafe = 0
    deltas: list[float] = []
    unsafe_groups: dict[str, dict[str, int]] = {}
    for run, base in comparable:
        pair_id = str(run.get("pair_id"))
        if str(base.get("config_sha256") or "") != str(run.get("config_sha256") or ""):
            provenance.append(f"baseline_config_mismatch:{pair_id}")
        if expected_run_lock_hash is not None and str(base.get("run_lock_sha256") or "") != expected_run_lock_hash:
            provenance.append(f"baseline_run_lock:{pair_id}")
        if not bool(base.get("ok")):
            provenance.append(f"baseline_run_not_ok:{pair_id}")
        if expected_router_calibration is not None:
            observed_base_router = base.get("router_calibration")
            if not isinstance(observed_base_router, Mapping) or dict(observed_base_router) != dict(expected_router_calibration):
                provenance.append(f"baseline_router_calibration:{pair_id}")
        if require_spectator_follow and (
            int(base.get("spectator_follow_updates", 0)) <= 0
            or base.get("spectator_follow_error") not in (None, "")
        ):
            provenance.append(f"baseline_spectator_follow:{pair_id}")
        if not str(run.get("manifest_kind", "")).startswith("h6_vla75") or not str(base.get("manifest_kind", "")).startswith("h6_vla75"):
            provenance.append(f"nonformal_manifest_kind:{pair_id}")
        if expected_manifest_kind is not None and (
            run.get("manifest_kind") != expected_manifest_kind
            or base.get("manifest_kind") != expected_manifest_kind
        ):
            provenance.append(f"manifest_lineage_pair:{pair_id}")
        if lineage_id is not None and base.get("schema_version") != "safedrive.h6.vla75.run.v2":
            provenance.append(f"baseline_run_schema:{pair_id}")
        # Materialize the baseline decision list before the strict formal
        # provenance audit below.  Keeping this assignment ahead of the
        # lineage-only checks prevents a stale value from a previous pair (or
        # an unbound local on the first pair) from being used when validating
        # baseline hashes and seed isolation.
        base_decisions = list(base.get("decisions", ()))
        if lineage_id is not None:
            # The Classic-only arm does not score World online, but it still
            # carries the same candidate/code/worktree identity so the paired
            # comparison can be reproduced without an implicit untracked
            # baseline.
            for base_tick, base_decision in enumerate(base_decisions):
                if not isinstance(base_decision, Mapping):
                    provenance.append(f"baseline_malformed_decision:{pair_id}:{base_tick}")
                    continue
                for field in required_decision_fields:
                    if field not in base_decision:
                        provenance.append(
                            f"baseline_decision_field_missing:{pair_id}:{base_tick}:{field}"
                        )
                base_raw_reasons = base_decision.get("raw_gate_reasons")
                if not isinstance(base_raw_reasons, Mapping):
                    provenance.append(f"baseline_raw_gate_reasons_missing:{pair_id}:{base_tick}")
                else:
                    for reason_name in raw_reason_counts:
                        if reason_name not in base_raw_reasons:
                            provenance.append(
                                f"baseline_raw_gate_reason_missing:{pair_id}:{base_tick}:{reason_name}"
                            )
                base_hashes = base_decision.get("candidate_hashes") or base_decision.get("candidate_sha256")
                if not isinstance(base_hashes, Mapping) or len(base_hashes) < 2:
                    provenance.append(f"baseline_candidate_hashes:{pair_id}:{base_tick}")
                base_manifest = base_decision.get("candidates")
                if not isinstance(base_manifest, Mapping):
                    provenance.append(f"baseline_candidate_manifest:{pair_id}:{base_tick}")
                else:
                    base_sources = {
                        _v2_source(
                            value.get("source")
                            or (
                                value.get("provenance", {}).get("source")
                                if isinstance(value.get("provenance"), Mapping)
                                else None
                            )
                        )
                        for value in base_manifest.values()
                        if isinstance(value, Mapping)
                    }
                    if not {"vla", "expert"}.issubset(base_sources):
                        provenance.append(
                            f"baseline_candidate_manifest_sources:{pair_id}:{base_tick}"
                        )
                    if isinstance(base_hashes, Mapping) and {
                        str(key) for key in base_hashes
                    } != {str(key) for key in base_manifest}:
                        provenance.append(
                            f"baseline_candidate_hash_ids:{pair_id}:{base_tick}"
                        )
                if not base_decision.get("model_hash") and not (base_decision.get("world_score") or {}).get("model_hash"):
                    provenance.append(f"baseline_model_hash:{pair_id}:{base_tick}")
                elif expected_model_hash is not None:
                    base_model = base_decision.get("model_hash") or (base_decision.get("world_score") or {}).get("model_hash")
                    if base_model != expected_model_hash:
                        provenance.append(f"baseline_model_hash_lock:{pair_id}:{base_tick}")
                if not base_decision.get("feature_schema") and not (base_decision.get("world_score") or {}).get("feature_schema"):
                    provenance.append(f"baseline_feature_schema:{pair_id}:{base_tick}")
                elif expected_feature_schema is not None:
                    base_feature = base_decision.get("feature_schema") or (base_decision.get("world_score") or {}).get("feature_schema")
                    if base_feature != expected_feature_schema:
                        provenance.append(f"baseline_feature_schema_lock:{pair_id}:{base_tick}")
                base_worktree = base.get("worktree")
                base_worktree = base_worktree if isinstance(base_worktree, Mapping) else {}
                if isinstance(lock_worktree, Mapping) and lock_worktree:
                    for identity_key in (
                        "head",
                        "branch",
                        "full_worktree_hash",
                        "untracked_manifest_sha256",
                    ):
                        expected_identity = lock_worktree.get(identity_key)
                        if expected_identity is None:
                            continue
                        if base_worktree.get(identity_key) != expected_identity:
                            provenance.append(
                                f"baseline_worktree_identity_lock:{pair_id}:{base_tick}:{identity_key}"
                            )
                if not base_decision.get("worktree_hash") and not base.get("worktree_hash") and not (
                    base_worktree.get("full_worktree_hash") or base_worktree.get("worktree_diff_sha256")
                ):
                    provenance.append(f"baseline_worktree_hash:{pair_id}:{base_tick}")
                elif expected_worktree_hash is not None:
                    base_worktree_value = (
                        base_decision.get("worktree_hash")
                        or base.get("worktree_hash")
                        or base_worktree.get("full_worktree_hash")
                        or base_worktree.get("worktree_diff_sha256")
                    )
                    if base_worktree_value != expected_worktree_hash:
                        provenance.append(f"baseline_worktree_hash_lock:{pair_id}:{base_tick}")
        physical_hash = run.get("physical_sha256")
        if physical_hash != base.get("physical_sha256") or not physical_hash:
            provenance.append(f"physical_pair_mismatch:{pair_id}")
        elif lineage_id is not None and (
            not isinstance(physical_hash, str)
            or len(physical_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in physical_hash)
        ):
            provenance.append(f"physical_hash:{pair_id}")
        reset = dict(run.get("reset_comparison") or {})
        if not bool(reset.get("comparable")):
            provenance.append(f"reset_not_comparable:{pair_id}")
        if len(base_decisions) != int(base.get("ticks_executed", 0)):
            provenance.append(f"baseline_decision_tick_mismatch:{pair_id}")
        if lineage_id is not None or run_lock is not None:
            # The whole-card budget applies to both paired arms.  World-only
            # incremental allocation is checked on the target arm below;
            # Classic-only baseline has no World delta but still consumes the
            # same GPU and therefore must report a finite peak.
            whole = base.get("whole_gpu_peak_gb")
            if max_whole_gpu_peak_gib is not None and whole is None:
                provenance.append(f"baseline_whole_gpu_peak_missing:{pair_id}")
                failures.add("resource_gpu")
            if max_whole_gpu_peak_gib is not None:
                try:
                    whole_value = float(whole)
                except (TypeError, ValueError):
                    whole_value = float("nan")
                if not math.isfinite(whole_value):
                    provenance.append(f"baseline_whole_gpu_peak_missing:{pair_id}")
                    failures.add("resource_gpu")
                elif whole_value > max_whole_gpu_peak_gib + 1e-12:
                    failures.add("resource_gpu")
                else:
                    observed_max_whole_gpu_peak_gib = max(
                        observed_max_whole_gpu_peak_gib, whole_value
                    )
        if int(base.get("vla_executed_ticks", 0)) > 0 or any(
            _v2_source(
                item.get("applied_source")
                or item.get("applied_candidate_source")
                or item.get("executed_source")
                or _v2_applied_source(item)[0]
            ) == "vla"
            for item in base_decisions
            if isinstance(item, Mapping)
        ):
            provenance.append(f"baseline_not_classic_only:{pair_id}")
        t_unsafe = _unsafe(run)
        b_unsafe = _unsafe(base)
        scenario = dict(run.get("scenario") or {})
        unsafe_group = "|".join(
            str(scenario.get(key, "unknown"))
            for key in ("map_name", "family", "weather")
        )
        group_slot = unsafe_groups.setdefault(
            unsafe_group,
            {"target_unsafe_pairs": 0, "baseline_unsafe_pairs": 0, "target_only_unsafe_pairs": 0},
        )
        group_slot["target_unsafe_pairs"] += int(t_unsafe)
        group_slot["baseline_unsafe_pairs"] += int(b_unsafe)
        target_unsafe += int(t_unsafe)
        baseline_unsafe += int(b_unsafe)
        if t_unsafe and not b_unsafe:
            target_only_unsafe.append(pair_id)
            group_slot["target_only_unsafe_pairs"] += 1
        try:
            deltas.append(float(run["route_progress_m"]) - float(base["route_progress_m"]))
        except (KeyError, TypeError, ValueError):
            provenance.append(f"progress_missing:{pair_id}")
    paired_n = len(comparable)
    target_unsafe_rate = target_unsafe / max(1, paired_n)
    baseline_unsafe_rate = baseline_unsafe / max(1, paired_n)
    unsafe_delta = target_unsafe_rate - baseline_unsafe_rate
    if unsafe_delta > max_unsafe_delta + 1e-12:
        failures.add("unsafe_delta")
    if forbid_target_only_unsafe and target_only_unsafe:
        failures.add("target_only_unsafe")
    progress = _progress_ci(deltas)
    if not math.isfinite(progress["lower_95"]) or progress["lower_95"] < 0.0:
        failures.add("progress_lower_95")

    scorer_p99 = _p99(scorer_latencies)
    deadline_misses += sum(int(run.get("scorer_deadline_misses", 0)) for run in target)
    if scorer_p99 > scorer_deadline_ms or deadline_misses:
        failures.add("scorer_resource")
    switch_rate = total_switches / max(total_sim_s, 30.0) * 30.0 if total_sim_s > 0.0 else float("inf")
    if switch_rate > float(max_switches_per_30s) + 1e-12:
        failures.add("switch_rate")
    if ping_pong_runs:
        failures.add("ping_pong")
    if provenance:
        failures.add("provenance")
    observed_worktree = None
    if target and isinstance(target[0].get("worktree"), Mapping):
        observed_worktree = dict(target[0]["worktree"])
    elif isinstance(lock_worktree, Mapping):
        observed_worktree = dict(lock_worktree)
    identity = {
        "head": None if observed_worktree is None else observed_worktree.get("head"),
        "branch": None if observed_worktree is None else observed_worktree.get("branch"),
        "full_worktree_hash": (
            None
            if observed_worktree is None
            else observed_worktree.get("full_worktree_hash")
        ),
        "untracked_manifest_sha256": (
            None
            if observed_worktree is None
            else observed_worktree.get("untracked_manifest_sha256")
        ),
        "run_lock_sha256": expected_run_lock_hash,
        "scoped_runtime_sha256": (
            None
            if not isinstance(run_lock, Mapping)
            else run_lock.get("scoped_runtime_sha256")
        ),
    }
    return {
        "schema_version": "safedrive.vla75.acceptance.v2",
        "passed": not failures,
        "failures": sorted(failures),
        "contract": {
            "target_world_vla_preference": target_world_vla_preference,
            "target_actual_vla_coverage": target_actual_vla_coverage,
            "max_classic_mrm_share": max_classic_mrm_share,
            "max_unsafe_delta": max_unsafe_delta,
            "forbid_target_only_unsafe": forbid_target_only_unsafe,
            "scorer_deadline_ms": scorer_deadline_ms,
            "max_switches_per_30s": max_switches_per_30s,
            "ping_pong_window_ticks": ping_pong_window_ticks,
            "max_vla_risk": max_vla_risk,
            "lineage_id": lineage_id,
            "config_sha256": expected_hash,
            "run_lock_sha256": expected_run_lock_hash,
        },
        "coverage": {
            "raw_world_vla_preference": raw_coverage,
            "actual_applied_vla": actual_coverage,
            "classic_mrm_share": classic_mrm_share,
            "target_world_vla_preference": target_world_vla_preference,
            "target_actual_vla_coverage": target_actual_vla_coverage,
            "max_classic_mrm_share": max_classic_mrm_share,
            "raw_world_high_ticks": raw_high,
            "raw_world_pair_scored_ticks": raw_pair_scored,
            "all_decision_ticks": total_decisions,
            "applied_vla_ticks": applied_counts["vla"],
            "applied_expert_ticks": applied_counts["expert"],
            "applied_mrm_ticks": applied_counts["mrm"],
        },
        "matrix": matrix_report,
        "raw_gate_reasons": {
            "failed_tick_counts": raw_reason_counts,
            "by_group": raw_group,
        },
        "transitions": {
            "selected_to_safety_to_applied": transition_matrix,
            "selected_to_safety_to_applied_by_group": transition_groups,
            "invalid_applied_bindings": invalid_applied,
            "guard_verdicts": guard_verdict_counts,
            "world_defer_ticks": world_defer_count,
            "safety_fallback_ticks": safety_fallback_count,
            "repair_methods": repair_method_counts,
        },
        "safety": {
            "target_unsafe_rate": target_unsafe_rate,
            "baseline_unsafe_rate": baseline_unsafe_rate,
            "unsafe_delta": unsafe_delta,
            "max_unsafe_delta": max_unsafe_delta,
            "target_only_unsafe_pairs": sorted(target_only_unsafe),
            "paired_runs": paired_n,
            "absolute_target_unsafe_pairs": target_unsafe,
            "absolute_baseline_unsafe_pairs": baseline_unsafe,
            "by_map_family_weather": unsafe_groups,
        },
        "progress": progress,
        "resource": {
            "scorer_p99_ms": scorer_p99,
            "deadline_ms": scorer_deadline_ms,
            "deadline_misses": deadline_misses,
            "observed_max_world_incremental_gpu_gib": observed_max_world_incremental_gib,
            "observed_max_whole_gpu_peak_gib": observed_max_whole_gpu_peak_gib,
            "max_world_incremental_gpu_gib": max_world_incremental_gib,
            "max_whole_gpu_peak_gib": max_whole_gpu_peak_gib,
            "max_vla_risk": max_vla_risk,
        },
        "stability": {
            "actual_switches": total_switches,
            "sim_seconds": total_sim_s,
            "switches_per_30s": switch_rate,
            "max_switches_per_30s": max_switches_per_30s,
            "max_window_switches": max_window_switches,
            "ping_pong_runs": sorted(set(ping_pong_runs)),
        },
        "identity": identity,
        "provenance_failures": sorted(set(provenance)),
    }


def evaluate_vla90_gate(
    runs: Sequence[Mapping[str, Any]],
    *,
    target_arm: str = "on",
    baseline_arm: str = "off",
    target_vla_coverage: float = 0.90,
    max_unsafe_delta: float = 0.01,
    scorer_deadline_ms: float = 50.0,
) -> dict[str, Any]:
    rows = [_dict(run) for run in runs]
    target = [run for run in rows if run.get("arm") == target_arm]
    baseline = {run["pair_id"]: run for run in rows if run.get("arm") == baseline_arm}
    failures: list[str] = []
    if not target:
        failures.append("no_target_runs")

    actual_sources: list[str] = []
    world_vla_high_score_ticks = 0
    world_pair_scored_ticks = 0
    scorer_latencies: list[float] = []
    total_switches = 0
    total_sim_s = 0.0
    provenance_failures: list[str] = []
    ping_pong_runs: list[str] = []
    for run in target:
        pair_id = str(run.get("pair_id"))
        if run.get("config_sha256") != H6_VLA90_CONFIG_SHA256:
            provenance_failures.append(f"nonformal_config:{pair_id}")
        decisions = list(run.get("decisions", ()))
        if not bool(run.get("ok")):
            provenance_failures.append(f"run_not_ok:{pair_id}")
        if len(decisions) != int(run.get("ticks_executed", 0)):
            provenance_failures.append(f"decision_tick_mismatch:{pair_id}")
        run_sources = []
        for decision in decisions:
            source = decision.get("executed_source")
            if source not in {"vla", "expert", "mrm"}:
                provenance_failures.append(f"missing_executed_source:{pair_id}:{decision.get('tick')}")
                source = "mrm"
            run_sources.append(str(source))
            actual_sources.append(str(source))
            latency = decision.get("scorer_latency_ms")
            if latency is not None:
                scorer_latencies.append(float(latency))

            routing = dict(decision.get("routing") or {})
            eligible = set(routing.get("pass_candidate_ids", ()))
            preference = tuple(routing.get("preference_order", ()))
            selected = routing.get("selected_candidate_id")
            world_score = dict(decision.get("world_score") or {})
            predictions = list(world_score.get("predictions") or ())
            vla_prediction = next(
                (
                    item
                    for item in predictions
                    if str(item.get("candidate_key", "")).endswith(":vla")
                ),
                None,
            )
            expert_prediction = next(
                (
                    item
                    for item in predictions
                    if str(item.get("candidate_key", "")).endswith(":expert")
                ),
                None,
            )
            if vla_prediction is None or expert_prediction is None:
                provenance_failures.append(
                    f"missing_world_pair_score:{pair_id}:{decision.get('tick')}"
                )
            else:
                world_pair_scored_ticks += 1
                try:
                    vla_score = float(vla_prediction["deployment_score"])
                    expert_score = float(expert_prediction["deployment_score"])
                    vla_trust = float(vla_prediction["trust_probability"])
                    vla_risk = float(vla_prediction["unsafe_probability"])
                    trust_threshold = float(world_score["trust_threshold"])
                    risk_ceiling = float(world_score["risk_ceiling"])
                except (KeyError, TypeError, ValueError):
                    provenance_failures.append(
                        f"malformed_world_pair_score:{pair_id}:{decision.get('tick')}"
                    )
                else:
                    values = (
                        vla_score,
                        expert_score,
                        vla_trust,
                        vla_risk,
                        trust_threshold,
                        risk_ceiling,
                    )
                    if not all(math.isfinite(value) for value in values):
                        provenance_failures.append(
                            f"nonfinite_world_pair_score:{pair_id}:{decision.get('tick')}"
                        )
                    elif (
                        vla_score + 1e-12 >= expert_score
                        and vla_trust + 1e-12 >= trust_threshold
                        and vla_risk <= risk_ceiling + 1e-12
                    ):
                        world_vla_high_score_ticks += 1
            if preference and selected != preference[0]:
                provenance_failures.append(f"preference_head_mismatch:{pair_id}:{decision.get('tick')}")
            for candidate_id, guard in dict(decision.get("guard") or {}).items():
                if guard.get("verdict") == "REJECT" and (
                    candidate_id in eligible or candidate_id in preference
                ):
                    provenance_failures.append(f"guard_reject_reached_world:{pair_id}:{decision.get('tick')}")

            repair = decision.get("repair")
            if repair and repair.get("success"):
                margins = repair.get("margins", ())
                if any(bool(item.get("hard")) and float(item.get("margin", -1.0)) < 0.0 for item in margins):
                    provenance_failures.append(f"repair_not_revalidated:{pair_id}:{decision.get('tick')}")
            if selected and str(selected).endswith(":vla") and source == "expert":
                arbitration = dict(decision.get("arbitration") or {})
                notes = set(arbitration.get("notes", ()))
                audits = {item.get("candidate_id"): item for item in arbitration.get("audits", ())}
                vla_audit = audits.get(selected, {})
                if (
                    "preferred_vla_repair_failed_try_expert" not in notes
                    and vla_audit.get("final_ok") is not False
                ):
                    provenance_failures.append(f"expert_fallback_without_vla_failure:{pair_id}:{decision.get('tick')}")
        if _ping_pong(run_sources):
            ping_pong_runs.append(pair_id)
        total_switches += int(run.get("actual_switch_count", 0))
        total_sim_s += 0.05 * len(decisions)

    total_decisions = len(actual_sources)
    vla_executed = actual_sources.count("vla")
    expert_executed = actual_sources.count("expert")
    mrm = actual_sources.count("mrm")
    coverage = vla_executed / total_decisions if total_decisions else 0.0
    # Denominator is every decision tick, not only ticks where both candidates
    # happened to survive.  Missing an Expert/VLA pair therefore cannot inflate
    # the claim that World truly rated VLA higher in 90% of situations.
    world_vla_coverage = (
        world_vla_high_score_ticks / total_decisions if total_decisions else 0.0
    )
    if world_vla_coverage + 1e-12 < target_vla_coverage:
        failures.append("world_vla_preference")
    if coverage + 1e-12 < target_vla_coverage:
        failures.append("actual_vla_coverage")

    comparable = [(run, baseline[run["pair_id"]]) for run in target if run.get("pair_id") in baseline]
    if len(comparable) != len(target):
        failures.append("missing_paired_baseline")
    for run, base in comparable:
        pair_id = str(run.get("pair_id"))
        if base.get("config_sha256") != H6_VLA90_CONFIG_SHA256:
            provenance_failures.append(f"nonformal_baseline_config:{base.get('pair_id')}")
        if not bool(base.get("ok")):
            provenance_failures.append(f"baseline_run_not_ok:{pair_id}")
        if (
            run.get("manifest_kind") != "h6_fresh"
            or base.get("manifest_kind") != "h6_fresh"
        ):
            provenance_failures.append(f"nonformal_manifest_kind:{pair_id}")
        if (
            not run.get("physical_sha256")
            or run.get("physical_sha256") != base.get("physical_sha256")
        ):
            provenance_failures.append(f"physical_pair_mismatch:{pair_id}")
        reset = dict(run.get("reset_comparison") or {})
        if not bool(reset.get("comparable")):
            provenance_failures.append(f"reset_not_comparable:{pair_id}")
        baseline_decisions = list(base.get("decisions", ()))
        if len(baseline_decisions) != int(base.get("ticks_executed", 0)):
            provenance_failures.append(f"baseline_decision_tick_mismatch:{pair_id}")
        if int(base.get("vla_executed_ticks", 0)) != 0 or any(
            decision.get("executed_source") == "vla" for decision in baseline_decisions
        ):
            provenance_failures.append(f"baseline_not_classic_only:{pair_id}")
    target_unsafe_rate = sum(int(_unsafe(run)) for run, _ in comparable) / max(1, len(comparable))
    baseline_unsafe_rate = sum(int(_unsafe(base)) for _, base in comparable) / max(1, len(comparable))
    unsafe_delta = target_unsafe_rate - baseline_unsafe_rate
    if unsafe_delta > max_unsafe_delta + 1e-12:
        failures.append("unsafe_delta")

    progress = _progress_ci(
        [float(run["route_progress_m"]) - float(base["route_progress_m"]) for run, base in comparable]
    )
    if not math.isfinite(progress["lower_95"]) or progress["lower_95"] < 0.0:
        failures.append("progress_lower_95")

    scorer_p99 = _p99(scorer_latencies)
    deadline_misses = sum(int(run.get("scorer_deadline_misses", 0)) for run in target)
    if scorer_p99 > scorer_deadline_ms or deadline_misses:
        failures.append("scorer_resource")

    switch_rate = total_switches / total_sim_s if total_sim_s > 0.0 else float("inf")
    if switch_rate > 2.0 / 30.0 + 1e-12:
        failures.append("switch_rate")
    if ping_pong_runs:
        failures.append("ping_pong")
    if provenance_failures:
        failures.append("provenance")

    failures = sorted(set(failures))
    return {
        "schema_version": "safedrive.vla90.acceptance.v1",
        "passed": not failures,
        "failures": failures,
        "coverage": {
            "actual_vla": coverage,
            "world_vla_preference": world_vla_coverage,
            "target": target_vla_coverage,
            "vla_ticks": vla_executed,
            "expert_ticks": expert_executed,
            "mrm_ticks": mrm,
            "all_decision_ticks": total_decisions,
            "world_pair_scored_ticks": world_pair_scored_ticks,
            "world_vla_high_score_ticks": world_vla_high_score_ticks,
        },
        "safety": {
            "target_unsafe_rate": target_unsafe_rate,
            "baseline_unsafe_rate": baseline_unsafe_rate,
            "unsafe_delta": unsafe_delta,
            "max_unsafe_delta": max_unsafe_delta,
            "paired_runs": len(comparable),
        },
        "progress": progress,
        "resource": {
            "scorer_p99_ms": scorer_p99,
            "deadline_ms": scorer_deadline_ms,
            "deadline_misses": deadline_misses,
        },
        "stability": {
            "actual_switches": total_switches,
            "sim_seconds": total_sim_s,
            "switches_per_second": switch_rate,
            "ping_pong_runs": sorted(set(ping_pong_runs)),
        },
        "provenance_failures": sorted(set(provenance_failures)),
    }


__all__ = ["evaluate_vla90_gate", "evaluate_vla75_gate"]
