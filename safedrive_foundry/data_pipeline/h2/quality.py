"""Frozen pilot/full-matrix quality gates for H2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .config import H2_CONFIG_SHA256


DECISIVE = "CANDIDATE_WIN"


@dataclass(frozen=True)
class H2GateReport:
    scope: str
    passed: bool
    metrics: Mapping[str, Any]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "passed": self.passed, "metrics": dict(self.metrics), "failures": list(self.failures)}


def _source_by_candidate(record: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["candidate_id"]): str(item["source"]) for item in record.get("candidates", [])}


def audit_h2_gate(
    records: Iterable[Mapping[str, Any]],
    *,
    scope: str,
    manifest_valid: bool = True,
    dataset_bytes: int = 0,
    whole_gpu_peak_gb: float = 0.0,
) -> H2GateReport:
    rows = list(records)
    config_valid = all(row.get("config_sha256") == H2_CONFIG_SHA256 for row in rows)
    valid = [row for row in rows if row.get("terminal_status") == "VALID_PAIR"]
    eligible = [
        row for row in rows
        if len(row.get("candidates", [])) == 2
        and all(item.get("guard", {}).get("verdict") == "PASS" for item in row.get("candidates", []))
        and row.get("anchor", {}).get("selection_space") == "DISTINCT"
        and int(row.get("vla_forward_count", 0)) == 1
        and bool(row.get("anchor", {}).get("swap_invariant", False))
    ]
    labels = [row.get("label") or {} for row in valid]
    decisive = [label for label in labels if label.get("verdict") == DECISIVE]
    source_wins = {"expert": 0, "vla": 0}
    for row in valid:
        label = row.get("label") or {}
        if label.get("verdict") != DECISIVE:
            continue
        source = _source_by_candidate(row).get(str(label.get("winner_candidate_id")))
        if source in source_wins:
            source_wins[source] += 1
    map_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    weather_counts: dict[str, int] = {}
    expert_slots: list[int] = []
    hash_preserved = 0
    branch_total = 0
    execution_violations: list[str] = []
    for row in valid:
        scenario = row.get("scenario", {})
        map_counts[str(scenario.get("map_name"))] = map_counts.get(str(scenario.get("map_name")), 0) + 1
        family_counts[str(scenario.get("family"))] = family_counts.get(str(scenario.get("family")), 0) + 1
        weather_counts[str(scenario.get("weather"))] = weather_counts.get(str(scenario.get("weather")), 0) + 1
        for candidate in row.get("candidates", []):
            if candidate.get("source") == "expert":
                expert_slots.append(int(candidate.get("slot", -1)))
        for branch in row.get("branches", []):
            branch_total += 1
            if branch.get("pre_binding_trajectory_sha256") == branch.get("post_binding_trajectory_sha256"):
                hash_preserved += 1
            candidate_id = branch.get("candidate_id")
            if (
                branch.get("safety_input_id") != candidate_id
                or not branch.get("safety_executed_id")
                or branch.get("applied_id") != branch.get("safety_executed_id")
            ):
                execution_violations.append(f"{row.get('pair_id')}:execution_binding")
            errors = " ".join(str(item).lower() for item in branch.get("errors", ()))
            if any(token in errors for token in ("orphan", "first-available", "cross_source")):
                execution_violations.append(f"{row.get('pair_id')}:fallback")
    for row in rows:
        if int(row.get("vla_forward_count", 0)) > 1:
            execution_violations.append(f"{row.get('pair_id')}:second_forward")
    expert_slot_zero_rate = (
        sum(slot == 0 for slot in expert_slots) / len(expert_slots) if expert_slots else 0.0
    )
    decisive_count = len(decisive)
    source_majority = max(source_wins.values(), default=0) / decisive_count if decisive_count else 0.0
    swap_invariant = all(bool(row.get("anchor", {}).get("swap_invariant", False)) for row in valid) if valid else False
    metrics: dict[str, Any] = {
        "planned": 15 if scope == "pilot" else 120,
        "terminal": len(rows),
        "eligible_distinct": len(eligible),
        "valid_pairs": len(valid),
        "decisive_labels": decisive_count,
        "source_wins": source_wins,
        "map_valid_pairs": map_counts,
        "family_valid_pairs": family_counts,
        "weather_valid_pairs": weather_counts,
        "expert_slot_zero_rate": expert_slot_zero_rate,
        "swap_invariance": swap_invariant,
        "source_only_majority_baseline": source_majority,
        "trajectory_hash_preservation_rate": hash_preserved / branch_total if branch_total else 0.0,
        "manifest_valid": manifest_valid,
        "config_sha256": H2_CONFIG_SHA256,
        "config_valid": config_valid,
        "execution_violations": execution_violations,
        "dataset_bytes": dataset_bytes,
        "whole_gpu_peak_gb": whole_gpu_peak_gb,
    }
    failures: list[str] = []
    planned = int(metrics["planned"])
    if len(rows) != planned:
        failures.append("terminal_count")
    if scope == "pilot":
        if len(eligible) < 8:
            failures.append("pilot_eligible_distinct")
        if len(valid) < 7:
            failures.append("pilot_valid_pairs")
        if execution_violations:
            failures.append("pilot_execution_contract")
    else:
        if len(eligible) < 80:
            failures.append("eligible_distinct")
        if len(valid) < 72:
            failures.append("valid_pairs")
        if any(map_counts.get(name, 0) < 18 for name in ("Town01", "Town03", "Town05")):
            failures.append("per_map_valid_pairs")
        families = ("free_flow", "slow_lead", "stopped_lead", "cut_in", "red_light_hold")
        if any(family_counts.get(name, 0) < 8 for name in families):
            failures.append("per_family_valid_pairs")
        if any(weather_counts.get(name, 0) < 30 for name in ("ClearNoon", "CloudyNoon")):
            failures.append("per_weather_valid_pairs")
        if decisive_count < 24 or decisive_count < 0.25 * len(valid):
            failures.append("decisive_labels")
        for source in ("expert", "vla"):
            if source_wins[source] < 5 or (decisive_count and source_wins[source] < 0.20 * decisive_count):
                failures.append(f"{source}_wins")
        if not 0.40 <= expert_slot_zero_rate <= 0.60:
            failures.append("slot_balance")
        if not swap_invariant:
            failures.append("swap_invariance")
        if source_majority > 0.80 + 1e-12:
            failures.append("source_only_majority")
        if metrics["trajectory_hash_preservation_rate"] != 1.0:
            failures.append("trajectory_hash_preservation")
        if execution_violations:
            failures.append("execution_contract")
        if dataset_bytes > 15 * 1024**3:
            failures.append("dataset_size")
        if whole_gpu_peak_gb > 14.5 + 1e-12:
            failures.append("gpu_peak")
    if not manifest_valid:
        failures.append("manifest_hashes")
    if not config_valid:
        failures.append("config_hashes")
    return H2GateReport(scope, not failures, metrics, tuple(dict.fromkeys(failures)))


__all__ = ["H2GateReport", "audit_h2_gate"]
