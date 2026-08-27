"""H5 closed-loop data contracts.

These records are intentionally outcome-only and contain no online Oracle,
future labels, or Regression inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from data_pipeline.h2.contracts import ScenarioKey
from data_pipeline.h3.contracts import stable_sha256


@dataclass(frozen=True)
class H5Scenario:
    pair_id: str
    scenario: ScenarioKey
    physical_sha256: str
    manifest_kind: str  # "h2" | "challenge"
    arm_order: tuple[str, ...]
    physical: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "pair_id": self.pair_id,
            "scenario": self.scenario.to_dict(),
            "physical_sha256": self.physical_sha256,
            "manifest_kind": self.manifest_kind,
            "arm_order": list(self.arm_order),
        }
        payload["sha256"] = stable_sha256(payload)
        return payload


@dataclass(frozen=True)
class H5DecisionRecord:
    tick: int
    carla_frame: int
    simulation_time_s: float
    anchor_id: str
    candidate_sha256: Mapping[str, str]
    guard: Mapping[str, Any]
    routing: Mapping[str, Any]
    selected_source: str | None
    scorer_latency_ms: float | None
    scorer_disposition: str | None
    scorer_defer_reason: str | None
    safety_decision_kind: str | None
    applied_mode: str | None
    switch_count: int
    defer_count: int
    fallback_count: int
    generation_latency_s: Mapping[str, float]
    tick_wall_ms: float
    generation_attempts: tuple[Mapping[str, Any], ...] = ()
    candidates: Mapping[str, Any] = field(default_factory=dict)
    world_features: Mapping[str, Any] = field(default_factory=dict)
    executed_candidate_id: str | None = None
    executed_source: str | None = None
    repair: Mapping[str, Any] | None = None
    arbitration: Mapping[str, Any] | None = None
    world_score: Mapping[str, Any] | None = None
    red_light_active: bool = False
    # VLA75 provenance is additive; all fields are optional so H5/v1 records
    # deserialize unchanged.  The collector fills them after Guard, World,
    # Safety and the control bind have each produced their authoritative id.
    raw_preferred_candidate_id: str | None = None
    raw_preferred_source: str | None = None
    raw_gate_reasons: Mapping[str, Any] = field(default_factory=dict)
    stabilized_preferred_candidate_id: str | None = None
    stabilized_preferred_source: str | None = None
    selected_candidate_id: str | None = None
    selected_candidate_source: str | None = None
    safety_executed_candidate_id: str | None = None
    safety_executed_source: str | None = None
    applied_candidate_id: str | None = None
    applied_source: str | None = None
    applied_candidate_source: str | None = None
    repair_input_id: str | None = None
    repair_output_id: str | None = None
    repair_method: str | None = None
    repair_success: bool | None = None
    repair_final_validation: bool | None = None
    model_hash: str | None = None
    feature_schema: str | None = None
    worktree_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep the historical H5/v1 JSON shape byte-compatible when a caller
        # constructs a record without the additive VLA75 audit fields.  The
        # v75 collector writes non-default values (or the explicit empty
        # audit map where required) and therefore still serializes the full
        # provenance contract.  This avoids making old Evidence appear to
        # have been produced by the new contract merely because it was
        # round-tripped through the typed adapter.
        additive_defaults = {
            "generation_attempts": (),
            "candidates": {},
            "world_features": {},
            "executed_candidate_id": None,
            "executed_source": None,
            "repair": None,
            "arbitration": None,
            "world_score": None,
            "red_light_active": False,
            "raw_preferred_candidate_id": None,
            "raw_preferred_source": None,
            "raw_gate_reasons": {},
            "stabilized_preferred_candidate_id": None,
            "stabilized_preferred_source": None,
            "selected_candidate_id": None,
            "selected_candidate_source": None,
            "safety_executed_candidate_id": None,
            "safety_executed_source": None,
            "applied_candidate_id": None,
            "applied_source": None,
            "applied_candidate_source": None,
            "repair_input_id": None,
            "repair_output_id": None,
            "repair_method": None,
            "repair_success": None,
            "repair_final_validation": None,
            "model_hash": None,
            "feature_schema": None,
            "worktree_hash": None,
        }
        for key, default in additive_defaults.items():
            if payload.get(key) == default:
                payload.pop(key, None)
        return payload


@dataclass(frozen=True)
class H5RunRecord:
    schema_version: str
    dataset_id: str
    run_id: str
    pair_id: str
    scenario: ScenarioKey
    physical_sha256: str
    manifest_kind: str
    arm: str
    arm_order_index: int
    reset_signature: Mapping[str, Any]
    reset_comparison: Mapping[str, Any] | None
    route_progress_m: float
    route_completed: bool
    collision_count: int
    red_light_violation: bool
    off_corridor_duration_s: float
    jerk_rms_mps3: float
    acceleration_rms_mps2: float
    lateral_acceleration_rms_mps2: float
    switch_count: int
    defer_count: int
    fallback_count: int
    safety_fallback_count: int
    deadline_misses: int
    scorer_deadline_misses: int
    p50_scorer_ms: float | None
    p95_scorer_ms: float | None
    p99_scorer_ms: float | None
    whole_gpu_peak_gb: float
    vla_forward_count: int
    ticks_executed: int
    cleanup_complete: bool
    ok: bool
    errors: tuple[str, ...] = ()
    decisions: tuple[H5DecisionRecord, ...] = ()
    timeline: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    worktree: Mapping[str, Any] = field(default_factory=dict)
    config_sha256: str = ""
    vla_executed_ticks: int = 0
    expert_executed_ticks: int = 0
    mrm_ticks: int = 0
    actual_switch_count: int = 0
    pre_roll: tuple[Mapping[str, Any], ...] = ()
    initial_route_progress_m: float | None = None
    red_light_stop_progress_m: float | None = None
    # Additive fields used by the H6 VLA75 run contract.  Keeping them at the
    # end with defaults means old H5/v1 JSON remains deserializable while a
    # v2 run can be round-tripped through the typed contract as well as the
    # file-backed H5Store.
    run_lock_sha256: str | None = None
    spectator_follow_updates: int = 0
    spectator_follow_error: str | None = None
    phase_boundaries: Mapping[str, int] = field(default_factory=dict)
    # World-only incremental allocation is distinct from the whole-card peak
    # sampled by the collector.  It is optional for legacy H5/v1 runs and
    # mandatory (and finite) only when the vla75 formal gate audits resources.
    world_incremental_gpu_gib: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario"] = self.scenario.to_dict()
        # Preserve the historical H5/v1 record shape when this typed adapter
        # is used to round-trip an old run.  VLA75 collectors populate the
        # additive fields (or the explicit non-empty phase/provenance values),
        # so those records retain the complete v2 payload.
        additive_defaults = {
            "vla_executed_ticks": 0,
            "expert_executed_ticks": 0,
            "mrm_ticks": 0,
            "actual_switch_count": 0,
            "pre_roll": (),
            "initial_route_progress_m": None,
            "red_light_stop_progress_m": None,
            "run_lock_sha256": None,
            "spectator_follow_updates": 0,
            "spectator_follow_error": None,
            "phase_boundaries": {},
            "world_incremental_gpu_gib": None,
        }
        for key, default in additive_defaults.items():
            if payload.get(key) == default:
                payload.pop(key, None)
        return payload

    @property
    def content_sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("content_sha256", None)
        return stable_sha256(payload)


def run_from_dict(payload: Mapping[str, Any]) -> H5RunRecord:
    values = dict(payload)
    values["scenario"] = ScenarioKey(**values["scenario"])
    values["decisions"] = tuple(H5DecisionRecord(**d) for d in values.get("decisions", ()))
    values["timeline"] = tuple(dict(x) for x in values.get("timeline", ()))
    values["events"] = tuple(dict(x) for x in values.get("events", ()))
    values["pre_roll"] = tuple(dict(x) for x in values.get("pre_roll", ()))
    values["errors"] = tuple(str(x) for x in values.get("errors", ()))
    return H5RunRecord(**values)


__all__ = [
    "H5DecisionRecord",
    "H5RunRecord",
    "H5Scenario",
    "run_from_dict",
]
