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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario"] = self.scenario.to_dict()
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
    values["errors"] = tuple(str(x) for x in values.get("errors", ()))
    return H5RunRecord(**values)


__all__ = [
    "H5DecisionRecord",
    "H5RunRecord",
    "H5Scenario",
    "run_from_dict",
]
