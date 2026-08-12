"""Pair comparability gate and failure classification (R2 §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from driving_vla.evaluation.paired_contract import (
    EXPECTED_PRIMARY_TICKS,
    K2AnchorArtifactV1,
    MeasuredInitialState,
)


# Hard tolerances (R2 §6.4)
POS_TOL_M = 0.02
YAW_TOL_DEG = 0.2
VEL_TOL_MPS = 0.05
MIN_MPC_SOLVED = 48
EXPECTED_TICKS = EXPECTED_PRIMARY_TICKS

FAILURE_CODES = (
    "INITIAL_STATE_MISMATCH",
    "ANCHOR_BUNDLE_MISMATCH",
    "SPAWN_FAILED",
    "SCRIPT_PHASE_MISMATCH",
    "SENSOR_SYNC_FAILURE",
    "TICK_OWNER_CONFLICT",
    "SERVER_CRASH_OR_HANG",
    "MPC_DEADLINE_UNRELIABLE",
    "EXECUTION_BINDING_FAILURE",
    "GUARD_REJECT",
    "CLEANUP_FAILURE",
    "REGISTRY_HASH_MISMATCH",
    "INCOMPLETE_PRIMARY_HORIZON",
    "CONFIG_HASH_MISMATCH",
)

STATUS_COMPARABLE = "COMPARABLE"
STATUS_INCOMPARABLE = "INCOMPARABLE"


@dataclass(frozen=True)
class BranchExecutionReport:
    """Minimal branch-side facts needed for comparability (no oracle metrics)."""

    candidate_index: int
    candidate_id: str
    anchor_artifact_hash: str
    measured_initial_state: MeasuredInitialState
    registry_hash: str
    scenario_id: str
    seed_id: str
    model_retimer_hash: str
    executor_config_hash: str
    completed_primary_ticks: int
    mpc_solved_ticks: int
    mpc_timeout_ticks: int
    mpc_fallback_ticks: int
    selected_ids_consistent: bool
    executed_ids_consistent: bool
    source_ids_consistent: bool
    sensor_sync_ok: bool = True
    tick_owner_ok: bool = True
    cleanup_ok: bool = True
    spawn_ok: bool = True
    server_ok: bool = True
    failure_codes: tuple[str, ...] = ()
    # Per-actor script phase at branch start (name -> phase)
    script_phase: Mapping[str, str] = field(default_factory=dict)
    weather: Mapping[str, Any] = field(default_factory=dict)
    traffic_light_state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparabilityResult:
    status: str
    reasons: tuple[str, ...]
    failure_codes: tuple[str, ...]
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def comparable(self) -> bool:
        return self.status == STATUS_COMPARABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "comparable": self.comparable,
            "reasons": list(self.reasons),
            "failure_codes": list(self.failure_codes),
            "details": dict(self.details),
        }


def _angle_diff_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def _pose_deltas(a: MeasuredInitialState, b: MeasuredInitialState) -> dict[str, Any]:
    actors_a = {x.name: x for x in a.actors}
    actors_b = {x.name: x for x in b.actors}
    names = sorted(set(actors_a) | set(actors_b))
    deltas: dict[str, Any] = {}
    for name in names:
        if name not in actors_a or name not in actors_b:
            deltas[name] = {"missing": True}
            continue
        pa, pb = actors_a[name], actors_b[name]
        dx = pa.transform.x - pb.transform.x
        dy = pa.transform.y - pb.transform.y
        dz = pa.transform.z - pb.transform.z
        pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_err = _angle_diff_deg(pa.transform.yaw_deg, pb.transform.yaw_deg)
        # linear speed / velocity components
        dvx = pa.velocity.vx - pb.velocity.vx
        dvy = pa.velocity.vy - pb.velocity.vy
        dvz = pa.velocity.vz - pb.velocity.vz
        vel_err = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        deltas[name] = {
            "position_err_m": pos_err,
            "yaw_err_deg": yaw_err,
            "velocity_err_mps": vel_err,
            "script_phase_a": pa.script_phase,
            "script_phase_b": pb.script_phase,
        }
    return deltas


def evaluate_pair_comparability(
    *,
    anchor: K2AnchorArtifactV1,
    branch0: BranchExecutionReport,
    branch1: BranchExecutionReport,
    expected_registry_hash: str,
    expected_scenario_id: str,
    expected_seed_id: str,
    expected_model_retimer_hash: str,
    expected_executor_config_hash: str,
) -> ComparabilityResult:
    """Apply hard comparability gate (R2 §6.4)."""
    reasons: list[str] = []
    codes: list[str] = []
    details: dict[str, Any] = {}

    # 1. registry/scenario/seed/config/model/executor hash
    if branch0.registry_hash != expected_registry_hash or branch1.registry_hash != expected_registry_hash:
        reasons.append("registry_hash_mismatch")
        codes.append("REGISTRY_HASH_MISMATCH")
    if (
        branch0.scenario_id != expected_scenario_id
        or branch1.scenario_id != expected_scenario_id
        or branch0.seed_id != expected_seed_id
        or branch1.seed_id != expected_seed_id
        or anchor.scenario_id != expected_scenario_id
        or anchor.seed_id != expected_seed_id
    ):
        reasons.append("scenario_seed_identity_mismatch")
        codes.append("CONFIG_HASH_MISMATCH")
    if (
        branch0.model_retimer_hash != expected_model_retimer_hash
        or branch1.model_retimer_hash != expected_model_retimer_hash
    ):
        reasons.append("model_retimer_hash_mismatch")
        codes.append("CONFIG_HASH_MISMATCH")
    if (
        branch0.executor_config_hash != expected_executor_config_hash
        or branch1.executor_config_hash != expected_executor_config_hash
    ):
        reasons.append("executor_config_hash_mismatch")
        codes.append("CONFIG_HASH_MISMATCH")

    # 2. same anchor artifact hash on both branches
    ah = anchor.artifact_content_hash()
    if branch0.anchor_artifact_hash != ah or branch1.anchor_artifact_hash != ah:
        reasons.append("anchor_artifact_hash_mismatch")
        codes.append("ANCHOR_BUNDLE_MISMATCH")
    if branch0.anchor_artifact_hash != branch1.anchor_artifact_hash:
        reasons.append("branch_anchor_hashes_differ")
        codes.append("ANCHOR_BUNDLE_MISMATCH")

    # 3. Guard OK on anchor
    if anchor.guard_status != "OK":
        reasons.append(f"guard_not_ok:{anchor.guard_status}")
        codes.append("GUARD_REJECT")

    # 4–6. measured state tolerances + phases
    deltas = _pose_deltas(branch0.measured_initial_state, branch1.measured_initial_state)
    details["pose_deltas"] = deltas
    for name, d in deltas.items():
        if d.get("missing"):
            reasons.append(f"actor_missing:{name}")
            codes.append("INITIAL_STATE_MISMATCH")
            continue
        if float(d["position_err_m"]) > POS_TOL_M:
            reasons.append(f"position_tol:{name}:{d['position_err_m']:.4f}")
            codes.append("INITIAL_STATE_MISMATCH")
        if float(d["yaw_err_deg"]) > YAW_TOL_DEG:
            reasons.append(f"yaw_tol:{name}:{d['yaw_err_deg']:.4f}")
            codes.append("INITIAL_STATE_MISMATCH")
        if float(d["velocity_err_mps"]) > VEL_TOL_MPS:
            reasons.append(f"velocity_tol:{name}:{d['velocity_err_mps']:.4f}")
            codes.append("INITIAL_STATE_MISMATCH")
        if str(d.get("script_phase_a")) != str(d.get("script_phase_b")):
            reasons.append(f"script_phase:{name}")
            codes.append("SCRIPT_PHASE_MISMATCH")

    # traffic light / weather phase
    if dict(branch0.traffic_light_state) != dict(branch1.traffic_light_state):
        reasons.append("traffic_light_state_mismatch")
        codes.append("SCRIPT_PHASE_MISMATCH")
    # weather: exact registry weather should match; allow only if both empty
    if dict(branch0.weather) != dict(branch1.weather):
        reasons.append("weather_mismatch")
        codes.append("INITIAL_STATE_MISMATCH")

    # Absolute CARLA frame IDs always differ across cold rebuilds in one server
    # session. R2 §6.4 "frame phase" means script/light/phase alignment (checked
    # above), not equal absolute frame numbers. Record for audit only.
    details["simulation_frames"] = {
        "branch0": int(branch0.measured_initial_state.simulation_frame),
        "branch1": int(branch1.measured_initial_state.simulation_frame),
    }

    # 7. 50 primary ticks
    for label, br in (("branch0", branch0), ("branch1", branch1)):
        if br.completed_primary_ticks < EXPECTED_TICKS:
            reasons.append(f"incomplete_horizon:{label}:{br.completed_primary_ticks}")
            codes.append("INCOMPLETE_PRIMARY_HORIZON")

    # 8. sensor/tick/cleanup/server
    for label, br in (("branch0", branch0), ("branch1", branch1)):
        if not br.spawn_ok:
            reasons.append(f"spawn_failed:{label}")
            codes.append("SPAWN_FAILED")
        if not br.sensor_sync_ok:
            reasons.append(f"sensor_sync:{label}")
            codes.append("SENSOR_SYNC_FAILURE")
        if not br.tick_owner_ok:
            reasons.append(f"tick_owner:{label}")
            codes.append("TICK_OWNER_CONFLICT")
        if not br.cleanup_ok:
            reasons.append(f"cleanup:{label}")
            codes.append("CLEANUP_FAILURE")
        if not br.server_ok:
            reasons.append(f"server:{label}")
            codes.append("SERVER_CRASH_OR_HANG")
        for c in br.failure_codes:
            if c not in codes:
                codes.append(str(c))
            reasons.append(f"branch_failure:{label}:{c}")

    # 9. candidate binding
    for label, br, idx in (
        ("branch0", branch0, 0),
        ("branch1", branch1, 1),
    ):
        if br.candidate_index != idx:
            reasons.append(f"candidate_index:{label}:{br.candidate_index}")
            codes.append("EXECUTION_BINDING_FAILURE")
        if not br.selected_ids_consistent or not br.executed_ids_consistent or not br.source_ids_consistent:
            reasons.append(f"id_binding:{label}")
            codes.append("EXECUTION_BINDING_FAILURE")

    # 10. MPC solved >= 48/50
    for label, br in (("branch0", branch0), ("branch1", branch1)):
        if br.mpc_solved_ticks < MIN_MPC_SOLVED:
            reasons.append(
                f"mpc_unreliable:{label}:solved={br.mpc_solved_ticks}/"
                f"timeout={br.mpc_timeout_ticks}/fallback={br.mpc_fallback_ticks}"
            )
            codes.append("MPC_DEADLINE_UNRELIABLE")
        # Remaining ticks must be explicit timeout/fallback
        remaining = br.completed_primary_ticks - br.mpc_solved_ticks
        explicit = br.mpc_timeout_ticks + br.mpc_fallback_ticks
        if remaining > 0 and explicit < remaining:
            reasons.append(f"mpc_unaccounted:{label}:remaining={remaining},explicit={explicit}")
            codes.append("MPC_DEADLINE_UNRELIABLE")

    # de-dupe codes preserving order
    seen: set[str] = set()
    uniq_codes: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            uniq_codes.append(c)

    if reasons:
        return ComparabilityResult(
            status=STATUS_INCOMPARABLE,
            reasons=tuple(reasons),
            failure_codes=tuple(uniq_codes),
            details=details,
        )
    return ComparabilityResult(
        status=STATUS_COMPARABLE,
        reasons=(),
        failure_codes=(),
        details=details,
    )


def classify_external_failure(code: str) -> str:
    """Map a failure code to a coarse bucket for ledgers."""
    c = str(code)
    if c not in FAILURE_CODES and c not in {
        "REGISTRY_HASH_MISMATCH",
        "INCOMPLETE_PRIMARY_HORIZON",
        "CONFIG_HASH_MISMATCH",
    }:
        return "UNKNOWN"
    if c in {
        "SERVER_CRASH_OR_HANG",
        "SENSOR_SYNC_FAILURE",
        "TICK_OWNER_CONFLICT",
    }:
        return "EXTERNAL_SERVER"
    if c in {
        "SPAWN_FAILED",
        "INITIAL_STATE_MISMATCH",
        "SCRIPT_PHASE_MISMATCH",
        "CLEANUP_FAILURE",
    }:
        return "FIXTURE_OR_RUNNER"
    if c in {"GUARD_REJECT", "ANCHOR_BUNDLE_MISMATCH", "EXECUTION_BINDING_FAILURE"}:
        return "ANCHOR_OR_BINDING"
    if c == "MPC_DEADLINE_UNRELIABLE":
        return "EXECUTOR_DEADLINE"
    return "OTHER"
