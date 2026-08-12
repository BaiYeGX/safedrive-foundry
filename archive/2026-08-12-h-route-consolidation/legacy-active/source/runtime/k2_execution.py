"""R1 K2 selection and PathManager/SpeedPlanner binding (fail-closed).

Also provides Spatial K2 V2 selection (R2-X) without weakening V1 contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Union

from driving_vla.model.k2_builder import (
    GUARD_OK,
    K2ExecutionSpec,
    K2PredictionBundle,
    verify_execution_spatial_binding,
)
from driving_vla.model.k2_spatial_types import (
    GUARD_OK as GUARD_OK_V2,
    K2ExecutionSpecV2,
    K2PredictionBundleV2,
)
from driving_vla.model.k2_v3_types import K2CandidateV3, K2PredictionBundleV3
from driving_vla.runtime.path_manager import EgoPose, PathUpdate, VLAPathManager
from driving_vla.runtime.vla_speed_planner import VLASpeedDecision, VLASpeedPlanner


class K2SelectionError(RuntimeError):
    """Fail-closed selection / binding error (no silent first-candidate fallback)."""


@dataclass(frozen=True)
class K2Selection:
    candidate_id: str
    candidate_index: int
    execution_spec: Union[K2ExecutionSpec, K2ExecutionSpecV2, K2CandidateV3]
    mode: str
    bundle: Union[K2PredictionBundle, K2PredictionBundleV2, K2PredictionBundleV3]


@dataclass(frozen=True)
class K2ApplyResult:
    selected_candidate_id: str
    executed_candidate_id: str
    source_id: str
    speed_decision: VLASpeedDecision
    path_update: PathUpdate
    selection: K2Selection


def select_k2(
    bundle: K2PredictionBundle,
    *,
    mode: Literal["top1", "force"] = "top1",
    force_index: int | None = None,
    require_guard_ok: bool = True,
) -> K2Selection:
    """Select a K2 candidate by explicit top1 or forced index.

    - top1 always uses ``bundle.top1_index`` (not probability argmax).
    - force accepts only 0 or 1.
    - Guard non-OK / missing ID / hash mismatch → :class:`K2SelectionError`.
    - Never falls back to "first available" candidate.
    """
    mode_l = str(mode).strip().lower()
    if mode_l not in {"top1", "force"}:
        raise K2SelectionError(f"unsupported selection mode: {mode}")

    if require_guard_ok and bundle.guard_status != GUARD_OK:
        raise K2SelectionError(
            f"guard_not_ok: status={bundle.guard_status} reasons={bundle.guard_reasons}"
        )
    if bundle.build_error:
        raise K2SelectionError(f"build_error: {bundle.build_error}")

    if mode_l == "top1":
        idx = int(bundle.top1_index)
    else:
        if force_index is None:
            raise K2SelectionError("force mode requires force_index 0 or 1")
        idx = int(force_index)
        if idx not in (0, 1):
            raise K2SelectionError(f"force_index must be 0 or 1, got {idx}")

    if idx < 0 or idx >= len(bundle.candidates):
        raise K2SelectionError(f"candidate index out of range: {idx}")

    cand = bundle.candidates[idx]
    cid = cand.candidate_id
    if not cid:
        raise K2SelectionError("empty candidate_id")

    spec = bundle.execution_specs.get(cid)
    if spec is None:
        raise K2SelectionError(f"orphan_candidate_id: {cid}")
    if spec.candidate_id != cid:
        raise K2SelectionError(f"execution_spec id mismatch: {spec.candidate_id} != {cid}")
    bind_reasons = verify_execution_spatial_binding(bundle, spec)
    if bind_reasons:
        raise K2SelectionError(
            "execution_spatial_binding_failed: " + ",".join(bind_reasons)
        )

    return K2Selection(
        candidate_id=cid,
        candidate_index=idx,
        execution_spec=spec,
        mode=mode_l,
        bundle=bundle,
    )


def make_source_id(frame_id: str, candidate_id: str) -> str:
    return f"{frame_id}:{candidate_id}"


def select_k2_spatial(
    bundle: K2PredictionBundleV2,
    *,
    mode: Literal["top1", "force"] = "top1",
    force_index: int | None = None,
    require_guard_ok: bool = True,
) -> K2Selection:
    """Select Spatial K2 V2 candidate.

    Semantics (R2-X contract):
    - top1 / World path: unavailable selected candidate → deterministic candidate 0.
    - force mode: unavailable target → **fail-closed** (no silent fallback to 0).
    - Always re-validate execution-spec content hashes before return.
    """
    from driving_vla.model.k2_spatial_guard import verify_execution_spec_content_hash

    mode_l = str(mode).strip().lower()
    if mode_l not in {"top1", "force"}:
        raise K2SelectionError(f"unsupported selection mode: {mode}")
    if require_guard_ok and bundle.guard_status != GUARD_OK_V2:
        raise K2SelectionError(
            f"guard_not_ok_v2: status={bundle.guard_status} reasons={bundle.guard_reasons}"
        )
    if bundle.build_error:
        raise K2SelectionError(f"build_error: {bundle.build_error}")
    if mode_l == "top1":
        idx = int(bundle.top1_index)
    else:
        if force_index is None:
            raise K2SelectionError("force mode requires force_index 0 or 1")
        idx = int(force_index)
        if idx not in (0, 1):
            raise K2SelectionError(f"force_index must be 0 or 1, got {idx}")
    if idx < 0 or idx >= len(bundle.candidates):
        raise K2SelectionError(f"candidate index out of range: {idx}")
    cand = bundle.candidates[idx]
    if not cand.available:
        if mode_l == "force":
            raise K2SelectionError(
                f"unavailable_candidate_force_fail_closed: index={idx} "
                f"id={cand.candidate_id} reason={cand.availability_reason}"
            )
        # top1 / world: deterministic fallback to candidate 0
        idx = 0
        cand = bundle.candidates[0]
        if not cand.available:
            raise K2SelectionError("candidate0_unavailable")
    cid = cand.candidate_id
    spec = bundle.execution_specs.get(cid)
    if spec is None:
        raise K2SelectionError(f"orphan_candidate_id: {cid}")
    if spec.candidate_id != cid:
        raise K2SelectionError(f"execution_spec id mismatch: {spec.candidate_id} != {cid}")
    bind_reasons = verify_execution_spec_content_hash(bundle, cid)
    if bind_reasons:
        raise K2SelectionError(
            "execution_spec_content_hash_failed: " + ",".join(bind_reasons)
        )
    return K2Selection(
        candidate_id=cid,
        candidate_index=idx,
        execution_spec=spec,
        mode=mode_l,
        bundle=bundle,
    )


def select_k2_semantic_v3(
    bundle: K2PredictionBundleV3,
    *,
    mode: Literal["top1", "force"] = "top1",
    force_index: int | None = None,
    require_guard_ok: bool = True,
) -> K2Selection:
    """Select a route-bound K2 V3 candidate without weakening its Guard.

    In top1 mode, an unavailable or rejected learned top1 may fall back only
    to the other candidate preserving the exact same route/topology binding.
    Forced evaluation never falls back.
    """
    from driving_vla.model.k2_v3_guard import (
        GUARD_OK as GUARD_OK_V3,
        attach_k2_v3_guard,
    )

    mode_l = str(mode).strip().lower()
    if mode_l not in {"top1", "force"}:
        raise K2SelectionError(f"unsupported selection mode: {mode}")
    guarded = (
        bundle
        if "candidate_valid" in bundle.guard_metrics
        else attach_k2_v3_guard(bundle)
    )
    if require_guard_ok and guarded.guard_status != GUARD_OK_V3:
        raise K2SelectionError(
            f"guard_not_ok_v3: status={guarded.guard_status} "
            f"reasons={guarded.guard_reasons}"
        )
    valid = dict(guarded.guard_metrics.get("candidate_valid") or {})
    if mode_l == "force":
        if force_index is None:
            raise K2SelectionError("force mode requires force_index 0 or 1")
        idx = int(force_index)
        if idx not in (0, 1):
            raise K2SelectionError(f"force_index must be 0 or 1, got {idx}")
        ordered = (idx,)
    else:
        ordered = (int(guarded.top1_index), 1 - int(guarded.top1_index))

    selected_index: int | None = None
    for idx in ordered:
        candidate = guarded.candidates[idx]
        if bool(valid.get(candidate.candidate_id, False)):
            selected_index = idx
            break
    if selected_index is None:
        target = "forced candidate" if mode_l == "force" else "all candidates"
        raise K2SelectionError(f"{target} rejected by K2 V3 Guard")
    candidate = guarded.candidates[selected_index]
    # These bindings are deliberately repeated at the execution boundary.
    if candidate.route_hash != guarded.route_context.route_hash:
        raise K2SelectionError("v3 execution route_hash mismatch")
    if candidate.topology_hash != guarded.route_context.topology_hash:
        raise K2SelectionError("v3 execution topology_hash mismatch")
    return K2Selection(
        candidate_id=candidate.candidate_id,
        candidate_index=selected_index,
        execution_spec=candidate,
        mode=mode_l,
        bundle=guarded,
    )


def apply_k2_to_executors(
    selection: K2Selection,
    *,
    speed_planner: VLASpeedPlanner,
    path_manager: VLAPathManager,
    ego: EgoPose,
    stamp_s: float,
    frame_id: str,
    dt_s: float,
    ego_speed_mps: float | None = None,
    nav_target_map_xy: tuple[float, float] | None = None,
) -> K2ApplyResult:
    """Bind selected execution_spec to existing SpeedPlanner + PathManager."""
    spec = selection.execution_spec
    source_id = make_source_id(frame_id, selection.candidate_id)
    ego_v = float(ego.speed_mps if ego_speed_mps is None else ego_speed_mps)

    speed_decision = speed_planner.update(
        spec.speed_samples_mps,
        dt_s=float(dt_s),
        ego_speed_mps=ego_v,
    )
    path_update = path_manager.update(
        spec.spatial_path_xy,
        ego=ego,
        target_speed_mps=float(speed_decision.target_speed_mps),
        stamp_s=float(stamp_s),
        source_id=source_id,
        nav_target_map_xy=nav_target_map_xy,
    )

    executed = selection.candidate_id
    # Prefer raw source_id (always set from our update); committed should also
    # propagate it after R1 PathManager bind.
    raw_sid = ""
    if path_update.raw is not None:
        raw_sid = str(getattr(path_update.raw, "source_id", "") or "")
    committed_sid = ""
    if path_update.committed is not None:
        committed_sid = str(getattr(path_update.committed, "source_id", "") or "")
    bind_sid = raw_sid or committed_sid or source_id
    if path_update.accepted and selection.candidate_id not in bind_sid:
        raise K2SelectionError(
            f"executed_source_id_mismatch: expected substring {selection.candidate_id!r} "
            f"in raw={raw_sid!r} committed={committed_sid!r}"
        )

    return K2ApplyResult(
        selected_candidate_id=selection.candidate_id,
        executed_candidate_id=executed,
        source_id=source_id,
        speed_decision=speed_decision,
        path_update=path_update,
        selection=selection,
    )


def selection_event_fields(selection: K2Selection) -> Mapping[str, Any]:
    """Minimal audit fields for runner evidence events."""
    b = selection.bundle
    if isinstance(b, K2PredictionBundleV3):
        c = b.candidates[selection.candidate_index]
        alternative = b.candidates[1]
        return {
            "schema_version": b.schema_version,
            "generated_candidate_ids": [
                candidate.candidate_id for candidate in b.candidates
            ],
            "top1_index": int(b.top1_index),
            "selected_candidate_id": selection.candidate_id,
            "selected_candidate_index": selection.candidate_index,
            "selection_mode": selection.mode,
            "route_maneuver": b.route_context.maneuver.value,
            "alternative_kind": c.alternative_kind.value,
            "target_lane_side": c.target_lane_side.value,
            "maneuver_phase": c.maneuver_phase.value,
            "route_hash": c.route_hash,
            "topology_hash": c.topology_hash,
            "intent_hash": c.intent_hash,
            "spatial_path_hash": c.spatial_path_hash,
            "timed_trajectory_hash": c.timed_trajectory_hash,
            "bundle_hash": b.bundle_hash,
            "guard_status": b.guard_status,
            "guard_reasons": list(b.guard_reasons),
            "candidate_valid": dict(
                b.guard_metrics.get("candidate_valid") or {}
            ),
            "k_eff": int(b.guard_metrics.get("k_eff", 0)),
            "alternative_slot_kind": alternative.alternative_kind.value,
            "alternative_slot_available": bool(alternative.available),
            "alternative_slot_target_lane_side": (
                alternative.target_lane_side.value
            ),
            "alternative_slot_maneuver_phase": (
                alternative.maneuver_phase.value
            ),
            "alternative_slot_availability_reason": (
                alternative.availability_reason
            ),
            "alternative_slot_spatial_path_xy": [
                list(point) for point in alternative.spatial_path_xy
            ],
            "alternative_slot_speed_samples_mps": [
                float(value) for value in alternative.speed_samples_mps
            ],
            "probability_source": b.probability_source,
            "config_hash": b.config_hash,
        }
    d = b.diagnostics
    return {
        "generated_candidate_ids": list(b.candidate_ids()),
        "top1_index": int(b.top1_index),
        "selected_candidate_id": selection.candidate_id,
        "selected_candidate_index": selection.candidate_index,
        "selection_mode": selection.mode,
        "native_path_hash": b.native_path_hash,
        "timed_trajectory_hash": selection.execution_spec.timed_trajectory_hash,
        "retimer_version": b.retimer_version,
        "branch_type": b.branch_type,
        "guard_status": b.guard_status,
        "guard_reasons": list(b.guard_reasons),
        "probability_source": b.probability_source,
        "probability_margin": b.probability_margin,
        "mean_speed_gap_mps": d.mean_speed_gap_mps,
        "final_progress_gap_m": d.final_progress_gap_m,
        "max_position_separation_m": d.max_position_separation_m,
        "collapsed": d.collapsed,
        "collapse_reason": d.collapse_reason,
        "selection_space_eligible": d.selection_space_eligible,
        "path_speed_cap_active": d.path_speed_cap_active,
        "config_hash": b.config_hash,
    }
