"""C3 supervision recovery and semantic audit for the frozen C2 release.

The C2 ``anchor.route`` is an observable reference line.  It is an input to
the Classic planner and is therefore not, by itself, an expert driving label.
This module replays the same native Classic planner from the saved observable
anchor, binds the replay to the original proposal's raw/canonical identities,
and exposes only the actually supported interval to the SFT adapter.

The replay is intentionally separate from the training command.  It produces
a small, reviewable JSON index containing the new derived hashes and every
per-root decision while leaving the frozen C2 release untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from classic_stack.geometry import ReferencePath
from classic_stack.planning.frenet import ActorState, PlanRequest
from driving_vla.adapter.policy_adapter import ObservationBundle
from driving_vla.hybrid.contracts import ObservableAnchor
from driving_vla.hybrid.generators import ClassicExpertGenerator, route_revision_sha256
from driving_vla.model.canonicalizer import stable_sha256
from driving_vla.model.sft import (
    EXECUTION_DT_S,
    RELEASE_ID,
    QUALITY_PROFILE,
    ROUTE_STEPS,
    SPEED_STEPS,
    SPEED_DT_S,
    cumulative_arclength,
    map_to_ego,
    sample_polyline,
)
from safety_kernel.contracts.types import (
    ObservationPrivilege,
    ObservableSnapshot,
    TrackedObject,
    TrafficLightObs,
)


SCHEMA_VERSION = "safedrive.c3.native_reconstruction.v1"
SOURCE_KIND = "native_classic_planner_trajectory"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _resolve_path(raw: str | Path) -> Path:
    """Resolve both the WSL paths recorded by C2 and Windows paths."""

    value = str(raw)
    candidate = Path(value)
    if candidate.exists():
        return candidate
    if value.startswith("/mnt/") and len(value) > 6:
        drive = value[5].upper()
        converted = Path(f"{drive}:/" + value[7:].replace("/", "/"))
        if converted.exists():
            return converted
    return candidate


def resolve_release_base(release_root: Path, release: Mapping[str, Any]) -> Path:
    raw = release.get("raw_data_references", {}).get("base")
    if not raw:
        raise ValueError("c3_release_base_reference_missing")
    path = _resolve_path(raw)
    if not path.is_absolute():
        path = (release_root / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _as_tuple_xy(rows: Sequence[Sequence[Any]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(row[0]), float(row[1])) for row in rows)


def observable_anchor_from_record(record: Mapping[str, Any]) -> ObservableAnchor:
    """Rehydrate the observable H1 anchor without importing oracle labels."""

    summary = record.get("observable_anchor")
    snapshot_raw = record.get("observable_snapshot")
    if not isinstance(summary, Mapping) or not isinstance(snapshot_raw, Mapping):
        raise ValueError("c3_anchor_observable_fields_missing")
    actors = tuple(
        TrackedObject(
            actor_id=str(row["actor_id"]),
            class_name=str(row["class_name"]),
            x=float(row["x"]),
            y=float(row["y"]),
            yaw=float(row["yaw"]),
            vx=float(row["vx"]),
            vy=float(row["vy"]),
            length_m=float(row["length_m"]),
            width_m=float(row["width_m"]),
            observed_time_s=float(row["observed_time_s"]),
            lost=bool(row.get("lost", False)),
            source=str(row.get("source", "observable")),
            cov_xx=float(row.get("cov_xx", 0.25)),
            cov_yy=float(row.get("cov_yy", 0.25)),
        )
        for row in snapshot_raw.get("actors", ())
    )
    lights = tuple(
        TrafficLightObs(
            light_id=str(row.get("light_id", "")),
            state=str(row.get("state", "unknown")),
            distance_m=float(row["distance_m"]),
            observed_time_s=float(row["observed_time_s"]),
            stop_line_distance_m=(
                None
                if row.get("stop_line_distance_m") is None
                else float(row["stop_line_distance_m"])
            ),
            controls_ego_lane=(
                None
                if row.get("controls_ego_lane") is None
                else bool(row["controls_ego_lane"])
            ),
        )
        for row in snapshot_raw.get("traffic_lights", ())
    )
    privilege = ObservationPrivilege(str(snapshot_raw.get("privilege", "observable")))
    snapshot = ObservableSnapshot(
        run_id=str(snapshot_raw["run_id"]),
        frame_id=str(snapshot_raw["frame_id"]),
        scenario_id=str(snapshot_raw["scenario_id"]),
        simulation_time_s=float(snapshot_raw["simulation_time_s"]),
        wall_time_s=float(snapshot_raw["wall_time_s"]),
        ego_x=float(snapshot_raw["ego_x"]),
        ego_y=float(snapshot_raw["ego_y"]),
        ego_yaw=float(snapshot_raw["ego_yaw"]),
        ego_v=float(snapshot_raw["ego_v"]),
        ego_a=float(snapshot_raw.get("ego_a", 0.0)),
        observed_time_s=float(snapshot_raw["observed_time_s"]),
        freshness_s=float(snapshot_raw.get("freshness_s", 0.0)),
        speed_limit_mps=(
            None
            if snapshot_raw.get("speed_limit_mps") is None
            else float(snapshot_raw["speed_limit_mps"])
        ),
        actors=actors,
        traffic_lights=lights,
        corridor_centerline=_as_tuple_xy(snapshot_raw.get("corridor_centerline", ())),
        corridor_half_width_m=float(snapshot_raw.get("corridor_half_width_m", 1.75)),
        privilege=privilege,
        oracle_fields=dict(snapshot_raw.get("oracle_fields", {})),
        schema_version=str(snapshot_raw.get("schema_version", "safedrive.safety.contracts.v1")),
        coordinate_frame=str(snapshot_raw.get("coordinate_frame", "map")),
    )
    history = tuple(
        (
            float(row["ego_x"]),
            float(row["ego_y"]),
            float(row["ego_yaw"]),
            float(row["ego_speed_mps"]),
        )
        for row in record.get("observable_history", ())
    )
    bundle = ObservationBundle(
        run_id=str(summary["run_id"]),
        frame_id=str(summary["frame_id"]),
        scenario_id=str(summary["scenario_id"]),
        simulation_time_s=float(summary["simulation_time_s"]),
        wall_time_s=float(summary["wall_time_s"]),
        carla_frame=int(summary["carla_frame"]),
        ego_x=float(summary["ego"]["x"]),
        ego_y=float(summary["ego"]["y"]),
        ego_yaw=float(summary["ego"]["yaw"]),
        ego_v=float(summary["ego"]["v"]),
        route_xy=_as_tuple_xy(record.get("route", ())),
        ego_history=history,
        meta={
            "c3_input_source": "frozen_c2_observable_anchor",
            "history_speed_semantics": "recorded_history_audited_separately",
        },
    )
    return ObservableAnchor(
        observation_id=str(summary["observation_id"]),
        bundle=bundle,
        safety_snapshot=snapshot,
        route_revision=str(summary["route_revision"]),
        sensor_frames={str(k): int(v) for k, v in summary["sensor_frames"].items()},
        sensor_timestamps_s={str(k): float(v) for k, v in summary["sensor_timestamps_s"].items()},
    )


def _expert_branch(pair: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    for proposal in pair.get("proposals", ()):
        if proposal.get("kind") != "nominal" or proposal.get("audit_source") != "expert":
            continue
        proposal_id = proposal.get("proposal_id")
        branch = next(
            (item for item in pair.get("branches", ()) if item.get("proposal_id") == proposal_id),
            None,
        )
        if branch is not None:
            return proposal, branch
    raise ValueError("c3_expert_nominal_branch_missing")


def _plan_request(anchor: ObservableAnchor, generator: ClassicExpertGenerator) -> PlanRequest:
    """Mirror ClassicExpertGenerator.generate's request construction."""

    route = anchor.bundle.route_xy
    reference = ReferencePath.from_xy([point[0] for point in route], [point[1] for point in route])
    s0, d0 = reference.project(anchor.bundle.ego_x, anchor.bundle.ego_y)
    if abs(d0) > generator.planner.config.road_half_width_m:
        raise ValueError(f"expert_ego_outside_route:d={d0:.6f}")
    actors = tuple(
        ActorState(
            actor_id=actor.actor_id,
            x=actor.x,
            y=actor.y,
            yaw=actor.yaw,
            speed_mps=math.hypot(actor.vx, actor.vy),
            length_m=actor.length_m,
            width_m=actor.width_m,
            model="cv",
        )
        for actor in anchor.safety_snapshot.actors
        if not actor.lost
    )
    red_distances = [
        light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m
        for light in anchor.safety_snapshot.traffic_lights
        if light.state.lower() == "red"
        and light.controls_ego_lane is not False
        and (light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m) >= 0.0
    ]
    stop_s = min(red_distances) if red_distances else None
    target_speed = min(6.0, float(anchor.safety_snapshot.speed_limit_mps)) if anchor.safety_snapshot.speed_limit_mps is not None else 6.0
    return PlanRequest(
        reference=reference,
        v0=max(0.0, anchor.bundle.ego_v),
        a0=float(anchor.safety_snapshot.ego_a),
        s0=max(0.0, min(s0, reference.length)),
        d0=d0,
        scenario_kind="stop" if stop_s is not None else "follow",
        actors=actors,
        target_speed_mps=max(0.0, target_speed),
        stop_s=None if stop_s is None else max(0.0, stop_s - 1.0),
        seed=int(anchor.bundle.carla_frame),
    )


def _raw_plan_rows(plan: Any) -> list[dict[str, float]]:
    if plan.trajectory is None:
        return []
    return [
        {
            "t": float(point.t),
            "x": float(point.x),
            "y": float(point.y),
            "yaw": float(point.yaw),
            "v": float(point.v),
            "a": float(point.a),
            "kappa": float(point.kappa),
            "jerk": float(point.jerk),
        }
        for point in plan.trajectory.points
    ]


def _canonical_rows(candidate: Any) -> list[dict[str, float]]:
    return [
        {
            "t": float(point.t),
            "x": float(point.x),
            "y": float(point.y),
            "yaw": float(point.yaw),
            "v": float(point.v),
            "a": float(point.a),
            "kappa": float(point.kappa),
        }
        for point in candidate.candidate.points
    ]


def _route_from_native(anchor: ObservableAnchor, raw_rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, float]:
    origin = (anchor.bundle.ego_x, anchor.bundle.ego_y)
    points = np.asarray([origin] + [(float(row["x"]), float(row["y"])) for row in raw_rows], dtype=np.float64)
    if points.shape[0] < 2 or float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) <= 1e-8:
        # A planner that remains at the anchor is a valid stop observation.  A
        # single point is represented by point zero only; no fake forward point
        # is introduced to make a route look complete.
        return np.zeros((ROUTE_STEPS, 2), dtype=np.float64), np.asarray([True] + [False] * (ROUTE_STEPS - 1)), 0.0
    support = float(cumulative_arclength(points)[-1])
    sampled, mask = sample_polyline(points, np.arange(ROUTE_STEPS, dtype=np.float64))
    return map_to_ego(sampled, ego_x=anchor.bundle.ego_x, ego_y=anchor.bundle.ego_y, ego_yaw=anchor.bundle.ego_yaw), mask, support


def _quality_audit(anchor: ObservableAnchor, plan: Any, raw_rows: Sequence[Mapping[str, Any]], generator: ClassicExpertGenerator) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not bool(plan.ok) or plan.trajectory is None:
        reasons.append(f"planner_not_ok:{plan.failure_code}")
        return "REJECT", reasons
    if len(raw_rows) < 2:
        reasons.append("native_trajectory_too_short")
    times = np.asarray([float(row["t"]) for row in raw_rows], dtype=np.float64)
    xy = np.asarray([[float(row["x"]), float(row["y"])] for row in raw_rows], dtype=np.float64)
    speeds = np.asarray([float(row["v"]) for row in raw_rows], dtype=np.float64)
    accelerations = np.asarray([float(row["a"]) for row in raw_rows], dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (times, xy, speeds, accelerations)):
        reasons.append("native_trajectory_non_finite")
    if times.size >= 2 and not np.all(np.diff(times) > 0.0):
        reasons.append("native_time_not_increasing")
    if np.any(speeds < -1e-6):
        reasons.append("native_negative_speed")
    if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 5.0):
        reasons.append("native_position_jump")
    # The native planner's own kinematic limits are the audit reference.  A
    # replay that violates them is retained as a case but cannot be SFT.
    vehicle = generator.planner.config.vehicle
    if accelerations.size and np.max(np.abs(accelerations)) > float(vehicle.max_accel_mps2) + 1e-4:
        reasons.append("native_acceleration_limit")
    if accelerations.size >= 2 and times.size == accelerations.size:
        dt = np.diff(times)
        jerk = np.diff(accelerations) / np.maximum(dt, 1e-9)
        if np.max(np.abs(jerk)) > float(vehicle.max_jerk_mps3) + 1e-3:
            reasons.append("native_jerk_limit")
    # Reference-line deviation is checked in Frenet space; negative ego-frame
    # coordinates are legal and are never used as a rejection heuristic.
    reference = ReferencePath.from_xy(
        [point[0] for point in anchor.bundle.route_xy],
        [point[1] for point in anchor.bundle.route_xy],
    )
    deviations = [abs(reference.project(float(x), float(y))[1]) for x, y in xy]
    if deviations and max(deviations) > float(generator.planner.config.road_half_width_m) + 0.25:
        reasons.append("native_reference_deviation")
    # A red light is a planning constraint.  Preserve the sample as a case if
    # the replay crosses the chosen stop boundary instead of claiming it is a
    # normal expert label.
    red = [
        light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m
        for light in anchor.safety_snapshot.traffic_lights
        if light.state.lower() == "red" and light.controls_ego_lane is not False
    ]
    if red:
        stop_limit = min(red) - 1.0
        ref_s = [reference.project(float(x), float(y))[0] for x, y in xy]
        if ref_s and max(ref_s) > stop_limit + 0.75:
            reasons.append("native_red_light_stop_violation")
    return ("PASS" if not reasons else "REVIEW"), reasons


def reconstruct_one(
    *,
    base_root: Path,
    release_item: Mapping[str, Any],
    generator: ClassicExpertGenerator | None = None,
) -> dict[str, Any]:
    pair_id = str(release_item["pair_id"])
    anchor_path = base_root / "anchors" / f"{pair_id}.json"
    pair_path = base_root / "pairs" / f"{pair_id}.json"
    anchor_record = _load_json(anchor_path)
    pair = _load_json(pair_path)
    proposal_ref, branch = _expert_branch(pair)
    proposal_digest = str(proposal_ref["proposal_sha256"])
    proposal_path = base_root / "proposals" / f"{pair_id}__{proposal_digest}.json"
    proposal_file = _load_json(proposal_path)
    anchor = observable_anchor_from_record(anchor_record)
    generator = generator or ClassicExpertGenerator()
    expected_revision = route_revision_sha256(anchor.bundle.route_xy)
    provenance = proposal_file.get("provenance", {})
    if str(provenance.get("route_revision", "")) != expected_revision:
        raise ValueError("native_reconstruction_route_revision_mismatch")
    if str(provenance.get("generator_id", "")) != generator.generator_id:
        raise ValueError("native_reconstruction_generator_id_mismatch")
    if str(branch.get("proposal_sha256", "")) != proposal_digest:
        raise ValueError("native_reconstruction_branch_proposal_mismatch")
    candidate = generator.generate(anchor)
    canonical_rows = _canonical_rows(candidate)
    native_request = _plan_request(anchor, generator)
    native_plan = generator.planner.plan(native_request)
    raw_rows = _raw_plan_rows(native_plan)
    quality, quality_reasons = _quality_audit(anchor, native_plan, raw_rows, generator)
    canonical_hash = str(candidate.provenance.canonical_sha256)
    raw_hash = str(candidate.provenance.raw_sha256)
    original_canonical_hash = str(proposal_file.get("provenance", {}).get("canonical_sha256", proposal_digest))
    canonical_match = canonical_hash == proposal_digest and original_canonical_hash == proposal_digest
    raw_match = raw_hash == str(proposal_file.get("provenance", {}).get("raw_sha256", raw_hash))
    route_target, route_mask, support_m = _route_from_native(anchor, raw_rows)
    speed_target = map_to_ego(
        np.asarray([[row["x"], row["y"]] for row in canonical_rows], dtype=np.float64),
        ego_x=anchor.bundle.ego_x,
        ego_y=anchor.bundle.ego_y,
        ego_yaw=anchor.bundle.ego_yaw,
    ) if len(canonical_rows) == SPEED_STEPS else np.zeros((SPEED_STEPS, 2), dtype=np.float64)
    speed_mask = np.ones(SPEED_STEPS, dtype=bool) if len(canonical_rows) == SPEED_STEPS and canonical_match and quality == "PASS" else np.zeros(SPEED_STEPS, dtype=bool)
    # The history conflict is recorded as a data finding.  The current
    # SimLingo input contract consumes the current snapshot speed only; it does
    # not silently replace it with the future execution speed.
    history = anchor_record.get("observable_history", ())
    history_speed = None if not history else float(history[-1].get("ego_speed_mps"))
    history_disagreement = None if history_speed is None else history_speed - float(anchor.bundle.ego_v)
    failures = list(quality_reasons)
    if not canonical_match:
        failures.append("canonical_hash_mismatch")
    if not raw_match:
        failures.append("raw_hash_mismatch")
    if history_disagreement is not None and abs(history_disagreement) > 1e-8:
        failures.append("current_history_speed_semantics_conflict_recorded")
    status = "PASS" if quality == "PASS" and canonical_match and raw_match else "REVIEW"
    return {
        "pair_id": pair_id,
        "root_id": pair_id,
        "split": str(release_item.get("split", "")),
        "family": str(release_item.get("family", "")),
        "map": str(release_item.get("map", "")),
        "weather": str(release_item.get("weather", "")),
        "seed": int(release_item.get("seed", 0)),
        "anchor_path": str(anchor_path),
        "pair_path": str(pair_path),
        "proposal_path": str(proposal_path),
        "anchor_content_sha256": stable_sha256(anchor_record),
        "proposal_file_sha256": stable_sha256(proposal_file),
        "original_proposal_sha256": proposal_digest,
        "original_proposal_provenance": {
            "raw_sha256": str(provenance.get("raw_sha256", "")),
            "canonical_sha256": original_canonical_hash,
            "generator_id": str(provenance.get("generator_id", "")),
            "generator_hash": str(provenance.get("generator_hash", "")),
            "route_revision": expected_revision,
        },
        "reconstruction": {
            "status": status,
            "quality": quality,
            "quality_reasons": sorted(set(failures)),
            "source": SOURCE_KIND,
            "generator_id": generator.generator_id,
            "generator_hash": generator.generator_hash,
            "planner_config_hash": generator.planner.config_hash,
            "native_plan_ok": bool(native_plan.ok),
            "native_plan_failure_code": native_plan.failure_code,
            "native_plan_points": len(raw_rows),
            "native_plan_horizon_s": float(raw_rows[-1]["t"]) if raw_rows else 0.0,
            "native_support_m": support_m,
            "native_route_valid_points": int(route_mask.sum()),
            "canonical_points": len(canonical_rows),
            "canonical_raw_sha256": raw_hash,
            "canonical_sha256": canonical_hash,
            "raw_identity_match": raw_match,
            "canonical_identity_match": canonical_match,
            "raw_plan_sha256": stable_sha256(raw_rows),
            "raw_plan": raw_rows,
            "canonical": canonical_rows,
        },
        "route_target_ego": route_target.tolist(),
        "route_mask": route_mask.tolist(),
        "speed_target_ego": speed_target.tolist(),
        "speed_mask": speed_mask.tolist(),
        "route_support_m": support_m,
        "history_speed_mps": history_speed,
        "current_speed_mps": float(anchor.bundle.ego_v),
        "history_speed_disagreement_mps": history_disagreement,
        "input_contract": {
            "future_labels_in_input": False,
            "world_labels_in_sft": False,
            "current_speed_source": "observable_snapshot.ego_v",
            "history_speed_source": "stored_history_audit_only; conflict_not_replaced",
        },
    }


def reconstruct_release(
    release_root: Path | str,
    *,
    splits: Sequence[str] = ("train", "validation"),
) -> dict[str, Any]:
    """Replay all requested frozen-release roots and return a new index."""

    release_root = Path(release_root)
    release = _load_json(release_root / "release-index.json")
    if release.get("dataset_id") != RELEASE_ID or release.get("quality_profile") != QUALITY_PROFILE:
        raise ValueError("c3_release_identity_mismatch")
    if release.get("status") != "DEV_DATA_READY":
        raise ValueError("c3_release_not_ready")
    wanted = tuple(str(item) for item in splits)
    base_root = resolve_release_base(release_root, release)
    generator = ClassicExpertGenerator()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for release_item in release.get("samples", ()):
        if str(release_item.get("split", "")) not in wanted:
            continue
        try:
            rows.append(reconstruct_one(base_root=base_root, release_item=release_item, generator=generator))
        except Exception as exc:  # each root remains auditable
            failures.append({
                "pair_id": str(release_item.get("pair_id", "")),
                "split": str(release_item.get("split", "")),
                "reason": f"{type(exc).__name__}:{exc}",
            })
    rows.sort(key=lambda row: (row["split"], row["root_id"]))
    split_counts = {split: sum(row["split"] == split for row in rows) for split in wanted}
    quality_counts = {
        quality: sum(row["reconstruction"]["quality"] == quality for row in rows)
        for quality in sorted({row["reconstruction"]["quality"] for row in rows})
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "quality_profile": QUALITY_PROFILE,
        "release_root": str(release_root.resolve()),
        "base_root": str(base_root),
        "splits": list(wanted),
        "sample_count": len(rows),
        "split_counts": split_counts,
        "failure_count": len(failures),
        "failures": failures,
        "quality_counts": quality_counts,
        "native_generator_id": generator.generator_id,
        "native_generator_hash": generator.generator_hash,
        "native_planner_config_hash": generator.planner.config_hash,
        "route_contract": {
            "source": SOURCE_KIND,
            "reference_route_is_input_only": True,
            "arc_queries_m": list(range(ROUTE_STEPS)),
            "spacing_m": 1.0,
            "actual_support_only": True,
            "no_extrapolation": True,
        },
        "speed_contract": {
            "source": SOURCE_KIND,
            "steps": SPEED_STEPS,
            "dt_s": SPEED_DT_S,
            "canonicalizer_contract": "safedrive.trajectory_canonicalizer.v2",
            "actual_native_plan_required": True,
        },
        "execution_timeline_contract": {
            "dt_s": EXECUTION_DT_S,
            "use": "separate_outcome_and_case_audit",
            "not_used_as_native_sft_without_initial_state_proof": True,
        },
        "history_audit": {
            "current_speed_source": "observable_snapshot.ego_v",
            "zero_preserved": True,
            "history_conflict_count": sum(
                row.get("history_speed_disagreement_mps") is not None
                and abs(float(row["history_speed_disagreement_mps"])) > 1e-8
                for row in rows
            ),
            "history_is_not_used_to_overwrite_current": True,
        },
        "future_labels_in_input": False,
        "world_labels_in_sft": False,
        "rows": rows,
    }
    payload["content_sha256"] = stable_sha256(payload)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_KIND",
    "observable_anchor_from_record",
    "reconstruct_one",
    "reconstruct_release",
    "resolve_release_base",
]
