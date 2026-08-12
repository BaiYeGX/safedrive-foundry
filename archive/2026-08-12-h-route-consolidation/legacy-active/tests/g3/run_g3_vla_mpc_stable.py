#!/usr/bin/env python3
"""Stable pure-VLA spatial path → constrained MPC CARLA demonstration.

The CARLA map route is used only to form two coarse navigation targets for the
VLA.  The tracked path geometry comes exclusively from SimLingo.  Inference is
serialized with CARLA rendering by default to reduce D3D/CUDA contention on a
single Windows/WSL GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simlingo-main"))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import carla  # noqa: E402

from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.evaluation.maneuver_completion import (  # noqa: E402
    route_projection,
    signed_route_projection,
)
from driving_vla.model.neural_policy import (  # noqa: E402
    NeuralV0Policy,
    NeuralV1Policy,
    NeuralV2Policy,
    NeuralV3Policy,
)
from driving_vla.model.simlingo_contract import (  # noqa: E402
    SimLingoContractConfig,
    carla_bgra_to_bgr,
    ego_to_map,
    legacy_contract_config,
    navigation_targets as contract_navigation_targets,
    resolve_navigation_prompt_conditioning,
)
from driving_vla.model.simlingo_runtime import (  # noqa: E402
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_NATIVE_SIZE,
    SIMLINGO_CAMERA_XYZ,
    SimLingoNeuralRuntime,
)
from driving_vla.runtime.k2_execution import (  # noqa: E402
    K2SelectionError,
    apply_k2_to_executors,
    select_k2,
    select_k2_spatial,
    select_k2_semantic_v3,
    selection_event_fields,
)
from driving_vla.runtime.path_manager import (  # noqa: E402
    EgoPose,
    PathManagerConfig,
    SpatialPath,
    VLAPathManager,
    compare_native_dense_curvature,
    spatial_path_from_xy,
)
from driving_vla.runtime.lane_evidence import (  # noqa: E402
    DEFAULT_LANE_INVASION_GAP_S,
    LaneInvasionEpisodeBook,
    lane_center_error_stats,
    multi_deadband_sign_flips,
    observe_lane_oracle,
    steer_derivative_metrics,
)
from driving_vla.runtime.terminal_stop import (  # noqa: E402
    classify_terminal_stop,
    observe_traffic_light,
)
from driving_vla.runtime.stationary_requery import (  # noqa: E402
    StationaryRequeryConfig,
    StationaryRequeryDecision,
    evaluate_stationary_requery,
    requery_success,
)
from driving_vla.runtime.vehicle_geometry import (  # noqa: E402
    DEFAULT_EGO_BLUEPRINT,
    vehicle_geometry_from_carla_vehicle,
)
from driving_vla.runtime.vla_mpc_tracker import (  # noqa: E402
    ConstrainedVLAMPC,
    VLAMPCConfig,
)
from driving_vla.runtime.vla_speed_planner import (  # noqa: E402
    VLASpeedConfig,
    VLASpeedPlanner,
)
from runtime.carla_connection import (  # noqa: E402
    ConnectionResolver,
    build_carla_launch_arguments,
    launch_params_match,
    normalize_rhi,
    parse_offscreen_from_command_line,
    parse_resolution_from_command_line,
    parse_rhi_from_command_line,
    query_windows_carla_command_line,
)
from run_g3_vla_v0_visual_demo import build_route, free_spawn, set_spectator_follow  # noqa: E402

# Playable full-map packages (exclude multi-tile sublevels). Selected only at process start.


_SHARED_SIMLINGO_RUNTIME: SimLingoNeuralRuntime | None = None


def _simlingo_runtime_for_run(
    *, reuse_loaded: bool = False
) -> tuple[SimLingoNeuralRuntime, bool]:
    """Load SimLingo once for an in-process calibration campaign.

    The runtime is inference-only and clears its per-forward feature cache in
    ``SimLingoNeuralRuntime``.  Reusing it therefore avoids repeated backbone
    construction without carrying scenario, controller, or CARLA actor state
    between slots.
    """

    global _SHARED_SIMLINGO_RUNTIME
    if (
        reuse_loaded
        and _SHARED_SIMLINGO_RUNTIME is not None
        and bool(_SHARED_SIMLINGO_RUNTIME.load_report.ok)
    ):
        return _SHARED_SIMLINGO_RUNTIME, True
    runtime = SimLingoNeuralRuntime(device="cuda")
    load = runtime.load()
    if not load.ok:
        raise RuntimeError(f"SimLingo load failed: {load.error}")
    if reuse_loaded:
        _SHARED_SIMLINGO_RUNTIME = runtime
    return runtime, False


def _enough_vla_paths_for_run(
    *,
    accepts: int,
    event_count: int,
    route_change_committed: bool,
) -> bool:
    """Account for the intentional native route-change commitment window."""

    required = (
        1
        if route_change_committed
        else max(2, int(math.ceil(0.5 * max(event_count, 1))))
    )
    return int(accepts) >= required
DEMO_MAP_POOL: tuple[str, ...] = (
    "Town01",
    "Town02",
    "Town03",
    "Town03_Opt",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town10HD_Opt",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
)
# Random endurance runs intentionally avoid the three very large tiled towns:
# they have a much larger startup/failure surface and are opt-in by name.
# Prefer Town03_Opt over full Town03 when sampling randomly (full Town03 has
# hit startup ACCESS_VIOLATION on this host/driver combination).
DEFAULT_RANDOM_MAP_POOL: tuple[str, ...] = (
    "Town03_Opt",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town10HD_Opt",
)
CARLA_DEFAULT_ENGINE_INI = Path("/mnt/e/CARLA_0.9.16/CarlaUE4/Config/DefaultEngine.ini")
CARLA_MAPS_CONTENT = Path("/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps")
CARLA_START_TOML = ROOT / "safedrive_foundry" / "config" / "runtime" / "carla_start.toml"
# Large layered towns ship as Content/Carla/Maps/<Town>/<Town>.umap (not flat Maps/<Town>.umap).
NESTED_MAP_PACKAGE: frozenset[str] = frozenset({"Town11", "Town12", "Town13", "Town15"})
# CARLA Large Maps use tile streaming and require a hero actor near the active tile.
# Town15 is a nested large map (tile streaming); treat like 11–13 for spectator/tiles.
LARGE_MAPS: frozenset[str] = frozenset({"Town11", "Town12", "Town13", "Town15"})


def _write_json(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _camera_sensor_tick_s(vla_period_s: float, sim_dt_s: float, override_s: float) -> float:
    """Avoid rendering camera frames that the slow VLA will never consume."""
    if override_s > 0.0:
        return max(float(sim_dt_s), float(override_s))
    return max(float(sim_dt_s), float(vla_period_s))


def _mode_runs_forward(inference_mode: str) -> bool:
    return str(inference_mode) in {"full", "forward-only"}


def _attach_v3_native_feature_evidence(view: Any, policy: Any) -> Any:
    """Copy same-forward features onto the selected-path evidence view."""
    source = getattr(policy, "last_native", None)
    view.driving_feature = tuple(
        getattr(source, "driving_feature", ()) or ()
    )
    view.driving_feature_hash = str(
        getattr(source, "driving_feature_hash", "") or ""
    )
    view.driving_feature_raw_hash = str(
        getattr(source, "driving_feature_raw_hash", "") or ""
    )
    return view


def _v3_contract_auto_index(bundle: Any, *, should_yield: bool) -> int:
    """Choose the requested legal V3 slot without forcing a rejected branch."""
    preferred = (1, 0) if bool(should_yield) else (0, 1)
    valid = dict(bundle.guard_metrics.get("candidate_valid") or {})
    for index in preferred:
        candidate = bundle.candidates[index]
        if bool(candidate.available) and bool(
            valid.get(candidate.candidate_id, False)
        ):
            return int(index)
    # Preserve fail-closed selection diagnostics when neither candidate is
    # executable; the execution boundary will reject this preferred slot.
    return int(preferred[0])


def _v3_latched_conflict_side(
    previous: str,
    *,
    observed: str,
    conflict_active: bool,
    scenario_family: str,
) -> str:
    """Keep a cut-in's initial risk side stable until the conflict clears."""
    if "cut" not in str(scenario_family).lower():
        return str(observed)
    if not bool(conflict_active):
        return "none"
    prior = str(previous or "none").lower()
    if prior in {"left", "right"}:
        return prior
    current = str(observed or "none").lower()
    return current if current in {"left", "right"} else "none"


def _v3_latched_conflict_active(
    previous: bool,
    *,
    observed: bool,
    scenario_family: str,
    actor_present: bool,
    actor_lon_m: float | None,
    ego_route_error_m: float | None = None,
) -> bool:
    """Keep an overtake active through lateral separation until the pass."""
    family = str(scenario_family).lower()
    if "obstruction" not in family and "narrow" not in family:
        return bool(observed)
    if not bool(actor_present):
        return False
    if (
        actor_lon_m is not None
        and float(actor_lon_m) <= -2.0
        and ego_route_error_m is not None
        and float(ego_route_error_m) <= 0.60
    ):
        return False
    return bool(previous or observed)


def _v3_cut_in_conflict_active(
    previous: bool,
    *,
    completed: bool,
    observed: bool,
    actor_present: bool,
    actor_lon_m: float | None,
    actor_lat_m: float | None,
    initial_side: str,
    ego_route_error_m: float,
    clear_lateral_m: float = 4.5,
) -> bool:
    """Keep one cut-in avoidance direction until the actor traverses the lane.

    A current-state detector can briefly clear while the scripted actor is just
    outside its lateral threshold, then become true again after it has crossed
    the ego centreline.  Treating that second observation as a new cut-in makes
    the candidate reverse avoidance direction.  The persistent interaction
    state instead completes after a full side-to-side traversal (or a passed,
    rejoined actor) and cannot reactivate during the same fixture.
    """
    if completed:
        return False
    if not previous:
        return bool(observed)
    if not actor_present:
        return False
    if (
        actor_lon_m is not None
        and float(actor_lon_m) <= -2.0
        and abs(float(ego_route_error_m)) <= 0.60
    ):
        return False
    if actor_lat_m is None:
        return True
    side = str(initial_side or "none").lower()
    lateral = float(actor_lat_m)
    threshold = abs(float(clear_lateral_m))
    if side == "left" and lateral <= -threshold:
        return False
    if side == "right" and lateral >= threshold:
        return False
    return True


def _v3_overtake_phase(
    previous: str,
    *,
    actor_present: bool,
    actor_lon_m: float | None,
    ego_route_error_m: float,
) -> str:
    """Persistent observable overtake phase across VLA replans."""
    prior = str(previous or "DEPART").upper()
    if not actor_present:
        return "COMPLETE"
    passed = actor_lon_m is not None and float(actor_lon_m) <= -2.0
    rejoined = abs(float(ego_route_error_m)) <= 0.60
    if prior == "REJOIN":
        return "COMPLETE" if passed and rejoined else "REJOIN"
    if passed:
        return "REJOIN"
    if prior == "PASS" or abs(float(ego_route_error_m)) >= 2.50:
        return "PASS"
    return "DEPART"


def _v3_crossing_conflict_active(
    previous: bool,
    *,
    completed: bool = False,
    observed: bool,
    actor_lat_m: float | None,
    initial_side: str,
) -> bool:
    """Latch a crossing until the actor has traversed the ego corridor."""
    if completed:
        return False
    if not previous:
        return bool(observed)
    if actor_lat_m is None:
        return False
    side = str(initial_side or "none").lower()
    lateral = float(actor_lat_m)
    if side == "left" and lateral <= -8.0:
        return False
    if side == "right" and lateral >= 8.0:
        return False
    return True


def _v3_authorized_lane_corridor(
    route_context: Any,
    *,
    ego_x: float,
    ego_y: float,
    alternative_kind: str,
    target_lane_side: str,
) -> bool:
    """Prove that a legal lane transition remains inside its two-lane union."""
    maneuver = str(route_context.maneuver.value)
    is_turn = maneuver in {"TURN_LEFT", "TURN_RIGHT"}
    is_route_change = maneuver in {
        "ROUTE_CHANGE_LEFT",
        "ROUTE_CHANGE_RIGHT",
    }
    is_overtake = str(alternative_kind) == "SPATIAL_OVERTAKE"
    if is_turn:
        # CARLA can emit a lane-invasion event for the transverse end-cap of
        # a junction lane after the ego enters its registered exit.  Only a
        # tightly route-bound turn is authorized here; wrong exits and true
        # off-road motion remain independently fatal.
        return (
            route_projection(
                tuple(route_context.route_xy),
                float(ego_x),
                float(ego_y),
            )[1]
            <= 0.60 + 1.0e-6
        )
    if not (is_route_change or is_overtake):
        return False
    side_value = str(target_lane_side or "NONE")
    if side_value == "NONE" and is_route_change:
        side_value = (
            "LEFT" if maneuver == "ROUTE_CHANGE_LEFT" else "RIGHT"
        )
    try:
        from driving_vla.model.navigation_contract import TargetLaneSide

        lane = route_context.lane(TargetLaneSide(side_value))
    except (ValueError, AttributeError):
        return False
    paths = [
        tuple(route_context.route_xy),
        tuple(route_context.origin_lane_centerline_xy),
        tuple(lane.centerline_xy),
    ]
    distances = [
        route_projection(path, float(ego_x), float(ego_y))[1]
        for path in paths
        if len(path) >= 2
    ]
    if not distances:
        return False
    corridor_radius = (
        0.5
        * max(
            2.5,
            float(route_context.origin_lane_width_m),
            float(lane.lane_width_m),
        )
        + 0.30
    )
    return min(distances) <= corridor_radius + 1.0e-6


def _v3_native_route_change_commitment(
    route_context: Any,
    spatial_path_xy: Any,
    *,
    target_error_max_m: float | None = None,
) -> dict[str, Any]:
    """Decide whether one Guard-approved native path completes a lane change.

    Once such a path is committed, the live runner tracks that exact native
    ``pred_route`` instead of repeatedly asking the model to perform another
    lane change while the first one is still underway.  No map path is emitted
    here; target/origin centerlines are used only to audit the native endpoint.
    """

    maneuver = str(getattr(route_context.maneuver, "value", route_context.maneuver))
    if maneuver not in {"ROUTE_CHANGE_LEFT", "ROUTE_CHANGE_RIGHT"}:
        return {"ready": False, "reason": "NOT_ROUTE_CHANGE"}
    path = tuple((float(point[0]), float(point[1])) for point in spatial_path_xy)
    if len(path) < 2:
        return {"ready": False, "reason": "PATH_TOO_SHORT"}
    from driving_vla.model.navigation_contract import TargetLaneSide

    side = (
        TargetLaneSide.LEFT
        if maneuver == "ROUTE_CHANGE_LEFT"
        else TargetLaneSide.RIGHT
    )
    lane = route_context.lane(side)
    if len(lane.centerline_xy) < 2:
        return {"ready": False, "reason": "TARGET_LANE_MISSING"}
    endpoint = path[-1]
    target_error = float(
        route_projection(lane.centerline_xy, endpoint[0], endpoint[1])[1]
    )
    effective_target_error_max_m = (
        0.5 * max(2.5, float(lane.lane_width_m))
        if target_error_max_m is None
        else float(target_error_max_m)
    )
    origin_paths = (
        route_context.origin_lane_centerline_xy,
        route_context.route_xy,
    )
    origin_errors = [
        float(route_projection(polyline, endpoint[0], endpoint[1])[1])
        for polyline in origin_paths
        if len(polyline) >= 2
    ]
    origin_error = min(origin_errors, default=float("inf"))
    ready = bool(
        target_error <= effective_target_error_max_m + 1.0e-6
        and target_error + 1.0e-6 < origin_error
    )
    return {
        "ready": ready,
        "reason": "TARGET_LANE_ENDPOINT" if ready else "ENDPOINT_NOT_COMMITTED",
        "target_lane_side": side.value,
        "target_error_m": target_error,
        "origin_error_m": origin_error,
        "target_error_max_m": effective_target_error_max_m,
        "endpoint_xy": [endpoint[0], endpoint[1]],
    }


DEFAULT_COLLISION_EPISODE_GAP_S = 0.50
# Live E0 2026-07-19: DX12 eliminates CUDA-forward↔D3D11 hang on this host.
DEFAULT_RHI = "dx12"


def _road_surface_diagnosis(
    carla_map: Any,
    location: Any,
    ego_yaw: float,
    *,
    carla_module: Any,
) -> dict[str, Any]:
    """Evidence-only road/surface classification (never used as control centerline)."""
    diag: dict[str, Any] = {
        "on_driving": False,
        "lane_type": None,
        "lane_type_name": None,
        "is_junction": None,
        "distance_to_driving_m": None,
        "road_heading_deg": None,
        "heading_error_deg": None,
        "surface_class": "unknown",
        "flags": [],
    }
    try:
        any_wp = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla_module.LaneType.Any,
        )
        drive_wp = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
        if any_wp is not None:
            lt = int(any_wp.lane_type)
            diag["lane_type"] = lt
            # CARLA LaneType is a bitfield; name best-effort via enum members.
            names = []
            for name in (
                "Driving",
                "Shoulder",
                "Sidewalk",
                "Parking",
                "Bidirectional",
                "Median",
                "Border",
                "Restricted",
                "Parking",
                "Bidirectional",
                "Biking",
                "Shoulder",
                "Tram",
                "Rail",
                "OnRamp",
                "OffRamp",
                "Entry",
                "Exit",
                "OffRamp",
                "ConnectingRamp",
            ):
                try:
                    flag = int(getattr(carla_module.LaneType, name))
                except AttributeError:
                    continue
                if lt & flag:
                    names.append(name)
            diag["lane_type_name"] = "|".join(names) if names else str(lt)
            diag["is_junction"] = bool(getattr(any_wp, "is_junction", False))
        if drive_wp is not None:
            dx = float(location.x) - float(drive_wp.transform.location.x)
            dy = float(location.y) - float(drive_wp.transform.location.y)
            dist = math.hypot(dx, dy)
            diag["distance_to_driving_m"] = dist
            road_yaw = math.radians(float(drive_wp.transform.rotation.yaw))
            diag["road_heading_deg"] = math.degrees(road_yaw)
            err = abs(math.degrees(math.atan2(math.sin(ego_yaw - road_yaw), math.cos(ego_yaw - road_yaw))))
            diag["heading_error_deg"] = err
            diag["on_driving"] = dist <= 2.0
        flags: list[str] = []
        name = str(diag.get("lane_type_name") or "")
        if not diag["on_driving"]:
            flags.append("away_from_driving")
        if "Sidewalk" in name:
            flags.append("sidewalk")
        if "Shoulder" in name or "Border" in name or "Median" in name:
            flags.append("non_driving_lane")
        if "Parking" in name:
            flags.append("parking")
        if "Biking" in name:
            flags.append("biking")
        # Vegetation / grass often reports as None driving + far distance without sidewalk bit.
        dist = diag.get("distance_to_driving_m")
        if dist is not None and float(dist) > 3.5 and "Sidewalk" not in name:
            flags.append("likely_greenbelt_or_terrain")
        if not flags and diag["on_driving"]:
            diag["surface_class"] = "driving"
        elif "sidewalk" in flags:
            diag["surface_class"] = "sidewalk"
        elif "likely_greenbelt_or_terrain" in flags:
            diag["surface_class"] = "greenbelt_or_terrain"
        elif flags:
            diag["surface_class"] = "non_driving_lane"
        else:
            diag["surface_class"] = "unknown"
        diag["flags"] = flags
    except Exception as exc:  # pragma: no cover - live CARLA only
        diag["error"] = f"{type(exc).__name__}: {exc}"
    return diag


def _path_xy_as_lists(path: Any) -> list[list[float]] | None:
    if path is None:
        return None
    if isinstance(path, (list, tuple)):
        out: list[list[float]] = []
        for point in path:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                out.append([float(point[0]), float(point[1])])
        return out
    xs = getattr(path, "x", None)
    ys = getattr(path, "y", None)
    if xs is not None and ys is not None:
        try:
            return [
                [float(x), float(y)]
                for x, y in zip(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
            ]
        except (TypeError, ValueError):
            return None
    return None


class CollisionEpisodeBook:
    """Merge sustained contact into collision episodes with pre-impact snapshots."""

    def __init__(self, *, gap_s: float = DEFAULT_COLLISION_EPISODE_GAP_S) -> None:
        self.gap_s = float(gap_s)
        self.raw_event_count = 0
        self.total_impulse = 0.0
        self.episodes: list[dict[str, Any]] = []
        self._open: dict[str, Any] | None = None
        self._current_sim_s = 0.0
        self._latest_snapshot: dict[str, Any] = {}
        self.lock = threading.Lock()

    @property
    def episode_count(self) -> int:
        return len(self.episodes) + (1 if self._open is not None else 0)

    def set_sim_s(self, sim_s: float) -> None:
        with self.lock:
            self._current_sim_s = float(sim_s)

    def update_control_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Store the most recent control/path state for the next collision."""
        with self.lock:
            self._latest_snapshot = dict(snapshot)

    def _close_open_unlocked(self) -> None:
        if self._open is not None:
            self.episodes.append(self._open)
            self._open = None

    def on_collision(
        self,
        *,
        impulse: float,
        other_type: str | None,
        other_id: int | None,
        vehicle_xy: tuple[float, float] | None,
        vehicle_yaw: float | None = None,
        sim_s: float | None = None,
    ) -> dict[str, Any]:
        """Register one raw contact; returns the active episode dict."""
        with self.lock:
            stamp = float(self._current_sim_s if sim_s is None else sim_s)
            magnitude = float(impulse)
            self.raw_event_count += 1
            self.total_impulse += magnitude
            other = str(other_type or "unknown")
            oid = None if other_id is None else int(other_id)
            pose = {
                "x": None if vehicle_xy is None else float(vehicle_xy[0]),
                "y": None if vehicle_xy is None else float(vehicle_xy[1]),
                "yaw": None if vehicle_yaw is None else float(vehicle_yaw),
            }
            continuous = (
                self._open is not None
                and self._open.get("last_sim_s") is not None
                and (stamp - float(self._open["last_sim_s"])) <= self.gap_s
            )
            if continuous:
                episode = self._open
                assert episode is not None
                episode["raw_event_count"] = int(episode["raw_event_count"]) + 1
                episode["impulse_sum"] = float(episode["impulse_sum"]) + magnitude
                episode["impulse_max"] = max(float(episode["impulse_max"]), magnitude)
                episode["last_sim_s"] = stamp
                episode["duration_s"] = stamp - float(episode["first_sim_s"])
                if other and other not in episode["other_types"]:
                    episode["other_types"].append(other)
                if oid is not None and oid not in episode["other_ids"]:
                    episode["other_ids"].append(oid)
            else:
                self._close_open_unlocked()
                episode = {
                    "episode_index": len(self.episodes),
                    "first_sim_s": stamp,
                    "last_sim_s": stamp,
                    "duration_s": 0.0,
                    "raw_event_count": 1,
                    "impulse_sum": magnitude,
                    "impulse_max": magnitude,
                    "other_type": other,
                    "other_types": [other] if other else [],
                    "other_ids": [] if oid is None else [oid],
                    "vehicle_pose_first": pose,
                    "pre_collision": dict(self._latest_snapshot),
                }
                self._open = episode
            return dict(episode)

    def finalize(self) -> list[dict[str, Any]]:
        with self.lock:
            self._close_open_unlocked()
            return list(self.episodes)

    def summary(self) -> dict[str, Any]:
        episodes = self.finalize()
        first = episodes[0] if episodes else None
        return {
            "episode_count": len(episodes),
            "raw_event_count": self.raw_event_count,
            "total_impulse": self.total_impulse,
            "gap_s": self.gap_s,
            "first_collision_sim_s": None if first is None else first.get("first_sim_s"),
            "first_collision_other_type": None if first is None else first.get("other_type"),
            "first_collision_vehicle_pose": None
            if first is None
            else first.get("vehicle_pose_first"),
            "episodes": episodes,
        }


def _tail_motion_metrics(
    speed_samples_mps: list[float], *, sim_dt_s: float, window_s: float = 20.0
) -> dict[str, float]:
    """Measure whether an endurance run merely survived after becoming parked."""
    if not speed_samples_mps:
        return {"window_s": 0.0, "mean_speed_mps": 0.0, "moving_fraction": 0.0}
    count = max(1, int(math.ceil(float(window_s) / max(float(sim_dt_s), 1e-6))))
    tail = np.asarray(speed_samples_mps[-count:], dtype=float)
    return {
        "window_s": float(len(tail) * float(sim_dt_s)),
        "mean_speed_mps": float(np.mean(tail)),
        "moving_fraction": float(np.mean(tail >= 0.5)),
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _xml_local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _xml_find_text_recursive(root: ET.Element, wanted: str) -> str:
    """Match nested tags by local name; supports dotted Unreal names (RHI.RHIName)."""

    for element in root.iter():
        if _xml_local_tag(element.tag) == wanted:
            return (element.text or "").strip()
    # ElementTree XPath treats '.' specially; still try for simple undotted tags.
    if "." not in wanted:
        text = root.findtext(f".//{wanted}")
        if text is not None:
            return text.strip()
    return ""


def _capture_recent_carla_crash(
    evidence_dir: Path,
    *,
    run_wall_start_s: float,
    crash_root: Path = Path("/mnt/c/Users"),
) -> dict[str, Any] | None:
    """Copy and summarize only a CrashContext created during this run."""
    try:
        candidates = [
            path
            for path in crash_root.glob(
                "*/AppData/Local/CarlaUE4/Saved/Crashes/*/CrashContext.runtime-xml"
            )
            if path.stat().st_mtime >= float(run_wall_start_s) - 2.0
        ]
        if not candidates:
            return None
        source = max(candidates, key=lambda path: path.stat().st_mtime)
        destination = evidence_dir / "carla_crash_context_latest.xml"
        shutil.copy2(source, destination)
        root = ET.parse(source).getroot()
        wanted = (
            "CrashType",
            "ErrorMessage",
            "SecondsSinceStart",
            "Misc.PrimaryGPUBrand",
            "MemoryStats.AvailablePhysical",
            "MemoryStats.PeakUsedPhysical",
            "MemoryStats.bIsOOM",
            "RHI.RHIName",
            "RHI.AdapterName",
            "RHI.UserDriverVersion",
            "RHI.InternalDriverVersion",
            "RHI.DriverDate",
        )
        fields = {tag: _xml_find_text_recursive(root, tag) for tag in wanted}
        return {
            "captured": True,
            "source_mtime_s": source.stat().st_mtime,
            "source_path": str(source),
            "copied_file": destination.name,
            "fields": fields,
        }
    except (OSError, ET.ParseError) as exc:
        return {"captured": False, "capture_error": str(exc)}


def _classify_runtime_failure(
    error_text: str, crash_context: dict[str, Any] | None
) -> str:
    fields = crash_context.get("fields", {}) if isinstance(crash_context, dict) else {}
    crash_type = str(fields.get("CrashType", "") or "")
    crash_message = str(fields.get("ErrorMessage", "") or "")
    if crash_type == "GPUCrash":
        return "CARLA_GPU_CRASH"
    if "D3D" in crash_message.upper() or "DXGI_" in crash_message.upper():
        return "CARLA_D3D_CRASH"
    if crash_type == "Assert" and (
        "D3D" in crash_message.upper() or "DXGI_" in crash_message.upper() or "D3D11" in crash_message.upper()
    ):
        return "CARLA_D3D_CRASH"
    lowered = error_text.lower()
    tick_timeout = "time-out" in lowered and "simulator" in lowered
    has_context = bool(
        isinstance(crash_context, dict)
        and crash_context.get("captured")
        and (crash_type or crash_message)
    )
    if tick_timeout and not has_context:
        return "CARLA_SERVER_HANG_NO_CRASH_CONTEXT"
    if tick_timeout:
        return "CARLA_SERVER_UNRESPONSIVE"
    return "CLIENT_OR_MODEL_RUNTIME_ERROR"


def _query_driver_version() -> str | None:
    """Best-effort NVIDIA driver version from nvidia-smi (WSL or Windows)."""

    candidates = [
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        ["nvidia-smi.exe", "--query-gpu=driver_version", "--format=csv,noheader"],
    ]
    for command in candidates:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip().splitlines()[0].strip()
    return None


def _capture_carla_log_tail(
    evidence_dir: Path,
    *,
    max_lines: int = 80,
) -> dict[str, Any] | None:
    """Copy the newest CarlaUE4 log and keep a short tail in evidence."""

    search_roots = [
        Path("/mnt/c/Users"),
        Path("/mnt/e/CARLA_0.9.16"),
        Path("E:/CARLA_0.9.16"),
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        try:
            if root.name == "Users":
                candidates.extend(root.glob("*/AppData/Local/CarlaUE4/Saved/Logs/*.log"))
            else:
                candidates.extend(root.glob("**/Saved/Logs/*.log"))
                candidates.extend(root.glob("CarlaUE4/Saved/Logs/*.log"))
        except OSError:
            continue
    if not candidates:
        return None
    try:
        source = max(candidates, key=lambda path: path.stat().st_mtime)
        text = source.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        tail = lines[-max_lines:] if lines else []
        destination = evidence_dir / "carla_server_log_tail.txt"
        destination.write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")
        return {
            "source_path": str(source),
            "source_mtime_s": source.stat().st_mtime,
            "copied_file": destination.name,
            "tail_line_count": len(tail),
            "tail": tail,
        }
    except OSError as exc:
        return {"captured": False, "capture_error": str(exc)}


def _probe_server_health(resolver: ConnectionResolver) -> dict[str, Any]:
    """Snapshot process_state + RPC reachability without claiming READY for experiments."""

    try:
        report = resolver.preflight(retry_host=False)
        return {
            "process_state": report.process_state,
            "status": report.status,
            "tcp_reachable": report.tcp_reachable,
            "rpc_reachable": report.rpc_reachable,
            "error_code": report.error_code,
            "error_message": report.error_message,
            "map": report.map,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "process_state": "UNKNOWN",
            "status": "PROBE_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _polyline_s(points: list[tuple[float, float]]) -> np.ndarray:
    xy = np.asarray(points, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))))


def _point_at_s(points: list[tuple[float, float]], s: np.ndarray, query: float) -> tuple[float, float]:
    query = float(np.clip(query, 0.0, float(s[-1])))
    return float(np.interp(query, s, [p[0] for p in points])), float(
        np.interp(query, s, [p[1] for p in points])
    )


def _project_route_s(
    points: list[tuple[float, float]], s: np.ndarray, x: float, y: float, hint_s: float
) -> float:
    best_s, best_d2 = hint_s, float("inf")
    lo, hi = max(0.0, hint_s - 2.0), min(float(s[-1]), hint_s + 40.0)
    for i in range(len(points) - 1):
        if s[i + 1] < lo or s[i] > hi:
            continue
        a = np.asarray(points[i], dtype=float)
        b = np.asarray(points[i + 1], dtype=float)
        p = np.asarray([x, y], dtype=float)
        ab = b - a
        denom = float(ab @ ab)
        u = 0.0 if denom < 1e-12 else float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        q = a + u * ab
        d2 = float((p - q) @ (p - q))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(s[i] + u * (s[i + 1] - s[i]))
    return max(hint_s, best_s)


def _navigation_targets(
    route: list[tuple[float, float]],
    route_s: np.ndarray,
    ego: EgoPose,
    progress_s: float,
    *,
    contract: SimLingoContractConfig | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float, bool]:
    """Coarse VLA targets. Default: official RoutePlanner [1]/[2]; legacy: +15/+30 m."""
    cfg = contract or SimLingoContractConfig()
    result = contract_navigation_targets(
        route,
        ego_x=ego.x,
        ego_y=ego.y,
        ego_yaw=ego.yaw,
        progress_hint_s=progress_s,
        config=cfg,
    )
    return (
        result.target_ego_1,
        result.target_ego_2,
        result.progress_s,
        result.valid,
    )


def _navigation_targets_full(
    route: list[tuple[float, float]],
    ego: EgoPose,
    progress_s: float,
    *,
    contract: SimLingoContractConfig,
):
    return contract_navigation_targets(
        route,
        ego_x=ego.x,
        ego_y=ego.y,
        ego_yaw=ego.yaw,
        progress_hint_s=progress_s,
        config=contract,
    )


def _build_route_segment(
    world: carla.World,
    start: carla.Transform,
    length_m: float,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    points = [(float(x), float(y)) for x, y in build_route(world, start, length_m)]
    if len(points) < 2:
        raise RuntimeError("route builder returned fewer than two points")
    cumulative = _polyline_s(points)
    if float(cumulative[-1]) < 35.0:
        raise RuntimeError(f"route segment is only {float(cumulative[-1]):.1f}m")
    return points, cumulative


def _densify_registered_route_v3(
    world_map: Any,
    route_xyz: Any,
    *,
    step_m: float = 1.0,
) -> list[tuple[float, float]]:
    """Follow CARLA topology between frozen coarse route anchors.

    This refines navigation/topology only.  It is never passed to PathManager
    as executable geometry; SimLingo ``pred_route`` remains the tracked path.
    """
    anchors = [
        (float(point[0]), float(point[1]), float(point[2]))
        for point in route_xyz
    ]
    if len(anchors) < 2:
        raise ValueError("registered V3 route needs at least two anchors")
    output: list[tuple[float, float]] = []
    for segment_index, (start, target) in enumerate(
        zip(anchors, anchors[1:])
    ):
        chord = math.hypot(target[0] - start[0], target[1] - start[1])
        samples = max(2, int(math.ceil(chord / float(step_m))) + 1)
        for index in range(samples):
            alpha = index / max(samples - 1, 1)
            query = carla.Location(
                x=start[0] + alpha * (target[0] - start[0]),
                y=start[1] + alpha * (target[1] - start[1]),
                z=start[2] + alpha * (target[2] - start[2]),
            )
            waypoint = world_map.get_waypoint(
                query,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                raise RuntimeError(
                    f"registered route segment {segment_index} has no driving projection"
                )
            location = waypoint.transform.location
            point = (float(location.x), float(location.y))
            if not output or math.hypot(
                point[0] - output[-1][0], point[1] - output[-1][1]
            ) > 0.10:
                output.append(point)
    final = anchors[-1]
    output.append((final[0], final[1]))
    if float(_polyline_s(output)[-1]) < 35.0:
        raise RuntimeError("densified registered route is shorter than 35 m")
    return output


def _ego_pose(vehicle: carla.Vehicle) -> EgoPose:
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    return EgoPose(
        x=float(tf.location.x),
        y=float(tf.location.y),
        yaw=math.radians(float(tf.rotation.yaw)),
        speed_mps=math.hypot(float(vel.x), float(vel.y)),
    )


def _draw_polyline_map(
    world: carla.World,
    points_xy: list[tuple[float, float]] | None,
    *,
    color: carla.Color,
    z: float,
    thickness: float,
    life_s: float,
) -> None:
    if not points_xy or len(points_xy) < 2:
        return
    debug = world.debug
    for a, b in zip(points_xy, points_xy[1:]):
        debug.draw_line(
            carla.Location(float(a[0]), float(a[1]), z),
            carla.Location(float(b[0]), float(b[1]), z),
            thickness=thickness,
            color=color,
            life_time=float(life_s),
        )
    tip = points_xy[-1]
    debug.draw_point(
        carla.Location(float(tip[0]), float(tip[1]), z + 0.15),
        size=0.16,
        color=color,
        life_time=float(life_s),
    )


def _draw_path(
    world: carla.World,
    raw: SpatialPath | None,
    committed: SpatialPath | None,
    *,
    life_s: float,
    speed_wps_map_xy: list[tuple[float, float]] | None = None,
) -> None:
    """Green = raw pred_route; yellow = committed for MPC; cyan = pred_speed_wps.

    Drawn only on VLA updates (caller cadence).
    """
    debug = world.debug
    layers: list[tuple[SpatialPath | None, carla.Color, float, float]] = [
        (raw, carla.Color(0, 255, 0), 0.45, 0.12),
        (committed, carla.Color(255, 220, 0), 0.75, 0.20),
    ]
    for path, color, z, thickness in layers:
        if path is None or path.s.size < 2:
            continue
        stride = max(1, int(round(0.35 / max(float(np.median(np.diff(path.s))), 0.1))))
        points = path.as_xy()[::stride]
        if len(points) < 2:
            points = path.as_xy()
        _draw_polyline_map(
            world, points, color=color, z=z, thickness=thickness, life_s=life_s
        )
    if speed_wps_map_xy:
        # Cyan: speed head only — not tracked geometry.
        _draw_polyline_map(
            world,
            speed_wps_map_xy,
            color=carla.Color(0, 220, 255),
            z=0.55,
            thickness=0.10,
            life_s=life_s,
        )


def _resolve_ego_blueprint(
    world: carla.World, blueprint_id: str
) -> tuple[Any, str]:
    """Find ego blueprint or raise — never silent Audi/Tesla fallback."""
    requested = str(blueprint_id or DEFAULT_EGO_BLUEPRINT).strip()
    if not requested:
        requested = DEFAULT_EGO_BLUEPRINT
    library = world.get_blueprint_library()
    blueprint = None
    try:
        blueprint = library.find(requested)
    except Exception:
        blueprint = None
    if blueprint is None:
        # Some CARLA builds use IndexError; also try filter match.
        try:
            matches = list(library.filter(requested))
            if matches:
                blueprint = matches[0]
        except Exception:
            blueprint = None
    if blueprint is None:
        raise RuntimeError(
            f"ego vehicle blueprint not found: {requested!r}; "
            "refusing silent fallback to Audi/Tesla. "
            "Pass an existing --vehicle-blueprint or install the blueprint pack."
        )
    effective = str(getattr(blueprint, "id", requested) or requested)
    return blueprint, effective


def _spawn_ego(
    world: carla.World,
    *,
    large_map: bool,
    vehicle_blueprint: str = DEFAULT_EGO_BLUEPRINT,
) -> tuple[carla.Vehicle, carla.Transform, str]:
    spawn = free_spawn(world)
    blueprint, effective_bp = _resolve_ego_blueprint(world, vehicle_blueprint)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    if large_map:
        # A spectator close to the chosen spawn asks the server to stream that
        # tile before the hero exists.  Without this, Town13 may spawn against
        # an unloaded tile and crash inside Unreal with address 0x8.
        spectator_tf = carla.Transform(
            carla.Location(spawn.location.x - 8.0, spawn.location.y, spawn.location.z + 8.0),
            carla.Rotation(pitch=-25.0, yaw=spawn.rotation.yaw),
        )
        world.get_spectator().set_transform(spectator_tf)
        for _ in range(30):
            world.tick()
    for transform in [spawn, *world.get_map().get_spawn_points()[:40]]:
        lifted = carla.Transform(
            carla.Location(transform.location.x, transform.location.y, transform.location.z + 0.5),
            transform.rotation,
        )
        actor = world.try_spawn_actor(blueprint, lifted)
        if actor is not None:
            return actor, lifted, effective_bp
    raise RuntimeError(f"unable to spawn ego vehicle blueprint={effective_bp!r}")


def _spawn_ego_exact_v3(
    world: carla.World,
    spawn_value: dict[str, Any],
    *,
    vehicle_blueprint: str,
) -> tuple[carla.Vehicle, carla.Transform, str]:
    """Spawn a route-fixture ego exactly; never fall back to a free spawn."""
    blueprint, effective_bp = _resolve_ego_blueprint(world, vehicle_blueprint)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    transform = carla.Transform(
        carla.Location(
            x=float(spawn_value["x"]),
            y=float(spawn_value["y"]),
            z=float(spawn_value["z"]),
        ),
        carla.Rotation(
            roll=float(spawn_value.get("roll_deg", 0.0)),
            pitch=float(spawn_value.get("pitch_deg", 0.0)),
            yaw=float(spawn_value["yaw_deg"]),
        ),
    )
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        raise RuntimeError(
            "exact V3 route-fixture ego spawn failed; refusing free-spawn fallback"
        )
    return actor, transform, effective_bp


def _load_v3_route_fixture(
    path: str,
    *,
    maneuver: str,
    requested_map: str,
):
    from driving_vla.model.navigation_contract import (
        RouteContextV3,
        RouteManeuver,
        canonical_sha256,
    )

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "safedrive.r2_v3.route_fixtures.v1":
        raise ValueError("unsupported V3 route fixture schema")
    expected_hash = str(value.get("manifest_hash") or "")
    body = {key: item for key, item in value.items() if key != "manifest_hash"}
    if not expected_hash or canonical_sha256(body) != expected_hash:
        raise ValueError("V3 route fixture manifest hash mismatch")
    fixture_map = str(value.get("map_name") or "")
    if fixture_map != str(requested_map):
        raise ValueError(
            f"V3 route fixture map mismatch: {fixture_map!r} != {requested_map!r}"
        )
    selected = RouteManeuver(str(maneuver))
    raw = dict((value.get("routes") or {}).get(selected.value) or {})
    if not raw:
        raise ValueError(f"V3 route fixture missing {selected.value}")
    context = RouteContextV3.from_mapping(raw.get("route_context") or {})
    if context.maneuver is not selected:
        raise ValueError("V3 route fixture maneuver mismatch")
    spawn = dict(raw.get("ego_spawn_transform") or {})
    if not {"x", "y", "z", "yaw_deg"}.issubset(spawn):
        raise ValueError("V3 route fixture missing exact ego spawn transform")
    return context, spawn, expected_hash


def _map_content_path(map_name: str) -> str:
    """Unreal content path without the trailing asset name suffix.

    Flat towns:  /Game/Carla/Maps/Town04
    Nested:      /Game/Carla/Maps/Town13/Town13
    """
    if map_name in NESTED_MAP_PACKAGE or (CARLA_MAPS_CONTENT / map_name / f"{map_name}.umap").is_file():
        return f"/Game/Carla/Maps/{map_name}/{map_name}"
    return f"/Game/Carla/Maps/{map_name}"


def _map_asset_path(map_name: str) -> str:
    """Full DefaultEngine map asset id: /Game/.../Town13.Town13"""
    content = _map_content_path(map_name)
    return f"{content}.{map_name}"


def _write_startup_map(
    map_name: str,
    *,
    server_res_x: int,
    server_res_y: int,
    rhi: str = DEFAULT_RHI,
    render_offscreen: bool = False,
) -> str:
    """Pin CARLA process-start map/RHI via shared runtime helper + carla_start.toml.

    Mid-session client.load_world is intentionally avoided (D3D/shader risk).
    Default launch mode is manual-compatible (DefaultEngine.ini); explicit
    arguments remain available for regression. Returns the normalized arguments
    string for evidence only.
    """
    from runtime.carla_engine_config import map_content_path, pin_default_engine_config

    if map_name not in DEMO_MAP_POOL:
        raise ValueError(f"unsupported map {map_name!r}; pool={DEMO_MAP_POOL}")
    rhi_norm = normalize_rhi(rhi)
    content = map_content_path(map_name)
    umap = CARLA_MAPS_CONTENT / map_name / f"{map_name}.umap"
    if not umap.is_file():
        umap = CARLA_MAPS_CONTENT / f"{map_name}.umap"
    if not umap.is_file():
        raise FileNotFoundError(f"map package missing on disk for {map_name}: expected under {CARLA_MAPS_CONTENT}")
    arguments = build_carla_launch_arguments(
        map_content=content,
        res_x=int(server_res_x),
        res_y=int(server_res_y),
        quality_level="Low",
        rhi=rhi_norm,
        render_offscreen=bool(render_offscreen),
        rpc_port=2000,
        windowed=True,
        nosound=True,
    )
    pin_result = pin_default_engine_config(
        requested_map=map_name,
        requested_rhi=rhi_norm,
        path=CARLA_DEFAULT_ENGINE_INI if CARLA_DEFAULT_ENGINE_INI.is_file() else None,
    )
    if not pin_result.get("ok"):
        raise RuntimeError(
            f"DefaultEngine pin failed: {pin_result.get('error_code')} {pin_result.get('message')}"
        )
    if CARLA_START_TOML.is_file():
        toml = CARLA_START_TOML.read_text(encoding="utf-8", errors="replace")
        args_line = f'arguments = "{arguments}"'
        rhi_line = f'rhi = "{rhi_norm}"'
        off_line = f"render_offscreen = {'true' if render_offscreen else 'false'}"
        map_line = f'default_map = "{map_name}"'
        if re.search(r'(?m)^arguments\s*=', toml):
            toml = re.sub(r'(?m)^arguments\s*=\s*".*"$', args_line, toml)
        else:
            toml = toml.rstrip() + "\n" + args_line + "\n"
        if re.search(r'(?m)^rhi\s*=', toml):
            toml = re.sub(r'(?m)^rhi\s*=\s*".*"$', rhi_line, toml)
        else:
            toml = toml.rstrip() + "\n" + rhi_line + "\n"
        if re.search(r'(?m)^render_offscreen\s*=', toml):
            toml = re.sub(r'(?m)^render_offscreen\s*=\s*\S+$', off_line, toml)
        else:
            toml = toml.rstrip() + "\n" + off_line + "\n"
        if re.search(r'(?m)^default_map\s*=', toml):
            toml = re.sub(r'(?m)^default_map\s*=\s*".*"$', map_line, toml)
        else:
            toml = toml.rstrip() + "\n" + map_line + "\n"
        CARLA_START_TOML.write_text(toml, encoding="utf-8")
    print(
        f"startup pinned: map={map_name} rhi={rhi_norm} offscreen={bool(render_offscreen)} "
        f"args={arguments} engine_written={pin_result.get('written')}",
        flush=True,
    )
    return arguments


def _kill_carla_windows() -> None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        print("WARN: powershell not found; cannot stop CarlaUE4 from WSL", flush=True)
        return
    cmd = (
        "Get-Process -Name 'CarlaUE4*','UE4Editor*' -ErrorAction SilentlyContinue "
        "| Stop-Process -Force -ErrorAction SilentlyContinue"
    )
    try:
        subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", cmd],
            timeout=20.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"WARN: Carla stop failed: {exc}", flush=True)
    time.sleep(4.0)


def _map_matches(current_map: str, required: str) -> bool:
    return required in current_map or current_map.endswith(required)


def _resolve_requested_map(map_arg: str, *, seed: int | None) -> str:
    token = str(map_arg).strip()
    if token.lower() in {"random", "rand", "*"}:
        rng = random.Random(seed) if seed is not None else random.SystemRandom()
        choice = rng.choice(list(DEFAULT_RANDOM_MAP_POOL))
        print(
            f"random stable map selected: {choice} from {len(DEFAULT_RANDOM_MAP_POOL)} towns",
            flush=True,
        )
        return choice
    if token not in DEMO_MAP_POOL:
        raise SystemExit(
            f"unknown map {token!r}; use one of {list(DEMO_MAP_POOL)} or --map random"
        )
    return token


def _ensure_ready_map(
    resolver: ConnectionResolver,
    map_name: str,
    *,
    startup_timeout_s: float,
    server_res_x: int,
    server_res_y: int,
    force_cold_start: bool,
    rhi: str = DEFAULT_RHI,
    render_offscreen: bool = False,
) -> Any:
    rhi_norm = normalize_rhi(rhi)
    report = resolver.preflight()
    cmdline = query_windows_carla_command_line()
    launch_match = launch_params_match(
        cmdline, rhi=rhi_norm, render_offscreen=bool(render_offscreen)
    )
    map_ok = report.status == "READY" and _map_matches(str(report.map or ""), map_name)
    launch_ok = launch_match is not False
    if not force_cold_start and map_ok and launch_ok and report.status == "READY":
        if launch_match is None:
            print(
                f"preflight READY on map={report.map}; live cmdline RHI unknown — "
                f"assuming compatible (request rhi={rhi_norm} offscreen={bool(render_offscreen)})",
                flush=True,
            )
        else:
            print(
                f"preflight READY on required map={report.map} rhi={rhi_norm} "
                f"offscreen={bool(render_offscreen)}",
                flush=True,
            )
        if hasattr(report, "details"):
            report.details = dict(report.details or {})
            report.details["requested_rhi"] = rhi_norm
            report.details["effective_rhi"] = parse_rhi_from_command_line(cmdline or "") or rhi_norm
            report.details["render_offscreen"] = bool(render_offscreen)
            report.details["server_command_line"] = cmdline
        return report
    if force_cold_start and report.status == "READY":
        print(f"forced cold-start requested for map={map_name} rhi={rhi_norm}", flush=True)
    elif report.status == "READY" and not map_ok:
        print(
            f"map mismatch current={report.map} required={map_name}; cold-restart CarlaUE4",
            flush=True,
        )
    elif report.status == "READY" and launch_match is False:
        print(
            f"RHI/offscreen mismatch cmdline={cmdline!r} required rhi={rhi_norm} "
            f"offscreen={bool(render_offscreen)}; cold-restart CarlaUE4",
            flush=True,
        )
    else:
        print(
            f"preflight {report.status}/{report.error_code}; cold-start map={map_name} "
            f"rhi={rhi_norm} offscreen={bool(render_offscreen)}",
            flush=True,
        )
    launch_arguments = _write_startup_map(
        map_name,
        server_res_x=server_res_x,
        server_res_y=server_res_y,
        rhi=rhi_norm,
        render_offscreen=bool(render_offscreen),
    )
    _kill_carla_windows()
    # Shared map-aware ensure entry (same as `sdf sim ensure --map ...`).
    report = resolver.ensure(
        startup_timeout_seconds=float(startup_timeout_s),
        rhi=rhi_norm,
        render_offscreen=bool(render_offscreen),
        map_name=map_name,
        launch_mode="default_engine",
        auto_pin_default_engine=True,
    )
    if report.status != "READY":
        print("ensure not READY", report.status, report.error_code, report.error_message, flush=True)
        return report
    if not _map_matches(str(report.map or ""), map_name):
        print(
            f"MAP_MISMATCH after ensure current={report.map} required={map_name}",
            flush=True,
        )
        report.status = "BLOCKED"
        return report
    post_cmdline = query_windows_carla_command_line() or (
        f"CarlaUE4.exe {launch_arguments}"
    )
    if hasattr(report, "details"):
        report.details = dict(report.details or {})
        report.details["requested_rhi"] = rhi_norm
        report.details["effective_rhi"] = (
            parse_rhi_from_command_line(post_cmdline) or rhi_norm
        )
        report.details["render_offscreen"] = bool(render_offscreen)
        report.details["server_command_line"] = post_cmdline
        report.details["launch_arguments"] = launch_arguments
        res_x, res_y = parse_resolution_from_command_line(post_cmdline)
        report.details["server_resolution"] = [res_x, res_y]
    print(
        f"ensure READY map={report.map} rhi={rhi_norm} offscreen={bool(render_offscreen)}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stable pure VLA spatial path + constrained MPC")
    parser.add_argument(
        "--map",
        default="random",
        help="Town name, or 'random' to pick from the stable non-Large-Map pool (default: random)",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--sim-dt", type=float, default=0.05)
    parser.add_argument(
        "--vla-period-s",
        type=float,
        default=0.75,
        help="Low-rate VLA replanning period (default: 0.75s, selected by Town04 A/B).",
    )
    parser.add_argument(
        "--v-ref",
        "--max-speed",
        dest="v_ref",
        type=float,
        default=15.0,
        help="Absolute speed cap in m/s, not a forced cruise speed (default: 15.0).",
    )
    parser.add_argument(
        "--speed-gain",
        type=float,
        default=1.0,
        help=(
            "Multiplicative calibration for the VLA speed head before the cap "
            "(default: 1.0; official agent has no 1.5× gain)."
        ),
    )
    parser.add_argument(
        "--route-segment-m",
        type=float,
        default=600.0,
        help="Rolling coarse-navigation segment length; refreshed before exhaustion.",
    )
    parser.add_argument(
        "--coarse-route-spacing-m",
        type=float,
        default=None,
        help=(
            "Spacing used to resample dense map routes before official "
            "RoutePlanner[1]/[2]. Default: 10 m for frozen V3 scenarios, "
            "otherwise the legacy adapter default."
        ),
    )
    parser.add_argument(
        "--checkpoint-period-s",
        type=float,
        default=10.0,
        help="Write progress_latest.json at this simulated-time interval.",
    )
    parser.add_argument(
        "--cam-w",
        type=int,
        default=None,
        help="RGB width (default: 1024 official / 640 legacy).",
    )
    parser.add_argument(
        "--cam-h",
        type=int,
        default=None,
        help="RGB height (default: 512 official / 320 legacy).",
    )
    parser.add_argument(
        "--official-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Official SimLingo input contract: 1024x512 BGR-JPEG preprocess, "
            "RoutePlanner 7.5m targets [1]/[2], lateral_mode_flip off (default: on)."
        ),
    )
    parser.add_argument(
        "--enable-lateral-mode-flip",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override PathManager lateral_mode_flip (default: off under official contract).",
    )
    parser.add_argument(
        "--camera-sensor-tick-s",
        type=float,
        default=0.0,
        help="RGB render period; 0 derives it from --vla-period-s (recommended).",
    )
    parser.add_argument("--server-res-x", type=int, default=640)
    parser.add_argument("--server-res-y", type=int, default=360)
    parser.add_argument(
        "--inference-mode",
        choices=("full", "forward-only", "model-resident", "camera-only"),
        default="full",
        help=(
            "D3D isolation mode: full drives; forward-only runs real VLA forwards while "
            "diagnostic modes hold the vehicle braked."
        ),
    )
    parser.add_argument(
        "--gpu-idle-guard-ms",
        type=float,
        default=50.0,
        help="Host pause after a CUDA forward before the next CARLA tick (default: 50ms).",
    )
    parser.add_argument("--steer-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for --map random")
    parser.add_argument(
        "--vla-version",
        choices=("v0", "v1", "v2", "v3"),
        default="v0",
        help=(
            "VLA policy version: v0 = K1; v1 = longitudinal K2; "
            "v2 = learned Spatial K2 development endurance; "
            "v3 = route-bound mixed-semantic K2."
        ),
    )
    parser.add_argument(
        "--force-candidate-index",
        type=int,
        choices=(0, 1),
        default=None,
        help="K2 only: force candidate 0/1. Default: bundle top1.",
    )
    parser.add_argument(
        "--spatial-head-checkpoint",
        default="",
        help="Required with --vla-version v2 or learned v3.",
    )
    parser.add_argument(
        "--v3-mode",
        choices=("teacher", "learned"),
        default="teacher",
        help="K2 V3 semantic source (default: teacher contract smoke).",
    )
    parser.add_argument(
        "--v3-route-fixture",
        default="",
        help=(
            "Frozen safedrive.r2_v3.route_fixtures.v1 JSON. Requires "
            "--vla-version v3 and --v3-route-maneuver; exact spawn only."
        ),
    )
    parser.add_argument(
        "--v3-route-maneuver",
        choices=(
            "FOLLOW_STRAIGHT",
            "FOLLOW_CURVE_LEFT",
            "FOLLOW_CURVE_RIGHT",
            "JUNCTION_STRAIGHT",
            "TURN_LEFT",
            "TURN_RIGHT",
            "ROUTE_CHANGE_LEFT",
            "ROUTE_CHANGE_RIGHT",
        ),
        default="",
    )
    parser.add_argument(
        "--v3-scenario-family",
        choices=(
            "auto",
            "clear",
            "lead_braking",
            "cut_in",
            "crossing",
            "merge",
            "obstruction",
            "traffic_control",
        ),
        default="auto",
        help="Registered Basic 1v1 family for teacher/contract smoke.",
    )
    parser.add_argument(
        "--v3-requested-overtake-side",
        choices=("NONE", "LEFT", "RIGHT"),
        default="NONE",
    )
    parser.add_argument(
        "--v3-scenario-registry",
        default="",
        help="Frozen Scenario Registry V1 TOML for exact Basic 1v1 spawning.",
    )
    parser.add_argument("--v3-scenario-id", default="")
    parser.add_argument("--v3-seed-id", default="")
    parser.add_argument(
        "--v3-contract-auto-select",
        action="store_true",
        default=False,
        help=(
            "Component-smoke selector: request candidate 1 while the current "
            "observable conflict/red light is active, otherwise candidate 0. "
            "Guard rejection and candidate unavailability still fail closed."
        ),
    )
    parser.add_argument(
        "--spatial-checkpoint-use",
        choices=(
            "development_live_smoke",
            "x5h_acceptance",
            "r2v3_blind_audit",
        ),
        default="development_live_smoke",
        help="Spatial head checkpoint contract use (default: development smoke).",
    )
    parser.add_argument(
        "--allow-v2-nominal-degraded-fallback",
        action="store_true",
        default=False,
        help=(
            "Development endurance only: when forced candidate 0 and the sole "
            "Guard failure is defensive spatial collapse, execute the hash-checked "
            "nominal candidate and record degraded_nominal_only."
        ),
    )
    parser.add_argument("--full-duration", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help=(
            "Draw at each VLA update: green=raw pred_route, yellow=committed MPC, "
            "cyan=pred_speed_wps. Disabled by default for D3D endurance stability."
        ),
    )
    parser.add_argument("--no-debug-draw", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=180.0,
        help="Carla cold-start handshake timeout after map pin",
    )
    parser.add_argument(
        "--no-map-restart",
        action="store_true",
        help="Do not kill/restart CARLA; fail on MAP_MISMATCH instead",
    )
    parser.add_argument(
        "--force-cold-start",
        action="store_true",
        help="Restart CARLA even when the requested map is already loaded (controlled A/B runs).",
    )
    parser.add_argument(
        "--rhi",
        choices=("dx11", "dx12"),
        default=DEFAULT_RHI,
        help=f"CARLA RHI for cold-start via carla_start.toml / sdf sim ensure (default: {DEFAULT_RHI}).",
    )
    parser.add_argument(
        "--render-offscreen",
        action="store_true",
        default=False,
        help="Cold-start CARLA with -RenderOffScreen (must match ensure launch; not no-rendering mode).",
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(ROOT / "docs/runtime-evidence/g3-05/vla_mpc_stable"),
    )
    parser.add_argument(
        "--vehicle-blueprint",
        default=DEFAULT_EGO_BLUEPRINT,
        help=(
            f"Ego vehicle blueprint id (default: {DEFAULT_EGO_BLUEPRINT}). "
            "Missing blueprint raises RuntimeError — no Audi/Tesla silent fallback."
        ),
    )
    parser.add_argument(
        "--enable-stationary-requery",
        action="store_true",
        default=False,
        help=(
            "Experimental: when stationary with sustained VLA stop, re-query once "
            "with a 1.5 m/s speed hint. Default OFF for pure VLA+MPC."
        ),
    )
    parser.add_argument(
        "--reuse-simlingo-runtime",
        action="store_true",
        default=False,
        help=(
            "Reuse one already-loaded SimLingo runtime when main() is invoked "
            "repeatedly in the same process (campaign collection only)."
        ),
    )
    args = parser.parse_args()
    contract_cfg = (
        SimLingoContractConfig(official_contract=True)
        if bool(args.official_contract)
        else legacy_contract_config()
    )
    if args.coarse_route_spacing_m is not None:
        if float(args.coarse_route_spacing_m) <= 0.0:
            parser.error("--coarse-route-spacing-m must be positive")
        contract_cfg.densify_ds_m = float(args.coarse_route_spacing_m)
    elif args.vla_version == "v3" and args.v3_scenario_registry:
        # Leaderboard's official global plan is sparse and keeps navigation
        # decision anchors.  Registry routes are dense topology polylines; a
        # 10 m resample preserves coarse navigation without leaking an MPC
        # centerline and gives target_point2 useful look-ahead at junctions.
        contract_cfg.densify_ds_m = 10.0
    # Camera defaults follow contract when CLI did not override.
    if args.cam_w is None:
        args.cam_w = int(contract_cfg.camera_width)
    if args.cam_h is None:
        args.cam_h = int(contract_cfg.camera_height)
    if args.enable_lateral_mode_flip is not None:
        contract_cfg.lateral_mode_flip_enabled = bool(args.enable_lateral_mode_flip)
    else:
        contract_cfg.lateral_mode_flip_enabled = not bool(args.official_contract)
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if args.sim_dt <= 0.0:
        parser.error("--sim-dt must be positive")
    if args.vla_period_s <= 0.0:
        parser.error("--vla-period-s must be positive")
    if args.v_ref < 0.0:
        parser.error("--v-ref speed cap must be non-negative")
    if args.speed_gain < 0.0:
        parser.error("--speed-gain must be non-negative")
    if args.camera_sensor_tick_s < 0.0:
        parser.error("--camera-sensor-tick-s must be non-negative")
    if args.gpu_idle_guard_ms < 0.0:
        parser.error("--gpu-idle-guard-ms must be non-negative")
    if args.cam_w <= 0 or args.cam_h <= 0 or args.server_res_x <= 0 or args.server_res_y <= 0:
        parser.error("camera and server resolutions must be positive")
    if args.force_cold_start and args.no_map_restart:
        parser.error("--force-cold-start conflicts with --no-map-restart")
    if args.force_candidate_index is not None and args.vla_version not in {"v1", "v2", "v3"}:
        parser.error("--force-candidate-index requires --vla-version v1, v2, or v3")
    if args.vla_version == "v2" and not args.spatial_head_checkpoint:
        parser.error("--vla-version v2 requires --spatial-head-checkpoint")
    if (
        args.vla_version == "v3"
        and args.v3_mode == "learned"
        and not args.spatial_head_checkpoint
    ):
        parser.error("--vla-version v3 --v3-mode learned requires --spatial-head-checkpoint")
    if bool(args.v3_route_fixture) != bool(args.v3_route_maneuver):
        parser.error(
            "--v3-route-fixture and --v3-route-maneuver must be provided together"
        )
    if args.v3_route_fixture and args.vla_version != "v3":
        parser.error("--v3-route-fixture requires --vla-version v3")
    registry_args = (
        bool(args.v3_scenario_registry),
        bool(args.v3_scenario_id),
        bool(args.v3_seed_id),
    )
    if any(registry_args) and not all(registry_args):
        parser.error(
            "--v3-scenario-registry, --v3-scenario-id and --v3-seed-id "
            "must be provided together"
        )
    if args.v3_scenario_registry and args.vla_version != "v3":
        parser.error("--v3-scenario-registry requires --vla-version v3")
    if args.v3_contract_auto_select and args.vla_version != "v3":
        parser.error(
            "--v3-contract-auto-select requires --vla-version v3"
        )
    if args.v3_scenario_registry and args.v3_route_fixture:
        parser.error(
            "route and actor fixtures must be authored in one scenario registry; "
            "external --v3-route-fixture cannot be combined"
        )
    if args.allow_v2_nominal_degraded_fallback and not (
        args.vla_version == "v2" and args.force_candidate_index == 0
    ):
        parser.error(
            "--allow-v2-nominal-degraded-fallback requires "
            "--vla-version v2 --force-candidate-index 0"
        )
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    requested_map = _resolve_requested_map(args.map, seed=args.seed)
    scenario_fixture_v3 = None
    scenario_fixture_route_context_v3 = None
    scenario_registry_hash_v3 = ""
    scenario_fixture_maneuver_v3 = None
    if args.v3_scenario_registry:
        from driving_vla.evaluation.scenario_registry import (
            load_scenario_registry,
        )
        from driving_vla.model.navigation_contract import RouteManeuver

        registry_v3 = load_scenario_registry(args.v3_scenario_registry)
        scenario_fixture_v3 = registry_v3.get(
            args.v3_scenario_id, args.v3_seed_id
        )
        scenario_registry_hash_v3 = str(
            registry_v3.registry_sha256
            or registry_v3.compute_registry_sha256()
        )
        if scenario_fixture_v3.map_name != requested_map:
            parser.error(
                "V3 scenario fixture map does not match --map: "
                f"{scenario_fixture_v3.map_name} != {requested_map}"
            )
        if abs(float(scenario_fixture_v3.sim_dt_s) - float(args.sim_dt)) > 1.0e-9:
            parser.error("V3 scenario fixture sim_dt_s does not match --sim-dt")
        navigation = dict(scenario_fixture_v3.route.navigation_context or {})
        if navigation.get("maneuver"):
            scenario_fixture_maneuver_v3 = RouteManeuver(
                str(navigation["maneuver"])
            )
        if navigation.get("frozen_context_json"):
            from driving_vla.model.navigation_contract import RouteContextV3

            scenario_fixture_route_context_v3 = RouteContextV3.from_mapping(
                json.loads(str(navigation["frozen_context_json"]))
            )
    effective_v3_family = (
        str(scenario_fixture_v3.family)
        if scenario_fixture_v3 is not None
        else (
            "clear"
            if args.v3_scenario_family == "auto"
            else str(args.v3_scenario_family)
        )
    )
    if (
        scenario_fixture_v3 is not None
        and args.v3_scenario_family != "auto"
        and args.v3_scenario_family != scenario_fixture_v3.family
    ):
        parser.error(
            "--v3-scenario-family conflicts with frozen registry family"
        )
    route_fixture_context_v3 = None
    route_fixture_spawn_v3 = None
    route_fixture_manifest_hash_v3 = ""
    if args.v3_route_fixture:
        (
            route_fixture_context_v3,
            route_fixture_spawn_v3,
            route_fixture_manifest_hash_v3,
        ) = _load_v3_route_fixture(
            args.v3_route_fixture,
            maneuver=args.v3_route_maneuver,
            requested_map=requested_map,
        )
    requested_rhi = normalize_rhi(args.rhi)
    render_offscreen = bool(args.render_offscreen)
    camera_sensor_tick_s = _camera_sensor_tick_s(
        float(args.vla_period_s), float(args.sim_dt), float(args.camera_sensor_tick_s)
    )
    debug_draw = bool(args.debug_draw and not args.no_debug_draw)
    driver_version = _query_driver_version()
    run_config: dict[str, Any] = {
        "status": "INITIALIZING",
        "run_id": f"g3-vla-mpc-{int(time.time() * 1000)}",
        "requested_map": requested_map,
        "map_arg": str(args.map),
        "inference_mode": str(args.inference_mode),
        "duration_s": float(args.duration_s),
        "sim_dt_s": float(args.sim_dt),
        "vla_period_s": float(args.vla_period_s),
        "camera_sensor_tick_s": camera_sensor_tick_s,
        "camera_resolution": [int(args.cam_w), int(args.cam_h)],
        "camera_fov_deg": float(SIMLINGO_CAMERA_FOV_DEG),
        "camera_mount_xyz": list(SIMLINGO_CAMERA_XYZ),
        "server_resolution": [int(args.server_res_x), int(args.server_res_y)],
        "official_contract": bool(contract_cfg.official_contract),
        "simlingo_contract": contract_cfg.evidence_dict(),
        "path_manager_gates": {
            "enable_lateral_mode_flip": bool(contract_cfg.lateral_mode_flip_enabled),
            "enable_early_lane_change": bool(contract_cfg.early_lane_change_enabled),
        },
        "requested_rhi": requested_rhi,
        "effective_rhi": None,
        "effective_rhi_source": None,
        "rhi_match_requested": None,
        "rhi_independently_verified": False,
        "render_offscreen": render_offscreen,
        "server_command_line": None,
        "driver_version": driver_version,
        "requested_vehicle_blueprint": str(args.vehicle_blueprint),
        "effective_vehicle_blueprint": None,
        "enable_stationary_requery": bool(args.enable_stationary_requery),
        "speed_cap_mps": float(args.v_ref),
        "speed_gain": float(args.speed_gain),
        "steer_sign": float(args.steer_sign),
        "route_segment_m": float(args.route_segment_m),
        "checkpoint_period_s": float(args.checkpoint_period_s),
        "gpu_idle_guard_ms": float(args.gpu_idle_guard_ms),
        "debug_draw": debug_draw,
        "seed": args.seed,
        "startup_timeout_s": float(args.startup_timeout_s),
        "no_map_restart": bool(args.no_map_restart),
        "force_cold_start": bool(args.force_cold_start),
        "reuse_simlingo_runtime": bool(args.reuse_simlingo_runtime),
        "vla_version": str(args.vla_version),
        "force_candidate_index": args.force_candidate_index,
        "spatial_head_checkpoint": (
            str(Path(args.spatial_head_checkpoint))
            if args.spatial_head_checkpoint
            else None
        ),
        "spatial_checkpoint_use": str(args.spatial_checkpoint_use),
        "v3_mode": str(args.v3_mode),
        "v3_route_fixture": (
            str(Path(args.v3_route_fixture)) if args.v3_route_fixture else None
        ),
        "v3_route_maneuver": str(args.v3_route_maneuver or ""),
        "v3_route_fixture_manifest_hash": route_fixture_manifest_hash_v3,
        "v3_scenario_family": str(args.v3_scenario_family),
        "v3_requested_overtake_side": str(
            args.v3_requested_overtake_side
        ),
        "v3_effective_scenario_family": effective_v3_family,
        "v3_scenario_registry": (
            str(Path(args.v3_scenario_registry))
            if args.v3_scenario_registry
            else None
        ),
        "v3_scenario_registry_hash": scenario_registry_hash_v3,
        "v3_scenario_id": str(args.v3_scenario_id or ""),
        "v3_seed_id": str(args.v3_seed_id or ""),
        "v3_contract_auto_select": bool(args.v3_contract_auto_select),
        "allow_v2_nominal_degraded_fallback": bool(
            args.allow_v2_nominal_degraded_fallback
        ),
        "branch_type": (
            "longitudinal_temporal"
            if str(args.vla_version) == "v1"
            else (
                "learned_spatial_semantic"
                if str(args.vla_version) == "v2"
                else (
                    "learned_route_bound_mixed_semantic"
                    if str(args.vla_version) == "v3"
                    else "k1_single"
                )
            )
        ),
        "wall_start_s": time.time(),
    }
    configured_maneuver_v3 = (
        scenario_fixture_maneuver_v3
        or (
            route_fixture_context_v3.maneuver
            if route_fixture_context_v3 is not None
            else None
        )
    )
    route_change_hlc_v3 = bool(
        str(args.vla_version) == "v3"
        and configured_maneuver_v3 is not None
        and str(getattr(configured_maneuver_v3, "value", configured_maneuver_v3))
        in {"ROUTE_CHANGE_LEFT", "ROUTE_CHANGE_RIGHT"}
    )
    # Actual prompt contract.  V3 route changes use the released checkpoint's
    # native HLC mode until the registered target lane is reached.
    if bool(contract_cfg.official_contract):
        run_config["simlingo_contract"]["prompt_mode"] = (
            "route_maneuver_dynamic_hlc"
            if route_change_hlc_v3
            else "target_point"
        )
        run_config["simlingo_contract"]["command_text"] = None
        run_config["simlingo_contract"]["command_text_config_default"] = (
            "Command: follow the road."
        )
        run_config["simlingo_contract"]["actual_runtime_command_text"] = None
        run_config["simlingo_contract"]["route_change_hlc_enabled"] = (
            route_change_hlc_v3
        )
        run_config["simlingo_contract"]["route_change_native_commitment"] = (
            "hold one Guard-approved native pred_route whose endpoint is within "
            "the registered target-lane boundary; release on target lane "
            "or MPC 5s stale limit"
            if route_change_hlc_v3
            else None
        )
    else:
        run_config["simlingo_contract"]["prompt_mode"] = "legacy_command_text"
        run_config["simlingo_contract"]["actual_runtime_command_text"] = (
            contract_cfg.command_text
        )
    _write_json(evidence_dir / "run_config.json", run_config)
    resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=15.0)
    if args.no_map_restart:
        report = resolver.preflight()
        if report.status != "READY":
            print("preflight not READY", report.status, report.error_code, flush=True)
            return 75
        if not _map_matches(str(report.map or ""), requested_map):
            print(
                f"MAP_MISMATCH current={report.map} required={requested_map}",
                flush=True,
            )
            return 75
        cmdline = query_windows_carla_command_line()
        match = launch_params_match(
            cmdline, rhi=requested_rhi, render_offscreen=render_offscreen
        )
        if match is False:
            print(
                f"LAUNCH_PARAM_MISMATCH under --no-map-restart cmdline={cmdline!r} "
                f"required rhi={requested_rhi} offscreen={render_offscreen}",
                flush=True,
            )
            return 75
    else:
        report = _ensure_ready_map(
            resolver,
            requested_map,
            startup_timeout_s=float(args.startup_timeout_s),
            server_res_x=int(args.server_res_x),
            server_res_y=int(args.server_res_y),
            force_cold_start=bool(args.force_cold_start),
            rhi=requested_rhi,
            render_offscreen=render_offscreen,
        )
        if getattr(report, "status", None) != "READY":
            return 75
    client, report = resolver.connect(report=report)
    # Large layered towns need a long RPC budget; 20s is enough for Town0x/10 only.
    rpc_timeout_s = 120.0 if requested_map in NESTED_MAP_PACKAGE else 30.0
    client.set_timeout(rpc_timeout_s)
    world = client.get_world()
    current_map = str(world.get_map().name)
    if not _map_matches(current_map, requested_map):
        print(f"MAP_MISMATCH current={current_map} required={requested_map}", flush=True)
        print("Cold-restart CarlaUE4 with the required startup map.", flush=True)
        return 75
    details = getattr(report, "details", {}) or {}
    cmdline = details.get("server_command_line") or query_windows_carla_command_line()
    parsed_rhi = parse_rhi_from_command_line(cmdline or "")
    if parsed_rhi is not None:
        effective_rhi = parsed_rhi
        effective_rhi_source = "server_command_line"
    elif details.get("effective_rhi") is not None and not bool(args.no_map_restart):
        # Cold-start path may record launch RHI before cmdline is queryable.
        effective_rhi = details.get("effective_rhi")
        effective_rhi_source = "cold_start_details"
    elif not bool(args.no_map_restart):
        # We just launched with requested_rhi; still not an independent cmdline proof.
        effective_rhi = requested_rhi
        effective_rhi_source = "cold_start_requested_assumed"
    else:
        # Attach mode: never claim DX12 solely from requested_rhi without cmdline -dx12.
        effective_rhi = None
        effective_rhi_source = "unknown_attach_no_cmdline_rhi"
    rhi_match = effective_rhi is not None and str(effective_rhi) == str(requested_rhi)
    rhi_verified = (
        parsed_rhi is not None and str(parsed_rhi) == str(requested_rhi)
    )
    run_config["status"] = "CONNECTED"
    run_config["actual_map"] = current_map
    run_config["requested_rhi"] = requested_rhi
    run_config["effective_rhi"] = effective_rhi
    run_config["effective_rhi_source"] = effective_rhi_source
    run_config["rhi_match_requested"] = rhi_match
    run_config["rhi_independently_verified"] = bool(rhi_verified)
    run_config["render_offscreen"] = (
        bool(details["render_offscreen"])
        if "render_offscreen" in details
        else (
            parse_offscreen_from_command_line(cmdline or "")
            if cmdline
            else render_offscreen
        )
    )
    run_config["server_command_line"] = cmdline
    run_config["driver_version"] = driver_version
    if details.get("server_resolution"):
        run_config["server_resolution_effective"] = details.get("server_resolution")
    _write_json(evidence_dir / "run_config.json", run_config)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.sim_dt)
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = max(2, int(math.ceil(float(args.sim_dt) / 0.01)))
    if requested_map in LARGE_MAPS:
        if hasattr(settings, "tile_stream_distance"):
            settings.tile_stream_distance = 2000.0
        if hasattr(settings, "actor_active_distance"):
            settings.actor_active_distance = 2000.0
        if hasattr(settings, "spectator_as_ego"):
            settings.spectator_as_ego = True
    world.apply_settings(settings)
    # Give streaming maps a few synchronous ticks before actor surgery.
    for _ in range(20 if requested_map in NESTED_MAP_PACKAGE else 5):
        world.tick()

    ego: carla.Vehicle | None = None
    camera: carla.Sensor | None = None
    collision_sensor: carla.Sensor | None = None
    lane_sensor: carla.Sensor | None = None
    scenario_session_v3 = None
    apply_scenario_scripts_v3 = None
    live_state: dict[str, Any] = {
        "phase": "world_setup",
        "last_operation": "apply_synchronous_settings",
        "last_checkpoint": None,
    }
    try:
        for actor_filter in ("sensor.*", "vehicle.*"):
            for actor in list(world.get_actors().filter(actor_filter)):
                try:
                    actor.destroy()
                except Exception as exc:  # pragma: no cover - live only
                    print(f"WARN destroy {actor_filter} id={getattr(actor, 'id', '?')}: {exc}", flush=True)
        if scenario_fixture_v3 is not None:
            from driving_vla.evaluation.fixture_runtime import (
                _apply_scripts,
                apply_weather,
                open_fixture_session,
            )

            apply_weather(world, scenario_fixture_v3.weather)
            scenario_session_v3 = open_fixture_session(
                client,
                world,
                scenario_fixture_v3,
                settle_ticks=8,
            )
            apply_scenario_scripts_v3 = _apply_scripts
            ego = next(
                spawned.actor
                for spawned in scenario_session_v3.spawned
                if spawned.role == "ego"
            )
            spawn = ego.get_transform()
            effective_vehicle_bp = str(
                getattr(ego, "type_id", scenario_fixture_v3.ego.blueprint)
            )
        elif route_fixture_context_v3 is not None:
            assert route_fixture_spawn_v3 is not None
            ego, spawn, effective_vehicle_bp = _spawn_ego_exact_v3(
                world,
                route_fixture_spawn_v3,
                vehicle_blueprint=str(args.vehicle_blueprint),
            )
        else:
            ego, spawn, effective_vehicle_bp = _spawn_ego(
                world,
                large_map=requested_map in LARGE_MAPS,
                vehicle_blueprint=str(args.vehicle_blueprint),
            )
        run_config["requested_vehicle_blueprint"] = str(args.vehicle_blueprint)
        run_config["effective_vehicle_blueprint"] = effective_vehicle_bp
        _write_json(evidence_dir / "run_config.json", run_config)
        if requested_map in LARGE_MAPS and hasattr(settings, "spectator_as_ego"):
            settings.spectator_as_ego = False
            world.apply_settings(settings)
        for _ in range(12):
            if scenario_session_v3 is None:
                ego.apply_control(carla.VehicleControl(brake=0.5))
            elif (
                scenario_session_v3 is not None
                and apply_scenario_scripts_v3 is not None
            ):
                apply_scenario_scripts_v3(
                    scenario_session_v3,
                    simulation_time_since_anchor_s=0.0,
                    include_ego=False,
                )
            world.tick()

        if scenario_fixture_v3 is not None:
            route_xy = _densify_registered_route_v3(
                world.get_map(),
                scenario_fixture_v3.route.waypoints,
            )
            route_s = _polyline_s(route_xy)
        elif route_fixture_context_v3 is not None:
            route_xy = [
                (float(x), float(y))
                for x, y in route_fixture_context_v3.route_xy
            ]
            route_s = _polyline_s(route_xy)
            _write_json(
                evidence_dir / "route_context_v3.json",
                route_fixture_context_v3.to_dict(),
            )
        else:
            route_xy, route_s = _build_route_segment(
                world,
                spawn,
                max(100.0, float(args.route_segment_m)),
            )
        route_progress = 0.0
        completed_route_progress = 0.0
        route_refreshes = 0
        route_refresh_failures = 0

        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(args.cam_w))
        cam_bp.set_attribute("image_size_y", str(args.cam_h))
        cam_bp.set_attribute("fov", str(SIMLINGO_CAMERA_FOV_DEG))
        cam_bp.set_attribute("sensor_tick", f"{camera_sensor_tick_s:.6f}")
        camera = world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(*SIMLINGO_CAMERA_XYZ)),
            attach_to=ego,
        )
        camera_parent = getattr(camera, "parent", None)
        if camera_parent is None or int(camera_parent.id) != int(ego.id):
            raise RuntimeError("front_rgb_camera_not_attached_to_ego")
        collision_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=ego,
        )
        lane_sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=ego,
        )
        image_lock = threading.Lock()
        latest_image: dict[str, Any] = {
            "rgb": None,
            "image_layout": "bgr" if contract_cfg.official_contract else "rgb",
            "frame": -1,
            "received": 0,
            "wall_time_s": None,
            "sim_time_s": None,
        }
        event_lock = threading.Lock()
        collision_book = CollisionEpisodeBook(gap_s=DEFAULT_COLLISION_EPISODE_GAP_S)
        lane_invasion_book = LaneInvasionEpisodeBook(gap_s=DEFAULT_LANE_INVASION_GAP_S)
        has_collided = False
        ever_moved = False
        road_events: dict[str, Any] = {
            "collisions": 0,
            "collision_episodes": 0,
            "collision_raw_events": 0,
            "collision_impulse": 0.0,
            "lane_invasions": 0,
            "lane_invasion_raw_events": 0,
            "lane_invasion_episodes": 0,
            "first_collision_sim_s": None,
            "first_collision_other_type": None,
        }

        def on_image(image: carla.Image) -> None:
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            if contract_cfg.official_contract:
                # Keep BGR (CARLA BGRA[:3]) for official JPEG path — no RGB swap here.
                frame = carla_bgra_to_bgr(bgra)
                layout = "bgr"
            else:
                frame = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
                layout = "rgb"
            with image_lock:
                latest_image["rgb"] = frame
                latest_image["image_layout"] = layout
                latest_image["frame"] = int(image.frame)
                latest_image["received"] = int(latest_image["received"]) + 1
                latest_image["wall_time_s"] = time.time()
                latest_image["sim_time_s"] = float(getattr(image, "timestamp", 0.0) or 0.0)

        camera.listen(on_image)

        def on_collision(event: carla.CollisionEvent) -> None:
            nonlocal has_collided
            impulse = event.normal_impulse
            magnitude = math.sqrt(float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2)
            other = getattr(event, "other_actor", None)
            other_type = getattr(other, "type_id", None) if other is not None else None
            other_id = getattr(other, "id", None) if other is not None else None
            vehicle_xy = None
            vehicle_yaw = None
            try:
                transform = event.actor.get_transform()
                vehicle_xy = (float(transform.location.x), float(transform.location.y))
                vehicle_yaw = math.radians(float(transform.rotation.yaw))
            except Exception:
                pass
            episode = collision_book.on_collision(
                impulse=magnitude,
                other_type=str(other_type) if other_type is not None else "unknown",
                other_id=None if other_id is None else int(other_id),
                vehicle_xy=vehicle_xy,
                vehicle_yaw=vehicle_yaw,
            )
            with event_lock:
                has_collided = True
                road_events["collision_raw_events"] = collision_book.raw_event_count
                road_events["collision_impulse"] = collision_book.total_impulse
                road_events["collision_episodes"] = collision_book.episode_count
                # Keep "collisions" as episode count for acceptance gates.
                road_events["collisions"] = road_events["collision_episodes"]
                if road_events["first_collision_sim_s"] is None:
                    road_events["first_collision_sim_s"] = episode.get("first_sim_s")
                    road_events["first_collision_other_type"] = episode.get("other_type")
                    road_events["first_collision_vehicle_pose"] = episode.get(
                        "vehicle_pose_first"
                    )

        def on_lane_invasion(event: carla.LaneInvasionEvent) -> None:
            markings = getattr(event, "crossed_lane_markings", None)
            frame = getattr(event, "frame", None)
            pose_now = None
            try:
                tf = event.actor.get_transform()
                pose_now = {
                    "x": float(tf.location.x),
                    "y": float(tf.location.y),
                    "yaw": math.radians(float(tf.rotation.yaw)),
                }
            except Exception:
                pose_now = None
            lane_invasion_book.on_invasion(
                crossed_markings=markings,
                carla_frame=None if frame is None else int(frame),
                ego_pose=pose_now,
            )
            with event_lock:
                road_events["lane_invasion_raw_events"] = lane_invasion_book.raw_event_count
                road_events["lane_invasion_episodes"] = lane_invasion_book.episode_count
                # Keep integer counter as episode count for backwards-compatible readers.
                road_events["lane_invasions"] = road_events["lane_invasion_episodes"]

        collision_sensor.listen(on_collision)
        lane_sensor.listen(on_lane_invasion)
        for _ in range(10):
            ego.apply_control(carla.VehicleControl(brake=0.5))
            world.tick()

        vehicle_geometry = vehicle_geometry_from_carla_vehicle(
            ego, blueprint_id=effective_vehicle_bp
        )
        wheelbase = float(vehicle_geometry.wheelbase_m)
        max_steer_rad = float(vehicle_geometry.max_steer_rad)
        track_width_m = float(vehicle_geometry.track_width_m)
        run_config["vehicle_geometry"] = vehicle_geometry.as_dict()
        torch = None
        policy = None
        if args.inference_mode != "camera-only":
            import torch as torch_module

            torch = torch_module
            runtime, runtime_reused = _simlingo_runtime_for_run(
                reuse_loaded=bool(args.reuse_simlingo_runtime)
            )
            run_config["simlingo_runtime_reused"] = bool(runtime_reused)
            if str(args.vla_version) == "v1":
                policy = NeuralV1Policy(runtime=runtime, keep_on_gpu=True)
            elif str(args.vla_version) == "v2":
                policy = NeuralV2Policy(
                    runtime=runtime,
                    spatial_head_checkpoint=str(args.spatial_head_checkpoint),
                    keep_on_gpu=True,
                    lazy=True,
                    require_driving_feature=True,
                    device="cuda",
                    checkpoint_use=str(args.spatial_checkpoint_use),
                )
            elif str(args.vla_version) == "v3":
                policy = NeuralV3Policy(
                    runtime=runtime,
                    semantic_head_checkpoint=(
                        str(args.spatial_head_checkpoint)
                        if args.spatial_head_checkpoint
                        else None
                    ),
                    teacher_mode=str(args.v3_mode) == "teacher",
                    keep_on_gpu=True,
                    lazy=True,
                    device="cuda",
                    checkpoint_use=str(args.spatial_checkpoint_use),
                )
            else:
                policy = NeuralV0Policy(runtime=runtime, keep_on_gpu=True)
            policy.ensure_loaded()
        cuda_alloc_mb = (
            float(torch.cuda.memory_allocated() / 1024**2) if torch is not None else 0.0
        )
        run_config["model_resident_vram_mb"] = cuda_alloc_mb
        _write_json(evidence_dir / "run_config.json", run_config)
        print(
            f"map={current_map} mode={args.inference_mode} "
            f"vehicle={effective_vehicle_bp} "
            f"camera={args.cam_w}x{args.cam_h}@{camera_sensor_tick_s:.2f}s "
            f"mount={SIMLINGO_CAMERA_XYZ} "
            f"wheelbase={wheelbase:.3f} track={track_width_m:.3f} "
            f"max_steer={math.degrees(max_steer_rad):.1f}deg "
            f"geom={vehicle_geometry.geometry_source}/{vehicle_geometry.validation_status} "
            f"cuda_alloc={cuda_alloc_mb:.0f}MB",
            flush=True,
        )

        speed_cap = max(0.0, float(args.v_ref))
        path_manager = None
        speed_planner = None
        tracker = None
        does_forward = _mode_runs_forward(args.inference_mode)
        if does_forward:
            path_manager = VLAPathManager(
                PathManagerConfig(
                    # Raw 20-point polylines show κ≈0.25–0.35 from point noise
                    # even on gentle roads, so reject only clearly infeasible shapes.
                    max_abs_curvature=0.30,
                    max_switch_lateral_5m=1.0,
                    max_switch_heading_5m_deg=12.0,
                    enable_lateral_mode_flip=bool(contract_cfg.lateral_mode_flip_enabled),
                    enable_early_lane_change=bool(contract_cfg.early_lane_change_enabled),
                )
            )
            speed_planner = VLASpeedPlanner(
                VLASpeedConfig(
                    max_speed_mps=speed_cap,
                    calibration_gain=max(0.0, float(args.speed_gain)),
                    max_accel_mps2=2.50,
                )
            )
        if args.inference_mode == "full":
            tracker = ConstrainedVLAMPC(
                VLAMPCConfig(
                    control_dt_s=float(args.sim_dt),
                    prediction_dt_s=0.10,
                    horizon=20,
                    wheelbase_m=wheelbase,
                    max_steer_rad=max_steer_rad,
                    max_speed_mps=speed_cap,
                    max_accel_mps2=2.50,
                    max_lateral_accel_mps2=1.50,
                    solver_deadline_ms=30.0,
                )
            )

        sim_start = float(world.get_snapshot().timestamp.elapsed_seconds)
        live_state["phase"] = "running"
        live_state["sim_start_s"] = sim_start
        next_inference_sim_s = sim_start
        last_speed_update_sim_s = sim_start
        next_checkpoint_sim_s = sim_start + max(1.0, float(args.checkpoint_period_s))
        last_xy = (_ego_pose(ego).x, _ego_pose(ego).y)
        distance_m = 0.0
        steps = 0
        accepts = 0
        reject_reasons: Counter[str] = Counter()
        solver_modes: Counter[str] = Counter()
        solver_statuses: Counter[str] = Counter()
        steer_values: list[float] = []
        cte_values: list[float] = []
        actual_speed_values: list[float] = []
        target_speed_values: list[float] = []
        path_age_values: list[float] = []
        peak_vram_values: list[float] = []
        inference_ms: list[float] = []
        events: list[dict[str, Any]] = []
        route_mission_base_v3 = (
            route_fixture_context_v3 or scenario_fixture_route_context_v3
        )
        v3_conflict_side_latched = "none"
        v3_interaction_conflict_latched = False
        v3_overtake_phase = "DEPART"
        v3_overtake_complete_sim_s: float | None = None
        v3_cut_in_complete = False
        v3_crossing_initial_side = "none"
        v3_crossing_complete = False
        v3_route_change_latch_active = False
        v3_route_change_latch_source = ""
        v3_route_change_latch_stamp_s: float | None = None
        v3_route_change_target_hold_start_s: float | None = None
        v3_route_change_target_hold_progress: float | None = None
        v3_route_change_latch_events: list[dict[str, Any]] = []
        # During a native route-change commitment, keep executing the exact
        # same-forward speed samples while suppressing a second VLA forward.
        # This avoids freezing the first 20-Hz slew-limited speed (often
        # ~0.1 m/s) for the entire lane-change window.  Geometry/source/hash
        # remain untouched; only execution metadata is refreshed.
        v3_route_change_latched_speed_samples: tuple[float, ...] | None = None
        v3_route_change_execution_refreshes = 0
        if route_mission_base_v3 is not None:
            _write_json(
                evidence_dir / "route_context_v3.json",
                route_mission_base_v3.to_dict(),
            )
        hang_trace: dict[str, Any] = {
            "forwards": [],
            "last_tick": None,
            "last_camera": None,
            "n_forward": 0,
        }
        driving_trace: list[dict[str, Any]] = []
        long_horizon_trace: list[dict[str, Any]] = []
        last_long_trace_lane_invasions = 0
        first_offroad_snapshot: dict[str, Any] | None = None
        first_long_stop_snapshot: dict[str, Any] | None = None
        reanchor_snapshots: list[dict[str, Any]] = []
        stationary_requery_log: list[dict[str, Any]] = []
        consecutive_vla_stop_frames = 0
        last_requery_sim_s: float | None = None
        requery_cfg = StationaryRequeryConfig()
        enable_stationary_requery = bool(args.enable_stationary_requery)
        last_route_progress_m = 0.0
        server_zombie = False
        # Seed last_good with real vehicle height (large maps may be z≈100m+).
        try:
            _ego_tf0 = ego.get_transform()
            spectator_state = {
                "x": float(_ego_tf0.location.x),
                "y": float(_ego_tf0.location.y),
                "z": float(_ego_tf0.location.z),
                "yaw": float(_ego_tf0.rotation.yaw),
            }
        except Exception:
            spectator_state = {
                "x": last_xy[0],
                "y": last_xy[1],
                "z": 1.0,
                "yaw": 0.0,
            }
        previous_steer_norm = 0.0
        previous_sign = 0
        sign_flips = 0
        offroad_steps = 0
        offroad_steps_raw = 0
        authorized_non_driving_steps = 0
        route_target_invalid = 0
        next_route_retry_sim_s = sim_start
        carla_map = world.get_map()
        # User-facing chase view follows every simulation tick.  The model RGB
        # camera is rigidly attached to ego above; this keeps CARLA spectator
        # motion equally continuous during all formal tests.
        spectator_period_steps = 1
        spectator_follow_ok = bool(
            set_spectator_follow(world, ego, last_good=spectator_state)
        )
        if not spectator_follow_ok:
            raise RuntimeError("spectator_follow_initialization_failed")
        lane_oracle_samples: list[dict[str, Any]] = []
        traffic_light_samples: list[dict[str, Any]] = []
        control_seq_full: list[dict[str, Any]] = []
        committed_source_counts: Counter[str] = Counter()
        raw_kappa_at_1m_series: list[float] = []
        committed_kappa_at_1m_series: list[float] = []
        raw_kappa_max_series: list[float] = []
        committed_kappa_max_series: list[float] = []

        while True:
            loop_wall = time.perf_counter()
            snapshot = world.get_snapshot()
            sim_s = float(snapshot.timestamp.elapsed_seconds)
            if sim_s - sim_start >= float(args.duration_s):
                break
            pose = _ego_pose(ego)
            collision_book.set_sim_s(sim_s)
            lane_invasion_book.set_sim_s(sim_s)
            if pose.speed_mps >= 0.5:
                ever_moved = True
            actual_speed_values.append(pose.speed_mps)
            distance_m += math.hypot(pose.x - last_xy[0], pose.y - last_xy[1])
            last_xy = (pose.x, pose.y)
            route_progress = _project_route_s(route_xy, route_s, pose.x, pose.y, route_progress)
            # Observation-only lane / traffic-light oracles (never control).
            lane_oracle = observe_lane_oracle(
                carla_map,
                ego.get_location(),
                pose.yaw,
                carla_module=carla,
            )
            lane_oracle_samples.append(lane_oracle)
            tl_obs = observe_traffic_light(ego, carla_module=carla)
            tl_obs["sim_s"] = sim_s
            tl_obs["speed_mps"] = pose.speed_mps
            traffic_light_samples.append(tl_obs)
            driving_waypoint = None
            try:
                driving_waypoint = carla_map.get_waypoint(
                    ego.get_location(),
                    project_to_road=False,
                    lane_type=carla.LaneType.Driving,
                )
                if driving_waypoint is None:
                    offroad_steps += 1
                    offroad_steps_raw += 1
            except RuntimeError:
                offroad_steps += 1
                offroad_steps_raw += 1

            if v3_route_change_latch_active:
                current_road_id = (
                    None
                    if driving_waypoint is None
                    else int(getattr(driving_waypoint, "road_id", 0))
                )
                current_lane_id = (
                    None
                    if driving_waypoint is None
                    else int(getattr(driving_waypoint, "lane_id", 0))
                )
                target_reached = bool(
                    route_mission_base_v3 is not None
                    and current_lane_id is not None
                    and route_mission_base_v3.target_lane_id is not None
                    and int(current_lane_id)
                    == int(route_mission_base_v3.target_lane_id)
                    and (
                        route_mission_base_v3.target_road_id is None
                        or (
                            current_road_id is not None
                            and int(current_road_id)
                            == int(route_mission_base_v3.target_road_id)
                        )
                    )
                )
                if target_reached:
                    if v3_route_change_target_hold_start_s is None:
                        v3_route_change_target_hold_start_s = sim_s
                        v3_route_change_target_hold_progress = route_progress
                else:
                    v3_route_change_target_hold_start_s = None
                    v3_route_change_target_hold_progress = None
                target_lane_stable = bool(
                    target_reached
                    and v3_route_change_target_hold_progress is not None
                    and route_progress
                    - float(v3_route_change_target_hold_progress)
                    >= 8.0
                )
                committed_latch = (
                    None if path_manager is None else path_manager.committed
                )
                latch_age_s = (
                    float("inf")
                    if committed_latch is None
                    else max(0.0, sim_s - float(committed_latch.stamp_s))
                )
                stale_limit_s = float(
                    getattr(
                        getattr(tracker, "config", None),
                        "path_stale_zero_s",
                        5.0,
                    )
                )
                if route_change_hlc_v3:
                    # A native route-change commitment is intentionally
                    # held long enough to cross and stabilize in the target
                    # lane.  The ordinary 5 s stale limit can release just
                    # as the vehicle first touches the lane, after which the
                    # next native proposal pulls it back to origin.
                    stale_limit_s = max(stale_limit_s, 8.0)
                release_reason = (
                    "TARGET_LANE_STABLE_8M"
                    if target_lane_stable
                    else (
                        "COMMITTED_PATH_MISSING"
                        if committed_latch is None
                        else (
                            "COMMITTED_PATH_STALE"
                            if latch_age_s >= stale_limit_s
                            else ""
                        )
                    )
                )
                if release_reason:
                    v3_route_change_latch_events.append(
                        {
                            "sim_s": sim_s,
                            "event": "RELEASE",
                            "reason": release_reason,
                            "source_id": v3_route_change_latch_source,
                            "latch_age_s": latch_age_s,
                            "current_road_id": current_road_id,
                            "current_lane_id": current_lane_id,
                        }
                    )
                    v3_route_change_latch_active = False
                    v3_route_change_latch_source = ""
                    v3_route_change_latch_stamp_s = None
                    v3_route_change_target_hold_start_s = None
                    v3_route_change_target_hold_progress = None
                    v3_route_change_latched_speed_samples = None

            if (
                v3_route_change_latch_active
                and speed_planner is not None
                and path_manager is not None
                and v3_route_change_latched_speed_samples is not None
            ):
                # Reuse only the speed sequence bound to the committed native
                # path.  No new model output, route steering, or map centerline
                # is introduced while the maneuver is latched.
                speed_dt = max(
                    float(args.sim_dt),
                    sim_s - float(last_speed_update_sim_s),
                )
                committed_speed = speed_planner.update(
                    v3_route_change_latched_speed_samples,
                    dt_s=speed_dt,
                    ego_speed_mps=pose.speed_mps,
                )
                last_speed_update_sim_s = sim_s
                moving_heartbeat = pose.speed_mps >= 0.25
                if path_manager.update_committed_execution(
                    target_speed_mps=committed_speed.target_speed_mps,
                    stamp_s=sim_s if moving_heartbeat else None,
                ):
                    if moving_heartbeat:
                        v3_route_change_execution_refreshes += 1

            if (
                route_fixture_context_v3 is None
                and scenario_fixture_v3 is None
                and
                float(route_s[-1]) - route_progress < 60.0
                and sim_s >= next_route_retry_sim_s
            ):
                try:
                    refreshed_xy, refreshed_s = _build_route_segment(
                        world,
                        ego.get_transform(),
                        max(100.0, float(args.route_segment_m)),
                    )
                    completed_route_progress += route_progress
                    route_xy, route_s = refreshed_xy, refreshed_s
                    route_progress = 0.0
                    route_refreshes += 1
                    print(
                        f"route refreshed n={route_refreshes} length={float(route_s[-1]):.1f}m",
                        flush=True,
                    )
                except Exception as exc:
                    route_refresh_failures += 1
                    next_route_retry_sim_s = sim_s + 5.0
                    print(f"WARN route refresh failed: {exc}", flush=True)

            live_state["sim_elapsed_s"] = sim_s - sim_start
            live_state["steps"] = steps
            if (
                does_forward
                and sim_s >= next_inference_sim_s
                and not v3_route_change_latch_active
            ):
                assert policy is not None
                assert speed_planner is not None
                assert path_manager is not None
                with image_lock:
                    image = None if latest_image["rgb"] is None else latest_image["rgb"].copy()
                    camera_frame = int(latest_image["frame"])
                    image_layout = str(latest_image.get("image_layout") or "rgb")
                if image is not None:
                    nav_full = _navigation_targets_full(
                        route_xy, pose, route_progress, contract=contract_cfg
                    )
                    tp1, tp2 = nav_full.target_ego_1, nav_full.target_ego_2
                    route_progress = nav_full.progress_s
                    navigation_valid = nav_full.valid
                    if not navigation_valid:
                        route_target_invalid += 1
                        print(
                            f"WARN coarse route target is invalid/exhausted; hold last VLA path "
                            f"tp1=({tp1[0]:.1f},{tp1[1]:.1f}) mode={nav_full.mode}",
                            flush=True,
                        )
                    else:
                        # First-start assist only: never after a collision episode.
                        # Distinct from post-collision stop (real 0 m/s into VLA).
                        # Official runtime uses command_text=None (target_point only).
                        actual_command_text = (
                            None
                            if contract_cfg.official_contract
                            else contract_cfg.command_text
                        )
                        meta: dict[str, Any] = {
                            "target_ego_1": tp1,
                            "target_ego_2": tp2,
                            "has_collided": bool(has_collided),
                            "official_contract": bool(contract_cfg.official_contract),
                            "image_layout": image_layout,
                            "route_progress_s": float(route_progress),
                            "command_text": actual_command_text,
                            "prompt_mode": (
                                "target_point"
                                if contract_cfg.official_contract
                                else "legacy_command_text"
                            ),
                            "camera_mount_xyz": list(SIMLINGO_CAMERA_XYZ),
                            "target1_distance_m": float(nav_full.target1_distance_m),
                            "target2_distance_m": float(nav_full.target2_distance_m),
                            "nav_mode": nav_full.mode,
                        }
                        if str(args.vla_version) == "v3":
                            from driving_vla.runtime.navigation_topology import (
                                observe_traffic_control_v3,
                                observe_route_context_v3,
                            )
                            from driving_vla.runtime.basic1v1_observable import (
                                basic1v1_conflict_active,
                                conflict_side_from_scene,
                                observe_basic1v1_actor,
                            )

                            route_index = max(
                                0,
                                min(
                                    len(route_xy) - 2,
                                    int(np.searchsorted(route_s, route_progress)),
                                ),
                            )
                            mission_route_xy = tuple(
                                route_xy[
                                    route_index : min(
                                        len(route_xy),
                                        route_index
                                        + (
                                            150
                                            if route_mission_base_v3 is None
                                            else 60
                                        ),
                                    )
                                ]
                            )
                            actor_observables = [
                                actor
                                for actor in world.get_actors()
                                if int(getattr(actor, "id", -1))
                                != int(getattr(ego, "id", -2))
                                and (
                                    str(getattr(actor, "type_id", "")).startswith(
                                        "vehicle."
                                    )
                                    or str(
                                        getattr(actor, "type_id", "")
                                    ).startswith("walker.")
                                )
                            ]
                            signal_state_v3, stop_line_distance_v3 = (
                                observe_traffic_control_v3(ego)
                            )
                            observed_context_v3 = observe_route_context_v3(
                                world_map=carla_map,
                                ego=ego,
                                route_xy=mission_route_xy,
                                actors=actor_observables,
                                traffic_signal_state=signal_state_v3,
                                stop_line_distance_m=stop_line_distance_v3,
                                explicit_maneuver=scenario_fixture_maneuver_v3,
                            )
                            if route_mission_base_v3 is None:
                                route_mission_base_v3 = observed_context_v3
                                _write_json(
                                    evidence_dir / "route_context_v3.json",
                                    route_mission_base_v3.to_dict(),
                                )
                            from dataclasses import replace as dataclass_replace

                            route_context_v3 = dataclass_replace(
                                observed_context_v3,
                                maneuver=route_mission_base_v3.maneuver,
                                route_xy=route_mission_base_v3.route_xy,
                                origin_road_id=route_mission_base_v3.origin_road_id,
                                origin_lane_id=route_mission_base_v3.origin_lane_id,
                                origin_lane_width_m=(
                                    route_mission_base_v3.origin_lane_width_m
                                ),
                                origin_lane_centerline_xy=(
                                    route_mission_base_v3.origin_lane_centerline_xy
                                ),
                                target_road_id=route_mission_base_v3.target_road_id,
                                target_lane_id=route_mission_base_v3.target_lane_id,
                                entry_signature=route_mission_base_v3.entry_signature,
                                exit_signature=route_mission_base_v3.exit_signature,
                                left_lane=route_mission_base_v3.left_lane,
                                right_lane=route_mission_base_v3.right_lane,
                                route_hash="",
                                topology_hash="",
                            )
                            meta["route_context_v3"] = (
                                route_context_v3.to_dict()
                            )
                            current_road_id_v3 = (
                                None
                                if driving_waypoint is None
                                else int(getattr(driving_waypoint, "road_id", 0))
                            )
                            current_lane_id_v3 = (
                                None
                                if driving_waypoint is None
                                else int(getattr(driving_waypoint, "lane_id", 0))
                            )
                            nav_prompt_v3 = resolve_navigation_prompt_conditioning(
                                maneuver=route_context_v3.maneuver,
                                target_distance_m=float(
                                    nav_full.target1_distance_m
                                ),
                                current_road_id=current_road_id_v3,
                                current_lane_id=current_lane_id_v3,
                                target_road_id=route_context_v3.target_road_id,
                                target_lane_id=route_context_v3.target_lane_id,
                            )
                            meta["prompt_mode"] = nav_prompt_v3.eval_route_as
                            meta["command_text"] = nav_prompt_v3.command_text
                            meta["navigation_prompt_conditioning"] = (
                                nav_prompt_v3.to_dict()
                            )
                            meta["current_road_id_v3"] = current_road_id_v3
                            meta["current_lane_id_v3"] = current_lane_id_v3
                            scene_v3 = observe_basic1v1_actor(
                                ego=ego,
                                actors=actor_observables,
                            )
                            meta["observable_scene_v1"] = scene_v3.to_dict()
                            meta["scenario_family_v3"] = str(
                                effective_v3_family
                            )
                            observed_conflict_active_v3 = basic1v1_conflict_active(
                                scenario_family=effective_v3_family,
                                scene=scene_v3,
                            )
                            _route_error_signed_v3 = signed_route_projection(
                                route_mission_base_v3.route_xy,
                                pose.x,
                                pose.y,
                            )[1]
                            _family_v3_lower = str(
                                effective_v3_family
                            ).lower()
                            if (
                                "obstruction" in _family_v3_lower
                                or "narrow" in _family_v3_lower
                            ):
                                _previous_overtake_phase_v3 = (
                                    v3_overtake_phase
                                )
                                v3_overtake_phase = _v3_overtake_phase(
                                    v3_overtake_phase,
                                    actor_present=scene_v3.actor_present,
                                    actor_lon_m=scene_v3.actor_lon_m,
                                    ego_route_error_m=_route_error_signed_v3,
                                )
                                v3_interaction_conflict_latched = (
                                    v3_overtake_phase != "COMPLETE"
                                )
                                if (
                                    v3_overtake_phase == "COMPLETE"
                                    and _previous_overtake_phase_v3
                                    != "COMPLETE"
                                ):
                                    v3_overtake_complete_sim_s = float(sim_s)
                            elif "cross" in _family_v3_lower:
                                _observed_side_v3 = conflict_side_from_scene(
                                    scene_v3
                                )
                                if (
                                    not v3_interaction_conflict_latched
                                    and observed_conflict_active_v3
                                    and _observed_side_v3 in {"left", "right"}
                                ):
                                    v3_crossing_initial_side = (
                                        _observed_side_v3
                                    )
                                _crossing_was_active_v3 = (
                                    v3_interaction_conflict_latched
                                )
                                v3_interaction_conflict_latched = (
                                    _v3_crossing_conflict_active(
                                        v3_interaction_conflict_latched,
                                        completed=v3_crossing_complete,
                                        observed=observed_conflict_active_v3,
                                        actor_lat_m=scene_v3.actor_lat_m,
                                        initial_side=v3_crossing_initial_side,
                                    )
                                )
                                if (
                                    _crossing_was_active_v3
                                    and not v3_interaction_conflict_latched
                                ):
                                    v3_crossing_complete = True
                            elif "cut" in _family_v3_lower:
                                _observed_side_v3 = conflict_side_from_scene(
                                    scene_v3
                                )
                                if (
                                    not v3_interaction_conflict_latched
                                    and not v3_cut_in_complete
                                    and observed_conflict_active_v3
                                    and _observed_side_v3 in {"left", "right"}
                                ):
                                    v3_conflict_side_latched = (
                                        _observed_side_v3
                                    )
                                _cut_in_was_active_v3 = (
                                    v3_interaction_conflict_latched
                                )
                                cut_in_clear_lateral_m = 4.5
                                if v3_conflict_side_latched in {"left", "right"}:
                                    away_lane_side = (
                                        "RIGHT"
                                        if v3_conflict_side_latched == "left"
                                        else "LEFT"
                                    )
                                    if not route_context_v3.lane(
                                        away_lane_side
                                    ).authorized:
                                        # Edge-lane cut-ins cannot legally
                                        # move farther into the opposite
                                        # lane.  Once the actor has crossed
                                        # the ego lane centre (~2 m), the
                                        # temporal fallback may resume.
                                        cut_in_clear_lateral_m = 2.0
                                v3_interaction_conflict_latched = (
                                    _v3_cut_in_conflict_active(
                                        v3_interaction_conflict_latched,
                                        completed=v3_cut_in_complete,
                                        observed=observed_conflict_active_v3,
                                        actor_present=scene_v3.actor_present,
                                        actor_lon_m=scene_v3.actor_lon_m,
                                        actor_lat_m=scene_v3.actor_lat_m,
                                        initial_side=v3_conflict_side_latched,
                                        ego_route_error_m=(
                                            _route_error_signed_v3
                                        ),
                                        clear_lateral_m=cut_in_clear_lateral_m,
                                    )
                                )
                                if (
                                    _cut_in_was_active_v3
                                    and not v3_interaction_conflict_latched
                                ):
                                    v3_cut_in_complete = True
                            else:
                                v3_interaction_conflict_latched = (
                                    _v3_latched_conflict_active(
                                        v3_interaction_conflict_latched,
                                        observed=observed_conflict_active_v3,
                                        scenario_family=effective_v3_family,
                                        actor_present=scene_v3.actor_present,
                                        actor_lon_m=scene_v3.actor_lon_m,
                                        ego_route_error_m=abs(
                                            _route_error_signed_v3
                                        ),
                                    )
                                )
                            conflict_active_v3 = (
                                v3_interaction_conflict_latched
                            )
                            v3_conflict_side_latched = (
                                _v3_latched_conflict_side(
                                    v3_conflict_side_latched,
                                    observed=conflict_side_from_scene(scene_v3),
                                    conflict_active=conflict_active_v3,
                                    scenario_family=effective_v3_family,
                                )
                            )
                            meta["conflict_side"] = v3_conflict_side_latched
                            meta["conflict_active_v3"] = conflict_active_v3
                            meta["ego_route_error_m"] = (
                                _route_error_signed_v3
                            )
                            meta["overtake_phase_v3"] = v3_overtake_phase
                            meta["requested_overtake_side"] = str(
                                args.v3_requested_overtake_side
                            )
                        if (
                            args.inference_mode == "full"
                            and not has_collided
                            and not ever_moved
                            and pose.speed_mps <= 0.05
                        ):
                            meta["startup_speed_assist_mps"] = 3.0
                        obs = ObservationBundle(
                            run_id="g3_vla_mpc_stable",
                            frame_id=f"carla-{camera_frame}",
                            scenario_id="pure_vla_straight_arc",
                            simulation_time_s=sim_s,
                            wall_time_s=time.time(),
                            carla_frame=camera_frame,
                            ego_x=pose.x,
                            ego_y=pose.y,
                            ego_yaw=pose.yaw,
                            ego_v=pose.speed_mps,
                            route_xy=tuple(route_xy),
                            front_rgb=image,
                            meta=meta,
                        )
                        if str(args.vla_version) == "v3":
                            anchor_rgb_path = evidence_dir / "anchor_front_rgb.npy"
                            if not anchor_rgb_path.exists():
                                np.save(anchor_rgb_path, image)
                        forward_seq = int(hang_trace["n_forward"]) + 1
                        forward_wall_start = time.time()
                        forward_sim_start = sim_s
                        hang_trace["n_forward"] = forward_seq
                        live_state["last_operation"] = "vla_forward"
                        live_state["forward_seq"] = forward_seq
                        live_state["forward_start_sim_s"] = forward_sim_start
                        live_state["forward_start_wall_s"] = forward_wall_start
                        infer_start = time.perf_counter()
                        k2_event_fields: dict[str, Any] = {}
                        frame_id_str = f"carla-{camera_frame}"
                        if str(args.vla_version) in {"v1", "v2", "v3"}:
                            # K2: one forward → guarded bundle → force/top1 → same executors.
                            bundle = policy.predict_bundle(obs)
                            latency_ms = (time.perf_counter() - infer_start) * 1000.0
                            sel_mode = (
                                "force"
                                if args.force_candidate_index is not None
                                else "top1"
                            )
                            if str(args.vla_version) == "v2":
                                assert isinstance(policy, NeuralV2Policy)
                                degraded_nominal_only = False
                                try:
                                    selection = select_k2_spatial(
                                        bundle,
                                        mode=sel_mode,  # type: ignore[arg-type]
                                        force_index=args.force_candidate_index,
                                    )
                                except K2SelectionError:
                                    reasons = set(bundle.guard_reasons)
                                    allow_nominal = bool(
                                        args.allow_v2_nominal_degraded_fallback
                                        and args.force_candidate_index == 0
                                        and reasons
                                        and reasons <= {"SPATIAL_COLLAPSE_ELIGIBLE"}
                                    )
                                    if not allow_nominal:
                                        committed_hold = path_manager.committed
                                        if (
                                            args.inference_mode != "full"
                                            or committed_hold is None
                                            or tracker is None
                                        ):
                                            raise
                                        # Rolling fail-closed behavior: never apply
                                        # the rejected bundle. Track only the last
                                        # previously accepted committed path and retry
                                        # at the next VLA period. Existing stale-path
                                        # limits stop the vehicle after 5 s without a
                                        # valid refresh.
                                        latency_ms = (
                                            time.perf_counter() - infer_start
                                        ) * 1000.0
                                        reason_key = "v2_guard_hold_last:" + ",".join(
                                            sorted(reasons)
                                        )
                                        reject_reasons[reason_key] += 1
                                        inference_ms.append(latency_ms)
                                        peak_vram_values.append(
                                            float(policy.last_peak_vram_mb)
                                        )
                                        print(
                                            f"VLA v2 Guard reject: hold_last "
                                            f"reasons={sorted(reasons)} "
                                            f"infer={latency_ms:.0f}ms",
                                            flush=True,
                                        )
                                        command = tracker.step(
                                            committed_hold,
                                            pose,
                                            measured_steer_rad=(
                                                previous_steer_norm
                                                * max_steer_rad
                                                * float(args.steer_sign)
                                            ),
                                            now_s=sim_s,
                                        )
                                        solver_modes[command.mode] += 1
                                        solver_statuses[command.solver_status] += 1
                                        steer_norm = float(
                                            np.clip(
                                                float(args.steer_sign)
                                                * command.steer_rad
                                                / max(max_steer_rad, 1e-6),
                                                -1.0,
                                                1.0,
                                            )
                                        )
                                        throttle = float(
                                            np.clip(command.accel_mps2 / 2.5, 0.0, 1.0)
                                        )
                                        brake = float(
                                            np.clip(-command.accel_mps2 / 3.0, 0.0, 1.0)
                                        )
                                        ego.apply_control(
                                            carla.VehicleControl(
                                                steer=steer_norm,
                                                throttle=throttle,
                                                brake=brake,
                                            )
                                        )
                                        previous_steer_norm = steer_norm
                                        steer_values.append(steer_norm)
                                        cte_values.append(abs(command.lateral_error_m))
                                        target_speed_values.append(command.target_speed_mps)
                                        path_age_values.append(command.path_age_s)
                                        control_seq_full.append(
                                            {
                                                "sim_s": sim_s,
                                                "throttle": throttle,
                                                "brake": brake,
                                                "steer": steer_norm,
                                                "ego_speed_mps": pose.speed_mps,
                                                "mpc_mode": str(command.mode),
                                                "solver_status": str(
                                                    command.solver_status
                                                ),
                                                "solver_ms": float(command.solver_ms),
                                                "path_age_s": float(command.path_age_s),
                                                "guard_reject_hold_last": True,
                                                "guard_reasons": sorted(reasons),
                                            }
                                        )
                                        events.append(
                                            {
                                                "sim_s": sim_s,
                                                "camera_frame": camera_frame,
                                                "latency_ms": latency_ms,
                                                "vla_version": "v2",
                                                "accepted": False,
                                                "reason": reason_key,
                                                "guard_status": bundle.guard_status,
                                                "guard_reasons": sorted(reasons),
                                                "hold_last_committed": True,
                                                "committed_source_id": str(
                                                    getattr(
                                                        committed_hold,
                                                        "source_id",
                                                        "",
                                                    )
                                                    or ""
                                                ),
                                            }
                                        )
                                        next_inference_sim_s = (
                                            sim_s + float(args.vla_period_s)
                                        )
                                        if (
                                            scenario_session_v3 is not None
                                            and apply_scenario_scripts_v3
                                            is not None
                                        ):
                                            apply_scenario_scripts_v3(
                                                scenario_session_v3,
                                                simulation_time_since_anchor_s=max(
                                                    0.0,
                                                    sim_s
                                                    - sim_start
                                                    + float(args.sim_dt),
                                                ),
                                                include_ego=False,
                                            )
                                        world.tick()
                                        if steps % spectator_period_steps == 0:
                                            spectator_follow_ok = bool(set_spectator_follow(
                                                world,
                                                ego,
                                                last_good=spectator_state,
                                            ))
                                            if not spectator_follow_ok:
                                                raise RuntimeError(
                                                    "spectator_follow_update_failed"
                                                )
                                        steps += 1
                                        continue
                                    selection = select_k2_spatial(
                                        bundle,
                                        mode="force",
                                        force_index=0,
                                        require_guard_ok=False,
                                    )
                                    degraded_nominal_only = True
                                    print(
                                        "VLA v2 defensive collapse: "
                                        "degraded_nominal_only (candidate 0 hash-checked)",
                                        flush=True,
                                    )
                            elif str(args.vla_version) == "v3":
                                assert isinstance(policy, NeuralV3Policy)
                                degraded_nominal_only = False
                                v3_force_index = args.force_candidate_index
                                if args.v3_contract_auto_select:
                                    signal_value = str(
                                        route_context_v3.traffic_signal_state.value
                                    )
                                    v3_force_index = _v3_contract_auto_index(
                                        bundle,
                                        should_yield=bool(
                                            (obs.meta or {}).get(
                                                "conflict_active_v3",
                                                False,
                                            )
                                        )
                                        or signal_value
                                        in {"RED", "STOP_SIGN"},
                                    )
                                    sel_mode = "force"
                                selection = select_k2_semantic_v3(
                                    bundle,
                                    mode=sel_mode,  # type: ignore[arg-type]
                                    force_index=v3_force_index,
                                )
                            else:
                                assert isinstance(policy, NeuralV1Policy)
                                degraded_nominal_only = False
                                selection = select_k2(
                                    bundle,
                                    mode=sel_mode,  # type: ignore[arg-type]
                                    force_index=args.force_candidate_index,
                                )
                            nav1_map = tuple(nav_full.target_map_1)
                            nav2_map = tuple(nav_full.target_map_2)
                            applied = apply_k2_to_executors(
                                selection,
                                speed_planner=speed_planner,
                                path_manager=path_manager,
                                ego=pose,
                                stamp_s=sim_s,
                                frame_id=frame_id_str,
                                dt_s=max(
                                    float(args.sim_dt),
                                    sim_s - last_speed_update_sim_s,
                                ),
                                ego_speed_mps=pose.speed_mps,
                                nav_target_map_xy=nav1_map,
                            )
                            last_speed_update_sim_s = sim_s
                            speed = applied.speed_decision
                            update = applied.path_update
                            # Native-shaped view for shared evidence fields
                            native = type("NativeView", (), {})()
                            native.path_map_xy = selection.execution_spec.spatial_path_xy
                            native.speed_mps = selection.execution_spec.speed_samples_mps
                            native.peak_vram_mb = float(policy.last_peak_vram_mb)
                            native.latency_s = float(policy.last_latency_s)
                            native.route_ego_xy = ()
                            native.speed_wps_ego_xy = ()
                            if str(args.vla_version) == "v3":
                                _attach_v3_native_feature_evidence(
                                    native,
                                    policy,
                                )
                            # Prefer resolved targets from obs meta (same as V0 path)
                            t1 = (obs.meta or {}).get("resolved_target_ego_1") or (
                                obs.meta or {}
                            ).get("target_ego_1") or (0.0, 0.0)
                            t2 = (obs.meta or {}).get("resolved_target_ego_2") or (
                                obs.meta or {}
                            ).get("target_ego_2") or (0.0, 0.0)
                            native.target_ego_1 = (float(t1[0]), float(t1[1]))
                            native.target_ego_2 = (float(t2[0]), float(t2[1]))
                            if str(args.vla_version) == "v2":
                                k2_event_fields = {
                                    "generated_candidate_ids": list(bundle.candidate_ids()),
                                    "top1_index": int(bundle.top1_index),
                                    "selected_candidate_id": selection.candidate_id,
                                    "selected_candidate_index": selection.candidate_index,
                                    "selection_mode": selection.mode,
                                    "native_path_hash": bundle.native_path_hash,
                                    "timed_trajectory_hash": (
                                        selection.execution_spec.timed_trajectory_hash
                                    ),
                                    "branch_type": bundle.branch_type,
                                    "guard_status": bundle.guard_status,
                                    "guard_reasons": list(bundle.guard_reasons),
                                    "probability_source": bundle.probability_source,
                                    "config_hash": bundle.config_hash,
                                    "candidate_available": bool(
                                        bundle.candidates[selection.candidate_index].available
                                    ),
                                    "degraded_nominal_only": bool(
                                        degraded_nominal_only
                                    ),
                                }
                            else:
                                k2_event_fields = dict(
                                    selection_event_fields(selection)
                                )
                                if args.v3_contract_auto_select:
                                    k2_event_fields["selection_policy"] = (
                                        "component_smoke_current_observable"
                                    )
                            k2_event_fields["executed_candidate_id"] = (
                                applied.executed_candidate_id
                            )
                            k2_event_fields["path_source_id"] = applied.source_id
                            k2_event_fields["forward_count"] = int(
                                policy.last_forward_count
                            )
                            requery_decision = StationaryRequeryDecision(False, "disabled_v1")
                        else:
                            assert hasattr(policy, "predict_native")
                            native = policy.predict_native(obs)
                            latency_ms = (time.perf_counter() - infer_start) * 1000.0
                            speed = speed_planner.update(
                                native.speed_mps,
                                dt_s=max(
                                    float(args.sim_dt), sim_s - last_speed_update_sim_s
                                ),
                                ego_speed_mps=pose.speed_mps,
                            )
                            last_speed_update_sim_s = sim_s
                            # Coarse nav targets in map frame (for reanchor reverse veto).
                            nav1_map = tuple(nav_full.target_map_1)
                            nav2_map = tuple(nav_full.target_map_2)
                            pre_reanchor_committed = (
                                _path_xy_as_lists(path_manager.committed)
                                if path_manager.committed is not None
                                else None
                            )
                            pre_path_age = (
                                max(0.0, sim_s - float(path_manager.committed.stamp_s))
                                if path_manager.committed is not None
                                else None
                            )
                            path_nav_aligned = True
                            if path_manager.committed is not None:
                                try:
                                    path_nav_aligned = path_manager._nav_alignment_ok(  # noqa: SLF001
                                        path_manager.committed, pose, nav1_map
                                    )
                                except Exception:
                                    path_nav_aligned = True

                            # --- optional stationary re-query (default OFF; v0 only) ---
                            if (
                                speed.stop_requested
                                and pose.speed_mps <= requery_cfg.max_ego_speed_mps
                            ):
                                consecutive_vla_stop_frames += 1
                            else:
                                consecutive_vla_stop_frames = 0
                            if enable_stationary_requery:
                                requery_decision = evaluate_stationary_requery(
                                    ego_speed_mps=pose.speed_mps,
                                    stop_requested=bool(speed.stop_requested),
                                    consecutive_stop_frames=consecutive_vla_stop_frames,
                                    has_collided=bool(has_collided),
                                    navigation_valid=bool(navigation_valid),
                                    path_nav_aligned=bool(path_nav_aligned),
                                    already_active=False,
                                    last_trigger_sim_s=last_requery_sim_s,
                                    sim_s=sim_s,
                                    config=requery_cfg,
                                )
                            else:
                                requery_decision = StationaryRequeryDecision(
                                    False, "disabled"
                                )
                            update = None
                            if requery_decision.trigger and enable_stationary_requery:
                                prev_speed_cmd = float(speed_planner.target_speed_mps)
                                meta_rq = dict(meta)
                                meta_rq.update(
                                    {
                                        "vla_input_speed_mps": float(
                                            requery_decision.hint_speed_mps
                                        ),
                                        "stationary_requery": True,
                                    }
                                )
                                obs_rq = ObservationBundle(
                                    run_id="g3_vla_mpc_stable",
                                    frame_id=f"carla-{camera_frame}-requery",
                                    scenario_id="pure_vla_straight_arc",
                                    simulation_time_s=sim_s,
                                    wall_time_s=time.time(),
                                    carla_frame=camera_frame,
                                    ego_x=pose.x,
                                    ego_y=pose.y,
                                    ego_yaw=pose.yaw,
                                    ego_v=pose.speed_mps,
                                    route_xy=tuple(route_xy),
                                    front_rgb=image,
                                    meta=meta_rq,
                                )
                                rq_t0 = time.perf_counter()
                                native_rq = policy.predict_native(obs_rq)
                                rq_latency_ms = (time.perf_counter() - rq_t0) * 1000.0
                                speed_rq = speed_planner.update(
                                    native_rq.speed_mps,
                                    dt_s=max(float(args.sim_dt), 1e-3),
                                )
                                path_ok_candidate = (
                                    (not speed_rq.stop_requested)
                                    and float(speed_rq.target_speed_mps) >= 0.35
                                )
                                update_rq = None
                                if path_ok_candidate:
                                    update_rq = path_manager.update(
                                        native_rq.path_map_xy,
                                        ego=pose,
                                        target_speed_mps=speed_rq.target_speed_mps,
                                        stamp_s=sim_s,
                                        source_id=f"simlingo-requery-{camera_frame}",
                                        nav_target_map_xy=nav1_map,
                                    )
                                ok_rq = bool(
                                    update_rq is not None
                                    and requery_success(
                                        path_accepted=bool(update_rq.accepted),
                                        stop_requested_after=bool(
                                            speed_rq.stop_requested
                                        ),
                                        target_speed_after=float(
                                            speed_rq.target_speed_mps
                                        ),
                                    )
                                )
                                last_requery_sim_s = sim_s
                                stationary_requery_log.append(
                                    {
                                        "sim_s": sim_s,
                                        "vla_n_before": len(events) + 1,
                                        "reason": requery_decision.reason,
                                        "hint_speed_mps": requery_decision.hint_speed_mps,
                                        "resolved_vla_input_speed_mps": (
                                            obs_rq.meta or {}
                                        ).get("resolved_vla_input_speed_mps"),
                                        "raw_speed_samples": [
                                            float(v) for v in native_rq.speed_mps
                                        ],
                                        "vla_speed_raw_mps": speed_rq.raw_speed_mps,
                                        "desired_speed_mps": speed_rq.target_speed_mps,
                                        "stop_requested": speed_rq.stop_requested,
                                        "path_accepted": (
                                            None
                                            if update_rq is None
                                            else update_rq.accepted
                                        ),
                                        "path_reason": (
                                            None if update_rq is None else update_rq.reason
                                        ),
                                        "success": ok_rq,
                                        "latency_ms": rq_latency_ms,
                                    }
                                )
                                if ok_rq and update_rq is not None:
                                    native = native_rq
                                    speed = speed_rq
                                    update = update_rq
                                    latency_ms += rq_latency_ms
                                    consecutive_vla_stop_frames = 0
                                else:
                                    speed_planner.reset(target_speed_mps=prev_speed_cmd)
                                    speed = type(speed)(
                                        raw_speed_mps=speed.raw_speed_mps,
                                        calibrated_speed_mps=speed.calibrated_speed_mps,
                                        target_speed_mps=prev_speed_cmd,
                                        stop_requested=speed.stop_requested,
                                        valid=speed.valid,
                                    )
                            if update is None:
                                update = path_manager.update(
                                    native.path_map_xy,
                                    ego=pose,
                                    target_speed_mps=speed.target_speed_mps,
                                    stamp_s=sim_s,
                                    source_id=f"simlingo-{camera_frame}",
                                    nav_target_map_xy=nav1_map,
                                )

                        forward_wall_end = time.time()
                        hang_trace["forwards"].append(
                            {
                                "seq": forward_seq,
                                "sim_start_s": forward_sim_start,
                                "sim_end_s": sim_s,
                                "wall_start_s": forward_wall_start,
                                "wall_end_s": forward_wall_end,
                                "latency_ms": latency_ms,
                                "camera_frame": camera_frame,
                            }
                        )
                        live_state["forward_end_wall_s"] = forward_wall_end
                        live_state["last_successful_forward_seq"] = forward_seq
                        inference_ms.append(latency_ms)
                        peak_vram_values.append(float(native.peak_vram_mb))
                        # v0 pre_reanchor only; v1 still records path age from committed
                        if str(args.vla_version) in {"v1", "v2", "v3"}:
                            pre_reanchor_committed = (
                                _path_xy_as_lists(path_manager.committed)
                                if path_manager.committed is not None
                                else None
                            )
                            pre_path_age = (
                                max(0.0, sim_s - float(path_manager.committed.stamp_s))
                                if path_manager.committed is not None
                                else None
                            )

                        if update.accepted:
                            accepts += 1
                            # Fresh path / reanchor clears execution stale-stop and arms launch recovery.
                            path_age_for_notify = (
                                max(0.0, sim_s - float(update.committed.stamp_s))
                                if update.committed is not None
                                else 0.0
                            )
                            speed_planner.notify_path_accepted(
                                reanchor=str(update.reason) == "accepted_reanchor",
                                path_age_s=path_age_for_notify,
                            )
                            # Re-evaluate speed this frame so recovery arms immediately
                            # (planner update ran before path accept).
                            speed = speed_planner.update(
                                native.speed_mps,
                                dt_s=1e-3,
                                ego_speed_mps=pose.speed_mps,
                            )
                            if (
                                str(args.vla_version) == "v3"
                                and not bool(
                                    (obs.meta or {})
                                    .get("navigation_prompt_conditioning", {})
                                    .get("target_lane_reached", False)
                                )
                            ):
                                route_change_commitment = (
                                    _v3_native_route_change_commitment(
                                        route_context_v3,
                                        selection.execution_spec.spatial_path_xy,
                                    )
                                )
                                if bool(route_change_commitment.get("ready")):
                                    v3_route_change_latch_active = True
                                    v3_route_change_latch_source = str(
                                        applied.source_id
                                    )
                                    v3_route_change_latched_speed_samples = tuple(
                                        float(value)
                                        for value in selection.execution_spec.speed_samples_mps
                                    )
                                    v3_route_change_latch_stamp_s = sim_s
                                    v3_route_change_target_hold_start_s = None
                                    v3_route_change_target_hold_progress = None
                                    v3_route_change_latch_events.append(
                                        {
                                            "sim_s": sim_s,
                                            "event": "COMMIT",
                                            "source_id": v3_route_change_latch_source,
                                            **route_change_commitment,
                                        }
                                    )
                                    k2_event_fields[
                                        "route_change_native_commitment"
                                    ] = route_change_commitment
                        else:
                            reject_reasons[update.reason] += 1
                        path_age_now = (
                            max(0.0, sim_s - float(update.committed.stamp_s))
                            if update.committed is not None
                            else None
                        )
                        raw_lists = [list(point) for point in native.path_map_xy]
                        committed_lists = _path_xy_as_lists(update.committed)
                        road_diag = _road_surface_diagnosis(
                            carla_map,
                            ego.get_location(),
                            pose.yaw,
                            carla_module=carla,
                        )
                        offroad_now = not bool(road_diag.get("on_driving"))
                        curve_cmp = compare_native_dense_curvature(
                            native.path_map_xy,
                            ds_m=0.20,
                            hard_max=float(
                                getattr(path_manager.config, "hard_max_abs_curvature", 1.0)
                            ),
                        )
                        resolved_vla_speed = None
                        if obs.meta is not None:
                            resolved_vla_speed = obs.meta.get("resolved_vla_input_speed_mps")
                        if (
                            requery_decision.trigger
                            and stationary_requery_log
                            and stationary_requery_log[-1].get("success")
                        ):
                            resolved_vla_speed = stationary_requery_log[-1].get(
                                "resolved_vla_input_speed_mps", resolved_vla_speed
                            )
                        event = {
                            "sim_s": sim_s,
                            "camera_frame": camera_frame,
                            "latency_ms": latency_ms,
                            "peak_vram_mb": float(native.peak_vram_mb),
                            "vla_version": str(args.vla_version),
                            "ego": {
                                "x": pose.x,
                                "y": pose.y,
                                "yaw": pose.yaw,
                                "speed_mps": pose.speed_mps,
                            },
                            "route_progress_m": float(
                                completed_route_progress + route_progress
                            ),
                            "target_ego_1": list(native.target_ego_1),
                            "target_ego_2": list(native.target_ego_2),
                            "target_map_1": list(nav1_map),
                            "target_map_2": list(nav2_map),
                            "resolved_prompt_mode": (
                                None
                                if obs.meta is None
                                else obs.meta.get("resolved_prompt_mode")
                            ),
                            "resolved_command_text": (
                                None
                                if obs.meta is None
                                else obs.meta.get("resolved_command_text")
                            ),
                            "navigation_prompt_conditioning": (
                                None
                                if obs.meta is None
                                else obs.meta.get(
                                    "navigation_prompt_conditioning"
                                )
                            ),
                            "raw_path_map_xy": raw_lists,
                            "committed_path_map_xy": committed_lists,
                            "speed_wps_ego_xy": [
                                [float(x), float(y)] for x, y in (native.speed_wps_ego_xy or ())
                            ],
                            "path_kappa": {
                                "raw_max": (
                                    float(np.max(np.abs(update.raw.kappa)))
                                    if update.raw is not None and update.raw.kappa.size
                                    else None
                                ),
                                "committed_max": (
                                    float(np.max(np.abs(update.committed.kappa)))
                                    if update.committed is not None and update.committed.kappa.size
                                    else None
                                ),
                                "raw_at_1m": (
                                    float(
                                        np.interp(
                                            1.0,
                                            update.raw.s,
                                            update.raw.kappa,
                                        )
                                    )
                                    if update.raw is not None and update.raw.s.size
                                    else None
                                ),
                                "committed_at_1m": (
                                    float(
                                        np.interp(
                                            1.0,
                                            update.committed.s,
                                            update.committed.kappa,
                                        )
                                    )
                                    if update.committed is not None and update.committed.s.size
                                    else None
                                ),
                            },
                            "resolved_vla_input_speed_mps": resolved_vla_speed,
                            "vla_speed_samples_mps": [float(v) for v in native.speed_mps],
                            "vla_speed_raw_mps": speed.raw_speed_mps,
                            "vla_speed_calibrated_mps": speed.calibrated_speed_mps,
                            "desired_speed_mps": speed.target_speed_mps,
                            "vla_stop_requested": speed.stop_requested,
                            "vla_stop_source": getattr(speed, "stop_source", "none"),
                            "launch_confirm_count": getattr(
                                speed, "launch_confirm_count", 0
                            ),
                            "speed_recovery_active": getattr(
                                speed, "recovery_active", False
                            ),
                            "execution_stale_latched": bool(
                                getattr(speed_planner, "execution_stale_latched", False)
                            ),
                            "consecutive_vla_stop_frames": consecutive_vla_stop_frames,
                            "accepted": update.accepted,
                            "reason": update.reason,
                            "reanchor_pending_count": int(update.reanchor_pending_count),
                            "path_age_s": path_age_now,
                            "offroad": offroad_now,
                            "road_surface": road_diag,
                            "curvature_native_vs_dense": curve_cmp,
                            "quality": update.quality.__dict__,
                            "stationary_requery": (
                                stationary_requery_log[-1]
                                if (
                                    requery_decision.trigger
                                    and stationary_requery_log
                                    and abs(
                                        float(stationary_requery_log[-1].get("sim_s", -1))
                                        - float(sim_s)
                                    )
                                    < 1e-6
                                )
                                else None
                            ),
                        }
                        if k2_event_fields:
                            event.update(k2_event_fields)
                        if str(args.vla_version) == "v3":
                            event["driving_feature"] = (
                                None
                                if native.driving_feature is None
                                else [
                                    float(value)
                                    for value in native.driving_feature
                                ]
                            )
                            event["driving_feature_hash"] = str(
                                native.driving_feature_hash or ""
                            )
                            event["driving_feature_raw_hash"] = str(
                                native.driving_feature_raw_hash or ""
                            )
                            event["scenario_family_v3"] = str(
                                (obs.meta or {}).get("scenario_family_v3")
                                or "clear"
                            )
                            event["observable_scene_v1"] = dict(
                                (obs.meta or {}).get("observable_scene_v1")
                                or {}
                            )
                            event["conflict_side"] = str(
                                (obs.meta or {}).get("conflict_side") or "none"
                            )
                            event["conflict_active_v3"] = bool(
                                (obs.meta or {}).get("conflict_active_v3", False)
                            )
                            event["traffic_signal_state"] = str(
                                route_context_v3.traffic_signal_state.value
                            )
                            event["stop_line_distance_m"] = (
                                route_context_v3.stop_line_distance_m
                            )
                        events.append(event)
                        driving_trace.append(event)
                        if offroad_now and first_offroad_snapshot is None:
                            first_offroad_snapshot = {
                                "vla_n": len(events),
                                **{k: v for k, v in event.items() if k != "front_rgb"},
                                "road_surface": road_diag,
                            }
                        # First long stop: preserve the actual frame layout for replay.
                        committed_sid = (
                            None
                            if update.committed is None
                            else str(getattr(update.committed, "source_id", "") or "")
                        )
                        if committed_sid:
                            committed_source_counts[committed_sid] += 1
                        pk = event.get("path_kappa") or {}
                        if pk.get("raw_max") is not None:
                            raw_kappa_max_series.append(float(pk["raw_max"]))
                        if pk.get("committed_max") is not None:
                            committed_kappa_max_series.append(float(pk["committed_max"]))
                        if pk.get("raw_at_1m") is not None:
                            raw_kappa_at_1m_series.append(float(pk["raw_at_1m"]))
                        if pk.get("committed_at_1m") is not None:
                            committed_kappa_at_1m_series.append(
                                float(pk["committed_at_1m"])
                            )
                        event["committed_source_id"] = committed_sid
                        event["lane_oracle"] = {
                            **lane_oracle,
                            "oracle_only": True,
                        }
                        if (
                            first_long_stop_snapshot is None
                            and consecutive_vla_stop_frames >= requery_cfg.min_stop_frames
                            and pose.speed_mps <= requery_cfg.max_ego_speed_mps
                        ):
                            snapshot_layout = str(image_layout or "rgb").lower()
                            image_path = evidence_dir / (
                                f"first_long_stop_front_{snapshot_layout}.npy"
                            )
                            try:
                                np.save(image_path, np.asarray(image))
                            except OSError:
                                image_path = None  # type: ignore[assignment]
                            first_long_stop_snapshot = {
                                "vla_n": len(events),
                                "sim_s": sim_s,
                                "observation": {
                                    "ego_x": pose.x,
                                    "ego_y": pose.y,
                                    "ego_yaw": pose.yaw,
                                    "ego_v": pose.speed_mps,
                                    "target_ego_1": list(tp1),
                                    "target_ego_2": list(tp2),
                                    "resolved_vla_input_speed_mps": resolved_vla_speed,
                                    "vla_speed_samples_mps": [
                                        float(v) for v in native.speed_mps
                                    ],
                                    "front_image_file": (
                                        None if image_path is None else image_path.name
                                    ),
                                    "front_image_layout": snapshot_layout,
                                },
                                "event": {
                                    k: v
                                    for k, v in event.items()
                                    if k
                                    not in {
                                        # keep event JSON-light
                                    }
                                },
                            }
                        if update.reason in {"reanchor_pending", "accepted_reanchor"}:
                            reanchor_snapshots.append(
                                {
                                    "vla_n": len(events),
                                    "phase": update.reason,
                                    "pre_committed_path_map_xy": pre_reanchor_committed,
                                    "pre_path_age_s": pre_path_age,
                                    "post": {
                                        "reason": update.reason,
                                        "pending_count": int(
                                            update.reanchor_pending_count
                                        ),
                                        "raw_path_map_xy": raw_lists,
                                        "committed_path_map_xy": committed_lists,
                                        "path_age_s": path_age_now,
                                        "ego": event["ego"],
                                        "target_map_1": list(nav1_map),
                                        "curvature_native_vs_dense": curve_cmp,
                                    },
                                }
                            )
                        # green=pred_route, yellow=committed, cyan=pred_speed_wps
                        speed_wps_map = None
                        if native.speed_wps_ego_xy:
                            c_yaw, s_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
                            speed_wps_map = [
                                (
                                    pose.x + c_yaw * float(x) - s_yaw * float(y),
                                    pose.y + s_yaw * float(x) + c_yaw * float(y),
                                )
                                for x, y in native.speed_wps_ego_xy
                            ]
                        if debug_draw:
                            _draw_path(
                                world,
                                update.raw,
                                update.committed,
                                # Keep two refresh periods so the watchable overlay
                                # never blinks between rolling VLA updates.
                                life_s=max(0.10, float(args.vla_period_s) * 2.0),
                                speed_wps_map_xy=speed_wps_map,
                            )
                        print(
                            f"VLA n={len(events)} {update.reason} infer={latency_ms:.0f}ms "
                            f"tp1=({tp1[0]:.1f},{tp1[1]:.1f}) "
                            f"|tp|={nav_full.target1_distance_m:.1f}/{nav_full.target2_distance_m:.1f}m "
                            f"v_raw={speed.raw_speed_mps:.2f} v_cmd={speed.target_speed_mps:.2f} "
                            f"jump5={update.quality.switch_lateral_5m:.2f}m "
                            f"age={0.0 if path_age_now is None else path_age_now:.2f}s "
                            f"pend={update.reanchor_pending_count} "
                            f"surf={road_diag.get('surface_class')} "
                            f"contract={'official' if contract_cfg.official_contract else 'legacy'}",
                            flush=True,
                        )
                        if args.gpu_idle_guard_ms > 0.0:
                            time.sleep(float(args.gpu_idle_guard_ms) / 1000.0)
                next_inference_sim_s = sim_s + float(args.vla_period_s)

            committed = path_manager.committed if path_manager is not None else None
            control_throttle = 0.0
            control_brake = 1.0
            control_steer = 0.0
            last_mpc_cmd = None
            if args.inference_mode != "full" or committed is None:
                control_brake = 1.0
                ego.apply_control(carla.VehicleControl(brake=1.0))
            else:
                assert tracker is not None
                command = tracker.step(
                    committed,
                    pose,
                    measured_steer_rad=previous_steer_norm * max_steer_rad * float(args.steer_sign),
                    now_s=sim_s,
                )
                last_mpc_cmd = command
                # Execution-layer path-stale stop (distinct from VLA semantic stop).
                if (
                    speed_planner is not None
                    and float(command.freshness_speed_limit_mps) <= 0.05
                    and float(committed.target_speed_mps) > 0.50
                    and float(command.path_age_s) >= float(
                        getattr(tracker.config, "path_stale_zero_s", 5.0)
                    )
                    - 1e-6
                ):
                    speed_planner.notify_execution_stale_stop()
                solver_modes[command.mode] += 1
                solver_statuses[command.solver_status] += 1
                steer_norm = float(
                    np.clip(
                        float(args.steer_sign) * command.steer_rad / max(max_steer_rad, 1e-6),
                        -1.0,
                        1.0,
                    )
                )
                accel = command.accel_mps2
                throttle = float(np.clip(accel / 2.5, 0.0, 1.0))
                brake = float(np.clip(-accel / 3.0, 0.0, 1.0))
                control_throttle = throttle
                control_brake = brake
                control_steer = steer_norm
                ego.apply_control(carla.VehicleControl(steer=steer_norm, throttle=throttle, brake=brake))
                previous_steer_norm = steer_norm
                steer_values.append(steer_norm)
                cte_values.append(abs(command.lateral_error_m))
                target_speed_values.append(command.target_speed_mps)
                path_age_values.append(command.path_age_s)
                sign = 1 if steer_norm > 0.05 else (-1 if steer_norm < -0.05 else 0)
                if sign and previous_sign and sign != previous_sign:
                    sign_flips += 1
                if sign:
                    previous_sign = sign

            raw_path_lists = None
            committed_path_lists = None
            committed_sid_now = None
            if path_manager is not None:
                if path_manager.committed is not None:
                    committed_path_lists = _path_xy_as_lists(path_manager.committed)
                    committed_sid_now = str(
                        getattr(path_manager.committed, "source_id", "") or ""
                    )
                if events:
                    raw_path_lists = events[-1].get("raw_path_map_xy")
            ctrl_snapshot = {
                "sim_s": sim_s,
                "carla_frame": int(getattr(snapshot, "frame", -1)),
                "speed_mps": pose.speed_mps,
                "throttle": control_throttle,
                "brake": control_brake,
                "steer": control_steer,
                "raw_path_map_xy": raw_path_lists,
                "committed_path_map_xy": committed_path_lists,
                "committed_source_id": committed_sid_now,
                "path_age_s": path_age_values[-1] if path_age_values else None,
                "has_collided": bool(has_collided),
                "road_id": lane_oracle.get("road_id"),
                "lane_id": lane_oracle.get("lane_id"),
                "is_junction": lane_oracle.get("is_junction"),
                "vla_target_speed_mps": (
                    speed_planner.target_speed_mps if speed_planner is not None else None
                ),
            }
            collision_book.update_control_snapshot(ctrl_snapshot)
            lane_invasion_book.update_control_snapshot(ctrl_snapshot)
            # Attach latest controls onto the most recent driving_trace row.
            ctrl: dict[str, Any] = {
                "sim_s": sim_s,
                "throttle": control_throttle,
                "brake": control_brake,
                "steer": control_steer,
                "ego_speed_mps": pose.speed_mps,
            }
            if last_mpc_cmd is not None:
                ctrl["lateral_error_m"] = float(last_mpc_cmd.lateral_error_m)
                ctrl["heading_error_rad"] = float(last_mpc_cmd.heading_error_rad)
                ctrl["reference_curvature"] = float(
                    last_mpc_cmd.reference_curvature
                )
                ctrl["target_speed_mps"] = float(last_mpc_cmd.target_speed_mps)
                ctrl["mpc_mode"] = str(last_mpc_cmd.mode)
                ctrl["steer_rad"] = float(last_mpc_cmd.steer_rad)
                ctrl["steer_rate_rps"] = float(last_mpc_cmd.steer_rate_rps)
                ctrl["curve_speed_limit_mps"] = float(
                    last_mpc_cmd.curve_speed_limit_mps
                )
                ctrl["horizon_speed_limit_mps"] = float(
                    last_mpc_cmd.horizon_speed_limit_mps
                )
                ctrl["freshness_speed_limit_mps"] = float(
                    last_mpc_cmd.freshness_speed_limit_mps
                )
                ctrl["path_age_s"] = float(last_mpc_cmd.path_age_s)
                ctrl["freshness_regime"] = str(
                    getattr(last_mpc_cmd, "freshness_regime", "unknown")
                )
                ctrl["solver_status"] = str(last_mpc_cmd.solver_status)
                ctrl["solver_ms"] = float(last_mpc_cmd.solver_ms)
            if speed_planner is not None:
                ctrl["vla_stop_source"] = str(
                    getattr(speed_planner, "stop_source", "none")
                )
                ctrl["execution_stale_latched"] = bool(
                    getattr(speed_planner, "execution_stale_latched", False)
                )
            # Full 20 Hz control ring (single source for offline rate/accel metrics).
            control_seq_full.append(dict(ctrl))
            if driving_trace:
                driving_trace[-1]["control"] = ctrl
                # Per-event recent window (same stream, capped for size).
                seq = driving_trace[-1].setdefault("control_seq", [])
                if isinstance(seq, list):
                    seq.append(ctrl)
                    if len(seq) > 40:
                        del seq[:-40]
            last_route_progress_m = float(completed_route_progress + route_progress)

            if str(args.vla_version) == "v3":
                from driving_vla.runtime.basic1v1_observable import (
                    basic1v1_conflict_active,
                    conflict_side_from_scene,
                    observe_basic1v1_actor,
                )
                from driving_vla.runtime.navigation_topology import (
                    observe_traffic_control_v3,
                )

                actor_pool = (
                    [
                        spawned.actor
                        for spawned in scenario_session_v3.spawned
                        if spawned.role != "ego" and spawned.actor is not None
                    ]
                    if scenario_session_v3 is not None
                    else []
                )
                scene_tick = observe_basic1v1_actor(ego=ego, actors=actor_pool)
                scene_value = scene_tick.to_dict()
                conflict_active_tick = basic1v1_conflict_active(
                    scenario_family=effective_v3_family,
                    scene=scene_tick,
                )
                signal_tick, stop_line_tick = observe_traffic_control_v3(ego)
                latest_v3_event = events[-1] if events else {}
                route_trace_context = route_mission_base_v3
                route_maneuver_value = (
                    route_trace_context.maneuver.value
                    if route_trace_context is not None
                    else str(latest_v3_event.get("route_maneuver") or "")
                )
                current_lane_invasions = int(lane_invasion_book.episode_count)
                lane_invasion_tick = (
                    current_lane_invasions > last_long_trace_lane_invasions
                )
                last_long_trace_lane_invasions = current_lane_invasions
                alternative_kind_tick = str(
                    latest_v3_event.get("alternative_kind") or ""
                )
                target_lane_side_tick = str(
                    latest_v3_event.get("target_lane_side") or "NONE"
                )
                maneuver_phase_tick = str(
                    latest_v3_event.get("maneuver_phase") or ""
                )
                _family_trace_v3 = str(effective_v3_family).lower()
                _overtake_authorization_active_v3 = bool(
                    (
                        "obstruction" in _family_trace_v3
                        or "narrow" in _family_trace_v3
                    )
                    and (
                        v3_overtake_phase != "COMPLETE"
                        or (
                            v3_overtake_complete_sim_s is not None
                            and float(sim_s)
                            - float(v3_overtake_complete_sim_s)
                            <= 2.0
                        )
                    )
                )
                _authorization_kind_v3 = alternative_kind_tick
                _authorization_side_v3 = target_lane_side_tick
                if _overtake_authorization_active_v3:
                    _authorization_kind_v3 = "SPATIAL_OVERTAKE"
                    if _authorization_side_v3 == "NONE":
                        _authorization_side_v3 = str(
                            args.v3_requested_overtake_side
                        ).upper()
                authorized_union_tick = bool(
                    route_trace_context is not None
                    and _v3_authorized_lane_corridor(
                        route_trace_context,
                        ego_x=pose.x,
                        ego_y=pose.y,
                        alternative_kind=_authorization_kind_v3,
                        target_lane_side=_authorization_side_v3,
                    )
                )
                authorized_lane_crossing = bool(
                    (lane_invasion_tick or driving_waypoint is None)
                    and (
                        (
                            route_maneuver_value
                            in {"TURN_LEFT", "TURN_RIGHT"}
                            and bool(lane_oracle.get("is_junction", False))
                        )
                        or authorized_union_tick
                    )
                )
                if driving_waypoint is None and authorized_lane_crossing:
                    # CARLA can return no Driving waypoint for a few ticks on
                    # the broken-line gap between two legal lanes.  Preserve
                    # the raw count, but do not call a topology-authorized
                    # transition an off-road event.
                    offroad_steps = max(0, offroad_steps - 1)
                    authorized_non_driving_steps += 1
                actor_distance = scene_value.get("distance_m")
                actor_clearance = (
                    max(0.0, float(actor_distance) - 4.6)
                    if actor_distance is not None
                    else None
                )
                long_horizon_trace.append(
                    {
                        "simulation_time_s": float(sim_s - sim_start),
                        "ego_x": float(pose.x),
                        "ego_y": float(pose.y),
                        "ego_yaw_rad": float(pose.yaw),
                        "ego_v": float(pose.speed_mps),
                        "route_progress_m": float(
                            completed_route_progress + route_progress
                        ),
                        "road_id": lane_oracle.get("road_id"),
                        "lane_id": lane_oracle.get("lane_id"),
                        "is_junction": bool(lane_oracle.get("is_junction", False)),
                        "collision": bool(has_collided),
                        "offroad": driving_waypoint is None,
                        "lane_invasion": lane_invasion_tick,
                        "authorized_lane_crossing": authorized_lane_crossing,
                        "path_tracking_error_m": (
                            float(last_mpc_cmd.lateral_error_m)
                            if last_mpc_cmd is not None
                            else None
                        ),
                        "mpc_status": (
                            str(last_mpc_cmd.solver_status)
                            if last_mpc_cmd is not None
                            else ""
                        ),
                        "replan_id": (
                            str(latest_v3_event.get("camera_frame"))
                            if latest_v3_event.get("camera_frame") is not None
                            else ""
                        ),
                        "selected_candidate_id": str(
                            latest_v3_event.get("selected_candidate_id") or ""
                        ),
                        "executed_candidate_id": str(
                            latest_v3_event.get("executed_candidate_id") or ""
                        ),
                        "source_id": str(
                            latest_v3_event.get("path_source_id") or ""
                        ),
                        "candidate1_available": bool(
                            latest_v3_event.get(
                                "alternative_slot_available", False
                            )
                        ),
                        "alternative_kind": alternative_kind_tick,
                        "target_lane_side": target_lane_side_tick,
                        "maneuver_phase": maneuver_phase_tick,
                        "interaction_phase_v3": (
                            v3_overtake_phase
                            if (
                                "obstruction" in _family_trace_v3
                                or "narrow" in _family_trace_v3
                            )
                            else (
                                "COMPLETE"
                                if (
                                    v3_crossing_complete
                                    or v3_cut_in_complete
                                )
                                else (
                                    "ACTIVE"
                                    if v3_interaction_conflict_latched
                                    else "CLEAR"
                                )
                            )
                        ),
                        "route_maneuver": route_maneuver_value,
                        "route_hash": (
                            str(route_trace_context.route_hash)
                            if route_trace_context is not None
                            else str(latest_v3_event.get("route_hash") or "")
                        ),
                        "topology_hash": (
                            str(route_trace_context.topology_hash)
                            if route_trace_context is not None
                            else str(latest_v3_event.get("topology_hash") or "")
                        ),
                        "actor_lon_m": scene_value.get("actor_lon_m"),
                        "actor_lat_m": scene_value.get("actor_lat_m"),
                        "actor_clearance_m": actor_clearance,
                        "conflict_side": conflict_side_from_scene(scene_tick),
                        "conflict_active": bool(
                            latest_v3_event.get(
                                "conflict_active_v3",
                                conflict_active_tick,
                            )
                        ),
                        "traffic_signal_state": signal_tick.value,
                        "stop_line_distance_m": stop_line_tick,
                        "spectator_follow_ok": bool(spectator_follow_ok),
                    }
                )

            tick_wall_start = time.time()
            live_state["last_operation"] = "world_tick"
            live_state["tick_start_sim_s"] = sim_s
            live_state["tick_start_wall_s"] = tick_wall_start
            if (
                scenario_session_v3 is not None
                and apply_scenario_scripts_v3 is not None
            ):
                apply_scenario_scripts_v3(
                    scenario_session_v3,
                    simulation_time_since_anchor_s=max(
                        0.0, sim_s - sim_start + float(args.sim_dt)
                    ),
                    include_ego=False,
                )
            world.tick()
            tick_wall_end = time.time()
            hang_trace["last_tick"] = {
                "sim_start_s": sim_s,
                "wall_start_s": tick_wall_start,
                "wall_end_s": tick_wall_end,
                "steps": steps,
            }
            live_state["tick_end_wall_s"] = tick_wall_end
            live_state["last_operation"] = "post_tick"
            with image_lock:
                hang_trace["last_camera"] = {
                    "frame": int(latest_image["frame"]),
                    "frames_received": int(latest_image["received"]),
                    "wall_time_s": latest_image.get("wall_time_s"),
                    "sim_time_s": latest_image.get("sim_time_s"),
                }
                live_state["last_camera_frame"] = hang_trace["last_camera"]
            if steps % spectator_period_steps == 0:
                spectator_follow_ok = bool(
                    set_spectator_follow(
                        world,
                        ego,
                        last_good=spectator_state,
                    )
                )
                if not spectator_follow_ok:
                    raise RuntimeError("spectator_follow_update_failed")
            steps += 1
            if sim_s >= next_checkpoint_sim_s:
                with event_lock:
                    checkpoint_road_events = dict(road_events)
                with image_lock:
                    checkpoint_camera = {
                        "last_frame": int(latest_image["frame"]),
                        "frames_received": int(latest_image["received"]),
                        "last_frame_wall_s": latest_image.get("wall_time_s"),
                    }
                inference_stats = _percentiles(inference_ms)
                checkpoint_cte_rms = (
                    float(math.sqrt(np.mean(np.square(cte_values)))) if cte_values else None
                )
                checkpoint_saturation = (
                    float(np.mean(np.abs(np.asarray(steer_values)) > 0.95))
                    if steer_values
                    else None
                )
                checkpoint = {
                    "status": "RUNNING",
                    "run_id": run_config["run_id"],
                    "inference_mode": str(args.inference_mode),
                    "map": current_map,
                    "sim_elapsed_s": sim_s - sim_start,
                    "requested_duration_s": float(args.duration_s),
                    "distance_m": distance_m,
                    "route_progress_m": completed_route_progress + route_progress,
                    "route_refreshes": route_refreshes,
                    "vla_updates": len(events),
                    "vla_accepts": accepts,
                    "vla_accept_fraction": accepts / max(len(events), 1),
                    "vla_reject_reasons": dict(reject_reasons),
                    "inference_ms": inference_stats,
                    "peak_vram_mb": max(peak_vram_values, default=0.0),
                    "model_resident_vram_mb": cuda_alloc_mb,
                    "actual_speed_mps": pose.speed_mps,
                    "vla_speed_target_mps": (
                        speed_planner.target_speed_mps if speed_planner is not None else 0.0
                    ),
                    "path_age_s": path_age_values[-1] if path_age_values else None,
                    "control": {
                        "solver_modes": dict(solver_modes),
                        "solver_statuses": dict(solver_statuses),
                        "cte_rms_m": checkpoint_cte_rms,
                        "steer_abs_max": max((abs(value) for value in steer_values), default=0.0),
                        "steer_saturation_fraction": checkpoint_saturation,
                        "steer_sign_flips": sign_flips,
                        "steer_flip_rate_hz": sign_flips / max(sim_s - sim_start, 1.0),
                    },
                    "tail_motion": _tail_motion_metrics(
                        actual_speed_values,
                        sim_dt_s=float(args.sim_dt),
                        window_s=min(20.0, sim_s - sim_start),
                    ),
                    "route_target_invalid_count": route_target_invalid,
                    "camera": checkpoint_camera,
                    "road_events": checkpoint_road_events,
                    "offroad_fraction": offroad_steps / max(steps, 1),
                    "wall_time_s": time.time(),
                }
                _write_json(evidence_dir / "progress_latest.json", checkpoint)
                _write_json(evidence_dir / "vla_events_partial.json", events)
                live_state["last_checkpoint"] = checkpoint
                next_checkpoint_sim_s += max(1.0, float(args.checkpoint_period_s))
            remaining = float(args.sim_dt) - (time.perf_counter() - loop_wall)
            if remaining > 0.0:
                time.sleep(remaining)

        displacement = math.hypot(last_xy[0] - spawn.location.x, last_xy[1] - spawn.location.y)
        mpc_steps = int(solver_modes.get("mpc", 0))
        cte_rms_value = (
            float(math.sqrt(np.mean(np.square(cte_values)))) if cte_values else float("inf")
        )
        saturation_fraction = (
            float(np.mean(np.abs(np.asarray(steer_values)) > 0.95)) if steer_values else 1.0
        )
        distance_ratio = distance_m / max(displacement, 1e-3)
        total_route_progress = completed_route_progress + route_progress
        route_progress_efficiency = min(1.0, total_route_progress / max(distance_m, 1e-3))
        offroad_fraction = offroad_steps / max(steps, 1)
        with event_lock:
            final_road_events = dict(road_events)
        collision_summary = collision_book.summary()
        lane_invasion_summary = lane_invasion_book.summary()
        final_road_events = {
            **final_road_events,
            "collisions": int(collision_summary["episode_count"]),
            "collision_episodes": int(collision_summary["episode_count"]),
            "collision_raw_events": int(collision_summary["raw_event_count"]),
            "collision_impulse": float(collision_summary["total_impulse"]),
            "first_collision_sim_s": collision_summary.get("first_collision_sim_s"),
            "first_collision_other_type": collision_summary.get("first_collision_other_type"),
            "first_collision_vehicle_pose": collision_summary.get(
                "first_collision_vehicle_pose"
            ),
            "lane_invasions": int(lane_invasion_summary["episode_count"]),
            "lane_invasion_episodes": int(lane_invasion_summary["episode_count"]),
            "lane_invasion_raw_events": int(lane_invasion_summary["raw_event_count"]),
        }
        _write_json(evidence_dir / "collision_episodes.json", collision_summary)
        _write_json(evidence_dir / "lane_invasion_episodes.json", lane_invasion_summary)
        lane_oracle_summary = lane_center_error_stats(lane_oracle_samples)
        steer_series = [float(c.get("steer", 0.0)) for c in control_seq_full]
        steer_deriv = steer_derivative_metrics(
            steer_series, dt_s=float(args.sim_dt)
        )
        steer_sign_flips_multi = multi_deadband_sign_flips(
            steer_series, deadbands=(0.01, 0.03, 0.05)
        )
        curvature_sign_flips = {
            "raw_kappa_at_1m": multi_deadband_sign_flips(
                raw_kappa_at_1m_series, deadbands=(0.01, 0.03, 0.05)
            ),
            "committed_kappa_at_1m": multi_deadband_sign_flips(
                committed_kappa_at_1m_series, deadbands=(0.01, 0.03, 0.05)
            ),
        }
        _write_json(
            evidence_dir / "driving_trace.json",
            {
                "n": len(driving_trace),
                "updates": driving_trace,
                "first_offroad": first_offroad_snapshot,
                "first_long_stop": first_long_stop_snapshot,
                "reanchor_snapshots": reanchor_snapshots,
                "stationary_requery_log": stationary_requery_log,
                "enable_stationary_requery": enable_stationary_requery,
                "committed_source_counts": dict(committed_source_counts),
                "control_seq_n": len(control_seq_full),
                "control_seq_hz": 1.0 / max(float(args.sim_dt), 1e-6),
            },
        )
        _write_jsonl(
            evidence_dir / "long_horizon_trace.jsonl",
            long_horizon_trace,
        )
        # Full 20 Hz control stream (same samples used for rate/accel metrics).
        _write_json(
            evidence_dir / "control_seq.json",
            {
                "n": len(control_seq_full),
                "dt_s": float(args.sim_dt),
                "samples": control_seq_full,
            },
        )
        if first_offroad_snapshot is not None:
            _write_json(evidence_dir / "first_offroad_snapshot.json", first_offroad_snapshot)
        if first_long_stop_snapshot is not None:
            _write_json(
                evidence_dir / "first_long_stop_snapshot.json", first_long_stop_snapshot
            )
        if reanchor_snapshots:
            _write_json(evidence_dir / "reanchor_snapshots.json", reanchor_snapshots)
        if stationary_requery_log:
            _write_json(
                evidence_dir / "stationary_requery_log.json", stationary_requery_log
            )
        with image_lock:
            final_camera_frames = int(latest_image["received"])
        minimum_distance_m = max(20.0, min(100.0, float(args.duration_s)))
        tail_motion = _tail_motion_metrics(
            actual_speed_values,
            sim_dt_s=float(args.sim_dt),
            window_s=min(20.0, float(args.duration_s)),
        )
        terminal_stop = classify_terminal_stop(
            speed_samples_mps=actual_speed_values,
            traffic_samples=traffic_light_samples,
            sim_dt_s=float(args.sim_dt),
            window_s=min(20.0, float(args.duration_s)),
        )
        # Keep raw motion fraction for evidence; do not delete old metrics.
        tail_motion = {
            **tail_motion,
            "terminal_stop_classification": terminal_stop.get(
                "terminal_stop_classification"
            ),
            "tail_acceptance_ok": bool(terminal_stop.get("tail_acceptance_ok")),
            "traffic_light_at_end": terminal_stop.get("traffic_light_at_end"),
            "terminal_stop_reason": terminal_stop.get("reason"),
        }
        if args.inference_mode == "full":
            delegate_behavior_gates_v3 = bool(
                str(args.vla_version) == "v3"
                and scenario_fixture_v3 is not None
            )
            route_change_committed_v3 = bool(
                route_change_hlc_v3
                and any(
                    str(event.get("event") or "") == "COMMIT"
                    and bool(event.get("ready"))
                    for event in v3_route_change_latch_events
                )
            )
            acceptance = {
                "enough_vla_paths": _enough_vla_paths_for_run(
                    accepts=accepts,
                    event_count=len(events),
                    route_change_committed=route_change_committed_v3,
                ),
                "mpc_fraction_ge_0_95": mpc_steps >= 0.95 * max(len(steer_values), 1),
                "cte_rms_lt_0_50m": cte_rms_value < 0.50,
                "steer_saturation_lt_0_01": saturation_fraction < 0.01,
                "steer_flip_rate_lt_0_5hz": sign_flips / max(float(args.duration_s), 1.0) < 0.5,
                "route_progress_efficiency_ge_0_65": (
                    True
                    if delegate_behavior_gates_v3
                    else route_progress_efficiency >= 0.65
                ),
                "collision_free": int(final_road_events["collisions"]) == 0,
                "offroad_fraction_lt_0_02": offroad_fraction < 0.02,
                "minimum_distance": (
                    True
                    if delegate_behavior_gates_v3
                    else distance_m > minimum_distance_m
                ),
                # Motion OK, or legal red/yellow stop (avoids false fail at lights).
                "tail_moving_or_expected_traffic_stop": bool(
                    True
                    if delegate_behavior_gates_v3
                    else terminal_stop.get("tail_acceptance_ok")
                ),
            }
            result_pass = "DEMO_PASS"
            result_fail = "DEMO_FAIL"
        else:
            # Diagnostic modes intentionally hold the ego stopped. Their only
            # purpose is to attribute a D3D failure to camera rendering, model
            # residency, or full CUDA forward load.
            acceptance = {
                "duration_completed": steps >= max(1, int(0.95 * args.duration_s / args.sim_dt)),
                "camera_stream_alive": final_camera_frames >= max(
                    2, int(0.80 * args.duration_s / camera_sensor_tick_s)
                ),
                "vehicle_remained_stopped": max(actual_speed_values, default=0.0) < 0.5,
                "collision_free": int(final_road_events["collisions"]) == 0,
            }
            if args.inference_mode == "forward-only":
                expected_forwards = max(1, int(0.80 * args.duration_s / args.vla_period_s))
                acceptance["vla_forward_alive"] = len(events) >= expected_forwards
            result_pass = "DIAGNOSTIC_PASS"
            result_fail = "DIAGNOSTIC_FAIL"
        demo_pass = all(acceptance.values())
        summary = {
            "demo": "pure_vla_spatial_constrained_mpc_v1",
            "run_id": run_config["run_id"],
            "inference_mode": str(args.inference_mode),
            "map": current_map,
            "requested_map": requested_map,
            "map_arg": str(args.map),
            "v3_case": (
                {
                    "mode": str(args.v3_mode),
                    "scenario_family": effective_v3_family,
                    "scenario_id": str(args.v3_scenario_id or ""),
                    "seed_id": str(args.v3_seed_id or ""),
                    "route_maneuver": (
                        ""
                        if route_mission_base_v3 is None
                        else route_mission_base_v3.maneuver.value
                    ),
                    "route_hash": (
                        ""
                        if route_mission_base_v3 is None
                        else route_mission_base_v3.route_hash
                    ),
                    "route_fixture_manifest_hash": (
                        route_fixture_manifest_hash_v3
                    ),
                    "scenario_registry_hash": scenario_registry_hash_v3,
                    "contract_auto_select": bool(
                        args.v3_contract_auto_select
                    ),
                }
                if str(args.vla_version) == "v3"
                else None
            ),
            "speed_cap_mps": speed_cap,
            "speed_calibration_gain": float(args.speed_gain),
            "speed_semantics": "VLA speed head is primary; calibrated then capped; no positive speed floor",
            "steps": steps,
            "sim_duration_s": float(args.duration_s),
            "distance_m": distance_m,
            "displacement_m": displacement,
            "distance_over_displacement": distance_ratio,
            "route": {
                "coarse_navigation_only": True,
                "progress_m": total_route_progress,
                "progress_efficiency": route_progress_efficiency,
                "refreshes": route_refreshes,
                "refresh_failures": route_refresh_failures,
                "target_invalid_count": route_target_invalid,
                "segment_length_m": float(route_s[-1]),
            },
            "camera": {
                "width": args.cam_w,
                "height": args.cam_h,
                "sensor_tick_s": camera_sensor_tick_s,
                "frames_received": final_camera_frames,
                "fov_deg": SIMLINGO_CAMERA_FOV_DEG,
                "mount_xyz": list(SIMLINGO_CAMERA_XYZ),
            },
            "server_resolution_requested": [int(args.server_res_x), int(args.server_res_y)],
            "requested_vehicle_blueprint": str(args.vehicle_blueprint),
            "effective_vehicle_blueprint": effective_vehicle_bp,
            "vehicle": {
                "wheelbase_m": wheelbase,
                "track_width_m": track_width_m,
                "max_steer_rad": max_steer_rad,
                "geometry_source": vehicle_geometry.geometry_source,
                "validation_status": vehicle_geometry.validation_status,
                "fields": getattr(vehicle_geometry, "fields", {}),
                "geometry": vehicle_geometry.as_dict(),
            },
            "enable_stationary_requery": enable_stationary_requery,
            "vla_updates": len(events),
            "vla_accepts": accepts,
            "vla_reject_reasons": dict(reject_reasons),
            "route_change_native_commitment": {
                "active_at_end": bool(v3_route_change_latch_active),
                "source_id": v3_route_change_latch_source,
                "stamp_s": v3_route_change_latch_stamp_s,
                "execution_refreshes": int(v3_route_change_execution_refreshes),
                "same_forward_speed_bound": bool(
                    v3_route_change_latched_speed_samples is not None
                ),
                "events": v3_route_change_latch_events,
            },
            "committed_source_counts": {
                "vla_committed": int(committed_source_counts.get("vla_committed", 0)),
                "vla_committed_latest_fallback": int(
                    committed_source_counts.get("vla_committed_latest_fallback", 0)
                ),
                "all": dict(committed_source_counts),
            },
            "path_kappa_summary": {
                "raw_max": {
                    "max": max(raw_kappa_max_series, default=None),
                    "n": len(raw_kappa_max_series),
                },
                "committed_max": {
                    "max": max(committed_kappa_max_series, default=None),
                    "n": len(committed_kappa_max_series),
                },
            },
            "curvature_sign_flips": curvature_sign_flips,
            "steer_sign_flips_multi_deadband": steer_sign_flips_multi,
            "steer_derivatives": steer_deriv,
            "lane_oracle": lane_oracle_summary,
            "lane_invasion_episodes": {
                "episode_count": int(lane_invasion_summary["episode_count"]),
                "raw_event_count": int(lane_invasion_summary["raw_event_count"]),
                "evidence_file": "lane_invasion_episodes.json",
                "measurement_only": True,
            },
            "inference_ms": {
                "p50": float(np.percentile(inference_ms, 50)) if inference_ms else 0.0,
                "p95": float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
            },
            "speed_mps": {
                "actual_p50": float(np.percentile(actual_speed_values, 50)) if actual_speed_values else 0.0,
                "actual_p95": float(np.percentile(actual_speed_values, 95)) if actual_speed_values else 0.0,
                "actual_max": max(actual_speed_values, default=0.0),
                "controller_target_p50": float(np.percentile(target_speed_values, 50)) if target_speed_values else 0.0,
                "controller_target_p95": float(np.percentile(target_speed_values, 95)) if target_speed_values else 0.0,
            },
            "path_age_s": {
                "p95": float(np.percentile(path_age_values, 95)) if path_age_values else 0.0,
                "max": max(path_age_values, default=0.0),
            },
            "tail_motion": tail_motion,
            "terminal_stop_classification": terminal_stop.get(
                "terminal_stop_classification"
            ),
            "terminal_stop": terminal_stop,
            "traffic_light_samples_n": len(traffic_light_samples),
            # Pure motion observation (legacy threshold); not an acceptance gate alone.
            "tail_moving_fraction_ge_0_50_raw": float(tail_motion["moving_fraction"])
            >= 0.50,
            "peak_vram_mb": max(peak_vram_values, default=0.0),
            "model_resident_vram_mb": cuda_alloc_mb,
            "road_events": final_road_events,
            "collision_episodes": {
                "episode_count": collision_summary["episode_count"],
                "raw_event_count": collision_summary["raw_event_count"],
                "first_collision_sim_s": collision_summary.get("first_collision_sim_s"),
                "first_collision_other_type": collision_summary.get(
                    "first_collision_other_type"
                ),
                "evidence_file": "collision_episodes.json",
            },
            "driving_trace": {
                "n": len(driving_trace),
                "evidence_file": "driving_trace.json",
                "first_offroad_vla_n": (
                    None
                    if first_offroad_snapshot is None
                    else first_offroad_snapshot.get("vla_n")
                ),
                "reanchor_snapshot_n": len(reanchor_snapshots),
            },
            "offroad_steps": offroad_steps,
            "offroad_steps_raw": offroad_steps_raw,
            "authorized_non_driving_steps": authorized_non_driving_steps,
            "offroad_fraction_raw": offroad_steps_raw / max(steps, 1),
            "offroad_fraction": offroad_fraction,
            "solver_modes": dict(solver_modes),
            "solver_statuses": dict(solver_statuses),
            # Keep evidence strict JSON even when no controller sample was produced.
            "cte_rms_m": cte_rms_value if math.isfinite(cte_rms_value) else None,
            "steer_abs_max": max((abs(v) for v in steer_values), default=0.0),
            "steer_saturation_fraction": saturation_fraction,
            "steer_sign_flips": sign_flips,
            "geometry_source": "SimLingo native pred_route only; map route used for coarse VLA targets",
            "official_contract": bool(contract_cfg.official_contract),
            "simlingo_contract": run_config.get("simlingo_contract")
            or contract_cfg.evidence_dict(),
            "requested_rhi": requested_rhi,
            "effective_rhi": effective_rhi,
            "effective_rhi_source": effective_rhi_source,
            "rhi_independently_verified": bool(rhi_verified),
            "path_manager_gates": {
                "enable_lateral_mode_flip": bool(contract_cfg.lateral_mode_flip_enabled),
                "enable_early_lane_change": bool(contract_cfg.early_lane_change_enabled),
            },
            "gpu_schedule": (
                "VLA camera rate matched; runtime CUDA synchronize; configurable host idle guard "
                "before next world.tick"
            ),
            "acceptance": acceptance,
            "result": result_pass if demo_pass else result_fail,
            "verified": False,
        }
        stamp = int(time.time() * 1000)
        _write_json(evidence_dir / f"summary_{stamp}.json", summary)
        _write_json(evidence_dir / "latest_summary.json", summary)
        _write_json(evidence_dir / f"vla_events_{stamp}.json", events)
        _write_json(
            evidence_dir / "progress_latest.json",
            {
                "status": "COMPLETE",
                "result": summary["result"],
                "run_id": run_config["run_id"],
                "inference_mode": str(args.inference_mode),
                "map": current_map,
                "sim_elapsed_s": float(args.duration_s),
                "distance_m": distance_m,
                "route_progress_m": total_route_progress,
                "road_events": final_road_events,
                "offroad_fraction": offroad_fraction,
                "summary_file": f"summary_{stamp}.json",
            },
        )
        run_config["status"] = "COMPLETE"
        run_config["result"] = summary["result"]
        run_config["wall_end_s"] = time.time()
        _write_json(evidence_dir / "run_config.json", run_config)
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if demo_pass else 2
    except Exception as exc:  # pragma: no cover - requires live CARLA failure
        error_text = str(exc)
        failure_wall_s = time.time()
        lowered = error_text.lower()
        is_tick_timeout = "time-out" in lowered and "simulator" in lowered
        if is_tick_timeout:
            # Give Unreal a short window to flush CrashContext before scanning.
            time.sleep(4.0)
            if "server_zombie" in locals():
                server_zombie = True
            else:
                server_zombie = True
            live_state["server_zombie"] = True
        crash_context = _capture_recent_carla_crash(
            evidence_dir,
            run_wall_start_s=float(run_config["wall_start_s"]),
        )
        failure_class = _classify_runtime_failure(error_text, crash_context)
        hang = hang_trace if "hang_trace" in locals() else {"forwards": [], "n_forward": 0}
        forwards = list(hang.get("forwards") or [])
        last_forward = forwards[-1] if forwards else None
        n_forward_to_failure = int(hang.get("n_forward") or len(forwards) or 0)
        seconds_since_last_forward = None
        if last_forward is not None:
            seconds_since_last_forward = float(failure_wall_s) - float(
                last_forward.get("wall_end_s") or failure_wall_s
            )
        health = _probe_server_health(resolver)
        log_tail = _capture_carla_log_tail(evidence_dir)
        hang_evidence = {
            "n_forward_to_failure": n_forward_to_failure,
            "last_successful_forward": last_forward,
            "seconds_since_last_successful_forward": seconds_since_last_forward,
            "last_tick": hang.get("last_tick"),
            "last_camera": hang.get("last_camera"),
            "forward_timeline_tail": forwards[-5:],
            "crash_context_scan_delay_s": 4.0 if is_tick_timeout else 0.0,
            "process_state": health.get("process_state"),
            "rpc_status": {
                "status": health.get("status"),
                "tcp_reachable": health.get("tcp_reachable"),
                "rpc_reachable": health.get("rpc_reachable"),
                "error_code": health.get("error_code"),
                "error_message": health.get("error_message"),
            },
            "carla_log": {
                "source_path": (log_tail or {}).get("source_path"),
                "copied_file": (log_tail or {}).get("copied_file"),
                "tail_line_count": (log_tail or {}).get("tail_line_count"),
            }
            if log_tail
            else None,
        }
        live_state["hang_evidence"] = hang_evidence
        failure = {
            "status": "BLOCKED_EXTERNAL",
            "result": "RUNTIME_ERROR",
            "run_id": run_config["run_id"],
            "inference_mode": str(args.inference_mode),
            "requested_map": requested_map,
            "error_type": type(exc).__name__,
            "error": error_text,
            "failure_class": failure_class,
            "n_forward_to_failure": n_forward_to_failure,
            "seconds_since_last_successful_forward": seconds_since_last_forward,
            "hang_evidence": hang_evidence,
            "carla_crash_context": crash_context,
            "carla_process_state": health.get("process_state"),
            "carla_rpc_status": hang_evidence["rpc_status"],
            "last_live_state": live_state,
            "run_config": run_config,
            "traceback": traceback.format_exc(),
            "wall_time_s": failure_wall_s,
        }
        if "collision_book" in locals() and isinstance(collision_book, CollisionEpisodeBook):
            try:
                collision_summary = collision_book.summary()
                failure["collision_episodes"] = collision_summary
                _write_json(evidence_dir / "collision_episodes.json", collision_summary)
            except Exception:
                pass
        if "lane_invasion_book" in locals() and isinstance(
            lane_invasion_book, LaneInvasionEpisodeBook
        ):
            try:
                lane_summary = lane_invasion_book.summary()
                failure["lane_invasion_episodes"] = lane_summary
                _write_json(evidence_dir / "lane_invasion_episodes.json", lane_summary)
            except Exception:
                pass
        if "driving_trace" in locals() and isinstance(driving_trace, list):
            try:
                _write_json(
                    evidence_dir / "driving_trace.json",
                    {
                        "n": len(driving_trace),
                        "updates": driving_trace,
                        "first_offroad": locals().get("first_offroad_snapshot"),
                        "reanchor_snapshots": locals().get("reanchor_snapshots") or [],
                        "committed_source_counts": dict(
                            locals().get("committed_source_counts") or {}
                        ),
                    },
                )
            except Exception:
                pass
        if "long_horizon_trace" in locals() and isinstance(
            long_horizon_trace, list
        ):
            try:
                _write_jsonl(
                    evidence_dir / "long_horizon_trace.jsonl",
                    long_horizon_trace,
                )
            except Exception:
                pass
        if "control_seq_full" in locals() and isinstance(control_seq_full, list):
            try:
                _write_json(
                    evidence_dir / "control_seq.json",
                    {
                        "n": len(control_seq_full),
                        "dt_s": float(args.sim_dt),
                        "samples": control_seq_full,
                    },
                )
            except Exception:
                pass
        if str(args.vla_version) == "v3":
            last_v3_bundle = getattr(locals().get("policy"), "last_bundle", None)
            if last_v3_bundle is not None:
                try:
                    _write_json(
                        evidence_dir / "k2_v3_failure_bundle.json",
                        last_v3_bundle.to_dict(),
                    )
                except Exception:
                    pass
        _write_json(evidence_dir / "failure_latest.json", failure)
        if "events" in locals() and isinstance(events, list):
            _write_json(evidence_dir / "vla_events_partial.json", events)
        run_config["status"] = "FAILED"
        run_config["failure_class"] = failure["failure_class"]
        run_config["n_forward_to_failure"] = n_forward_to_failure
        run_config["wall_end_s"] = failure_wall_s
        _write_json(evidence_dir / "run_config.json", run_config)
        print(json.dumps(failure, indent=2), flush=True)
        return 75
    finally:
        zombie = bool(locals().get("server_zombie")) or bool(
            live_state.get("server_zombie") if isinstance(live_state, dict) else False
        )
        if zombie:
            # Zombie/hanging CARLA: one best-effort note, skip multi 30s destroy timeouts.
            print(
                "cleanup: server appears unresponsive; skipping actor destroy timeouts "
                "(single best-effort pass)",
                flush=True,
            )
            for actor in (lane_sensor, collision_sensor, camera, ego):
                if actor is None:
                    continue
                try:
                    if hasattr(actor, "stop"):
                        actor.stop()
                except Exception:
                    pass
        else:
            for sensor in (lane_sensor, collision_sensor):
                if sensor is not None:
                    try:
                        sensor.stop()
                        sensor.destroy()
                    except Exception:
                        pass
            if camera is not None:
                try:
                    camera.stop()
                    camera.destroy()
                except Exception:
                    pass
            if scenario_session_v3 is not None:
                try:
                    from driving_vla.evaluation.fixture_runtime import (
                        cleanup_session,
                    )

                    cleanup_session(scenario_session_v3, soft=True)
                    ego = None
                except Exception:
                    pass
            elif ego is not None:
                try:
                    ego.destroy()
                except Exception:
                    pass
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
