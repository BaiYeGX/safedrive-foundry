"""Observation-only lane invasion episodes and lane-center oracle.

These helpers produce evidence only.  Outputs must never enter VLA, PathManager,
MPC, speed planning, or control decisions.
"""

from __future__ import annotations

import math
import threading
from typing import Any

DEFAULT_LANE_INVASION_GAP_S = 0.50


def observe_lane_oracle(
    carla_map: Any,
    location: Any,
    ego_yaw: float,
    *,
    carla_module: Any,
) -> dict[str, Any]:
    """Nearest driving waypoint + signed lateral error to lane center.

    Always sets ``oracle_only=True``.  Never used for control.
    """
    out: dict[str, Any] = {
        "oracle_only": True,
        "road_id": None,
        "lane_id": None,
        "is_junction": None,
        "lane_width_m": None,
        "lane_center_error_m": None,
        "lane_heading_deg": None,
        "ok": False,
    }
    try:
        drive_wp = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
        if drive_wp is None:
            out["error"] = "no_driving_waypoint"
            return out
        out["road_id"] = int(drive_wp.road_id)
        out["lane_id"] = int(drive_wp.lane_id)
        out["is_junction"] = bool(getattr(drive_wp, "is_junction", False))
        try:
            out["lane_width_m"] = float(drive_wp.lane_width)
        except Exception:
            out["lane_width_m"] = None
        tf = drive_wp.transform
        dx = float(location.x) - float(tf.location.x)
        dy = float(location.y) - float(tf.location.y)
        road_yaw = math.radians(float(tf.rotation.yaw))
        out["lane_heading_deg"] = math.degrees(road_yaw)
        # Signed lateral in road frame: +left of lane heading.
        signed = -math.sin(road_yaw) * dx + math.cos(road_yaw) * dy
        out["lane_center_error_m"] = float(signed)
        out["ok"] = True
    except Exception as exc:  # pragma: no cover - live CARLA only
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def lane_center_error_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """RMS / P95 / max of |lane_center_error_m|, overall and by junction flag."""

    def _stats(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"n": 0, "rms": None, "p95": None, "max": None}
        arr = [abs(float(v)) for v in vals]
        n = len(arr)
        rms = math.sqrt(sum(v * v for v in arr) / n)
        ordered = sorted(arr)
        p95 = ordered[min(n - 1, int(math.ceil(0.95 * n) - 1))]
        return {"n": n, "rms": rms, "p95": p95, "max": max(arr)}

    all_e: list[float] = []
    junc: list[float] = []
    non_junc: list[float] = []
    for s in samples:
        if not s or not s.get("ok"):
            continue
        err = s.get("lane_center_error_m")
        if err is None:
            continue
        e = float(err)
        all_e.append(e)
        if s.get("is_junction"):
            junc.append(e)
        else:
            non_junc.append(e)
    return {
        "oracle_only": True,
        "all": _stats(all_e),
        "junction": _stats(junc),
        "non_junction": _stats(non_junc),
    }


class LaneInvasionEpisodeBook:
    """Merge short-window lane-marking callbacks into invasion episodes."""

    def __init__(self, *, gap_s: float = DEFAULT_LANE_INVASION_GAP_S) -> None:
        self.gap_s = float(gap_s)
        self.raw_event_count = 0
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
        with self.lock:
            self._latest_snapshot = dict(snapshot)

    def _close_open_unlocked(self) -> None:
        if self._open is not None:
            self.episodes.append(self._open)
            self._open = None

    @staticmethod
    def _markings_payload(crossed_markings: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not crossed_markings:
            return out
        for m in crossed_markings:
            entry: dict[str, Any] = {}
            try:
                mtype = getattr(m, "type", None)
                entry["type"] = str(mtype.name) if hasattr(mtype, "name") else str(mtype)
            except Exception:
                entry["type"] = str(getattr(m, "type", "unknown"))
            try:
                mcolor = getattr(m, "color", None)
                entry["color"] = (
                    str(mcolor.name) if hasattr(mcolor, "name") else str(mcolor)
                )
            except Exception:
                entry["color"] = str(getattr(m, "color", "unknown"))
            try:
                entry["lane_change"] = str(getattr(m, "lane_change", None))
            except Exception:
                entry["lane_change"] = None
            out.append(entry)
        return out

    def on_invasion(
        self,
        *,
        crossed_markings: Any = None,
        carla_frame: int | None = None,
        sim_s: float | None = None,
        ego_pose: dict[str, Any] | None = None,
        speed_mps: float | None = None,
        steer: float | None = None,
        throttle: float | None = None,
        brake: float | None = None,
        raw_path: Any = None,
        committed_path: Any = None,
        committed_source_id: str | None = None,
        road_id: int | None = None,
        lane_id: int | None = None,
        is_junction: bool | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            stamp = float(self._current_sim_s if sim_s is None else sim_s)
            self.raw_event_count += 1
            markings = self._markings_payload(crossed_markings)
            snap = dict(self._latest_snapshot)
            pose = dict(ego_pose or {})
            continuous = (
                self._open is not None
                and self._open.get("last_sim_s") is not None
                and (stamp - float(self._open["last_sim_s"])) <= self.gap_s
            )
            if continuous:
                episode = self._open
                assert episode is not None
                episode["raw_event_count"] = int(episode["raw_event_count"]) + 1
                episode["last_sim_s"] = stamp
                episode["duration_s"] = stamp - float(episode["first_sim_s"])
                episode["markings_union"].extend(markings)
                if carla_frame is not None:
                    episode["carla_frames"].append(int(carla_frame))
            else:
                self._close_open_unlocked()
                episode = {
                    "episode_index": len(self.episodes),
                    "first_sim_s": stamp,
                    "last_sim_s": stamp,
                    "duration_s": 0.0,
                    "raw_event_count": 1,
                    "carla_frame_first": None if carla_frame is None else int(carla_frame),
                    "carla_frames": [] if carla_frame is None else [int(carla_frame)],
                    "crossed_lane_markings": markings,
                    "markings_union": list(markings),
                    "ego_pose_first": pose,
                    "speed_mps": speed_mps if speed_mps is not None else snap.get("speed_mps"),
                    "steer": steer if steer is not None else snap.get("steer"),
                    "throttle": throttle if throttle is not None else snap.get("throttle"),
                    "brake": brake if brake is not None else snap.get("brake"),
                    "raw_path_map_xy": raw_path
                    if raw_path is not None
                    else snap.get("raw_path_map_xy"),
                    "committed_path_map_xy": committed_path
                    if committed_path is not None
                    else snap.get("committed_path_map_xy"),
                    "committed_source_id": committed_source_id
                    if committed_source_id is not None
                    else snap.get("committed_source_id"),
                    "road_id": road_id if road_id is not None else snap.get("road_id"),
                    "lane_id": lane_id if lane_id is not None else snap.get("lane_id"),
                    "is_junction": is_junction
                    if is_junction is not None
                    else snap.get("is_junction"),
                    "pre_invasion_control": dict(snap),
                }
                self._open = episode
            return dict(episode)

    def finalize(self) -> list[dict[str, Any]]:
        with self.lock:
            self._close_open_unlocked()
            return list(self.episodes)

    def summary(self) -> dict[str, Any]:
        episodes = self.finalize()
        return {
            "episode_count": len(episodes),
            "raw_event_count": self.raw_event_count,
            "gap_s": self.gap_s,
            "episodes": episodes,
            "evidence_note": "measurement_only; not an acceptance gate",
        }


# ---------------------------------------------------------------------------
# Control / curvature sign-flip metrics (observation only)
# ---------------------------------------------------------------------------


def multi_deadband_sign_flips(
    values: list[float],
    *,
    deadbands: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> dict[str, int]:
    """Count sign flips of a scalar series under several deadbands."""
    out: dict[str, int] = {}
    for db in deadbands:
        flips = 0
        prev = 0
        for v in values:
            x = float(v)
            sign = 1 if x > db else (-1 if x < -db else 0)
            if sign and prev and sign != prev:
                flips += 1
            if sign:
                prev = sign
        out[f"deadband_{db:g}"] = flips
    return out


def steer_derivative_metrics(
    steer_values: list[float],
    *,
    dt_s: float,
) -> dict[str, Any]:
    """|Δsteer| and finite-difference rate / acceleration percentiles."""
    if len(steer_values) < 2 or dt_s <= 0.0:
        return {
            "n": len(steer_values),
            "delta_steer_abs": {"p50": None, "p95": None, "p99": None, "max": None},
            "steer_rate_abs": {"p50": None, "p95": None, "p99": None, "max": None},
            "steer_accel_abs": {"p50": None, "p95": None, "p99": None, "max": None},
        }
    arr = [float(v) for v in steer_values]
    deltas = [abs(arr[i] - arr[i - 1]) for i in range(1, len(arr))]
    rates = [d / dt_s for d in deltas]
    accels: list[float] = []
    for i in range(1, len(rates)):
        accels.append(abs(rates[i] - rates[i - 1]) / dt_s)

    def _pct(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"p50": None, "p95": None, "p99": None, "max": None}
        import numpy as np

        a = np.asarray(vals, dtype=float)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)),
        }

    return {
        "n": len(arr),
        "dt_s": float(dt_s),
        "delta_steer_abs": _pct(deltas),
        "steer_rate_abs": _pct(rates),
        "steer_accel_abs": _pct(accels),
    }


__all__ = [
    "DEFAULT_LANE_INVASION_GAP_S",
    "LaneInvasionEpisodeBook",
    "observe_lane_oracle",
    "lane_center_error_stats",
    "multi_deadband_sign_flips",
    "steer_derivative_metrics",
]
