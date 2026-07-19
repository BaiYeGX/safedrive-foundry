"""Yaw-invariant vehicle geometry estimation for G3 MPC.

Wheel positions from CARLA may be in vehicle-local or world coordinates and may
use centimetres.  Axis extents along world X are *not* used — they swap with yaw.

Each of wheelbase / track / max_steer is validated independently; a single field
failure must not discard other valid physics fields.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

# Default ego for pure VLA + MPC live (CARLA 0.9.16).
DEFAULT_EGO_BLUEPRINT = "vehicle.mercedes.coupe_2020"

# Explicit, audited per-field fallback only when that field is unusable.
# Never applied silently; field-level source always names this path.
MERCEDES_COUPE_2020_VALIDATED_FALLBACK: dict[str, float] = {
    "wheelbase_m": 2.70,
    "track_width_m": 1.55,
    "max_steer_rad": 0.70,
}

# CARLA Mercedes coupe reports ~70° (~1.222 rad); allow up to ~80°.
MAX_STEER_PLAUSIBLE_RAD = 1.40
MIN_STEER_PLAUSIBLE_RAD = 0.20
MIN_WHEELBASE_M = 1.8
MAX_WHEELBASE_M = 4.0
MIN_TRACK_M = 0.8
MAX_TRACK_M = 2.5


@dataclass(frozen=True)
class FieldGeometry:
    """One scalar geometry field with independent provenance."""

    value: float
    source: str
    validation_status: str
    physics_value: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VehicleGeometryResult:
    """Structured geometry for evidence + MPC config."""

    wheelbase_m: float
    track_width_m: float
    max_steer_rad: float
    geometry_source: str
    validation_status: str
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    wheel_positions_raw: list[list[float]] = field(default_factory=list)
    wheel_positions_m: list[list[float]] = field(default_factory=list)
    units_detected: str | None = None
    n_wheels: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_xy_array(wheel_positions: Sequence[Sequence[float]]) -> np.ndarray:
    pts: list[list[float]] = []
    for p in wheel_positions:
        if p is None:
            continue
        if len(p) < 2:
            continue
        pts.append([float(p[0]), float(p[1])])
    if len(pts) < 2:
        raise ValueError(f"need at least 2 wheel XY points, got {len(pts)}")
    return np.asarray(pts, dtype=float)


def normalize_wheel_units(
    wheel_xy: Sequence[Sequence[float]],
) -> tuple[np.ndarray, str]:
    """Detect cm vs m and return points in metres."""
    pts = _as_xy_array(wheel_xy)
    span = float(np.max(np.ptp(pts, axis=0))) if pts.size else 0.0
    max_abs = float(np.max(np.abs(pts))) if pts.size else 0.0
    # CARLA physics wheels are typically centimetres (~100–200 for half-length).
    if span > 20.0 or max_abs > 20.0:
        return pts / 100.0, "cm_to_m"
    return pts, "m"


def estimate_wheelbase_track_m(
    wheel_xy_m: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Rotation-invariant wheelbase and track from 2D wheel points (metres).

    Uses PCA principal axes, then clusters projections into front/rear (major)
    and left/right (minor).  Does **not** assume world X is vehicle forward.
    """
    pts = _as_xy_array(wheel_xy_m)
    mean = pts.mean(axis=0)
    centered = pts - mean
    if centered.shape[0] >= 2:
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov), 0.0], [0.0, float(cov)]])
        if cov.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            major = eigvecs[:, order[0]]
            minor = eigvecs[:, order[1]]
        else:
            major = np.array([1.0, 0.0])
            minor = np.array([0.0, 1.0])
    else:
        major = np.array([1.0, 0.0])
        minor = np.array([0.0, 1.0])
    major = major / max(float(np.linalg.norm(major)), 1e-12)
    minor = minor / max(float(np.linalg.norm(minor)), 1e-12)

    proj_major = centered @ major
    proj_minor = centered @ minor

    def _two_cluster_span(proj: np.ndarray) -> float:
        if proj.size < 2:
            return 0.0
        order = np.argsort(proj)
        if proj.size >= 4:
            low = proj[order[:2]]
            high = proj[order[-2:]]
            return abs(float(high.mean()) - float(low.mean()))
        mid = float(np.median(proj))
        lo = proj[proj <= mid]
        hi = proj[proj >= mid]
        if lo.size == 0 or hi.size == 0:
            return float(np.ptp(proj))
        return abs(float(hi.mean()) - float(lo.mean()))

    wheelbase = max(0.0, float(_two_cluster_span(proj_major)))
    track = max(0.0, float(_two_cluster_span(proj_minor)))
    return wheelbase, track


def plausible_wheelbase_m(wheelbase_m: float) -> bool:
    return MIN_WHEELBASE_M <= float(wheelbase_m) <= MAX_WHEELBASE_M


def plausible_track_width_m(
    track_width_m: float, *, wheelbase_m: float | None = None
) -> bool:
    t = float(track_width_m)
    if not (MIN_TRACK_M <= t <= MAX_TRACK_M):
        return False
    # Guard axis swap when a reliable wheelbase is available.
    if wheelbase_m is not None and plausible_wheelbase_m(wheelbase_m):
        if t > float(wheelbase_m) * 0.95:
            return False
    return True


def plausible_max_steer_rad(max_steer_rad: float) -> bool:
    return MIN_STEER_PLAUSIBLE_RAD <= float(max_steer_rad) <= MAX_STEER_PLAUSIBLE_RAD


def _resolve_field(
    *,
    name: str,
    physics_value: float | None,
    plausible: bool,
    allow_mercedes_fallback: bool,
    is_mercedes: bool,
    notes: list[str],
) -> FieldGeometry:
    fb = MERCEDES_COUPE_2020_VALIDATED_FALLBACK[name]
    if physics_value is not None and plausible:
        return FieldGeometry(
            value=float(physics_value),
            source="physics_wheels_pca" if name != "max_steer_rad" else "physics_wheels",
            validation_status="ok",
            physics_value=float(physics_value),
        )
    if physics_value is not None and not plausible:
        notes.append(f"{name}_physics_implausible:{physics_value:.6g}")
    elif physics_value is None:
        notes.append(f"{name}_physics_missing")

    if allow_mercedes_fallback and is_mercedes:
        return FieldGeometry(
            value=float(fb),
            source="mercedes_coupe_2020_validated_fallback",
            validation_status="fallback_used",
            physics_value=None if physics_value is None else float(physics_value),
            notes=[f"{name}_explicit_mercedes_validated_fallback"],
        )
    raise RuntimeError(
        f"geometry field {name} unusable and no allowed validated fallback "
        f"(physics={physics_value!r})"
    )


def estimate_vehicle_geometry_from_wheels(
    wheel_positions: Sequence[Sequence[float]],
    *,
    max_steer_deg: float | None = None,
    max_steer_rad: float | None = None,
    allow_mercedes_fallback: bool = True,
    blueprint_id: str | None = None,
) -> VehicleGeometryResult:
    """Estimate geometry with independent per-field validation and fallback."""
    notes: list[str] = []
    raw_list: list[list[float]] = []
    for p in wheel_positions:
        if p is None or len(p) < 2:
            continue
        row = [float(p[0]), float(p[1])]
        if len(p) > 2:
            row.append(float(p[2]))
        raw_list.append(row)
    n_wheels = len(raw_list)

    steer_physics: float | None = None
    if max_steer_rad is not None:
        steer_physics = float(max_steer_rad)
    elif max_steer_deg is not None and float(max_steer_deg) > 1.0:
        steer_physics = math.radians(float(max_steer_deg))

    wheel_m_list: list[list[float]] = []
    units: str | None = None
    wb_physics: float | None = None
    tr_physics: float | None = None
    try:
        if n_wheels < 2:
            raise ValueError("fewer than 2 wheel positions")
        pts_m, units = normalize_wheel_units(raw_list)
        wheel_m_list = pts_m.tolist()
        wb_physics, tr_physics = estimate_wheelbase_track_m(pts_m)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"wheel_pca_error:{type(exc).__name__}:{exc}")

    bp = str(blueprint_id or "")
    is_mercedes = (not bp) or ("mercedes" in bp.lower()) or (bp == DEFAULT_EGO_BLUEPRINT)
    if not bp:
        notes.append("blueprint_unspecified_mercedes_fallback_allowed_by_caller")

    # Independent field resolution — steer failure must not wipe wb/track.
    try:
        wb_field = _resolve_field(
            name="wheelbase_m",
            physics_value=wb_physics,
            plausible=wb_physics is not None and plausible_wheelbase_m(wb_physics),
            allow_mercedes_fallback=allow_mercedes_fallback,
            is_mercedes=is_mercedes,
            notes=notes,
        )
        # Track uses accepted wheelbase when physics wb is good, else fallback wb
        # only for axis-swap check after wb is resolved.
        tr_plausible = False
        if tr_physics is not None:
            ref_wb = (
                float(wb_physics)
                if wb_physics is not None and plausible_wheelbase_m(wb_physics)
                else float(wb_field.value)
            )
            tr_plausible = plausible_track_width_m(tr_physics, wheelbase_m=ref_wb)
        tr_field = _resolve_field(
            name="track_width_m",
            physics_value=tr_physics,
            plausible=tr_plausible,
            allow_mercedes_fallback=allow_mercedes_fallback,
            is_mercedes=is_mercedes,
            notes=notes,
        )
        steer_field = _resolve_field(
            name="max_steer_rad",
            physics_value=steer_physics,
            plausible=(
                steer_physics is not None and plausible_max_steer_rad(steer_physics)
            ),
            allow_mercedes_fallback=allow_mercedes_fallback,
            is_mercedes=is_mercedes,
            notes=notes,
        )
    except RuntimeError:
        raise

    statuses = {
        "wheelbase_m": wb_field.validation_status,
        "track_width_m": tr_field.validation_status,
        "max_steer_rad": steer_field.validation_status,
    }
    sources = {
        "wheelbase_m": wb_field.source,
        "track_width_m": tr_field.source,
        "max_steer_rad": steer_field.source,
    }
    n_ok = sum(1 for s in statuses.values() if s == "ok")
    n_fb = sum(1 for s in statuses.values() if s == "fallback_used")
    if n_ok == 3:
        overall_source = "physics_wheels_pca"
        overall_status = "ok"
    elif n_fb == 3:
        overall_source = "mercedes_coupe_2020_validated_fallback"
        overall_status = "fallback_used"
    else:
        overall_source = "mixed_physics_and_fallback"
        overall_status = "partial_fallback"

    fields = {
        "wheelbase_m": wb_field.as_dict(),
        "track_width_m": tr_field.as_dict(),
        "max_steer_rad": steer_field.as_dict(),
    }
    if n_fb:
        notes.append("explicit_per_field_fallback")
        notes.append(f"field_sources={sources}")

    return VehicleGeometryResult(
        wheelbase_m=float(wb_field.value),
        track_width_m=float(tr_field.value),
        max_steer_rad=float(steer_field.value),
        geometry_source=overall_source,
        validation_status=overall_status,
        fields=fields,
        wheel_positions_raw=raw_list,
        wheel_positions_m=wheel_m_list,
        units_detected=units,
        n_wheels=n_wheels,
        notes=list(notes),
    )


def vehicle_geometry_from_carla_vehicle(
    vehicle: Any,
    *,
    blueprint_id: str | None = None,
) -> VehicleGeometryResult:
    """Read CARLA physics wheels and estimate geometry (live helper)."""
    bp = blueprint_id
    if bp is None:
        try:
            bp = str(vehicle.type_id)
        except Exception:
            bp = DEFAULT_EGO_BLUEPRINT

    positions: list[list[float]] = []
    max_steer_deg = 0.0
    try:
        phys = vehicle.get_physics_control()
        wheels = list(getattr(phys, "wheels", []) or [])
        for w in wheels:
            pos = getattr(w, "position", None)
            if pos is not None:
                positions.append(
                    [float(pos.x), float(pos.y), float(getattr(pos, "z", 0.0))]
                )
            try:
                max_steer_deg = max(
                    max_steer_deg, float(getattr(w, "max_steer_angle", 0.0))
                )
            except (TypeError, ValueError):
                pass
    except Exception as exc:  # noqa: BLE001
        result = estimate_vehicle_geometry_from_wheels(
            [],
            blueprint_id=bp,
            allow_mercedes_fallback=True,
        )
        return VehicleGeometryResult(
            wheelbase_m=result.wheelbase_m,
            track_width_m=result.track_width_m,
            max_steer_rad=result.max_steer_rad,
            geometry_source=result.geometry_source,
            validation_status=result.validation_status,
            fields=result.fields,
            wheel_positions_raw=result.wheel_positions_raw,
            wheel_positions_m=result.wheel_positions_m,
            units_detected=result.units_detected,
            n_wheels=result.n_wheels,
            notes=list(result.notes)
            + [f"physics_control_error:{type(exc).__name__}:{exc}"],
        )

    return estimate_vehicle_geometry_from_wheels(
        positions,
        max_steer_deg=max_steer_deg if max_steer_deg > 1.0 else None,
        blueprint_id=bp,
        allow_mercedes_fallback=True,
    )


__all__ = [
    "DEFAULT_EGO_BLUEPRINT",
    "MERCEDES_COUPE_2020_VALIDATED_FALLBACK",
    "MAX_STEER_PLAUSIBLE_RAD",
    "FieldGeometry",
    "VehicleGeometryResult",
    "normalize_wheel_units",
    "estimate_wheelbase_track_m",
    "plausible_wheelbase_m",
    "plausible_track_width_m",
    "plausible_max_steer_rad",
    "estimate_vehicle_geometry_from_wheels",
    "vehicle_geometry_from_carla_vehicle",
]
