"""Scenario Registry V1 loader, validation, freeze, and actor scripts.

R2 freezes fixtures before any candidate outcome is observed. This module
rejects free spawn, random TM-driven core actors, and candidate-dependent
scripts.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from driving_vla.evaluation.paired_contract import (
    ContractError,
    TransformPose,
    VelocityState,
    canonical_json_bytes,
    content_hash,
    sha256_hex,
)

REGISTRY_SCHEMA = "safedrive.g4a.scenario_registry.v1"
REGISTRY_VERSION_DEFAULT = "v1"
ALLOWED_FAMILIES = frozenset({"lead_braking", "cut_in", "crossing"})
ALLOWED_VEHICLE_SCRIPTS = frozenset(
    {
        "hold",
        "constant_throttle",
        "constant_brake",
        "piecewise_vehicle_control",
    }
)
ALLOWED_WALKER_SCRIPTS = frozenset({"walker_control"})
FORBIDDEN_SCRIPT_KEYS = frozenset(
    {
        "candidate_id",
        "oracle",
        "ego_future",
        "random",
        "traffic_manager",
        "distance_trigger",
        "free_spawn",
        "fallback_spawn",
    }
)
REQUIRED_SCENARIO_IDS = (
    "lead_brake_moderate",
    "lead_brake_hard",
    "cut_in_early",
    "cut_in_late",
    "cross_vehicle_clear",
    "cross_vehicle_tight",
)
REQUIRED_SEEDS = ("seed_a", "seed_b")


class RegistryError(ContractError):
    """Invalid registry TOML or freeze attempt."""


@dataclass(frozen=True)
class WeatherSpec:
    preset: str
    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    wetness: float
    fog_density: float = 0.0
    fog_distance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "cloudiness": float(self.cloudiness),
            "precipitation": float(self.precipitation),
            "precipitation_deposits": float(self.precipitation_deposits),
            "wind_intensity": float(self.wind_intensity),
            "sun_azimuth_angle": float(self.sun_azimuth_angle),
            "sun_altitude_angle": float(self.sun_altitude_angle),
            "wetness": float(self.wetness),
            "fog_density": float(self.fog_density),
            "fog_distance": float(self.fog_distance),
        }


@dataclass(frozen=True)
class ScriptKnot:
    t_s: float
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    # walker
    direction_x: float = 1.0
    direction_y: float = 0.0
    direction_z: float = 0.0
    speed: float = 0.0
    jump: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_s": float(self.t_s),
            "throttle": float(self.throttle),
            "brake": float(self.brake),
            "steer": float(self.steer),
            "hand_brake": bool(self.hand_brake),
            "reverse": bool(self.reverse),
            "direction_x": float(self.direction_x),
            "direction_y": float(self.direction_y),
            "direction_z": float(self.direction_z),
            "speed": float(self.speed),
            "jump": bool(self.jump),
        }


@dataclass(frozen=True)
class ActorScriptSpec:
    script_type: str
    knots: tuple[ScriptKnot, ...] = ()
    # constant modes
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "script_type": self.script_type,
            "throttle": float(self.throttle),
            "brake": float(self.brake),
            "steer": float(self.steer),
            "knots": [k.to_dict() for k in self.knots],
        }

    def control_at(self, simulation_time_since_anchor_s: float) -> dict[str, Any]:
        """Pure simulation-time script; never reads candidate/oracle/ego future."""
        t = float(simulation_time_since_anchor_s)
        st = self.script_type
        if st == "hold":
            return {
                "kind": "vehicle",
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0,
                "hand_brake": False,
                "reverse": False,
                "phase": f"hold@t={t:.3f}",
            }
        if st == "constant_throttle":
            return {
                "kind": "vehicle",
                "throttle": float(self.throttle),
                "brake": 0.0,
                "steer": float(self.steer),
                "hand_brake": False,
                "reverse": False,
                "phase": f"constant_throttle@t={t:.3f}",
            }
        if st == "constant_brake":
            return {
                "kind": "vehicle",
                "throttle": 0.0,
                "brake": float(self.brake),
                "steer": float(self.steer),
                "hand_brake": False,
                "reverse": False,
                "phase": f"constant_brake@t={t:.3f}",
            }
        if st == "piecewise_vehicle_control":
            if not self.knots:
                raise RegistryError("piecewise_vehicle_control requires knots")
            knot = self.knots[0]
            for k in self.knots:
                if t + 1e-9 >= k.t_s:
                    knot = k
                else:
                    break
            return {
                "kind": "vehicle",
                "throttle": float(knot.throttle),
                "brake": float(knot.brake),
                "steer": float(knot.steer),
                "hand_brake": bool(knot.hand_brake),
                "reverse": bool(knot.reverse),
                "phase": f"piecewise@t={t:.3f}->knot_t={knot.t_s:.3f}",
            }
        if st == "walker_control":
            if not self.knots:
                raise RegistryError("walker_control requires knots")
            knot = self.knots[0]
            for k in self.knots:
                if t + 1e-9 >= k.t_s:
                    knot = k
                else:
                    break
            return {
                "kind": "walker",
                "direction_x": float(knot.direction_x),
                "direction_y": float(knot.direction_y),
                "direction_z": float(knot.direction_z),
                "speed": float(knot.speed),
                "jump": bool(knot.jump),
                "phase": f"walker@t={t:.3f}->knot_t={knot.t_s:.3f}",
            }
        raise RegistryError(f"unsupported script_type: {st}")


@dataclass(frozen=True)
class RegistryActor:
    name: str
    role: str
    blueprint: str
    transform: TransformPose
    initial_velocity: VelocityState
    bounding_box_extent_m: tuple[float, float, float]
    script: ActorScriptSpec
    spawn_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "blueprint": self.blueprint,
            "transform": self.transform.raw_dict(),
            "initial_velocity": self.initial_velocity.raw_dict(),
            "bounding_box_extent_m": list(self.bounding_box_extent_m),
            "script": self.script.to_dict(),
            "spawn_order": int(self.spawn_order),
        }


@dataclass(frozen=True)
class RouteSpec:
    identity: str
    waypoints: tuple[tuple[float, float, float], ...]  # x,y,z
    target_speed_mps: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "waypoints": [list(w) for w in self.waypoints],
            "target_speed_mps": float(self.target_speed_mps),
        }


@dataclass(frozen=True)
class SensorContract:
    front_rgb: Mapping[str, Any]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"front_rgb": dict(self.front_rgb), "extra": dict(self.extra)}


@dataclass(frozen=True)
class ScenarioSeedFixture:
    """Fully expanded scenario × seed configuration (no free spawn)."""

    schema_version: str
    registry_version: str
    scenario_id: str
    family: str
    seed_id: str
    map_name: str
    weather: WeatherSpec
    sim_dt_s: float
    duration_s: float
    ego: RegistryActor
    route: RouteSpec
    actors: tuple[RegistryActor, ...]
    traffic_light: Mapping[str, Any]
    sensor_contract: SensorContract
    vla_config_ref: str
    mpc_config_ref: str
    executor_config_ref: str
    expected_decision_anchor_time_s: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "scenario_id": self.scenario_id,
            "family": self.family,
            "seed_id": self.seed_id,
            "map_name": self.map_name,
            "weather": self.weather.to_dict(),
            "sim_dt_s": float(self.sim_dt_s),
            "duration_s": float(self.duration_s),
            "ego": self.ego.to_dict(),
            "route": self.route.to_dict(),
            "actors": [a.to_dict() for a in self.actors],
            "traffic_light": dict(self.traffic_light),
            "sensor_contract": self.sensor_contract.to_dict(),
            "vla_config_ref": self.vla_config_ref,
            "mpc_config_ref": self.mpc_config_ref,
            "executor_config_ref": self.executor_config_ref,
            "expected_decision_anchor_time_s": float(self.expected_decision_anchor_time_s),
            "notes": self.notes,
        }

    def requested_initial_state_hash(self) -> str:
        return content_hash(self.to_dict(), nibble=64)


@dataclass(frozen=True)
class ScenarioRegistryV1:
    schema_version: str
    registry_version: str
    fixtures: tuple[ScenarioSeedFixture, ...]
    source_path: str | None = None
    frozen: bool = False
    registry_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_SCHEMA:
            raise RegistryError(
                f"registry schema must be {REGISTRY_SCHEMA}, got {self.schema_version}"
            )

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((f.scenario_id, f.seed_id) for f in self.fixtures)

    def get(self, scenario_id: str, seed_id: str) -> ScenarioSeedFixture:
        for f in self.fixtures:
            if f.scenario_id == scenario_id and f.seed_id == seed_id:
                return f
        raise RegistryError(f"fixture not found: {scenario_id}/{seed_id}")

    def families(self) -> set[str]:
        return {f.family for f in self.fixtures}

    def scenario_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for f in self.fixtures:
            if f.scenario_id not in seen:
                seen.append(f.scenario_id)
        return tuple(seen)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "fixtures": [f.to_dict() for f in self.fixtures],
        }

    def compute_registry_sha256(self) -> str:
        return sha256_hex(canonical_json_bytes(self.canonical_payload()))

    def freeze_manifest(self) -> dict[str, Any]:
        reg_hash = self.compute_registry_sha256()
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "registry_sha256": reg_hash,
            "n_scenarios": len(self.scenario_ids()),
            "n_seeds_per_scenario": len(REQUIRED_SEEDS),
            "n_pairs": len(self.fixtures),
            "families": sorted(self.families()),
            "scenario_ids": list(self.scenario_ids()),
            "pairs": [{"scenario_id": s, "seed_id": d} for s, d in self.pairs()],
            "requested_initial_state_hashes": {
                f"{f.scenario_id}/{f.seed_id}": f.requested_initial_state_hash()
                for f in self.fixtures
            },
            "source_path": self.source_path,
            "frozen": True,
        }


def _pose_from_mapping(m: Mapping[str, Any], name: str) -> TransformPose:
    required = ("x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg")
    for k in required:
        if k not in m:
            raise RegistryError(f"{name} missing transform field {k}")
    # Reject free/fallback spawn markers
    for bad in ("free_spawn", "fallback", "any_spawn", "random_spawn"):
        if bad in m and m[bad]:
            raise RegistryError(f"{name}: forbidden spawn mode marker {bad}")
    return TransformPose(
        x=float(m["x"]),
        y=float(m["y"]),
        z=float(m["z"]),
        roll_deg=float(m["roll_deg"]),
        pitch_deg=float(m["pitch_deg"]),
        yaw_deg=float(m["yaw_deg"]),
    )


def _velocity_from_mapping(m: Mapping[str, Any] | None) -> VelocityState:
    if not m:
        return VelocityState(0.0, 0.0, 0.0)
    return VelocityState(
        vx=float(m.get("vx", 0.0)),
        vy=float(m.get("vy", 0.0)),
        vz=float(m.get("vz", 0.0)),
        wx=float(m.get("wx", 0.0)),
        wy=float(m.get("wy", 0.0)),
        wz=float(m.get("wz", 0.0)),
    )


def _script_from_mapping(m: Mapping[str, Any], name: str) -> ActorScriptSpec:
    if not isinstance(m, Mapping):
        raise RegistryError(f"{name}: script must be a table")
    for bad in FORBIDDEN_SCRIPT_KEYS:
        if bad in m:
            raise RegistryError(f"{name}: forbidden script key {bad}")
    st = str(m.get("script_type", "")).strip()
    if not st:
        raise RegistryError(f"{name}: script_type required")
    if st not in ALLOWED_VEHICLE_SCRIPTS | ALLOWED_WALKER_SCRIPTS:
        raise RegistryError(f"{name}: unsupported script_type {st}")
    knots_raw = m.get("knots", []) or []
    knots: list[ScriptKnot] = []
    prev_t = -1e18
    for i, k in enumerate(knots_raw):
        if not isinstance(k, Mapping):
            raise RegistryError(f"{name}: knot {i} must be a table")
        for bad in FORBIDDEN_SCRIPT_KEYS:
            if bad in k:
                raise RegistryError(f"{name}: forbidden knot key {bad}")
        t_s = float(k.get("t_s", 0.0))
        if t_s + 1e-12 < prev_t:
            raise RegistryError(f"{name}: knots must be non-decreasing in t_s")
        prev_t = t_s
        knots.append(
            ScriptKnot(
                t_s=t_s,
                throttle=float(k.get("throttle", 0.0)),
                brake=float(k.get("brake", 0.0)),
                steer=float(k.get("steer", 0.0)),
                hand_brake=bool(k.get("hand_brake", False)),
                reverse=bool(k.get("reverse", False)),
                direction_x=float(k.get("direction_x", 1.0)),
                direction_y=float(k.get("direction_y", 0.0)),
                direction_z=float(k.get("direction_z", 0.0)),
                speed=float(k.get("speed", 0.0)),
                jump=bool(k.get("jump", False)),
            )
        )
    if st == "piecewise_vehicle_control" and not knots:
        raise RegistryError(f"{name}: piecewise_vehicle_control requires knots")
    if st == "walker_control" and not knots:
        raise RegistryError(f"{name}: walker_control requires knots")
    return ActorScriptSpec(
        script_type=st,
        knots=tuple(knots),
        throttle=float(m.get("throttle", 0.0)),
        brake=float(m.get("brake", 0.0)),
        steer=float(m.get("steer", 0.0)),
    )


def _actor_from_mapping(m: Mapping[str, Any], *, default_role: str) -> RegistryActor:
    name = str(m.get("name", "")).strip()
    if not name:
        raise RegistryError("actor name required")
    role = str(m.get("role", default_role)).strip()
    if role not in {"ego", "npc"}:
        raise RegistryError(f"actor {name}: role must be ego or npc")
    bp = str(m.get("blueprint", "")).strip()
    if not bp:
        raise RegistryError(f"actor {name}: blueprint required")
    if bool(m.get("free_spawn", False)) or bool(m.get("fallback_spawn", False)):
        raise RegistryError(f"actor {name}: free/fallback spawn forbidden")
    if bool(m.get("autopilot", False)) or bool(m.get("traffic_manager", False)):
        raise RegistryError(f"actor {name}: Traffic Manager / autopilot forbidden for core actors")
    tf = m.get("transform")
    if not isinstance(tf, Mapping):
        raise RegistryError(f"actor {name}: exact transform required")
    extent = m.get("bounding_box_extent_m", [2.0, 1.0, 0.8])
    if len(extent) != 3:
        raise RegistryError(f"actor {name}: bounding_box_extent_m must be length 3")
    script = _script_from_mapping(m.get("script") or {"script_type": "hold"}, name)
    return RegistryActor(
        name=name,
        role=role,
        blueprint=bp,
        transform=_pose_from_mapping(tf, name),
        initial_velocity=_velocity_from_mapping(m.get("initial_velocity")),
        bounding_box_extent_m=(float(extent[0]), float(extent[1]), float(extent[2])),
        script=script,
        spawn_order=int(m.get("spawn_order", 0)),
    )


def _weather_from_mapping(m: Mapping[str, Any]) -> WeatherSpec:
    required = (
        "preset",
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_azimuth_angle",
        "sun_altitude_angle",
        "wetness",
    )
    for k in required:
        if k not in m:
            raise RegistryError(f"weather missing field {k}")
    return WeatherSpec(
        preset=str(m["preset"]),
        cloudiness=float(m["cloudiness"]),
        precipitation=float(m["precipitation"]),
        precipitation_deposits=float(m["precipitation_deposits"]),
        wind_intensity=float(m["wind_intensity"]),
        sun_azimuth_angle=float(m["sun_azimuth_angle"]),
        sun_altitude_angle=float(m["sun_altitude_angle"]),
        wetness=float(m["wetness"]),
        fog_density=float(m.get("fog_density", 0.0)),
        fog_distance=float(m.get("fog_distance", 0.0)),
    )


def _route_from_mapping(m: Mapping[str, Any]) -> RouteSpec:
    identity = str(m.get("identity", "")).strip()
    if not identity:
        raise RegistryError("route.identity required")
    wps_raw = m.get("waypoints") or []
    if len(wps_raw) < 2:
        raise RegistryError("route must have at least 2 waypoints")
    wps: list[tuple[float, float, float]] = []
    for i, w in enumerate(wps_raw):
        if not isinstance(w, (list, tuple)) or len(w) < 2:
            raise RegistryError(f"route waypoint {i} invalid")
        z = float(w[2]) if len(w) > 2 else 0.0
        wps.append((float(w[0]), float(w[1]), z))
    return RouteSpec(
        identity=identity,
        waypoints=tuple(wps),
        target_speed_mps=float(m.get("target_speed_mps", 8.0)),
    )


def _validate_registry_shape(
    fixtures: Sequence[ScenarioSeedFixture],
    *,
    registry_version: str,
) -> None:
    if len(fixtures) != 12:
        raise RegistryError(f"R2 freezes 12 pairs, got {len(fixtures)}")
    scenario_ids = {f.scenario_id for f in fixtures}
    if registry_version == REGISTRY_VERSION_DEFAULT and scenario_ids != set(REQUIRED_SCENARIO_IDS):
        raise RegistryError(
            f"scenario_ids must be exactly {REQUIRED_SCENARIO_IDS}, got {sorted(scenario_ids)}"
        )
    if registry_version != REGISTRY_VERSION_DEFAULT and len(scenario_ids) != 6:
        raise RegistryError(
            f"post-v1 registry must contain exactly 6 scenarios, got {len(scenario_ids)}"
        )
    families = {f.family for f in fixtures}
    if families != ALLOWED_FAMILIES:
        raise RegistryError(f"families must be exactly {sorted(ALLOWED_FAMILIES)}, got {sorted(families)}")
    by_scenario: dict[str, set[str]] = {}
    seen_pair: set[tuple[str, str]] = set()
    for f in fixtures:
        key = (f.scenario_id, f.seed_id)
        if key in seen_pair:
            raise RegistryError(f"duplicate scenario/seed pair: {key}")
        seen_pair.add(key)
        by_scenario.setdefault(f.scenario_id, set()).add(f.seed_id)
        if f.family not in ALLOWED_FAMILIES:
            raise RegistryError(f"{f.scenario_id}: invalid family {f.family}")
        if f.seed_id not in REQUIRED_SEEDS:
            raise RegistryError(f"{f.scenario_id}: seed_id must be seed_a or seed_b")
        if abs(f.sim_dt_s - 0.05) > 1e-9:
            raise RegistryError(f"{f.scenario_id}/{f.seed_id}: sim_dt_s must be 0.05")
        if f.duration_s + 1e-9 < 2.5:
            raise RegistryError(f"{f.scenario_id}/{f.seed_id}: duration_s must cover primary 2.5s")
        if f.map_name not in {"Town03", "Town03_Opt"}:
            # Prefer Town03; other towns only if explicitly allowed later.
            raise RegistryError(f"{f.scenario_id}: first registry version freezes Town03*")
        if f.ego.role != "ego":
            raise RegistryError(f"{f.scenario_id}: ego role mismatch")
        # Overlap check (axis-aligned XY distance between centers)
        centers = [(f.ego.name, f.ego.transform.x, f.ego.transform.y, f.ego.bounding_box_extent_m)]
        for a in f.actors:
            centers.append((a.name, a.transform.x, a.transform.y, a.bounding_box_extent_m))
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                ni, xi, yi, ei = centers[i]
                nj, xj, yj, ej = centers[j]
                min_sep = float(ei[0] + ej[0]) * 0.5 + 0.5
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                if dist < min_sep:
                    raise RegistryError(
                        f"{f.scenario_id}/{f.seed_id}: actors {ni} and {nj} overlap "
                        f"(dist={dist:.3f} < {min_sep:.3f})"
                    )
        # Route length rough check
        total = 0.0
        wps = f.route.waypoints
        for i in range(1, len(wps)):
            total += ((wps[i][0] - wps[i - 1][0]) ** 2 + (wps[i][1] - wps[i - 1][1]) ** 2) ** 0.5
        if total < 15.0:
            raise RegistryError(
                f"{f.scenario_id}/{f.seed_id}: route too short ({total:.2f}m < 15m)"
            )
    for sid in sorted(scenario_ids):
        seeds = by_scenario.get(sid, set())
        if seeds != set(REQUIRED_SEEDS):
            raise RegistryError(f"{sid}: must have exactly seeds {REQUIRED_SEEDS}, got {sorted(seeds)}")


def load_scenario_registry(path: Path | str) -> ScenarioRegistryV1:
    """Load and fully validate registry TOML."""
    p = Path(path)
    if not p.is_file():
        raise RegistryError(f"registry file not found: {p}")
    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    meta = raw.get("registry") or raw.get("meta") or {}
    schema_version = str(meta.get("schema_version", REGISTRY_SCHEMA))
    registry_version = str(meta.get("registry_version", REGISTRY_VERSION_DEFAULT))
    if schema_version != REGISTRY_SCHEMA:
        raise RegistryError(f"unsupported schema_version {schema_version}")

    defaults = raw.get("defaults") or {}
    scenarios = raw.get("scenarios") or {}
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise RegistryError("scenarios table required")

    fixtures: list[ScenarioSeedFixture] = []
    for scenario_id, sc in scenarios.items():
        if not isinstance(sc, Mapping):
            raise RegistryError(f"scenario {scenario_id} must be a table")
        family = str(sc.get("family", "")).strip()
        if family not in ALLOWED_FAMILIES:
            raise RegistryError(f"{scenario_id}: invalid family {family}")
        map_name = str(sc.get("map_name", defaults.get("map_name", "Town03")))
        sim_dt_s = float(sc.get("sim_dt_s", defaults.get("sim_dt_s", 0.05)))
        duration_s = float(sc.get("duration_s", defaults.get("duration_s", 5.0)))
        weather = _weather_from_mapping(sc.get("weather") or defaults.get("weather") or {})
        route = _route_from_mapping(sc.get("route") or {})
        sensor = SensorContract(
            front_rgb=dict((sc.get("sensor_contract") or defaults.get("sensor_contract") or {}).get(
                "front_rgb",
                {
                    "width": 1024,
                    "height": 512,
                    "fov": 110.0,
                    "attach": "ego",
                },
            )),
            extra=dict((sc.get("sensor_contract") or {}).get("extra", {})),
        )
        traffic_light = dict(sc.get("traffic_light") or defaults.get("traffic_light") or {"policy": "freeze_green"})
        vla_ref = str(sc.get("vla_config_ref", defaults.get("vla_config_ref", "config/vla/k2_v1.toml")))
        mpc_ref = str(sc.get("mpc_config_ref", defaults.get("mpc_config_ref", "config/control/mpc_pid_baseline.toml")))
        exec_ref = str(
            sc.get("executor_config_ref", defaults.get("executor_config_ref", "g3_stable_vla_mpc"))
        )
        expected_anchor = float(
            sc.get("expected_decision_anchor_time_s", defaults.get("expected_decision_anchor_time_s", 0.0))
        )
        notes = str(sc.get("notes", ""))

        seeds = sc.get("seeds") or {}
        if not isinstance(seeds, Mapping) or not seeds:
            raise RegistryError(f"{scenario_id}: seeds table required")
        for seed_id, seed_body in seeds.items():
            if not isinstance(seed_body, Mapping):
                raise RegistryError(f"{scenario_id}/{seed_id}: seed body must be a table")
            ego = _actor_from_mapping(seed_body.get("ego") or sc.get("ego") or {}, default_role="ego")
            actors_raw = seed_body.get("actors") or sc.get("actors") or []
            if not isinstance(actors_raw, list):
                raise RegistryError(f"{scenario_id}/{seed_id}: actors must be a list")
            actors = tuple(_actor_from_mapping(a, default_role="npc") for a in actors_raw)
            # Per-seed weather/route overrides optional
            w = _weather_from_mapping(seed_body["weather"]) if "weather" in seed_body else weather
            r = _route_from_mapping(seed_body["route"]) if "route" in seed_body else route
            fixtures.append(
                ScenarioSeedFixture(
                    schema_version=schema_version,
                    registry_version=registry_version,
                    scenario_id=str(scenario_id),
                    family=family,
                    seed_id=str(seed_id),
                    map_name=map_name,
                    weather=w,
                    sim_dt_s=sim_dt_s,
                    duration_s=duration_s,
                    ego=ego,
                    route=r,
                    actors=actors,
                    traffic_light=dict(seed_body.get("traffic_light") or traffic_light),
                    sensor_contract=sensor,
                    vla_config_ref=vla_ref,
                    mpc_config_ref=mpc_ref,
                    executor_config_ref=exec_ref,
                    expected_decision_anchor_time_s=float(
                        seed_body.get("expected_decision_anchor_time_s", expected_anchor)
                    ),
                    notes=notes,
                )
            )

    # Stable order: required scenario list × seed_a, seed_b
    scenario_order = (
        list(REQUIRED_SCENARIO_IDS)
        if registry_version == REGISTRY_VERSION_DEFAULT
        else sorted(str(sid) for sid in scenarios)
    )
    order_index = {
        (sid, seed): i
        for i, (sid, seed) in enumerate(
            (sid, seed) for sid in scenario_order for seed in REQUIRED_SEEDS
        )
    }
    fixtures_sorted = sorted(
        fixtures,
        key=lambda f: order_index.get((f.scenario_id, f.seed_id), 10_000),
    )
    _validate_registry_shape(fixtures_sorted, registry_version=registry_version)
    reg = ScenarioRegistryV1(
        schema_version=schema_version,
        registry_version=registry_version,
        fixtures=tuple(fixtures_sorted),
        source_path=str(p.as_posix()),
        frozen=False,
        registry_sha256=None,
    )
    # Attach computed hash without breaking frozen dataclass
    object.__setattr__(reg, "registry_sha256", reg.compute_registry_sha256())
    return reg


def freeze_registry(registry: ScenarioRegistryV1) -> ScenarioRegistryV1:
    """Mark registry frozen; returns a copy with frozen=True and sha256 set."""
    sha = registry.compute_registry_sha256()
    return ScenarioRegistryV1(
        schema_version=registry.schema_version,
        registry_version=registry.registry_version,
        fixtures=registry.fixtures,
        source_path=registry.source_path,
        frozen=True,
        registry_sha256=sha,
    )


DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "g4a" / "scenario_registry_v1.toml"
)
