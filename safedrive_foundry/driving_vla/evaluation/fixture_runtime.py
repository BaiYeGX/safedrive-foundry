"""CARLA fixture helpers for R2 registry dry-run and paired cold rebuilds.

Exact spawn only — no free_spawn fallback. Actor scripts are simulation-time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from driving_vla.evaluation.paired_contract import (
    MEASURED_STATE_SCHEMA,
    ActorSnapshot,
    MeasuredInitialState,
    TransformPose,
    VelocityState,
)
from driving_vla.evaluation.scenario_registry import (
    RegistryActor,
    ScenarioSeedFixture,
)


class FixtureError(RuntimeError):
    """Spawn / script / cleanup failure (fail closed)."""


@dataclass
class SpawnedActor:
    name: str
    role: str
    blueprint: str
    actor: Any  # carla.Actor
    requested: RegistryActor


@dataclass
class FixtureSession:
    client: Any
    world: Any
    fixture: ScenarioSeedFixture
    spawned: list[SpawnedActor] = field(default_factory=list)
    sim_time0: float = 0.0
    frame0: int = 0
    cleaned: bool = False


def _carla_transform(pose: TransformPose) -> Any:
    import carla

    return carla.Transform(
        carla.Location(x=float(pose.x), y=float(pose.y), z=float(pose.z)),
        carla.Rotation(
            roll=float(pose.roll_deg),
            pitch=float(pose.pitch_deg),
            yaw=float(pose.yaw_deg),
        ),
    )


def _pose_from_carla(tf: Any) -> TransformPose:
    loc = tf.location
    rot = tf.rotation
    return TransformPose(
        x=float(loc.x),
        y=float(loc.y),
        z=float(loc.z),
        roll_deg=float(rot.roll),
        pitch_deg=float(rot.pitch),
        yaw_deg=float(rot.yaw),
    )


def _velocity_from_carla(actor: Any) -> VelocityState:
    v = actor.get_velocity()
    w = actor.get_angular_velocity()
    return VelocityState(
        vx=float(v.x),
        vy=float(v.y),
        vz=float(v.z),
        wx=float(w.x),
        wy=float(w.y),
        wz=float(w.z),
    )


def connect_world(
    *,
    host: str = "127.0.0.1",
    port: int = 2000,
    timeout_s: float = 60.0,
    map_name: str = "Town03",
    sim_dt_s: float = 0.05,
    sync: bool = True,
    retries: int = 3,
) -> tuple[Any, Any]:
    """Single client connection; caller is sole tick owner when sync=True.

    Default timeout 60s (was 10s): WSL↔Windows RPC + GPU load can stall
    get_world/tick beyond 10s without CARLA being "off". Retries only on
    transient RPC timeouts; map mismatch still fail-closed immediately.
    """
    import carla

    last_exc: BaseException | None = None
    for attempt in range(max(1, int(retries))):
        try:
            client = carla.Client(host, int(port))
            client.set_timeout(float(timeout_s))
            world = client.get_world()
            current = world.get_map().name
            # Prefer already-loaded Town03; avoid mid-session load_world when possible.
            if map_name not in current and not current.endswith(map_name):
                raise FixtureError(
                    f"map mismatch: server={current!r} required={map_name!r}; "
                    "restart CARLA with Town03 rather than mid-session load_world"
                )
            settings = world.get_settings()
            settings.synchronous_mode = bool(sync)
            settings.fixed_delta_seconds = float(sim_dt_s)
            world.apply_settings(settings)
            if sync:
                world.tick()
            return client, world
        except FixtureError:
            raise
        except Exception as exc:  # noqa: BLE001 — RPC retry
            last_exc = exc
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(2.0 * (attempt + 1))
    raise FixtureError(
        f"connect_world failed after {retries} tries host={host}:{port}: {last_exc}"
    ) from last_exc


def restore_async(world: Any) -> None:
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except Exception:
        pass


def _find_blueprint(world: Any, bp_id: str) -> Any:
    bp_lib = world.get_blueprint_library()
    try:
        return bp_lib.find(bp_id)
    except Exception:
        matches = list(bp_lib.filter(bp_id))
        if matches:
            return matches[0]
        # common alias fallbacks
        aliases = {
            "vehicle.bmw.grandtour": "vehicle.lincoln.mkz_2017",
            "vehicle.tesla.model3": "vehicle.tesla.model3",
        }
        alt = aliases.get(bp_id)
        if alt and alt != bp_id:
            try:
                return bp_lib.find(alt)
            except Exception:
                pass
        raise FixtureError(f"blueprint not found: {bp_id}")


def exact_spawn_actor(world: Any, spec: RegistryActor) -> Any:
    """Spawn at exact transform. Never falls back to another spawn point."""
    import carla

    bp = _find_blueprint(world, spec.blueprint)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero" if spec.role == "ego" else spec.name)
    tf = _carla_transform(spec.transform)
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        raise FixtureError(
            f"SPAWN_FAILED exact transform for {spec.name} "
            f"bp={spec.blueprint} x={spec.transform.x:.3f} y={spec.transform.y:.3f} "
            f"z={spec.transform.z:.3f} yaw={spec.transform.yaw_deg:.2f}"
        )
    # Set initial velocity if vehicle
    if hasattr(actor, "set_target_velocity"):
        iv = spec.initial_velocity
        actor.set_target_velocity(carla.Vector3D(iv.vx, iv.vy, iv.vz))
        actor.set_target_angular_velocity(carla.Vector3D(iv.wx, iv.wy, iv.wz))
    return actor


def _reapply_exact_state(session: FixtureSession) -> None:
    """Pin transform+velocity after settle so cold rebuilds share initial state."""
    import carla

    for sp in session.spawned:
        spec = sp.requested
        tf = _carla_transform(spec.transform)
        try:
            sp.actor.set_simulate_physics(False)
        except Exception:
            pass
        sp.actor.set_transform(tf)
        if hasattr(sp.actor, "set_target_velocity"):
            iv = spec.initial_velocity
            sp.actor.set_target_velocity(carla.Vector3D(iv.vx, iv.vy, iv.vz))
            sp.actor.set_target_angular_velocity(carla.Vector3D(iv.wx, iv.wy, iv.wz))
        try:
            sp.actor.set_simulate_physics(True)
        except Exception:
            pass
        # re-apply once physics is on
        if hasattr(sp.actor, "set_target_velocity"):
            iv = spec.initial_velocity
            sp.actor.set_target_velocity(carla.Vector3D(iv.vx, iv.vy, iv.vz))
            sp.actor.set_target_angular_velocity(carla.Vector3D(iv.wx, iv.wy, iv.wz))


def open_fixture_session(
    client: Any,
    world: Any,
    fixture: ScenarioSeedFixture,
    *,
    settle_ticks: int = 5,
) -> FixtureSession:
    """Spawn ego + NPCs for one fixture; settle a few ticks; no free spawn."""
    # Clear leftovers from prior failed cleanups before exact spawn.
    purge_episode_actors(world, client=client)
    ordered: list[RegistryActor] = [fixture.ego, *sorted(fixture.actors, key=lambda a: a.spawn_order)]
    session = FixtureSession(client=client, world=world, fixture=fixture)
    try:
        for spec in ordered:
            actor = exact_spawn_actor(world, spec)
            session.spawned.append(
                SpawnedActor(
                    name=spec.name,
                    role=spec.role,
                    blueprint=spec.blueprint,
                    actor=actor,
                    requested=spec,
                )
            )
            world.tick()
        for _ in range(max(0, int(settle_ticks))):
            _apply_scripts(session, simulation_time_since_anchor_s=0.0)
            world.tick()
        # Critical for paired comparability: re-pin exact registry state after
        # physics settle so branch-0/1 cold rebuilds share measured initial pose.
        _reapply_exact_state(session)
        world.tick()
        _reapply_exact_state(session)
        world.tick()
        snap = world.get_snapshot()
        session.frame0 = int(snap.frame)
        session.sim_time0 = float(snap.timestamp.elapsed_seconds)
    except Exception:
        cleanup_session(session, soft=True)
        raise
    return session


def _apply_scripts(session: FixtureSession, *, simulation_time_since_anchor_s: float) -> None:
    import carla

    for sp in session.spawned:
        ctrl = sp.requested.script.control_at(simulation_time_since_anchor_s)
        if ctrl.get("kind") == "vehicle" and hasattr(sp.actor, "apply_control"):
            sp.actor.apply_control(
                carla.VehicleControl(
                    throttle=float(ctrl["throttle"]),
                    brake=float(ctrl["brake"]),
                    steer=float(ctrl["steer"]),
                    hand_brake=bool(ctrl.get("hand_brake", False)),
                    reverse=bool(ctrl.get("reverse", False)),
                )
            )
        elif ctrl.get("kind") == "walker" and hasattr(sp.actor, "apply_control"):
            direction = carla.Vector3D(
                float(ctrl["direction_x"]),
                float(ctrl["direction_y"]),
                float(ctrl["direction_z"]),
            )
            sp.actor.apply_control(
                carla.WalkerControl(
                    direction=direction,
                    speed=float(ctrl["speed"]),
                    jump=bool(ctrl.get("jump", False)),
                )
            )


def step_fixture(
    session: FixtureSession,
    *,
    n_ticks: int,
    sim_dt_s: float,
) -> list[dict[str, Any]]:
    """Advance scripts by simulation time; return lightweight per-tick states."""
    records: list[dict[str, Any]] = []
    for i in range(int(n_ticks)):
        t = (i + 1) * float(sim_dt_s)
        _apply_scripts(session, simulation_time_since_anchor_s=t)
        session.world.tick()
        ego = next((s for s in session.spawned if s.role == "ego"), None)
        rec: dict[str, Any] = {"tick": i, "t_s": t}
        if ego is not None:
            tf = ego.actor.get_transform()
            rec["ego_x"] = float(tf.location.x)
            rec["ego_y"] = float(tf.location.y)
            rec["ego_v"] = math.hypot(
                float(ego.actor.get_velocity().x), float(ego.actor.get_velocity().y)
            )
        for sp in session.spawned:
            if sp.role == "ego":
                continue
            phase = sp.requested.script.control_at(t)["phase"]
            rec[f"{sp.name}_phase"] = phase
            tf = sp.actor.get_transform()
            rec[f"{sp.name}_x"] = float(tf.location.x)
            rec[f"{sp.name}_y"] = float(tf.location.y)
        records.append(rec)
    return records


def measure_initial_state(session: FixtureSession) -> MeasuredInitialState:
    world = session.world
    snap = world.get_snapshot()
    weather = world.get_weather()
    actors: list[ActorSnapshot] = []
    phases: dict[str, str] = {}
    for sp in session.spawned:
        ctrl = {}
        if hasattr(sp.actor, "get_control"):
            c = sp.actor.get_control()
            ctrl = {
                "throttle": float(getattr(c, "throttle", 0.0)),
                "brake": float(getattr(c, "brake", 0.0)),
                "steer": float(getattr(c, "steer", 0.0)),
            }
        phase = sp.requested.script.control_at(0.0)["phase"]
        phases[sp.name] = phase
        extent = sp.requested.bounding_box_extent_m
        if hasattr(sp.actor, "bounding_box"):
            e = sp.actor.bounding_box.extent
            extent = (float(e.x), float(e.y), float(e.z))
        actors.append(
            ActorSnapshot(
                name=sp.name,
                role=sp.role,
                blueprint=sp.blueprint,
                transform=_pose_from_carla(sp.actor.get_transform()),
                velocity=_velocity_from_carla(sp.actor),
                control=ctrl,
                script_phase=phase,
                bounding_box_extent_m=extent,
            )
        )
    settings = world.get_settings()
    return MeasuredInitialState(
        schema_version=MEASURED_STATE_SCHEMA,
        map_name=session.fixture.map_name,
        open_drive_identity=str(world.get_map().name),
        world_settings={
            "synchronous_mode": bool(settings.synchronous_mode),
            "fixed_delta_seconds": float(settings.fixed_delta_seconds or 0.0),
        },
        weather={
            "cloudiness": float(weather.cloudiness),
            "precipitation": float(weather.precipitation),
            "precipitation_deposits": float(weather.precipitation_deposits),
            "wind_intensity": float(weather.wind_intensity),
            "sun_azimuth_angle": float(weather.sun_azimuth_angle),
            "sun_altitude_angle": float(weather.sun_altitude_angle),
            "wetness": float(weather.wetness),
            "fog_density": float(weather.fog_density),
            "fog_distance": float(weather.fog_distance),
        },
        actors=tuple(actors),
        traffic_light_state=dict(session.fixture.traffic_light),
        route_anchor={"identity": session.fixture.route.identity},
        sensor_calibration=session.fixture.sensor_contract.to_dict(),
        carla_server_epoch=str(getattr(session.client, "get_server_version", lambda: "unknown")()),
        carla_version=str(getattr(session.client, "get_client_version", lambda: "unknown")()),
        simulation_frame=int(snap.frame),
        simulation_time_s=float(snap.timestamp.elapsed_seconds),
        actor_script_phase=phases,
    )


def purge_episode_actors(world: Any, *, client: Any | None = None) -> list[str]:
    """Best-effort destroy leftover vehicles/walkers/sensors before a new episode.

    Prevents actor leaks after a prior CLEANUP_FAILURE from wedging the next pair.
    """
    notes: list[str] = []
    if client is not None:
        try:
            client.set_timeout(5.0)
        except Exception:
            pass
    try:
        actors = list(world.get_actors())
    except Exception as exc:  # noqa: BLE001
        return [f"list_actors:{exc}"]
    for a in actors:
        try:
            t = a.type_id if hasattr(a, "type_id") else ""
        except Exception:
            t = ""
        if not (
            str(t).startswith("vehicle.")
            or str(t).startswith("walker.")
            or str(t).startswith("sensor.")
        ):
            continue
        try:
            a.destroy()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"purge:{t}:{exc}")
    try:
        world.tick()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"purge_tick:{exc}")
    return notes


def cleanup_session(session: FixtureSession, *, soft: bool = False) -> None:
    """Destroy spawned actors. soft=True never raises (dead-server tolerant)."""
    if session.cleaned:
        return
    errors: list[str] = []
    # Short RPC timeout so a dead CARLA does not burn 60s × N actors.
    try:
        session.client.set_timeout(3.0)
    except Exception:
        pass
    for sp in list(reversed(session.spawned)):
        try:
            if sp.actor is not None:
                sp.actor.destroy()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sp.name}:{exc}")
    session.spawned.clear()
    try:
        for _ in range(2):
            session.world.tick()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tick:{exc}")
    try:
        session.client.set_timeout(60.0)
    except Exception:
        pass
    session.cleaned = True
    if errors and not soft:
        raise FixtureError("CLEANUP_FAILURE: " + "; ".join(errors))


def apply_weather(world: Any, weather_spec: Any) -> None:
    import carla

    w = weather_spec.to_dict() if hasattr(weather_spec, "to_dict") else dict(weather_spec)
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=float(w.get("cloudiness", 0.0)),
            precipitation=float(w.get("precipitation", 0.0)),
            precipitation_deposits=float(w.get("precipitation_deposits", 0.0)),
            wind_intensity=float(w.get("wind_intensity", 0.0)),
            sun_azimuth_angle=float(w.get("sun_azimuth_angle", 0.0)),
            sun_altitude_angle=float(w.get("sun_altitude_angle", 70.0)),
            wetness=float(w.get("wetness", 0.0)),
            fog_density=float(w.get("fog_density", 0.0)),
            fog_distance=float(w.get("fog_distance", 0.0)),
        )
    )


def nearest_driving_waypoint(
    world: Any,
    x: float,
    y: float,
    z: float = 0.5,
    *,
    lane_type: Any = None,
) -> dict[str, float]:
    """Utility for registry authoring only (not a runtime free-spawn fallback)."""
    import carla

    cmap = world.get_map()
    loc = carla.Location(x=float(x), y=float(y), z=float(z))
    wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        raise FixtureError(f"no driving waypoint near ({x}, {y})")
    tf = wp.transform
    return {
        "x": float(tf.location.x),
        "y": float(tf.location.y),
        "z": float(tf.location.z + 0.5),
        "yaw_deg": float(tf.rotation.yaw),
        "road_id": int(wp.road_id),
        "lane_id": int(wp.lane_id),
        "s": float(wp.s),
    }


def compare_measured(
    a: MeasuredInitialState,
    b: MeasuredInitialState,
    *,
    pos_tol_m: float = 0.02,
    yaw_tol_deg: float = 0.2,
    vel_tol_mps: float = 0.05,
) -> list[str]:
    """Return mismatch reasons (empty if within dry-run tolerances)."""
    reasons: list[str] = []
    actors_a = {x.name: x for x in a.actors}
    actors_b = {x.name: x for x in b.actors}
    if set(actors_a) != set(actors_b):
        reasons.append(f"actor_set {sorted(actors_a)} != {sorted(actors_b)}")
        return reasons
    for name in sorted(actors_a):
        pa, pb = actors_a[name], actors_b[name]
        dx = pa.transform.x - pb.transform.x
        dy = pa.transform.y - pb.transform.y
        dz = pa.transform.z - pb.transform.z
        pos = math.sqrt(dx * dx + dy * dy + dz * dz)
        if pos > pos_tol_m:
            reasons.append(f"{name}:pos={pos:.4f}")
        dyaw = abs((pa.transform.yaw_deg - pb.transform.yaw_deg + 180.0) % 360.0 - 180.0)
        if dyaw > yaw_tol_deg:
            reasons.append(f"{name}:yaw={dyaw:.4f}")
        dvx = pa.velocity.vx - pb.velocity.vx
        dvy = pa.velocity.vy - pb.velocity.vy
        dvz = pa.velocity.vz - pb.velocity.vz
        vel = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        if vel > vel_tol_mps:
            reasons.append(f"{name}:vel={vel:.4f}")
        if pa.script_phase != pb.script_phase:
            reasons.append(f"{name}:script_phase")
    return reasons
