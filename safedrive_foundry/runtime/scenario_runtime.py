"""Single-owner CARLA scenario runtime with deterministic lifecycle management.

The module intentionally imports neither CARLA nor ROS.  Production callers
pass CARLA client/world objects, while tests use protocol-compatible fakes.
All tick, control, sensor admission and cleanup ownership is kept here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    import fcntl  # type: ignore
except ImportError:  # Windows — file locks use msvcrt or no-op for preflight
    fcntl = None  # type: ignore

from .identity import RunIdentity
from .profiles import RuntimeProfile


class RuntimeViolation(RuntimeError):
    """Raised when lifecycle, ownership, or frame contracts are violated."""


class TickLeaseUnavailable(RuntimeViolation):
    """Raised when another runtime already owns the CARLA tick."""


class CleanupFailed(RuntimeViolation):
    """Raised when cleanup or post-cleanup verification is not successful."""


@dataclass(frozen=True)
class FrameHeader:
    """The one frame identity shared by camera, ego, actors, control and clock."""

    identity: RunIdentity
    carla_frame: int
    simulation_time: float
    wall_time: float
    coordinate_frame: str = "carla_map"


@dataclass(frozen=True)
class ActorSpec:
    name: str
    blueprint: str
    transform: Any
    role: str  # ``ego`` or ``npc``
    spawn_order: int
    autopilot: bool = False


@dataclass(frozen=True)
class SensorSpec:
    name: str
    blueprint: str
    transform: Any
    parent: str
    spawn_order: int
    attributes: Mapping[str, str] = field(default_factory=dict)
    delivery: str = "frame"  # ``frame`` participates in barrier; ``event`` is sparse.

    def __post_init__(self) -> None:
        if self.delivery not in {"frame", "event"}:
            raise ValueError("sensor delivery must be 'frame' or 'event'")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    map_name: str
    actors: tuple[ActorSpec, ...] = ()
    sensors: tuple[SensorSpec, ...] = ()
    traffic_manager_port: int = 8000
    traffic_manager_seed: int = 0
    sensor_timeout_seconds: float = 1.0
    weather: Mapping[str, float] = field(default_factory=dict)

    def ordered_actors(self) -> tuple[ActorSpec, ...]:
        return tuple(sorted(self.actors, key=lambda item: (item.spawn_order, item.name)))

    def ordered_sensors(self) -> tuple[SensorSpec, ...]:
        return tuple(sorted(self.sensors, key=lambda item: (item.spawn_order, item.name)))


class TickLease:
    """Cross-process exclusive lease for a CARLA endpoint."""

    _held_paths: set[Path] = set()
    _guard = threading.Lock()

    def __init__(self, path: Path, owner: str) -> None:
        self.path = path
        self.owner = owner
        self._handle: Any | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resolved = self.path.resolve()
        with self._guard:
            if resolved in self._held_paths:
                raise TickLeaseUnavailable(f"tick lease already held: {resolved}")
            handle = self.path.open("a+", encoding="utf-8")
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    # Windows: process-local set already guards same-process double-acquire.
                    try:
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError as exc:
                        handle.close()
                        raise TickLeaseUnavailable(
                            f"tick lease already held: {resolved}"
                        ) from exc
            except BlockingIOError as exc:
                handle.close()
                raise TickLeaseUnavailable(f"tick lease already held: {resolved}") from exc
            handle.seek(0)
            handle.truncate()
            json.dump({"owner": self.owner, "acquired_wall_time": time.time()}, handle, sort_keys=True)
            handle.flush()
            self._held_paths.add(resolved)
            self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        resolved = self.path.resolve()
        with self._guard:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            else:
                try:
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            self._handle.close()
            self._held_paths.discard(resolved)
            self._handle = None


class RunRegistry:
    """Durable attempt registry with a non-success finalization phase."""

    RUNNING = "RUNNING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    CRASHED = "CRASHED"
    TERMINAL = frozenset({COMPLETED, FAILED, INTERRUPTED, CLEANUP_FAILED, CRASHED})

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS scenario_attempts (
                run_id TEXT PRIMARY KEY, scenario_id TEXT NOT NULL, attempt_id INTEGER NOT NULL,
                config_hash TEXT NOT NULL, status TEXT NOT NULL, started_wall_time REAL NOT NULL,
                ended_wall_time REAL, failure_code TEXT, actor_manifest_json TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def begin(
        self,
        identity: RunIdentity,
        config_hash: str,
        actor_manifest: Sequence[Mapping[str, Any]],
        *,
        allow_existing_running: bool = False,
    ) -> None:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT status FROM scenario_attempts WHERE run_id = ?", (identity.run_id,)
            ).fetchone()
            if existing is not None:
                if allow_existing_running and existing[0] == self.RUNNING:
                    return
                raise RuntimeViolation(f"run_id already registered with status={existing[0]}")
            connection.execute(
                "INSERT INTO scenario_attempts VALUES (?, ?, ?, ?, 'RUNNING', ?, NULL, NULL, ?)",
                (
                    identity.run_id,
                    identity.scenario_id,
                    identity.attempt_id,
                    config_hash,
                    time.time(),
                    json.dumps(list(actor_manifest), sort_keys=True),
                ),
            )

    def status(self, run_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM scenario_attempts WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else str(row[0])

    def record_finalizing(self, run_id: str) -> None:
        """Move RUNNING to FINALIZING before any cleanup begins."""

        with self._connect() as connection:
            current = connection.execute("SELECT status FROM scenario_attempts WHERE run_id=?", (run_id,)).fetchone()
            if current is None:
                raise RuntimeViolation(f"unknown run_id={run_id}")
            if current[0] == self.FINALIZING:
                return
            if current[0] != self.RUNNING:
                raise RuntimeViolation(f"cannot finalize status={current[0]}")
            connection.execute(
                "UPDATE scenario_attempts SET status=? WHERE run_id=? AND status=?",
                (self.FINALIZING, run_id, self.RUNNING),
            )

    def finish(self, run_id: str, status: str, failure_code: str | None = None) -> None:
        if status not in self.TERMINAL:
            raise ValueError(f"invalid terminal run status={status}")
        with self._connect() as connection:
            current = connection.execute("SELECT status FROM scenario_attempts WHERE run_id=?", (run_id,)).fetchone()
            if current is None:
                raise RuntimeViolation(f"unknown run_id={run_id}")
            current_status = str(current[0])
            if current_status == status:
                return
            allowed = {self.FINALIZING}
            if status in {self.FAILED, self.INTERRUPTED, self.CRASHED, self.CLEANUP_FAILED}:
                allowed.add(self.RUNNING)
            if current_status not in allowed:
                raise RuntimeViolation(f"cannot finish {current_status} as {status}")
            connection.execute(
                "UPDATE scenario_attempts SET status=?, ended_wall_time=?, failure_code=? "
                "WHERE run_id=? AND status=?",
                (status, time.time(), failure_code, run_id, current_status),
            )

    def mark_crashed(self, run_id: str, *, exit_code: int | None, detail: str) -> None:
        """Parent-side terminal transition for a child killed by native signal."""
        code = f"child_exit={exit_code}; {detail}"
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, failure_code FROM scenario_attempts WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise RuntimeViolation(f"unknown run_id={run_id}")
            if row[0] == self.CRASHED:
                return
            previous = f"previous={row[0]}:{row[1]}; " if row[1] else f"previous={row[0]}; "
            connection.execute(
                "UPDATE scenario_attempts SET status=?, ended_wall_time=?, failure_code=? WHERE run_id=?",
                (self.CRASHED, time.time(), previous + code, run_id),
            )

    def record(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM scenario_attempts WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["actor_manifest"] = json.loads(result.pop("actor_manifest_json"))
        return result


@dataclass
class _SensorBarrier:
    names: set[str]
    received: dict[str, int] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def accept(self, name: str, measurement: Any) -> None:
        frame = getattr(measurement, "frame", None)
        if isinstance(frame, int):
            with self._lock:
                self.received[name] = frame
                self.measurements[name] = measurement

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.received)

    def measurement(self, name: str, frame: int) -> Any:
        """Return the latest payload only when it belongs to ``frame``."""

        with self._lock:
            if name not in self.names:
                raise RuntimeViolation(f"unknown_sensor name={name}")
            actual = self.received.get(name)
            if actual != frame or name not in self.measurements:
                raise RuntimeViolation(
                    f"sensor_frame_unavailable name={name} expected={frame} actual={actual}"
                )
            return self.measurements[name]

    def await_frame(self, frame: int, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if all(self.snapshot().get(name) == frame for name in self.names):
                return
            time.sleep(0.001)
        missing = sorted(name for name in self.names if self.snapshot().get(name) != frame)
        raise RuntimeViolation(f"sensor_barrier_timeout frame={frame} sensors={missing}")


@dataclass
class _SensorEventBuffer:
    """Thread-safe append-only buffer for sparse collision/lane events."""

    names: set[str]
    events: dict[str, list[tuple[int, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def accept(self, name: str, measurement: Any) -> None:
        frame = getattr(measurement, "frame", None)
        if name not in self.names or not isinstance(frame, int):
            return
        with self._lock:
            self.events.setdefault(name, []).append((frame, measurement))

    def read(self, name: str, since_frame: int, through_frame: int | None = None) -> tuple[Any, ...]:
        if name not in self.names:
            raise RuntimeViolation(f"unknown_event_sensor name={name}")
        upper = int(through_frame) if through_frame is not None else 2**63 - 1
        with self._lock:
            return tuple(
                measurement
                for frame, measurement in self.events.get(name, ())
                if int(since_frame) <= frame <= upper
            )


class _SensorCallbackGate:
    """Admission gate and worker queue for sensor callbacks during shutdown."""

    def __init__(self, barrier: _SensorBarrier, event_buffer: _SensorEventBuffer) -> None:
        self.barrier = barrier
        self.event_buffer = event_buffer
        self._accepting = True
        self._inflight = 0
        self._lock = threading.Condition(threading.Lock())
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue()
        self._worker_errors: list[BaseException] = []
        self.before_enqueue: Callable[[], None] | None = None
        self.process_item: Callable[[], None] | None = None
        self._worker = threading.Thread(target=self._run, name="sdf-sensor-worker", daemon=False)
        self._worker.start()

    def callback(self, name: str) -> Callable[[Any], None]:
        def receive(measurement: Any) -> None:
            with self._lock:
                if not self._accepting:
                    return
                self._inflight += 1
            try:
                if self.before_enqueue is not None:
                    self.before_enqueue()
                self._queue.put((name, measurement))
            finally:
                with self._lock:
                    self._inflight -= 1
                    self._lock.notify_all()

        return receive

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self.process_item is not None:
                    self.process_item()
                name, measurement = item
                if name in self.barrier.names:
                    self.barrier.accept(name, measurement)
                else:
                    self.event_buffer.accept(name, measurement)
            except BaseException as exc:  # worker errors become cleanup failures
                with self._lock:
                    self._worker_errors.append(exc)
            finally:
                self._queue.task_done()

    def stop_admission_and_wait(self) -> None:
        with self._lock:
            self._accepting = False
            while self._inflight:
                self._lock.wait()

    def flush(self) -> None:
        """Wait until every callback admitted so far has been processed."""

        self._queue.join()

    def drain_and_join(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._queue.join()
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            raise CleanupFailed("sensor worker did not join")
        with self._lock:
            if self._worker_errors:
                raise CleanupFailed(f"sensor worker failed: {type(self._worker_errors[0]).__name__}")


class ScenarioRuntime:
    """Owns exactly one CARLA lifecycle from settings through cleanup."""

    def __init__(
        self,
        *,
        client: Any,
        identity: RunIdentity,
        profile: RuntimeProfile,
        registry: RunRegistry,
        lease_path: Path,
        owner: str = "sdf.g1-02.runtime",
        adopt_existing_running: bool = False,
    ) -> None:
        profile.assert_valid()
        self.client, self.identity, self.profile, self.registry = client, identity, profile, registry
        self.lease = TickLease(lease_path, owner)
        self.adopt_existing_running = adopt_existing_running
        self.world: Any | None = None
        self.spec: ScenarioSpec | None = None
        self._original_settings: Any | None = None
        self._original_weather: Any | None = None
        self._traffic_manager: Any | None = None
        self._actors: dict[str, Any] = {}
        self._sensors: dict[str, Any] = {}
        self._actor_specs: dict[str, ActorSpec] = {}
        self._sensor_ids: dict[str, int] = {}
        self._actor_ids: dict[str, int] = {}
        self._barrier: _SensorBarrier | None = None
        self._event_buffer: _SensorEventBuffer | None = None
        self._callback_gate: _SensorCallbackGate | None = None
        self._started = False
        self._registry_started = False
        self._closing = False
        self._closed = False
        self._terminal_status: str | None = None
        self._close_error: BaseException | None = None
        self._lifecycle = threading.Condition(threading.RLock())
        self._tick_lock = threading.Lock()

    @staticmethod
    def config_hash(spec: ScenarioSpec, profile: RuntimeProfile) -> str:
        def transform_payload(transform: Any) -> Any:
            location = getattr(transform, "location", None)
            rotation = getattr(transform, "rotation", None)
            if location is not None and rotation is not None:
                return {
                    "location": {"x": float(location.x), "y": float(location.y), "z": float(location.z)},
                    "rotation": {"pitch": float(rotation.pitch), "yaw": float(rotation.yaw), "roll": float(rotation.roll)},
                }
            return str(transform)

        payload = {
            "scenario": {
                "scenario_id": spec.scenario_id,
                "map_name": spec.map_name,
                "actors": [
                    {
                        "name": item.name, "blueprint": item.blueprint,
                        "transform": transform_payload(item.transform), "role": item.role,
                        "spawn_order": item.spawn_order, "autopilot": item.autopilot,
                    }
                    for item in spec.ordered_actors()
                ],
                "sensors": [
                    {
                        "name": item.name, "blueprint": item.blueprint,
                        "transform": transform_payload(item.transform), "parent": item.parent,
                        "spawn_order": item.spawn_order,
                        "attributes": dict(sorted(item.attributes.items())),
                        "delivery": item.delivery,
                    }
                    for item in spec.ordered_sensors()
                ],
                "traffic_manager_port": spec.traffic_manager_port,
                "traffic_manager_seed": spec.traffic_manager_seed,
                "sensor_timeout_seconds": spec.sensor_timeout_seconds,
                "weather": dict(sorted(spec.weather.items())),
            },
            "profile": asdict(profile),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def actor_manifest(spec: ScenarioSpec) -> list[dict[str, Any]]:
        return [
            {"name": actor.name, "blueprint": actor.blueprint, "role": actor.role, "spawn_order": actor.spawn_order}
            for actor in spec.ordered_actors()
        ]

    def start(self, spec: ScenarioSpec) -> None:
        with self._lifecycle:
            if self._started or self._closing or self._closed:
                raise RuntimeViolation("runtime is already started or closed")
            if spec.scenario_id != self.identity.scenario_id:
                raise RuntimeViolation("scenario_id does not match RunIdentity")
        self.lease.acquire()
        self.spec = spec
        self._actor_specs = {item.name: item for item in spec.ordered_actors()}
        try:
            self.world = self.client.get_world()
            current_map = str(self.world.get_map().name)
            actual_map = current_map.rsplit("/", 1)[-1]
            if actual_map != spec.map_name:
                raise RuntimeViolation(
                    f"map_mismatch actual={current_map} expected={spec.map_name}; cold restart required"
                )
            self._configure_world()
            self._configure_traffic_manager(spec)
            self.registry.begin(
                self.identity,
                self.config_hash(spec, self.profile),
                self.actor_manifest(spec),
                allow_existing_running=self.adopt_existing_running,
            )
            self._registry_started = True
            self._spawn_all(spec)
            with self._lifecycle:
                self._started = True
        except BaseException as exc:
            with self._lifecycle:
                self._started = False
            if self._registry_started:
                try:
                    self._finalize("FAILED", failure_code=type(exc).__name__)
                except BaseException:
                    pass
            else:
                with contextlib.suppress(BaseException):
                    self._cleanup_impl()
            raise

    def _configure_world(self) -> None:
        assert self.world is not None
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.profile.fixed_delta_seconds
        settings.substepping = True
        settings.max_substep_delta_time = self.profile.max_substep_delta_time
        settings.max_substeps = self.profile.max_substeps
        self.world.apply_settings(settings)
        if self.spec is not None and self.spec.weather:
            getter = getattr(self.world, "get_weather", None)
            setter = getattr(self.world, "set_weather", None)
            if getter is None or setter is None:
                raise RuntimeViolation("world weather API unavailable")
            self._original_weather = getter()
            weather = getter()
            for name, value in sorted(self.spec.weather.items()):
                if not hasattr(weather, name):
                    raise RuntimeViolation(f"unknown weather attribute: {name}")
                setattr(weather, name, float(value))
            setter(weather)

    def _configure_traffic_manager(self, spec: ScenarioSpec) -> None:
        # Skip TM when no actor uses autopilot. Creating a TM binds a Windows-side
        # port and can fail with "bind error" after prior crashes even if unused.
        if not any(actor.autopilot for actor in spec.actors):
            self._traffic_manager = None
            return
        self._traffic_manager = self.client.get_trafficmanager(spec.traffic_manager_port)
        self._traffic_manager.set_synchronous_mode(True)
        self._traffic_manager.set_random_device_seed(spec.traffic_manager_seed)

    def _spawn_all(self, spec: ScenarioSpec) -> None:
        assert self.world is not None
        blueprints = self.world.get_blueprint_library()
        for actor_spec in spec.ordered_actors():
            bp_found = blueprints.find(actor_spec.blueprint)
            actor = self.world.try_spawn_actor(bp_found, actor_spec.transform)
            if actor is None:
                try:
                    import carla as _carla
                    raised = _carla.Transform(
                        _carla.Location(
                            x=float(actor_spec.transform.location.x),
                            y=float(actor_spec.transform.location.y),
                            z=float(actor_spec.transform.location.z) + 0.8,
                        ),
                        actor_spec.transform.rotation,
                    )
                    actor = self.world.try_spawn_actor(bp_found, raised)
                except Exception:
                    pass
            if actor is None:
                raise RuntimeViolation(f"spawn_failed actor={actor_spec.name}")
            self._actors[actor_spec.name] = actor
            self._actor_ids[actor_spec.name] = int(getattr(actor, "id", -1))
            if actor_spec.autopilot:
                actor.set_autopilot(True, spec.traffic_manager_port)
        frame_names = {sensor.name for sensor in spec.sensors if sensor.delivery == "frame"}
        event_names = {sensor.name for sensor in spec.sensors if sensor.delivery == "event"}
        self._barrier = _SensorBarrier(frame_names)
        self._event_buffer = _SensorEventBuffer(event_names)
        self._callback_gate = _SensorCallbackGate(self._barrier, self._event_buffer)
        for sensor_spec in spec.ordered_sensors():
            parent = self._actors.get(sensor_spec.parent)
            if parent is None:
                raise RuntimeViolation(f"sensor parent not spawned: {sensor_spec.parent}")
            blueprint = blueprints.find(sensor_spec.blueprint)
            if sensor_spec.attributes:
                setter = getattr(blueprint, "set_attribute", None)
                if setter is None:
                    raise RuntimeViolation(
                        f"sensor_blueprint_attributes_unsupported name={sensor_spec.name}"
                    )
                for key, value in sorted(sensor_spec.attributes.items()):
                    setter(str(key), str(value))
            sensor = self.world.spawn_actor(blueprint, sensor_spec.transform, attach_to=parent)
            sensor.listen(self._callback_gate.callback(sensor_spec.name))
            self._sensors[sensor_spec.name] = sensor
            self._sensor_ids[sensor_spec.name] = int(getattr(sensor, "id", -1))

    def tick(self, control: Any, apply_control: Callable[[Any, Any], None] | None = None) -> FrameHeader:
        if self.spec is None:
            raise RuntimeViolation("runtime is not started")
        ego_names = [item.name for item in self.spec.ordered_actors() if item.role == "ego"]
        if len(ego_names) != 1:
            raise RuntimeViolation("scenario requires exactly one ego actor for control")
        return self.tick_controls({ego_names[0]: control}, apply_control=apply_control)

    def tick_controls(
        self,
        controls: Mapping[str, Any],
        apply_control: Callable[[Any, Any], None] | None = None,
    ) -> FrameHeader:
        """Apply ego/NPC controls, then advance the single owned tick exactly once."""

        with self._lifecycle:
            if not self._started or self._closing or self._closed or self.world is None or self.spec is None:
                raise RuntimeViolation("runtime is not started")
        error: BaseException | None = None
        result: FrameHeader | None = None
        with self._tick_lock:
            try:
                unknown = sorted(set(controls) - set(self._actors))
                if unknown:
                    raise RuntimeViolation(f"unknown controlled actors: {unknown}")
                applier = apply_control or (lambda actor, command: actor.apply_control(command))
                for name in sorted(controls):
                    applier(self._actors[name], controls[name])
                frame = int(self.world.tick())
                snapshot = self.world.get_snapshot()
                if int(snapshot.frame) != frame:
                    raise RuntimeViolation(f"snapshot_frame_mismatch tick={frame} snapshot={snapshot.frame}")
                if self._barrier is not None and self._barrier.names:
                    self._barrier.await_frame(frame, self.spec.sensor_timeout_seconds)
                if self._callback_gate is not None:
                    self._callback_gate.flush()
                result = FrameHeader(
                    identity=self.identity,
                    carla_frame=frame,
                    simulation_time=float(snapshot.timestamp.elapsed_seconds),
                    wall_time=time.time(),
                )
            except BaseException as exc:
                error = exc
        if error is not None:
            self.abort(type(error).__name__)
            raise error
        assert result is not None
        return result

    def sensor_measurement(self, name: str, frame: int) -> Any:
        """Read one payload already admitted by this runtime's frame barrier.

        This method never waits and never advances CARLA.  Callers must first
        obtain ``frame`` from :meth:`tick`; stale, future and unknown frames fail
        closed rather than returning a nearby measurement.
        """

        with self._lifecycle:
            if not self._started or self._closing or self._closed or self._barrier is None:
                raise RuntimeViolation("runtime is not started")
            return self._barrier.measurement(name, int(frame))

    def sensor_events(
        self, name: str, *, since_frame: int, through_frame: int | None = None
    ) -> tuple[Any, ...]:
        """Return sparse events already admitted in the requested inclusive frame range."""

        with self._lifecycle:
            if not self._started or self._closing or self._closed or self._event_buffer is None:
                raise RuntimeViolation("runtime is not started")
            if self._callback_gate is not None:
                self._callback_gate.flush()
            return self._event_buffer.read(name, int(since_frame), through_frame)

    def complete(self) -> None:
        self._finalize("COMPLETED")

    def abort(self, failure_code: str) -> None:
        self._finalize("INTERRUPTED", failure_code=failure_code)

    def close(self) -> None:
        """Idempotent non-success close; COMPLETED is only set by ``complete``."""

        self._finalize("INTERRUPTED", failure_code="close")

    def _finalize(self, requested_status: str, failure_code: str | None = None) -> None:
        with self._lifecycle:
            if self._closed:
                if requested_status == "COMPLETED" and self._terminal_status != RunRegistry.COMPLETED:
                    raise RuntimeViolation(f"runtime already closed as {self._terminal_status}")
                if self._close_error is not None:
                    raise self._close_error
                return
            if self._closing:
                while not self._closed:
                    self._lifecycle.wait()
                if requested_status == "COMPLETED" and self._terminal_status != RunRegistry.COMPLETED:
                    raise RuntimeViolation(f"runtime already closed as {self._terminal_status}")
                if self._close_error is not None:
                    raise self._close_error
                return
            self._closing = True
            self._started = False  # stop new control/tick admission

        # Wait for an in-progress tick/control call before touching CARLA.
        with self._tick_lock:
            pass

        cleanup_error: BaseException | None = None
        if self._registry_started:
            with contextlib.suppress(BaseException):
                self.registry.record_finalizing(self.identity.run_id)
        try:
            self._cleanup_impl()
        except BaseException as exc:
            cleanup_error = exc

        terminal = requested_status
        if cleanup_error is not None:
            terminal = RunRegistry.CLEANUP_FAILED
        if self._registry_started:
            try:
                self.registry.finish(self.identity.run_id, terminal, failure_code or (type(cleanup_error).__name__ if cleanup_error else None))
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
                terminal = RunRegistry.CLEANUP_FAILED
                with contextlib.suppress(BaseException):
                    self.registry.finish(self.identity.run_id, terminal, type(exc).__name__)

        with self._lifecycle:
            self._terminal_status = terminal
            self._close_error = cleanup_error
            self._closed = True
            self._closing = False
            self._lifecycle.notify_all()
        if cleanup_error is not None:
            if isinstance(cleanup_error, CleanupFailed):
                raise cleanup_error
            raise CleanupFailed(str(cleanup_error)) from cleanup_error

    def _cleanup_impl(self) -> None:
        """Ordered, observable cleanup; lease release is deliberately last."""

        errors: list[str] = []
        cleanup_sensor_refs = dict(self._sensors)
        cleanup_actor_refs = dict(self._actors)
        # 1. stop sensor production first, then close callback admission.
        for name, sensor in list(self._sensors.items()):
            try:
                sensor.stop()
            except BaseException as exc:
                errors.append(f"sensor.stop({name}): {type(exc).__name__}")
        gate = self._callback_gate
        if gate is not None:
            try:
                gate.stop_admission_and_wait()  # 2-3: gate + in-flight callbacks
                gate.drain_and_join()  # 4-5: queue drain + worker join
            except BaseException as exc:
                errors.append(str(exc))

        # 6. sensor -> NPC -> Ego, with no new callbacks admitted.
        for name, sensor in reversed(list(self._sensors.items())):
            try:
                result = sensor.destroy()
                if result is False:
                    errors.append(f"sensor.destroy({name}) returned false")
            except BaseException as exc:
                errors.append(f"sensor.destroy({name}): {type(exc).__name__}")
        self._sensors.clear()
        for role in ("npc", "ego"):
            for name, actor in list(self._actors.items()):
                actor_spec = self._actor_specs.get(name, ActorSpec(name, "", None, role, 0))
                if actor_spec.role != role:
                    continue
                try:
                    # Traffic Manager can still issue a command to an
                    # autopilot NPC while it is being destroyed.  Withdraw
                    # autopilot control while the actor is alive; TM itself is
                    # restored only after the required sensor -> NPC -> Ego
                    # destruction order.
                    if role == "npc" and actor_spec.autopilot and hasattr(actor, "set_autopilot"):
                        actor.set_autopilot(False, self.spec.traffic_manager_port if self.spec else 8000)
                    result = actor.destroy()
                    if result is False:
                        errors.append(f"actor.destroy({name}) returned false")
                except BaseException as exc:
                    errors.append(f"actor.destroy({name}): {type(exc).__name__}")
                self._actors.pop(name, None)
        for name, actor in list(self._actors.items()):
            with contextlib.suppress(BaseException):
                actor.destroy()
            self._actors.pop(name, None)

        # 7. restore Traffic Manager/weather, then 8. world settings.
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except BaseException as exc:
                errors.append(f"traffic_manager.restore: {type(exc).__name__}")
        if self.world is not None and self._original_weather is not None:
            try:
                self.world.set_weather(self._original_weather)
            except BaseException as exc:
                errors.append(f"weather.restore: {type(exc).__name__}")
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except BaseException as exc:
                errors.append(f"world_settings.restore: {type(exc).__name__}")

        # 9. post-cleanup verification happens before lease release.
        try:
            self._verify_cleanup(cleanup_actor_refs, cleanup_sensor_refs)
        except BaseException as exc:
            errors.append(str(exc))

        # 10. lease release is always last, even if verification failed.
        try:
            self.lease.release()
        except BaseException as exc:
            errors.append(f"tick_lease.release: {type(exc).__name__}")
        self._callback_gate = None
        self._barrier = None
        self._event_buffer = None
        if errors:
            raise CleanupFailed("; ".join(errors))

    def _verify_cleanup(self, actor_refs: Mapping[str, Any], sensor_refs: Mapping[str, Any]) -> None:
        if self.world is not None and callable(getattr(self.world, "get_actors", None)):
            target_ids = {
                name: actor_id
                for name, actor_id in {**self._actor_ids, **self._sensor_ids}.items()
                if actor_id >= 0
            }
            deadline = time.monotonic() + 3.0
            residue = target_ids
            while residue and time.monotonic() < deadline:
                active_ids = {int(actor.id) for actor in self.world.get_actors()}
                residue = {name: actor_id for name, actor_id in target_ids.items() if actor_id in active_ids}
                if residue:
                    time.sleep(0.05)
            if residue:
                raise CleanupFailed(f"cleanup verification found actor residue: {residue}")
        else:
            # Protocol-compatible fake actors expose ``destroyed``.
            for name, actor in {**actor_refs, **sensor_refs}.items():
                if getattr(actor, "destroyed", True) is not True:
                    raise CleanupFailed(f"cleanup verification actor remains: {name}")
        if self.world is not None and self._original_settings is not None:
            current = self.world.get_settings()
            fields = ("synchronous_mode", "fixed_delta_seconds", "substepping", "max_substep_delta_time", "max_substeps")
            mismatches = [field for field in fields if getattr(current, field, None) != getattr(self._original_settings, field, None)]
            if mismatches:
                raise CleanupFailed(f"world settings not restored: {mismatches}")
