from __future__ import annotations

import copy
import os
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from runtime import (  # noqa: E402
    ActorSpec,
    RunRegistry,
    RuntimeIdentityFactory,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    TickLeaseUnavailable,
    load_runtime_profiles,
)
from runtime.scenario_runtime import RuntimeViolation  # noqa: E402


@dataclass
class FakeSettings:
    synchronous_mode: bool = False
    fixed_delta_seconds: float | None = None
    substepping: bool = False
    max_substep_delta_time: float | None = None
    max_substeps: int | None = None


class FakeTimestamp:
    def __init__(self, seconds: float) -> None:
        self.elapsed_seconds = seconds


class FakeSnapshot:
    def __init__(self, frame: int, seconds: float) -> None:
        self.frame = frame
        self.timestamp = FakeTimestamp(seconds)


class FakeActor:
    _next_id = 100

    def __init__(self, name: str) -> None:
        self.id = FakeActor._next_id
        FakeActor._next_id += 1
        self.name, self.destroyed, self.control, self.autopilot = name, False, None, None

    def apply_control(self, control: object) -> None:
        self.control = control

    def set_autopilot(self, enabled: bool, port: int) -> None:
        self.autopilot = (enabled, port)

    def destroy(self) -> None:
        self.destroyed = True


class FakeSensor(FakeActor):
    def __init__(self, name: str, world: "FakeWorld") -> None:
        super().__init__(name)
        self.world, self.callback, self.stopped = world, None, False

    def listen(self, callback: object) -> None:
        self.callback = callback

    def stop(self) -> None:
        self.stopped = True


class FakeBlueprints:
    @staticmethod
    def find(name: str) -> str:
        return name


class FakeWorld:
    def __init__(self, map_name: str = "Town01") -> None:
        self.map = type("Map", (), {"name": map_name})()
        self.settings, self.frame, self.applied_settings = FakeSettings(), 0, []
        self.actors: list[FakeActor] = []
        self.sensors: list[FakeSensor] = []
        self.emit_sensors = True

    def get_map(self) -> object:
        return self.map

    def get_actors(self) -> list[FakeActor]:
        return [actor for actor in [*self.actors, *self.sensors] if not actor.destroyed]

    def get_settings(self) -> FakeSettings:
        return copy.deepcopy(self.settings)

    def apply_settings(self, settings: FakeSettings) -> None:
        self.settings = copy.deepcopy(settings)
        self.applied_settings.append(copy.deepcopy(settings))

    def get_blueprint_library(self) -> FakeBlueprints:
        return FakeBlueprints()

    def try_spawn_actor(self, blueprint: str, transform: object) -> FakeActor:
        actor = FakeActor(str(transform))
        self.actors.append(actor)
        return actor

    def spawn_actor(self, blueprint: str, transform: object, attach_to: FakeActor) -> FakeSensor:
        sensor = FakeSensor(str(transform), self)
        self.sensors.append(sensor)
        return sensor

    def tick(self) -> int:
        self.frame += 1
        if self.emit_sensors:
            measurement = type("Measurement", (), {"frame": self.frame})()
            for sensor in self.sensors:
                if sensor.callback is not None:
                    sensor.callback(measurement)
        return self.frame

    def get_snapshot(self) -> FakeSnapshot:
        return FakeSnapshot(self.frame, self.frame * float(self.settings.fixed_delta_seconds))


class FakeTrafficManager:
    def __init__(self) -> None:
        self.sync_calls: list[bool] = []
        self.seed: int | None = None

    def set_synchronous_mode(self, value: bool) -> None:
        self.sync_calls.append(value)

    def set_random_device_seed(self, value: int) -> None:
        self.seed = value


class FakeClient:
    def __init__(self, world: FakeWorld | None = None) -> None:
        self.world = world or FakeWorld()
        self.loaded_maps: list[str] = []
        self.traffic_manager, self.tm_port = FakeTrafficManager(), None

    def get_world(self) -> FakeWorld:
        return self.world

    def load_world(self, name: str) -> FakeWorld:
        self.loaded_maps.append(name)
        self.world.map.name = name
        return self.world

    def get_trafficmanager(self, port: int) -> FakeTrafficManager:
        self.tm_port = port
        return self.traffic_manager


class G102ScenarioRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.registry, self.lease = RunRegistry(root / "runs.sqlite3"), root / "tick.lock"
        self.profile = load_runtime_profiles(ROOT / "safedrive_foundry/config/runtime_profiles.toml")["throughput_20hz"]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def identity(self, attempt: int = 0):
        return RuntimeIdentityFactory.create({
            "experiment_id": "g1-02", "scenario_id": "Town01.lifecycle", "attempt_id": attempt,
            "server_epoch": "carla-0.9.16-test", "producer_version": "test",
        })

    @staticmethod
    def spec(*, timeout: float = 0.05) -> ScenarioSpec:
        return ScenarioSpec(
            scenario_id="Town01.lifecycle", map_name="Town01", traffic_manager_port=8123,
            traffic_manager_seed=17, sensor_timeout_seconds=timeout,
            actors=(
                ActorSpec("npc-b", "vehicle.npc", "npc-b", "npc", 2, True),
                ActorSpec("ego", "vehicle.ego", "ego", "ego", 0),
                ActorSpec("npc-a", "vehicle.npc", "npc-a", "npc", 1, True),
            ),
            sensors=(SensorSpec("front_camera", "sensor.camera.rgb", "camera", "ego", 0),),
        )

    def runtime(self, client: FakeClient, attempt: int = 0) -> ScenarioRuntime:
        return ScenarioRuntime(client=client, identity=self.identity(attempt), profile=self.profile,
                               registry=self.registry, lease_path=self.lease)

    def test_empty_scene_and_single_ego_lifecycle(self) -> None:
        empty = ScenarioSpec("Town01.lifecycle", "Town01")
        runtime = self.runtime(FakeClient())
        runtime.start(empty)
        with self.assertRaises(RuntimeViolation):
            runtime.tick({"throttle": 0.0})
        runtime.abort("no_ego_expected")

        runtime = self.runtime(FakeClient(), attempt=1)
        runtime.start(self.spec())
        header = runtime.tick({"throttle": 0.2})
        self.assertEqual(header.carla_frame, 1)
        self.assertEqual(header.identity.run_id, self.identity(1).run_id)
        self.assertEqual(runtime._actors["ego"].control, {"throttle": 0.2})
        runtime.complete()
        self.assertEqual(self.registry.status(self.identity(1).run_id), "COMPLETED")

    def test_npc_tm_is_seeded_and_spawn_manifest_is_reproducible(self) -> None:
        first_client, second_client = FakeClient(), FakeClient()
        first, second = self.runtime(first_client), self.runtime(second_client, attempt=1)
        first.start(self.spec())
        first_manifest = ScenarioRuntime.actor_manifest(self.spec())
        self.assertEqual([actor.name for actor in self.spec().ordered_actors()], ["ego", "npc-a", "npc-b"])
        self.assertEqual(first_client.tm_port, 8123)
        self.assertEqual(first_client.traffic_manager.seed, 17)
        first.abort("test")
        second.start(self.spec())
        self.assertEqual(first_manifest, ScenarioRuntime.actor_manifest(self.spec()))
        second.complete()

    def test_sensor_timeout_marks_attempt_interrupted_and_cleans_up(self) -> None:
        client = FakeClient()
        client.world.emit_sensors = False
        runtime = self.runtime(client)
        runtime.start(self.spec(timeout=0.005))
        with self.assertRaisesRegex(RuntimeViolation, "sensor_barrier_timeout"):
            runtime.tick({})
        self.assertEqual(self.registry.status(self.identity().run_id), "INTERRUPTED")
        self.assertTrue(all(actor.destroyed for actor in client.world.actors))
        self.assertTrue(all(sensor.destroyed and sensor.stopped for sensor in client.world.sensors))
        self.assertFalse(client.world.settings.synchronous_mode)

    def test_second_runtime_cannot_acquire_tick_lease(self) -> None:
        first = self.runtime(FakeClient())
        second = self.runtime(FakeClient(), attempt=1)
        first.start(self.spec())
        with self.assertRaises(TickLeaseUnavailable):
            second.start(self.spec())
        first.abort("test")

    def test_map_load_and_forced_interrupt_are_non_success(self) -> None:
        client = FakeClient(FakeWorld("WrongTown"))
        runtime = self.runtime(client)
        runtime.start(self.spec())
        self.assertEqual(client.loaded_maps, ["Town01"])
        original_tick = client.world.tick
        client.world.tick = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            runtime.tick({})
        client.world.tick = original_tick
        self.assertEqual(self.registry.status(self.identity().run_id), "INTERRUPTED")

    def test_client_connection_failure_releases_lease(self) -> None:
        class BadClient:
            def get_world(self):
                raise ConnectionError("CARLA 127.0.0.1:2999 refused connection")

        with self.assertRaisesRegex(ConnectionError, "2999"):
            self.runtime(BadClient()).start(self.spec())
        retry = self.runtime(FakeClient(), attempt=1)
        retry.start(self.spec())
        retry.abort("test")

    def test_config_hash_accepts_noncopyable_carla_like_transform(self) -> None:
        class NonCopyableTransform:
            location = type("Location", (), {"x": 1.0, "y": 2.0, "z": 3.0})()
            rotation = type("Rotation", (), {"pitch": 4.0, "yaw": 5.0, "roll": 6.0})()

            def __deepcopy__(self, memo):
                raise RuntimeError("CARLA Transform cannot be deep-copied")

        spec = ScenarioSpec(
            "Town01.lifecycle", "Town01",
            actors=(ActorSpec("ego", "vehicle.ego", NonCopyableTransform(), "ego", 0),),
        )
        first = ScenarioRuntime.config_hash(spec, self.profile)
        second = ScenarioRuntime.config_hash(spec, self.profile)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_delayed_camera_callback_is_drained_before_close(self) -> None:
        runtime = self.runtime(FakeClient())
        runtime.start(self.spec())
        entered, release = threading.Event(), threading.Event()

        def delay_callback() -> None:
            entered.set()
            self.assertTrue(release.wait(2.0))

        assert runtime._callback_gate is not None
        runtime._callback_gate.before_enqueue = delay_callback
        tick_error: list[BaseException] = []
        tick_thread = threading.Thread(target=lambda: _capture(runtime.tick, {}, tick_error))
        tick_thread.start()
        self.assertTrue(entered.wait(2.0))
        close_thread = threading.Thread(target=runtime.complete)
        close_thread.start()
        time.sleep(0.02)
        self.assertTrue(close_thread.is_alive(), "close must wait for in-flight callback")
        release.set()
        tick_thread.join(2.0)
        close_thread.join(2.0)
        self.assertFalse(tick_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertEqual(tick_error, [])
        self.assertEqual(self.registry.status(self.identity().run_id), "COMPLETED")

    def test_nonempty_callback_queue_is_drained_and_worker_joined(self) -> None:
        runtime = self.runtime(FakeClient())
        runtime.start(self.spec())
        assert runtime._callback_gate is not None
        processing, release = threading.Event(), threading.Event()

        def delay_worker() -> None:
            processing.set()
            self.assertTrue(release.wait(2.0))

        runtime._callback_gate.process_item = delay_worker
        callback = runtime._callback_gate.callback("front_camera")
        callback(type("Measurement", (), {"frame": 7})())
        self.assertTrue(processing.wait(2.0))
        close_thread = threading.Thread(target=runtime.complete)
        close_thread.start()
        time.sleep(0.02)
        self.assertTrue(close_thread.is_alive())
        release.set()
        close_thread.join(2.0)
        self.assertFalse(close_thread.is_alive())
        self.assertEqual(self.registry.status(self.identity().run_id), "COMPLETED")

    def test_double_close_and_complete_after_close_are_idempotent(self) -> None:
        runtime = self.runtime(FakeClient())
        runtime.start(self.spec())
        runtime.close()
        runtime.close()
        self.assertEqual(self.registry.status(self.identity().run_id), "INTERRUPTED")
        with self.assertRaises(RuntimeViolation):
            runtime.complete()

        completed = self.runtime(FakeClient(), attempt=1)
        completed.start(self.spec())
        completed.complete()
        completed.close()
        self.assertEqual(self.registry.status(self.identity(1).run_id), "COMPLETED")

    def test_spawn_midway_exception_is_not_success_and_cleans_prior_actor(self) -> None:
        class SpawnFailureWorld(FakeWorld):
            def try_spawn_actor(self, blueprint: str, transform: object) -> FakeActor | None:
                if len(self.actors) == 1:
                    return None
                return super().try_spawn_actor(blueprint, transform)

        world = SpawnFailureWorld()
        runtime = self.runtime(FakeClient(world))
        with self.assertRaises(RuntimeViolation):
            runtime.start(self.spec())
        self.assertEqual(self.registry.status(self.identity().run_id), "FAILED")
        self.assertEqual(world.get_actors(), [])

    def test_cleanup_verification_failure_forbids_completed(self) -> None:
        class StickyWorld(FakeWorld):
            def get_actors(self) -> list[FakeActor]:
                return [*self.actors, *self.sensors]

        runtime = self.runtime(FakeClient(StickyWorld()))
        runtime.start(self.spec())
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            runtime.complete()
        self.assertEqual(self.registry.status(self.identity().run_id), "CLEANUP_FAILED")

    def test_completed_is_written_only_after_cleanup(self) -> None:
        observed: list[str | None] = []

        class TrackingRuntime(ScenarioRuntime):
            def _cleanup_impl(self):  # type: ignore[override]
                observed.append(self.registry.status(self.identity.run_id))
                return super()._cleanup_impl()

        runtime = TrackingRuntime(
            client=FakeClient(), identity=self.identity(), profile=self.profile,
            registry=self.registry, lease_path=self.lease,
        )
        runtime.start(self.spec())
        runtime.complete()
        self.assertEqual(observed, ["FINALIZING"])
        self.assertEqual(self.registry.status(self.identity().run_id), "COMPLETED")

    def test_parent_marks_child_sigabrt_as_crashed(self) -> None:
        identity = self.identity()
        self.registry.begin(identity, "parent-config", [])
        child = subprocess.run(
            [sys.executable, "-c", "import os; os.abort()"],
            env={**os.environ, "PYTHONFAULTHANDLER": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(child.returncode, 0)
        self.registry.mark_crashed(identity.run_id, exit_code=child.returncode, detail="native signal")
        self.assertEqual(self.registry.status(identity.run_id), "CRASHED")


def _capture(function: object, argument: object, errors: list[BaseException]) -> None:
    try:
        function(argument)  # type: ignore[operator]
    except BaseException as exc:
        errors.append(exc)


if __name__ == "__main__":
    unittest.main()
