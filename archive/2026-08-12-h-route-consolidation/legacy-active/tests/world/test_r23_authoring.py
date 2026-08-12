from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from driving_vla.evaluation.fixture_runtime import purge_episode_actors
from driving_vla.evaluation.paired_live import (
    SensorBundle,
    _destroy_sensors,
    _release_decision_controls,
)
from runtime.carla_connection import _linux_carla_process_running
from scripts.r23_author_scenarios import (
    _adjacent,
    _actor_script_lines,
    _topology_signature,
    _weather_lines,
)


def waypoint(
    *,
    road: int,
    lane: int,
    s: float,
    x: float,
    y: float,
    z: float,
    yaw: float,
):
    return SimpleNamespace(
        road_id=road,
        lane_id=lane,
        s=s,
        is_junction=False,
        lane_width=3.5,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=x, y=y, z=z),
            rotation=SimpleNamespace(yaw=yaw),
        ),
    )


class R23AuthoringTest(unittest.TestCase):
    def test_adjacent_lane_must_be_same_type_and_travel_direction(self) -> None:
        same = SimpleNamespace(lane_id=-2, lane_type=1)
        oncoming = SimpleNamespace(lane_id=1, lane_type=1)
        base = SimpleNamespace(
            lane_id=-1,
            lane_type=1,
            get_left_lane=lambda: oncoming,
            get_right_lane=lambda: same,
        )
        self.assertIs(_adjacent(base), same)
        base.get_right_lane = lambda: None
        self.assertIsNone(_adjacent(base))

    def test_release_decision_controls_clears_camera_wait_ego_brake(self) -> None:
        controls = []
        ego_actor = SimpleNamespace(apply_control=lambda control: controls.append(control))
        session = SimpleNamespace(
            spawned=[SimpleNamespace(role="ego", actor=ego_actor)]
        )
        fake_carla = SimpleNamespace(
            VehicleControl=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        with patch.dict("sys.modules", {"carla": fake_carla}):
            _release_decision_controls(session)
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0].brake, 0.0)
        self.assertEqual(controls[0].throttle, 0.0)

    def test_sensor_cleanup_releases_proxies_before_batch_destroy(self) -> None:
        events = []

        class Sensor:
            def __init__(self, actor_id):
                self.id = actor_id

            def stop(self):
                events.append(("stop", self.id))

        class Client:
            def apply_batch_sync(self, commands, do_tick):
                self.proxies_released = (
                    bundle.camera is None
                    and bundle.collision is None
                    and bundle.lane is None
                )
                events.append(("batch", tuple(commands)))
                return []

        bundle = SensorBundle(
            camera=Sensor(1), collision=Sensor(2), lane=Sensor(3)
        )
        client = Client()
        fake_carla = SimpleNamespace(
            command=SimpleNamespace(DestroyActor=lambda actor_id: actor_id)
        )
        with patch.dict("sys.modules", {"carla": fake_carla}):
            _destroy_sensors(bundle, client=client)
        self.assertTrue(client.proxies_released)
        self.assertEqual(events[-1], ("batch", (3, 2, 1)))

    def test_purge_restores_rpc_timeout_after_short_cleanup_window(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.timeouts = []

            def set_timeout(self, value):
                self.timeouts.append(float(value))

            def apply_batch_sync(self, commands, do_tick):
                return [SimpleNamespace(has_error=lambda: False) for _ in commands]

        actor = SimpleNamespace(id=7, type_id="vehicle.test")
        world = SimpleNamespace(get_actors=lambda: [actor], tick=lambda: 1)
        client = FakeClient()
        fake_carla = SimpleNamespace(
            command=SimpleNamespace(DestroyActor=lambda actor_id: actor_id)
        )
        with patch.dict("sys.modules", {"carla": fake_carla}):
            self.assertEqual(purge_episode_actors(world, client=client), [])
        self.assertEqual(client.timeouts, [5.0, 60.0])

    @patch("runtime.carla_connection.subprocess.run")
    def test_linux_carla_probe_uses_exact_process_name(self, run) -> None:
        run.return_value = SimpleNamespace(
            returncode=1,
            stdout="",
        )
        self.assertEqual(_linux_carla_process_running(), "NOT_RUNNING")
        command = run.call_args.args[0]
        self.assertIn("-x", command)
        self.assertNotIn("-f", command)

    def test_topology_signature_is_stable_and_location_sensitive(self) -> None:
        route = [
            waypoint(
                road=1,
                lane=-1,
                s=index * 8.0,
                x=index * 8.0,
                y=0.0,
                z=0.1 * index,
                yaw=2.0 * index,
            )
            for index in range(5)
        ]
        first_hash, first_body = _topology_signature("Town03", route)
        second_hash, second_body = _topology_signature("Town03", route)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_body, second_body)
        shifted = [
            waypoint(
                road=1,
                lane=-1,
                s=80.0 + index * 8.0,
                x=80.0 + index * 8.0,
                y=0.0,
                z=0.1 * index,
                yaw=2.0 * index,
            )
            for index in range(5)
        ]
        self.assertNotEqual(
            first_hash, _topology_signature("Town03", shifted)[0]
        )

    def test_v2_conditions_bind_weather_and_actor_mode(self) -> None:
        self.assertTrue(any("ClearNoon" in row for row in _weather_lines("s", 0)))
        self.assertTrue(any("HardRainNight" in row for row in _weather_lines("s", 5)))
        fixed = _actor_script_lines("a", "crossing", 2)
        reactive = _actor_script_lines("a", "crossing", 5)
        self.assertFalse(any("reactive_yield" in row for row in fixed))
        self.assertTrue(any("reactive_yield" in row for row in reactive))


if __name__ == "__main__":
    unittest.main()
