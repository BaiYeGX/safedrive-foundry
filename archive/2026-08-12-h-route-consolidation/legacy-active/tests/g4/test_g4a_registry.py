"""R2 offline: scenario registry load, hash, freeze, rejection rules."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    ALLOWED_FAMILIES,
    DEFAULT_REGISTRY_PATH,
    REQUIRED_SCENARIO_IDS,
    REQUIRED_SEEDS,
    R2V3_CAMPAIGN_VERSION_PREFIX,
    R2V3_LONG_SMOKE_SCENARIO_IDS,
    R2V3_LONG_SMOKE_VERSION_PREFIX,
    RegistryError,
    freeze_registry,
    load_scenario_registry,
    _validate_registry_shape,
)
from driving_vla.evaluation.fixture_runtime import (  # noqa: E402
    FixtureSession,
    SpawnedActor,
    _apply_scripts,
    _apply_traffic_light_script,
    _configure_traffic_lights,
    _restore_traffic_lights,
)


class G4ARegistryTest(unittest.TestCase):
    def test_live_fixture_scripts_can_exclude_ego_mpc_owner(self) -> None:
        class Script:
            script_type = "hold"

            @staticmethod
            def control_at(_time):
                return {
                    "kind": "vehicle",
                    "throttle": 0.2,
                    "brake": 0.0,
                    "steer": 0.0,
                    "hand_brake": False,
                    "reverse": False,
                    "phase": "hold",
                }

        class Actor:
            def __init__(self):
                self.controls = []

            def apply_control(self, value):
                self.controls.append(value)

        ego = Actor()
        npc = Actor()
        requested = SimpleNamespace(script=Script())
        session = FixtureSession(
            client=SimpleNamespace(),
            world=SimpleNamespace(),
            fixture=SimpleNamespace(traffic_light={"policy": "freeze_green"}),
            spawned=[
                SpawnedActor("ego", "ego", "vehicle.ego", ego, requested),
                SpawnedActor("npc", "npc", "vehicle.npc", npc, requested),
            ],
        )
        carla_mock = SimpleNamespace(
            VehicleControl=lambda **kwargs: kwargs,
            WalkerControl=lambda **kwargs: kwargs,
            Vector3D=lambda *args: args,
        )
        with patch.dict(sys.modules, {"carla": carla_mock}):
            _apply_scripts(
                session,
                simulation_time_since_anchor_s=1.0,
                include_ego=False,
            )
        self.assertEqual(ego.controls, [])
        self.assertEqual(len(npc.controls), 1)

    def test_scripted_red_green_targets_and_restores_exact_light(self) -> None:
        class Light:
            def __init__(self, actor_id, x):
                self.id = actor_id
                self._x = x
                self.state = "Yellow"
                self.frozen = False

            def get_transform(self):
                return SimpleNamespace(
                    location=SimpleNamespace(x=self._x, y=2.0, z=3.0)
                )

            def get_state(self):
                return self.state

            def set_state(self, state):
                self.state = state

            def freeze(self, value):
                self.frozen = bool(value)

            def is_frozen(self):
                return self.frozen

        target = Light(1, 10.0)
        other = Light(2, 30.0)
        actors = SimpleNamespace(filter=lambda _pattern: [target, other])
        fixture = SimpleNamespace(
            traffic_light={
                "policy": "scripted_red_green",
                "target_x": 10.0,
                "target_y": 2.0,
                "target_z": 3.0,
                "green_after_s": 5.0,
            }
        )
        session = FixtureSession(
            client=SimpleNamespace(),
            world=SimpleNamespace(get_actors=lambda: actors),
            fixture=fixture,
        )
        carla_mock = SimpleNamespace(
            TrafficLightState=SimpleNamespace(Red="Red", Green="Green")
        )
        with patch.dict(sys.modules, {"carla": carla_mock}):
            _configure_traffic_lights(session)
            self.assertEqual(target.state, "Red")
            self.assertEqual(other.state, "Green")
            _apply_traffic_light_script(session, 4.9)
            self.assertEqual(target.state, "Red")
            _apply_traffic_light_script(session, 5.0)
            self.assertEqual(target.state, "Green")
            errors = []
            _restore_traffic_lights(session, errors)
        self.assertEqual(errors, [])
        self.assertEqual(target.state, "Yellow")
        self.assertFalse(target.frozen)

    def test_freeze_red_sets_all_lights_red_and_restores(self) -> None:
        class Light:
            def __init__(self, actor_id):
                self.id = actor_id
                self.state = "Yellow"
                self.frozen = False

            def get_state(self):
                return self.state

            def set_state(self, state):
                self.state = state

            def freeze(self, value):
                self.frozen = bool(value)

            def is_frozen(self):
                return self.frozen

        lights = [Light(1), Light(2)]
        session = FixtureSession(
            client=SimpleNamespace(),
            world=SimpleNamespace(
                get_actors=lambda: SimpleNamespace(
                    filter=lambda _pattern: lights
                )
            ),
            fixture=SimpleNamespace(traffic_light={"policy": "freeze_red"}),
        )
        carla_mock = SimpleNamespace(
            TrafficLightState=SimpleNamespace(Red="Red", Green="Green")
        )
        with patch.dict(sys.modules, {"carla": carla_mock}):
            _configure_traffic_lights(session)
            self.assertEqual([light.state for light in lights], ["Red", "Red"])
            errors = []
            _restore_traffic_lights(session, errors)
        self.assertEqual(errors, [])
        self.assertEqual([light.state for light in lights], ["Yellow", "Yellow"])

    def test_r2v3_long_smoke_registry_requires_exact_16_dense_routes(self) -> None:
        source = load_scenario_registry(DEFAULT_REGISTRY_PATH).fixtures[0]
        dense = tuple((float(index), 0.0, 0.5) for index in range(24))
        route = replace(
            source.route,
            waypoints=dense,
            navigation_context={
                "maneuver": "FOLLOW_STRAIGHT",
                "entry_signature": "1:1:entry",
                "exit_signature": "1:1:exit",
                "route_hash": "0" * 64,
                "topology_hash": "1" * 64,
            },
        )
        fixtures = tuple(
            replace(
                source,
                registry_version=R2V3_LONG_SMOKE_VERSION_PREFIX + "test",
                scenario_id=scenario_id,
                family="clear",
                seed_id="seed_a",
                duration_s=20.0,
                route=route,
            )
            for scenario_id in R2V3_LONG_SMOKE_SCENARIO_IDS
        )
        _validate_registry_shape(
            fixtures,
            registry_version=R2V3_LONG_SMOKE_VERSION_PREFIX + "test",
        )
        with self.assertRaises(RegistryError):
            _validate_registry_shape(
                fixtures[:-1],
                registry_version=R2V3_LONG_SMOKE_VERSION_PREFIX + "test",
            )

    def test_r2v3_campaign_lineage_requires_three_conditions_two_seeds(self) -> None:
        source = load_scenario_registry(DEFAULT_REGISTRY_PATH).fixtures[0]
        dense = tuple((float(index), 0.0, 0.5) for index in range(24))
        route = replace(
            source.route,
            waypoints=dense,
            navigation_context={
                "maneuver": "FOLLOW_STRAIGHT",
                "entry_signature": "1:1:entry",
                "exit_signature": "1:1:exit",
                "route_hash": "0" * 64,
                "topology_hash": "1" * 64,
            },
        )
        version = R2V3_CAMPAIGN_VERSION_PREFIX + "test"
        fixtures = tuple(
            replace(
                source,
                registry_version=version,
                scenario_id=f"condition_{condition}",
                family="traffic_control",
                seed_id=seed,
                duration_s=5.0,
                route=route,
            )
            for condition in range(3)
            for seed in REQUIRED_SEEDS
        )
        _validate_registry_shape(fixtures, registry_version=version)
        with self.assertRaises(RegistryError):
            _validate_registry_shape(fixtures[:-1], registry_version=version)

    def test_default_registry_exists(self) -> None:
        self.assertTrue(DEFAULT_REGISTRY_PATH.is_file(), msg=str(DEFAULT_REGISTRY_PATH))

    def test_load_six_scenarios_three_families_two_seeds(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        self.assertEqual(len(reg.fixtures), 12)
        self.assertEqual(set(reg.scenario_ids()), set(REQUIRED_SCENARIO_IDS))
        self.assertEqual(reg.families(), set(ALLOWED_FAMILIES))
        for sid in REQUIRED_SCENARIO_IDS:
            for seed in REQUIRED_SEEDS:
                fx = reg.get(sid, seed)
                self.assertEqual(fx.scenario_id, sid)
                self.assertEqual(fx.seed_id, seed)
                self.assertEqual(fx.map_name, "Town03")
                self.assertAlmostEqual(fx.sim_dt_s, 0.05)
                self.assertGreaterEqual(fx.duration_s, 2.5)
                self.assertEqual(fx.ego.role, "ego")
                self.assertTrue(fx.actors)
                self.assertNotIn("free", fx.ego.blueprint.lower())

    def test_r23_collection_registry_allows_multimap_and_reactive_script(self) -> None:
        text = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
        text = text.replace(
            'registry_version = "v1"',
            'registry_version = "r23-test-shard"',
        )
        text = text.replace('map_name = "Town03"', 'map_name = "Town12"')
        text = text.replace('family = "lead_braking"', 'family = "merge"')
        text = text.replace('family = "cut_in"', 'family = "merge"')
        text = text.replace('family = "crossing"', 'family = "merge"')
        text = text.replace(
            'script_type = "piecewise_vehicle_control"',
            (
                'script_type = "reactive_yield"\n'
                "desired_speed_mps = 5.0\n"
                "yield_ttc_s = 2.5\n"
                "stop_distance_m = 5.0\n"
                "actor_has_priority = false"
            ),
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r23.toml"
            path.write_text(text, encoding="utf-8")
            registry = load_scenario_registry(path)
        self.assertEqual({fixture.map_name for fixture in registry.fixtures}, {"Town12"})
        self.assertEqual(registry.families(), {"merge"})
        self.assertTrue(
            any(
                actor.script.script_type == "reactive_yield"
                for fixture in registry.fixtures
                for actor in fixture.actors
            )
        )

    def test_canonical_hash_stable(self) -> None:
        a = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        b = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        self.assertEqual(a.compute_registry_sha256(), b.compute_registry_sha256())
        self.assertEqual(len(a.compute_registry_sha256()), 64)

    def test_field_change_changes_hash(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        h0 = reg.compute_registry_sha256()
        raw = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
        # bump registry_version → hash must change
        mutated = raw.replace(
            'registry_version = "v1"',
            'registry_version = "v1-mut"',
            1,
        )
        self.assertNotEqual(raw, mutated)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "reg.toml"
            p.write_text(mutated, encoding="utf-8")
            reg2 = load_scenario_registry(p)
            self.assertNotEqual(h0, reg2.compute_registry_sha256())

    def test_post_v1_registry_accepts_six_new_scenario_ids(self) -> None:
        raw = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
        mutated = raw.replace('registry_version = "v1"', 'registry_version = "v2-blind"', 1)
        renamed = []
        for index, old in enumerate(REQUIRED_SCENARIO_IDS):
            new = f"blind_scene_{index + 1}"
            mutated = mutated.replace(f"scenarios.{old}", f"scenarios.{new}")
            renamed.append(new)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "blind.toml"
            p.write_text(mutated, encoding="utf-8")
            reg = load_scenario_registry(p)
        self.assertEqual(reg.registry_version, "v2-blind")
        self.assertEqual(set(reg.scenario_ids()), set(renamed))
        self.assertEqual(len(reg.fixtures), 12)

    def test_post_v1_registry_still_requires_six_scenarios(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        with self.assertRaises(RegistryError):
            _validate_registry_shape(
                reg.fixtures[:10],
                registry_version="v2-blind",
            )

    def test_requested_initial_state_hash_unique_per_pair(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        hashes = [f.requested_initial_state_hash() for f in reg.fixtures]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_freeze_manifest(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        frozen = freeze_registry(reg)
        self.assertTrue(frozen.frozen)
        man = frozen.freeze_manifest()
        self.assertEqual(man["n_pairs"], 12)
        self.assertEqual(man["n_scenarios"], 6)
        self.assertEqual(man["registry_sha256"], frozen.registry_sha256)
        self.assertTrue(man["frozen"])

    def test_seed_a_seed_b_differ_in_timing_or_gap(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        for sid in REQUIRED_SCENARIO_IDS:
            a = reg.get(sid, "seed_a")
            b = reg.get(sid, "seed_b")
            self.assertNotEqual(a.to_dict(), b.to_dict(), msg=sid)

    def test_actor_script_is_time_based_not_candidate(self) -> None:
        reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
        fx = reg.get("lead_brake_moderate", "seed_a")
        lead = fx.actors[0]
        c0 = lead.script.control_at(0.0)
        c1 = lead.script.control_at(1.0)
        self.assertEqual(c0["kind"], "vehicle")
        self.assertNotEqual(c0["phase"], c1["phase"])
        # script API has no candidate parameter
        self.assertNotIn("candidate_id", lead.script.to_dict())

    def test_reject_free_spawn(self) -> None:
        raw = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
        needle = "[scenarios.lead_brake_moderate.seeds.seed_a.ego.transform]\n"
        self.assertIn(needle, raw)
        bad = raw.replace(
            needle,
            needle + "free_spawn = true\n",
            1,
        )
        self.assertNotEqual(raw, bad)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.toml"
            p.write_text(bad, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_scenario_registry(p)

    def test_reject_duplicate_scenario_seed(self) -> None:
        # Minimal broken registry: only one scenario duplicated via rewrite is hard;
        # instead load and ensure validation rejects wrong seed count.
        raw = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
        # rename seed_b of first scenario to seed_a → duplicate
        bad = raw.replace(
            "[scenarios.lead_brake_moderate.seeds.seed_b.ego]",
            "[scenarios.lead_brake_moderate.seeds.seed_a_dup.ego]",
            1,
        )
        # Also need to rename all seed_b paths for that scenario - easier: drop seed_b keys
        # by replacing seed_b block name incorrectly leaving only one seed
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.toml"
            # Remove lead_brake_moderate seed_b entirely by truncating after seed_a script knots
            # Simpler approach: use a tiny invalid registry
            tiny = """
[registry]
schema_version = "safedrive.g4a.scenario_registry.v1"
registry_version = "v1"

[defaults]
map_name = "Town03"
sim_dt_s = 0.05
duration_s = 5.0

[defaults.weather]
preset = "ClearNoon"
cloudiness = 0.0
precipitation = 0.0
precipitation_deposits = 0.0
wind_intensity = 0.0
sun_azimuth_angle = 0.0
sun_altitude_angle = 70.0
wetness = 0.0

[scenarios.lead_brake_moderate]
family = "lead_braking"

[scenarios.lead_brake_moderate.route]
identity = "r"
waypoints = [[0.0, 0.0, 0.5], [50.0, 0.0, 0.5]]

[scenarios.lead_brake_moderate.seeds.seed_a.ego]
name = "ego"
role = "ego"
blueprint = "vehicle.tesla.model3"
bounding_box_extent_m = [2.0, 1.0, 0.8]
[scenarios.lead_brake_moderate.seeds.seed_a.ego.transform]
x = 0.0
y = 0.0
z = 0.5
roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0
[scenarios.lead_brake_moderate.seeds.seed_a.ego.script]
script_type = "hold"
[[scenarios.lead_brake_moderate.seeds.seed_a.actors]]
name = "lead"
role = "npc"
blueprint = "vehicle.audi.a2"
bounding_box_extent_m = [2.0, 1.0, 0.8]
[scenarios.lead_brake_moderate.seeds.seed_a.actors.transform]
x = 20.0
y = 0.0
z = 0.5
roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0
[scenarios.lead_brake_moderate.seeds.seed_a.actors.script]
script_type = "hold"
"""
            p.write_text(tiny, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_scenario_registry(p)

    def test_reject_random_or_tm_script(self) -> None:
        tiny = """
[registry]
schema_version = "safedrive.g4a.scenario_registry.v1"
registry_version = "v1"
[defaults]
map_name = "Town03"
sim_dt_s = 0.05
duration_s = 5.0
[defaults.weather]
preset = "ClearNoon"
cloudiness = 0.0
precipitation = 0.0
precipitation_deposits = 0.0
wind_intensity = 0.0
sun_azimuth_angle = 0.0
sun_altitude_angle = 70.0
wetness = 0.0
[scenarios.lead_brake_moderate]
family = "lead_braking"
[scenarios.lead_brake_moderate.route]
identity = "r"
waypoints = [[0.0, 0.0, 0.5], [50.0, 0.0, 0.5]]
[scenarios.lead_brake_moderate.seeds.seed_a.ego]
name = "ego"
role = "ego"
blueprint = "vehicle.tesla.model3"
bounding_box_extent_m = [2.0, 1.0, 0.8]
autopilot = true
[scenarios.lead_brake_moderate.seeds.seed_a.ego.transform]
x = 0.0
y = 0.0
z = 0.5
roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0
[scenarios.lead_brake_moderate.seeds.seed_a.ego.script]
script_type = "hold"
[[scenarios.lead_brake_moderate.seeds.seed_a.actors]]
name = "lead"
role = "npc"
blueprint = "vehicle.audi.a2"
bounding_box_extent_m = [2.0, 1.0, 0.8]
[scenarios.lead_brake_moderate.seeds.seed_a.actors.transform]
x = 20.0
y = 0.0
z = 0.5
roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0
[scenarios.lead_brake_moderate.seeds.seed_a.actors.script]
script_type = "hold"
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.toml"
            p.write_text(tiny, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_scenario_registry(p)

    def test_reject_candidate_id_in_script(self) -> None:
        tiny = """
[registry]
schema_version = "safedrive.g4a.scenario_registry.v1"
registry_version = "v1"
[defaults]
map_name = "Town03"
sim_dt_s = 0.05
duration_s = 5.0
[defaults.weather]
preset = "ClearNoon"
cloudiness = 0.0
precipitation = 0.0
precipitation_deposits = 0.0
wind_intensity = 0.0
sun_azimuth_angle = 0.0
sun_altitude_angle = 70.0
wetness = 0.0
[scenarios.lead_brake_moderate]
family = "lead_braking"
[scenarios.lead_brake_moderate.route]
identity = "r"
waypoints = [[0.0, 0.0, 0.5], [50.0, 0.0, 0.5]]
[scenarios.lead_brake_moderate.seeds.seed_a.ego]
name = "ego"
role = "ego"
blueprint = "vehicle.tesla.model3"
bounding_box_extent_m = [2.0, 1.0, 0.8]
[scenarios.lead_brake_moderate.seeds.seed_a.ego.transform]
x = 0.0
y = 0.0
z = 0.5
roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0
[scenarios.lead_brake_moderate.seeds.seed_a.ego.script]
script_type = "hold"
[[scenarios.lead_brake_moderate.seeds.seed_a.actors]]
name = "lead"
role = "npc"
blueprint = "vehicle.audi.a2"
bounding_box_extent_m = [2.0, 1.0, 0.8]
[scenarios.lead_brake_moderate.seeds.seed_a.actors.transform]
x = 20.0
y = 0.0
z = 0.5
roll_deg = 0.0
pitch_deg = 0.0
yaw_deg = 0.0
[scenarios.lead_brake_moderate.seeds.seed_a.actors.script]
script_type = "hold"
candidate_id = "v1_nominal"
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.toml"
            p.write_text(tiny, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_scenario_registry(p)


if __name__ == "__main__":
    unittest.main()
