from __future__ import annotations

import sys
import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from runtime import ContractViolation, RuntimeIdentityFactory, StatusJsonAdapter, load_runtime_profiles  # noqa: E402
from runtime.profiles import RuntimeProfile  # noqa: E402


class G101ContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = {
            "experiment_id": "g1-contracts",
            "scenario_id": "Town01.route.001",
            "attempt_id": 2,
            "server_epoch": "carla-0.9.16-epoch-a",
            "producer_version": "0.1.0",
        }

    def test_identity_is_deterministic_and_cross_run_isolated(self) -> None:
        first = RuntimeIdentityFactory.create(self.inputs)
        second = RuntimeIdentityFactory.create(dict(self.inputs))
        self.assertEqual(first, second)
        code = (
            "import json; from runtime import RuntimeIdentityFactory; "
            f"print(RuntimeIdentityFactory.create(json.loads({json.dumps(self.inputs)!r})).run_id)"
        )
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "safedrive_foundry")}
        other_process = subprocess.check_output(
            [sys.executable, "-c", code], text=True, env=environment
        ).strip()
        self.assertEqual(first.run_id, other_process)
        changed = RuntimeIdentityFactory.create({**self.inputs, "attempt_id": 3})
        self.assertNotEqual(first.run_id, changed.run_id)
        self.assertNotIn("episode_id", first.to_dict())

    def test_profiles_are_carla_legal(self) -> None:
        profiles = load_runtime_profiles(ROOT / "safedrive_foundry" / "config" / "runtime_profiles.toml")
        self.assertEqual(profiles["throughput_20hz"].fixed_delta_seconds, 0.05)
        self.assertEqual(profiles["control_50hz"].control_period_ms, 20)
        self.assertEqual(profiles["control_50hz"].validate(), [])

    def test_invalid_profile_fails_fast(self) -> None:
        invalid = RuntimeProfile("control_50hz", 0.02, 0.01, 1, 20)
        self.assertTrue(invalid.validate())
        with self.assertRaises(ValueError):
            invalid.assert_valid()

    def test_legacy_json_adapter_validates_schema(self) -> None:
        identity = RuntimeIdentityFactory.create(self.inputs)
        payload = {
            "schema": "safedrive.carla.status.v2",
            "episode_id": "compat-only",
            "carla_frame": 91,
            "simulation_seconds": 4.55,
            "publisher_wall_time": 1720000000.0,
        }
        frame = StatusJsonAdapter().parse(payload, identity)
        self.assertEqual(frame.identity.run_id, identity.run_id)
        self.assertEqual(frame.carla_frame, 91)
        with self.assertRaises(ContractViolation):
            StatusJsonAdapter().parse({**payload, "schema": "free-form"}, identity)


if __name__ == "__main__":
    unittest.main()
