from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry" / "ros_ws" / "src" / "safedrive_carla_bridge"))

from safedrive_carla_bridge.doctor import (  # noqa: E402
    classify_carla_probe,
    classify_disk_probe,
    classify_gpu_probe,
)
from safedrive_carla_bridge.sync_contract import (  # noqa: E402
    FAIL,
    SyncConfig,
    build_deterministic_trace,
    compare_traces,
    run_contract_fault_injection,
    run_deterministic_smoke,
    validate_carla_settings,
)


class G005SyncContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SyncConfig()
        self.assertEqual(self.config.validate(), [])

    def test_fixed_step_and_substep_constraints(self) -> None:
        settings = SimpleNamespace(
            synchronous_mode=True,
            fixed_delta_seconds=0.05,
            substepping=True,
            max_substep_delta_time=0.01,
            max_substeps=5,
        )
        self.assertEqual(validate_carla_settings(settings, self.config), [])
        invalid = SyncConfig(max_substeps=4)
        self.assertTrue(invalid.validate())

    def test_same_seed_is_reproducible(self) -> None:
        first = build_deterministic_trace(seed=42, steps=8, config=self.config)
        second = build_deterministic_trace(seed=42, steps=8, config=self.config)
        result = compare_traces(first, second, tolerance_seconds=1e-6)
        self.assertTrue(result["pass"], result)

    def test_contract_faults_are_rejected(self) -> None:
        results = run_contract_fault_injection(self.config)
        observed = {item["id"]: item["observed"]["code"] for item in results}
        self.assertEqual(
            observed,
            {
                "duplicate_tick": "duplicate_tick",
                "missing_frame": "missing_frame",
                "stale_message": "stale_message",
                "multiple_tick_masters": "multiple_tick_masters",
            },
        )

    def test_checkpoint_resume_matches_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoint = directory / "checkpoint.json"
            resumed_path = directory / "resumed.json"
            status, trace = run_deterministic_smoke(
                seed=9,
                steps=9,
                config=self.config,
                checkpoint_path=checkpoint,
                trace_path=resumed_path,
                interrupt_after=3,
            )
            self.assertEqual(status, "INTERRUPTED")
            self.assertIsNone(trace)
            status, resumed = run_deterministic_smoke(
                seed=9,
                steps=9,
                config=self.config,
                checkpoint_path=checkpoint,
                trace_path=resumed_path,
                resume=True,
            )
            self.assertEqual(status, "COMPLETED")
            expected = build_deterministic_trace(seed=9, steps=9, config=self.config)
            self.assertTrue(compare_traces(expected, resumed or {}, tolerance_seconds=1e-6)["pass"])

    def test_environment_fault_classifiers(self) -> None:
        self.assertEqual(
            classify_carla_probe(socket_open=False, api_available=False, handshake_ok=False).code,
            "carla_not_started",
        )
        self.assertEqual(
            classify_carla_probe(socket_open=True, api_available=True, handshake_ok=False).code,
            "port_conflict_or_non_carla_service",
        )
        self.assertEqual(
            classify_carla_probe(
                socket_open=True,
                api_available=True,
                handshake_ok=True,
                client_version="0.9.15",
                server_version="0.9.16",
            ).code,
            "carla_version_mismatch",
        )
        self.assertEqual(classify_gpu_probe(False).code, "gpu_not_visible")
        self.assertEqual(classify_disk_probe(1.0, 20.0).code, "low_disk_space")
        self.assertEqual(classify_gpu_probe(False).status, FAIL)


if __name__ == "__main__":
    unittest.main()
