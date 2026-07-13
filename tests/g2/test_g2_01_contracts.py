"""G2-01 contract schema, hash, and serialization tests."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from safety_kernel import (  # noqa: E402
    SCHEMA_VERSION,
    PolicyCandidate,
    PolicyCandidateSet,
    TrajectoryPoint,
    contracts_schema_hash,
    load_safety_config,
    config_sha256,
)
from safety_kernel.contracts import (  # noqa: E402
    CandidateSource,
    candidate_set_from_dict,
    candidate_set_to_dict,
    validate_candidate_set_schema,
)
from safety_kernel.contracts.serialize import ContractSchemaError  # noqa: E402


def _point(t: float, x: float = 0.0, v: float = 5.0) -> TrajectoryPoint:
    return TrajectoryPoint(t=t, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0)


def _candidate(**kwargs) -> PolicyCandidate:
    points = kwargs.pop("points", tuple(_point(0.2 * i, x=1.0 * i) for i in range(16)))
    base = dict(
        candidate_id="c1",
        source=CandidateSource.CLASSIC,
        generated_time_s=1.0,
        valid_until_s=1.2,
        probability=1.0,
        points=points,
    )
    base.update(kwargs)
    return PolicyCandidate(**base)


def _set(cands=None) -> PolicyCandidateSet:
    return PolicyCandidateSet(
        run_id="run-1",
        frame_id="f-1",
        scenario_id="sc-1",
        model_id="classic",
        carla_frame=10,
        simulation_time_s=1.0,
        wall_time_s=100.0,
        candidates=tuple(cands or [_candidate()]),
        schema_version=SCHEMA_VERSION,
    )


class G201ContractsTests(unittest.TestCase):
    def test_schema_version_and_hash_stable(self) -> None:
        h1 = contracts_schema_hash()
        h2 = contracts_schema_hash()
        self.assertEqual(len(h1), 64)
        self.assertEqual(h1, h2)
        self.assertEqual(SCHEMA_VERSION, "safedrive.safety.contracts.v1")

    def test_config_versioned_and_hashed(self) -> None:
        cfg = load_safety_config()
        digest = config_sha256(cfg.raw_toml)
        self.assertEqual(len(digest), 64)
        self.assertIn("safety", cfg.schema_version)
        self.assertAlmostEqual(cfg.control_period_s, 0.02)

    def test_roundtrip_candidate_set(self) -> None:
        original = _set()
        payload = candidate_set_to_dict(original)
        restored = candidate_set_from_dict(payload)
        self.assertEqual(restored.run_id, original.run_id)
        self.assertEqual(len(restored.candidates), 1)
        self.assertEqual(restored.candidates[0].candidate_id, "c1")
        self.assertEqual(restored.schema_version, SCHEMA_VERSION)

    def test_missing_fields_rejected(self) -> None:
        payload = candidate_set_to_dict(_set())
        del payload["run_id"]
        with self.assertRaises(ContractSchemaError):
            candidate_set_from_dict(payload)

    def test_wrong_schema_version_rejected(self) -> None:
        payload = candidate_set_to_dict(_set())
        payload["schema_version"] = "not.a.schema"
        with self.assertRaises(ContractSchemaError):
            candidate_set_from_dict(payload)

    def test_nan_in_point_rejected_at_schema(self) -> None:
        payload = candidate_set_to_dict(_set())
        payload["candidates"][0]["points"][0]["x"] = float("nan")
        with self.assertRaises(ContractSchemaError):
            candidate_set_from_dict(payload)

    def test_validate_candidate_set_schema_helper(self) -> None:
        ok = validate_candidate_set_schema(candidate_set_to_dict(_set()))
        self.assertEqual(ok, [])
        bad = validate_candidate_set_schema({"schema_version": SCHEMA_VERSION})
        self.assertTrue(bad)


if __name__ == "__main__":
    unittest.main()
