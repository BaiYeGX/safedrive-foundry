"""Offline boundary tests for the H6 VLA75 v2 contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h6.acceptance import evaluate_vla75_gate  # noqa: E402
from data_pipeline.h6.config import (  # noqa: E402
    H6_VLA75_FORMAL_LINEAGES,
    h6_vla75_config_sha256,
)
from data_pipeline.h6.matrix import load_h6_vla75_matrix  # noqa: E402


def _decision(tick: int, *, raw_ok: bool = True, applied: str = "vla") -> dict:
    vla_score, expert_score = (2.0, 1.0) if raw_ok else (1.0, 2.0)
    applied_id = f"frame:{applied}" if applied in {"vla", "expert"} else None
    return {
        "tick": tick,
        "world_score": {
            "raw_preference_order": ["frame:vla", "frame:expert"],
            "predictions": [
                {
                    "candidate_key": "frame:vla",
                    "deployment_score": vla_score,
                    "preference_utility": vla_score,
                    "trust_probability": 0.95,
                    "unsafe_probability": 0.01,
                },
                {
                    "candidate_key": "frame:expert",
                    "deployment_score": expert_score,
                    "preference_utility": expert_score,
                    "trust_probability": 0.95,
                    "unsafe_probability": 0.01,
                },
            ],
            "trust_threshold": 0.5,
            "risk_ceiling": 0.2,
            "model_hash": "model",
            "feature_schema": "features",
        },
        "candidates": {
            "frame:vla": {"provenance": {"source": "vla"}},
            "frame:expert": {"provenance": {"source": "classic"}},
        },
        "selected_candidate_id": applied_id,
        "selected_candidate_source": applied,
        "safety_executed_candidate_id": applied_id,
        "safety_executed_source": applied,
        "applied_candidate_id": applied_id,
        "applied_source": applied,
        "applied_candidate_source": applied,
        "applied_mode": "TRACK_APPROVED" if applied in {"vla", "expert"} else "MINIMAL_RISK_BRAKE",
        "candidate_hashes": {"frame:vla": "v", "frame:expert": "e"},
        "model_hash": "model",
        "feature_schema": "features",
        "worktree_hash": "worktree",
        "scorer_latency_ms": 1.0,
        "repair": None,
    }


def _runs(*, raw_high: int = 600, applied_vla: int = 450, unsafe: bool = False):
    config = h6_vla75_config_sha256("a")
    target_decisions = [
        _decision(
            tick,
            raw_ok=tick < raw_high,
            applied="vla" if tick < applied_vla else "expert",
        )
        for tick in range(600)
    ]
    target = {
        "schema_version": "safedrive.h6.vla75.run.v2",
        "pair_id": "pair-a",
        "arm": "on",
        "ok": True,
        "ticks_executed": 600,
        "decisions": target_decisions,
        "scenario": {"map_name": "Town01", "family": "free_flow", "weather": "ClearNoon"},
        "config_sha256": config,
        "manifest_kind": "h6_vla75_fresh_lineage_a",
        "physical_sha256": "physical",
        "reset_comparison": {"comparable": True},
        "route_progress_m": 10.0,
        "collision_count": int(unsafe),
        "red_light_violation": False,
        "off_corridor_duration_s": 0.0,
        "spectator_follow_updates": 2,
        "spectator_follow_error": None,
    }
    baseline = {
        "schema_version": "safedrive.h6.vla75.run.v2",
        "pair_id": "pair-a",
        "arm": "off",
        "ok": True,
        "ticks_executed": 600,
        "decisions": [
            {
                "tick": tick,
                "applied_candidate_id": "frame:expert",
                "applied_source": "expert",
                "applied_mode": "TRACK_APPROVED",
                "candidates": {"frame:expert": {"provenance": {"source": "classic"}}},
            }
            for tick in range(600)
        ],
        "scenario": target["scenario"],
        "config_sha256": config,
        "manifest_kind": target["manifest_kind"],
        "physical_sha256": "physical",
        "route_progress_m": 10.0,
        "spectator_follow_updates": 2,
        "spectator_follow_error": None,
    }
    return [target, baseline]


class VLA75AcceptanceBoundaryTest(unittest.TestCase):
    def test_exact_600_tick_boundaries(self):
        passed = evaluate_vla75_gate(_runs())
        self.assertTrue(passed["passed"], passed)
        self.assertEqual(passed["coverage"]["raw_world_high_ticks"], 600)

        raw_boundary = evaluate_vla75_gate(_runs(raw_high=540))
        self.assertTrue(raw_boundary["passed"], raw_boundary)
        self.assertFalse(evaluate_vla75_gate(_runs(raw_high=539))["passed"])
        self.assertIn("world_vla_preference", evaluate_vla75_gate(_runs(raw_high=539))["failures"])

        self.assertTrue(evaluate_vla75_gate(_runs(applied_vla=450))["passed"])
        below = evaluate_vla75_gate(_runs(applied_vla=449))
        self.assertFalse(below["passed"])
        self.assertIn("actual_vla_coverage", below["failures"])
        self.assertIn("classic_mrm_share", below["failures"])

    def test_target_only_unsafe_and_missing_pair_are_hard_failures(self):
        result = evaluate_vla75_gate(_runs(unsafe=True))
        self.assertFalse(result["passed"])
        self.assertIn("target_only_unsafe", result["failures"])

        missing = evaluate_vla75_gate(_runs()[:1])
        self.assertFalse(missing["passed"])
        self.assertIn("missing_paired_baseline", missing["failures"])

    def test_lineages_have_fixed_disjoint_cardinality(self):
        self.assertEqual(set(H6_VLA75_FORMAL_LINEAGES), {"a", "b", "c"})
        pairs = set()
        seeds = set()
        for lineage, lineage_seeds in H6_VLA75_FORMAL_LINEAGES.items():
            self.assertEqual(len(load_h6_vla75_matrix(lineage, False)), 12)
            self.assertEqual(len(load_h6_vla75_matrix(lineage, True)), 108)
            self.assertTrue(pairs.isdisjoint({row.pair_id for row in load_h6_vla75_matrix(lineage, True)}))
            self.assertTrue(seeds.isdisjoint(lineage_seeds))
            pairs.update(row.pair_id for row in load_h6_vla75_matrix(lineage, True))
            seeds.update(lineage_seeds)


if __name__ == "__main__":
    unittest.main()
