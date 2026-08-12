"""Formal Spatial K2 promotion must bind a genuinely new blind registry."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

SPEC = importlib.util.spec_from_file_location(
    "r2x_promote_formal_checkpoint",
    ROOT / "scripts" / "r2x_promote_formal_checkpoint.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _offline() -> dict:
    return {
        "schema_version": "safedrive.r2x.offline_exec.v3",
        "head_status": "OK",
        "eligible_guard_ok_rate": 1.0,
        "eligible_spatial_sep_rate": 0.875,
        "eligible_proposal_valid_rate": 0.875,
        "learned_confidence_diagnostic": {
            "recall": 0.875,
            "specificity": 0.786,
        },
    }


def _card() -> dict:
    return {
        "new_blind_exam_registry_required": True,
        "r2k_pilot_allowed": False,
    }


class FormalPromotionGateTest(unittest.TestCase):
    def test_accepts_new_zero_overlap_blind_registry(self) -> None:
        audit = MODULE.evaluate_formal_inputs(
            offline=_offline(),
            card=_card(),
            train_pairs={("train_cut_in", "seed_a")},
            blind_pairs={("blind_cut_in", "seed_a")},
            registry_version="v2-blind",
        )
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["overlap"], [])

    def test_rejects_v1_even_when_pairs_do_not_overlap(self) -> None:
        audit = MODULE.evaluate_formal_inputs(
            offline=_offline(),
            card=_card(),
            train_pairs={("train_cut_in", "seed_a")},
            blind_pairs={("other", "seed_a")},
            registry_version="v1",
        )
        self.assertFalse(audit["ok"])
        self.assertFalse(audit["gates"]["new_blind_registry"])

    def test_rejects_training_pair_overlap(self) -> None:
        audit = MODULE.evaluate_formal_inputs(
            offline=_offline(),
            card=_card(),
            train_pairs={("same", "seed_a")},
            blind_pairs={("same", "seed_a")},
            registry_version="v2-blind",
        )
        self.assertFalse(audit["ok"])
        self.assertFalse(audit["gates"]["blind_pair_overlap_zero"])

    def test_rejects_old_offline_schema(self) -> None:
        offline = _offline()
        offline["schema_version"] = "safedrive.r2x.offline_exec.v2"
        audit = MODULE.evaluate_formal_inputs(
            offline=offline,
            card=_card(),
            train_pairs=set(),
            blind_pairs={("blind", "seed_a")},
            registry_version="v2-blind",
        )
        self.assertFalse(audit["ok"])


if __name__ == "__main__":
    unittest.main()
