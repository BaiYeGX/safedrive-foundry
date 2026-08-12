"""Regression tests for the R2 V4 collection/formal transition contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.r2_v4_freeze_pending_blind import freeze
from scripts.r2_v4_make_collection_checkpoint import make_checkpoint
from safedrive_foundry.driving_vla.model.checkpoint_contract import (
    CheckpointContractError,
    validate_checkpoint_for_use,
)


ROOT = Path(__file__).resolve().parents[2]
TOKEN_SAMPLE = ROOT / "docs/runtime-evidence/r2x-feature-probe-rawcheck/anchors/real_b8e4124a7d33/raw_tokens_fp16.npy"


class R2V4ContractClosureTest(unittest.TestCase):
    def test_neutral_collection_checkpoint_is_not_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = make_checkpoint(TOKEN_SAMPLE, Path(tmp) / "neutral")
            checkpoint = Path(result["checkpoint"])
            self.assertEqual(result["token_dim"], 896)
            self.assertEqual(
                validate_checkpoint_for_use(checkpoint, "collection_anchor")["status"],
                "HEAD_TRAINED_NOT_FORMAL",
            )
            with self.assertRaises(CheckpointContractError):
                validate_checkpoint_for_use(checkpoint, "r2v4_formal")
            with self.assertRaises(CheckpointContractError):
                validate_checkpoint_for_use(checkpoint, "r2v4_blind_audit")

    def test_pending_blind_requires_exactly_one_locked_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = make_checkpoint(TOKEN_SAMPLE, Path(tmp) / "neutral")
            checkpoint = Path(result["checkpoint"])
            formal = checkpoint.parent / "formal.json"
            formal.write_text(
                json.dumps(
                    {
                        "locked_test_opened_once": True,
                        "test_evaluation_count": 1,
                        "all_hard_gates_pass": True,
                        "reports": [
                            {
                                "checkpoint_sha256": result["checkpoint_sha256"],
                                "test": {"map": "Town13"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pending = freeze(checkpoint, formal)
            self.assertEqual(pending["status"], "R2V4_FROZEN_PENDING_BLIND")
            self.assertEqual(
                validate_checkpoint_for_use(checkpoint, "r2v4_blind_audit")["status"],
                "R2V4_FROZEN_PENDING_BLIND",
            )
            with self.assertRaises(CheckpointContractError):
                validate_checkpoint_for_use(checkpoint, "r3_final_head_formal")


if __name__ == "__main__":
    unittest.main()
