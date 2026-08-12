from __future__ import annotations

import unittest

from driving_vla.evaluation.r2_world_ready_v3 import (
    build_calibration_manifest_v3,
)
from scripts.r2_v3_promote_formal import _validate_campaign_reports


class R2V3PromotionContractTest(unittest.TestCase):
    def test_teacher_reports_must_cover_exactly_360_frozen_slots(self) -> None:
        manifest = build_calibration_manifest_v3()
        rows = [
            {
                "slot_id": slot["slot_id"],
                "status": "COMPLETED",
            }
            for slot in manifest["slots"]
        ]
        hashes = _validate_campaign_reports(
            [
                {
                    "manifest_hash": manifest["manifest_hash"],
                    "mode": "teacher",
                    "rows": rows,
                }
            ],
            calibration=manifest,
        )
        self.assertEqual(len(hashes), 1)

        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, "exactly cover 360"):
            _validate_campaign_reports(
                [
                    {
                        "manifest_hash": manifest["manifest_hash"],
                        "mode": "teacher",
                        "rows": rows,
                    }
                ],
                calibration=manifest,
            )

    def test_learned_collection_cannot_seed_formal_dataset(self) -> None:
        manifest = build_calibration_manifest_v3()
        with self.assertRaisesRegex(ValueError, "teacher collection"):
            _validate_campaign_reports(
                [
                    {
                        "manifest_hash": manifest["manifest_hash"],
                        "mode": "learned",
                        "rows": [],
                    }
                ],
                calibration=manifest,
            )


if __name__ == "__main__":
    unittest.main()
