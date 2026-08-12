"""Tests for retiring the observed v4 exam into development data."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.development_split import (  # noqa: E402
    REQUIRED_DEVELOPMENT_FAMILIES,
    assign_balanced_episode_split,
)


class DevelopmentSplitTest(unittest.TestCase):
    def test_episode_grouping_and_old_exam_retirement(self) -> None:
        rows = []
        for family in REQUIRED_DEVELOPMENT_FAMILIES:
            for episode_index in range(3):
                for frame in range(2):
                    rows.append(
                        {
                            "sample_id": f"{family}-{episode_index}-{frame}",
                            "episode_id": f"{family}-episode-{episode_index}",
                            "scenario_family": family,
                            "split_id": "holdout_exam",
                            "is_r2_exam_fixture": True,
                            "alternative_available": episode_index == 0,
                            "driving_feature": [float(frame), float(episode_index)],
                        }
                    )
        split, manifest = assign_balanced_episode_split(rows)
        self.assertTrue(manifest["new_blind_exam_required"])
        self.assertEqual(manifest["n_val_episodes"], len(REQUIRED_DEVELOPMENT_FAMILIES))
        by_episode = {}
        for row in split:
            by_episode.setdefault(row["episode_id"], set()).add(row["split_id"])
            self.assertFalse(row["is_r2_exam_fixture"])
            self.assertTrue(row["was_r2_v4_exam_fixture"])
        self.assertTrue(all(len(values) == 1 for values in by_episode.values()))
        self.assertFalse(
            {
                row["episode_id"] for row in split if row["split_id"] == "train"
            }
            & {
                row["episode_id"] for row in split if row["split_id"] == "val"
            }
        )


if __name__ == "__main__":
    unittest.main()
