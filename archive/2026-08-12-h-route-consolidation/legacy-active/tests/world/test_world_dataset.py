from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from driving_vla.world.dataset import (
    ActionBranchDataset,
    assign_group_splits,
    quality_report,
    write_dataset,
)

from .helpers import make_sample


class WorldDatasetTest(unittest.TestCase):
    def test_group_split_has_no_lineage_overlap(self) -> None:
        samples = [make_sample(i, winner=i % 2) for i in range(8)]
        splits = assign_group_splits(samples)
        group_splits: dict[str, set[str]] = {}
        for sample in samples:
            group_splits.setdefault(sample.identity.group_key, set()).add(
                splits[sample.identity.sample_id]
            )
        self.assertTrue(all(len(values) == 1 for values in group_splits.values()))

    def test_immutable_dataset_roundtrip(self) -> None:
        samples = [make_sample(i, winner=i % 2) for i in range(6)]
        splits = {sample.identity.sample_id: "train" for sample in samples}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            manifest = write_dataset(
                samples,
                root,
                split_by_sample=splits,
                shard_size=2,
                quality_thresholds={
                    "min_comparable": 6,
                    "min_decisive": 6,
                    "min_wins_per_slot": 3,
                    "min_future_coverage": 1.0,
                },
            )
            self.assertEqual(manifest["sample_count"], 6)
            dataset = ActionBranchDataset(root)
            self.assertEqual(len(dataset), 6)
            self.assertEqual(dataset[2].sample.identity.sample_id, "sample-2")
            self.assertEqual(len(list(dataset.iter_split("train"))), 6)
            with self.assertRaisesRegex(Exception, "already exists"):
                write_dataset(samples, root, split_by_sample=splits)

    def test_quality_gate_reports_weak_signal(self) -> None:
        samples = [make_sample(i, tie=True) for i in range(4)]
        splits = {sample.identity.sample_id: "test" for sample in samples}
        report = quality_report(samples, splits)
        self.assertFalse(report["all_hard_gates_pass"])
        self.assertEqual(report["decisive_count"], 0)
        self.assertEqual(report["split_quality"]["test"]["samples"], 4)
        self.assertEqual(report["split_quality"]["test"]["dual"], 4)
