from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.action_signal import action_signal_report  # noqa: E402
from driving_vla.world.contracts import (  # noqa: E402
    ActionBranchSample,
    SampleIdentity,
    content_hash,
)
from driving_vla.world.dataset import (  # noqa: E402
    ActionBranchDataset,
    ActionBranchDatasetV1,
    write_dataset,
)
from driving_vla.world.final_quality import validate_final_dataset  # noqa: E402


def _sample(index: int, *, reactive: bool = True) -> ActionBranchSample:
    identity = SampleIdentity(
        sample_id=f"sample-{index}",
        pair_id=f"pair-{index}",
        scenario_id=f"scenario-{index}",
        seed_id="seed_a",
        group_key=f"Town03|cut_in|lineage-{index}",
        family="cut_in",
        map_name="Town03",
        initial_state_hash="a" * 16,
        observation_hash="b" * 16,
        anchor_artifact_hash="c" * 16,
        model_hash="d" * 16,
        guard_hash="e" * 16,
        executor_hash="f" * 16,
        source_manifest_hash="1" * 16,
    )
    actor_future = np.zeros((2, 8, 10, 6), dtype=np.float32)
    actor_future[1, 0, :, 0] = 1.0
    sample = ActionBranchSample(
        identity=identity,
        ego_history=np.zeros((5, 11), dtype=np.float32),
        ego_history_mask=np.ones((5,), dtype=bool),
        actor_history=np.zeros((8, 5, 14), dtype=np.float32),
        actor_history_mask=np.zeros((8, 5), dtype=bool),
        road=np.zeros((3, 16, 6), dtype=np.float32),
        road_mask=np.zeros((3, 16), dtype=bool),
        candidates=np.zeros((2, 10, 8), dtype=np.float32),
        candidate_mask=np.ones((2,), dtype=bool),
        actor_future=actor_future,
        actor_future_mask=np.pad(
            np.ones((2, 1, 10), dtype=bool), ((0, 0), (0, 7), (0, 0))
        ),
        outcomes=np.zeros((2, 8), dtype=np.float32),
        outcome_mask=np.ones((2,), dtype=bool),
        rank_target=1.0,
        rank_mask=True,
        rank_weight=1.0,
        tie_target=False,
        comparable=True,
        unavailable_reasons=(None, None),
        audit={
            "reactive_actor_present": reactive,
            "namespace": "r3_final_head_formal",
            "r2_checkpoint_sha256": "a" * 64,
            "repeat_group": f"group-{index}",
            "aa_noise_identity": content_hash(
                {
                    "namespace": "r3_aa_noise_probe",
                    "repeat_group": f"group-{index}",
                    "candidate_id": "v3_nominal_progress",
                }
            ),
        },
    )
    sample.validate()
    return sample


class R3ActionSignalTest(unittest.TestCase):
    def test_action_signal_exceeds_aa_noise(self) -> None:
        report = action_signal_report(
            [_sample(0), _sample(1)],
            aa_p99=0.1,
        )
        self.assertEqual(report["reactive_count"], 2)
        self.assertEqual(report["action_sensitive_count"], 2)
        self.assertTrue(report["gate_reactive_fraction"])

    def test_v1_dataset_namespace_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            manifest = write_dataset(
                [_sample(0)],
                root,
                split_by_sample={"sample-0": "train"},
                quality_thresholds={"min_comparable": 1, "min_decisive": 1, "min_wins_per_slot": 1},
                namespace="r3_final_head_formal",
                r2_checkpoint_sha256="a" * 64,
            )
            self.assertEqual(
                manifest["schema_version"],
                "safedrive.action_branch_dataset_manifest.v1",
            )
            loaded = ActionBranchDataset(root)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.manifest["namespace"], "r3_final_head_formal")
            loaded_v1 = ActionBranchDatasetV1(root)
            self.assertEqual(len(loaded_v1), 1)
            report = validate_final_dataset(
                loaded,
                checkpoint_sha256="a" * 64,
            )
            self.assertFalse(report["all_hard_gates_pass"])
            self.assertFalse(report["hard_gates"]["completed_minimum"])


if __name__ == "__main__":
    unittest.main()
