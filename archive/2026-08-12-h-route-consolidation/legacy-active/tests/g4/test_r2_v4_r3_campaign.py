from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "scripts"))

from r2_v4_campaign import build_manifest  # noqa: E402
from r2_v4_audit_manifest import build_audit_manifest  # noqa: E402
from r2_v4_core_manifest import build_core_manifest  # noqa: E402
from r3_v1_campaign import build_campaign  # noqa: E402
from r2_v4_train_heads import (  # noqa: E402
    TOKEN_MODES,
    V4Dataset,
    _normalization,
    load_rows,
    train,
)
from r2_v4_freeze_smoke_manifests import build_manifest as build_smoke_manifest  # noqa: E402
from r2_v4_formal_eval import evaluate as evaluate_v4_formal  # noqa: E402
from driving_vla.evaluation.scenario_registry import R2V3_LONG_SMOKE_SCENARIO_IDS  # noqa: E402
from driving_vla.model.semantic_mode_heads_v4 import AUX_DIM  # noqa: E402
from driving_vla.model.v4_token_features import TOTAL_TOKEN_COUNT  # noqa: E402
from driving_vla.evaluation.actor_future_collector import (  # noqa: E402
    capture_observable_scene_t0,
)


class CampaignManifestTest(unittest.TestCase):
    def test_learned_smoke_manifests_are_frozen_16_and_32_pairs(self) -> None:
        fixtures = []
        for seed in ("seed_a", "seed_b"):
            for scenario_id in R2V3_LONG_SMOKE_SCENARIO_IDS:
                fixtures.append(
                    SimpleNamespace(
                        scenario_id=scenario_id,
                        seed_id=seed,
                        map_name="Town03",
                        family="clear",
                        route=SimpleNamespace(navigation_context={"maneuver": "FOLLOW_STRAIGHT"}),
                    )
                )
        smoke16 = build_smoke_manifest(fixtures, expected_cases=16, source_registry_hashes=["a" * 64])
        smoke32 = build_smoke_manifest(fixtures, expected_cases=32, source_registry_hashes=["a" * 64])
        self.assertEqual(len(smoke16["cases"]), 16)
        self.assertFalse(any(row["unseen_seed_or_route"] for row in smoke16["cases"]))
        self.assertEqual(len(smoke32["cases"]), 32)
        self.assertTrue(all(row["unseen_seed_or_route"] for row in smoke32["cases"]))
        self.assertEqual(len({(row["scenario_id"], row["seed_id"]) for row in smoke32["cases"]}), 32)

    def test_v4_observable_scene_can_exclude_offline_family_label(self) -> None:
        def actor(x: float, role: str) -> SimpleNamespace:
            vehicle = SimpleNamespace(
                get_transform=lambda: SimpleNamespace(
                    location=SimpleNamespace(x=x, y=0.0, z=0.0),
                    rotation=SimpleNamespace(yaw=0.0),
                ),
                get_velocity=lambda: SimpleNamespace(x=1.0, y=0.0, z=0.0),
                get_angular_velocity=lambda: SimpleNamespace(z=0.0),
                get_acceleration=lambda: SimpleNamespace(x=0.0, y=0.0),
                bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0)),
            )
            return SimpleNamespace(
                actor=vehicle,
                name=role,
                role=role,
                blueprint="vehicle.test",
            )

        scene = capture_observable_scene_t0(
            scenario_id="v4-scene",
            seed_id="seed_a",
            spawned_actors=[actor(0.0, "ego"), actor(10.0, "npc")],
            route_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
            map_name="Town03",
            family=None,
            simulation_time_s=0.0,
            frame=1,
        )
        self.assertNotIn("family", scene)
        historical = capture_observable_scene_t0(
            scenario_id="v3-scene",
            seed_id="seed_a",
            spawned_actors=[actor(0.0, "ego")],
            route_waypoints=[],
            map_name="Town03",
            family="clear",
            simulation_time_s=0.0,
            frame=1,
        )
        self.assertEqual(historical["family"], "clear")

    def test_r2_v4_counts_map_ood_and_pilot(self) -> None:
        manifest = build_manifest()
        self.assertEqual(len(manifest["lineages"]), 84)
        self.assertEqual(len(manifest["slots"]), 1008)
        self.assertTrue(
            all(
                len(row.get("expected_maneuver_params") or ()) == 6
                for row in manifest["slots"]
            )
        )
        self.assertEqual(len(manifest["pilot_lineages"]), 12)
        self.assertEqual(
            {row["map_name"] for row in manifest["lineages"] if row["split"] == "test"},
            {"Town13"},
        )
        expected_kinds = {"NONE", "TEMPORAL_YIELD", "SPATIAL_AVOID", "SPATIAL_OVERTAKE"}
        expected_sides = {"NONE", "LEFT", "RIGHT"}
        for split in ("train", "val", "test"):
            rows = [row for row in manifest["slots"] if row["split"] == split]
            self.assertTrue(expected_kinds.issubset({row["expected_kind"] for row in rows}))
            self.assertTrue(expected_sides.issubset({row["expected_side"] for row in rows}))
            self.assertEqual({bool(row["expected_available"]) for row in rows}, {False, True})
            self.assertTrue(
                set(
                    (
                        "FOLLOW_STRAIGHT",
                        "FOLLOW_CURVE_LEFT",
                        "FOLLOW_CURVE_RIGHT",
                        "JUNCTION_STRAIGHT",
                        "TURN_LEFT",
                        "TURN_RIGHT",
                        "ROUTE_CHANGE_LEFT",
                        "ROUTE_CHANGE_RIGHT",
                    )
                ).issubset({row["route_maneuver"] for row in rows})
            )
        groups = {}
        for row in manifest["slots"]:
            groups.setdefault(row["root_group"], set()).add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in groups.values()))

    def test_r2_v4_blind_and_core_manifests_are_frozen_before_outcomes(self) -> None:
        audit = build_audit_manifest(build_manifest())
        self.assertEqual(len(audit["fixtures"]), 84)
        self.assertEqual(len(audit["pairs"]), 252)
        self.assertTrue(audit["locked_test"])
        core = build_core_manifest()
        self.assertEqual(len(core["pairs"]), 12)
        self.assertEqual({row["family"] for row in core["pairs"]}, {"lead_braking", "cut_in", "crossing", "merge", "obstruction", "clear"})

    def test_r3_v1_counts_and_checkpoint_binding(self) -> None:
        campaign = build_campaign("a" * 64)
        self.assertEqual(len(campaign["lineages"]), 168)
        self.assertEqual(len(campaign["slots"]), 2016)
        self.assertEqual(len(campaign["reserve_lineages"]), 42)
        self.assertEqual(len(campaign["reserve_slots"]), 504)
        self.assertEqual(len(campaign["development_slots"]), 512)
        self.assertNotIn("test", {str(row["split"]) for row in campaign["development_slots"]})
        self.assertEqual(len(campaign["noise_probe"]), 168)
        for split in ("train", "val", "test"):
            rows = [row for row in campaign["lineages"] if row["split"] == split]
            self.assertEqual({row["family"] for row in rows}, set(campaign["families"]))
            self.assertEqual(
                {row["actor_controller_kind"] for row in campaign["slots"] if row["split"] == split},
                {"fixed", "reactive"},
            )
        self.assertTrue(all(str(row.get("repeat_group") or "") for row in campaign["slots"]))
        self.assertTrue(all(len(str(row.get("aa_noise_identity") or "")) == 64 for row in campaign["slots"]))
        self.assertEqual(
            {row["repeat_group"] for row in campaign["noise_probe"]},
            {row["lineage_id"] for row in campaign["lineages"]},
        )
        self.assertEqual(
            {row["family"] for row in campaign["reserve_lineages"]},
            {"lead_braking", "cut_in", "crossing", "merge", "obstruction", "clear"},
        )
        self.assertTrue(
            {row["map_name"] for row in campaign["reserve_lineages"]}
            >= {"Town03", "Town04", "Town05", "Town06", "Town10HD", "Town12"}
        )
        self.assertEqual(campaign["r2_checkpoint_sha256"], "a" * 64)
        self.assertEqual(
            {row["map_name"] for row in campaign["lineages"] if row["split"] == "test"},
            {"Town13"},
        )


class V4TrainingSmokeTest(unittest.TestCase):
    def test_locked_test_rejects_multiple_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "exactly once"):
                evaluate_v4_formal(
                    Path(tmp) / "sealed.jsonl",
                    [Path(tmp) / "candidate_a.pt", Path(tmp) / "candidate_b.pt"],
                    "cpu",
                    evaluate_test=True,
                )

    def test_parameter_supervision_cannot_silently_default_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "raw.npy"
            np.save(raw_path, np.zeros((1, TOTAL_TOKEN_COUNT, 8), dtype=np.float32))
            row = {
                "row_id": "missing-params",
                "lineage_id": "lineage-missing-params",
                "split": "train",
                "v4_raw_tensor_path": str(raw_path),
                "v4_token_raw_content_hash": hashlib.sha256(
                    np.zeros((1, TOTAL_TOKEN_COUNT, 8), dtype=np.float16).tobytes()
                ).hexdigest(),
                "v4_aux": [0.0] * AUX_DIM,
                "target_kind": "NONE",
                "target_side": "NONE",
                "available": False,
            }
            path = root / "missing.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "maneuver_params supervision"):
                load_rows(path)

    def test_synthetic_overfit_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.RandomState(9)
            records = []
            for index in range(12):
                raw_path = root / f"raw_{index}.npy"
                tokens = rng.randn(1, TOTAL_TOKEN_COUNT, 8).astype(np.float32)
                np.save(raw_path, tokens)
                kind = (index % 4)
                records.append(
                    {
                        "row_id": f"row-{index}",
                        "lineage_id": f"lineage-{index}",
                        "split": "train" if index < 8 else "val",
                "v4_raw_tensor_path": str(raw_path),
                "v4_token_raw_content_hash": hashlib.sha256(
                    tokens.astype(np.float16).tobytes()
                ).hexdigest(),
                        "v4_aux": rng.randn(AUX_DIM).tolist(),
                        "target_kind": ("NONE", "SPATIAL_AVOID", "SPATIAL_OVERTAKE", "TEMPORAL_YIELD")[kind],
                        "target_side": ("NONE", "LEFT", "RIGHT")[index % 3],
                        "available": kind != 0,
                        "maneuver_params": [0.5] * 6,
                    }
                )
            data = root / "data.jsonl"
            data.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            loaded = load_rows(data)
            self.assertEqual(len(loaded), 12)
            mean, std = _normalization([row for row in loaded if row.split == "train"])
            views = {
                mode: V4Dataset(loaded[:1], mean, std, token_mode=mode)[0]
                for mode in TOKEN_MODES
            }
            self.assertTrue(torch.equal(views["structured-only"][0], torch.zeros_like(views["structured-only"][0])))
            self.assertTrue(torch.allclose(views["mean64"][0], views["mean64"][0][0:1].expand_as(views["mean64"][0])))
            self.assertTrue(torch.equal(views["token-aware"][1][56:168], torch.zeros(112)))
            self.assertFalse(torch.equal(views["token-aware+history"][1][56:168], torch.zeros(112)))
            report = train(
                Namespace(
                    data=str(data),
                    output_root=str(root / "out"),
                    run_name="smoke",
                    steps=20,
                    batch_size=4,
                    learning_rate=3e-4,
                    weight_decay=1e-4,
                    dropout=0.2,
                    eval_every=10,
                    patience_evals=2,
                    seed=3407,
                    device="cpu",
                    overfit_32=False,
                    evaluate_test=False,
                )
            )
            self.assertEqual(report["schema_version"], "safedrive.k2.v4.semantic_head_checkpoint.v1")
            self.assertTrue(Path(report["checkpoint"]).is_file())
            self.assertGreater(report["parameter_count"], 300_000)


if __name__ == "__main__":
    unittest.main()
