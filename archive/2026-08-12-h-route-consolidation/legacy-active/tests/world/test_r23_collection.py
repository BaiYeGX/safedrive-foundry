from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from driving_vla.evaluation.observable_history import (
    ObservableHistoryError,
    ObservableHistoryRecorder,
    merge_history_into_scene,
)
from driving_vla.evaluation.r23_campaign import (
    build_phase_manifest,
    default_lineage_bank,
    default_lineage_bank_v2,
    load_lineage_bank,
)
from driving_vla.evaluation.r23_collection import (
    CAMPAIGN_CHECKPOINT_SCHEMA,
    R23CollectionError,
    R23CollectionSampleV1,
    content_hash,
    validate_completed_joint_artifacts,
    validate_campaign_manifest,
    validate_checkpoint,
)
from driving_vla.evaluation.reactive_actor import (
    ReactiveActorContractError,
    deterministic_reactive_control,
)
from driving_vla.model.checkpoint_contract import (
    CheckpointContractError,
    validate_checkpoint_for_use,
)


class FakeActor:
    def __init__(self, x: float, speed: float = 1.0) -> None:
        self.x = x
        self.speed = speed
        self.bounding_box = SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0))

    def get_transform(self):
        return SimpleNamespace(
            location=SimpleNamespace(x=self.x, y=1.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )

    def get_velocity(self):
        return SimpleNamespace(x=self.speed, y=0.0, z=0.0)

    def get_acceleration(self):
        return SimpleNamespace(x=0.1, y=0.0, z=0.0)

    def get_angular_velocity(self):
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)


def spawned(name: str, role: str, actor: FakeActor):
    return SimpleNamespace(
        name=name,
        role=role,
        blueprint="vehicle.test",
        actor=actor,
    )


class R23CollectionTest(unittest.TestCase):
    def test_collection_anchor_rejects_offline_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "head.pt"
            checkpoint.write_bytes(b"head")
            import hashlib

            sha = hashlib.sha256(b"head").hexdigest()
            ok = {
                "status": "OK",
                "checkpoint_sha256": sha,
                "allowed_uses": ["formal_offline", "r2k_pilot"],
                "forbidden_uses": [],
            }
            audit = validate_checkpoint_for_use(
                checkpoint, "collection_anchor", manifest=ok
            )
            self.assertTrue(audit["ok"])
            offline_only = {
                **ok,
                "status": "HEAD_TRAINED_NOT_FORMAL",
                "allowed_uses": ["offline_diagnostic"],
            }
            with self.assertRaises(CheckpointContractError):
                validate_checkpoint_for_use(
                    checkpoint, "collection_anchor", manifest=offline_only
                )

    def test_default_campaign_exact_scale_and_split_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.json"
            bank_path.write_text(json.dumps(default_lineage_bank()), encoding="utf-8")
            lineages = load_lineage_bank(bank_path)
            r2 = build_phase_manifest(
                lineages=lineages,
                phase="r2_calibration",
                campaign_id="r2",
                r2_checkpoint_sha256="0" * 64,
                collection_config_sha256="1" * 64,
            )
            world = build_phase_manifest(
                lineages=lineages,
                phase="world_formal",
                campaign_id="world",
                r2_checkpoint_sha256="2" * 64,
                collection_config_sha256="1" * 64,
            )
            self.assertEqual(validate_campaign_manifest(r2)["n_slots"], 360)
            audit = validate_campaign_manifest(world)
            self.assertEqual(audit["n_slots"], 1200)
            self.assertEqual(audit["n_reserve"], 192)
            self.assertEqual(audit["n_lineages"], 84)
            split_by_lineage: dict[str, set[str]] = {}
            for row in world["slots"]:
                split_by_lineage.setdefault(row["lineage_id"], set()).add(row["split"])
            self.assertTrue(all(len(values) == 1 for values in split_by_lineage.values()))

    def test_v2_campaign_scale_marginals_and_independent_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank-v2.json"
            bank = default_lineage_bank_v2()
            bank_path.write_text(json.dumps(bank), encoding="utf-8")
            lineages = load_lineage_bank(bank_path)
            r2 = build_phase_manifest(
                lineages=lineages,
                phase="r2_calibration",
                campaign_id="r2-v2",
                r2_checkpoint_sha256="1" * 64,
                collection_config_sha256="2" * 64,
                expected_base_lineages=84,
                reserve_lineages=0,
                completed_target=1008,
                campaign_version="v2",
                conditions=tuple(bank["conditions"]),
            )
            r2_audit = validate_campaign_manifest(r2)
            self.assertEqual(r2_audit["n_slots"], 1008)
            self.assertEqual(r2_audit["n_lineages"], 84)
            base_rows = [row for row in r2["slots"] if not row["reserve"]]
            map_lineages = {
                (row["map_name"], row["lineage_id"]) for row in base_rows
            }
            family_lineages = {
                (row["family"], row["lineage_id"]) for row in base_rows
            }
            self.assertTrue(
                all(
                    sum(name == map_name for name, _ in map_lineages) == 12
                    for map_name in bank["maps"]
                )
            )
            self.assertTrue(
                all(
                    sum(name == family for name, _ in family_lineages) == 14
                    for family in bank["families"]
                )
            )
            split_lineages = {}
            for row in base_rows:
                split_lineages.setdefault(row["split"], set()).add(row["lineage_id"])
            self.assertEqual(
                {key: len(value) for key, value in split_lineages.items()},
                {"train": 60, "val": 12, "holdout": 12},
            )
            for split in ("train", "val", "holdout"):
                rows = [
                    row
                    for row in base_rows
                    if row["lineage_id"] in split_lineages[split]
                ]
                self.assertEqual(
                    {row["map_name"] for row in rows}, set(bank["maps"])
                )
                self.assertEqual(
                    {row["family"] for row in rows}, set(bank["families"])
                )

            world = build_phase_manifest(
                lineages=lineages,
                phase="world_formal",
                campaign_id="world-v2",
                r2_checkpoint_sha256="3" * 64,
                collection_config_sha256="2" * 64,
                expected_base_lineages=168,
                reserve_lineages=42,
                completed_target=2000,
                campaign_version="v2",
                conditions=tuple(bank["conditions"]),
            )
            world_audit = validate_campaign_manifest(world)
            self.assertEqual(world_audit["n_slots"], 2520)
            self.assertEqual(world_audit["n_reserve"], 504)
            self.assertEqual(world_audit["n_lineages"], 210)
            base_ids = {
                row["lineage_id"] for row in world["slots"] if not row["reserve"]
            }
            reserve_ids = {
                row["lineage_id"] for row in world["slots"] if row["reserve"]
            }
            self.assertFalse(base_ids.intersection(reserve_ids))

    def test_checkpoint_must_be_contiguous_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(default_lineage_bank()), encoding="utf-8")
            manifest = build_phase_manifest(
                lineages=load_lineage_bank(path),
                phase="r2_calibration",
                campaign_id="r2",
                r2_checkpoint_sha256="0" * 64,
                collection_config_sha256="1" * 64,
            )
            first, second = manifest["slots"][:2]
            checkpoint = {
                "schema_version": CAMPAIGN_CHECKPOINT_SCHEMA,
                "manifest_content_hash": manifest["manifest_content_hash"],
                "last_completed_index": 0,
                "results": [{"slot_id": first["slot_id"], "status": "COMPLETED"}],
            }
            self.assertEqual(validate_checkpoint(checkpoint, manifest)["next_index"], 1)
            checkpoint["results"] = [{"slot_id": second["slot_id"], "status": "COMPLETED"}]
            with self.assertRaisesRegex(R23CollectionError, "contiguous"):
                validate_checkpoint(checkpoint, manifest)

    def test_manifest_hash_and_frozen_r2_binding(self) -> None:
        bank = default_lineage_bank()
        body = dict(bank)
        body.pop("bank_content_hash")
        self.assertEqual(bank["bank_content_hash"], content_hash(body))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            with self.assertRaisesRegex(R23CollectionError, "frozen R2"):
                build_phase_manifest(
                    lineages=load_lineage_bank(path),
                    phase="world_formal",
                    campaign_id="world",
                    r2_checkpoint_sha256="0" * 64,
                    collection_config_sha256="1" * 64,
                )

    def test_observable_history_has_five_real_distinct_frames(self) -> None:
        ego = FakeActor(0.0, 2.0)
        npc = FakeActor(5.0, 1.0)
        recorder = ObservableHistoryRecorder(scenario_id="s", seed_id="a")
        for index in range(21):
            time_s = index * 0.05
            ego.x = 2.0 * time_s
            npc.x = 5.0 + time_s
            recorder.record(
                simulation_time_s=time_s,
                frame=index,
                spawned_actors=[
                    spawned("ego", "ego", ego),
                    spawned("npc", "vehicle", npc),
                ],
            )
        history = recorder.finalize(anchor_time_s=1.0)
        self.assertEqual(history["history_count"], 5)
        self.assertEqual(len(set(history["frame_ids"])), 5)
        self.assertEqual(
            [round(x["dt"], 2) for x in history["ego_history"]],
            [-0.8, -0.6, -0.4, -0.2, 0.0],
        )
        scene = merge_history_into_scene({"ego": history["ego_history"][-1]}, history)
        self.assertEqual(len(scene["ego_history"]), 5)
        self.assertEqual(len(scene["actor_histories"][0]["history"]), 5)

    def test_observable_history_fails_without_coverage(self) -> None:
        recorder = ObservableHistoryRecorder(scenario_id="s", seed_id="a")
        recorder.record(
            simulation_time_s=1.0,
            frame=1,
            spawned_actors=[spawned("ego", "ego", FakeActor(0.0))],
        )
        with self.assertRaises(ObservableHistoryError):
            recorder.finalize(anchor_time_s=1.0)

    def test_reactive_controller_is_candidate_blind(self) -> None:
        control = deterministic_reactive_control(
            {
                "relative_distance_m": 3.0,
                "closing_speed_mps": 2.0,
                "actor_speed_mps": 4.0,
                "lane_conflict": True,
                "actor_has_priority": False,
            }
        )
        self.assertGreater(control["brake"], 0.0)
        with self.assertRaises(ReactiveActorContractError):
            deterministic_reactive_control(
                {"candidate_id": "defensive", "relative_distance_m": 3.0}
            )

    def test_joint_sample_keeps_failed_and_singleton_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            path.write_text(json.dumps(default_lineage_bank()), encoding="utf-8")
            manifest = build_phase_manifest(
                lineages=load_lineage_bank(path),
                phase="r2_calibration",
                campaign_id="r2",
                r2_checkpoint_sha256="1" * 64,
                collection_config_sha256="2" * 64,
            )
            slot = manifest["slots"][0]
            from driving_vla.evaluation.r23_collection import CollectionSlot

            sample = R23CollectionSampleV1(
                slot=CollectionSlot.from_dict(slot),
                status="SINGLETON",
                observable_path="observable.json",
                anchor_artifact_path="anchor.json",
                feature_path="feature.json",
                branch_paths=("branch-0", None),
                unavailable_reasons=(None, "NO_ALTERNATIVE"),
                provenance={
                    "registry_sha256": "3" * 64,
                    "r2_checkpoint_sha256": "1" * 64,
                    "executor_sha256": "4" * 64,
                    "collection_config_sha256": "2" * 64,
                },
            )
            self.assertEqual(sample.to_dict()["status"], "SINGLETON")

    def test_completed_joint_artifacts_bind_forward_feature_and_five_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            anchor = attempt / "anchor"
            anchor.mkdir()
            forward_id = "forward-1"
            feature_hash = "f" * 64
            (anchor / "feature.json").write_text(
                json.dumps(
                    {
                        "scenario_id": "scenario",
                        "seed_id": "seed_a",
                        "ego_v": 4.0,
                        "driving_feature": [0.0] * 64,
                        "driving_feature_hash": feature_hash,
                        "backbone_forward_id": forward_id,
                    }
                ),
                encoding="utf-8",
            )
            (anchor / "feature_lineage.json").write_text(
                json.dumps(
                    {
                        "backbone_forward_id": forward_id,
                        "driving_feature_hash": feature_hash,
                        "native_feature_hash": feature_hash,
                    }
                ),
                encoding="utf-8",
            )
            (anchor / "anchor_bundle_v2.json").write_text(
                json.dumps({"backbone_forward_id": forward_id}),
                encoding="utf-8",
            )
            (anchor / "anchor_front_rgb.npy").write_bytes(b"npy-audit")
            history = {
                "history_count": 5,
                "frame_ids": [1, 2, 3, 4, 5],
                "ego_history": [{"valid": True}] * 5,
            }
            (anchor / "observable_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            for branch in (0, 1):
                branch_dir = attempt / f"branch-{branch}"
                branch_dir.mkdir()
                (branch_dir / "observable_scene_t0.json").write_text(
                    json.dumps(
                        {
                            "ego_history": [{"valid": True}] * 5,
                            "actor_histories": [
                                {"history": [{"valid": index > 1} for index in range(5)]}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            audit = validate_completed_joint_artifacts(
                attempt_dir=attempt,
                expected_scenario_id="scenario",
                expected_seed_id="seed_a",
            )
            self.assertEqual(audit["history_count"], 5)
            self.assertEqual(audit["backbone_forward_id"], forward_id)

            feature = json.loads((anchor / "feature.json").read_text(encoding="utf-8"))
            feature["backbone_forward_id"] = "different-forward"
            (anchor / "feature.json").write_text(
                json.dumps(feature), encoding="utf-8"
            )
            with self.assertRaisesRegex(R23CollectionError, "forward lineage"):
                validate_completed_joint_artifacts(
                    attempt_dir=attempt,
                    expected_scenario_id="scenario",
                    expected_seed_id="seed_a",
                )
