"""X5A driving feature contract: raw/full_pool/mean64 + fail-closed."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.driving_feature import (  # noqa: E402
    DRIVING_FEAT_DIM,
    DrivingFeatureError,
    build_context_vector,
    extract_driving_feature_bundle,
    feature_vector_hash,
    linear_probe_labels_accuracy,
    observable_scene_vector,
    pool_driving_adaptor_output,
)


class DrivingFeatureExtractTest(unittest.TestCase):
    def test_observable_actor_context_is_runtime_safe_and_in_96_dim_abi(self) -> None:
        scene = {
            "actor_present": True,
            "actor_lon_m": 12.0,
            "actor_lat_m": -1.5,
            "actor_speed_mps": 2.0,
            # Privileged values are deliberately ignored by the encoder.
            "candidate_id": "forbidden",
            "oracle_winner": 1,
            "future_x": [999.0],
            "scenario_family": "forbidden",
        }
        vec = observable_scene_vector(scene, ego_v=5.0)
        self.assertEqual(len(vec), 8)
        self.assertEqual(vec[0], 1.0)
        privileged_changed = dict(scene)
        privileged_changed.update(
            candidate_id="other",
            oracle_winner=0,
            future_x=[-999.0],
            scenario_family="other",
        )
        self.assertEqual(
            vec, observable_scene_vector(privileged_changed, ego_v=5.0)
        )
        ctx = build_context_vector(
            [(0.0, 0.0), (1.0, 0.0)],
            ego_v=5.0,
            base_speed_mps=5.0,
            driving_feature=[0.1] * 64,
            observable_scene=scene,
        )
        self.assertEqual(len(ctx), 96)
        self.assertEqual(ctx[16:24], vec)

    def test_missing_actor_has_zero_observable_context(self) -> None:
        self.assertEqual(
            observable_scene_vector(
                {
                    "actor_present": False,
                    "actor_lon_m": None,
                    "actor_lat_m": None,
                    "actor_speed_mps": None,
                },
                ego_v=4.0,
            ),
            [0.0] * 8,
        )

    def test_defensive_speed_margin_only_preserves_teacher_gap(self) -> None:
        import torch

        from driving_vla.model.spatial_mode_heads import (
            defensive_speed_margin_loss,
        )

        target = torch.tensor([0.85, 1.0], dtype=torch.float32)
        good = torch.tensor([0.84, 1.0], dtype=torch.float32)
        collapsed = torch.tensor([0.97, 1.0], dtype=torch.float32)
        self.assertAlmostEqual(
            float(defensive_speed_margin_loss(good, target).item()), 0.0, places=6
        )
        self.assertGreater(
            float(defensive_speed_margin_loss(collapsed, target).item()), 0.0
        )

    def test_mean64_and_full_pool_from_btc(self) -> None:
        # [B,T,C] with C=128, T=4 — left-biased tokens
        rng = np.random.RandomState(0)
        arr = rng.randn(1, 4, 128).astype(np.float32)
        arr[:, :, :10] += 5.0  # strong early channels
        b = extract_driving_feature_bundle(arr, require=True)
        self.assertTrue(b.ok)
        self.assertEqual(len(b.mean64), DRIVING_FEAT_DIM)
        self.assertEqual(b.full_pool_dim, 128)
        self.assertEqual(len(b.full_pool), 128)
        self.assertEqual(len(b.raw_shape), 3)
        self.assertTrue(b.raw_content_hash)
        self.assertEqual(b.mean64_hash, feature_vector_hash(b.mean64))
        self.assertEqual(b.full_pool_hash, feature_vector_hash(b.full_pool))
        # mean64 is first 64 of full pool
        self.assertEqual(list(b.mean64), list(b.full_pool[:64]))

    def test_deterministic_hash(self) -> None:
        arr = np.ones((2, 8, 32), dtype=np.float32)
        b1 = extract_driving_feature_bundle(arr, require=True)
        b2 = extract_driving_feature_bundle(arr, require=True)
        self.assertEqual(b1.raw_content_hash, b2.raw_content_hash)
        self.assertEqual(b1.mean64_hash, b2.mean64_hash)

    def test_none_require_raises(self) -> None:
        with self.assertRaises(DrivingFeatureError):
            extract_driving_feature_bundle(None, require=True)

    def test_none_soft_returns_not_ok(self) -> None:
        b = extract_driving_feature_bundle(None, require=False)
        self.assertFalse(b.ok)
        self.assertTrue(b.error)

    def test_legacy_pool_zeros_on_none(self) -> None:
        z = pool_driving_adaptor_output(None, require=False)
        self.assertEqual(len(z), DRIVING_FEAT_DIM)
        self.assertTrue(all(x == 0.0 for x in z))

    def test_legacy_pool_require_raises(self) -> None:
        with self.assertRaises(DrivingFeatureError):
            pool_driving_adaptor_output(None, require=True)

    def test_all_zeros_mean64_fails_require(self) -> None:
        arr = np.zeros((1, 3, 16), dtype=np.float32)
        with self.assertRaises(DrivingFeatureError):
            extract_driving_feature_bundle(arr, require=True)

    def test_linear_probe_separates_left_right_synthetic(self) -> None:
        # Strong channel-0 separation + low noise; enough samples for holdout
        feats = []
        labels = []
        for i in range(80):
            v = np.zeros(64, dtype=np.float64)
            lab = 0 if i < 40 else 1
            v[0] = -5.0 if lab == 0 else 5.0
            v += np.random.RandomState(i).randn(64) * 0.05
            feats.append(v.tolist())
            labels.append(lab)
        rep = linear_probe_labels_accuracy(feats, labels, seed=1, train_ratio=0.75)
        self.assertGreaterEqual(rep["accuracy"], 0.9)
        self.assertEqual(rep["ok"], 1.0)


class DevelopmentSmokeCheckpointContractTest(unittest.TestCase):
    def test_development_live_smoke_requires_explicit_allow(self) -> None:
        from driving_vla.model.checkpoint_contract import (
            CheckpointContractError,
            require_checkpoint_for_use,
            write_checkpoint_manifest,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "head.pt"
            checkpoint.write_bytes(b"diagnostic-head")
            manifest_path = root / "CHECKPOINT_STATUS.json"
            write_checkpoint_manifest(
                manifest_path,
                checkpoint_path=checkpoint,
                status="HEAD_TRAINED_NOT_FORMAL",
                allowed_uses=["offline_diagnostic"],
                forbidden_uses=["x5h_acceptance", "r2k_pilot"],
            )
            with self.assertRaises(CheckpointContractError):
                require_checkpoint_for_use(
                    checkpoint, "development_live_smoke"
                )
            write_checkpoint_manifest(
                manifest_path,
                checkpoint_path=checkpoint,
                status="HEAD_TRAINED_NOT_FORMAL",
                allowed_uses=[
                    "offline_diagnostic",
                    "development_live_smoke",
                ],
                forbidden_uses=["x5h_acceptance", "r2k_pilot"],
            )
            result = require_checkpoint_for_use(
                checkpoint, "development_live_smoke"
            )
            self.assertTrue(result["ok"])

    def test_formal_checkpoint_is_bound_to_exact_blind_registry(self) -> None:
        from driving_vla.model.checkpoint_contract import (
            CheckpointContractError,
            require_checkpoint_blind_registry,
            write_checkpoint_manifest,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "head.pt"
            checkpoint.write_bytes(b"formal-head")
            write_checkpoint_manifest(
                root / "CHECKPOINT_STATUS.json",
                checkpoint_path=checkpoint,
                status="OK",
                allowed_uses=["x5h_acceptance", "r2k_pilot"],
                forbidden_uses=[],
                extra={
                    "blind_registry_sha256": "a" * 64,
                    "blind_registry_version": "v2-blind",
                    "blind_pair_overlap_zero": True,
                },
            )
            self.assertTrue(
                require_checkpoint_blind_registry(checkpoint, "a" * 64)["ok"]
            )
            with self.assertRaises(CheckpointContractError):
                require_checkpoint_blind_registry(checkpoint, "b" * 64)


class NativePathPredictionFeatureFieldsTest(unittest.TestCase):
    def test_dataclass_has_feature_fields(self) -> None:
        from driving_vla.model.neural_policy import NativePathPrediction

        n = NativePathPrediction(
            path_map_xy=((0.0, 0.0), (1.0, 0.0)),
            speed_mps=(1.0,),
            target_ego_1=(1.0, 0.0),
            target_ego_2=(2.0, 0.0),
            latency_s=0.0,
            peak_vram_mb=0.0,
            driving_feature=(0.1,) * 64,
            driving_feature_hash="abc",
            driving_feature_ok=True,
        )
        self.assertTrue(n.driving_feature_ok)
        self.assertEqual(len(n.driving_feature), 64)


class RawDumpTest(unittest.TestCase):
    def test_dump_raw_tokens_fp16(self) -> None:
        import tempfile
        from pathlib import Path

        from driving_vla.model.driving_feature import dump_raw_tokens_fp16

        arr = np.random.RandomState(0).randn(1, 4, 32).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "raw.npy"
            h, shape, dtype = dump_raw_tokens_fp16(arr, p)
            self.assertTrue(p.is_file())
            self.assertEqual(shape, (1, 4, 32))
            self.assertTrue(h)
            loaded = np.load(str(p))
            self.assertEqual(loaded.dtype, np.float16)

    def test_extract_with_raw_path(self) -> None:
        import tempfile
        from pathlib import Path

        arr = np.random.RandomState(1).randn(1, 3, 40).astype(np.float32) + 0.5
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.npy"
            b = extract_driving_feature_bundle(arr, require=True, raw_tensor_path=str(p))
            self.assertTrue(b.ok)
            self.assertTrue(p.is_file())
            self.assertEqual(b.raw_tensor_path, str(p))


class CollectFailClosedCliTest(unittest.TestCase):
    def test_collect_requires_mode(self) -> None:
        import subprocess

        script = ROOT / "scripts" / "r2x_feature_collect.py"
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("synthetic", (r.stderr + r.stdout).lower())

    def test_synthetic_mode_ok(self) -> None:
        import subprocess
        import tempfile
        from pathlib import Path

        script = ROOT / "scripts" / "r2x_feature_collect.py"
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--synthetic-tensors",
                    "--n",
                    "8",
                    "--out",
                    td,
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            man = json.loads((Path(td) / "collect_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(man.get("is_real_simlingo_feature"))
            self.assertEqual(man.get("mode"), "synthetic_tensor")


class TeacherSkeletonTest(unittest.TestCase):
    def test_lateral_teacher_completes_smooth_out_and_rejoin(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            generate_lattice,
        )

        cfg = TeacherConfig(
            d_peaks_m=(1.0,),
            speed_scales=(1.0,),
            side_shift_start_s=(0.2,),
            side_shift_recover_s=(0.8,),
            require_execution_filters=False,
        )
        profile = list(generate_lattice(n_path=101, config=cfg)[0].raw_d)
        self.assertEqual(profile[0], 0.0)
        self.assertEqual(profile[-1], 0.0)
        self.assertGreater(max(profile), 0.99)
        peak_index = profile.index(max(profile))
        self.assertGreater(peak_index, 35)
        self.assertLess(peak_index, 65)
        self.assertLess(max(abs(b - a) for a, b in zip(profile, profile[1:])), 0.12)

    def test_teacher_v8_identity_binds_topology_authorization(self):
        from driving_vla.model.spatial_k2_teacher import TeacherConfig

        cfg = TeacherConfig()
        self.assertEqual(cfg.schema_version, "safedrive.k2_spatial_teacher.v8")
        self.assertEqual(
            cfg.teacher_id, "spatial_defensive_lattice_v8_topology_authorized"
        )

    def test_centered_vehicle_obstruction_is_not_fake_overtake_label(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        label = select_defensive_teacher(
            scenario_family="obstruction",
            conflict_side="left",
            config=TeacherConfig(require_execution_filters=False),
            allow_unit_test_stub_metrics=True,
        )
        self.assertFalse(label.alternative_available)
        self.assertEqual(
            label.availability_reason,
            "obstruction_requires_topology_authorization",
        )

    def test_adjacent_obstruction_can_be_labeled_only_when_authorized(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        label = select_defensive_teacher(
            scenario_family="obstruction",
            conflict_side="left",
            config=TeacherConfig(require_execution_filters=False),
            allow_unit_test_stub_metrics=True,
            privileged_scene={"adjacent_lane_authorized": True},
        )
        self.assertTrue(label.alternative_available)

    def test_teacher_v5_preserves_native_spatial_horizon(self) -> None:
        from driving_vla.model.frenet_codec import decode_frenet_residual_path
        from driving_vla.model.spatial_k2_teacher import native_horizon_raw_delta_s

        native = [(float(i), 0.0) for i in range(20)]
        raw_ds = native_horizon_raw_delta_s(native, len(native))
        _xy, s_values, _d = decode_frenet_residual_path(
            native, raw_ds, [0.0] * len(native)
        )
        self.assertGreaterEqual(s_values[-1] / 19.0, 0.99)

    def test_empty_road_no_alternative(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        lab = select_defensive_teacher(
            scenario_family="empty",
            conflict_side="empty",
            config=TeacherConfig(require_execution_filters=False),
            allow_unit_test_stub_metrics=True,
        )
        self.assertFalse(lab.alternative_available)
        self.assertIsNone(lab.defensive_residual)

    def test_centered_actor_is_not_misclassified_as_empty_road(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        lab = select_defensive_teacher(
            scenario_family="obstruction",
            conflict_side="none",
            config=TeacherConfig(require_execution_filters=False),
            allow_unit_test_stub_metrics=True,
        )
        self.assertNotEqual(
            lab.availability_reason, "empty_road_no_forced_swerve"
        )

    def test_available_center_teacher_never_uses_sub_floor_residual(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        cfg = TeacherConfig(require_execution_filters=False, min_lateral_sep_m=0.5)
        lab = select_defensive_teacher(
            scenario_family="lead_brake",
            conflict_side="center",
            config=cfg,
            allow_unit_test_stub_metrics=True,
        )
        if lab.alternative_available:
            self.assertGreaterEqual(
                abs(float(lab.defensive_residual["d_peak_m"])), 0.5
            )

    def test_production_guard_rejects_sub_floor_defensive_candidate(self) -> None:
        from driving_vla.model.spatial_k2_teacher import generate_lattice
        from driving_vla.model.teacher_offline_stages import stage_guard_v2

        candidate = next(
            item
            for item in generate_lattice(n_path=20)
            if abs(abs(item.d_peak_m) - 0.4) < 1.0e-9
            and abs(item.speed_scale - 1.0) < 1.0e-9
        )
        result = stage_guard_v2(
            {
                "scenario_family": "lead_brake",
                "conflict_side": "center",
                "native_path_xy": [[float(i), 0.0] for i in range(20)],
                "ego_v": 5.0,
                "d_peak_m": candidate.d_peak_m,
                "speed_scale": candidate.speed_scale,
                "raw_d": candidate.raw_d,
                "raw_delta_s": candidate.raw_delta_s,
            }
        )
        self.assertFalse(result.passed)
        self.assertIn("SPATIAL_COLLAPSE", result.reason)

    def test_speed_only_lattice_candidate_does_not_bypass_diversity(self) -> None:
        from driving_vla.model.spatial_k2_teacher import generate_lattice
        from driving_vla.model.teacher_offline_stages import stage_guard_v2

        candidate = next(
            item
            for item in generate_lattice(n_path=20)
            if abs(item.d_peak_m) < 1.0e-9
            and abs(item.speed_scale - 0.85) < 1.0e-9
        )
        result = stage_guard_v2(
            {
                "candidate_id": candidate.candidate_id,
                "scenario_family": "left_cut_in",
                "conflict_side": "left",
                "native_path_xy": [[float(i), 0.0] for i in range(20)],
                "ego_v": 5.0,
                "d_peak_m": candidate.d_peak_m,
                "speed_scale": candidate.speed_scale,
                "raw_d": candidate.raw_d,
                "raw_delta_s": candidate.raw_delta_s,
            }
        )
        self.assertFalse(result.passed)
        self.assertIn("SPATIAL_COLLAPSE", result.reason)

    def test_left_conflict_moves_right_negative_d(self) -> None:
        """+d is left normal; left conflict must select −d (away)."""
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            away_from_conflict_sign,
            generate_lattice,
            select_defensive_teacher,
        )

        self.assertEqual(away_from_conflict_sign("left"), -1.0)
        self.assertEqual(away_from_conflict_sign("right"), 1.0)
        lat = generate_lattice(n_path=20)
        self.assertGreater(len(lat), 10)
        lab = select_defensive_teacher(
            scenario_family="left_cut_in",
            conflict_side="left",
            config=TeacherConfig(require_execution_filters=False),
            allow_unit_test_stub_metrics=True,
        )
        self.assertTrue(lab.alternative_available)
        self.assertIsNotNone(lab.defensive_residual)
        d_peak = float(lab.defensive_residual["d_peak_m"])
        self.assertLess(d_peak, -0.4, msg="left conflict must not increase +d overlap")

    def test_right_conflict_moves_left_positive_d(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        lab = select_defensive_teacher(
            scenario_family="right_cut_in",
            conflict_side="right",
            config=TeacherConfig(require_execution_filters=False),
            allow_unit_test_stub_metrics=True,
        )
        self.assertTrue(lab.alternative_available)
        self.assertIsNotNone(lab.defensive_residual)
        d_peak = float(lab.defensive_residual["d_peak_m"])
        self.assertGreater(d_peak, 0.4, msg="right conflict must not increase -d overlap")

    def test_require_execution_filters_raises(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            select_defensive_teacher,
        )

        cfg = TeacherConfig(require_execution_filters=True)
        with self.assertRaises(RuntimeError):
            select_defensive_teacher(
                scenario_family="left_cut_in",
                conflict_side="left",
                config=cfg,
                execution_stages=None,
            )

    def test_dummy_stage_cannot_satisfy_production(self) -> None:
        from driving_vla.model.spatial_k2_teacher import (
            TeacherConfig,
            TeacherFilterResult,
            select_defensive_teacher,
            validate_production_stages,
        )

        def dummy(_p):
            return TeacherFilterResult("dummy", True, "always")

        with self.assertRaises(RuntimeError):
            validate_production_stages({"dummy": dummy})
        # incomplete production mapping
        with self.assertRaises(RuntimeError):
            select_defensive_teacher(
                scenario_family="left_cut_in",
                conflict_side="left",
                config=TeacherConfig(require_execution_filters=True),
                execution_stages={"guard_v2": dummy},
            )

    def test_old_checkpoint_formal_reject_before_load(self) -> None:
        from pathlib import Path

        from driving_vla.model.checkpoint_contract import (
            CheckpointContractError,
            require_checkpoint_for_use,
        )

        ckpt = (
            Path(__file__).resolve().parents[2]
            / "docs/runtime-evidence/r2x-training/checkpoints/v4_real_pathfix/spatial_heads_last.pt"
        )
        if not ckpt.is_file():
            self.skipTest("v4 checkpoint missing")
        with self.assertRaises(CheckpointContractError):
            require_checkpoint_for_use(ckpt, "x5h_acceptance")
        with self.assertRaises(CheckpointContractError):
            require_checkpoint_for_use(ckpt, "r2k_pilot")
        with self.assertRaises(CheckpointContractError):
            require_checkpoint_for_use(ckpt, "formal_offline")
        # historical may pass
        rep = require_checkpoint_for_use(ckpt, "historical_comparison")
        self.assertTrue(rep["ok"])

    def test_project_point_segment_frenet_roundtrip(self) -> None:
        import math

        from driving_vla.model.frenet_codec import (
            project_path_to_frenet,
            project_point_to_polyline,
        )

        # curved-ish polyline
        native = [(float(i), 0.1 * math.sin(i * 0.3)) for i in range(12)]
        # point offset to the left of first segment
        s, d, qx, qy, nx, ny = project_point_to_polyline(native, (1.0, 0.5))
        self.assertGreaterEqual(s, 0.0)
        # reconstruct
        rx, ry = qx + d * nx, qy + d * ny
        self.assertAlmostEqual(rx, 1.0, places=5)
        self.assertAlmostEqual(ry, 0.5, places=5)
        path = [(p[0] + 0.2 * nx, p[1] + 0.2 * ny) for p in native]
        ss, dd = project_path_to_frenet(native, path)
        self.assertEqual(len(ss), len(path))
        for i in range(1, len(ss)):
            self.assertGreaterEqual(ss[i] + 1e-6, ss[i - 1])


class RunAnchorV2ApiTest(unittest.TestCase):
    def test_run_anchor_v2_from_bundle_exists(self) -> None:
        from driving_vla.evaluation import paired_live

        self.assertTrue(callable(paired_live.run_anchor_v2_from_bundle))
        self.assertTrue(callable(paired_live.run_anchor_v2))


class NeuralV2PolicyOfflineTest(unittest.TestCase):
    def test_low_learned_confidence_does_not_veto_diverse_proposal(self) -> None:
        from driving_vla.model.neural_policy import (
            NativePathPrediction,
            NeuralV2Policy,
        )
        from driving_vla.model.spatial_mode_heads import SpatialHeadOutput

        policy = NeuralV2Policy(
            lazy=True, require_driving_feature=True, device="cpu"
        )
        path = tuple((float(index) * 1.2, 0.0) for index in range(20))
        native = NativePathPrediction(
            path_map_xy=path,
            speed_mps=(2.0,) * 5,
            target_ego_1=(5.0, 0.0),
            target_ego_2=(10.0, 0.0),
            latency_s=0.01,
            peak_vram_mb=100.0,
            driving_feature=tuple([0.1] * 64),
            driving_feature_hash="confidence-feature",
            driving_feature_raw_hash="confidence-raw",
            driving_feature_source="simlingo_driving_mean64_v1",
            driving_feature_ok=True,
        )

        class _Obs:
            front_rgb = None
            ego_x = 0.0
            ego_y = 0.0
            ego_yaw = 0.0
            ego_speed_mps = 2.0
            meta = {}
            route_xy = ((0.0, 0.0), (20.0, 0.0))

        nominal = SpatialHeadOutput(
            raw_delta_s=[0.5] * 20,
            raw_d=[0.0] * 20,
            speed_scale=1.0,
            available=True,
            avail_prob=1.0,
        )
        defensive = SpatialHeadOutput(
            raw_delta_s=[0.5] * 20,
            raw_d=[0.0, 0.5] + [2.0] * 18,
            speed_scale=0.85,
            available=False,
            avail_prob=0.01,
        )
        policy.v0.predict_native = lambda obs: native  # type: ignore[method-assign]
        policy.v0.resolve_vla_input_speed_mps = lambda obs: 2.0  # type: ignore[method-assign]
        policy.head.predict_modes = (  # type: ignore[method-assign]
            lambda *args, **kwargs: (nominal, defensive)
        )
        bundle = policy.predict_bundle(_Obs())  # type: ignore[arg-type]
        self.assertEqual(bundle.candidates[0].spatial_path_xy, path)
        self.assertEqual(bundle.candidates[0].proposal_path_hash, bundle.native_path_hash)
        self.assertTrue(all(abs(float(d)) < 1e-12 for d in bundle.candidates[0].frenet_d))
        self.assertTrue(bundle.candidates[1].available)
        self.assertAlmostEqual(bundle.candidates[1].probability, 0.01)
        self.assertEqual(
            bundle.set_diagnostics["availability_semantics"],
            "executability_only_v1",
        )
        self.assertFalse(
            bundle.set_diagnostics["learned_availability_decision"]
        )

    def test_v2_fail_closed_without_feature(self) -> None:
        from driving_vla.model.driving_feature import DrivingFeatureError
        from driving_vla.model.neural_policy import NeuralV2Policy, NativePathPrediction

        pol = NeuralV2Policy(lazy=True, require_driving_feature=True, device="cpu")
        # inject stub native without going through SimLingo
        bad = NativePathPrediction(
            path_map_xy=tuple((float(i), 0.0) for i in range(20)),
            speed_mps=(2.0,) * 5,
            target_ego_1=(5.0, 0.0),
            target_ego_2=(10.0, 0.0),
            latency_s=0.0,
            peak_vram_mb=0.0,
            driving_feature_ok=False,
            driving_feature_error="missing",
        )

        class _Obs:
            front_rgb = None
            ego_x = 0.0
            ego_y = 0.0
            ego_yaw = 0.0
            ego_speed_mps = 2.0
            meta = {}
            route_xy = ((0.0, 0.0), (10.0, 0.0))

        def fake_native(obs):
            return bad

        pol.v0.predict_native = fake_native  # type: ignore[method-assign]
        with self.assertRaises(DrivingFeatureError):
            pol.predict_bundle(_Obs())  # type: ignore[arg-type]

    def test_head_collapse_not_rescued_by_runtime_template(self) -> None:
        """Collapsed head residual must NOT be replaced by lattice hump."""
        from driving_vla.model.neural_policy import NeuralV2Policy, NativePathPrediction
        from driving_vla.model.spatial_mode_heads import SpatialHeadOutput

        pol = NeuralV2Policy(lazy=True, require_driving_feature=True, device="cpu")
        path = tuple((float(i) * 1.2, 0.0) for i in range(20))
        feat = [0.1] * 64
        good = NativePathPrediction(
            path_map_xy=path,
            speed_mps=(2.0,) * 5,
            target_ego_1=(5.0, 0.0),
            target_ego_2=(10.0, 0.0),
            latency_s=0.01,
            peak_vram_mb=100.0,
            driving_feature=tuple(feat),
            driving_feature_hash="h",
            driving_feature_raw_hash="r",
            driving_feature_source="simlingo_driving_mean64_v1",
            driving_feature_ok=True,
        )

        class _Obs:
            front_rgb = None
            ego_x = 0.0
            ego_y = 0.0
            ego_yaw = 0.0
            ego_speed_mps = 2.0
            meta = {}
            route_xy = ((0.0, 0.0), (20.0, 0.0))

        # Collapsed: both modes near-zero lateral, but head claims available
        flat = [0.0] * 20
        o0 = SpatialHeadOutput(
            raw_delta_s=[0.5] * 20,
            raw_d=list(flat),
            speed_scale=1.0,
            available=True,
            avail_prob=0.9,
        )
        o1 = SpatialHeadOutput(
            raw_delta_s=[0.5] * 20,
            raw_d=[0.05] * 20,  # tiny lateral → collapse
            speed_scale=1.0,
            available=True,
            avail_prob=0.9,
        )
        pol.v0.predict_native = lambda obs: good  # type: ignore[method-assign]
        pol.v0.resolve_vla_input_speed_mps = lambda obs: 2.0  # type: ignore[method-assign]
        pol.head.predict_modes = lambda *a, **k: (o0, o1)  # type: ignore[method-assign]
        bundle = pol.predict_bundle(_Obs())  # type: ignore[arg-type]
        # Must not invent large lateral template
        d1 = list(bundle.candidates[1].frenet_d)
        self.assertLess(max(abs(float(x)) for x in d1), 0.4)
        self.assertFalse(bool((bundle.set_diagnostics or {}).get("runtime_rescue")))
        # Collapse → unavailable or SPATIAL_COLLAPSE reject
        self.assertTrue(
            (not bundle.candidates[1].available)
            or ("SPATIAL_COLLAPSE" in str(bundle.guard_reasons))
            or bundle.guard_status != "OK"
        )

    def test_v2_builds_guarded_bundle_with_feature(self) -> None:
        from driving_vla.model.neural_policy import NeuralV2Policy, NativePathPrediction

        pol = NeuralV2Policy(lazy=True, require_driving_feature=True, device="cpu")
        # left-biased mean64
        feat = [0.0] * 64
        feat[0] = 3.0
        good = NativePathPrediction(
            path_map_xy=tuple((float(i) * 1.2, 0.0) for i in range(20)),
            speed_mps=(2.0,) * 5,
            target_ego_1=(5.0, 0.0),
            target_ego_2=(10.0, 0.0),
            latency_s=0.01,
            peak_vram_mb=100.0,
            driving_feature=tuple(feat),
            driving_feature_hash="testhash",
            driving_feature_raw_hash="rawhash",
            driving_feature_source="simlingo_driving_mean64_v1",
            driving_feature_ok=True,
        )

        class _Obs:
            front_rgb = None
            ego_x = 0.0
            ego_y = 0.0
            ego_yaw = 0.0
            ego_speed_mps = 2.0
            meta = {}
            route_xy = ((0.0, 0.0), (20.0, 0.0))

        pol.v0.predict_native = lambda obs: good  # type: ignore[method-assign]
        pol.v0.resolve_vla_input_speed_mps = lambda obs: 2.0  # type: ignore[method-assign]
        bundle = pol.predict_bundle(_Obs())  # type: ignore[arg-type]
        self.assertEqual(len(bundle.candidates), 2)
        self.assertTrue(bundle.backbone_forward_id)
        self.assertIn("driving_feature_hash", bundle.set_diagnostics)


if __name__ == "__main__":
    unittest.main()
