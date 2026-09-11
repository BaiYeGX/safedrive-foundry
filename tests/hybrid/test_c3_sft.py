"""Focused CPU checks for the C3 data adapter and metric contracts."""

from __future__ import annotations

import math
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "simlingo-main"))

from driving_vla.model.sft import (  # noqa: E402
    SFTSample,
    aggregate_metrics,
    deterministic_order,
    map_to_ego,
    masked_smooth_l1,
    native_route_target,
    native_route_target_with_report,
    native_speed_target,
    native_speed_target_from_timeline,
    root_metrics,
    sample_polyline,
)
import scripts.h6_cora_sft as c3_sft  # noqa: E402


def _sample(root_id: str, split: str = "validation") -> SFTSample:
    return SFTSample(
        root_id=root_id,
        pair_id=root_id,
        split=split,
        family="free_flow",
        map_name="Town01",
        weather="ClearNoon",
        seed=17,
        base_root="/tmp/c2",
        anchor_path="/tmp/anchor.json",
        image_path="/tmp/image.png",
        image_sha256="0" * 64,
        proposal_path="/tmp/proposal.json",
        proposal_sha256="1" * 64,
        teacher_source="expert",
        teacher_generator="classic-frenet-st@h1",
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_speed_mps=2.0,
        target_ego_1=(7.5, 0.0),
        target_ego_2=(15.0, 0.0),
        route_target=tuple((float(i), 0.0) for i in range(20)),
        route_mask=(True,) * 20,
        speed_target=tuple((float(i + 1), 0.0) for i in range(10)),
        speed_mask=(True,) * 10,
    )


class C3SamplingTests(unittest.TestCase):
    def test_polyline_sampling_preserves_short_route_mask(self) -> None:
        values, mask = sample_polyline([[0.0, 0.0], [2.0, 0.0]], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(values[:3], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        np.testing.assert_array_equal(mask, [True, True, True, False])
        self.assertEqual(values.shape, (4, 2))

    def test_native_route_uses_one_metre_arc_length_and_ego_frame(self) -> None:
        target, mask = native_route_target(
            [[10.0, 0.0], [10.0, 2.0], [12.0, 2.0]],
            ego_x=10.0,
            ego_y=0.0,
            ego_yaw=math.pi / 2.0,
            steps=6,
        )
        # Map +y is ego +x at a 90-degree yaw; the final query is beyond the
        # 4 m route and remains invalid instead of being extrapolated.
        np.testing.assert_allclose(target[:5], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, -1.0], [2.0, -2.0]], atol=1e-8)
        np.testing.assert_array_equal(mask, [True, True, True, True, True, False])

        behind, behind_mask = native_route_target(
            [[8.0, 0.0], [10.0, 0.0], [12.0, 0.0]],
            ego_x=10.0,
            ego_y=0.0,
            ego_yaw=0.0,
            steps=3,
        )
        # Projection removes the already-traversed prefix; all retained
        # targets are forward along the ordered route.
        np.testing.assert_allclose(behind, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], atol=1e-8)
        np.testing.assert_array_equal(behind_mask, [True, True, True])

    def test_route_projection_keeps_turns_and_reports_support(self) -> None:
        target, mask, report = native_route_target_with_report(
            [[-10.0, 0.0], [0.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
            ego_x=0.0,
            ego_y=1.0,
            ego_yaw=math.pi / 2.0,
            steps=20,
        )
        self.assertTrue(mask.all())
        self.assertAlmostEqual(report["projection_distance_m"], 0.0, places=8)
        self.assertGreaterEqual(report["support_m"], 19.0)
        # The turn is retained; negative map coordinates are not used as a
        # heuristic for dropping a legal route prefix.
        self.assertGreater(float(target[1, 0]), 0.0)
        self.assertLess(float(target[10, 1]), 0.0)

    def test_projection_rejects_later_crossing_hidden_by_adjacent_ties(self) -> None:
        with self.assertRaisesRegex(ValueError, "route_projection_ambiguous"):
            native_route_target_with_report(
                [[-10, 0], [0, 0], [10, 0], [10, 10], [0, 0], [0, -10]],
                ego_x=0.0, ego_y=0.0, ego_yaw=0.0,
            )

    def test_exact_19_metres_supports_twenty_points_without_provenance_claim(self) -> None:
        _, mask, report = native_route_target_with_report(
            [[0, 0], [19, 0]], ego_x=0.0, ego_y=0.0, ego_yaw=0.0,
        )
        self.assertEqual(int(mask.sum()), 20)
        self.assertEqual(report["support_m"], 19.0)
        self.assertEqual(report["source"], "ordered_polyline_geometry_only")

    def test_speed_sampling_requires_native_time_grid_and_keeps_missing(self) -> None:
        trajectory = [
            {"t": 0.25 * (i + 1), "x": float(i + 1), "y": 0.0}
            for i in range(9)
        ]
        target, mask, failures = native_speed_target(
            trajectory,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
        )
        self.assertEqual(mask.tolist(), [True] * 9 + [False])
        self.assertIn("trajectory_short:9<10", failures)
        self.assertIn("missing_step:9", failures)
        np.testing.assert_allclose(target[:2], [[1.0, 0.0], [2.0, 0.0]])

    def test_execution_timeline_sampling_masks_early_terminal_tail(self) -> None:
        timeline = [
            {"tick": i, "simulation_time_s": 10.0 + 0.05 * i, "x": 0.25, "y": 0.0}
            for i in range(5)
        ]
        target, mask, failures = native_speed_target_from_timeline(
            timeline,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
        )
        self.assertEqual(mask.tolist(), [True] + [False] * 9)
        np.testing.assert_allclose(target[0], [0.25, 0.0])
        self.assertIn("missing_execution_step:9", failures)

    def test_map_to_ego_is_inverse_rotation(self) -> None:
        result = map_to_ego([[1.0, 3.0]], ego_x=1.0, ego_y=1.0, ego_yaw=math.pi / 2.0)
        np.testing.assert_allclose(result, [[2.0, 0.0]], atol=1e-8)

    def test_deterministic_order_is_repeatable_without_replacement(self) -> None:
        samples = [_sample(f"r{i}", split="train") for i in range(12)]
        first = [sample.root_id for sample in deterministic_order(samples, seed=17)]
        second = [sample.root_id for sample in deterministic_order(samples, seed=17)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))


class C3MetricTests(unittest.TestCase):
    def test_root_metrics_honours_masks_and_finite_predictions(self) -> None:
        sample = _sample("root")
        route = np.asarray(sample.route_target, dtype=float)
        speed = np.asarray(sample.speed_target, dtype=float)
        route[3] += 1.0
        speed[4] += 2.0
        route[7, 0] = np.nan
        row = root_metrics({"route": route, "speed": speed}, sample)
        self.assertEqual(row["route_valid_points"], 20)
        self.assertEqual(row["route_scored_finite_points"], 19)
        self.assertEqual(row["speed_valid_points"], 10)
        self.assertFalse(row["route_finite"])
        self.assertFalse(row["prediction_valid"])
        self.assertIn("route_non_finite", row["failure_reasons"])
        self.assertIsNone(row["route_ade_m"])
        self.assertIsNotNone(row["route_ade_conditional_m"])

    def test_failed_point_cannot_improve_primary_ade_by_shrinking_support(self) -> None:
        sample = _sample("root")
        route = np.asarray(sample.route_target, dtype=float)
        route[5, 0] += 100
        finite = root_metrics({"route": route, "speed": sample.speed_target}, sample)
        self.assertEqual(finite["route_ade_m"], 5.0)
        route[5, 0] = np.nan
        failed = root_metrics({"route": route, "speed": sample.speed_target}, sample)
        self.assertIsNone(failed["route_ade_m"])
        self.assertEqual(failed["route_valid_points"], 20)
        self.assertEqual(failed["route_ade_conditional_m"], 0.0)
        result = aggregate_metrics([{**finite, "model": "m0"}, {**failed, "model": "m1"}])
        self.assertIsNone(result["metrics"]["route_ade_m"]["delta_m0_minus_m1"])
        self.assertEqual(result["metrics"]["route_ade_m"]["prediction_failed_root_count"]["m1"], 1)

    def test_missing_truth_is_distinct_from_prediction_failure(self) -> None:
        sample = SFTSample(**{**_sample("root").to_dict(), "route_mask": (False,) * 20, "speed_mask": (False,) * 10})
        row = root_metrics({"route": sample.route_target, "speed": sample.speed_target, "canonical_output_valid": True}, sample)
        self.assertTrue(row["prediction_valid"])
        self.assertIsNone(row["route_ade_m"])
        self.assertEqual(row["failure_reasons"], [])

    def test_multiple_anchors_do_not_increase_root_weight(self) -> None:
        rows = [{"root_id": "a", "model": "m1", "route_ade_m": 0.0, "prediction_valid": True}] * 9
        rows.append({"root_id": "b", "model": "m1", "route_ade_m": 10.0, "prediction_valid": True})
        result = aggregate_metrics(rows)["metrics"]
        self.assertEqual(result["route_ade_m"]["m1"], 5.0)
        self.assertEqual(result["root_count"]["m1"], 2)
        self.assertEqual(result["sample_count"]["m1"], 10)

    def test_route_fde_is_fixed_point_twenty_only(self) -> None:
        sample = _sample("root")
        route = np.asarray(sample.route_target, dtype=float)
        route[-1] += 2.0
        route[-2] += 10.0
        row = root_metrics({"route": route, "speed": np.asarray(sample.speed_target)}, sample)
        self.assertAlmostEqual(row["route_fde_m"], math.sqrt(8.0))
        sample_short = SFTSample(**{**sample.to_dict(), "route_mask": (True,) * 19 + (False,)})
        short_row = root_metrics({"route": route, "speed": np.asarray(sample.speed_target)}, sample_short)
        self.assertIsNone(short_row["route_fde_m"])
        self.assertFalse(short_row["route_fde_target_valid"])

    def test_aggregate_is_root_equal_and_bootstrap_is_seeded(self) -> None:
        rows = [
            {"root_id": "a", "model": "m0", "route_ade_m": 2.0, "route_fde_m": 2.0, "speed_wp_ade_m": 1.0, "prediction_valid": True},
            {"root_id": "b", "model": "m0", "route_ade_m": 4.0, "route_fde_m": 4.0, "speed_wp_ade_m": 3.0, "prediction_valid": True},
            {"root_id": "a", "model": "m1", "route_ade_m": 1.0, "route_fde_m": 1.0, "speed_wp_ade_m": 0.5, "prediction_valid": True},
            {"root_id": "b", "model": "m1", "route_ade_m": 3.0, "route_fde_m": 3.0, "speed_wp_ade_m": 2.0, "prediction_valid": True},
        ]
        result = aggregate_metrics(rows, bootstrap_rounds=1000, bootstrap_seed=71)
        route = result["metrics"]["route_ade_m"]
        self.assertEqual(route["m0"], 3.0)
        self.assertEqual(route["m1"], 2.0)
        self.assertEqual(route["delta_m0_minus_m1"], 1.0)
        self.assertEqual(route["paired_root_count"], 2)
        self.assertIsNotNone(route["bootstrap_ci95"])
        self.assertEqual(result["metrics"]["root_count"], {"m0": 2, "m1": 2})


class C3VerifierTests(unittest.TestCase):
    def test_changed_stage_arguments_are_rejected_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = c3_sft._parser().parse_args(["smoke", "--run-id", "c3-repair-test", "--output-root", temporary, "--device", "cuda"])
            run = c3_sft._ensure_run_dir(args)
            config = c3_sft._config_payload(args, model_identity={"path": str(Path(args.checkpoint).resolve())})
            c3_sft._write_json(run / "run_config.json", config)
            self.assertEqual(c3_sft._validate_run_config(run, args), config)
            for key, replacement in (("checkpoint", "/nonexistent/model.pt"), ("hydra_config", "/nonexistent/config.yaml"), ("internvl_root", "/nonexistent/internvl"), ("seed", 999), ("updates", 1), ("max_hours", 400)):
                with self.subTest(argument=key), patch.object(args, key, replacement):
                    with self.assertRaisesRegex(ValueError, "argument_conflict"):
                        c3_sft._validate_run_config(run, args)

    def test_redigested_configuration_does_not_override_stage_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = c3_sft._parser().parse_args(["verify", "--run-id", "c3-repair-test", "--output-root", temporary, "--device", "cpu"])
            run = c3_sft._ensure_run_dir(args)
            config = c3_sft._config_payload(args, model_identity={"path": str(Path(args.checkpoint).resolve())})
            config["updates"] = 1
            config.pop("content_sha256")
            config["content_sha256"] = c3_sft._sha_object(config)
            c3_sft._write_json(run / "run_config.json", config)
            with self.assertRaisesRegex(ValueError, "argument_conflict:updates"):
                c3_sft._validate_run_config(run, args)

    def test_smoke_success_string_cannot_hide_bad_measurements(self) -> None:
        smoke = {"status": "SMOKE_PASSED", "finite_grad": False, "lora_changed": False,
                 "driving_head_changed": True, "frozen_parameters_unchanged": True,
                 "nonzero_gradient_elements": 0, "checkpoint_roundtrip_max_abs": 100.0,
                 "checkpoint_roundtrip_tolerance": 1e-5}
        smoke["smoke_sha256"] = c3_sft._sha_object(smoke)
        errors = c3_sft._smoke_contract_errors(smoke, {"roundtrip_tolerance": 1e-5})
        self.assertIn("smoke_finite_grad", errors)
        self.assertIn("smoke_lora_changed", errors)
        self.assertIn("smoke_no_nonzero_gradient", errors)
        self.assertIn("smoke_roundtrip_measurement", errors)

    def test_training_log_is_bound_to_redundant_summary(self) -> None:
        rows = [{"update": 1, "epoch": 0}, {"update": 2, "epoch": 0}]
        summary = {"logs": rows}
        self.assertTrue(c3_sft._training_log_matches_summary(rows, summary))
        self.assertFalse(c3_sft._training_log_matches_summary(rows[:-1], summary))
        self.assertFalse(c3_sft._training_log_matches_summary(rows, {"logs": rows[:-1]}))


class C3TorchHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch unavailable in the lightweight test interpreter")

    def test_masked_loss_excludes_invalid_coordinates(self) -> None:
        import torch

        prediction = torch.tensor([[[0.0, 0.0], [100.0, 100.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(masked_smooth_l1(prediction, target, mask)), 0.0)
        with self.assertRaisesRegex(ValueError, "masked_loss_no_valid_elements"):
            masked_smooth_l1(prediction, target, torch.zeros_like(mask))

    def test_schedule_freezes_then_unfreezes_lora_and_decays_to_ten_percent(self) -> None:
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.lora_A = torch.nn.Parameter(torch.ones(2, 2))
                self.adaptors = torch.nn.Module()
                self.adaptors.driving = torch.nn.Module()
                self.adaptors.driving.route_head = torch.nn.Linear(2, 2)
                self.adaptors.driving.speed_wps_head = torch.nn.Linear(2, 2)
                self.vision_model = torch.nn.Linear(2, 2)
                self.wp_encoder = torch.nn.Linear(2, 2)

        model = Tiny()
        lora, heads = c3_sft._freeze_trainable(model)
        optimizer = torch.optim.AdamW(
            [
                {"name": "lora", "params": [p for _, p in lora], "lr": 0.0},
                {"name": "head", "params": [p for _, p in heads], "lr": 0.0},
            ]
        )
        first = c3_sft._set_schedule(optimizer, 1, total=200, head_lr=1e-4, lora_lr=2e-5, warmup=10, lora=lora)
        self.assertEqual(first["lora_lr"], 0.0)
        self.assertTrue(all(not parameter.requires_grad for _, parameter in lora))
        middle = c3_sft._set_schedule(optimizer, 20, total=200, head_lr=1e-4, lora_lr=2e-5, warmup=10, lora=lora)
        self.assertAlmostEqual(middle["lora_lr"], 2e-5)
        self.assertTrue(all(parameter.requires_grad for _, parameter in lora))
        final = c3_sft._set_schedule(optimizer, 200, total=200, head_lr=1e-4, lora_lr=2e-5, warmup=10, lora=lora)
        self.assertAlmostEqual(final["lora_lr"], 2e-6)
        self.assertAlmostEqual(final["head_lr"], 1e-5)
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("vision_model.") or name.startswith("wp_encoder.")))

    def test_resume_checkpoint_restores_whitelisted_parameters_and_optimizer(self) -> None:
        import tempfile
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.lora_A = torch.nn.Parameter(torch.ones(2, 2))
                self.adaptors = torch.nn.Module()
                self.adaptors.driving = torch.nn.Module()
                self.adaptors.driving.route_head = torch.nn.Linear(2, 2)
                self.adaptors.driving.speed_wps_head = torch.nn.Linear(2, 2)
                self.vision_model = torch.nn.Linear(2, 2)
                self.wp_encoder = torch.nn.Linear(2, 2)

        model = Tiny()
        lora, heads = c3_sft._freeze_trainable(model)
        optimizer = torch.optim.AdamW([p for _, p in lora + heads], lr=1e-3)
        loss = sum(parameter.float().sum() for _, parameter in lora + heads)
        loss.backward()
        optimizer.step()
        expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if (name.startswith("language_model.") and "lora_" in name) or name.startswith("adaptors.driving.route_head.") or name.startswith("adaptors.driving.speed_wps_head.")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            c3_sft._save_training_checkpoint(path, model, optimizer, update=4, sample_index=9, epoch=1, log=[])
            for name, parameter in model.named_parameters():
                if name in expected:
                    parameter.data.add_(10.0)
            restored_update, restored_index, restored_epoch, _ = c3_sft._restore_training_checkpoint(path, model, optimizer)
        self.assertEqual((restored_update, restored_index, restored_epoch), (4, 9, 1))
        for name, value in expected.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], value)


if __name__ == "__main__":
    unittest.main()
