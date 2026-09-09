"""Focused CPU checks for the C3 data adapter and metric contracts."""

from __future__ import annotations

import math
import sys
import unittest
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
        # The inserted ego origin is the first native point even when the
        # recorded route begins behind the car.
        np.testing.assert_allclose(behind, [[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]], atol=1e-8)
        np.testing.assert_array_equal(behind_mask, [True, True, True])

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
        self.assertEqual(row["route_valid_points"], 19)
        self.assertEqual(row["speed_valid_points"], 10)
        self.assertFalse(row["route_finite"])
        self.assertTrue(row["prediction_valid"])
        self.assertIsNotNone(row["route_ade_m"])

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
