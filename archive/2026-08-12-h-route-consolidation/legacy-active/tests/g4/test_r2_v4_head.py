from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.k2_v3_types import AlternativeKind  # noqa: E402
from driving_vla.model.k2_v3_codec import build_k2_v3_bundle  # noqa: E402
from driving_vla.model.k2_v3_guard import attach_k2_v3_guard  # noqa: E402
from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    TargetLaneSide,
    build_route_context,
)
from driving_vla.model.semantic_mode_heads_v4 import (  # noqa: E402
    AUX_DIM,
    SpatialSemanticHeadRuntimeV4,
    SpatialSemanticHeadV4,
    build_v4_aux_vector,
)
from driving_vla.model.v4_lora import (  # noqa: E402
    V4LoRAConfig,
    apply_v4_lora_qv,
)
from driving_vla.model.v4_token_features import (  # noqa: E402
    DrivingTokenBundleV4,
    DrivingTokenError,
    TOTAL_TOKEN_COUNT,
)
from scripts.r2_v4_train_heads import V4Row, _balanced_weights  # noqa: E402


def _line(y: float = 0.0):
    return tuple((float(index), float(y)) for index in range(40))


class R2V4TokenContractTest(unittest.TestCase):
    def test_ordered_tokens_split_and_reload_hash(self) -> None:
        raw = np.arange(1 * TOTAL_TOKEN_COUNT * 8, dtype=np.float32).reshape(
            1, TOTAL_TOKEN_COUNT, 8
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.npy"
            bundle = DrivingTokenBundleV4.from_adaptor_output(
                raw, raw_tensor_path=path
            )
            self.assertTrue(bundle.ok)
            self.assertEqual(bundle.raw_shape, (1, TOTAL_TOKEN_COUNT, 8))
            self.assertEqual(bundle.channel_dim, 8)
            self.assertEqual(len(bundle.raw_content_hash), 64)
            self.assertEqual(bundle.token_type_ids()[:20], (0,) * 20)
            self.assertEqual(bundle.token_type_ids()[20:], (1,) * 10)
            loaded = bundle.load_tokens()
            np.testing.assert_array_equal(loaded, raw[0].astype(np.float32))

    def test_wrong_token_count_fails_closed(self) -> None:
        with self.assertRaises(DrivingTokenError):
            DrivingTokenBundleV4.from_adaptor_output(np.zeros((1, 29, 8)))

    def test_split_metadata_is_immutable_contract(self) -> None:
        raw = np.zeros((1, TOTAL_TOKEN_COUNT, 8), dtype=np.float32)
        bundle = DrivingTokenBundleV4.from_adaptor_output(raw)
        forged = DrivingTokenBundleV4(
            **{
                **bundle.metadata(),
                "raw_shape": tuple(bundle.raw_shape),
                "route_token_count": 19,
            }
        )
        with self.assertRaises(DrivingTokenError):
            forged.require_ok()


class R2V4HeadTest(unittest.TestCase):
    def test_first_repair_class_weights_are_train_only_and_finite(self) -> None:
        rows = [
            V4Row(None, np.zeros((TOTAL_TOKEN_COUNT, 8), dtype=np.float32), (0.0,) * AUX_DIM, 0, 0, 0.0, (0.0,) * 6, "train", "g0", "r0"),
            V4Row(None, np.zeros((TOTAL_TOKEN_COUNT, 8), dtype=np.float32), (0.0,) * AUX_DIM, 0, 0, 0.0, (0.0,) * 6, "train", "g1", "r1"),
            V4Row(None, np.zeros((TOTAL_TOKEN_COUNT, 8), dtype=np.float32), (0.0,) * AUX_DIM, 1, 1, 1.0, (0.0,) * 6, "train", "g2", "r2"),
            V4Row(None, np.zeros((TOTAL_TOKEN_COUNT, 8), dtype=np.float32), (0.0,) * AUX_DIM, 2, 2, 1.0, (0.0,) * 6, "train", "g3", "r3"),
            V4Row(None, np.zeros((TOTAL_TOKEN_COUNT, 8), dtype=np.float32), (0.0,) * AUX_DIM, 3, 0, 1.0, (0.0,) * 6, "train", "g4", "r4"),
        ]
        weights = _balanced_weights(rows, field="kind", classes=4)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(tuple(weights.shape), (4,))
        self.assertLess(float(weights[0]), float(weights[1]))

    def test_aux_has_fixed_observable_abi(self) -> None:
        route = build_route_context(_line())
        aux = build_v4_aux_vector(
            native_path_xy=_line(),
            route_context=route,
            ego_v=4.0,
            base_speed_mps=5.0,
            observable_scene={
                "actor_present": True,
                "actor_lon_m": 8.0,
                "actor_lat_m": 1.0,
                "actor_speed_mps": 2.0,
            },
        )
        self.assertEqual(len(aux), AUX_DIM)
        self.assertTrue(np.isfinite(np.asarray(aux)).all())

    def test_model_forward_and_runtime_are_scenario_blind(self) -> None:
        model = SpatialSemanticHeadV4(token_dim=8)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 300_000)
        self.assertLess(sum(p.numel() for p in model.parameters()), 1_500_000)
        tokens = torch.randn(2, TOTAL_TOKEN_COUNT, 8)
        aux = torch.randn(2, AUX_DIM)
        out = model(tokens, aux)
        self.assertEqual(tuple(out["kind_logits"].shape), (2, 4))
        self.assertEqual(tuple(out["side_logits"].shape), (2, 3))
        self.assertEqual(tuple(out["maneuver_params"].shape), (2, 6))
        self.assertTrue(torch.isfinite(out["kind_logits"]).all())

        with tempfile.TemporaryDirectory() as tmp:
            raw = np.random.RandomState(4).randn(1, TOTAL_TOKEN_COUNT, 8).astype(
                np.float32
            )
            bundle = DrivingTokenBundleV4.from_adaptor_output(
                raw, raw_tensor_path=Path(tmp) / "raw.npy"
            )
            runtime = SpatialSemanticHeadRuntimeV4(model)
            output = runtime.predict(
                token_bundle=bundle,
                native_path_xy=_line(),
                route_context=build_route_context(_line()),
                ego_v=3.0,
                base_speed_mps=4.0,
                observable_scene={"actor_present": False},
            )
            self.assertIn(
                output.alternative_kind,
                {
                    AlternativeKind.NONE,
                    AlternativeKind.SPATIAL_AVOID,
                    AlternativeKind.SPATIAL_OVERTAKE,
                    AlternativeKind.TEMPORAL_YIELD,
                },
            )
            self.assertIn(
                output.target_lane_side,
                {TargetLaneSide.NONE, TargetLaneSide.LEFT, TargetLaneSide.RIGHT},
            )
            self.assertEqual(len(output.maneuver_params), 6)
            self.assertEqual(len(output.raw_head_output_hash), 64)

    def test_side_is_spatial_only(self) -> None:
        class FixedHead(SpatialSemanticHeadV4):
            def forward(self, tokens, aux):  # type: ignore[no-untyped-def]
                batch = tokens.shape[0]
                kind = torch.full((batch, 4), -10.0, device=tokens.device)
                kind[:, 0] = 10.0  # NONE
                side = torch.full((batch, 3), -10.0, device=tokens.device)
                side[:, 2] = 10.0  # RIGHT must be ignored for NONE
                return {
                    "kind_logits": kind,
                    "side_logits": side,
                    "avail_logit": torch.full((batch,), -10.0, device=tokens.device),
                    "maneuver_params": torch.full(
                        (batch, 6), 0.5, device=tokens.device
                    ),
                }

        with tempfile.TemporaryDirectory() as tmp:
            bundle = DrivingTokenBundleV4.from_adaptor_output(
                np.zeros((1, TOTAL_TOKEN_COUNT, 8), dtype=np.float32),
                raw_tensor_path=Path(tmp) / "raw.npy",
            )
            output = SpatialSemanticHeadRuntimeV4(FixedHead(token_dim=8)).predict(
                token_bundle=bundle,
                native_path_xy=_line(),
                route_context=build_route_context(_line()),
                ego_v=1.0,
                base_speed_mps=1.0,
                observable_scene={"actor_present": False},
            )
            self.assertEqual(output.alternative_kind, AlternativeKind.NONE)
            self.assertEqual(output.target_lane_side, TargetLaneSide.NONE)

    def test_bundle_preserves_raw_kind_side_and_has_no_rescue(self) -> None:
        model = SpatialSemanticHeadV4(token_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.random.RandomState(8).randn(1, TOTAL_TOKEN_COUNT, 8).astype(
                np.float32
            )
            bundle = DrivingTokenBundleV4.from_adaptor_output(
                raw, raw_tensor_path=Path(tmp) / "raw.npy"
            )
            runtime = SpatialSemanticHeadRuntimeV4(model)
            route = build_route_context(_line())
            decoded = runtime.build_bundle(
                token_bundle=bundle,
                native_path_xy=_line(),
                route_context=route,
                ego_v=2.0,
                base_speed_mps=4.0,
                observable_scene={"actor_present": False},
                observation_identity={},
                backbone_forward_id="forward-v4",
                base_checkpoint_hash="base",
                semantic_head_checkpoint_hash="head",
            )
            self.assertEqual(decoded.model_id, "sdf-k2-v4@semantic-head")
            self.assertEqual(decoded.candidates[1].head_lineage, "spatial_mode_head_v4")
            self.assertEqual(decoded.observation_identity["semantic_rescue_count"], 0)
            self.assertEqual(decoded.observation_identity["scenario_family_runtime_use"], 0)
            guarded = attach_k2_v3_guard(decoded)
            self.assertEqual(guarded.observation_identity["semantic_rescue_count"], 0)

    def test_runtime_rejects_family_label_even_when_called_directly(self) -> None:
        model = SpatialSemanticHeadV4(token_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = DrivingTokenBundleV4.from_adaptor_output(
                np.zeros((1, TOTAL_TOKEN_COUNT, 8), dtype=np.float32),
                raw_tensor_path=Path(tmp) / "raw.npy",
            )
            with self.assertRaises(ValueError):
                SpatialSemanticHeadRuntimeV4(model).predict(
                    token_bundle=bundle,
                    native_path_xy=_line(),
                    route_context=build_route_context(_line()),
                    ego_v=1.0,
                    base_speed_mps=1.0,
                    observable_scene={"family": "cut_in"},
                )

    def test_invalid_spatial_side_fails_closed_without_codec_crash(self) -> None:
        route = build_route_context(_line())
        bundle = build_k2_v3_bundle(
            native_path_xy=_line(),
            route_context=route,
            ego_v=2.0,
            base_speed_mps=4.0,
            alternative_kind=AlternativeKind.SPATIAL_OVERTAKE,
            alternative_available=True,
            alternative_reason="head",
            target_lane_side=TargetLaneSide.NONE,
            backbone_forward_id="forward-v4",
        )
        self.assertFalse(bundle.candidates[1].available)
        self.assertEqual(bundle.candidates[1].availability_reason, "HEAD_INVALID_SPATIAL_SIDE")
        self.assertEqual(bundle.candidates[1].target_lane_side, TargetLaneSide.NONE)

    def test_lora_fallback_is_exactly_qv_and_base_frozen(self) -> None:
        model = SpatialSemanticHeadV4(token_dim=8)
        install = apply_v4_lora_qv(model, V4LoRAConfig())
        self.assertEqual(install["installed_blocks"], [0, 1])
        self.assertEqual(install["trainable_parameters"], 8192)
        self.assertTrue(install["base_frozen"])
        tokens = torch.randn(2, TOTAL_TOKEN_COUNT, 8)
        aux = torch.randn(2, AUX_DIM)
        model.train()
        output = model(tokens, aux)
        sum(value.sum() for value in output.values()).backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if "q_lora" not in name and "v_lora" not in name
            )
        )


if __name__ == "__main__":
    unittest.main()
