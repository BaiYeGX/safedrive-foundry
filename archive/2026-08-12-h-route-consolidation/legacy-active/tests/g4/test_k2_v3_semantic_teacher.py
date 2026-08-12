from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.k2_v3_guard import attach_k2_v3_guard  # noqa: E402
from driving_vla.model.k2_v3_types import AlternativeKind  # noqa: E402
from driving_vla.model.navigation_contract import (  # noqa: E402
    LaneAccessV3,
    RouteManeuver,
    TargetLaneSide,
    TrafficSignalState,
    build_route_context,
)
from driving_vla.model.semantic_k2_teacher import (  # noqa: E402
    build_semantic_teacher_bundle_v3,
    select_semantic_teacher_v3,
)
from driving_vla.runtime.basic1v1_observable import (  # noqa: E402
    basic1v1_conflict_active,
    conflict_side_from_scene,
    observe_basic1v1_actor,
)
from driving_vla.model.semantic_mode_heads import (  # noqa: E402
    KIND_ORDER,
    SEMANTIC_CONTEXT_DIM,
    SIDE_ORDER,
    SpatialSemanticHeadRuntimeV3,
    SpatialSemanticHeadV3,
    navigation_context_vector,
)
from driving_vla.evaluation.k2_v3_artifact import (  # noqa: E402
    artifact_json_bytes_v3,
    bundle_from_artifact_json_v3,
)
from driving_vla.runtime.k2_execution import (  # noqa: E402
    K2SelectionError,
    select_k2_semantic_v3,
    selection_event_fields,
)


def _line(y: float = 0.0):
    return tuple((float(index), y) for index in range(40))


def _access(side: TargetLaneSide, y: float, *, clear: bool = True):
    return LaneAccessV3(
        side=side,
        exists=True,
        driving=True,
        same_direction=True,
        lane_change_allowed=True,
        currently_clear=clear,
        road_id=1,
        lane_id=2,
        lane_width_m=3.5,
        centerline_xy=_line(y),
        marking_type="Broken",
    )


class _AlwaysNoneHead(SpatialSemanticHeadV3):
    """Deterministic head used to exercise the observable fallback only."""

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(context.shape[0])
        kind = torch.zeros(
            (batch, len(KIND_ORDER)), dtype=context.dtype, device=context.device
        )
        kind[:, KIND_ORDER.index(AlternativeKind.NONE)] = 10.0
        side = torch.zeros(
            (batch, len(SIDE_ORDER)), dtype=context.dtype, device=context.device
        )
        side[:, SIDE_ORDER.index(TargetLaneSide.NONE)] = 10.0
        return {
            "kind_logits": kind,
            "side_logits": side,
            "avail_logit": torch.full(
                (batch,), -10.0, dtype=context.dtype, device=context.device
            ),
            "raw_delta_s": torch.zeros(
                (batch, self.n_path), dtype=context.dtype, device=context.device
            ),
            "raw_d": torch.zeros(
                (batch, self.n_path), dtype=context.dtype, device=context.device
            ),
            "maneuver_params": torch.full(
                (batch, 6), 0.5, dtype=context.dtype, device=context.device
            ),
        }


class SemanticTeacherV3Test(unittest.TestCase):
    def test_live_actor_observable_uses_carla_physical_left_semantics(self) -> None:
        def actor(actor_id: int, x: float, y: float):
            return SimpleNamespace(
                id=actor_id,
                get_transform=lambda: SimpleNamespace(
                    location=SimpleNamespace(x=x, y=y),
                    rotation=SimpleNamespace(yaw=0.0),
                ),
                get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0),
            )

        ego = actor(1, 0.0, 0.0)
        left = observe_basic1v1_actor(
            ego=ego,
            actors=[ego, actor(2, 8.0, -2.0)],
        )
        right = observe_basic1v1_actor(
            ego=ego,
            actors=[ego, actor(3, 8.0, 2.0)],
        )
        self.assertGreater(float(left.actor_lat_m), 0.0)
        self.assertEqual(conflict_side_from_scene(left), "left")
        self.assertLess(float(right.actor_lat_m), 0.0)
        self.assertEqual(conflict_side_from_scene(right), "right")

    def test_current_observable_conflict_clears_and_teacher_resumes(self) -> None:
        active_scene = {
            "actor_present": True,
            "actor_lon_m": 8.0,
            "actor_lat_m": 0.1,
            "actor_speed_mps": 0.0,
            "current_ttc_s": 2.0,
            "distance_m": 8.0,
        }
        cleared_scene = {
            **active_scene,
            "actor_lon_m": -5.0,
            "distance_m": 5.0,
        }
        self.assertTrue(
            basic1v1_conflict_active(
                scenario_family="lead_braking", scene=active_scene
            )
        )
        self.assertFalse(
            basic1v1_conflict_active(
                scenario_family="lead_braking", scene=cleared_scene
            )
        )
        label = select_semantic_teacher_v3(
            scenario_family="lead_braking",
            route_context=build_route_context(_line()),
            conflict_active=False,
        )
        self.assertEqual(label.alternative_kind, AlternativeKind.NONE)
        self.assertEqual(label.behavior, "CLEAR")

    def test_cut_in_side_comes_from_current_actor_state(self) -> None:
        scene = {
            "actor_present": True,
            "actor_lon_m": 10.0,
            "actor_lat_m": 2.0,
            "actor_speed_mps": 4.0,
            "current_ttc_s": 4.0,
            "distance_m": 10.2,
        }
        self.assertEqual(conflict_side_from_scene(scene), "left")
        self.assertTrue(
            basic1v1_conflict_active(scenario_family="cut_in", scene=scene)
        )

    def test_clear_is_none(self) -> None:
        label = select_semantic_teacher_v3(
            scenario_family="clear",
            route_context=build_route_context(_line()),
        )
        self.assertFalse(label.alternative_available)
        self.assertEqual(label.alternative_kind, AlternativeKind.NONE)

    def test_stopped_nominal_saturates_temporal_alternative_to_none(self) -> None:
        label = select_semantic_teacher_v3(
            scenario_family="crossing",
            route_context=build_route_context(_line()),
            conflict_active=True,
        )
        bundle = build_semantic_teacher_bundle_v3(
            native_path_xy=_line(),
            route_context=build_route_context(_line()),
            label=label,
            ego_v=0.2,
            base_speed_mps=0.5,
            backbone_forward_id="stopped-yield",
        )
        alternative = bundle.candidates[1]
        self.assertEqual(alternative.alternative_kind, AlternativeKind.NONE)
        self.assertFalse(alternative.available)
        self.assertEqual(
            alternative.availability_reason,
            "NOMINAL_ALREADY_MINIMAL_INTERVENTION",
        )

    def test_red_light_yield_stays_open_until_nominal_is_guard_safe(self) -> None:
        route = build_route_context(
            _line(),
            traffic_signal_state=TrafficSignalState.RED,
            stop_line_distance_m=2.0,
        )
        label = select_semantic_teacher_v3(
            scenario_family="traffic_control",
            route_context=route,
        )
        bundle = attach_k2_v3_guard(
            build_semantic_teacher_bundle_v3(
                native_path_xy=_line(),
                route_context=route,
                label=label,
                ego_v=0.2,
                base_speed_mps=0.5,
                backbone_forward_id="red-light-yield",
            )
        )
        self.assertEqual(
            bundle.candidates[1].alternative_kind,
            AlternativeKind.TEMPORAL_YIELD,
        )
        self.assertTrue(bundle.candidates[1].available)
        selection = select_k2_semantic_v3(
            bundle, mode="force", force_index=1
        )
        self.assertEqual(selection.candidate_index, 1)

    def test_cut_in_moves_away_from_conflict(self) -> None:
        route = build_route_context(
            _line(),
            right_lane=_access(TargetLaneSide.RIGHT, 3.5),
        )
        left_conflict = select_semantic_teacher_v3(
            scenario_family="cut_in",
            conflict_side="left",
            route_context=route,
        )
        self.assertEqual(
            left_conflict.alternative_kind, AlternativeKind.SPATIAL_AVOID
        )
        self.assertEqual(left_conflict.target_lane_side, TargetLaneSide.RIGHT)
        self.assertAlmostEqual(left_conflict.avoid_offset_m, 0.55)

    def test_cut_in_at_edge_lane_yields_when_away_side_is_not_legal(self) -> None:
        label = select_semantic_teacher_v3(
            scenario_family="cut_in",
            conflict_side="left",
            route_context=build_route_context(_line()),
        )
        self.assertEqual(label.alternative_kind, AlternativeKind.TEMPORAL_YIELD)
        self.assertEqual(label.target_lane_side, TargetLaneSide.NONE)
        self.assertEqual(
            label.availability_reason,
            "CUT_IN_NO_LEGAL_AWAY_SIDE_YIELD",
        )

    def test_junction_cut_in_yields_instead_of_swerve(self) -> None:
        label = select_semantic_teacher_v3(
            scenario_family="cut_in",
            conflict_side="left",
            route_context=build_route_context(
                _line(),
                maneuver=RouteManeuver.JUNCTION_STRAIGHT,
                junction_flags=[True] * 40,
            ),
        )
        self.assertEqual(label.alternative_kind, AlternativeKind.TEMPORAL_YIELD)

    def test_obstruction_overtakes_only_on_authorized_lane(self) -> None:
        authorized = build_route_context(
            _line(),
            left_lane=_access(TargetLaneSide.LEFT, 3.5),
        )
        label = select_semantic_teacher_v3(
            scenario_family="obstruction",
            route_context=authorized,
            requested_overtake_side=TargetLaneSide.LEFT,
        )
        self.assertEqual(label.alternative_kind, AlternativeKind.SPATIAL_OVERTAKE)
        bundle = attach_k2_v3_guard(
            build_semantic_teacher_bundle_v3(
                native_path_xy=_line(),
                route_context=authorized,
                label=label,
                ego_v=4.0,
                base_speed_mps=5.0,
                backbone_forward_id="teacher-forward",
            )
        )
        self.assertEqual(bundle.guard_status, "OK", bundle.guard_reasons)

        blocked = build_route_context(
            _line(),
            left_lane=_access(TargetLaneSide.LEFT, 3.5, clear=False),
        )
        blocked_label = select_semantic_teacher_v3(
            scenario_family="obstruction",
            route_context=blocked,
            requested_overtake_side=TargetLaneSide.LEFT,
        )
        self.assertEqual(
            blocked_label.alternative_kind, AlternativeKind.TEMPORAL_YIELD
        )

    def test_semantic_head_shapes_and_route_input(self) -> None:
        model = SpatialSemanticHeadV3()
        output = model(torch.zeros((3, SEMANTIC_CONTEXT_DIM)))
        self.assertEqual(tuple(output["kind_logits"].shape), (3, len(KIND_ORDER)))
        self.assertEqual(tuple(output["side_logits"].shape), (3, len(SIDE_ORDER)))
        self.assertEqual(tuple(output["raw_d"].shape), (3, 20))
        self.assertEqual(tuple(output["maneuver_params"].shape), (3, 6))

        straight = build_route_context(_line())
        left_turn = build_route_context(
            _line(),
            maneuver=RouteManeuver.TURN_LEFT,
            junction_flags=[True] * 40,
        )
        self.assertNotEqual(
            navigation_context_vector(straight),
            navigation_context_vector(left_turn),
        )

    def test_runtime_head_closes_clear_and_opens_observable_obstruction(self) -> None:
        route = build_route_context(
            _line(),
            left_lane=_access(TargetLaneSide.LEFT, 3.5),
        )
        runtime = SpatialSemanticHeadRuntimeV3(model=_AlwaysNoneHead())
        common = {
            "native_path_xy": _line(),
            "route_context": route,
            "ego_v": 4.0,
            "base_speed_mps": 5.0,
            "driving_feature": [0.0] * 64,
            "observation_identity": {"frame_id": "fallback-test"},
            "backbone_forward_id": "fallback-forward",
            "base_checkpoint_hash": "base",
            "spatial_head_checkpoint_hash": "head",
        }
        clear = runtime.build_bundle(
            **common,
            observable_scene={"actor_present": False},
            scenario_family="clear",
        )
        self.assertEqual(clear.candidates[1].alternative_kind, AlternativeKind.NONE)
        self.assertFalse(clear.candidates[1].available)

        obstruction = runtime.build_bundle(
            **common,
            observable_scene={
                "actor_present": True,
                "actor_lon_m": 10.0,
                "actor_lat_m": 0.0,
                "actor_speed_mps": 0.0,
            },
            scenario_family="obstruction",
            requested_overtake_side=TargetLaneSide.LEFT,
        )
        self.assertEqual(
            obstruction.candidates[1].alternative_kind,
            AlternativeKind.SPATIAL_OVERTAKE,
        )
        self.assertTrue(obstruction.candidates[1].available)
        self.assertEqual(
            obstruction.candidates[1].target_lane_side,
            TargetLaneSide.LEFT,
        )

        cut_in_route = build_route_context(
            _line(),
            right_lane=_access(TargetLaneSide.RIGHT, 3.5),
        )
        cut_in = runtime.build_bundle(
            **{**common, "route_context": cut_in_route},
            observable_scene={
                "actor_present": True,
                "actor_lon_m": 12.0,
                "actor_lat_m": -2.5,
                "actor_speed_mps": 2.5,
            },
            conflict_side="LEFT",
            scenario_family="cut_in",
        )
        self.assertEqual(
            cut_in.candidates[1].alternative_kind,
            AlternativeKind.SPATIAL_AVOID,
        )
        self.assertTrue(cut_in.candidates[1].available)
        self.assertEqual(
            cut_in.candidates[1].target_lane_side,
            TargetLaneSide.RIGHT,
        )

        # At the edge lane there may be no legal adjacent away lane.  The
        # interaction must then yield in place rather than crossing a solid
        # or opposite-direction boundary.
        edge_route = build_route_context(
            _line(),
            left_lane=LaneAccessV3(
                side=TargetLaneSide.LEFT,
                exists=True,
                driving=False,
                same_direction=False,
                lane_change_allowed=True,
                currently_clear=True,
                lane_width_m=3.5,
                centerline_xy=_line(3.5),
            ),
        )
        edge_cut_in = runtime.build_bundle(
            **{**common, "route_context": edge_route},
            observable_scene={
                "actor_present": True,
                "actor_lon_m": 12.0,
                "actor_lat_m": -2.5,
                "actor_speed_mps": 2.5,
            },
            conflict_side="RIGHT",
            scenario_family="cut_in",
        )
        self.assertEqual(
            edge_cut_in.candidates[1].alternative_kind,
            AlternativeKind.TEMPORAL_YIELD,
        )
        self.assertTrue(edge_cut_in.candidates[1].available)
        self.assertEqual(
            edge_cut_in.candidates[1].target_lane_side,
            TargetLaneSide.NONE,
        )
        self.assertEqual(
            edge_cut_in.candidates[1].availability_reason,
            "SCENARIO_CUT_IN_NO_LEGAL_AWAY_YIELD",
        )

        crossing = runtime.build_bundle(
            **common,
            observable_scene={
                "actor_present": True,
                "actor_lon_m": 14.0,
                "actor_lat_m": -2.0,
                "actor_speed_mps": 3.0,
            },
            conflict_side="LEFT",
            scenario_family="crossing",
        )
        self.assertEqual(
            crossing.candidates[1].alternative_kind,
            AlternativeKind.TEMPORAL_YIELD,
        )
        self.assertTrue(crossing.candidates[1].available)

    def test_v3_artifact_cold_roundtrip_and_runtime_selection(self) -> None:
        route = build_route_context(
            _line(),
            left_lane=_access(TargetLaneSide.LEFT, 3.5),
        )
        label = select_semantic_teacher_v3(
            scenario_family="obstruction",
            route_context=route,
            requested_overtake_side=TargetLaneSide.LEFT,
        )
        guarded = attach_k2_v3_guard(
            build_semantic_teacher_bundle_v3(
                native_path_xy=_line(),
                route_context=route,
                label=label,
                ego_v=4.0,
                base_speed_mps=5.0,
                backbone_forward_id="artifact-forward",
            )
        )
        rebuilt = bundle_from_artifact_json_v3(
            artifact_json_bytes_v3(guarded)
        )
        self.assertEqual(rebuilt.bundle_hash, guarded.bundle_hash)
        selection = select_k2_semantic_v3(
            rebuilt, mode="force", force_index=1
        )
        self.assertEqual(
            selection.execution_spec.alternative_kind,
            AlternativeKind.SPATIAL_OVERTAKE,
        )
        fields = selection_event_fields(selection)
        self.assertEqual(fields["route_hash"], route.route_hash)
        self.assertEqual(fields["topology_hash"], route.topology_hash)

    def test_v3_artifact_tamper_fails_closed(self) -> None:
        route = build_route_context(_line())
        label = select_semantic_teacher_v3(
            scenario_family="clear",
            route_context=route,
        )
        bundle = build_semantic_teacher_bundle_v3(
            native_path_xy=_line(),
            route_context=route,
            label=label,
            ego_v=4.0,
            base_speed_mps=5.0,
            backbone_forward_id="tamper-forward",
        )
        raw = json.loads(artifact_json_bytes_v3(bundle))
        raw["candidates"][0]["spatial_path_xy"][1][1] = 99.0
        with self.assertRaises(ValueError):
            bundle_from_artifact_json_v3(json.dumps(raw))
        guarded = attach_k2_v3_guard(bundle)
        with self.assertRaises(K2SelectionError):
            select_k2_semantic_v3(
                guarded, mode="force", force_index=1
            )


if __name__ == "__main__":
    unittest.main()
