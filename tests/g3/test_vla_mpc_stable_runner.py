from __future__ import annotations

import math
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "g3"))
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.simlingo_contract import (  # noqa: E402
    SimLingoContractConfig,
    legacy_contract_config,
)
from run_g3_vla_mpc_stable import (  # noqa: E402
    DEFAULT_RANDOM_MAP_POOL,
    DEFAULT_RHI,
    LARGE_MAPS,
    CollisionEpisodeBook,
    _camera_sensor_tick_s,
    _capture_recent_carla_crash,
    _classify_runtime_failure,
    _mode_runs_forward,
    _navigation_targets,
    _path_xy_as_lists,
    _polyline_s,
    _tail_motion_metrics,
    _xml_find_text_recursive,
)
from driving_vla.runtime.vehicle_geometry import (  # noqa: E402
    DEFAULT_EGO_BLUEPRINT,
    estimate_wheelbase_track_m,
)
from driving_vla.runtime.lane_evidence import (  # noqa: E402
    LaneInvasionEpisodeBook,
    multi_deadband_sign_flips,
)
from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.model.neural_policy import NeuralV0Policy  # noqa: E402
from driving_vla.runtime.path_manager import EgoPose, SpatialPath  # noqa: E402
from runtime.carla_connection import (  # noqa: E402
    build_carla_launch_arguments,
    count_offscreen_flags,
    count_rhi_flags,
    normalize_rhi,
    parse_offscreen_from_command_line,
    parse_rhi_from_command_line,
    rewrite_arguments_rhi_offscreen,
)
import xml.etree.ElementTree as ET  # noqa: E402


class StableRunnerHelpersTest(unittest.TestCase):
    def test_navigation_targets_are_forward_on_valid_route(self) -> None:
        route = [(float(x), 0.0) for x in range(0, 101)]
        # Legacy 15/30 m arc (old G3)
        first, second, progress, valid = _navigation_targets(
            route,
            _polyline_s(route),
            EgoPose(20.0, 0.0, 0.0, 5.0),
            0.0,
            contract=legacy_contract_config(),
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(first[0], 15.0)
        self.assertAlmostEqual(second[0], 30.0)
        self.assertAlmostEqual(progress, 20.0)
        # Official RoutePlanner: adjacent densified points after min_distance pop
        ofirst, osecond, oprogress, ovalid = _navigation_targets(
            route,
            _polyline_s(route),
            EgoPose(20.0, 0.0, 0.0, 5.0),
            0.0,
            contract=SimLingoContractConfig(official_contract=True),
        )
        self.assertTrue(ovalid)
        self.assertGreater(ofirst[0], 0.5)
        self.assertGreater(osecond[0], ofirst[0] - 0.1)
        self.assertLess(ofirst[0], 12.0)  # near planner horizon, not fixed 15/30
        self.assertAlmostEqual(oprogress, 20.0, places=0)

    def test_exhausted_route_is_reported_not_replaced_by_fake_straight_target(self) -> None:
        route = [(float(x), 0.0) for x in range(0, 41)]
        first, second, _progress, valid = _navigation_targets(
            route,
            _polyline_s(route),
            EgoPose(39.5, 0.0, 0.0, 5.0),
            38.0,
            contract=legacy_contract_config(),
        )
        self.assertFalse(valid)
        self.assertLess(first[0], 1.0)
        self.assertLess(second[0], 1.0)

    def test_curved_route_does_not_require_second_target_to_increase_ego_x(self) -> None:
        radius = 15.0
        theta = np.linspace(0.0, 2.2, 111)
        route = [
            (radius * math.sin(float(t)), radius * (1.0 - math.cos(float(t))))
            for t in theta
        ]
        first, second, _progress, valid = _navigation_targets(
            route,
            _polyline_s(route),
            EgoPose(0.0, 0.0, 0.0, 5.0),
            0.0,
            contract=legacy_contract_config(),
        )
        self.assertTrue(valid)
        self.assertLess(second[0], first[0] + 2.0)

    def test_default_random_pool_excludes_large_maps(self) -> None:
        self.assertTrue(set(DEFAULT_RANDOM_MAP_POOL).isdisjoint(LARGE_MAPS))

    def test_camera_rate_defaults_to_vla_rate(self) -> None:
        self.assertAlmostEqual(_camera_sensor_tick_s(0.75, 0.05, 0.0), 0.75)
        self.assertAlmostEqual(_camera_sensor_tick_s(0.75, 0.05, 0.20), 0.20)
        self.assertAlmostEqual(_camera_sensor_tick_s(0.75, 0.05, 0.01), 0.05)

    def test_forward_only_runs_model_without_enabling_full_drive_mode(self) -> None:
        self.assertTrue(_mode_runs_forward("forward-only"))
        self.assertTrue(_mode_runs_forward("full"))
        self.assertFalse(_mode_runs_forward("model-resident"))
        self.assertFalse(_mode_runs_forward("camera-only"))

    def test_tail_motion_rejects_a_run_that_finishes_parked(self) -> None:
        speeds = [5.0] * 800 + [0.0] * 400
        metrics = _tail_motion_metrics(speeds, sim_dt_s=0.05, window_s=20.0)
        self.assertAlmostEqual(metrics["window_s"], 20.0)
        self.assertEqual(metrics["moving_fraction"], 0.0)
        self.assertEqual(metrics["mean_speed_mps"], 0.0)

    def test_tail_motion_accepts_sustained_motion(self) -> None:
        metrics = _tail_motion_metrics([4.0] * 400, sim_dt_s=0.05, window_s=20.0)
        self.assertEqual(metrics["moving_fraction"], 1.0)
        self.assertAlmostEqual(metrics["mean_speed_mps"], 4.0)

    def test_recent_crash_context_is_copied_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crash_dir = root / "User" / "AppData/Local/CarlaUE4/Saved/Crashes/UE4CC-test"
            crash_dir.mkdir(parents=True)
            crash = crash_dir / "CrashContext.runtime-xml"
            crash.write_text(
                "<FGenericCrashContext><RuntimeProperties><CrashType>GPUCrash</CrashType>"
                "<ErrorMessage>D3D device lost</ErrorMessage>"
                "<MemoryStats.bIsOOM>0</MemoryStats.bIsOOM>"
                "<RHI.RHIName>D3D11</RHI.RHIName></RuntimeProperties></FGenericCrashContext>",
                encoding="utf-8",
            )
            evidence = root / "evidence"
            evidence.mkdir()
            result = _capture_recent_carla_crash(
                evidence,
                run_wall_start_s=time.time() - 1.0,
                crash_root=root,
            )
            assert result is not None
            self.assertTrue(result["captured"])
            self.assertEqual(result["fields"]["CrashType"], "GPUCrash")
            self.assertEqual(result["fields"]["MemoryStats.bIsOOM"], "0")
            self.assertTrue((evidence / "carla_crash_context_latest.xml").is_file())

    def test_d3d_assert_is_classified_from_nested_crash_context(self) -> None:
        crash_context = {
            "fields": {
                "CrashType": "Assert",
                "ErrorMessage": "D3D11Query failed with DXGI_ERROR_INVALID_CALL",
            }
        }
        self.assertEqual(
            _classify_runtime_failure("simulator time-out", crash_context),
            "CARLA_D3D_CRASH",
        )

    def test_old_crash_context_is_not_attributed_to_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crash_dir = root / "User" / "AppData/Local/CarlaUE4/Saved/Crashes/old"
            crash_dir.mkdir(parents=True)
            crash = crash_dir / "CrashContext.runtime-xml"
            crash.write_text("<FGenericCrashContext/>", encoding="utf-8")
            result = _capture_recent_carla_crash(
                root,
                run_wall_start_s=time.time() + 60.0,
                crash_root=root,
            )
            self.assertIsNone(result)

    def test_tick_timeout_without_crash_context_is_hang(self) -> None:
        self.assertEqual(
            _classify_runtime_failure(
                "time-out of 30000ms while waiting for the simulator, "
                "make sure the simulator is ready and connected to 127.0.0.1:2000",
                None,
            ),
            "CARLA_SERVER_HANG_NO_CRASH_CONTEXT",
        )
        self.assertEqual(
            _classify_runtime_failure(
                "time-out of 30000ms while waiting for the simulator",
                {"captured": False},
            ),
            "CARLA_SERVER_HANG_NO_CRASH_CONTEXT",
        )

    def test_nested_crash_context_fields_with_dotted_tags(self) -> None:
        xml = (
            "<FGenericCrashContext><RuntimeProperties>"
            "<CrashType>Assert</CrashType>"
            "<ErrorMessage>Assertion failed: D3D11Query.cpp:356 DXGI_ERROR_INVALID_CALL</ErrorMessage>"
            "<RHI.RHIName>D3D11</RHI.RHIName>"
            "<MemoryStats.bIsOOM>0</MemoryStats.bIsOOM>"
            "</RuntimeProperties></FGenericCrashContext>"
        )
        root = ET.fromstring(xml)
        self.assertEqual(_xml_find_text_recursive(root, "CrashType"), "Assert")
        self.assertEqual(_xml_find_text_recursive(root, "RHI.RHIName"), "D3D11")
        self.assertEqual(_xml_find_text_recursive(root, "MemoryStats.bIsOOM"), "0")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            crash_dir = base / "User" / "AppData/Local/CarlaUE4/Saved/Crashes/UE4CC-assert"
            crash_dir.mkdir(parents=True)
            crash = crash_dir / "CrashContext.runtime-xml"
            crash.write_text(xml, encoding="utf-8")
            evidence = base / "evidence"
            evidence.mkdir()
            result = _capture_recent_carla_crash(
                evidence,
                run_wall_start_s=time.time() - 1.0,
                crash_root=base,
            )
            assert result is not None
            self.assertTrue(result["captured"])
            self.assertEqual(result["fields"]["CrashType"], "Assert")
            self.assertEqual(result["fields"]["RHI.RHIName"], "D3D11")
            self.assertEqual(
                _classify_runtime_failure("simulator time-out", result),
                "CARLA_D3D_CRASH",
            )

    def test_build_dx11_command_contains_only_dx11(self) -> None:
        cmd = build_carla_launch_arguments(rhi="dx11", render_offscreen=False)
        self.assertIn("-dx11", cmd)
        self.assertNotIn("-dx12", cmd)
        self.assertEqual(count_rhi_flags(cmd), 1)
        self.assertEqual(count_offscreen_flags(cmd), 0)
        self.assertNotIn("-RenderOffScreen", cmd)

    def test_build_dx12_command_contains_only_dx12(self) -> None:
        cmd = build_carla_launch_arguments(rhi="dx12", render_offscreen=False)
        self.assertIn("-dx12", cmd)
        self.assertNotIn("-dx11", cmd)
        self.assertEqual(count_rhi_flags(cmd), 1)

    def test_offscreen_true_adds_exactly_one_flag(self) -> None:
        cmd = build_carla_launch_arguments(rhi="dx11", render_offscreen=True)
        self.assertEqual(count_offscreen_flags(cmd), 1)
        self.assertIn("-RenderOffScreen", cmd)
        self.assertEqual(parse_rhi_from_command_line(cmd), "dx11")
        self.assertTrue(parse_offscreen_from_command_line(cmd))

    def test_rewrite_strips_dual_rhi_and_applies_one(self) -> None:
        messy = (
            "/Game/Carla/Maps/Town04 -windowed -ResX=640 -ResY=360 "
            "-quality-level=Low -nosound -dx11 -dx12 -RenderOffScreen "
            "-carla-rpc-port=2000"
        )
        cleaned = rewrite_arguments_rhi_offscreen(messy, rhi="dx12", render_offscreen=False)
        self.assertEqual(count_rhi_flags(cleaned), 1)
        self.assertEqual(parse_rhi_from_command_line(cleaned), "dx12")
        self.assertEqual(count_offscreen_flags(cleaned), 0)
        self.assertNotIn("-dx11", cleaned)

    def test_normalize_rhi_aliases(self) -> None:
        self.assertEqual(normalize_rhi("DX12"), "dx12")
        self.assertEqual(normalize_rhi("d3d11"), "dx11")
        with self.assertRaises(ValueError):
            normalize_rhi("vulkan")

    def test_n_forward_to_failure_field_shape_in_classify_path(self) -> None:
        """Failure evidence must record forward count; shape checked offline."""
        hang_evidence = {
            "n_forward_to_failure": 79,
            "seconds_since_last_successful_forward": 1.2,
            "last_successful_forward": {
                "seq": 79,
                "wall_end_s": 100.0,
                "sim_end_s": 58.5,
            },
        }
        self.assertEqual(hang_evidence["n_forward_to_failure"], 79)
        self.assertEqual(
            _classify_runtime_failure(
                "time-out of 30000ms while waiting for the simulator",
                None,
            ),
            "CARLA_SERVER_HANG_NO_CRASH_CONTEXT",
        )

    def test_default_rhi_is_dx12(self) -> None:
        self.assertEqual(DEFAULT_RHI, "dx12")
        self.assertEqual(normalize_rhi(DEFAULT_RHI), "dx12")
        cmd = build_carla_launch_arguments(rhi=DEFAULT_RHI, render_offscreen=False)
        self.assertIn("-dx12", cmd)
        self.assertNotIn("-dx11", cmd)

    def test_default_ego_blueprint_is_mercedes(self) -> None:
        self.assertEqual(DEFAULT_EGO_BLUEPRINT, "vehicle.mercedes.coupe_2020")

    def test_geometry_yaw90_not_swapped(self) -> None:
        # 2.5m wheelbase × 1.5m track, rotated 90° in world frame.
        half_l, half_t = 1.25, 0.75
        # yaw=90: local (x_fwd, y_left) → world (-y, x)
        pts = [
            [-half_t, half_l],
            [half_t, half_l],
            [-half_t, -half_l],
            [half_t, -half_l],
        ]
        wb, tr = estimate_wheelbase_track_m(pts)
        self.assertAlmostEqual(wb, 2.5, places=5)
        self.assertAlmostEqual(tr, 1.5, places=5)

    def test_lane_invasion_episodes_merge(self) -> None:
        book = LaneInvasionEpisodeBook(gap_s=0.5)
        book.on_invasion(sim_s=1.0, carla_frame=1, road_id=1, lane_id=-1)
        book.on_invasion(sim_s=1.1, carla_frame=2, road_id=1, lane_id=-1)
        book.on_invasion(sim_s=3.0, carla_frame=3, road_id=2, lane_id=1)
        summary = book.summary()
        self.assertEqual(summary["raw_event_count"], 3)
        self.assertEqual(summary["episode_count"], 2)

    def test_multi_deadband_flips_observation_only(self) -> None:
        flips = multi_deadband_sign_flips(
            [0.1, -0.1, 0.1], deadbands=(0.01, 0.03, 0.05)
        )
        self.assertEqual(flips["deadband_0.01"], 2)

    def test_stopped_vehicle_feeds_real_zero_speed_to_vla(self) -> None:
        obs = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=0.0,
            meta={},
        )
        self.assertEqual(NeuralV0Policy.resolve_vla_input_speed_mps(obs), 0.0)
        moving = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=4.2,
            meta={},
        )
        self.assertAlmostEqual(NeuralV0Policy.resolve_vla_input_speed_mps(moving), 4.2)

    def test_startup_assist_disabled_after_collision(self) -> None:
        cold = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=0.0,
            meta={"startup_speed_assist_mps": 3.0, "has_collided": False},
        )
        self.assertAlmostEqual(NeuralV0Policy.resolve_vla_input_speed_mps(cold), 3.0)
        after_hit = ObservationBundle(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            simulation_time_s=1.0,
            ego_v=0.0,
            meta={"startup_speed_assist_mps": 3.0, "has_collided": True},
        )
        self.assertEqual(NeuralV0Policy.resolve_vla_input_speed_mps(after_hit), 0.0)

    def test_collision_events_merge_into_episodes_with_pre_snapshot(self) -> None:
        book = CollisionEpisodeBook(gap_s=0.5)
        book.set_sim_s(10.0)
        book.update_control_snapshot(
            {
                "sim_s": 10.0,
                "speed_mps": 5.5,
                "throttle": 0.4,
                "brake": 0.0,
                "steer": -0.1,
                "raw_path_map_xy": [[0.0, 0.0], [5.0, 0.0]],
                "committed_path_map_xy": [[0.0, 0.0], [4.0, 0.1]],
            }
        )
        book.on_collision(
            impulse=12.0,
            other_type="static.prop.streetbarrier",
            other_id=7,
            vehicle_xy=(1.0, 2.0),
            vehicle_yaw=0.3,
            sim_s=10.0,
        )
        book.on_collision(
            impulse=3.0,
            other_type="static.prop.streetbarrier",
            other_id=7,
            vehicle_xy=(1.05, 2.0),
            sim_s=10.2,
        )
        book.set_sim_s(12.0)
        book.update_control_snapshot(
            {
                "sim_s": 12.0,
                "speed_mps": 0.0,
                "throttle": 0.0,
                "brake": 1.0,
                "steer": 0.0,
                "raw_path_map_xy": [[1.0, 2.0], [2.0, 2.0]],
                "committed_path_map_xy": [[1.0, 2.0], [1.5, 2.0]],
            }
        )
        book.on_collision(
            impulse=8.0,
            other_type="vehicle.tesla.model3",
            other_id=9,
            vehicle_xy=(3.0, 4.0),
            sim_s=12.0,
        )
        summary = book.summary()
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["raw_event_count"], 3)
        first = summary["episodes"][0]
        second = summary["episodes"][1]
        self.assertEqual(first["raw_event_count"], 2)
        self.assertAlmostEqual(first["impulse_sum"], 15.0)
        self.assertEqual(first["other_type"], "static.prop.streetbarrier")
        self.assertEqual(first["vehicle_pose_first"]["x"], 1.0)
        self.assertEqual(first["pre_collision"]["speed_mps"], 5.5)
        self.assertEqual(first["pre_collision"]["throttle"], 0.4)
        self.assertEqual(first["pre_collision"]["raw_path_map_xy"][1], [5.0, 0.0])
        self.assertEqual(second["other_type"], "vehicle.tesla.model3")
        self.assertEqual(second["pre_collision"]["brake"], 1.0)
        self.assertEqual(summary["first_collision_sim_s"], 10.0)
        self.assertEqual(summary["first_collision_other_type"], "static.prop.streetbarrier")

    def test_path_xy_as_lists_from_spatial_path(self) -> None:
        path = SpatialPath(
            s=np.asarray([0.0, 1.0], dtype=float),
            x=np.asarray([0.0, 1.0], dtype=float),
            y=np.asarray([0.0, 0.5], dtype=float),
            yaw=np.asarray([0.0, 0.0], dtype=float),
            kappa=np.asarray([0.0, 0.0], dtype=float),
            target_speed_mps=5.0,
            stamp_s=1.0,
        )
        points = _path_xy_as_lists(path)
        assert points is not None
        self.assertEqual(points[0], [0.0, 0.0])
        self.assertEqual(points[1], [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
