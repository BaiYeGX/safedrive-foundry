from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.behavior import (  # noqa: E402
    BehaviorEvent,
    BehaviorState,
    BehaviorStateMachine,
)
from classic_stack.behavior.state_machine import BehaviorConfig  # noqa: E402
from classic_stack.map import (  # noqa: E402
    LaneGraphBuilder,
    LaneGraphQuery,
    load_carla_map,
    load_map_fixture,
)
from classic_stack.map.lane_graph import load_cached_graph  # noqa: E402
from classic_stack.route import RoutePlanner, RouteRequest  # noqa: E402
from classic_stack.route.planner import RouteError  # noqa: E402


MAPS = ("Town01", "Town03", "Town10HD")
CARLA_FIXTURE_DIR = ROOT / "safedrive_foundry" / "classic_stack" / "map" / "fixtures" / "carla"
CARLA_MANIFEST = CARLA_FIXTURE_DIR / "manifest.json"


class G103MapRouteBehaviorTests(unittest.TestCase):
    def test_three_maps_build_cache_query_and_detect_anomalies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            hashes = {}
            for name in MAPS:
                graph = load_map_fixture(name, cache_dir=cache_dir)
                self.assertTrue(graph.nodes, msg=f"{name} produced empty graph")
                self.assertTrue(graph.map_hash)
                hashes[name] = graph.map_hash
                query = LaneGraphQuery(graph)
                sample = next(iter(graph.driving_nodes()))
                self.assertEqual(query.get_node(sample.node_id).node_id, sample.node_id)
                self.assertIn("topology", graph.semantics.oracle_inputs)
                self.assertIn("navigation", graph.semantics.observable_inputs)
                cache_files = list(cache_dir.glob(f"{graph.map_name}-*.json"))
                self.assertEqual(len(cache_files), 1)
                reloaded = load_cached_graph(cache_files[0])
                self.assertEqual(reloaded.map_hash, graph.map_hash)
                self.assertEqual(set(reloaded.nodes), set(graph.nodes))

            # Town10HD fixture includes an orphan road with missing successor target.
            town10 = load_map_fixture("Town10HD")
            anomaly_codes = {item.code for item in town10.anomalies}
            self.assertTrue(
                {"MISSING_ENDPOINT", "DEAD_END_WITH_SUCCESSOR", "DISCONNECTED_COMPONENT"} & anomaly_codes,
                msg=f"expected topology anomalies, got {town10.anomalies}",
            )
            self.assertEqual(len(hashes), 3)
            self.assertEqual(len(set(hashes.values())), 3)

    def test_route_corridor_is_deterministic_and_semantic(self) -> None:
        graph = load_map_fixture("Town01")
        start = "R1:S0:L-1"
        goal = "R2:S0:L-1"
        planner = RoutePlanner(graph)
        request = RouteRequest(start_node_id=start, goal_node_ids=(goal,), seed=7)
        first = planner.plan(request)
        second = planner.plan(request)
        self.assertTrue(first.ok, first.message)
        self.assertTrue(second.ok, second.message)
        assert first.corridor is not None
        assert second.corridor is not None
        self.assertEqual(first.corridor.route_id, second.corridor.route_id)
        self.assertEqual(first.corridor.node_ids(), second.corridor.node_ids())
        self.assertEqual(first.corridor.map_hash, graph.map_hash)
        self.assertTrue(first.corridor.semantics["contains_road"])
        self.assertIn("road", {step.semantic for step in first.corridor.steps})
        # multi-goal prefers reachable legal goal deterministically
        multi = planner.plan(RouteRequest(start_node_id=start, goal_node_ids=(goal, "R2:S0:L-2"), seed=7))
        self.assertTrue(multi.ok)
        assert multi.corridor is not None
        self.assertIn(multi.corridor.goal_node_id, {goal, "R2:S0:L-2"})

    def test_route_lane_change_and_junction_semantics(self) -> None:
        town01 = load_map_fixture("Town01")
        planner = RoutePlanner(town01)
        lane_change = planner.plan(
            RouteRequest(
                start_node_id="R1:S0:L-1",
                goal_node_ids=("R2:S0:L-2",),
                seed=11,
                allow_lane_change=True,
            )
        )
        self.assertTrue(lane_change.ok, lane_change.message)
        assert lane_change.corridor is not None
        self.assertGreaterEqual(lane_change.corridor.lane_change_count, 1)
        self.assertTrue(lane_change.corridor.semantics["contains_lane_change"])
        self.assertTrue(any(step.maneuver in {"LEFT", "RIGHT"} for step in lane_change.corridor.steps))

        town03 = load_map_fixture("Town03")
        turn = RoutePlanner(town03).plan(
            RouteRequest(
                start_node_id="R10:S0:L-1",
                goal_node_ids=("R14:S0:L-1",),
                seed=3,
            )
        )
        self.assertTrue(turn.ok, turn.message)
        assert turn.corridor is not None
        self.assertGreaterEqual(turn.corridor.junction_count, 1)
        self.assertTrue(turn.corridor.semantics["contains_junction"])
        self.assertTrue(set(turn.corridor.road_sequence) & {"11", "13"}, msg=turn.corridor.road_sequence)

        straight = RoutePlanner(town03).plan(
            RouteRequest(
                start_node_id="R10:S0:L-1",
                goal_node_ids=("R12:S0:L-1",),
                seed=3,
            )
        )
        self.assertTrue(straight.ok, straight.message)
        assert straight.corridor is not None
        self.assertIn("12", straight.corridor.road_sequence)

    def test_unreachable_goal_reports_failure_node(self) -> None:
        graph = load_map_fixture("Town10HD")
        planner = RoutePlanner(graph)
        result = planner.plan(
            RouteRequest(
                start_node_id="R20:S0:L-1",
                goal_node_ids=("R99:S0:L-1",),
                seed=1,
            )
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, "UNREACHABLE")
        self.assertEqual(result.failure_node_id, "R20:S0:L-1")

        with self.assertRaises(RouteError):
            planner.plan(RouteRequest(start_node_id="missing", goal_node_ids=("R20:S0:L-1",), seed=0))

    def test_behavior_timeline_enter_hold_exit_timeout_suppress(self) -> None:
        machine = BehaviorStateMachine(BehaviorConfig(lane_change_timeout_s=2.0), route_id="route-demo")
        t0 = machine.handle(BehaviorEvent.ROUTE_STRAIGHT, timestamp_s=0.0)
        self.assertEqual(t0.phase, "hold")
        self.assertEqual(machine.state, BehaviorState.CRUISE)

        enter_follow = machine.handle(BehaviorEvent.LEAD_VEHICLE_DETECTED, timestamp_s=1.0)
        self.assertEqual(enter_follow.phase, "enter")
        self.assertEqual(machine.state, BehaviorState.FOLLOW)
        self.assertTrue(any(item.phase == "exit" for item in machine.history))

        hold = machine.handle(BehaviorEvent.TICK, timestamp_s=2.0)
        self.assertEqual(hold.phase, "hold")

        red = machine.handle(BehaviorEvent.RED_LIGHT, timestamp_s=3.0)
        self.assertEqual(red.to_state, BehaviorState.STOP)
        self.assertEqual(machine.goal.target_speed_mps, 0.0)
        self.assertIn("signal_state", machine.goal.oracle_inputs)
        self.assertIn("perceived_signal", machine.goal.observable_inputs)

        green = machine.handle(BehaviorEvent.GREEN_LIGHT, timestamp_s=4.0)
        self.assertEqual(green.to_state, BehaviorState.CRUISE)

        machine.handle(BehaviorEvent.LANE_CHANGE_REQUEST, timestamp_s=5.0, context={"target_lane_id": -2})
        self.assertEqual(machine.state, BehaviorState.LANE_CHANGE)
        timeout = machine.handle(BehaviorEvent.TIMEOUT, timestamp_s=8.0)
        self.assertEqual(timeout.to_state, BehaviorState.CRUISE)
        self.assertIn(timeout.phase, {"enter", "timeout"})

        machine.handle(BehaviorEvent.UNPROTECTED_LEFT, timestamp_s=9.0)
        self.assertEqual(machine.state, BehaviorState.YIELD)
        machine.handle(BehaviorEvent.CROSS_TRAFFIC_CLEAR, timestamp_s=10.0)
        self.assertEqual(machine.state, BehaviorState.CRUISE)

        machine.suppress(BehaviorEvent.OBSTACLE_AHEAD, reason="unit_test_suppress", timestamp_s=11.0)
        suppressed = machine.handle(BehaviorEvent.OBSTACLE_AHEAD, timestamp_s=12.0)
        self.assertTrue(suppressed.suppressed)
        self.assertEqual(suppressed.phase, "suppress")
        self.assertEqual(machine.state, BehaviorState.CRUISE)

        machine.unsuppress(BehaviorEvent.OBSTACLE_AHEAD)
        avoid = machine.handle(BehaviorEvent.OBSTACLE_AHEAD, timestamp_s=13.0)
        self.assertEqual(avoid.to_state, BehaviorState.AVOID)
        min_risk = machine.handle(BehaviorEvent.FORCE_MIN_RISK, timestamp_s=14.0)
        self.assertEqual(min_risk.to_state, BehaviorState.MIN_RISK)

        timeline = machine.timeline()
        phases = {item["phase"] for item in timeline}
        self.assertTrue({"enter", "hold", "exit", "suppress"} <= phases)
        self.assertTrue(any(item["event"] == "RED_LIGHT" for item in timeline))
        # no controls / trajectories
        self.assertFalse(machine.goal.metadata["emits_controls"])
        self.assertFalse(machine.goal.metadata["emits_local_trajectory"])

    def test_fixed_seed_scenarios_straight_turn_lane_change_red_unprotected(self) -> None:
        # Straight
        town03 = load_map_fixture("Town03")
        straight = RoutePlanner(town03).plan(
            RouteRequest(start_node_id="R10:S0:L-1", goal_node_ids=("R12:S0:L-1",), seed=42)
        )
        self.assertTrue(straight.ok)
        assert straight.corridor is not None
        behavior = BehaviorStateMachine(route_id=straight.corridor.route_id)
        behavior.handle(BehaviorEvent.ROUTE_STRAIGHT, timestamp_s=0.0)
        self.assertEqual(behavior.state, BehaviorState.CRUISE)

        # Turn + unprotected yield
        turn = RoutePlanner(town03).plan(
            RouteRequest(start_node_id="R10:S0:L-1", goal_node_ids=("R14:S0:L-1",), seed=42)
        )
        self.assertTrue(turn.ok)
        behavior.handle(BehaviorEvent.ROUTE_TURN, timestamp_s=1.0)
        behavior.handle(BehaviorEvent.UNPROTECTED_LEFT, timestamp_s=2.0)
        self.assertEqual(behavior.state, BehaviorState.YIELD)

        # Lane change
        town01 = load_map_fixture("Town01")
        lane_change_route = RoutePlanner(town01).plan(
            RouteRequest(start_node_id="R1:S0:L-1", goal_node_ids=("R2:S0:L-2",), seed=42)
        )
        self.assertTrue(lane_change_route.ok)
        behavior2 = BehaviorStateMachine(route_id=lane_change_route.corridor.route_id if lane_change_route.corridor else None)
        behavior2.handle(BehaviorEvent.LANE_CHANGE_REQUEST, timestamp_s=0.0, context={"target_lane_id": -2})
        self.assertEqual(behavior2.state, BehaviorState.LANE_CHANGE)
        behavior2.handle(BehaviorEvent.LANE_CHANGE_COMPLETE, timestamp_s=1.0)
        self.assertEqual(behavior2.state, BehaviorState.CRUISE)

        # Red light stop
        behavior3 = BehaviorStateMachine()
        behavior3.handle(BehaviorEvent.RED_LIGHT, timestamp_s=0.0)
        self.assertEqual(behavior3.state, BehaviorState.STOP)
        behavior3.handle(BehaviorEvent.GREEN_LIGHT, timestamp_s=1.0)
        self.assertEqual(behavior3.state, BehaviorState.CRUISE)

    def test_opendrive_builder_exposes_signal_and_stop_line_semantics(self) -> None:
        graph = load_map_fixture("Town01")
        self.assertTrue(graph.semantics.signals)
        self.assertTrue(graph.semantics.stop_lines)
        node = graph.nodes["R1:S0:L-1"]
        self.assertIsNotNone(node.stop_line_s)
        self.assertIn("sig-1", node.signal_ids)
        rebuilt = LaneGraphBuilder().build_from_path(
            ROOT / "safedrive_foundry" / "classic_stack" / "map" / "fixtures" / "Town01.xodr"
        )
        self.assertEqual(rebuilt.map_hash, graph.map_hash)

    def test_real_carla_opendrive_three_maps_build_cache_and_route(self) -> None:
        """Acceptance maps: CARLA 0.9.16 packaged OpenDRIVE (not synthetic fixtures)."""

        self.assertTrue(CARLA_MANIFEST.is_file(), "carla fixture manifest missing")
        manifest = json.loads(CARLA_MANIFEST.read_text(encoding="utf-8"))
        maps = manifest["maps"]
        self.assertEqual(set(maps), set(MAPS))

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            hashes: dict[str, str] = {}
            for name in MAPS:
                expected = maps[name]["sha256"]
                raw = (CARLA_FIXTURE_DIR / f"{name}.xodr").read_bytes()
                import hashlib

                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, msg=f"{name} xodr hash mismatch")
                graph = load_carla_map(name, cache_dir=cache_dir)
                self.assertEqual(graph.source_kind, "carla_packaged_opendrive")
                self.assertGreater(len(graph.nodes), 50, msg=f"{name} too small for real map")
                self.assertGreater(len(graph.edges), 20)
                self.assertTrue(graph.map_hash)
                hashes[name] = graph.map_hash
                query = LaneGraphQuery(graph)
                sample = next(iter(graph.driving_nodes()))
                self.assertEqual(query.get_node(sample.node_id).node_id, sample.node_id)
                # Prefer a driving node with outgoing edges; first dict order may be a dead-end lane.
                driving = [node for node in graph.driving_nodes() if graph.successors(node.node_id)]
                self.assertTrue(driving, msg=f"{name} has no driving node with successors")
                start = driving[0].node_id
                reachable = None
                for candidate in driving[1:80]:
                    if candidate.node_id != start and query.connected(start, candidate.node_id):
                        reachable = candidate.node_id
                        break
                self.assertIsNotNone(reachable, msg=f"{name} has no multi-hop driving path from {start}")
                plan = RoutePlanner(graph).plan(
                    RouteRequest(start_node_id=start, goal_node_ids=(reachable,), seed=7)
                )
                self.assertTrue(plan.ok, msg=f"{name} route failed: {plan.message}")
                assert plan.corridor is not None
                self.assertEqual(plan.corridor.map_hash, graph.map_hash)
                self.assertGreaterEqual(len(plan.corridor.steps), 1)

            self.assertEqual(len(set(hashes.values())), 3)


if __name__ == "__main__":
    unittest.main()

