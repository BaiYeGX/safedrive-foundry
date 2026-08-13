from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.control.controller import ControlLoop, EgoState  # noqa: E402
from driving_vla.adapter.policy_adapter import ObservationBundle  # noqa: E402
from driving_vla.hybrid.contracts import (  # noqa: E402
    GenerationAttempt,
    GuardResult,
    GuardVerdict,
    HybridCandidateSet,
    HybridContractError,
    HybridSource,
    ObservableAnchor,
    SelectionSpace,
)
from driving_vla.hybrid.generators import (  # noqa: E402
    ClassicExpertGenerator,
    NominalVLAGenerator,
    generate_hybrid_set,
    route_revision_sha256,
)
from driving_vla.hybrid.guard import CandidateGuard  # noqa: E402
from driving_vla.hybrid.pipeline import H1CandidatePipeline  # noqa: E402
from driving_vla.hybrid.router import FrozenH1Router  # noqa: E402
from driving_vla.model.canonicalizer import (  # noqa: E402
    CanonicalizationError,
    TrajectoryCanonicalizer,
    UpstreamPathSpeed,
    UpstreamTimedTrajectory,
    stable_sha256,
)
from driving_vla.model.nominal_policy import NominalVLAPolicy  # noqa: E402
from driving_vla.runtime.safety_control_bind import apply_safety_control  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ComponentAvailability,
    ObservableSnapshot,
    TrackedObject,
)
from safety_kernel.kernel import SafetyKernel  # noqa: E402


def make_anchor() -> ObservableAnchor:
    route = tuple((float(i), 0.0) for i in range(61))
    bundle = ObservationBundle(
        run_id="run-h1",
        frame_id="obs-20",
        scenario_id="h1-straight",
        simulation_time_s=1.0,
        wall_time_s=100.0,
        carla_frame=20,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=2.0,
        route_xy=route,
        front_rgb=np.zeros((512, 1024, 3), dtype=np.uint8),
        meta={
            "official_contract": True,
            "target_ego_1": (15.0, 0.0),
            "target_ego_2": (30.0, 0.0),
        },
    )
    snapshot = ObservableSnapshot(
        run_id=bundle.run_id,
        frame_id=bundle.frame_id,
        scenario_id=bundle.scenario_id,
        simulation_time_s=bundle.simulation_time_s,
        wall_time_s=bundle.wall_time_s,
        ego_x=bundle.ego_x,
        ego_y=bundle.ego_y,
        ego_yaw=bundle.ego_yaw,
        ego_v=bundle.ego_v,
        observed_time_s=bundle.simulation_time_s,
        freshness_s=0.0,
        speed_limit_mps=12.0,
        corridor_centerline=route,
        corridor_half_width_m=1.75,
        coordinate_frame="map",
    )
    return ObservableAnchor(
        observation_id=bundle.frame_id,
        bundle=bundle,
        safety_snapshot=snapshot,
        route_revision=route_revision_sha256(route),
        sensor_frames={"front_camera": bundle.carla_frame},
        sensor_timestamps_s={"front_camera": bundle.simulation_time_s},
    )


class FakeSimLingoRuntime:
    def __init__(self) -> None:
        self.load_report = SimpleNamespace(ok=True, error="")
        self.forward_count = 0

    def forward_numpy(self, _image, **_kwargs):
        self.forward_count += 1
        return SimpleNamespace(
            route_xy=np.column_stack((np.arange(0.0, 20.0), np.zeros(20))),
            speed_wps_xy=np.zeros((10, 2)),
            speed_mps=np.full(10, 2.0),
            latency_s=0.01,
            peak_vram_mb=123.0,
        )


def make_generated_set() -> tuple[HybridCandidateSet, FakeSimLingoRuntime]:
    anchor = make_anchor()
    runtime = FakeSimLingoRuntime()
    policy = NominalVLAPolicy(runtime=runtime)
    generated = generate_hybrid_set(
        anchor,
        ClassicExpertGenerator(),
        NominalVLAGenerator(policy, generator_hash="fake-simlingo-hash"),
    )
    return generated, runtime


def pass_result(candidate_id: str) -> GuardResult:
    return GuardResult(
        candidate_id=candidate_id,
        verdict=GuardVerdict.PASS,
        checks=(),
        reject_reasons=(),
        latency_ms=0.1,
    )


class CanonicalizerH1Test(unittest.TestCase):
    def test_path_speed_is_strict_t10(self) -> None:
        result = TrajectoryCanonicalizer().canonicalize_with_report(
            UpstreamPathSpeed(
                path_xy=tuple((float(i), 0.0) for i in range(20)),
                speed_mps=(2.0,) * 10,
                frame="map",
            ),
            to_map=False,
        )
        self.assertEqual(result.trajectory.t_steps, 10)
        self.assertEqual(result.report.source_points, 20)
        self.assertEqual(result.report.canonical_points, 10)
        self.assertEqual(result.report.coverage_shortfall_m, 0.0)

    def test_degenerate_and_insufficient_path_fail_closed(self) -> None:
        canonicalizer = TrajectoryCanonicalizer()
        with self.assertRaisesRegex(CanonicalizationError, "degenerate_path"):
            canonicalizer.canonicalize(
                UpstreamPathSpeed(path_xy=((0.0, 0.0),), speed_mps=(2.0,), frame="map")
            )
        with self.assertRaisesRegex(CanonicalizationError, "insufficient_path_coverage"):
            canonicalizer.canonicalize(
                UpstreamPathSpeed(
                    path_xy=((0.0, 0.0), (1.0, 0.0)),
                    speed_mps=(5.0,) * 10,
                    frame="map",
                )
            )

    def test_timed_resampling_requires_full_horizon(self) -> None:
        canonicalizer = TrajectoryCanonicalizer()
        rows = tuple(
            (i * 0.1, i * 0.2, 0.0, 0.0, 2.0, 0.0, 0.0)
            for i in range(31)
        )
        result = canonicalizer.canonicalize_timed(
            UpstreamTimedTrajectory(points=rows, frame="map"), to_map=False
        )
        self.assertEqual(result.trajectory.t_steps, 10)
        self.assertAlmostEqual(result.trajectory.points_xy_yaw_v_a_kappa[-1][0], 5.0)
        with self.assertRaisesRegex(CanonicalizationError, "timed_horizon_too_short"):
            canonicalizer.canonicalize_timed(
                UpstreamTimedTrajectory(points=rows[:20], frame="map"), to_map=False
            )


class HybridGenerationAndGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generated, cls.runtime = make_generated_set()

    def test_anchor_rejects_sensor_frame_mismatch(self) -> None:
        anchor = make_anchor()
        with self.assertRaisesRegex(HybridContractError, "sensor_frame_mismatch"):
            ObservableAnchor(
                observation_id=anchor.observation_id,
                bundle=anchor.bundle,
                safety_snapshot=anchor.safety_snapshot,
                route_revision=anchor.route_revision,
                sensor_frames={"front_camera": anchor.bundle.carla_frame + 1},
                sensor_timestamps_s={"front_camera": anchor.bundle.simulation_time_s},
            )

    def test_two_independent_sources_and_single_vla_forward(self) -> None:
        self.assertEqual(self.runtime.forward_count, 1)
        vla_policy_count = next(
            item for item in self.generated.attempts if item.source is HybridSource.VLA
        )
        self.assertTrue(vla_policy_count.success)
        self.assertEqual(len(self.generated.candidates), 2)
        self.assertEqual(
            {item.provenance.source for item in self.generated.candidates},
            {HybridSource.EXPERT, HybridSource.VLA},
        )
        self.assertTrue(all(len(item.candidate.points) == 10 for item in self.generated.candidates))
        vla = next(
            item for item in self.generated.candidates if item.provenance.source is HybridSource.VLA
        )
        self.assertEqual(vla.provenance.generator_hash, "fake-simlingo-hash")
        self.assertEqual(vla.candidate.source, CandidateSource.VLA_FAST)
        self.assertEqual(vla.candidate.dynamics_meta["observation_id"], "obs-20")

    def test_guard_passes_both_and_preserves_stage_order(self) -> None:
        guarded = CandidateGuard().evaluate(self.generated)
        expected_order = [
            "contract",
            "binding_freshness",
            "route_navigation",
            "dynamics_trackability",
            "observable_collision",
            "controller_feasibility",
        ]
        for item in guarded.candidates:
            self.assertTrue(item.guard.passed, item.guard.reject_reasons)
            seen = []
            for check in item.guard.checks:
                if not seen or seen[-1] != check.stage:
                    seen.append(check.stage)
            self.assertEqual(seen, expected_order)

    def test_generation_deadline_rejects_before_route_checks(self) -> None:
        original = self.generated.candidates[0]
        late = replace(
            original,
            provenance=replace(
                original.provenance, generation_latency_s=3.0, freshness_s=3.0
            ),
        )
        cset = HybridCandidateSet(self.generated.anchor, (late,), self.generated.attempts[:1])
        result = CandidateGuard().evaluate(cset).candidates[0].guard
        self.assertEqual(result.verdict, GuardVerdict.REJECT)
        self.assertIn("generation_deadline", result.reject_reasons[0])
        self.assertNotIn("route_navigation", [check.stage for check in result.checks])

    def test_off_corridor_reject_reason_is_candidate_local(self) -> None:
        original = self.generated.candidates[0]
        points = tuple(replace(point, y=3.0) for point in original.candidate.points)
        canonical_hash = stable_sha256(
            tuple((p.x, p.y, p.yaw, p.v, p.a, p.kappa) for p in points)
        )
        moved = replace(
            original,
            candidate=replace(original.candidate, points=points),
            provenance=replace(original.provenance, canonical_sha256=canonical_hash),
        )
        cset = HybridCandidateSet(self.generated.anchor, (moved,), self.generated.attempts[:1])
        result = CandidateGuard().evaluate(cset).candidates[0].guard
        self.assertEqual(result.verdict, GuardVerdict.REJECT)
        self.assertTrue(any("route_navigation:road:offroad" in reason for reason in result.reject_reasons))

    def test_route_revision_mismatch_rejects_in_binding_stage(self) -> None:
        original = self.generated.candidates[0]
        mismatched = replace(
            original,
            provenance=replace(original.provenance, route_revision="different-route"),
        )
        cset = HybridCandidateSet(
            self.generated.anchor, (mismatched,), self.generated.attempts[:1]
        )
        result = CandidateGuard().evaluate(cset).candidates[0].guard
        self.assertEqual(result.verdict, GuardVerdict.REJECT)
        self.assertTrue(any("observation_binding" in reason for reason in result.reject_reasons))
        self.assertEqual(result.checks[-1].stage, "binding_freshness")

    def test_non_finite_and_short_contract_fail_before_later_stages(self) -> None:
        original = self.generated.candidates[0]
        points = (replace(original.candidate.points[0], x=float("nan")),) + original.candidate.points[1:]
        malformed = replace(original, candidate=replace(original.candidate, points=points))
        cset = HybridCandidateSet(
            self.generated.anchor, (malformed,), self.generated.attempts[:1]
        )
        non_finite = CandidateGuard().evaluate(cset).candidates[0].guard
        self.assertEqual(non_finite.verdict, GuardVerdict.REJECT)
        self.assertEqual({check.stage for check in non_finite.checks}, {"contract"})

        short_points = original.candidate.points[:5]
        short = replace(
            original,
            candidate=replace(original.candidate, points=short_points),
            provenance=replace(
                original.provenance,
                canonical_sha256=stable_sha256(
                    tuple((p.x, p.y, p.yaw, p.v, p.a, p.kappa) for p in short_points)
                ),
            ),
        )
        short_set = HybridCandidateSet(
            self.generated.anchor, (short,), self.generated.attempts[:1]
        )
        short_result = CandidateGuard().evaluate(short_set).candidates[0].guard
        self.assertEqual(short_result.verdict, GuardVerdict.REJECT)
        self.assertTrue(any("h1_t_steps" in reason for reason in short_result.reject_reasons))

    def test_dynamics_and_collision_are_separate_guard_stages(self) -> None:
        original = self.generated.candidates[0]
        fast_points = tuple(replace(point, v=100.0) for point in original.candidate.points)
        fast = replace(
            original,
            candidate=replace(original.candidate, points=fast_points),
            provenance=replace(
                original.provenance,
                canonical_sha256=stable_sha256(
                    tuple((p.x, p.y, p.yaw, p.v, p.a, p.kappa) for p in fast_points)
                ),
            ),
        )
        no_limit_anchor = ObservableAnchor(
            observation_id=self.generated.anchor.observation_id,
            bundle=self.generated.anchor.bundle,
            safety_snapshot=replace(self.generated.anchor.safety_snapshot, speed_limit_mps=None),
            route_revision=self.generated.anchor.route_revision,
            sensor_frames=self.generated.anchor.sensor_frames,
            sensor_timestamps_s=self.generated.anchor.sensor_timestamps_s,
        )
        fast_set = HybridCandidateSet(
            no_limit_anchor, (fast,), self.generated.attempts[:1]
        )
        fast_result = CandidateGuard().evaluate(fast_set).candidates[0].guard
        self.assertTrue(any("dynamics_trackability:dynamics" in reason for reason in fast_result.reject_reasons))

        first = original.candidate.points[0]
        actor = TrackedObject(
            actor_id="blocking-actor",
            class_name="vehicle.test",
            x=first.x,
            y=first.y,
            yaw=0.0,
            vx=0.0,
            vy=0.0,
            length_m=4.5,
            width_m=1.9,
            observed_time_s=self.generated.anchor.simulation_time_s,
        )
        collision_anchor = ObservableAnchor(
            observation_id=self.generated.anchor.observation_id,
            bundle=self.generated.anchor.bundle,
            safety_snapshot=replace(self.generated.anchor.safety_snapshot, actors=(actor,)),
            route_revision=self.generated.anchor.route_revision,
            sensor_frames=self.generated.anchor.sensor_frames,
            sensor_timestamps_s=self.generated.anchor.sensor_timestamps_s,
        )
        collision_set = HybridCandidateSet(
            collision_anchor, (original,), self.generated.attempts[:1]
        )
        collision_result = CandidateGuard().evaluate(collision_set).candidates[0].guard
        self.assertTrue(any("observable_collision:collision" in reason for reason in collision_result.reject_reasons))

    def test_controller_infeasible_is_last_stage(self) -> None:
        class BrakeOnly:
            def set_trajectory(self, *_args) -> None:
                return None

            def step(self, *_args):
                return SimpleNamespace(steer=0.0, throttle=0.0, brake=1.0, mode="brake")

        cset = HybridCandidateSet(
            self.generated.anchor,
            (self.generated.candidates[0],),
            self.generated.attempts[:1],
        )
        result = CandidateGuard(control_factory=BrakeOnly).evaluate(cset).candidates[0].guard
        self.assertEqual(result.verdict, GuardVerdict.REJECT)
        self.assertTrue(any("controller_mode" in reason for reason in result.reject_reasons))


class H1RoutingAndSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generated, _ = make_generated_set()
        cls.generated = generated
        cls.guarded = CandidateGuard().evaluate(generated)

    def test_zero_one_two_pass_routing(self) -> None:
        router = FrozenH1Router()
        rejected = tuple(
            replace(
                item,
                guard=replace(
                    item.guard,
                    verdict=GuardVerdict.REJECT,
                    reject_reasons=("test_reject",),
                ),
            )
            for item in self.guarded.candidates
        )
        zero = router.route(replace(self.guarded, candidates=rejected))
        self.assertEqual(zero.selection_space, SelectionSpace.ZERO_PASS)
        self.assertIsNone(zero.selected_candidate_id)

        one_set = HybridCandidateSet(
            self.guarded.anchor,
            (self.guarded.candidates[0],),
            (self.guarded.attempts[0],),
        )
        one = router.route(one_set)
        self.assertEqual(one.selection_space, SelectionSpace.SINGLE_PASS)
        self.assertEqual(one.selected_candidate_id, self.guarded.candidates[0].candidate.candidate_id)

        two = router.route(self.guarded)
        self.assertEqual(two.selection_space, SelectionSpace.DISTINCT)
        self.assertIsNotNone(two.selected_candidate_id)
        self.assertEqual(set(two.pass_candidate_ids), {item.candidate.candidate_id for item in self.guarded.candidates})

    def test_near_duplicate_and_slot_permutation_are_stable(self) -> None:
        router = FrozenH1Router()
        vla = next(
            item for item in self.guarded.candidates if item.provenance.source is HybridSource.VLA
        )
        expert_id = f"{self.guarded.anchor.observation_id}:expert-copy"
        expert = replace(
            vla,
            candidate=replace(
                vla.candidate,
                candidate_id=expert_id,
                source=CandidateSource.CLASSIC,
            ),
            provenance=replace(
                vla.provenance,
                source=HybridSource.EXPERT,
                candidate_id=expert_id,
            ),
            guard=pass_result(expert_id),
        )
        vla = replace(vla, guard=pass_result(vla.candidate.candidate_id))
        first = HybridCandidateSet(self.guarded.anchor, (expert, vla), self.guarded.attempts)
        second = replace(first, candidates=(vla, expert))
        result_a, result_b = router.route(first), router.route(second)
        self.assertEqual(result_a.selection_space, SelectionSpace.NO_SELECTION_SPACE)
        self.assertEqual(result_a.selected_candidate_id, expert_id)
        self.assertEqual(result_a.selected_candidate_id, result_b.selected_candidate_id)

    def test_selected_safety_executed_and_applied_ids_are_continuous(self) -> None:
        result = H1CandidatePipeline().decide(self.generated)
        selected = result.routing.selected_candidate_id
        self.assertIsNotNone(selected)
        self.assertEqual(result.safety_input_ids, (selected,))
        self.assertEqual(result.safety.decision.executed_trajectory_id, selected)
        safety_input = FrozenH1Router.safety_input(result.guarded_set, result.routing)
        applied = apply_safety_control(
            result.safety.decision,
            safety_input,
            ControlLoop(),
            EgoState(0.0, 0.0, 0.0, 2.0),
            self.generated.anchor.simulation_time_s,
        )
        self.assertTrue(applied.is_track_approved)
        self.assertEqual(applied.executed_id, selected)

    def test_zero_pass_reaches_safety_fallback(self) -> None:
        rejected = tuple(
            replace(
                item,
                guard=GuardResult(
                    candidate_id=item.candidate.candidate_id,
                    verdict=GuardVerdict.REJECT,
                    checks=(),
                    reject_reasons=("test",),
                    latency_ms=0.1,
                ),
            )
            for item in self.guarded.candidates
        )
        cset = replace(self.guarded, candidates=rejected)
        router = FrozenH1Router()
        routing = router.route(cset)
        safety_input = router.safety_input(cset, routing)
        decision = SafetyKernel().tick(
            cset.anchor.safety_snapshot,
            safety_input,
            availability=ComponentAvailability(classic=True, vla=True, world=False, safety=True),
        ).decision
        self.assertIsNone(decision.executed_trajectory_id)
        self.assertIsNotNone(decision.fallback_request)


if __name__ == "__main__":
    unittest.main()
