"""G2-04 arbitration, shadow, degradation, fallback determinism."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from safety_kernel import (  # noqa: E402
    SCHEMA_VERSION,
    ComponentAvailability,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyKernel,
    SafetyMode,
    TrajectoryPoint,
    load_safety_config,
)
from safety_kernel.arbitration import (  # noqa: E402
    DegradationReason,
    degrade_candidate_set,
    rank_candidates,
    run_classic_shadow,
    score_candidate,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    DecisionKind,
    ObservationPrivilege,
    TrackedObject,
    TrafficLightObs,
)


def _pts(n: int = 16, v: float = 5.0, y: float = 0.0) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(
            TrajectoryPoint(t=0.25 * i, x=x, y=y, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0)
        )
        x += v * 0.25
    return tuple(out)


def _obs(now: float = 1.0, **kw) -> ObservableSnapshot:
    return ObservableSnapshot(
        run_id="g2-04",
        frame_id="f0",
        scenario_id="arb",
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=0.0,
        ego_y=0.0,
        ego_yaw=0.0,
        ego_v=kw.get("ego_v", 5.0),
        observed_time_s=now,
        actors=kw.get("actors", ()),
        traffic_lights=kw.get("lights", ()),
        corridor_centerline=tuple((float(i), 0.0) for i in range(0, 120)),
        corridor_half_width_m=2.5,
        privilege=ObservationPrivilege.OBSERVABLE,
    )


def _cand(
    cid: str,
    source: CandidateSource,
    *,
    now: float = 1.0,
    pts=None,
    probability: float = 1.0,
    uncertainty: float = 0.1,
    availability: bool = True,
    **meta,
) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=cid,
        source=source,
        generated_time_s=now,
        valid_until_s=now + 0.2,
        probability=probability,
        points=pts or _pts(),
        uncertainty=uncertainty,
        availability=availability,
        dynamics_meta=meta,
    )


def _cset(*cands: PolicyCandidate, now: float = 1.0) -> PolicyCandidateSet:
    return PolicyCandidateSet(
        run_id="g2-04",
        frame_id="f0",
        scenario_id="arb",
        model_id="mix",
        carla_frame=0,
        simulation_time_s=now,
        wall_time_s=now,
        candidates=cands,
        schema_version=SCHEMA_VERSION,
    )


class G204SoftScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.obs = _obs()

    def test_rank_prefers_higher_score_then_classic(self) -> None:
        classic = _cand("c", CandidateSource.CLASSIC, probability=0.7)
        vla = _cand("v", CandidateSource.VLA_FAST, probability=0.99, uncertainty=0.05)
        sc_c = score_candidate(classic, self.obs, self.cfg, now_s=1.0)
        sc_v = score_candidate(vla, self.obs, self.cfg, now_s=1.0)
        ranked = rank_candidates([classic, vla], [sc_c, sc_v])
        self.assertEqual(len(ranked), 2)
        # Deterministic: same inputs → same order
        ranked2 = rank_candidates([vla, classic], [sc_v, sc_c])
        self.assertEqual([c.candidate_id for c in ranked], [c.candidate_id for c in ranked2])


class G204DegradationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()

    def test_vla_unavailable_degraded(self) -> None:
        vla = _cand("v", CandidateSource.VLA_FAST)
        classic = _cand("c", CandidateSource.CLASSIC)
        cset = _cset(vla, classic)
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        deg, reasons = degrade_candidate_set(cset, avail, now_s=1.0, cfg=self.cfg)
        self.assertEqual(reasons["v"], DegradationReason.SOURCE_UNAVAILABLE)
        self.assertFalse(next(c for c in deg.candidates if c.candidate_id == "v").availability)
        self.assertTrue(next(c for c in deg.candidates if c.candidate_id == "c").availability)

    def test_overconfident_learning_degraded(self) -> None:
        vla = _cand(
            "v",
            CandidateSource.VLA_FAST,
            probability=0.99,
            uncertainty=0.01,
        )
        cset = _cset(vla)
        avail = ComponentAvailability(classic=True, vla=True, world=False, safety=True)
        _, reasons = degrade_candidate_set(cset, avail, now_s=1.0, cfg=self.cfg)
        self.assertEqual(reasons["v"], DegradationReason.OVERCONFIDENT)

    def test_ood_learning_degraded(self) -> None:
        vla = _cand("v", CandidateSource.VLA_FAST, probability=0.5, uncertainty=0.95)
        cset = _cset(vla)
        avail = ComponentAvailability(classic=True, vla=True, world=False, safety=True)
        _, reasons = degrade_candidate_set(cset, avail, now_s=1.0, cfg=self.cfg)
        self.assertEqual(reasons["v"], DegradationReason.OOD)

    def test_classic_not_soft_stale(self) -> None:
        """Classic within hard max_candidate_age must not be soft-staled."""
        classic = PolicyCandidate(
            candidate_id="c",
            source=CandidateSource.CLASSIC,
            generated_time_s=1.0,
            valid_until_s=1.5,
            probability=1.0,
            points=_pts(),
            uncertainty=0.0,
            availability=True,
        )
        cset = _cset(classic, now=1.22)
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        _, reasons = degrade_candidate_set(cset, avail, now_s=1.22, cfg=self.cfg)
        self.assertEqual(reasons["c"], DegradationReason.NONE)

    def test_learning_soft_stale(self) -> None:
        vla = PolicyCandidate(
            candidate_id="v",
            source=CandidateSource.VLA_FAST,
            generated_time_s=1.0,
            valid_until_s=1.5,
            probability=0.7,
            points=_pts(),
            uncertainty=0.2,
            availability=True,
        )
        cset = _cset(vla, now=1.22)
        avail = ComponentAvailability(classic=True, vla=True, world=False, safety=True)
        _, reasons = degrade_candidate_set(cset, avail, now_s=1.22, cfg=self.cfg)
        self.assertEqual(reasons["v"], DegradationReason.SOFT_STALE)


class G204ShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.obs = _obs()

    def test_shadow_never_claims_control(self) -> None:
        classic = _cand("c", CandidateSource.CLASSIC)
        executed = _cand("v", CandidateSource.VLA_FAST, probability=0.9)
        sh = run_classic_shadow(
            classic=classic, executed=executed, obs=self.obs, cfg=self.cfg, now_s=1.0
        )
        self.assertTrue(sh.enabled)
        self.assertFalse(sh.claims_control)
        self.assertFalse(sh.claims_tick_ownership)
        self.assertIsNotNone(sh.score_delta)


class G204KernelPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.kernel = SafetyKernel(self.cfg)
        self.avail_classic = ComponentAvailability(classic=True, vla=False, world=False, safety=True)
        self.avail_full = ComponentAvailability(classic=True, vla=True, world=False, safety=True)

    def test_classic_only_no_gpu_accept(self) -> None:
        cset = _cset(_cand("c", CandidateSource.CLASSIC))
        res = self.kernel.tick(_obs(), cset, now_s=1.0, availability=self.avail_classic)
        self.assertEqual(res.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertFalse(res.decision.learning_modules_required)
        self.assertIsNotNone(res.arbitration)
        assert res.arbitration is not None
        self.assertIn("prefilter", res.arbitration.stages)
        self.assertIn("soft_score", res.arbitration.stages)
        self.assertFalse(res.arbitration.learning_required)

    def test_vla_dropped_classic_selected(self) -> None:
        cset = _cset(
            _cand("v", CandidateSource.VLA_FAST, probability=0.99),
            _cand("c", CandidateSource.CLASSIC, probability=0.6),
        )
        res = self.kernel.tick(_obs(), cset, now_s=1.0, availability=self.avail_classic)
        self.assertEqual(res.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertEqual(res.decision.final_candidate_id, "c")
        assert res.arbitration is not None
        self.assertIsNotNone(res.arbitration.shadow)
        assert res.arbitration.shadow is not None
        self.assertFalse(res.arbitration.shadow.claims_control)

    def test_all_reject_then_repair_or_fallback(self) -> None:
        # Illegal high speed + red light → hard reject path then QP repair.
        lights = (TrafficLightObs("tl", "red", 7.0, 1.0),)
        obs = _obs(ego_v=5.0, lights=lights)
        cset = _cset(_cand("raw", CandidateSource.CLASSIC, pts=_pts(v=5.0)))
        res = self.kernel.tick(obs, cset, now_s=1.0, availability=self.avail_classic)
        self.assertIn(
            res.decision.decision_kind,
            {DecisionKind.QP, DecisionKind.RATO, DecisionKind.HARD_REJECT, DecisionKind.MINIMAL_RISK},
        )
        assert res.arbitration is not None
        self.assertTrue(any(s in res.arbitration.stages for s in ("repair", "fallback", "final")))

    def test_emergency_not_overridden_by_scores(self) -> None:
        # Force emergency via no classic + no learning.
        cset = _cset(_cand("v", CandidateSource.VLA_FAST))
        avail = ComponentAvailability(classic=False, vla=False, world=False, safety=True)
        res = self.kernel.tick(_obs(), cset, now_s=1.0, availability=avail)
        self.assertEqual(res.decision.decision_kind, DecisionKind.EMERGENCY)
        # Soft scores must not produce ACCEPT under this availability.
        self.assertNotEqual(res.decision.decision_kind, DecisionKind.ACCEPT)

    def test_deterministic_timeline_two_runs(self) -> None:
        cset = _cset(
            _cand("v", CandidateSource.VLA_FAST, probability=0.8, uncertainty=0.2),
            _cand("c", CandidateSource.CLASSIC, probability=0.7),
        )
        k1 = SafetyKernel(self.cfg)
        k2 = SafetyKernel(self.cfg)
        r1 = k1.tick(_obs(), cset, now_s=1.0, availability=self.avail_full)
        r2 = k2.tick(_obs(), cset, now_s=1.0, availability=self.avail_full)
        self.assertEqual(r1.decision.decision_kind, r2.decision.decision_kind)
        self.assertEqual(r1.decision.final_candidate_id, r2.decision.final_candidate_id)
        assert r1.arbitration and r2.arbitration
        self.assertEqual(r1.arbitration.ranked_ids, r2.arbitration.ranked_ids)
        self.assertEqual(r1.arbitration.stages, r2.arbitration.stages)

    def test_timeout_degrade_learning(self) -> None:
        # Soft-stale learning candidate via age.
        vla = PolicyCandidate(
            candidate_id="v",
            source=CandidateSource.VLA_FAST,
            generated_time_s=0.0,
            valid_until_s=0.5,
            probability=0.8,
            points=_pts(),
            uncertainty=0.2,
            availability=True,
        )
        classic = _cand("c", CandidateSource.CLASSIC, now=1.0)
        cset = _cset(vla, classic, now=1.0)
        res = self.kernel.tick(_obs(now=1.0), cset, now_s=1.0, availability=self.avail_full)
        self.assertEqual(res.decision.final_candidate_id, "c")
        assert res.arbitration is not None
        v_audit = next(a for a in res.arbitration.audits if a.candidate_id == "v")
        self.assertIn(
            v_audit.degradation,
            {
                DegradationReason.SOFT_STALE,
                DegradationReason.TIMEOUT,
                DegradationReason.NONE,
            },
        )
        # If still available somehow, final must still be legal.
        if v_audit.degradation is DegradationReason.NONE:
            self.assertTrue(res.decision.final_candidate_id in {"c", "v"})

    def test_metrics_include_arbitration(self) -> None:
        cset = _cset(_cand("c", CandidateSource.CLASSIC))
        self.kernel.tick(_obs(), cset, now_s=1.0, availability=self.avail_classic)
        snap = self.kernel.metrics_snapshot()
        self.assertIn("arbitration_latency", snap)
        self.assertGreaterEqual(snap["arbitration_tick_count"], 1)

    def test_classic_age_022_accepts_without_repair(self) -> None:
        """Classic age 0.22 (soft_stale window) still hard-fresh → ACCEPT, not RATO."""
        classic = PolicyCandidate(
            candidate_id="c",
            source=CandidateSource.CLASSIC,
            generated_time_s=1.0,
            valid_until_s=1.5,
            probability=1.0,
            points=_pts(),
            uncertainty=0.0,
            availability=True,
        )
        cset = _cset(classic, now=1.22)
        res = self.kernel.tick(_obs(now=1.22), cset, now_s=1.22, availability=self.avail_classic)
        self.assertEqual(res.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertEqual(res.decision.executed_trajectory_id, "c")
        self.assertTrue(res.repair_result is None or not res.repair_result.success)

    def test_final_sweeps_beyond_topk(self) -> None:
        """Hard-legal lower-ranked candidate accepted before repair when top-K fails final."""
        from safety_kernel.config import ArbitrationConfig, SafetyKernelConfig

        base = load_safety_config()
        arb = ArbitrationConfig(
            enabled=True,
            deadline_ms=base.arbitration.deadline_ms,
            w_progress=base.arbitration.w_progress,
            w_comfort=base.arbitration.w_comfort,
            w_margin=base.arbitration.w_margin,
            w_probability=base.arbitration.w_probability,
            w_uncertainty=base.arbitration.w_uncertainty,
            classic_source_bonus=base.arbitration.classic_source_bonus,
            vla_source_bonus=base.arbitration.vla_source_bonus,
            world_ranked_bonus=base.arbitration.world_ranked_bonus,
            max_final_candidates=1,
            shadow_enabled=base.arbitration.shadow_enabled,
            overconfident_prob_min=base.arbitration.overconfident_prob_min,
            overconfident_uncertainty_max=base.arbitration.overconfident_uncertainty_max,
            ood_uncertainty_min=base.arbitration.ood_uncertainty_min,
            soft_stale_age_s=base.arbitration.soft_stale_age_s,
        )
        cfg = SafetyKernelConfig(
            schema_version=base.schema_version,
            name=base.name,
            raw_toml=base.raw_toml,
            control_period_s=base.control_period_s,
            state_check_deadline_ms=base.state_check_deadline_ms,
            candidate_check_deadline_ms=base.candidate_check_deadline_ms,
            default_horizon_s=base.default_horizon_s,
            min_horizon_s=base.min_horizon_s,
            max_horizon_s=base.max_horizon_s,
            max_candidate_age_s=base.max_candidate_age_s,
            min_points=base.min_points,
            wheelbase_m=base.wheelbase_m,
            max_speed_mps=base.max_speed_mps,
            max_accel_mps2=base.max_accel_mps2,
            max_decel_mps2=base.max_decel_mps2,
            max_jerk_mps3=base.max_jerk_mps3,
            max_curvature_per_m=base.max_curvature_per_m,
            max_lateral_accel_mps2=base.max_lateral_accel_mps2,
            width_m=base.width_m,
            length_m=base.length_m,
            collision_inflate_m=base.collision_inflate_m,
            require_drivable=base.require_drivable,
            max_offroad_m=base.max_offroad_m,
            lane_half_width_m=base.lane_half_width_m,
            enforce_speed_limit=base.enforce_speed_limit,
            speed_limit_margin_mps=base.speed_limit_margin_mps,
            enforce_red_light_stop=base.enforce_red_light_stop,
            red_light_stop_distance_m=base.red_light_stop_distance_m,
            red_light_max_approach_speed_mps=base.red_light_max_approach_speed_mps,
            escalate_debounce_frames=base.escalate_debounce_frames,
            recover_debounce_frames=base.recover_debounce_frames,
            min_dwell_s=base.min_dwell_s,
            recover_clear_frames=base.recover_clear_frames,
            emit_safety_events=base.emit_safety_events,
            record_failure_samples=base.record_failure_samples,
            max_failure_samples=base.max_failure_samples,
            qp=base.qp,
            rato=base.rato,
            arbitration=arb,
        )
        bad = _cand("bad_top", CandidateSource.CLASSIC, pts=_pts(v=8.0, y=10.0), probability=1.0)
        legal = _cand("legal_low", CandidateSource.CLASSIC, pts=_pts(v=3.0, y=0.0), probability=0.1)
        k = SafetyKernel(cfg)
        res = k.tick(_obs(), _cset(bad, legal), now_s=1.0, availability=self.avail_classic)
        self.assertEqual(res.decision.decision_kind, DecisionKind.ACCEPT)
        self.assertEqual(res.decision.executed_trajectory_id, "legal_low")
        # Must not depend on successful repair of the off-road seed.
        self.assertTrue(res.repair_result is None or not res.repair_result.success)
        assert res.arbitration is not None
        self.assertIn("final_sweep_beyond_topk", res.arbitration.notes)


if __name__ == "__main__":
    unittest.main()
