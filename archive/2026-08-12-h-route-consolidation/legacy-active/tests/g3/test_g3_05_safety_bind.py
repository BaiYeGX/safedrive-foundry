"""S1: Safety control bind must never bypass kernel (no force throttle / first-available)."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from classic_stack.control.controller import EgoState  # noqa: E402
from driving_vla.runtime.safety_control_bind import (  # noqa: E402
    AppliedMode,
    apply_safety_control,
    evaluate_episode_status,
    resolve_executable_candidate,
)
from safety_kernel.contracts.types import (  # noqa: E402
    DecisionKind,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyDecision,
    SafetyMode,
    TrajectoryPoint,
    CandidateSource,
)


def _pts(n: int = 10) -> tuple[TrajectoryPoint, ...]:
    return tuple(
        TrajectoryPoint(t=(i + 1) * 0.25, x=float(i), y=0.0, yaw=0.0, kappa=0.0, v=4.0, a=0.0)
        for i in range(n)
    )


def _cand(cid: str, *, avail: bool = True) -> PolicyCandidate:
    return PolicyCandidate(
        candidate_id=cid,
        source=CandidateSource.VLA_FAST,
        generated_time_s=1.0,
        valid_until_s=2.0,
        probability=0.9,
        points=_pts(),
        availability=avail,
    )


def _decision(
    kind: DecisionKind,
    *,
    exec_id: str | None = None,
    accepted: PolicyCandidate | None = None,
) -> SafetyDecision:
    return SafetyDecision(
        decision_id="d1",
        run_id="r",
        frame_id="f",
        prefilter_candidate_ids=(),
        final_candidate_id=exec_id,
        pre_repair_trajectory_id=None,
        post_repair_trajectory_id=None,
        executed_trajectory_id=exec_id,
        constraint_margins=(),
        decision_kind=kind,
        modification_norm=0.0,
        slack=0.0,
        progress_loss=0.0,
        solver_status="ok",
        latency_ms=1.0,
        state_before=SafetyMode.NORMAL,
        state_after=SafetyMode.NORMAL if kind is DecisionKind.ACCEPT else SafetyMode.EMERGENCY,
        recovery_conditions=(),
        fallback_request=None,
        reject_reasons=(),
        accepted_candidate=accepted,
    )


@dataclass
class _Cmd:
    throttle: float
    brake: float
    steer: float
    mode: str = "mpc"
    solver_ms: float = 0.0
    e2e_ms: float = 0.0
    deadline_miss: bool = False
    reason: str = ""


class _MockControl:
    def __init__(self) -> None:
        self.traj_ids: list[str] = []
        self.steps = 0

    def set_trajectory(self, traj: Any, now_s: float) -> None:
        self.traj_ids.append(getattr(traj, "trajectory_id", "x"))

    def step(self, ego: EgoState, now_s: float, **kwargs: Any) -> _Cmd:
        self.steps += 1
        return _Cmd(throttle=0.4, brake=0.0, steer=0.1)


class TestResolveExecutable(unittest.TestCase):
    def test_no_first_available_on_missing_id(self) -> None:
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("a"), _cand("b")),
            schema_version="1",
        )
        d = _decision(DecisionKind.ACCEPT, exec_id=None)
        got, notes = resolve_executable_candidate(d, cset)
        self.assertIsNone(got)
        self.assertIn("no_executed_trajectory_id", notes)

    def test_orphan_id_not_first_available(self) -> None:
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("a"),),
            schema_version="1",
        )
        d = _decision(DecisionKind.ACCEPT, exec_id="missing")
        got, notes = resolve_executable_candidate(d, cset)
        self.assertIsNone(got)
        self.assertIn("exec_id_orphan", notes)

    def test_match_by_id(self) -> None:
        c = _cand("want")
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("other"), c),
            schema_version="1",
        )
        d = _decision(DecisionKind.ACCEPT, exec_id="want")
        got, _ = resolve_executable_candidate(d, cset)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.candidate_id, "want")


class TestApplySafetyControl(unittest.TestCase):
    def setUp(self) -> None:
        self.ctrl = _MockControl()
        self.ego = EgoState(x=0.0, y=0.0, yaw=0.0, v=1.0, steer=0.05)

    def test_emergency_forces_brake_no_track(self) -> None:
        d = _decision(DecisionKind.EMERGENCY, exec_id=None)
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("a"),),
            schema_version="1",
        )
        out = apply_safety_control(d, cset, self.ctrl, self.ego, 1.0)
        self.assertEqual(out.applied_mode, AppliedMode.EMERGENCY_BRAKE)
        self.assertEqual(out.throttle, 0.0)
        self.assertGreater(out.brake, 0.5)
        self.assertEqual(self.ctrl.steps, 0)
        self.assertEqual(self.ctrl.traj_ids, [])

    def test_accept_tracks_approved(self) -> None:
        c = _cand("ok")
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(c,),
            schema_version="1",
        )
        d = _decision(DecisionKind.ACCEPT, exec_id="ok", accepted=c)
        out = apply_safety_control(d, cset, self.ctrl, self.ego, 1.0)
        self.assertEqual(out.applied_mode, AppliedMode.TRACK_APPROVED)
        self.assertEqual(out.executed_id, "ok")
        self.assertGreater(out.throttle, 0.0)
        self.assertEqual(self.ctrl.steps, 1)
        self.assertEqual(len(self.ctrl.traj_ids), 1)

    def test_missing_id_no_first_available_drive(self) -> None:
        cset = PolicyCandidateSet(
            run_id="r",
            frame_id="f",
            scenario_id="s",
            model_id="m",
            carla_frame=0,
            simulation_time_s=1.0,
            wall_time_s=1.0,
            candidates=(_cand("a"),),
            schema_version="1",
        )
        d = _decision(DecisionKind.ACCEPT, exec_id=None)
        out = apply_safety_control(d, cset, self.ctrl, self.ego, 1.0)
        self.assertNotEqual(out.applied_mode, AppliedMode.TRACK_APPROVED)
        self.assertEqual(out.throttle, 0.0)
        self.assertEqual(self.ctrl.steps, 0)


class TestEpisodeStatus(unittest.TestCase):
    def test_old_emergency_long_run_fails(self) -> None:
        status, reasons = evaluate_episode_status(
            steps=120,
            max_steps=120,
            min_steps=80,
            distance_m=140.0,
            decisions=["EMERGENCY"] * 120,
            n_track_approved=0,
            n_emergency=120,
            sources_seen=["vla_fast"],
            camera_frames=120,
            fault="",
            mode="VLA_SAFETY",
            classic_current_frame=False,
        )
        self.assertEqual(status, "FAILED")
        self.assertTrue(any("emergency" in r or "track" in r for r in reasons))

    def test_timeout_empty_sources_long_distance_fails(self) -> None:
        status, reasons = evaluate_episode_status(
            steps=100,
            max_steps=100,
            min_steps=80,
            distance_m=138.0,
            decisions=["ACCEPT"] * 15,
            n_track_approved=0,
            n_emergency=0,
            sources_seen=[],
            camera_frames=100,
            fault="timeout",
            mode="VLA_SAFETY",
            classic_current_frame=False,
        )
        self.assertEqual(status, "FAILED")
        self.assertTrue(any("timeout_long" in r for r in reasons))

    def test_healthy_episode_completes(self) -> None:
        status, reasons = evaluate_episode_status(
            steps=100,
            max_steps=100,
            min_steps=80,
            distance_m=50.0,
            decisions=["ACCEPT"] * 80 + ["QP"] * 20,
            n_track_approved=90,
            n_emergency=5,
            sources_seen=["vla_fast"],
            camera_frames=100,
            fault="",
            mode="VLA_SAFETY",
            classic_current_frame=False,
        )
        self.assertEqual(status, "COMPLETED", reasons)
        self.assertEqual(reasons, [])


class TestAssertG3CloseRejectsLegacy(unittest.TestCase):
    def test_legacy_summary_fails_close_gate(self) -> None:
        """Old neural_live latest must not satisfy strengthened close logic fields."""
        legacy = ROOT / "docs/runtime-evidence/g3-05/neural_live/latest_live_summary.json"
        if not legacy.is_file():
            self.skipTest("legacy evidence not present")
        data = json.loads(legacy.read_text(encoding="utf-8"))
        # Structural defects used by evaluate_episode_status / assert
        self.assertTrue(data.get("all_ok"))
        for r in data.get("results") or []:
            self.assertNotIn("n_track_approved", r)
            tail = r.get("decision_tail") or []
            if float(r.get("distance_m") or 0) > 20 and tail:
                self.assertTrue(all(d == "EMERGENCY" for d in tail))
        # Running assert_g3_close as subprocess would need env; field-level is enough here
        status, reasons = evaluate_episode_status(
            steps=int((data["results"][0]).get("steps") or 0),
            max_steps=120,
            min_steps=80,
            distance_m=float((data["results"][0]).get("distance_m") or 0),
            decisions=list((data["results"][0]).get("decision_tail") or []),
            n_track_approved=0,
            n_emergency=15,
            sources_seen=list((data["results"][0]).get("sources_seen") or []),
            camera_frames=int((data["results"][0]).get("camera_frames") or 0),
            fault="",
            mode="VLA_SAFETY",
            classic_current_frame=False,
        )
        self.assertEqual(status, "FAILED")
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
