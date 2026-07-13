"""G2-01 safety state machine debounce / dwell / hysteresis tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from safety_kernel import ComponentAvailability, SafetyMode, load_safety_config  # noqa: E402
from safety_kernel.contracts.types import DecisionKind, SafetyDecision  # noqa: E402
from safety_kernel.state_machine import SafetyStateMachine  # noqa: E402
from safety_kernel.adapters.ros_safety_status import (  # noqa: E402
    ROS_CAUTION,
    ROS_CLEAR,
    ROS_EMERGENCY,
    ROS_INTERVENE,
    safety_mode_to_ros_level,
)


def _decision(kind: DecisionKind, frame: str = "f") -> SafetyDecision:
    return SafetyDecision(
        decision_id="d1",
        run_id="r",
        frame_id=frame,
        prefilter_candidate_ids=(),
        final_candidate_id=None,
        pre_repair_trajectory_id=None,
        post_repair_trajectory_id=None,
        executed_trajectory_id=None,
        constraint_margins=(),
        decision_kind=kind,
        modification_norm=0.0,
        slack=0.0,
        progress_loss=0.0,
        solver_status="n/a",
        latency_ms=0.1,
        state_before=SafetyMode.NORMAL,
        state_after=SafetyMode.NORMAL,
        recovery_conditions=(),
        fallback_request=None,
        reject_reasons=(),
    )


class G201StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_safety_config()
        self.sm = SafetyStateMachine(self.cfg)
        self.avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)

    def test_starts_normal(self) -> None:
        self.assertEqual(self.sm.mode, SafetyMode.NORMAL)

    def test_qp_decision_stays_normal(self) -> None:
        tr = self.sm.step(_decision(DecisionKind.QP), self.avail, now_s=0.0, frame_id="qp0")
        self.assertIsNone(tr)
        self.assertEqual(self.sm.mode, SafetyMode.NORMAL)

    def test_emergency_immediate(self) -> None:
        tr = self.sm.step(_decision(DecisionKind.EMERGENCY), self.avail, now_s=0.0, frame_id="e1")
        self.assertIsNotNone(tr)
        assert tr is not None
        self.assertEqual(tr.to_state, SafetyMode.EMERGENCY)
        self.assertEqual(self.sm.mode, SafetyMode.EMERGENCY)

    def test_escalate_debounce_minimal_risk(self) -> None:
        # First frames may not commit due to debounce.
        transitions = []
        for i in range(self.cfg.escalate_debounce_frames + 2):
            t = i * self.cfg.control_period_s + self.cfg.min_dwell_s
            tr = self.sm.step(
                _decision(DecisionKind.MINIMAL_RISK, frame=f"f{i}"),
                self.avail,
                now_s=t,
                frame_id=f"f{i}",
            )
            if tr:
                transitions.append(tr)
        self.assertTrue(transitions)
        self.assertEqual(self.sm.mode, SafetyMode.MINIMAL_RISK)

    def test_recovery_requires_hysteresis(self) -> None:
        self.sm.reset(SafetyMode.DEGRADED, now_s=0.0)
        # Single ACCEPT should not recover immediately.
        tr = self.sm.step(
            _decision(DecisionKind.ACCEPT),
            self.avail,
            now_s=self.cfg.min_dwell_s + 0.01,
            frame_id="r0",
        )
        self.assertIsNone(tr)
        self.assertEqual(self.sm.mode, SafetyMode.DEGRADED)
        # Many clear frames after dwell.
        for i in range(self.cfg.recover_clear_frames + 1):
            t = self.cfg.min_dwell_s + 0.02 * (i + 1)
            tr = self.sm.step(_decision(DecisionKind.ACCEPT, frame=f"c{i}"), self.avail, now_s=t, frame_id=f"c{i}")
        self.assertEqual(self.sm.mode, SafetyMode.NORMAL)

    def test_errors_not_swallowed(self) -> None:
        self.sm.record_error("unit_test_error")
        self.assertEqual(self.sm.last_error, "unit_test_error")

    def test_ros_level_mapping(self) -> None:
        self.assertEqual(safety_mode_to_ros_level(SafetyMode.NORMAL), ROS_CLEAR)
        self.assertEqual(safety_mode_to_ros_level(SafetyMode.DEGRADED), ROS_CAUTION)
        self.assertEqual(safety_mode_to_ros_level(SafetyMode.MINIMAL_RISK), ROS_INTERVENE)
        self.assertEqual(safety_mode_to_ros_level(SafetyMode.EMERGENCY), ROS_EMERGENCY)


if __name__ == "__main__":
    unittest.main()
