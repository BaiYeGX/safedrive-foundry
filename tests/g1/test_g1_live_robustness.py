"""Offline unit tests for live robustness helpers (no CARLA)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "tests" / "g1"))

# Load live script as a module without requiring carla at import time.
_spec = importlib.util.spec_from_file_location(
    "run_g1_classic_expert_live",
    ROOT / "tests" / "g1" / "run_g1_classic_expert_live.py",
)
assert _spec and _spec.loader
_live = importlib.util.module_from_spec(_spec)
# dataclasses need the module registered before exec_module
sys.modules[_spec.name] = _live
try:
    _spec.loader.exec_module(_live)
    HAS_LIVE = True
    _IMPORT_ERR: BaseException | None = None
except Exception as exc:  # noqa: BLE001
    HAS_LIVE = False
    _IMPORT_ERR = exc


@unittest.skipUnless(HAS_LIVE, "live module import failed (carla?)")
class LiveRobustnessHelpers(unittest.TestCase):
    def test_stuck_triggers_after_hold(self) -> None:
        w = _live.StuckWatch(v_max=0.35, prog_eps=0.4, hold_s=1.0, grace_s=0.5)
        self.assertFalse(w.update(now_s=0.0, v=0.0, progress_m=0.0))
        self.assertFalse(w.update(now_s=0.6, v=0.1, progress_m=0.0))  # in grace end
        self.assertFalse(w.update(now_s=1.0, v=0.1, progress_m=0.0))
        self.assertTrue(w.update(now_s=2.1, v=0.1, progress_m=0.0))

    def test_stuck_resets_on_progress(self) -> None:
        w = _live.StuckWatch(v_max=0.35, prog_eps=0.4, hold_s=1.0, grace_s=0.0)
        self.assertFalse(w.update(now_s=0.0, v=0.1, progress_m=0.0))
        self.assertFalse(w.update(now_s=0.5, v=0.1, progress_m=0.0))
        self.assertFalse(w.update(now_s=1.0, v=0.1, progress_m=1.0))  # progress
        self.assertFalse(w.update(now_s=1.5, v=0.1, progress_m=1.0))
        self.assertTrue(w.update(now_s=2.6, v=0.1, progress_m=1.0))

    def test_segment_cruise_uturn_slower(self) -> None:
        self.assertLess(
            _live.segment_cruise_mps("uturn"),
            _live.segment_cruise_mps("approach"),
        )


if __name__ == "__main__":
    unittest.main()
