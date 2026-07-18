from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "g3"))

from run_g3_vla_mpc_stable import (  # noqa: E402
    DEFAULT_RANDOM_MAP_POOL,
    LARGE_MAPS,
    _navigation_targets,
    _polyline_s,
)
from driving_vla.runtime.path_manager import EgoPose  # noqa: E402


class StableRunnerHelpersTest(unittest.TestCase):
    def test_navigation_targets_are_forward_on_valid_route(self) -> None:
        route = [(float(x), 0.0) for x in range(0, 101)]
        first, second, progress, valid = _navigation_targets(
            route,
            _polyline_s(route),
            EgoPose(20.0, 0.0, 0.0, 5.0),
            0.0,
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(first[0], 15.0)
        self.assertAlmostEqual(second[0], 30.0)
        self.assertAlmostEqual(progress, 20.0)

    def test_exhausted_route_is_reported_not_replaced_by_fake_straight_target(self) -> None:
        route = [(float(x), 0.0) for x in range(0, 41)]
        first, second, _progress, valid = _navigation_targets(
            route,
            _polyline_s(route),
            EgoPose(39.5, 0.0, 0.0, 5.0),
            38.0,
        )
        self.assertFalse(valid)
        self.assertLess(first[0], 1.0)
        self.assertLess(second[0], 1.0)

    def test_default_random_pool_excludes_large_maps(self) -> None:
        self.assertTrue(set(DEFAULT_RANDOM_MAP_POOL).isdisjoint(LARGE_MAPS))


if __name__ == "__main__":
    unittest.main()
