from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.navigation_contract import (  # noqa: E402
    RouteManeuver,
    build_route_context,
)
from driving_vla.world.contracts import WorldContractError  # noqa: E402
from driving_vla.world.navigation_batch import (  # noqa: E402
    NAVIGATION_FEATURES,
    RouteBoundWorldBatch,
    WorldNavigationCondition,
    v3_world_audit_fields,
)
from tests.world.helpers import make_sample  # noqa: E402


def _route():
    points = tuple((float(index), 0.0) for index in range(24))
    return build_route_context(
        points,
        maneuver=RouteManeuver.JUNCTION_STRAIGHT,
        junction_flags=[True] * len(points),
    )


class RouteBoundWorldBatchTest(unittest.TestCase):
    def test_navigation_is_input_and_candidates_share_binding(self) -> None:
        context = _route()
        condition = WorldNavigationCondition.from_route_context(context)
        sample = make_sample()
        sample.audit = v3_world_audit_fields(
            condition=condition,
            candidate_route_hashes=[context.route_hash] * 2,
            candidate_topology_hashes=[context.topology_hash] * 2,
            guard_candidate_valid=[True, True],
        )
        batch = RouteBoundWorldBatch.from_samples([sample], [context])
        self.assertEqual(batch.navigation.shape, (1, NAVIGATION_FEATURES))
        self.assertEqual(
            batch.route_maneuvers, (RouteManeuver.JUNCTION_STRAIGHT.value,)
        )
        self.assertEqual(batch.route_hashes, (context.route_hash,))

    def test_wrong_candidate_route_fails_closed(self) -> None:
        context = _route()
        condition = WorldNavigationCondition.from_route_context(context)
        sample = make_sample()
        sample.audit = v3_world_audit_fields(
            condition=condition,
            candidate_route_hashes=[context.route_hash, "bad-route"],
            candidate_topology_hashes=[context.topology_hash] * 2,
            guard_candidate_valid=[True, True],
        )
        with self.assertRaises(WorldContractError):
            RouteBoundWorldBatch.from_samples([sample], [context])

    def test_guard_rejected_candidate_cannot_be_world_available(self) -> None:
        context = _route()
        condition = WorldNavigationCondition.from_route_context(context)
        sample = make_sample()
        sample.audit = v3_world_audit_fields(
            condition=condition,
            candidate_route_hashes=[context.route_hash] * 2,
            candidate_topology_hashes=[context.topology_hash] * 2,
            guard_candidate_valid=[True, False],
        )
        with self.assertRaises(WorldContractError):
            RouteBoundWorldBatch.from_samples([sample], [context])


if __name__ == "__main__":
    unittest.main()
