"""Yaw-invariant vehicle geometry unit tests (no CARLA)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.runtime.vehicle_geometry import (  # noqa: E402
    DEFAULT_EGO_BLUEPRINT,
    MAX_STEER_PLAUSIBLE_RAD,
    MERCEDES_COUPE_2020_VALIDATED_FALLBACK,
    estimate_vehicle_geometry_from_wheels,
    estimate_wheelbase_track_m,
    normalize_wheel_units,
    plausible_max_steer_rad,
)


def _rect_wheels(wheelbase: float, track: float, yaw_rad: float) -> list[list[float]]:
    """Four wheels of a rectangle; yaw rotates the vehicle body."""
    half_l = wheelbase / 2.0
    half_t = track / 2.0
    local = [
        [half_l, half_t],
        [half_l, -half_t],
        [-half_l, half_t],
        [-half_l, -half_t],
    ]
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out: list[list[float]] = []
    for x, y in local:
        out.append([c * x - s * y, s * x + c * y])
    return out


class VehicleGeometryYawInvariantTest(unittest.TestCase):
    def test_wheelbase_track_invariant_under_yaw(self) -> None:
        wb, tr = 2.5, 1.5
        for deg in (0.0, 45.0, 90.0, 180.0):
            pts = _rect_wheels(wb, tr, math.radians(deg))
            est_wb, est_tr = estimate_wheelbase_track_m(pts)
            self.assertAlmostEqual(
                est_wb,
                wb,
                places=5,
                msg=f"wheelbase changed at yaw={deg}",
            )
            self.assertAlmostEqual(
                est_tr,
                tr,
                places=5,
                msg=f"track changed at yaw={deg}",
            )

    def test_yaw_90_does_not_swap_1_5_as_wheelbase(self) -> None:
        """Regression: max(x)-min(x) at yaw=90° would report track as wheelbase."""
        pts = _rect_wheels(2.5, 1.5, math.radians(90.0))
        # World-X extent is track (1.5), not wheelbase — old bug.
        xs = [p[0] for p in pts]
        self.assertAlmostEqual(max(xs) - min(xs), 1.5, places=5)
        est_wb, est_tr = estimate_wheelbase_track_m(pts)
        self.assertAlmostEqual(est_wb, 2.5, places=5)
        self.assertAlmostEqual(est_tr, 1.5, places=5)
        self.assertNotAlmostEqual(est_wb, 1.5, places=2)

    def test_cm_units_normalized(self) -> None:
        # Same 2.5×1.5 rectangle in centimetres.
        pts_cm = [[x * 100.0, y * 100.0] for x, y in _rect_wheels(2.5, 1.5, 0.0)]
        pts_m, units = normalize_wheel_units(pts_cm)
        self.assertEqual(units, "cm_to_m")
        est_wb, est_tr = estimate_wheelbase_track_m(pts_m)
        self.assertAlmostEqual(est_wb, 2.5, places=4)
        self.assertAlmostEqual(est_tr, 1.5, places=4)

    def test_world_frame_offset_invariant(self) -> None:
        pts = _rect_wheels(2.5, 1.5, math.radians(33.0))
        shifted = [[x + 100.0, y - 50.0] for x, y in pts]
        a = estimate_wheelbase_track_m(pts)
        b = estimate_wheelbase_track_m(shifted)
        self.assertAlmostEqual(a[0], b[0], places=5)
        self.assertAlmostEqual(a[1], b[1], places=5)

    def test_full_estimate_ok_on_valid_physics(self) -> None:
        pts = _rect_wheels(2.70, 1.55, math.radians(90.0))
        result = estimate_vehicle_geometry_from_wheels(
            pts,
            max_steer_deg=40.0,
            blueprint_id=DEFAULT_EGO_BLUEPRINT,
            allow_mercedes_fallback=True,
        )
        self.assertEqual(result.geometry_source, "physics_wheels_pca")
        self.assertEqual(result.validation_status, "ok")
        self.assertAlmostEqual(result.wheelbase_m, 2.70, places=4)
        self.assertAlmostEqual(result.track_width_m, 1.55, places=4)

    def test_implausible_wheels_use_field_fallback_but_keep_good_steer(self) -> None:
        # Degenerate wheels → wb/track fallback; 40° steer stays physics.
        bad = [[0.0, 0.0], [0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]
        result = estimate_vehicle_geometry_from_wheels(
            bad,
            max_steer_deg=40.0,
            blueprint_id=DEFAULT_EGO_BLUEPRINT,
            allow_mercedes_fallback=True,
        )
        self.assertEqual(result.validation_status, "partial_fallback")
        self.assertAlmostEqual(
            result.wheelbase_m,
            MERCEDES_COUPE_2020_VALIDATED_FALLBACK["wheelbase_m"],
        )
        self.assertEqual(result.fields["wheelbase_m"]["validation_status"], "fallback_used")
        self.assertEqual(result.fields["max_steer_rad"]["validation_status"], "ok")
        self.assertAlmostEqual(result.max_steer_rad, math.radians(40.0), places=5)

    def test_mercedes_70deg_steer_accepted_with_physics_wb_track(self) -> None:
        """Live regression: 70° (~1.222 rad) must not wholesale-fallback geometry."""
        pts = _rect_wheels(2.833, 1.598, math.radians(90.0))
        steer = math.radians(70.0)
        self.assertTrue(plausible_max_steer_rad(steer))
        self.assertLess(steer, MAX_STEER_PLAUSIBLE_RAD)
        result = estimate_vehicle_geometry_from_wheels(
            pts,
            max_steer_rad=steer,
            blueprint_id=DEFAULT_EGO_BLUEPRINT,
            allow_mercedes_fallback=True,
        )
        self.assertEqual(result.geometry_source, "physics_wheels_pca")
        self.assertEqual(result.validation_status, "ok")
        self.assertAlmostEqual(result.wheelbase_m, 2.833, places=3)
        self.assertAlmostEqual(result.track_width_m, 1.598, places=3)
        self.assertAlmostEqual(result.max_steer_rad, steer, places=5)
        for name in ("wheelbase_m", "track_width_m", "max_steer_rad"):
            self.assertEqual(result.fields[name]["validation_status"], "ok")

    def test_bad_steer_does_not_discard_valid_wheelbase_track(self) -> None:
        pts = _rect_wheels(2.833, 1.598, 0.0)
        # Absurd steer (200°) → only steer falls back.
        result = estimate_vehicle_geometry_from_wheels(
            pts,
            max_steer_deg=200.0,
            blueprint_id=DEFAULT_EGO_BLUEPRINT,
            allow_mercedes_fallback=True,
        )
        self.assertEqual(result.validation_status, "partial_fallback")
        self.assertAlmostEqual(result.wheelbase_m, 2.833, places=3)
        self.assertAlmostEqual(result.track_width_m, 1.598, places=3)
        self.assertEqual(result.fields["wheelbase_m"]["validation_status"], "ok")
        self.assertEqual(result.fields["track_width_m"]["validation_status"], "ok")
        self.assertEqual(result.fields["max_steer_rad"]["validation_status"], "fallback_used")
        self.assertAlmostEqual(
            result.max_steer_rad,
            MERCEDES_COUPE_2020_VALIDATED_FALLBACK["max_steer_rad"],
        )

    def test_non_mercedes_implausible_raises(self) -> None:
        bad = [[0.0, 0.0], [0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]
        with self.assertRaises(RuntimeError):
            estimate_vehicle_geometry_from_wheels(
                bad,
                max_steer_deg=40.0,
                blueprint_id="vehicle.audi.a2",
                allow_mercedes_fallback=True,
            )

    def test_default_blueprint_constant(self) -> None:
        self.assertEqual(DEFAULT_EGO_BLUEPRINT, "vehicle.mercedes.coupe_2020")


if __name__ == "__main__":
    unittest.main()
