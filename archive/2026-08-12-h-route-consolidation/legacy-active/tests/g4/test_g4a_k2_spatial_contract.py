"""R2-G/X0 Spatial K2 V2 contract + Guard V2 + selector force semantics."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.k2_spatial_builder import (  # noqa: E402
    build_spatial_k2_bundle_from_residuals,
    synthetic_diverse_residuals,
)
from driving_vla.model.k2_spatial_guard import (  # noqa: E402
    attach_spatial_guard,
    validate_k2_spatial_bundle,
    verify_execution_spec_content_hash,
)
from driving_vla.model.k2_spatial_types import (  # noqa: E402
    CURVATURE_ENVELOPE,
    GUARD_OK,
    GUARD_REJECT,
    HEAD_LINEAGE_INVALID,
    NATIVE_CORRIDOR,
    PATH_TOO_SHORT,
    PROPOSAL_PATH_HASH_MISMATCH,
    SPATIAL_COLLAPSE_ELIGIBLE,
    TIMED_XTRACK,
    load_k2_spatial_config,
    stable_hash_xy,
)
from driving_vla.model.k2_builder import load_k2_config  # noqa: E402
from driving_vla.runtime.k2_execution import K2SelectionError, select_k2_spatial  # noqa: E402


def _native_line(n: int = 20, ds: float = 1.0):
    return tuple((float(i * ds), 0.0) for i in range(n))


def _diverse_bundle(*, defensive_available: bool = True, eligible: bool = True):
    native = _native_line(20)
    nom, alt = synthetic_diverse_residuals(20, lateral_sign=1.0, lineage="contract_probe")
    alt["raw_d"] = [min(2.5, 0.35 * i) for i in range(20)]
    bundle = build_spatial_k2_bundle_from_residuals(
        native_path_xy=native,
        ego_xy=(0.0, 0.0),
        ego_v=5.0,
        base_speed_mps=6.0,
        residual_nominal=nom,
        residual_defensive=alt,
        observation_identity={"frame_id": "t0"},
        backbone_forward_id="fwd-1",
        defensive_available=defensive_available,
        defensive_reason="ok" if defensive_available else "NO_ALTERNATIVE",
    )
    return replace(
        bundle,
        set_diagnostics={
            **dict(bundle.set_diagnostics),
            "eligible_for_diversity": eligible and defensive_available,
        },
    )


class SpatialK2ContractTest(unittest.TestCase):
    def test_load_v2_toml(self) -> None:
        cfg = load_k2_spatial_config()
        self.assertEqual(cfg.k, 2)
        self.assertLessEqual(cfg.max_lateral_residual_m, 1.0)
        self.assertGreaterEqual(cfg.ambiguity_min_spatial_sep_m, 0.5)
        self.assertGreater(cfg.native_corridor_cross_track_max_m, 0.0)
        self.assertGreater(cfg.hard_max_abs_curvature, 0.0)
        self.assertEqual(
            cfg.availability_semantics, "executability_only_v1"
        )
        self.assertTrue(cfg.learned_confidence_non_blocking)
        self.assertTrue(cfg.nominal_anchor_exact)
        # formal lineage uses full SHA256 (64 hex chars)
        self.assertEqual(len(cfg.config_hash()), 64)

    def test_learned_confidence_is_probability_not_executability(self) -> None:
        native = _native_line(20)
        nom, alt = synthetic_diverse_residuals(
            20, lateral_sign=1.0, lineage="contract_probe"
        )
        alt["raw_d"] = [min(2.5, 0.35 * i) for i in range(20)]
        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=(0.0, 0.0),
            ego_v=5.0,
            base_speed_mps=6.0,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={},
            backbone_forward_id="fwd-confidence",
            defensive_available=True,
            defensive_reason="PROPOSAL_SPATIALLY_DISTINCT",
            nominal_probability=0.99,
            defensive_probability=0.01,
            probability_source="learned_mode_confidence_non_blocking",
        )
        guarded = attach_spatial_guard(
            bundle, require_diversity_if_eligible=True
        )
        self.assertEqual(guarded.guard_status, GUARD_OK)
        self.assertTrue(guarded.candidates[1].available)
        self.assertAlmostEqual(guarded.candidates[1].probability, 0.01)
        selected = select_k2_spatial(
            guarded, mode="force", force_index=1
        )
        self.assertEqual(selected.candidate_index, 1)

    def test_config_hash_covers_corridor_params(self) -> None:
        cfg = load_k2_spatial_config()
        h1 = cfg.config_hash()
        # mutate a previously-unhashed field via replace on dataclass
        from dataclasses import replace as dc_replace

        cfg2 = dc_replace(cfg, native_corridor_cross_track_max_m=cfg.native_corridor_cross_track_max_m + 0.1)
        self.assertNotEqual(h1, cfg2.config_hash())

    def test_diverse_bundle_guard_ok(self) -> None:
        bundle = _diverse_bundle()
        guarded = attach_spatial_guard(bundle, require_diversity_if_eligible=True)
        self.assertEqual(guarded.guard_status, GUARD_OK, msg=guarded.guard_reasons)
        c0, c1 = guarded.candidates
        self.assertNotEqual(c0.spatial_path_xy, c1.spatial_path_xy)
        self.assertNotEqual(c0.proposal_path_hash, c1.proposal_path_hash)
        # geometry metrics populated
        self.assertIn("max_lateral_separation_m", guarded.set_diagnostics)
        self.assertEqual(c0.spatial_path_xy, bundle.native_path_xy)
        self.assertEqual(c0.proposal_path_hash, bundle.native_path_hash)

    def test_same_direction_excursion_is_not_candidate_diversity(self) -> None:
        """Two paths far from native but close to each other must collapse."""
        from dataclasses import replace as dc_replace

        native = _native_line(20)
        cfg = dc_replace(load_k2_spatial_config(), nominal_anchor_exact=False)
        common = {
            "raw_delta_s": [0.5] * 20,
            "speed_scale": 1.0,
            "head_lineage": "spatial_mode_head",
        }
        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=(0.0, 0.0),
            ego_v=5.0,
            base_speed_mps=6.0,
            residual_nominal={**common, "raw_d": [1.0] * 20},
            residual_defensive={**common, "raw_d": [1.2] * 20},
            observation_identity={},
            backbone_forward_id="fwd-same-direction",
            defensive_available=True,
            config=cfg,
        )
        status, reasons, metrics = validate_k2_spatial_bundle(
            bundle, config=cfg, require_diversity_if_eligible=True
        )
        self.assertEqual(status, GUARD_REJECT)
        self.assertTrue(any(SPATIAL_COLLAPSE_ELIGIBLE in r for r in reasons))
        self.assertLess(
            metrics["max_inter_candidate_lateral_separation_m"],
            cfg.ambiguity_min_spatial_sep_m,
        )
        self.assertGreater(
            metrics["max_native_excursion_candidate0_m"],
            cfg.ambiguity_min_spatial_sep_m,
        )

    def test_fixed_bias_lineage_rejected(self) -> None:
        native = _native_line(20)
        nom, alt = synthetic_diverse_residuals(20, lineage="contract_probe")
        alt["raw_d"] = [min(2.5, 0.35 * i) for i in range(20)]
        alt["head_lineage"] = "fixed_bias"
        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=(0.0, 0.0),
            ego_v=5.0,
            base_speed_mps=6.0,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={},
            backbone_forward_id="fwd-1",
        )
        status, reasons, _ = validate_k2_spatial_bundle(
            bundle, require_diversity_if_eligible=False
        )
        self.assertEqual(status, GUARD_REJECT)
        self.assertTrue(any(HEAD_LINEAGE_INVALID in r for r in reasons))

    def test_collapse_eligible_rejected(self) -> None:
        native = _native_line(20)
        nom, alt = synthetic_diverse_residuals(20, lineage="contract_probe")
        alt["raw_d"] = [0.0] * 20
        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=(0.0, 0.0),
            ego_v=5.0,
            base_speed_mps=6.0,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={},
            backbone_forward_id="fwd-1",
        )
        bundle = replace(bundle, set_diagnostics={"eligible_for_diversity": True})
        status, reasons, _ = validate_k2_spatial_bundle(
            bundle, require_diversity_if_eligible=True
        )
        self.assertEqual(status, GUARD_REJECT)
        self.assertTrue(any(SPATIAL_COLLAPSE_ELIGIBLE in r for r in reasons))

    def test_short_defensive_spatial_horizon_rejected(self) -> None:
        """A lateral proposal that ends early is a stop-path, not a pass route."""
        native = _native_line(20)
        nom, alt = synthetic_diverse_residuals(20, lineage="contract_probe")
        alt["raw_d"] = [min(2.5, 0.35 * i) for i in range(20)]
        alt["raw_delta_s"] = [-0.2] * 20
        bundle = build_spatial_k2_bundle_from_residuals(
            native_path_xy=native,
            ego_xy=(0.0, 0.0),
            ego_v=5.0,
            base_speed_mps=6.0,
            residual_nominal=nom,
            residual_defensive=alt,
            observation_identity={},
            backbone_forward_id="fwd-short-horizon",
            defensive_available=True,
        )
        status, reasons, metrics = validate_k2_spatial_bundle(
            bundle, require_diversity_if_eligible=True
        )
        self.assertEqual(status, GUARD_REJECT)
        self.assertTrue(
            any("SPATIAL_HORIZON_COLLAPSE_ELIGIBLE" in r for r in reasons)
        )
        self.assertLess(
            metrics["eligible_spatial_horizon_ratio"],
            load_k2_spatial_config().min_eligible_spatial_horizon_ratio,
        )

    def test_path_hash_spoof_rejected(self) -> None:
        bundle = _diverse_bundle()
        c0, c1 = bundle.candidates
        spoofed = replace(c1, proposal_path_hash="deadbeefdeadbeef")
        bundle2 = replace(bundle, candidates=(c0, spoofed))
        status, reasons, _ = validate_k2_spatial_bundle(
            bundle2, require_diversity_if_eligible=False
        )
        self.assertEqual(status, GUARD_REJECT)
        self.assertTrue(any(PROPOSAL_PATH_HASH_MISMATCH in r for r in reasons))

    def test_t10_spec_path_mismatch_rejected(self) -> None:
        bundle = _diverse_bundle()
        c0, c1 = bundle.candidates
        specs = dict(bundle.execution_specs)
        bad_path = tuple((float(i), 5.0) for i in range(20))
        from driving_vla.model.k2_spatial_types import K2ExecutionSpecV2

        specs[c1.candidate_id] = K2ExecutionSpecV2(
            candidate_id=c1.candidate_id,
            spatial_path_xy=bad_path,
            speed_samples_mps=specs[c1.candidate_id].speed_samples_mps,
            spatial_path_hash=stable_hash_xy(bad_path),
            timed_trajectory_hash=c1.timed_trajectory_hash,
            native_anchor_hash=c1.native_anchor_hash,
        )
        bundle2 = replace(bundle, execution_specs=specs)
        status, reasons, _ = validate_k2_spatial_bundle(
            bundle2, require_diversity_if_eligible=False
        )
        self.assertEqual(status, GUARD_REJECT)
        self.assertTrue(any("path_spec" in r or "TIMED" in r for r in reasons))

    def test_native_corridor_config_loaded(self) -> None:
        cfg = load_k2_spatial_config()
        self.assertAlmostEqual(cfg.native_corridor_cross_track_max_m, 1.25)

    def test_force_unavailable_fail_closed(self) -> None:
        bundle = _diverse_bundle(defensive_available=False, eligible=False)
        # Guard OK when not requiring diversity
        guarded = attach_spatial_guard(bundle, require_diversity_if_eligible=False)
        self.assertEqual(guarded.guard_status, GUARD_OK, msg=guarded.guard_reasons)
        # force 0 ok
        s0 = select_k2_spatial(guarded, mode="force", force_index=0)
        self.assertEqual(s0.candidate_index, 0)
        # force 1 must fail-closed
        with self.assertRaises(K2SelectionError) as ctx:
            select_k2_spatial(guarded, mode="force", force_index=1)
        self.assertIn("unavailable_candidate_force_fail_closed", str(ctx.exception))

    def test_top1_unavailable_falls_back_to_zero(self) -> None:
        bundle = _diverse_bundle(defensive_available=False, eligible=False)
        # Force top1_index=1 while cand1 unavailable
        c0, c1 = bundle.candidates
        bundle = replace(bundle, top1_index=1, candidates=(c0, c1))
        guarded = attach_spatial_guard(bundle, require_diversity_if_eligible=False)
        sel = select_k2_spatial(guarded, mode="top1")
        self.assertEqual(sel.candidate_index, 0)
        self.assertEqual(sel.candidate_id, c0.candidate_id)

    def test_selection_rehashes_execution_spec(self) -> None:
        bundle = _diverse_bundle()
        guarded = attach_spatial_guard(bundle, require_diversity_if_eligible=True)
        c0 = guarded.candidates[0]
        reasons = verify_execution_spec_content_hash(guarded, c0.candidate_id)
        self.assertEqual(reasons, [])
        # tamper hash after guard
        specs = dict(guarded.execution_specs)
        bad = replace(specs[c0.candidate_id], spatial_path_hash="0000000000000000")
        specs[c0.candidate_id] = bad
        tampered = replace(guarded, execution_specs=specs, guard_status=GUARD_OK)
        with self.assertRaises(K2SelectionError):
            select_k2_spatial(tampered, mode="force", force_index=0, require_guard_ok=False)

    def test_contract_probe_lineage_default(self) -> None:
        nom, alt = synthetic_diverse_residuals(10)
        self.assertEqual(nom["head_lineage"], "contract_probe")
        self.assertEqual(alt["head_lineage"], "contract_probe")

    def test_v1_config_still_loads(self) -> None:
        cfg = load_k2_config()
        self.assertEqual(cfg.branch_type, "longitudinal_temporal")

    def test_force_0_1_different_path_hash(self) -> None:
        guarded = attach_spatial_guard(_diverse_bundle(), require_diversity_if_eligible=True)
        self.assertEqual(guarded.guard_status, GUARD_OK, msg=guarded.guard_reasons)
        s0 = select_k2_spatial(guarded, mode="force", force_index=0)
        s1 = select_k2_spatial(guarded, mode="force", force_index=1)
        self.assertNotEqual(
            s0.execution_spec.spatial_path_hash, s1.execution_spec.spatial_path_hash
        )


class OfflineEvalGateLogicTest(unittest.TestCase):
    """Unit-test the eligible-denominator gate math without loading a head."""

    def test_pass_spatial_requires_eligible_rate(self) -> None:
        # 1/12 sep must NOT pass under v2 gate (0.70)
        n_elig = 8
        n_elig_sep = 1
        rate = n_elig_sep / n_elig
        pass_spatial = rate >= 0.70
        self.assertFalse(pass_spatial)
        # old buggy rule would have been True
        old_buggy = (1 / 12) >= 0.5 or 1 > 0
        self.assertTrue(old_buggy)

    def test_availability_specificity(self) -> None:
        # all predicted available when 4 negatives → specificity 0
        y_true = [True] * 8 + [False] * 4
        y_pred = [True] * 12
        tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
        fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
        spec = tn / max(tn + fp, 1)
        self.assertEqual(spec, 0.0)


class DatasetSplitContractTest(unittest.TestCase):
    def test_stable_seed_not_python_hash(self) -> None:
        from scripts.r2x_teacher_generate import _stable_seed_from_str

        a = _stable_seed_from_str("pair_abc", base=7)
        b = _stable_seed_from_str("pair_abc", base=7)
        self.assertEqual(a, b)
        self.assertNotEqual(a, _stable_seed_from_str("pair_xyz", base=7))

    def test_path_group_key_stable(self) -> None:
        from scripts.r2x_teacher_generate import _path_group_key

        n1 = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        n2 = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        n3 = [[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]]
        self.assertEqual(_path_group_key(n1), _path_group_key(n2))
        self.assertNotEqual(_path_group_key(n1), _path_group_key(n3))


if __name__ == "__main__":
    unittest.main()
