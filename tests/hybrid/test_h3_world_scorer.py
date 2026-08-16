"""Unit tests for H3 World Scorer data pipeline, models, baselines and runtime."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from safedrive_foundry.data_pipeline.h3.baselines import (
    BASELINE_NAMES,
    _candidate_only_score,
    _cv_ctrv_score,
    _final_speed,
    _hand_reward,
    _planned_jerk,
    _planned_length,
    baseline_winner,
    evaluate_baseline,
)
from safedrive_foundry.data_pipeline.h3.contracts import (
    H3_CONFIG,
    H3_CONFIG_SHA256,
    H3_SCHEMA_VERSION,
    SplitRow,
    WorldPrediction,
    WorldScoreResult,
    stable_sha256,
)
from safedrive_foundry.data_pipeline.h3.dataset import (
    CandidateExample,
    FORBIDDEN_FEATURE_TOKENS,
    H3DatasetError,
    PairExample,
    build_split_manifest,
    lineage_key,
    load_examples,
)
from safedrive_foundry.data_pipeline.h3.evaluate import swap_consistency
from safedrive_foundry.data_pipeline.h3.model import (
    CANDIDATE_DIM,
    CANDIDATE_STEPS,
    CONTEXT_DIM,
    WorldScorerModel,
    predict_model,
)
from safedrive_foundry.data_pipeline.h3.runtime import H3WorldRouter, WorldScorer
from driving_vla.adapter.policy_adapter import ObservationBundle
from driving_vla.hybrid.contracts import (
    GuardResult,
    GuardVerdict,
    HybridCandidate,
    HybridCandidateSet,
    HybridSource,
    ObservableAnchor,
    PolicyCandidate,
    WorldDisposition,
)
from driving_vla.hybrid.generators import (
    ClassicExpertGenerator,
    NominalVLAGenerator,
    generate_hybrid_set,
    route_revision_sha256,
)
from driving_vla.model.nominal_policy import NominalVLAPolicy
from safety_kernel.contracts.types import ObservableSnapshot


def _make_dummy_candidate(key: str, length: float = 10.0, speed: float = 5.0, acc: float = 0.5) -> CandidateExample:
    context = tuple([0.0] * CONTEXT_DIM)
    points = []
    for i in range(CANDIDATE_STEPS):
        t = i * 0.25
        dx = (length / CANDIDATE_STEPS) * i
        points.append((dx / 50.0, 0.0, 0.0, 1.0, speed / 10.0, acc / 10.0, 0.0, t / 2.5))
    return CandidateExample(
        candidate_key=key,
        context=context,
        candidate=tuple(tuple(row) for row in points),
        progress_m=length,
        jerk_rms_mps3=0.1,
        risk=False,
        h1_soft_score=length - 0.35 * 0.1,
    )


def _make_dummy_pair(pair_id: str = "pair-001", winner_idx: int | None = 0) -> PairExample:
    cand0 = _make_dummy_candidate("c0", length=15.0, speed=6.0, acc=0.2)
    cand1 = _make_dummy_candidate("c1", length=10.0, speed=4.0, acc=0.8)
    return PairExample(
        pair_id=pair_id,
        map_name="Town01",
        family="free_flow",
        seed=0,
        weather="ClearNoon",
        split="dev_fold_1",
        candidates=(cand0, cand1),
        winner_index=winner_idx,
        tie=winner_idx is None,
    )


def _make_anchor() -> ObservableAnchor:
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


class TestH3WorldScorer(unittest.TestCase):
    """Test suite for H3 World Scorer pipeline and contracts."""

    def test_h3_contracts_and_hashes(self) -> None:
        self.assertEqual(H3_CONFIG["schema_version"], H3_SCHEMA_VERSION)
        self.assertIsInstance(H3_CONFIG_SHA256, str)
        self.assertEqual(len(H3_CONFIG_SHA256), 64)

        row = SplitRow("p1", "Town01", "free_flow", 0, "ClearNoon", "Town01|free_flow|0", 0, "dev_fold_1", True)
        d = row.to_dict()
        self.assertEqual(d["pair_id"], "p1")
        self.assertEqual(d["map"], "Town01")

        pred = WorldPrediction("c0", 1.5, 12.0, -0.5, 0.1, -1.0, 0.25)
        self.assertEqual(pred.candidate_key, "c0")
        self.assertAlmostEqual(pred.utility, 1.5)
        self.assertAlmostEqual(pred.risk_logit, 0.25)

    def test_baselines_pure_computation(self) -> None:
        cand0 = _make_dummy_candidate("c0", length=20.0, speed=8.0, acc=0.5)
        cand1 = _make_dummy_candidate("c1", length=10.0, speed=4.0, acc=0.2)
        pair = PairExample("p1", "Town01", "free_flow", 0, "ClearNoon", "dev_fold_1", (cand0, cand1), 0, False)

        self.assertGreater(_planned_length(cand0), _planned_length(cand1))
        self.assertGreater(_final_speed(cand0), _final_speed(cand1))
        self.assertGreaterEqual(_planned_jerk(cand0), 0.0)

        # Baseline winners
        self.assertEqual(baseline_winner(pair, "planned_length"), 0)
        self.assertEqual(baseline_winner(pair, "final_speed"), 0)
        self.assertEqual(baseline_winner(pair, "candidate_only"), 0)
        self.assertEqual(baseline_winner(pair, "cv_ctrv"), 0)

        # Evaluation metrics
        eval_res = evaluate_baseline([pair], "planned_length")
        self.assertEqual(eval_res["correct"], 1)
        self.assertEqual(eval_res["accuracy"], 1.0)
        self.assertEqual(eval_res["mean_progress_regret_m"], 0.0)

    def test_h1_soft_selector_is_not_hand_reward_alias(self) -> None:
        # The H1 baseline must come from the frozen Safety soft score attached
        # to each example, not from the hand_reward formula.
        cand0 = _make_dummy_candidate("c0", length=20.0, speed=8.0, acc=0.5)
        cand1 = _make_dummy_candidate("c1", length=10.0, speed=4.0, acc=0.2)
        pair = PairExample("p1", "Town01", "free_flow", 0, "ClearNoon", "dev_fold_1", (cand0, cand1), 0, False)
        self.assertNotEqual(cand0.h1_soft_score, cand0.candidate[0][0] * 50.0)
        self.assertIn(baseline_winner(pair, "h1_soft_selector"), {0, 1})

    def test_model_forward_and_swap_invariance(self) -> None:
        model = WorldScorerModel(d_model=64, layers=1, heads=2, ffn=128)
        model.eval()

        context = torch.zeros((2, CONTEXT_DIM), dtype=torch.float32)
        candidate = torch.zeros((2, CANDIDATE_STEPS, CANDIDATE_DIM), dtype=torch.float32)

        out = model(context, candidate)
        self.assertEqual(out.shape, (2, 6))

        # Test predict_model and swap consistency
        pair = _make_dummy_pair("p1")
        p0, p1 = predict_model(model, pair)
        self.assertIsInstance(p0.utility, float)
        self.assertIsInstance(p1.utility, float)

        swap_res = swap_consistency([model], [pair])
        self.assertTrue(swap_res["passed"])
        self.assertLessEqual(swap_res["max_error"], 1e-6)

    def test_runtime_world_scorer_and_defer(self) -> None:
        model = WorldScorerModel(d_model=64, layers=1, heads=2, ffn=128)
        model.eval()
        scorer = WorldScorer([model])

        cand0 = _make_dummy_candidate("c0", length=15.0)
        cand1 = _make_dummy_candidate("c1", length=10.0)

        res = scorer.score_pair(
            ("c0", cand0.context, cand0.candidate),
            ("c1", cand1.context, cand1.candidate),
        )
        self.assertIn(res.disposition, {"ranked", "defer_low_confidence"})
        self.assertEqual(len(res.predictions), 2)

        # Test H3WorldRouter deferral
        router = H3WorldRouter(scorer)
        anchor = _make_anchor()
        runtime = FakeSimLingoRuntime()
        policy = NominalVLAPolicy(runtime=runtime)
        generated = generate_hybrid_set(
            anchor,
            ClassicExpertGenerator(),
            NominalVLAGenerator(policy, generator_hash="fake-simlingo-hash"),
        )
        # Mark both guard passed
        passed_candidates = tuple(
            replace(
                item,
                guard=GuardResult(
                    candidate_id=item.provenance.candidate_id,
                    verdict=GuardVerdict.PASS,
                    checks=(),
                    reject_reasons=(),
                    latency_ms=0.1,
                ),
            )
            for item in generated.candidates
        )
        candidate_set = replace(generated, candidates=passed_candidates)

        # Missing features should defer gracefully to DEFERRED_LOW_CONFIDENCE
        routed = router.route(candidate_set, features=None)
        self.assertEqual(routed.world, WorldDisposition.DEFERRED_LOW_CONFIDENCE)


if __name__ == "__main__":
    unittest.main()
