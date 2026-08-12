from __future__ import annotations

import math

import numpy as np

from driving_vla.world.contracts import (
    ACTOR_FEATURES,
    CANDIDATE_FEATURES,
    EGO_FEATURES,
    FUTURE_FEATURES,
    HISTORY,
    K,
    MAX_ACTORS,
    MAX_ROAD_POINTS,
    MAX_ROAD_POLYLINES,
    OUTCOME_FEATURES,
    ROAD_FEATURES,
    T,
    ActionBranchSample,
    SampleIdentity,
)


def make_sample(
    index: int = 0,
    *,
    winner: int | None = 0,
    tie: bool = False,
    candidate1_available: bool = True,
    family: str = "crossing",
) -> ActionBranchSample:
    ego = np.zeros((HISTORY, EGO_FEATURES), dtype=np.float32)
    ego[-1] = (0, 0, 0, 1, 5, 0, 0, 0, 0, 1, 0)
    ego_mask = np.zeros(HISTORY, dtype=bool)
    ego_mask[-1] = True
    actors = np.zeros((MAX_ACTORS, HISTORY, ACTOR_FEATURES), dtype=np.float32)
    actors[0, -1] = (8, 1, 0, 1, -1, 0, 0, 0, 4.5, 1.8, 0, 0, 0, 1)
    actor_mask = np.zeros((MAX_ACTORS, HISTORY), dtype=bool)
    actor_mask[0, -1] = True
    road = np.zeros(
        (MAX_ROAD_POLYLINES, MAX_ROAD_POINTS, ROAD_FEATURES), dtype=np.float32
    )
    road_mask = np.zeros((MAX_ROAD_POLYLINES, MAX_ROAD_POINTS), dtype=bool)
    for i in range(MAX_ROAD_POINTS):
        road[0, i] = (float(i), 0.0, 0.0, 1.0, 0.0, 8.0)
        road_mask[0, i] = True
    candidates = np.zeros((K, T, CANDIDATE_FEATURES), dtype=np.float32)
    for k in range(K):
        for t in range(T):
            candidates[k, t] = (
                (t + 1) * (1.0 - 0.2 * k),
                0.6 * k * math.sin((t + 1) / T * math.pi),
                0.0,
                1.0,
                4.0 - k,
                -0.2 * k,
                0.02 * k,
                (t + 1) * 0.25,
            )
    candidate_mask = np.asarray([True, candidate1_available], dtype=bool)
    future = np.zeros((K, MAX_ACTORS, T, FUTURE_FEATURES), dtype=np.float32)
    future_mask = np.zeros((K, MAX_ACTORS, T), dtype=bool)
    for k in range(K):
        if not candidate_mask[k]:
            continue
        for t in range(T):
            future[k, 0, t] = (8 - 0.25 * (t + 1), 1 + 0.05 * k, 0, 1, -1, 0)
            future_mask[k, 0, t] = True
    outcomes = np.zeros((K, OUTCOME_FEATURES), dtype=np.float32)
    outcomes[0] = (0, 0, 2.5, 3.0, 10.0, 0.5, 1.0, 2.5)
    outcomes[1] = (0, 0, 2.5, 5.0, 8.0, 0.4, 1.0, 2.5)
    outcome_mask = candidate_mask.copy()
    rank_mask = bool(candidate_mask.all()) and (tie or winner is not None)
    rank_target = 0.0 if tie else (1.0 if winner == 0 else -1.0)
    h = f"{index + 1:064x}"
    identity = SampleIdentity(
        sample_id=f"sample-{index}",
        pair_id=f"pair-{index}",
        scenario_id=f"scenario-{index // 2}",
        seed_id=f"seed-{index % 2}",
        group_key=f"Town03|{family}|lineage-{index // 2}",
        family=family,
        map_name="Town03",
        initial_state_hash=h,
        observation_hash=h,
        anchor_artifact_hash=h,
        model_hash=h,
        guard_hash=h,
        executor_hash=h,
        source_manifest_hash=h,
    )
    sample = ActionBranchSample(
        identity=identity,
        ego_history=ego,
        ego_history_mask=ego_mask,
        actor_history=actors,
        actor_history_mask=actor_mask,
        road=road,
        road_mask=road_mask,
        candidates=candidates,
        candidate_mask=candidate_mask,
        actor_future=future,
        actor_future_mask=future_mask,
        outcomes=outcomes,
        outcome_mask=outcome_mask,
        rank_target=rank_target,
        rank_mask=rank_mask,
        rank_weight=1.0 if rank_mask else 0.0,
        tie_target=tie,
        comparable=bool(candidate_mask.all()),
        unavailable_reasons=(None, None if candidate1_available else "NO_ALTERNATIVE"),
        audit={"pair_label": "TIE" if tie else "TOP1_BEST"},
    )
    sample.validate()
    return sample
