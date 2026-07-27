"""Debug / fingerprint V1 residual helpers (NOT R1 acceptance).

Formal real-K2 acceptance uses ``NeuralV1Policy`` + ``k2_builder`` (path retiming).
This module keeps the legacy fingerprint ``V1Policy`` and residual/lateral-bias
utilities for offline debug only. Do not treat lateral_bias forks as neural K2.
"""

from __future__ import annotations

import math
from typing import Sequence

from driving_vla.adapter.policy_adapter import ObservationBundle, TrajectoryArray
from driving_vla.model.backbone_loader import SimLingoCheckpointHandle, V0Policy
from driving_vla.schema.trajectory_contract import DT_S, T_STEPS


def _history_features(
    history: Sequence[tuple[float, float, float, float]],
    current: tuple[float, float, float, float],
) -> list[float]:
    """Pack up to 4 history steps + current into flat vector (x,y,yaw,v each)."""
    feats: list[float] = []
    hist = list(history)[-4:]
    while len(hist) < 4:
        hist.insert(0, current)
    for x, y, yaw, v in hist:
        feats.extend([float(x), float(y), float(yaw), float(v)])
    feats.extend([float(current[0]), float(current[1]), float(current[2]), float(current[3])])
    return feats


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _apply_residual(base: TrajectoryArray, scale: float, lateral_bias: float) -> TrajectoryArray:
    """Apply lateral offset + speed scale, then recompute yaw / a / kappa."""
    raw: list[tuple[float, float, float]] = []  # x, y, v
    for i, (x, y, yaw, v, _a, _kappa) in enumerate(base.points_xy_yaw_v_a_kappa):
        nx = -math.sin(yaw)
        ny = math.cos(yaw)
        # grow lateral offset along horizon so mid/end points diverge
        lat = lateral_bias * (i + 1) / T_STEPS
        x2 = x + lat * nx
        y2 = y + lat * ny
        v2 = max(0.3, float(v) * scale)
        raw.append((x2, y2, v2))

    pts: list[tuple[float, float, float, float, float, float]] = []
    prev_v = raw[0][2]
    for i, (x, y, v) in enumerate(raw):
        if i + 1 < len(raw):
            dx = raw[i + 1][0] - x
            dy = raw[i + 1][1] - y
        elif i > 0:
            dx = x - raw[i - 1][0]
            dy = y - raw[i - 1][1]
        else:
            dx, dy = 1.0, 0.0
        yaw = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
        a = (v - prev_v) / DT_S
        if i == 0:
            kappa = 0.0
        else:
            dyaw = _wrap_angle(yaw - pts[-1][2])
            ds = max(math.hypot(x - pts[-1][0], y - pts[-1][1]), 1e-3)
            kappa = dyaw / ds
        pts.append((x, y, yaw, v, a, kappa))
        prev_v = v

    return TrajectoryArray(
        points_xy_yaw_v_a_kappa=tuple(pts),
        probability=0.0,
        uncertainty=0.0,
        candidate_id="tmp",
    )


class V1Policy:
    """K=2 nominal/conservative anchored on V0 tau_0 + history-conditioned scales."""

    model_id = "sdf-vla-v1@0.1.0"
    k = 2

    def __init__(self, handle: SimLingoCheckpointHandle) -> None:
        self.v0 = V0Policy(handle)

    def predict_arrays(self, obs: ObservationBundle) -> list[TrajectoryArray]:
        base_list = self.v0.predict_arrays(obs)
        base = base_list[0]
        cur = (obs.ego_x, obs.ego_y, obs.ego_yaw, obs.ego_v)
        feats = _history_features(obs.ego_history, cur)
        hist_v = feats[3::4]
        mean_v = sum(hist_v) / len(hist_v)
        nom_scale = max(0.7, min(1.15, mean_v / max(obs.ego_v, 0.5) if obs.ego_v > 0.1 else 1.0))
        cons_scale = 0.65 * nom_scale
        # Spatial fork: conservative offset laterally so position ADE can rank.
        nom = _apply_residual(base, nom_scale, lateral_bias=0.0)
        cons = _apply_residual(base, cons_scale, lateral_bias=0.6)
        p_nom, p_cons = 0.62, 0.38
        return [
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=nom.points_xy_yaw_v_a_kappa,
                probability=p_nom,
                uncertainty=0.12,
                candidate_id="v1_nominal",
                intended_action="nominal",
            ),
            TrajectoryArray(
                points_xy_yaw_v_a_kappa=cons.points_xy_yaw_v_a_kappa,
                probability=p_cons,
                uncertainty=0.22,
                candidate_id="v1_conservative",
                intended_action="conservative",
            ),
        ]


def oracle_best_of_k(
    candidates: Sequence[TrajectoryArray],
    *,
    expert: TrajectoryArray | None = None,
) -> tuple[int, float]:
    """Return (best_index, score). If expert given, min ADE; else max probability."""
    if not candidates:
        return -1, float("nan")
    if expert is None:
        best_i = max(range(len(candidates)), key=lambda i: candidates[i].probability)
        return best_i, candidates[best_i].probability
    best_i = 0
    best_ade = float("inf")
    for i, c in enumerate(candidates):
        ade = 0.0
        n = min(len(c.points_xy_yaw_v_a_kappa), len(expert.points_xy_yaw_v_a_kappa))
        for j in range(n):
            dx = c.points_xy_yaw_v_a_kappa[j][0] - expert.points_xy_yaw_v_a_kappa[j][0]
            dy = c.points_xy_yaw_v_a_kappa[j][1] - expert.points_xy_yaw_v_a_kappa[j][1]
            ade += math.hypot(dx, dy)
        ade /= max(n, 1)
        if ade < best_ade:
            best_ade = ade
            best_i = i
    return best_i, best_ade


def detect_collapse(candidates: Sequence[TrajectoryArray], *, eps: float = 0.05) -> bool:
    """True if all candidates nearly identical in position (collapsed)."""
    if len(candidates) < 2:
        return False
    a = candidates[0].points_xy_yaw_v_a_kappa
    for c in candidates[1:]:
        b = c.points_xy_yaw_v_a_kappa
        max_d = 0.0
        for p, q in zip(a, b):
            max_d = max(max_d, math.hypot(p[0] - q[0], p[1] - q[1]))
        if max_d > eps:
            return False
    return True


def max_position_separation(candidates: Sequence[TrajectoryArray]) -> float:
    if len(candidates) < 2:
        return 0.0
    a = candidates[0].points_xy_yaw_v_a_kappa
    b = candidates[1].points_xy_yaw_v_a_kappa
    return max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(a, b))
