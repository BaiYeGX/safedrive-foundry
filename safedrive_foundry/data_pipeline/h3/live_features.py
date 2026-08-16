"""Online feature extraction for H3 WorldScorer.

The produced vectors must be byte-identical to the offline H3v2 feature view.
Only observable anchor/history/route/candidate trajectory fields are read.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from driving_vla.hybrid.contracts import ObservableAnchor
from safety_kernel.contracts.serialize import candidate_to_dict

from .dataset import _candidate_tensor, _context_vector


def build_live_features(
    anchor: ObservableAnchor,
    observable_history: Sequence[Mapping[str, Any]],
    candidates: Sequence[Any],
) -> dict[str, tuple[Sequence[float], Sequence[Sequence[float]]]]:
    """Return {candidate_id: (context_vector, candidate_tensor)}.

    ``candidates`` are H1 ``HybridCandidate`` wrappers or Safety
    ``PolicyCandidate`` objects; only their candidate identity and trajectory
    are used.
    """
    snapshot = asdict(anchor.safety_snapshot) if hasattr(anchor.safety_snapshot, "__dataclass_fields__") else dict(anchor.safety_snapshot)
    record: dict[str, Any] = {
        "anchor": {
            "observable_snapshot": snapshot,
        },
        "observable_history": [dict(row) for row in observable_history],
        "route": list(anchor.bundle.route_xy),
    }
    features: dict[str, tuple[Sequence[float], Sequence[Sequence[float]]]] = {}
    context = _context_vector(record)
    for item in candidates:
        candidate = getattr(item, "candidate", item)
        payload = candidate_to_dict(candidate)
        payload.setdefault("trajectory", [point_to_dict_public(point) for point in candidate.points])
        features[str(candidate.candidate_id)] = (
            tuple(context),
            tuple(tuple(float(value) for value in row) for row in _candidate_tensor(record, payload)),
        )
    return features


def point_to_dict_public(point: Any) -> dict[str, float]:
    return {
        "t": float(point.t),
        "x": float(point.x),
        "y": float(point.y),
        "yaw": float(point.yaw),
        "v": float(point.v),
        "a": float(point.a),
        "kappa": float(point.kappa),
    }


__all__ = ["build_live_features"]
