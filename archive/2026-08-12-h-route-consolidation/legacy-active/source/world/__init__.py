"""Candidate-conditioned World-V0.

This package is runtime-safe: it must not import paired-evaluation oracle code.
Oracle-only label collection lives under :mod:`driving_vla.evaluation`.
"""

from .contracts import (
    ActionBranchSample,
    WorldBatch,
    WorldContractError,
    WorldPrediction,
)
from .dataset import ActionBranchDataset, ActionBranchDatasetV1
from .model_v0 import WorldV0, WorldV0Config
from .navigation_batch import (
    RouteBoundWorldBatch,
    WorldNavigationCondition,
)

__all__ = [
    "ActionBranchSample",
    "ActionBranchDataset",
    "ActionBranchDatasetV1",
    "WorldBatch",
    "WorldContractError",
    "WorldPrediction",
    "RouteBoundWorldBatch",
    "WorldNavigationCondition",
    "WorldV0",
    "WorldV0Config",
]
