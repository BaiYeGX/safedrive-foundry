"""H2 paired-outcome contracts, storage, offline labels and audits."""

from .contracts import (
    ActorInitialState,
    BranchOutcome,
    CandidateSnapshot,
    H2ContractError,
    OracleLabel,
    OracleVerdict,
    PairRecord,
    PairTerminalStatus,
    ResetComparison,
    ResetSignature,
    ScenarioKey,
    h3_feature_view,
    stable_sha256,
)
from .config import H2_CONFIG, H2_CONFIG_SHA256, config_identity
from .matrix import FIXED_MATRIX, PILOT_MATRIX, MatrixEntry, materialize_matrix
from .store import PairedOutcomeStore

__all__ = [
    "ActorInitialState",
    "BranchOutcome",
    "CandidateSnapshot",
    "FIXED_MATRIX",
    "H2ContractError",
    "MatrixEntry",
    "OracleLabel",
    "OracleVerdict",
    "PILOT_MATRIX",
    "PairRecord",
    "H2_CONFIG", "H2_CONFIG_SHA256", "config_identity",
    "PairTerminalStatus",
    "PairedOutcomeStore",
    "ResetComparison",
    "ResetSignature",
    "ScenarioKey",
    "h3_feature_view",
    "materialize_matrix",
    "stable_sha256",
]
