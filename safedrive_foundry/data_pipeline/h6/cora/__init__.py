"""H6-CORA C2 counterfactual potential-outcome data subsystem."""

from .config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256
from .contracts import (
    BRANCH_OUTCOME_SCHEMA,
    FEATURE_SCHEMA,
    PAIR_INDEX_SCHEMA,
    PROPOSAL_SCHEMA,
    ROOT_ANCHOR_SCHEMA,
    CoraBranchOutcome,
    CoraPairEdge,
    CoraProposal,
    CoraRootRecord,
    OutcomeValue,
)
from .feature import build_cora_feature_view, validate_cora_feature_view
from .interventions import derive_interventions
from .matrix import (
    CORA_DATA_MATRIX,
    CORA_FORMAL_MATRIX,
    CORA_MATRIX_SHA256,
    CORA_SMOKE_ROOT_IDS,
    materialize_cora_matrix,
)
from .outcomes import (
    PUBLIC_DERIVATION_VERSION,
    PUBLIC_LABEL_SCHEMA,
    PUBLIC_OUTCOME_HEADS,
    derive_public_outcome_heads,
    materialize_public_labels,
    validate_public_outcome_heads,
)
from .repair_labels import SCHEMA as REPAIR_LABEL_SCHEMA, TRACE_SCHEMA as SAFETY_TRACE_SCHEMA
from .scenarios import materialize_cora_physical_scenario

__all__ = [
    "BRANCH_OUTCOME_SCHEMA",
    "CORA_C2_CONFIG",
    "CORA_C2_CONFIG_SHA256",
    "CORA_DATA_MATRIX",
    "CORA_FORMAL_MATRIX",
    "CORA_MATRIX_SHA256",
    "CORA_SMOKE_ROOT_IDS",
    "CoraBranchOutcome",
    "CoraPairEdge",
    "CoraProposal",
    "CoraRootRecord",
    "FEATURE_SCHEMA",
    "OutcomeValue",
    "PAIR_INDEX_SCHEMA",
    "PROPOSAL_SCHEMA",
    "PUBLIC_DERIVATION_VERSION",
    "PUBLIC_LABEL_SCHEMA",
    "PUBLIC_OUTCOME_HEADS",
    "REPAIR_LABEL_SCHEMA",
    "SAFETY_TRACE_SCHEMA",
    "ROOT_ANCHOR_SCHEMA",
    "build_cora_feature_view",
    "derive_interventions",
    "derive_public_outcome_heads",
    "materialize_cora_matrix",
    "materialize_cora_physical_scenario",
    "materialize_public_labels",
    "validate_cora_feature_view",
    "validate_public_outcome_heads",
]
