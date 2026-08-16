"""H3v2 candidate-conditioned World scorer data and training utilities.

Offline training consumes the frozen H2 and H3 Challenge stores through the
observable feature view.  Runtime code never imports the offline Oracle.
"""

from .contracts import (
    H3_CONFIG,
    H3_CONFIG_SHA256,
    H3_CONTEXT_DIM,
    H3_SCHEMA_VERSION,
    H3_SPLIT_VERSION,
)
from .dataset import (
    H3DatasetError,
    PairExample,
    build_split_manifest,
    leakage_audit,
    load_examples,
    write_split_manifest,
)

__all__ = [
    "H3_CONFIG",
    "H3_CONFIG_SHA256",
    "H3_CONTEXT_DIM",
    "H3_SCHEMA_VERSION",
    "H3_SPLIT_VERSION",
    "H3DatasetError",
    "PairExample",
    "build_split_manifest",
    "leakage_audit",
    "load_examples",
    "write_split_manifest",
]
