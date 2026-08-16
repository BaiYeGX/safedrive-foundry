"""Frozen H4 locked-evaluation configuration.

All values in this module are part of the H4 freeze.  They must be reviewed and
committed before the first locked test-label read.  Changing these constants
after test labels have been read invalidates the H4 evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from data_pipeline.h3.contracts import stable_sha256

H4_SCHEMA_VERSION = "safedrive.h4.locked_eval.v1"
H4_EVIDENCE_VERSION = "safedrive.h4.evidence.v1"

# H3 artifact identities that H4 locks onto.
H3_RUN_ID = "h3-v2-20260815d-final"
H3_EVIDENCE_SHA256 = "f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e"
H3_SPLIT_MANIFEST_SHA256 = "17dedd305aaf2933266a15345926f035aa7ebcd3210b6c636cc92d99e676b08c"
H3_CONFIG_SHA256 = "a1354b8e0ac9ff1d0cd5dba121b0097084509bcd0c8af68792a933dfe562aedb"

H2_DATASET_ID = "h2-gatepass-20260813-routefix"
H2_PHYSICAL_MANIFEST_SHA256 = "6e74a789647182d9333cd99a69305bc2700a95216ebc7f34d2af21024a6d48ed"
H2_STORE_MANIFEST_SHA256 = "22d11961c74509843a1df6ea453794fad2519fcc42077540c33ce46e9f3c3524"

CHALLENGE_DATASET_ID = "h3-challenge-v2-20260815d-dev"
CHALLENGE_PHYSICAL_MANIFEST_SHA256 = "7b27cc140dda3bc1bcf1d1587f95616dcd3c120c9596b0642caf63bd98a029d9"
CHALLENGE_STORE_MANIFEST_SHA256 = "a87871c9c858d440f7b4d6553663ca63e43469f510e950e19d84ee06d3aa35ef"

# Final H3 deployment checkpoints (5-seed ensemble).
FINAL_CHECKPOINTS: dict[str, dict[str, str]] = {
    "11": {
        "path": "generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-11.pt",
        "sha256": "7f9adf45a50fd521da6f1766d8ed89a7be77f20dca7b088ef2c1505e8b42fb11",
    },
    "23": {
        "path": "generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-23.pt",
        "sha256": "50454c1178b9b59f7572d45f21a6475a0cdba7bf2c5fd00120b92c80b5394a09",
    },
    "37": {
        "path": "generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-37.pt",
        "sha256": "b2be92e1480fa1b2ef8bfc6cb074ca118dc75a602e5f61e20ce0c8f56d8e8d43",
    },
    "53": {
        "path": "generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-53.pt",
        "sha256": "c1b9cdb057eae056e32b69bfcca0f359a96bb2a2021f4a5d95f8d7c35dcf04b0",
    },
    "71": {
        "path": "generated/h3/h3-v2-20260815d-final/checkpoints/final/seed-71.pt",
        "sha256": "e3ca937f7fd3e9ada03f1f79ede2aaffb263045373834bf10f8d4dc20c56d8b5",
    },
}

H4_CONFIG: dict[str, Any] = {
    "schema_version": H4_SCHEMA_VERSION,
    "evidence_version": H4_EVIDENCE_VERSION,
    "h3_run_id": H3_RUN_ID,
    "h3_evidence_sha256": H3_EVIDENCE_SHA256,
    "h3_split_manifest_sha256": H3_SPLIT_MANIFEST_SHA256,
    "h3_config_sha256": H3_CONFIG_SHA256,
    "h2_dataset_id": H2_DATASET_ID,
    "h2_physical_manifest_sha256": H2_PHYSICAL_MANIFEST_SHA256,
    "h2_store_manifest_sha256": H2_STORE_MANIFEST_SHA256,
    "challenge_dataset_id": CHALLENGE_DATASET_ID,
    "challenge_physical_manifest_sha256": CHALLENGE_PHYSICAL_MANIFEST_SHA256,
    "challenge_store_manifest_sha256": CHALLENGE_STORE_MANIFEST_SHA256,
    "final_checkpoints": FINAL_CHECKPOINTS,
    "temperature": 0.05000531921999756,
    "runtime": {
        "defer_margin": 0.05,
        "max_uncertainty": 0.35,
        "deadline_ms": 50.0,
        "max_incremental_gpu_gib": 1.5,
    },
    "bootstrap": {
        "seed": 7,
        "rounds": 10000,
    },
    "min_decisive_for_claim": 20,
    "baselines": (
        "no_action",
        "candidate_only",
        "planned_length",
        "final_speed",
        "planned_jerk",
        "cv_ctrv",
        "h1_soft_selector",
        "hand_reward",
        "comfort",
    ),
    "ece_bins": 10,
}

H4_CONFIG_SHA256 = stable_sha256(H4_CONFIG)

__all__ = [
    "FINAL_CHECKPOINTS",
    "H4_CONFIG",
    "H4_CONFIG_SHA256",
    "H4_EVIDENCE_VERSION",
    "H4_SCHEMA_VERSION",
    "H3_CONFIG_SHA256",
    "H3_EVIDENCE_SHA256",
    "H3_RUN_ID",
    "H3_SPLIT_MANIFEST_SHA256",
    "H2_DATASET_ID",
    "H2_PHYSICAL_MANIFEST_SHA256",
    "H2_STORE_MANIFEST_SHA256",
    "CHALLENGE_DATASET_ID",
    "CHALLENGE_PHYSICAL_MANIFEST_SHA256",
    "CHALLENGE_STORE_MANIFEST_SHA256",
]
