"""Frozen H5 closed-loop on/off experiment configuration.

All values in this module are part of the H5 freeze.  Changing any value after
the first real CARLA run invalidates the H5 evidence.  This module does not
read any H4 test labels.
"""

from __future__ import annotations

from typing import Any

from data_pipeline.h3.contracts import stable_sha256
from data_pipeline.h4.contracts import (
    CHALLENGE_PHYSICAL_MANIFEST_SHA256,
    FINAL_CHECKPOINTS,
    H2_PHYSICAL_MANIFEST_SHA256,
    H3_EVIDENCE_SHA256,
    H3_SPLIT_MANIFEST_SHA256,
    H4_CONFIG,
)

H5_SCHEMA_VERSION = "safedrive.h5.closed_loop.v1"
H5_EVIDENCE_VERSION = "safedrive.h5.evidence.v1"

# Risk calibration produced by scripts/h5_calibrate_risk.py on H3 dev folds.
H5_RISK_CALIBRATION_REL = "generated/h5/risk_calibration.json"
H5_RISK_CALIBRATION_SHA256 = "14da362af461200013efad914d278abdbe973b683fd37e014b2d57326b1f581f"
H5_RISK_DEFER_PROBABILITY = 0.37027421649586634
H5_RISK_CALIBRATION_METHOD = "separable_midpoint"

# Probability calibration floor fixed by H4/H5 to avoid saturated probabilities.
H5_PROBABILITY_TEMPERATURE_FLOOR = 0.5

# Temperature calibration produced by scripts/h5_calibrate_temperature.py on H3 dev folds.
H5_TEMPERATURE_CALIBRATION_REL = "generated/h5/temperature_calibration.json"
H5_TEMPERATURE_CALIBRATION_SHA256 = "8cb5c60c387a0bab5c8ed895846ee084bb5422bbcf06770e0ecb16f6758c44ce"

# Frozen H5 scenario inputs.
H2_PHYSICAL_MANIFEST_REL = "generated/h2/paired-outcomes/h2-gatepass-20260813-routefix/scenario_manifest.json"
CHALLENGE_PHYSICAL_MANIFEST_REL = "generated/h3/carla-challenge-v2/h3-challenge-v2-20260815d-dev/scenario_manifest.json"
H3_SPLIT_MANIFEST_REL = "docs/runtime-evidence/h3/h3-v2-20260815d-final/split_manifest.json"
H4_EVIDENCE_REL = "docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json"

# H5 arms.
ARMS = ("off", "on", "defer")

H5_CONFIG: dict[str, Any] = {
    "schema_version": H5_SCHEMA_VERSION,
    "evidence_version": H5_EVIDENCE_VERSION,
    "h4_run_id": "h4-locked-20260816-final",
    "h4_evidence_sha256": "35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4",
    "h3_evidence_sha256": H3_EVIDENCE_SHA256,
    "h3_split_manifest_sha256": H3_SPLIT_MANIFEST_SHA256,
    "h2_physical_manifest_sha256": H2_PHYSICAL_MANIFEST_SHA256,
    "challenge_physical_manifest_sha256": CHALLENGE_PHYSICAL_MANIFEST_SHA256,
    "final_checkpoints": FINAL_CHECKPOINTS,
    "temperature": float(H4_CONFIG["temperature"]),
    "probability_temperature_floor": H5_PROBABILITY_TEMPERATURE_FLOOR,
    "risk_defer_probability": H5_RISK_DEFER_PROBABILITY,
    "risk_calibration_sha256": H5_RISK_CALIBRATION_SHA256,
    "risk_calibration_method": H5_RISK_CALIBRATION_METHOD,
    "temperature_calibration_sha256": H5_TEMPERATURE_CALIBRATION_SHA256,
    "runtime": {
        "defer_margin": float(H4_CONFIG["runtime"]["defer_margin"]),
        "max_uncertainty": float(H4_CONFIG["runtime"]["max_uncertainty"]),
        "scorer_deadline_ms": float(H4_CONFIG["runtime"]["deadline_ms"]),
        "max_incremental_gpu_gib": float(H4_CONFIG["runtime"]["max_incremental_gpu_gib"]),
    },
    "router": {
        "min_hold_ticks": 10,
        "hysteresis_margin": 0.05,
        "emergency_switch_margin": 1.5,
        "single_pass_grace_ticks": 3,
    },
    "matrix": {
        "full_split": "test",
        "full_valid_only": True,
        "pilot_count": 12,
        "decision_ticks": 50,
        "pre_roll_ticks": "from_scenario_script",
        "arm_order_seed": "safedrive.h5.arm_order.v1",
    },
    "acceptance": {
        "bootstrap_seed": 7,
        "bootstrap_rounds": 10000,
        "progress_ci_lower": 0.0,
        "safety_noninferior": True,
        "chattering_noninferior": True,
    },
}

H5_CONFIG_SHA256 = stable_sha256(H5_CONFIG)

__all__ = [
    "ARMS",
    "CHALLENGE_PHYSICAL_MANIFEST_REL",
    "H2_PHYSICAL_MANIFEST_REL",
    "H3_SPLIT_MANIFEST_REL",
    "H4_EVIDENCE_REL",
    "H5_CONFIG",
    "H5_CONFIG_SHA256",
    "H5_EVIDENCE_VERSION",
    "H5_PROBABILITY_TEMPERATURE_FLOOR",
    "H5_RISK_CALIBRATION_REL",
    "H5_RISK_CALIBRATION_SHA256",
    "H5_RISK_DEFER_PROBABILITY",
    "H5_SCHEMA_VERSION",
    "H5_TEMPERATURE_CALIBRATION_REL",
    "H5_TEMPERATURE_CALIBRATION_SHA256",
]
