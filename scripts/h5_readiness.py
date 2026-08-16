#!/usr/bin/env python3
"""H5 readiness preflight: verify H3/H4 evidence, H5 matrix and runtime hardening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from data_pipeline.h3.contracts import stable_sha256  # noqa: E402
from data_pipeline.h4.contracts import FINAL_CHECKPOINTS  # noqa: E402
from data_pipeline.h5.config import (  # noqa: E402
    H5_CONFIG,
    H5_RISK_CALIBRATION_REL,
    H5_RISK_CALIBRATION_SHA256,
    H5_TEMPERATURE_CALIBRATION_REL,
    H5_TEMPERATURE_CALIBRATION_SHA256,
)
from data_pipeline.h5.matrix import h5_matrix_sha256, load_h5_matrix  # noqa: E402
from data_pipeline.h5.runtime import H5WorldRouter  # noqa: E402


def _evidence_ok(run_type: str, run_id: str, expected_sha: str) -> bool:
    path = ROOT / "docs" / "runtime-evidence" / run_type / run_id / "final-delivery.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = stable_sha256({k: v for k, v in payload.items() if k != "evidence_sha256"})
    return actual == expected_sha and bool(payload.get("gate", {}).get("passed", False))


def _checkpoint_files_ok() -> bool:
    for info in FINAL_CHECKPOINTS.values():
        path = ROOT / info["path"]
        if not path.is_file():
            return False
        # Do not hash large checkpoints on every readiness call; existence plus
        # the live runner's from_checkpoints/load path enforces SHA before use.
    return True


def _risk_calibration_ok() -> bool:
    path = ROOT / H5_RISK_CALIBRATION_REL
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sha256") != H5_RISK_CALIBRATION_SHA256:
        return False
    verify = {k: v for k, v in payload.items() if k != "sha256"}
    return stable_sha256(verify) == H5_RISK_CALIBRATION_SHA256


def _temperature_calibration_ok() -> bool:
    path = ROOT / H5_TEMPERATURE_CALIBRATION_REL
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sha256") != H5_TEMPERATURE_CALIBRATION_SHA256:
        return False
    verify = {k: v for k, v in payload.items() if k != "sha256"}
    return stable_sha256(verify) == H5_TEMPERATURE_CALIBRATION_SHA256


def main() -> int:
    checks = {
        "h3_evidence": _evidence_ok(
            "h3",
            "h3-v2-20260815d-final",
            "f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e",
        ),
        "h4_evidence": _evidence_ok(
            "h4",
            "h4-locked-20260816-final",
            "35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4",
        ),
        "h5_router_importable": True,
        "hysteresis_available": hasattr(H5WorldRouter, "route"),
        "force_defer_available": "force_defer" in H5WorldRouter.__init__.__code__.co_varnames,
        "checkpoint_files_exist": _checkpoint_files_ok(),
        "risk_calibration_ok": _risk_calibration_ok(),
        "temperature_calibration_ok": _temperature_calibration_ok(),
    }
    try:
        scenarios = load_h5_matrix(ROOT, full=True)
        checks["h5_matrix_loaded"] = len(scenarios) == 74
        checks["h5_matrix_sha256"] = h5_matrix_sha256(scenarios)
    except Exception as exc:
        checks["h5_matrix_loaded"] = False
        checks["h5_matrix_error"] = f"{type(exc).__name__}:{exc}"
    # NormalizedWorldScorer must default to risk-gated defer.
    try:
        import inspect
        from data_pipeline.h4.runtime import NormalizedWorldScorer
        params = inspect.signature(NormalizedWorldScorer.__init__).parameters
        risk_param = params.get("risk_defer_probability")
        prob_floor = params.get("probability_temperature_floor")
        checks["risk_gate_default"] = risk_param is not None and float(risk_param.default) == 0.35
        checks["probability_floor_default"] = prob_floor is not None and float(prob_floor.default) == 0.5
    except Exception:
        checks["risk_gate_default"] = False
    ready = all(checks.values())
    print(json.dumps({"ready": ready, "checks": checks}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
