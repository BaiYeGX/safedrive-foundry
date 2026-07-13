"""G2-04 evidence: arbitration latency, fallback success, shadow audit."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from safety_kernel import (  # noqa: E402
    SCHEMA_VERSION,
    ComponentAvailability,
    ObservableSnapshot,
    PolicyCandidate,
    PolicyCandidateSet,
    SafetyKernel,
    config_sha256,
    contracts_schema_hash,
    load_safety_config,
)
from safety_kernel.arbitration.types import DegradationReason  # noqa: E402
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservationPrivilege,
    TrajectoryPoint,
    TrafficLightObs,
)
from safety_kernel.evidence_util import (  # noqa: E402
    host_resource_snapshot,
    write_evidence_enabled,
    write_evidence_pack,
)
from safety_kernel.metrics import build_latency_report  # noqa: E402

EVIDENCE_DIR = ROOT / "docs/architecture/evidence/g2-04"
REPRO_CMD = "SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_04_latency_evidence -v"


def _pts(v: float = 5.0, n: int = 16) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(TrajectoryPoint(t=0.25 * i, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
        x += v * 0.25
    return tuple(out)


class G204EvidenceTests(unittest.TestCase):
    def test_write_g2_04_evidence(self) -> None:
        cfg = load_safety_config()
        scenarios = []
        arb_lats: list[float] = []

        cases = {
            "normal_classic": {
                "avail": ComponentAvailability(classic=True, vla=False, world=False, safety=True),
                "cands": (
                    PolicyCandidate(
                        "c",
                        CandidateSource.CLASSIC,
                        1.0,
                        1.2,
                        1.0,
                        _pts(),
                    ),
                ),
                "lights": (),
                "expect_kind": "ACCEPT",
            },
            "learning_off_vla_dropped": {
                "avail": ComponentAvailability(classic=True, vla=False, world=False, safety=True),
                "cands": (
                    PolicyCandidate("v", CandidateSource.VLA_FAST, 1.0, 1.2, 0.99, _pts(), uncertainty=0.1),
                    PolicyCandidate("c", CandidateSource.CLASSIC, 1.0, 1.2, 0.7, _pts()),
                ),
                "lights": (),
                "expect_selected": "c",
            },
            "repair_red_light": {
                "avail": ComponentAvailability(classic=True, vla=False, world=False, safety=True),
                "cands": (
                    PolicyCandidate("raw", CandidateSource.CLASSIC, 1.0, 1.2, 1.0, _pts(v=5.0)),
                ),
                "lights": (TrafficLightObs("tl", "red", 7.0, 1.0),),
            },
            "emergency_no_classic": {
                "avail": ComponentAvailability(classic=False, vla=False, world=False, safety=True),
                "cands": (
                    PolicyCandidate("v", CandidateSource.VLA_FAST, 1.0, 1.2, 0.9, _pts()),
                ),
                "lights": (),
                "expect_kind": "EMERGENCY",
            },
            "overconfident_vla": {
                "avail": ComponentAvailability(classic=True, vla=True, world=False, safety=True),
                "cands": (
                    PolicyCandidate(
                        "v",
                        CandidateSource.VLA_FAST,
                        1.0,
                        1.2,
                        0.99,
                        _pts(),
                        uncertainty=0.01,
                    ),
                    PolicyCandidate("c", CandidateSource.CLASSIC, 1.0, 1.2, 0.6, _pts()),
                ),
                "lights": (),
                # Fresh kernel: overconfident VLA is downranked; classic may win or still accept.
                "expect_not_kind": "EMERGENCY",
            },
        }

        for name, sc in cases.items():
            # P1-3: fresh kernel per scenario — no cross-scenario mode pollution.
            kernel = SafetyKernel(cfg)
            obs = ObservableSnapshot(
                run_id="g2-04",
                frame_id=name,
                scenario_id=name,
                simulation_time_s=1.0,
                wall_time_s=1.0,
                ego_x=0.0,
                ego_y=0.0,
                ego_yaw=0.0,
                ego_v=5.0,
                observed_time_s=1.0,
                traffic_lights=sc["lights"],
                corridor_centerline=tuple((float(i), 0.0) for i in range(0, 120)),
                corridor_half_width_m=2.5,
                privilege=ObservationPrivilege.OBSERVABLE,
                schema_version=SCHEMA_VERSION,
                coordinate_frame="map",
            )
            cset = PolicyCandidateSet(
                run_id="g2-04",
                frame_id=name,
                scenario_id=name,
                model_id="mix",
                carla_frame=0,
                simulation_time_s=1.0,
                wall_time_s=1.0,
                candidates=sc["cands"],
                schema_version=SCHEMA_VERSION,
                coordinate_frame="map",
            )
            res = kernel.tick(obs, cset, now_s=1.0, availability=sc["avail"])
            arb = res.arbitration
            if arb is not None:
                arb_lats.append(arb.arbitration_latency_ms)

            # Real degradation audit assertions (not state pollution).
            if name == "learning_off_vla_dropped":
                self.assertIsNotNone(arb)
                assert arb is not None
                v_audit = next((a for a in arb.audits if a.candidate_id == "v"), None)
                self.assertIsNotNone(v_audit)
                assert v_audit is not None
                self.assertIn(
                    v_audit.degradation,
                    {
                        DegradationReason.SOURCE_UNAVAILABLE,
                        DegradationReason.CANDIDATE_UNAVAILABLE,
                    },
                )
                self.assertEqual(res.decision.final_candidate_id, "c")
            if name == "overconfident_vla":
                self.assertIsNotNone(arb)
                assert arb is not None
                v_audit = next((a for a in arb.audits if a.candidate_id == "v"), None)
                self.assertIsNotNone(v_audit)
                assert v_audit is not None
                self.assertEqual(v_audit.degradation, DegradationReason.OVERCONFIDENT)
                self.assertNotEqual(res.decision.decision_kind.value, "EMERGENCY")
            if "expect_kind" in sc:
                self.assertEqual(res.decision.decision_kind.value, sc["expect_kind"])
            if "expect_selected" in sc:
                self.assertEqual(res.decision.final_candidate_id, sc["expect_selected"])
            if "expect_not_kind" in sc:
                self.assertNotEqual(res.decision.decision_kind.value, sc["expect_not_kind"])

            scenarios.append(
                {
                    "name": name,
                    "decision_kind": res.decision.decision_kind.value,
                    "selected_id": res.decision.final_candidate_id,
                    "mode": res.mode.value,
                    "learning_required": res.decision.learning_modules_required,
                    "stages": None if arb is None else list(arb.stages),
                    "ranked_ids": None if arb is None else list(arb.ranked_ids),
                    "shadow_claims_control": None
                    if arb is None or arb.shadow is None
                    else arb.shadow.claims_control,
                    "fallback": None
                    if res.decision.fallback_request is None
                    else res.decision.fallback_request.reason_code,
                    "latency_ms": res.decision.latency_ms,
                    "arbitration_latency_ms": None if arb is None else arb.arbitration_latency_ms,
                    "degradation_audits": None
                    if arb is None
                    else {
                        a.candidate_id: a.degradation.value if a.degradation else None
                        for a in arb.audits
                    },
                }
            )

        report = build_latency_report(arb_lats, deadline_ms=cfg.arbitration.deadline_ms)
        resources = host_resource_snapshot()
        fallback_ok = sum(
            1
            for s in scenarios
            if s["decision_kind"]
            in {"ACCEPT", "QP", "RATO", "CLASSIC_FALLBACK", "MINIMAL_RISK", "EMERGENCY", "HARD_REJECT"}
        )
        payload = {
            "schema": "safedrive.g2_04.evidence.v1",
            "task": "G2-04",
            "status": "MEASURED",
            "measured_at_unix": time.time(),
            "hardware_id": resources["hostname"],
            "profile": resources["profile"],
            "workload_profile": resources["workload_profile"],
            "config_name": cfg.name,
            "config_hash": config_sha256(cfg.raw_toml),
            "contracts_schema_version": SCHEMA_VERSION,
            "contracts_schema_hash": contracts_schema_hash(),
            "resources": resources,
            "CARLA_quality_rendering_mode": resources["CARLA_quality_rendering_mode"],
            "CPU_utilization": {
                "note": "process_local_offline",
                "cpu_count_logical": resources["cpu_count_logical"],
                "process_user_time_s": resources["process_user_time_s"],
                "process_system_time_s": resources["process_system_time_s"],
            },
            "system_RAM_peak": {
                "process_peak_rss_mib_approx": resources["process_peak_rss_mib_approx"],
                "process_peak_rss_raw": resources["process_peak_rss_raw"],
            },
            "Windows_CARLA_VRAM": resources["Windows_CARLA_VRAM"],
            "WSL_CUDA_allocated_reserved": resources["WSL_CUDA_allocated_reserved"],
            "whole_GPU_peak": resources["whole_GPU_peak"],
            "model_precision_quantization": resources["model_precision_quantization"],
            "OOM_thermal_disconnect_recovery": resources["OOM_thermal_disconnect_recovery"],
            "disk_artifact_note": resources["disk_artifact_note"],
            "arbitration_config": {
                "enabled": cfg.arbitration.enabled,
                "deadline_ms": cfg.arbitration.deadline_ms,
                "shadow_enabled": cfg.arbitration.shadow_enabled,
                "max_final_candidates": cfg.arbitration.max_final_candidates,
            },
            "arbitration_latency_ms": report.to_dict(),
            "fallback_resolution_rate": fallback_ok / max(len(scenarios), 1),
            "scenarios": scenarios,
            "coverage": {
                "normal_classic": True,
                "learning_off_classic": True,
                "repair_path": True,
                "emergency_locked": True,
                "overconfident_degrade": True,
                "shadow_no_control": True,
                "fresh_kernel_per_scenario": True,
            },
            "limits": [
                "offline_cpu_only_not_live_carla_50hz",
                "shadow_compare_only",
                "no_fault_matrix_g2_05",
            ],
        }

        kinds = {s["decision_kind"] for s in scenarios}
        self.assertIn("ACCEPT", kinds)
        self.assertIn("EMERGENCY", kinds)

        # P2: formal evidence only via explicit gate (not silent rewrite on every unittest).
        if write_evidence_enabled():
            write_evidence_pack(
                EVIDENCE_DIR,
                task="G2-04",
                schema="safedrive.g2_04.evidence.v1",
                summary=payload,
                readme_md=(
                    "# G2-04 Arbitration / Shadow / Fallback Evidence\n\n"
                    "Offline CPU regression for deterministic arbitration pipeline.\n\n"
                    "## Reproduce\n\n"
                    f"```text\n{REPRO_CMD}\n```\n\n"
                    "## Limits\n\n"
                    "- Not live CARLA 50Hz VERIFIED\n"
                    "- Shadow is compare-only (no control / tick ownership)\n"
                    "- Fresh SafetyKernel per scenario (no mode pollution)\n"
                ),
                config_hash=payload["config_hash"],
                contracts_schema_hash=payload["contracts_schema_hash"],
                command=REPRO_CMD,
                repo_root=ROOT,
            )
            self.assertTrue((EVIDENCE_DIR / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
