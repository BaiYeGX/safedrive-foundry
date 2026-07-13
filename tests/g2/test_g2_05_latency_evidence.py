"""G2-05 stage acceptance evidence (offline): fault matrix + mode comparison."""

from __future__ import annotations

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
    RepairInterface,
    SafetyKernel,
    TrajectoryPoint,
    config_sha256,
    contracts_schema_hash,
    load_safety_config,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservationPrivilege,
    TrackedObject,
)
from safety_kernel.evidence_util import (  # noqa: E402
    host_resource_snapshot,
    write_evidence_enabled,
    write_evidence_pack,
)
from safety_kernel.faults import (  # noqa: E402
    DEFAULT_MATRIX,
    apply_fault_to_obs,
    apply_fault_to_set,
    expected_action_holds,
)
from safety_kernel.metrics import build_latency_report  # noqa: E402

EVIDENCE_DIR = ROOT / "docs/architecture/evidence/g2-05"
REPRO_CMD = "SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_05_latency_evidence -v"


def _pts(v: float = 6.0, n: int = 16) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(TrajectoryPoint(t=0.25 * i, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
        x += v * 0.25
    return tuple(out)


class G205EvidenceTests(unittest.TestCase):
    def test_write_g2_05_evidence(self) -> None:
        cfg = load_safety_config()
        iface = RepairInterface(cfg)
        avail = ComponentAvailability(classic=True, vla=True, world=False, safety=True)
        now = 1.0
        lead = TrackedObject("lead", "vehicle", 14.0, 0.0, 0.0, 3.0, 0.0, 4.5, 1.8, now)
        base_obs = ObservableSnapshot(
            run_id="g2-05",
            frame_id="f0",
            scenario_id="stage",
            simulation_time_s=now,
            wall_time_s=now,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=6.0,
            observed_time_s=now,
            actors=(lead,),
            corridor_centerline=tuple((float(i), 0.0) for i in range(0, 150)),
            corridor_half_width_m=2.5,
            privilege=ObservationPrivilege.OBSERVABLE,
            schema_version=SCHEMA_VERSION,
            coordinate_frame="map",
        )
        base_cand = PolicyCandidate(
            candidate_id="raw",
            source=CandidateSource.CLASSIC,
            generated_time_s=now,
            valid_until_s=now + 0.2,
            probability=1.0,
            points=_pts(v=6.0),
        )
        base_cset = PolicyCandidateSet(
            run_id="g2-05",
            frame_id="f0",
            scenario_id="stage",
            model_id="classic",
            carla_frame=0,
            simulation_time_s=now,
            wall_time_s=now,
            candidates=(base_cand,),
            schema_version=SCHEMA_VERSION,
            coordinate_frame="map",
        )

        fault_rows = []
        tick_lats: list[float] = []
        for fault in DEFAULT_MATRIX:
            # Fresh kernel per fault — no EMERGENCY dwell pollution across matrix rows.
            kernel = SafetyKernel(cfg)
            obs = apply_fault_to_obs(base_obs, fault, now_s=now)
            cset = apply_fault_to_set(base_cset, fault, now_s=now)
            t0 = time.perf_counter()
            res = kernel.tick(obs, cset, now_s=now, availability=avail)
            lat = (time.perf_counter() - t0) * 1000.0
            tick_lats.append(lat)
            repair_ok = None if res.repair_result is None else res.repair_result.success
            self.assertTrue(
                expected_action_holds(
                    fault,
                    res.decision.decision_kind.value,
                    repair_success=repair_ok,
                ),
                msg=f"{fault.fault_id.value} expected_action failed",
            )
            fault_rows.append(
                {
                    **fault.to_dict(),
                    "decision_kind": res.decision.decision_kind.value,
                    "mode": res.mode.value,
                    "selected_id": res.decision.final_candidate_id,
                    "learning_required": res.decision.learning_modules_required,
                    "reject_reasons": list(res.decision.reject_reasons)[:8],
                    "tick_latency_ms": lat,
                    "repair_success": repair_ok,
                    "expected_action_ok": True,
                }
            )

        cmp = iface.compare_all(
            base_cand,
            base_obs,
            now_s=now,
            reject_hints=["c:collision"],
        )
        mode_cmp = {
            mode: {
                "success": r.success,
                "status": r.solver_trace.status.value,
                "progress_ratio": r.metrics.progress_ratio,
                "modification_norm": r.metrics.modification_norm,
                "comfort_jerk_rms": r.metrics.comfort_jerk_rms,
                "safety_margin_m": r.metrics.safety_margin_m,
                "unjustified_stop": r.metrics.unjustified_stop,
            }
            for mode, r in cmp.items()
        }

        normal = SafetyKernel(cfg).tick(
            base_obs,
            base_cset,
            now_s=now,
            availability=ComponentAvailability(classic=True, vla=False, world=False, safety=True),
        )

        report = build_latency_report(tick_lats, deadline_ms=50.0)
        resources = host_resource_snapshot()
        payload = {
            "schema": "safedrive.g2_05.evidence.v1",
            "task": "G2-05",
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
            "claim_c3_offline": {
                "modes_compared": list(mode_cmp.keys()),
                "mode_comparison": mode_cmp,
                "note": "Offline fair compare; not live CARLA C3 VERIFIED",
            },
            "learning_off_normal_decision": normal.decision.decision_kind.value,
            "learning_off_learning_required": normal.decision.learning_modules_required,
            "fault_matrix": fault_rows,
            "tick_latency_ms": report.to_dict(),
            "oracle_runtime_separation": "runtime_uses_observable_only",
            "live_carla_short_loop": "NOT_RUN",
            "coverage": {
                "stale_obs": True,
                "packet_drop": True,
                "out_of_order": True,
                "localization_bias": True,
                "missed_actor": True,
                "actor_offset": True,
                "stale_candidate": True,
                "numeric_nan": True,
                "vision_soft_degrade": True,
                "low_attachment": True,
                "actuator_saturation": True,
                "solver_timeout": True,
                "model_timeout": True,
                "raw_rule_hardreject_qp_rato": True,
                "learning_modules_off": True,
                "state_lock_no_qp_on_stale_obs": True,
                "fresh_kernel_per_fault": True,
            },
            "limits": [
                "offline_cpu_only_not_live_carla_50hz",
                "no_live_short_loop_classic_safety",
                "c3_not_live_verified",
                "collision_cv_circular_envelope_not_full_polygon",
                "red_light_near_zone_speed_gate_not_full_stopline_planner",
            ],
            "negative_results": [
                "Missed-actor fault can make Raw appear safe under Observable-only (documented detection gap)",
                "Live short-loop stage gate not executed in this environment",
            ],
            "remediation_notes": [
                "state hard faults lock tick (no ACCEPT/QP/RATO on untrusted obs)",
                "soft_stale applies to learning sources only",
                "NaN/Inf actors state-lock; identity contract hard-checked",
                "solver timeout never executes timed-out solution",
            ],
        }
        stale_row = next(r for r in fault_rows if r["fault_id"] == "stale_obs")
        self.assertIn(
            stale_row["decision_kind"],
            {"MINIMAL_RISK", "EMERGENCY", "HARD_REJECT"},
        )
        self.assertNotIn(stale_row["decision_kind"], {"ACCEPT", "QP", "RATO"})
        self.assertEqual(len(fault_rows), len(DEFAULT_MATRIX))
        self.assertFalse(payload["learning_off_learning_required"])

        if write_evidence_enabled():
            write_evidence_pack(
                EVIDENCE_DIR,
                task="G2-05",
                schema="safedrive.g2_05.evidence.v1",
                summary=payload,
                readme_md=(
                    "# G2-05 Fault Matrix & Safety Stage Evidence (Offline)\n\n"
                    "Offline CPU fault injection and mode comparison. **Not** live CARLA short-loop VERIFIED.\n\n"
                    "## Semantics\n\n"
                    "- State hard faults (stale obs, privilege, non-finite ego/actors) **lock** the tick.\n"
                    "- Soft-stale / OOD / overconfident gates apply to **learning** sources only.\n"
                    "- Solver timeout never executes a timed-out solution.\n"
                    "- Matrix includes low_attachment / actuator_saturation / solver_timeout / model_timeout.\n\n"
                    "## Reproduce\n\n"
                    f"```text\n{REPRO_CMD}\n```\n\n"
                    "## Limits\n\n"
                    "- No live CARLA short closed loop\n"
                    "- C3 claim remains offline MEASURED, not live VERIFIED\n"
                ),
                config_hash=payload["config_hash"],
                contracts_schema_hash=payload["contracts_schema_hash"],
                command=REPRO_CMD,
                repo_root=ROOT,
                extra_manifest={"live_carla": False},
            )
            self.assertTrue((EVIDENCE_DIR / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
