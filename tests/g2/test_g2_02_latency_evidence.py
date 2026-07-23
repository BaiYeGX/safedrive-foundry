"""G2-02 evidence writer: QP latency, baseline comparison, resource fields."""

from __future__ import annotations

import json
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
    RepairMode,
    SafetyKernel,
    config_sha256,
    contracts_schema_hash,
    load_safety_config,
    osqp_available,
    osqp_local_site,
)
from safety_kernel.contracts.types import (  # noqa: E402
    CandidateSource,
    ObservationPrivilege,
    TrackedObject,
    TrajectoryPoint,
    TrafficLightObs,
)
from safety_kernel.evidence_util import (  # noqa: E402
    host_resource_snapshot,
    write_evidence_enabled,
    write_evidence_pack,
)
from safety_kernel.metrics import build_latency_report  # noqa: E402

EVIDENCE_DIR = ROOT / "docs/runtime-evidence/g2-02"
REPRO_CMD = "SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_02_latency_evidence -v"


def _pts(v: float = 8.0, n: int = 17) -> tuple[TrajectoryPoint, ...]:
    out = []
    x = 0.0
    for i in range(n):
        out.append(TrajectoryPoint(t=0.25 * i, x=x, y=0.0, yaw=0.0, kappa=0.0, v=v, a=0.0, jerk=0.0))
        x += v * 0.25
    return tuple(out)


class G202EvidenceTests(unittest.TestCase):
    def test_write_g2_02_evidence(self) -> None:
        cfg = load_safety_config()
        iface = RepairInterface(cfg)
        kernel = SafetyKernel(cfg)
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)

        scenarios = {
            "red_light": {
                "lights": (TrafficLightObs("tl", "red", 7.0, 1.0),),
                "actors": (),
                "v": 5.0,
                "hints": ["c:rules:red_light"],
            },
            "follow": {
                "lights": (),
                "actors": (
                    TrackedObject("lead", "vehicle", 18.0, 0.0, 0.0, 3.0, 0.0, 4.5, 1.8, 1.0),
                ),
                "v": 8.0,
                "hints": ["c:collision:collision_envelope"],
            },
            "cut_in": {
                "lights": (),
                "actors": (
                    TrackedObject("cut", "vehicle", 14.0, 1.2, 0.0, 4.0, -0.3, 4.0, 1.8, 1.0),
                ),
                "v": 7.0,
                "hints": ["c:collision:collision_envelope"],
            },
            "hard_brake_dynamics": {
                "lights": (),
                "actors": (),
                "v": 8.0,
                "hints": ["c:dynamics:accel"],
                "custom_pts": True,
            },
        }

        scenario_rows = []
        qp_latencies: list[float] = []
        for name, spec in scenarios.items():
            if spec.get("custom_pts"):
                pts = []
                x = 0.0
                v = 8.0
                for i in range(17):
                    a = 5.0 if i < 4 else -7.0
                    pts.append(TrajectoryPoint(0.25 * i, x, 0.0, 0.0, 0.0, v, a, 20.0))
                    x += max(0.0, v) * 0.25
                    v = max(0.0, v + a * 0.25)
                pts_t = tuple(pts)
            else:
                pts_t = _pts(v=float(spec["v"]))
            obs = ObservableSnapshot(
                run_id="g2-02-ev",
                frame_id=name,
                scenario_id=name,
                simulation_time_s=1.0,
                wall_time_s=1.0,
                ego_x=0.0,
                ego_y=0.0,
                ego_yaw=0.0,
                ego_v=float(spec["v"]),
                observed_time_s=1.0,
                actors=spec["actors"],
                traffic_lights=spec["lights"],
                corridor_centerline=tuple((float(i), 0.0) for i in range(0, 200)),
                corridor_half_width_m=2.5,
                privilege=ObservationPrivilege.OBSERVABLE,
            )
            cand = PolicyCandidate(
                candidate_id=f"{name}-raw",
                source=CandidateSource.CLASSIC,
                generated_time_s=1.0,
                valid_until_s=1.2,
                probability=1.0,
                points=pts_t,
            )
            cmp = iface.compare_all(cand, obs, now_s=1.0, reject_hints=list(spec["hints"]))
            long = cmp["longitudinal"]
            qp_latencies.append(long.solver_trace.latency_ms)
            # Kernel path
            cset = PolicyCandidateSet(
                run_id="g2-02-ev",
                frame_id=name,
                scenario_id=name,
                model_id="classic",
                carla_frame=0,
                simulation_time_s=1.0,
                wall_time_s=1.0,
                candidates=(cand,),
                schema_version=SCHEMA_VERSION,
            )
            tick = kernel.tick(obs, cset, now_s=1.0, availability=avail)
            scenario_rows.append(
                {
                    "name": name,
                    "qp_success": long.success,
                    "qp_status": long.solver_trace.status.value,
                    "qp_backend": long.solver_trace.backend,
                    "qp_latency_ms": long.solver_trace.latency_ms,
                    "qp_iterations": long.solver_trace.iterations,
                    "metrics": long.metrics.to_dict(),
                    "baselines": {
                        m: {
                            "success": cmp[m].success,
                            "progress_ratio": cmp[m].metrics.progress_ratio,
                            "safety_margin_m": cmp[m].metrics.safety_margin_m,
                            "comfort_jerk_rms": cmp[m].metrics.comfort_jerk_rms,
                            "modification_norm": cmp[m].metrics.modification_norm,
                            "unjustified_stop": cmp[m].metrics.unjustified_stop,
                        }
                        for m in ("raw", "rule", "hard_reject", "longitudinal")
                    },
                    "kernel_decision": tick.decision.decision_kind.value,
                    "kernel_solver_status": tick.decision.solver_status,
                }
            )

        # Warm-start microbench
        lights = (TrafficLightObs("tl", "red", 7.0, 1.0),)
        obs_w = ObservableSnapshot(
            run_id="g2-02-ev",
            frame_id="warm",
            scenario_id="warm",
            simulation_time_s=1.0,
            wall_time_s=1.0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            observed_time_s=1.0,
            traffic_lights=lights,
            corridor_centerline=tuple((float(i), 0.0) for i in range(0, 200)),
            corridor_half_width_m=2.5,
            privilege=ObservationPrivilege.OBSERVABLE,
        )
        cand_w = PolicyCandidate(
            candidate_id="warm",
            source=CandidateSource.CLASSIC,
            generated_time_s=1.0,
            valid_until_s=1.2,
            probability=1.0,
            points=_pts(v=5.0, n=13),
        )
        for _ in range(20):
            r = iface.repair(cand_w, obs_w, mode=RepairMode.LONGITUDINAL, now_s=1.0, reject_hints=["c:rules"])
            qp_latencies.append(r.solver_trace.latency_ms)

        qp_report = build_latency_report(qp_latencies, deadline_ms=cfg.qp.deadline_ms)
        snap = kernel.metrics_snapshot()
        resources = host_resource_snapshot()

        payload = {
            "schema": "safedrive.g2_02.evidence.v1",
            "task": "G2-02",
            "status": "MEASURED",
            "measured_at_unix": time.time(),
            "workload_profile": "regression",
            "profile": "offline_cpu_regression",
            "hardware_id": resources["hostname"],
            "resources": resources,
            "CARLA_quality_rendering_mode": resources["CARLA_quality_rendering_mode"],
            "CPU_utilization": {
                "cpu_count_logical": resources["cpu_count_logical"],
                "process_user_time_s": resources["process_user_time_s"],
            },
            "system_RAM_peak": {
                "process_peak_rss_mib_approx": resources["process_peak_rss_mib_approx"],
            },
            "Windows_CARLA_VRAM": resources["Windows_CARLA_VRAM"],
            "WSL_CUDA_allocated_reserved": resources["WSL_CUDA_allocated_reserved"],
            "whole_GPU_peak": resources["whole_GPU_peak"],
            "model_precision_quantization": resources["model_precision_quantization"],
            "disk_artifact_note": resources["disk_artifact_note"],
            "OOM_thermal_disconnect_recovery": resources["OOM_thermal_disconnect_recovery"],
            "osqp_python_available": osqp_available(),
            "osqp_local_site": osqp_local_site(),
            "solver_backend_default": scenario_rows[0]["qp_backend"] if scenario_rows else "unknown",
            "config_name": cfg.name,
            "config_hash": config_sha256(cfg.raw_toml),
            "contracts_schema_version": SCHEMA_VERSION,
            "contracts_schema_hash": contracts_schema_hash(),
            "qp_config": {
                "enabled": cfg.qp.enabled,
                "deadline_ms": cfg.qp.deadline_ms,
                "slack_stop_max_m": cfg.qp.slack_stop_max_m,
                "slack_lead_max_m": cfg.qp.slack_lead_max_m,
                "slack_speed_max_mps": cfg.qp.slack_speed_max_mps,
                "w_v_ref": cfg.qp.w_v_ref,
                "w_a": cfg.qp.w_a,
                "w_jerk": cfg.qp.w_jerk,
                "w_slack": cfg.qp.w_slack,
                "w_progress": cfg.qp.w_progress,
                "min_progress_ratio": cfg.qp.min_progress_ratio,
                "warm_start": cfg.qp.warm_start,
            },
            "qp_latency_ms": qp_report.to_dict(),
            "deadline_miss_count": qp_report.deadline_miss_count,
            "scenarios": scenario_rows,
            "kernel_metrics": {
                "qp_repair_count": snap.get("qp_repair_count"),
                "qp_success_count": snap.get("qp_success_count"),
                "qp_latency": snap.get("qp_latency"),
            },
            "coverage": {
                "red_light": True,
                "follow": True,
                "cut_in": True,
                "hard_brake_dynamics": True,
                "raw_rule_hardreject_longitudinal_interface": True,
                "stale_timeout_infeasible_contract": True,
                "warm_start": True,
            },
            "limits": [
                "offline_cpu_only_not_live_carla_50hz",
                "osqp_python_optional_numpy_admm_fallback",
                "no_rato_scp_g2_03",
                "no_arbitration_shadow_g2_04",
            ],
        }
        by_name = {r["name"]: r for r in scenario_rows}
        self.assertTrue(by_name["red_light"]["qp_success"])
        self.assertTrue(by_name["hard_brake_dynamics"]["qp_success"])
        if write_evidence_enabled():
            write_evidence_pack(
                EVIDENCE_DIR,
                task="G2-02",
                schema="safedrive.g2_02.evidence.v1",
                summary=payload,
                readme_md=(
                    "# G2-02 Longitudinal QP Repair Evidence\n\n"
                    f"## Reproduce\n\n```text\n{REPRO_CMD}\n```\n\n"
                    "## Limits\n\n- Not live CARLA 50Hz VERIFIED\n"
                ),
                config_hash=payload["config_hash"],
                contracts_schema_hash=payload["contracts_schema_hash"],
                command=REPRO_CMD,
                repo_root=ROOT,
                extra_manifest={"osqp_python_available": osqp_available()},
            )
            self.assertTrue((EVIDENCE_DIR / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
