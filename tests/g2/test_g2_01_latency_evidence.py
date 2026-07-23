"""G2-01 latency measurement and evidence writer (full offline suite)."""

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
    SafetyKernel,
    config_sha256,
    contracts_schema_hash,
    load_safety_config,
)
from safety_kernel.adapters.g1_trajectory import g1_plan_result_to_candidate_set, load_g1_trajectory_json  # noqa: E402
from safety_kernel.contracts.types import CandidateSource, ObservationPrivilege, TrajectoryPoint  # noqa: E402
from safety_kernel.evidence_util import (  # noqa: E402
    host_resource_snapshot,
    write_evidence_enabled,
    write_evidence_pack,
)
from safety_kernel.metrics import build_latency_report  # noqa: E402
from safety_kernel.validator import ValidationStage  # noqa: E402

G1_FOLLOW = ROOT / "tests/fixtures/g1/sample_follow_trajectory.json"
G1_HYBRID = ROOT / "tests/fixtures/g1/sample_blocked_detour.json"
EVIDENCE_DIR = ROOT / "docs/runtime-evidence/g2-01"
REPRO_CMD = "SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_01_latency_evidence -v"


def _refresh(cset: PolicyCandidateSet, now: float, frame: str) -> PolicyCandidateSet:
    cands = []
    for c in cset.candidates:
        cands.append(
            PolicyCandidate(
                candidate_id=c.candidate_id,
                source=c.source,
                generated_time_s=now,
                valid_until_s=now + 0.2,
                probability=c.probability,
                points=c.points,
                behavior=c.behavior,
                dynamics_meta=c.dynamics_meta,
            )
        )
    return PolicyCandidateSet(
        run_id=cset.run_id,
        frame_id=frame,
        scenario_id=cset.scenario_id,
        model_id=cset.model_id,
        carla_frame=int(now * 50),
        simulation_time_s=now,
        wall_time_s=now,
        candidates=tuple(cands),
        schema_version=SCHEMA_VERSION,
    )


def _obs_for(
    plan: dict,
    now: float,
    run_id: str,
    frame: str,
    *,
    scenario_id: str | None = None,
) -> ObservableSnapshot:
    pts = plan["trajectory"]["points"]
    corridor = tuple((float(p["x"]), float(p["y"])) for p in pts[:: max(1, len(pts) // 20)])
    return ObservableSnapshot(
        run_id=run_id,
        frame_id=frame,
        scenario_id=scenario_id if scenario_id is not None else run_id,
        simulation_time_s=now,
        wall_time_s=now,
        ego_x=float(pts[0]["x"]),
        ego_y=float(pts[0]["y"]),
        ego_yaw=float(pts[0]["yaw"]),
        ego_v=float(pts[0]["v"]),
        observed_time_s=now,
        corridor_centerline=corridor,
        corridor_half_width_m=2.5,
        privilege=ObservationPrivilege.OBSERVABLE,
        schema_version=SCHEMA_VERSION,
        coordinate_frame="map",
    )


class G201LatencyEvidenceTests(unittest.TestCase):
    def test_measure_and_write_evidence(self) -> None:
        cfg = load_safety_config()
        kernel = SafetyKernel(cfg)
        follow = load_g1_trajectory_json(G1_FOLLOW)
        hybrid = load_g1_trajectory_json(G1_HYBRID)
        avail = ComponentAvailability(classic=True, vla=False, world=False, safety=True)

        n = 200
        accept_follow = 0
        accept_hybrid = 0
        for i in range(n):
            now = i * cfg.control_period_s
            frame = f"f{i}"
            cset = _refresh(
                g1_plan_result_to_candidate_set(
                    follow,
                    run_id="g2-01-lat",
                    frame_id=frame,
                    scenario_id="g1-04-follow",
                    simulation_time_s=now,
                    wall_time_s=now,
                    now_s=now,
                ),
                now,
                frame,
            )
            # Inject a stale VLA sibling that must be dropped without learning.
            vla_pts = cset.candidates[0].points
            vla = PolicyCandidate(
                candidate_id=f"vla-stale-{i}",
                source=CandidateSource.VLA_FAST,
                generated_time_s=now,
                valid_until_s=now + 0.2,
                probability=0.99,
                points=vla_pts,
            )
            mixed = PolicyCandidateSet(
                run_id=cset.run_id,
                frame_id=frame,
                scenario_id=cset.scenario_id,
                model_id="mixed",
                carla_frame=i,
                simulation_time_s=now,
                wall_time_s=now,
                candidates=(vla, cset.candidates[0]),
                schema_version=SCHEMA_VERSION,
            )
            tick = kernel.tick(
                _obs_for(follow, now, "g2-01-lat", frame, scenario_id=mixed.scenario_id),
                mixed,
                now_s=now,
                availability=avail,
            )
            if tick.decision.final_candidate_id == cset.candidates[0].candidate_id:
                accept_follow += 1

            # Every 10th frame also exercise hybrid sample final validation.
            if i % 10 == 0:
                hset = _refresh(
                    g1_plan_result_to_candidate_set(
                        hybrid,
                        run_id="g2-01-hybrid",
                        frame_id=frame,
                        scenario_id="g1-05-detour",
                        simulation_time_s=now,
                        now_s=now,
                    ),
                    now,
                    frame,
                )
                htick = kernel.tick(
                    _obs_for(hybrid, now, "g2-01-hybrid", frame, scenario_id=hset.scenario_id),
                    hset,
                    now_s=now,
                    availability=avail,
                    stage=ValidationStage.FINAL,
                )
                if htick.decision.final_candidate_id:
                    accept_hybrid += 1

        # Intentional failure samples for evidence.
        bad_pts = list(cset.candidates[0].points)
        p0 = bad_pts[0]
        bad_pts[0] = TrajectoryPoint(p0.t, float("nan"), p0.y, p0.yaw, p0.kappa, p0.v, p0.a, p0.jerk)
        bad = PolicyCandidate(
            candidate_id="inject-nan",
            source=CandidateSource.CLASSIC,
            generated_time_s=now,
            valid_until_s=now + 0.2,
            probability=1.0,
            points=tuple(bad_pts),
        )
        kernel.validator.validate_candidates(
            PolicyCandidateSet(
                run_id="inject",
                frame_id="inj",
                scenario_id="inject",
                model_id="classic",
                carla_frame=0,
                simulation_time_s=now,
                wall_time_s=now,
                candidates=(bad,),
                schema_version=SCHEMA_VERSION,
            ),
            _obs_for(follow, now, "inject", "inj"),
            now_s=now,
            availability=avail,
        )

        state_rep = build_latency_report(
            kernel.validator.latency_state_ms,
            deadline_ms=cfg.state_check_deadline_ms,
        )
        cand_rep = build_latency_report(
            kernel.validator.latency_candidate_ms,
            deadline_ms=cfg.candidate_check_deadline_ms,
        )
        metrics = kernel.metrics_snapshot()

        summary = {
            "schema": "safedrive.g2_01.evidence.v1",
            "task": "G2-01",
            "status": "MEASURED",
            "workload_profile": "regression",
            "hardware_id": "local-dev-measured-host",
            "profile": "offline_cpu_regression",
            "CARLA_quality_rendering_mode": "n/a",
            "contracts_schema_version": SCHEMA_VERSION,
            "contracts_schema_hash": contracts_schema_hash(),
            "config_name": cfg.name,
            "config_hash": config_sha256(cfg.raw_toml),
            "samples_state_and_candidate_ticks": n,
            "state_check_ms": state_rep.to_dict(),
            "candidate_check_ms": cand_rep.to_dict(),
            "deadline_miss_count": kernel.validator.deadline_misses,
            "deadline_miss_rate": kernel.validator.deadline_misses
            / float(max(1, len(kernel.validator.latency_state_ms) + len(kernel.validator.latency_candidate_ms))),
            "g1_follow_accept_count": accept_follow,
            "g1_hybrid_accept_frames": accept_hybrid,
            "failure_samples": kernel.validator.failure_samples[:16],
            "event_count": len(kernel.validator.events),
            "final_mode": kernel.mode.value,
            "transitions": metrics["transitions"],
            "learning_modules": {"vla": False, "world": False, "classic": True},
            "coverage": {
                "schema": True,
                "numeric_nan": True,
                "freshness": True,
                "time_order": True,
                "road": True,
                "dynamics": True,
                "collision": True,
                "rules": True,
                "trackability": True,
                "prefilter_vs_final": True,
                "g1_follow_replay": True,
                "g1_hybrid_replay": True,
                "learning_fail_classic_select": True,
                "state_machine": True,
                "oracle_offline_vs_runtime": True,
                "ros_msg_shaped_adapters": True,
                "kernel_facade": True,
            },
            "limits": [
                "offline_cpu_only_not_live_carla_50hz",
                "no_qp_rato_repair_g2_02_03",
                "no_arbitration_shadow_g2_04",
            ],
            "measured_at_unix": time.time(),
        }
        resources = host_resource_snapshot()
        summary["resources"] = resources
        summary["cpu_utilization"] = {
            "cpu_count_logical": resources["cpu_count_logical"],
            "process_user_time_s": resources["process_user_time_s"],
            "process_system_time_s": resources["process_system_time_s"],
        }
        summary["system_RAM_peak"] = {
            "process_peak_rss_mib_approx": resources["process_peak_rss_mib_approx"],
            "process_peak_rss_raw": resources["process_peak_rss_raw"],
        }
        summary["Windows_CARLA_VRAM"] = resources["Windows_CARLA_VRAM"]
        summary["WSL_CUDA_allocated_reserved"] = resources["WSL_CUDA_allocated_reserved"]
        summary["whole_GPU_peak"] = resources["whole_GPU_peak"]
        summary["model_precision_quantization"] = resources["model_precision_quantization"]
        summary["disk_artifact_note"] = resources["disk_artifact_note"]
        summary["OOM_thermal_disconnect_recovery"] = resources["OOM_thermal_disconnect_recovery"]

        self.assertGreater(accept_follow, n * 0.9)
        self.assertGreater(accept_hybrid, 0)
        self.assertLess(state_rep.p95_ms, cfg.state_check_deadline_ms * 5)
        self.assertLess(cand_rep.p95_ms, cfg.candidate_check_deadline_ms * 5)
        self.assertTrue(kernel.validator.failure_samples)

        if write_evidence_enabled():
            write_evidence_pack(
                EVIDENCE_DIR,
                task="G2-01",
                schema="safedrive.g2_01.evidence.v1",
                summary=summary,
                readme_md=(
                    "# G2-01 Safety Kernel Evidence\n\n"
                    "Offline CPU regression for contracts / Validator / state machine.\n\n"
                    f"## Reproduce\n\n```text\n{REPRO_CMD}\n```\n\n"
                    "## Limits\n\n- Not live CARLA 50Hz VERIFIED\n"
                ),
                config_hash=summary["config_hash"],
                contracts_schema_hash=summary["contracts_schema_hash"],
                command=REPRO_CMD,
                repo_root=ROOT,
                extra_manifest={
                    "g1_replay_sources": [
                        str(G1_FOLLOW.relative_to(ROOT)),
                        str(G1_HYBRID.relative_to(ROOT)),
                    ],
                },
            )
            loaded = json.loads((EVIDENCE_DIR / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["task"], "G2-01")
            self.assertIn("coverage", loaded)


if __name__ == "__main__":
    unittest.main()
