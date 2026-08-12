"""Live paired evaluation for one frozen K2 V3 anchor.

Exactly one SimLingo/head forward creates the anchor artifact.  Candidate
branches cold-rebuild the same registered fixture and never call the model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from driving_vla.adapter.policy_adapter import ObservationBundle
from driving_vla.evaluation.comparability import evaluate_pair_comparability
from driving_vla.evaluation.fixture_runtime import (
    FixtureError,
    apply_weather,
    cleanup_session,
    connect_world,
    measure_initial_state,
    open_fixture_session,
    restore_async,
)
from driving_vla.evaluation.k2_v3_artifact import K2AnchorArtifactV3
from driving_vla.evaluation.oracle_v2 import evaluate_pair_oracle_v2
from driving_vla.evaluation.paired_contract import (
    ObservationFingerprint,
    compute_pair_id,
    compute_run_id,
    content_hash,
    sha256_hex,
)
from driving_vla.evaluation.paired_live import (
    EXECUTOR_CONFIG_HASH,
    _attach_sensors,
    _destroy_sensors,
    _ego_pose,
    _file_sha256,
    _prepare_decision_state,
    _route_xy,
    _wait_camera,
    run_branch,
    set_spectator_follow,
)
from driving_vla.evaluation.scenario_registry import ScenarioSeedFixture
from driving_vla.model.k2_v3_types import load_k2_v3_config
from driving_vla.model.navigation_contract import RouteContextV3
from driving_vla.model.simlingo_contract import (
    SimLingoContractConfig,
    navigation_targets as contract_navigation_targets,
    resolve_navigation_prompt_conditioning,
)
from driving_vla.model.simlingo_runtime import (
    SIMLINGO_CAMERA_FOV_DEG,
    SIMLINGO_CAMERA_XYZ,
    SimLingoNeuralRuntime,
)
from driving_vla.runtime.basic1v1_observable import observe_basic1v1_actor
from driving_vla.runtime.navigation_topology import observe_traffic_control_v3

REPORT_SCHEMA = "safedrive.r2_v3.live_pair.v1"


def build_v3_model_identity(checkpoint: Path | str) -> dict[str, str]:
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError(f"K2 V3 checkpoint missing: {path}")
    config = load_k2_v3_config()
    checkpoint_hash = _file_sha256(path)
    model_hash = content_hash(
        {
            "policy": "NeuralV3Policy",
            "checkpoint_sha256": checkpoint_hash,
            "k2_v3_config_hash": config.config_hash(),
            "schema": config.schema_version,
        },
        nibble=64,
    )
    return {
        "checkpoint_sha256": checkpoint_hash,
        "k2_v3_config_hash": config.config_hash(),
        "model_retimer_hash": model_hash,
    }


def _frozen_route_context(fixture: ScenarioSeedFixture) -> RouteContextV3:
    navigation = dict(fixture.route.navigation_context or {})
    frozen = navigation.get("frozen_context_json")
    if not frozen:
        raise FixtureError("V3 fixture lacks frozen_context_json")
    value = json.loads(str(frozen))
    if not isinstance(value, Mapping):
        raise FixtureError("V3 frozen route context must be an object")
    return RouteContextV3.from_mapping(value)


def _current_route_context(
    base: RouteContextV3,
    *,
    ego: Any,
) -> RouteContextV3:
    signal, stop_line = observe_traffic_control_v3(ego)
    return replace(
        base,
        traffic_signal_state=signal,
        stop_line_distance_m=stop_line,
        route_hash="",
        topology_hash="",
    )


def run_anchor_v3(
    *,
    client: Any,
    world: Any,
    fixture: ScenarioSeedFixture,
    policy: Any,
    pair_id: str,
    model_checkpoint_hash: str,
    registry_hash: str,
    evidence_dir: Path,
    forward_counter: dict[str, int],
    spectator_follow: bool = True,
    include_scenario_family: bool = True,
) -> tuple[K2AnchorArtifactV3, dict[str, Any]]:
    apply_weather(world, fixture.weather)
    session = open_fixture_session(client, world, fixture, settle_ticks=8)
    sensors = None
    try:
        ego = next(item.actor for item in session.spawned if item.role == "ego")
        if spectator_follow:
            ok, error = set_spectator_follow(world, ego)
            if not ok:
                raise FixtureError(
                    f"spectator_follow_initialization_failed:{error}"
                )
        sensors = _attach_sensors(world, ego)
        image = _wait_camera(sensors, world, session)
        _prepare_decision_state(
            session,
            world,
            collect_observable_history=False,
        )
        image = _wait_camera(
            sensors,
            world,
            session,
            min_frames=1,
            max_ticks=10,
        )
        measured = measure_initial_state(session)
        pose = _ego_pose(ego)
        route_xy = _route_xy(fixture)
        base_context = _frozen_route_context(fixture)
        route_context = _current_route_context(base_context, ego=ego)
        actors = [
            item.actor for item in session.spawned if item.role != "ego"
        ]
        scene = observe_basic1v1_actor(ego=ego, actors=actors).to_dict()
        contract = SimLingoContractConfig(official_contract=True)
        contract.densify_ds_m = 10.0
        targets = contract_navigation_targets(
            route_xy,
            ego_x=pose.x,
            ego_y=pose.y,
            ego_yaw=pose.yaw,
            progress_hint_s=0.0,
            config=contract,
        )
        current_waypoint = world.get_map().get_waypoint(
            ego.get_location(),
            project_to_road=False,
        )
        current_road_id = (
            None
            if current_waypoint is None
            else int(getattr(current_waypoint, "road_id", 0))
        )
        current_lane_id = (
            None
            if current_waypoint is None
            else int(getattr(current_waypoint, "lane_id", 0))
        )
        nav_prompt = resolve_navigation_prompt_conditioning(
            maneuver=route_context.maneuver,
            target_distance_m=float(targets.target1_distance_m),
            current_road_id=current_road_id,
            current_lane_id=current_lane_id,
            target_road_id=route_context.target_road_id,
            target_lane_id=route_context.target_lane_id,
        )
        snapshot = world.get_snapshot()
        frame_id = f"anchor-v3-{int(snapshot.frame)}"
        obs_meta = {
            "official_contract": True,
            "image_layout": "bgr",
            "target_ego_1": targets.target_ego_1,
            "target_ego_2": targets.target_ego_2,
            "command_text": nav_prompt.command_text,
            "prompt_mode": nav_prompt.eval_route_as,
            "navigation_prompt_conditioning": nav_prompt.to_dict(),
            "current_road_id_v3": current_road_id,
            "current_lane_id_v3": current_lane_id,
            "camera_mount_xyz": list(SIMLINGO_CAMERA_XYZ),
            "route_context_v3": route_context.to_dict(),
            "observable_scene_v1": scene,
            "semantic_head_checkpoint_hash": model_checkpoint_hash,
        }
        if include_scenario_family:
            # Historical V3 audit field; V4 callers explicitly disable it.
            obs_meta["scenario_family_v3"] = fixture.family
        obs = ObservationBundle(
            run_id=pair_id,
            frame_id=frame_id,
            scenario_id=fixture.scenario_id,
            simulation_time_s=float(snapshot.timestamp.elapsed_seconds),
            wall_time_s=time.time(),
            carla_frame=int(snapshot.frame),
            ego_x=pose.x,
            ego_y=pose.y,
            ego_yaw=pose.yaw,
            ego_v=pose.speed_mps,
            route_xy=route_xy,
            front_rgb=image,
            meta=obs_meta,
        )
        forward_counter["n"] = int(forward_counter.get("n", 0)) + 1
        if forward_counter["n"] != 1:
            raise FixtureError("V3 paired anchor requires exactly one forward")
        started = time.perf_counter()
        bundle = policy.predict_bundle(obs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if int(policy.last_forward_count) != 1:
            raise FixtureError(
                f"V3 policy forward_count={policy.last_forward_count}, expected 1"
            )
        if bundle.guard_status != "OK":
            # Preserve the complete rejected bundle before aborting the slot.
            # A terse failure string is not enough to audit native geometry,
            # pointwise kappa, or raw-head binding, and interrupted slots must
            # remain recoverable rather than being overwritten on resume.
            failure_anchor = Path(evidence_dir) / "anchor"
            failure_anchor.mkdir(parents=True, exist_ok=True)
            (failure_anchor / "k2_v3_failure_bundle.json").write_text(
                json.dumps(bundle.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (failure_anchor / "guard_reject.json").write_text(
                json.dumps(
                    {
                        "schema_version": "safedrive.k2.v3.guard_reject.v1",
                        "guard_status": bundle.guard_status,
                        "guard_reasons": list(bundle.guard_reasons),
                        "guard_metrics": dict(bundle.guard_metrics),
                        "model_checkpoint_hash": model_checkpoint_hash,
                        "registry_hash": registry_hash,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise FixtureError(
                "V3_GUARD_REJECT:" + ",".join(bundle.guard_reasons)
            )
        fingerprint = ObservationFingerprint(
            front_rgb_sha256=sha256_hex(
                np.ascontiguousarray(image).tobytes()
            ),
            image_height=int(image.shape[0]),
            image_width=int(image.shape[1]),
            image_channels=int(image.shape[2]),
            image_layout="bgr",
            ego_observable={
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
                "v": pose.speed_mps,
            },
            route_targets=[
                list(targets.target_ego_1),
                list(targets.target_ego_2),
            ],
            camera_frame={
                "mount_xyz": list(SIMLINGO_CAMERA_XYZ),
                "fov": SIMLINGO_CAMERA_FOV_DEG,
                "carla_frame": int(snapshot.frame),
            },
            k2_bundle_hash=str(bundle.bundle_hash),
        )
        artifact = K2AnchorArtifactV3(
            pair_id=pair_id,
            scenario_id=fixture.scenario_id,
            seed_id=fixture.seed_id,
            anchor_run_id=compute_run_id(
                pair_id=pair_id,
                role="anchor",
                attempt_id=0,
            ),
            anchor_carla_frame=int(snapshot.frame),
            anchor_simulation_time_s=float(
                snapshot.timestamp.elapsed_seconds
            ),
            requested_initial_state_hash=(
                fixture.requested_initial_state_hash()
            ),
            measured_initial_state_hash=measured.measured_hash(),
            observation_fingerprint=fingerprint.to_dict(),
            model_checkpoint_hash=model_checkpoint_hash,
            executor_config_hash=EXECUTOR_CONFIG_HASH,
            bundle=bundle,
        )
        raw = artifact.to_json_bytes()
        roundtrip = K2AnchorArtifactV3.from_json_bytes(raw)
        if (
            roundtrip.artifact_content_hash()
            != artifact.artifact_content_hash()
        ):
            raise FixtureError("V3 anchor round-trip hash mismatch")
        anchor_dir = evidence_dir / "anchor"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        (anchor_dir / "anchor_bundle_v3.json").write_bytes(raw)
        np.save(anchor_dir / "anchor_front_rgb.npy", image)
        meta = {
            "pair_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "registry_hash": registry_hash,
            "artifact_schema": artifact.schema_version,
            "artifact_content_hash": artifact.artifact_content_hash(),
            "bundle_hash": bundle.bundle_hash,
            "guard_status": bundle.guard_status,
            "guard_reasons": list(bundle.guard_reasons),
            "candidate_ids": [
                candidate.candidate_id for candidate in bundle.candidates
            ],
            "candidate_available": [
                bool(candidate.available) for candidate in bundle.candidates
            ],
            "top1_index": int(bundle.top1_index),
            "route_hash": bundle.route_context.route_hash,
            "topology_hash": bundle.route_context.topology_hash,
            "forward_count": 1,
            "latency_ms": latency_ms,
            "peak_vram_mb": float(policy.last_peak_vram_mb),
            "navigation_prompt_conditioning": nav_prompt.to_dict(),
            "resolved_prompt_mode": obs.meta.get("resolved_prompt_mode"),
            "resolved_command_text": obs.meta.get("resolved_command_text"),
        }
        (anchor_dir / "run_config.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact, meta
    finally:
        if sensors is not None:
            _destroy_sensors(sensors, client=client)
        cleanup_session(session, soft=True)


def _safe(metrics: Any) -> bool:
    return bool(
        int(metrics.collision_episode_count) == 0
        and float(metrics.offroad_fraction) < 0.02
        and bool(metrics.completed_primary_horizon)
    )


def run_pair_v3(
    *,
    registry: Any,
    fixture: ScenarioSeedFixture,
    checkpoint: Path | str,
    evidence_dir: Path,
    host: str = "127.0.0.1",
    port: int = 2000,
    device: str = "cuda",
    shared_policy: Any | None = None,
    branch_order: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Execute one V3 paired fixture into an exclusive evidence directory."""
    if evidence_dir.exists():
        raise FileExistsError(f"refusing existing V3 pair dir {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    identity = build_v3_model_identity(checkpoint)
    registry_hash = str(
        registry.registry_sha256 or registry.compute_registry_sha256()
    )
    pair_id = compute_pair_id(
        scenario_registry_hash=registry_hash,
        scenario_id=fixture.scenario_id,
        seed_id=fixture.seed_id,
        model_checkpoint_config_retimer_hash=identity[
            "model_retimer_hash"
        ],
        executor_config_hash=EXECUTOR_CONFIG_HASH,
    )
    client = None
    world = None
    forward_counter = {"n": 0}
    try:
        client, world = connect_world(
            host=host,
            port=port,
            map_name=fixture.map_name,
            sim_dt_s=float(fixture.sim_dt_s),
            sync=True,
            timeout_s=90.0,
            retries=3,
        )
        if shared_policy is None:
            from driving_vla.model.neural_policy import NeuralV3Policy

            runtime = SimLingoNeuralRuntime(device=device)
            load = runtime.load()
            if not load.ok:
                raise RuntimeError(f"SimLingo load failed: {load.error}")
            policy = NeuralV3Policy(
                runtime=runtime,
                semantic_head_checkpoint=str(checkpoint),
                teacher_mode=False,
                keep_on_gpu=True,
                lazy=False,
                device=device,
                checkpoint_use="r2v3_blind_audit",
            )
        else:
            policy = shared_policy
        policy.ensure_loaded()
        artifact, anchor = run_anchor_v3(
            client=client,
            world=world,
            fixture=fixture,
            policy=policy,
            pair_id=pair_id,
            model_checkpoint_hash=identity["checkpoint_sha256"],
            registry_hash=registry_hash,
            evidence_dir=evidence_dir,
            forward_counter=forward_counter,
        )
        valid = dict(
            artifact.bundle.guard_metrics.get("candidate_valid") or {}
        )
        available = {
            index
            for index, candidate in enumerate(artifact.candidates)
            if bool(candidate.available)
            and bool(valid.get(candidate.candidate_id, False))
        }
        if 0 not in available:
            raise FixtureError("V3 nominal candidate unavailable at anchor")
        results = {}
        for index in branch_order:
            if index not in available:
                continue
            results[index] = run_branch(
                client=client,
                world=world,
                fixture=fixture,
                artifact=artifact,
                force_index=index,
                pair_id=pair_id,
                registry_hash=registry_hash,
                model_retimer_hash=identity["model_retimer_hash"],
                evidence_dir=evidence_dir,
                forward_counter=forward_counter,
                policy=policy,
                collect_actor_future=False,
            )
        comparable = False
        comparability: dict[str, Any] = {
            "status": "SINGLETON_NO_RANKING",
            "comparable": False,
            "reasons": ["NO_ALTERNATIVE"],
            "failure_codes": [],
        }
        oracle = None
        if 1 in results:
            comparison = evaluate_pair_comparability(
                anchor=artifact,  # duck-typed V3 anchor identity
                branch0=results[0].report,
                branch1=results[1].report,
                expected_registry_hash=registry_hash,
                expected_scenario_id=fixture.scenario_id,
                expected_seed_id=fixture.seed_id,
                expected_model_retimer_hash=identity[
                    "model_retimer_hash"
                ],
                expected_executor_config_hash=EXECUTOR_CONFIG_HASH,
            )
            comparability = comparison.to_dict()
            comparable = bool(comparison.comparable)
            oracle = evaluate_pair_oracle_v2(
                pair_id=pair_id,
                scenario_id=fixture.scenario_id,
                seed_id=fixture.seed_id,
                family=fixture.family,
                comparable=comparable,
                top1_index=artifact.top1_index,
                metrics0=results[0].metrics if comparable else None,
                metrics1=results[1].metrics if comparable else None,
                incomparable_reasons=comparison.reasons,
                candidate_ids=(
                    artifact.candidates[0].candidate_id,
                    artifact.candidates[1].candidate_id,
                ),
            )
        winner = (
            None if oracle is None else oracle.oracle_candidate_index
        )
        decisive = bool(
            comparable
            and oracle is not None
            and oracle.oracle_decision_level not in {
                None,
                "tie_top1",
            }
        )
        safe = {
            index: _safe(result.metrics)
            for index, result in results.items()
        }
        both_bad = bool(
            oracle.both_bad
            if oracle is not None
            else not safe.get(0, False)
        )
        guard_mpc_failure = bool(
            not comparable
            and 1 in available
            and any(
                code == "MPC_DEADLINE_UNRELIABLE"
                for code in comparability.get("failure_codes", ())
            )
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "pair_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "family": fixture.family,
            "maneuver": artifact.route_context.maneuver.value,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "registry_hash": registry_hash,
            "artifact_content_hash": artifact.artifact_content_hash(),
            "bundle_hash": artifact.bundle.bundle_hash,
            "forward_count_total": int(forward_counter["n"]),
            "candidate1_available": 1 in available,
            "comparable": comparable,
            "decisive": decisive,
            "winner": winner,
            "pair_label": (
                "NO_ALTERNATIVE" if oracle is None else oracle.pair_label
            ),
            "both_bad": both_bad,
            "safe_candidate_exists": any(safe.values()),
            "guard_mpc_failure": guard_mpc_failure,
            "comparability": comparability,
            "oracle": None if oracle is None else oracle.to_dict(),
            "anchor": anchor,
            "branches": {
                str(index): result.summary
                for index, result in results.items()
            },
        }
        (evidence_dir / "pair_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception as exc:
        (evidence_dir / "failure.json").write_text(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if world is not None:
            restore_async(world)


__all__ = [
    "REPORT_SCHEMA",
    "build_v3_model_identity",
    "run_anchor_v3",
    "run_pair_v3",
]
