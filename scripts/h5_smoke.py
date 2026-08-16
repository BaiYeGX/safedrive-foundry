#!/usr/bin/env python3
"""Bounded Town03 H5 smoke: Expert/VLA -> Guard -> H5WorldRouter -> Safety -> control.

This is a live-entry smoke for H5, not the full closed-loop experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

import carla  # noqa: E402
import torch  # noqa: E402

from classic_stack.control.config import config_sha256 as control_config_sha256  # noqa: E402
from classic_stack.control.controller import ControlLoop  # noqa: E402
from driving_vla.hybrid import (  # noqa: E402
    ClassicExpertGenerator,
    H1CandidatePipeline,
    NominalVLAGenerator,
    generate_hybrid_set,
    simlingo_generator_hash,
)
from driving_vla.model.nominal_policy import NominalVLAPolicy  # noqa: E402
from runtime import (  # noqa: E402
    ActorSpec,
    RunIdentity,
    RunRegistry,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    load_runtime_profiles,
)
from runtime.carla_connection import ConnectionResolver, READY  # noqa: E402
from safety_kernel.config import config_sha256 as safety_config_sha256  # noqa: E402

from data_pipeline.h4.contracts import FINAL_CHECKPOINTS, H4_CONFIG
from data_pipeline.h4.runtime import NormalizedWorldScorer
from data_pipeline.h5.runtime import H5WorldRouter

from scripts.h1_hybrid_smoke import (  # noqa: E402
    _build_anchor,
    _ego_state,
    _map_basename,
    _require_clean_scene,
    _route_and_spawn,
    _worktree_identity,
)


def _load_scorer() -> NormalizedWorldScorer:
    import json as _json
    from data_pipeline.h3.model import load_model

    evidence = _json.loads(
        (ROOT / "docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json").read_text()
    )
    stats = evidence["normalization_stats"]
    stats_list = [(float(item["mean"]), float(item["std"])) for item in stats["items"]]
    models = []
    for seed, info in FINAL_CHECKPOINTS.items():
        model, _ = load_model(ROOT / info["path"], device="cuda")
        models.append(model)
    return NormalizedWorldScorer(models, stats_list, device="cuda", temperature=float(H4_CONFIG["temperature"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--ticks", type=int, default=1)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, object] = {
        "schema_version": "safedrive.h5.live_smoke.v1",
        "ok": False,
        "map_requested": args.map,
        "worktree": _worktree_identity(),
        "started_wall_time_s": time.time(),
    }
    runtime: ScenarioRuntime | None = None
    registry: RunRegistry | None = None
    run_id: str | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")
        resolver = ConnectionResolver(ROOT, expected_version="0.9.16", timeout_seconds=10.0)
        report = resolver.preflight()
        payload["connection"] = report.to_dict()
        if report.status != READY:
            raise RuntimeError(f"CARLA_NOT_READY:{report.error_code}:{report.error_message}")
        client, _ = resolver.connect(report=report)
        world = client.get_world()
        if _map_basename(str(world.get_map().name)) != _map_basename(args.map):
            raise RuntimeError(f"MAP_MISMATCH:{world.get_map().name}!={args.map}")
        _require_clean_scene(world)
        spawn, route = _route_and_spawn(world.get_map())

        policy = NominalVLAPolicy(keep_on_gpu=True)
        policy.ensure_loaded()
        vla_hash = simlingo_generator_hash(policy)
        classic = ClassicExpertGenerator()
        vla = NominalVLAGenerator(policy, generator_hash=vla_hash)

        run_id = f"h5-town03-{time.time_ns()}"
        identity = RunIdentity(
            experiment_id="h5-world-on-off",
            run_id=run_id,
            scenario_id="h5-town03-smoke",
            attempt_id=0,
            server_epoch=f"carla-{report.server_version}-{report.process_state}",
            producer_version="h5-live-smoke-v1",
        )
        profile = load_runtime_profiles(
            ROOT / "safedrive_foundry/config/runtime_profiles.toml"
        )["throughput_20hz"]
        camera = carla.Transform(
            carla.Location(x=1.4, y=0.0, z=1.8)
        )
        spec = ScenarioSpec(
            scenario_id=identity.scenario_id,
            map_name=_map_basename(str(world.get_map().name)),
            actors=(ActorSpec("ego", "vehicle.tesla.model3", spawn, "ego", 0, False),),
            sensors=(SensorSpec(
                "front_camera", "sensor.camera.rgb", camera, "ego", 0,
                {"image_size_x": "1280", "image_size_y": "720", "fov": "110"},
            ),),
            traffic_manager_seed=20260812,
            sensor_timeout_seconds=10.0,
        )
        registry = RunRegistry(args.evidence_dir / "run_registry.sqlite3")
        runtime = ScenarioRuntime(
            client=client,
            identity=identity,
            profile=profile,
            registry=registry,
            lease_path=ROOT / ".runtime" / f"tick-lease-{run_id}.lock",
            owner="sdf.h5.live_smoke",
        )
        runtime.start(spec)
        if args.ticks < 1:
            raise RuntimeError("TICKS_MUST_BE_POSITIVE")
        header = runtime.tick(carla.VehicleControl(throttle=0.0, brake=1.0))
        scorer = _load_scorer()
        h5_router = H5WorldRouter(scorer, min_hold_ticks=5, hysteresis_margin=0.05)
        pipeline = H1CandidatePipeline(router=h5_router)
        control_loop = ControlLoop()
        from driving_vla.runtime.safety_control_bind import apply_safety_control

        routing_history = []
        last_routing = None
        last_safety = None
        last_applied = None
        execution_frame = None
        for step in range(args.ticks):
            anchor = _build_anchor(runtime, header, route)
            generated = generate_hybrid_set(anchor, classic, vla)
            if len(generated.candidates) != 2 or not all(attempt.success for attempt in generated.attempts):
                raise RuntimeError("BOTH_SOURCES_NOT_GENERATED")
            result = pipeline.decide(generated)
            if result.routing.selected_candidate_id is None:
                raise RuntimeError("NO_SELECTED_CANDIDATE")
            routing_history.append(result.routing.to_dict())
            last_routing = result.routing
            last_safety = result.to_dict().get("safety")

            ego_actor = runtime._actors["ego"]
            ego_state, _ = _ego_state(ego_actor)
            applied = apply_safety_control(
                result.safety.decision,
                result.guarded_set.to_policy_candidate_set(
                    tuple(
                        item.candidate
                        for item in result.guarded_set.candidates
                        if item.candidate.candidate_id == result.routing.selected_candidate_id
                    )
                ),
                control_loop,
                ego_state,
                anchor.simulation_time_s,
            )
            if not applied.is_track_approved:
                raise RuntimeError(f"CONTROL_NOT_TRACK_APPROVED:{applied.applied_mode.value}")
            last_applied = applied
            header = runtime.tick(
                carla.VehicleControl(throttle=applied.throttle, brake=applied.brake, steer=applied.steer)
            )
            execution_frame = int(header.carla_frame)

        runtime.complete()
        runtime = None
        payload["runtime_cleanup"] = registry.record(run_id)
        payload.update({
            "ok": True,
            "run_id": run_id,
            "ticks_requested": args.ticks,
            "routing_history": routing_history,
            "routing": last_routing.to_dict(),
            "router_metrics": h5_router.metrics(),
            "online_history_ticks": len(pipeline._ego_history),
            "safety": last_safety,
            "applied_control": last_applied.to_dict(),
            "execution_frame": execution_frame,
            "vla_forward_count": policy.forward_count,
            "configs": {
                "scenario_sha256": ScenarioRuntime.config_hash(spec, profile),
                "classic_sha256": classic.generator_hash,
                "safety_sha256": safety_config_sha256(pipeline.safety.config.raw_toml),
                "control_sha256": control_config_sha256(control_loop.config.raw_toml),
            },
        })
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}:{exc}"
        if runtime is not None:
            try:
                runtime.abort(type(exc).__name__)
                if registry is not None and run_id is not None:
                    payload["runtime_cleanup"] = registry.record(run_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                payload["cleanup_error"] = f"{type(cleanup_exc).__name__}:{cleanup_exc}"
    finally:
        payload["ended_wall_time_s"] = time.time()
        output = args.evidence_dir / "h5_smoke.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"ok": payload["ok"], "evidence": str(output), "error": payload.get("error")}, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
