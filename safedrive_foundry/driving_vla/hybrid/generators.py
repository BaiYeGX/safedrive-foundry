"""Independent Classic Expert and nominal SimLingo generators for H1."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

from classic_stack.geometry import ReferencePath
from classic_stack.planning.frenet import ActorState, FrenetPlanner, PlanRequest
from driving_vla.adapter.policy_adapter import TrajectoryArray, arrays_to_candidate_set
from driving_vla.hybrid.contracts import (
    CandidateProvenance,
    GenerationAttempt,
    HybridCandidate,
    HybridCandidateSet,
    HybridSource,
    ObservableAnchor,
)
from driving_vla.model.canonicalizer import (
    CanonicalizationResult,
    TrajectoryCanonicalizer,
    UpstreamTimedTrajectory,
    stable_sha256,
)
from driving_vla.model.lineage import file_sha256
from driving_vla.model.nominal_policy import NominalVLAPolicy
from safety_kernel.contracts.types import CandidateSource, PolicyCandidate


class CandidateGenerator(Protocol):
    source: HybridSource

    def generate(self, anchor: ObservableAnchor) -> HybridCandidate: ...


def route_revision_sha256(route_xy: tuple[tuple[float, float], ...]) -> str:
    """Stable revision for one observable route polyline."""

    points = tuple((round(float(x), 6), round(float(y), 6)) for x, y in route_xy)
    return stable_sha256({"coordinate_frame": "map", "route_xy": points})


def combined_generator_hash(**parts: str) -> str:
    return stable_sha256(dict(sorted(parts.items())))


def simlingo_generator_hash(policy: NominalVLAPolicy) -> str:
    """Hash the actual checkpoint/config used by a loaded nominal policy."""

    policy.ensure_loaded()
    runtime = policy.runtime
    assert runtime is not None
    ckpt = Path(runtime.ckpt_path)
    hydra = Path(runtime.hydra_config)
    if not ckpt.is_file() or not hydra.is_file():
        raise FileNotFoundError("SimLingo checkpoint/config missing")
    return combined_generator_hash(
        model_id=policy.model_id,
        checkpoint_sha256=file_sha256(ckpt),
        hydra_sha256=file_sha256(hydra),
    )


def _candidate_with_provenance(
    *,
    source: HybridSource,
    safety_source: CandidateSource,
    anchor: ObservableAnchor,
    trajectory: TrajectoryArray,
    canonical: CanonicalizationResult,
    generator_id: str,
    generator_hash: str,
    generation_latency_s: float,
    generated_wall_time_s: float,
    uncertainty: float,
) -> HybridCandidate:
    candidate_id = f"{anchor.observation_id}:{source.value}"
    array = TrajectoryArray(
        points_xy_yaw_v_a_kappa=trajectory.points_xy_yaw_v_a_kappa,
        probability=1.0,
        uncertainty=float(uncertainty),
        candidate_id=candidate_id,
        behavior=trajectory.behavior,
        intended_action=source.value,
    )
    provenance = CandidateProvenance(
        source=source,
        candidate_id=candidate_id,
        observation_id=anchor.observation_id,
        frame_id=anchor.bundle.frame_id,
        carla_frame=anchor.bundle.carla_frame,
        simulation_time_s=anchor.bundle.simulation_time_s,
        route_revision=anchor.route_revision,
        generator_id=generator_id,
        generator_hash=generator_hash,
        raw_sha256=canonical.report.input_sha256,
        canonical_sha256=canonical.report.canonical_sha256,
        canonicalizer_version=canonical.report.version,
        canonicalization_error_m=canonical.report.max_resample_error_m,
        coverage_shortfall_m=canonical.report.coverage_shortfall_m,
        generation_latency_s=float(generation_latency_s),
        generated_wall_time_s=float(generated_wall_time_s),
        freshness_s=float(generation_latency_s),
        coordinate_frame="map",
    )
    cset = arrays_to_candidate_set(
        [array],
        anchor.bundle,
        model_id=generator_id,
        source=safety_source,
        now_s=anchor.bundle.simulation_time_s,
        valid_for_s=0.25,
        coordinate_frame="map",
        dynamics_meta={
            "h1_source": source.value,
            "observation_id": anchor.observation_id,
            "frame_id": anchor.bundle.frame_id,
            "carla_frame": anchor.bundle.carla_frame,
            "simulation_time_s": anchor.bundle.simulation_time_s,
            "route_revision": anchor.route_revision,
            "generator_id": generator_id,
            "generator_hash": generator_hash,
            "raw_sha256": canonical.report.input_sha256,
            "canonical_sha256": canonical.report.canonical_sha256,
            "canonicalizer_version": canonical.report.version,
            "canonicalization_error_m": canonical.report.max_resample_error_m,
            "coverage_shortfall_m": canonical.report.coverage_shortfall_m,
            "generation_latency_s": generation_latency_s,
            "generated_wall_time_s": generated_wall_time_s,
            "coordinate_frame": "map",
        },
    )
    candidate: PolicyCandidate = cset.candidates[0]
    return HybridCandidate(candidate=candidate, provenance=provenance)


class ClassicExpertGenerator:
    """One deterministic Frenet/ST proposal from the observable anchor."""

    source = HybridSource.EXPERT
    generator_id = "classic-frenet-st@h1"

    def __init__(
        self,
        planner: FrenetPlanner | None = None,
        canonicalizer: TrajectoryCanonicalizer | None = None,
    ) -> None:
        self.planner = planner or FrenetPlanner()
        self.canonicalizer = canonicalizer or TrajectoryCanonicalizer()
        self.generator_hash = self.planner.config_hash

    def generate(self, anchor: ObservableAnchor) -> HybridCandidate:
        t0 = time.perf_counter()
        route = anchor.bundle.route_xy
        reference = ReferencePath.from_xy(
            [point[0] for point in route], [point[1] for point in route]
        )
        s0, d0 = reference.project(anchor.bundle.ego_x, anchor.bundle.ego_y)
        if abs(d0) > self.planner.config.road_half_width_m:
            raise RuntimeError(f"expert_ego_outside_route:d={d0:.3f}")
        actors = tuple(
            ActorState(
                actor_id=actor.actor_id,
                x=actor.x,
                y=actor.y,
                yaw=actor.yaw,
                speed_mps=math.hypot(actor.vx, actor.vy),
                length_m=actor.length_m,
                width_m=actor.width_m,
                model="cv",
            )
            for actor in anchor.safety_snapshot.actors
            if not actor.lost
        )
        red_distances = [
            (light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m)
            for light in anchor.safety_snapshot.traffic_lights
            if light.state.lower() == "red"
            and light.controls_ego_lane is not False
            and (light.stop_line_distance_m if light.stop_line_distance_m is not None else light.distance_m) >= 0.0
        ]
        stop_s = min(red_distances) if red_distances else None
        scenario_kind = "stop" if stop_s is not None else "follow"
        speed_limit = anchor.safety_snapshot.speed_limit_mps
        target_speed = min(6.0, float(speed_limit)) if speed_limit is not None else 6.0
        plan = self.planner.plan(
            PlanRequest(
                reference=reference,
                v0=max(0.0, anchor.bundle.ego_v),
                a0=float(anchor.safety_snapshot.ego_a),
                s0=max(0.0, min(s0, reference.length)),
                d0=d0,
                scenario_kind=scenario_kind,
                actors=actors,
                target_speed_mps=max(0.0, target_speed),
                stop_s=None if stop_s is None else max(0.0, stop_s - 1.0),
                seed=int(anchor.bundle.carla_frame),
            )
        )
        if not plan.ok or plan.trajectory is None:
            raise RuntimeError(f"expert_plan_failed:{plan.failure_code}")
        raw = UpstreamTimedTrajectory(
            points=tuple(
                (point.t, point.x, point.y, point.yaw, point.v, point.a, point.kappa)
                for point in plan.trajectory.points
            ),
            frame="map",
        )
        canonical = self.canonicalizer.canonicalize_timed(raw, to_map=False)
        elapsed = time.perf_counter() - t0
        return _candidate_with_provenance(
            source=self.source,
            safety_source=CandidateSource.CLASSIC,
            anchor=anchor,
            trajectory=canonical.trajectory,
            canonical=canonical,
            generator_id=self.generator_id,
            generator_hash=self.generator_hash,
            generation_latency_s=elapsed,
            generated_wall_time_s=time.time(),
            uncertainty=0.0,
        )


class NominalVLAGenerator:
    """One real nominal SimLingo forward; no derived second candidate."""

    source = HybridSource.VLA

    def __init__(self, policy: NominalVLAPolicy, *, generator_hash: str) -> None:
        if not generator_hash:
            raise ValueError("generator_hash is required")
        self.policy = policy
        self.generator_id = policy.model_id
        self.generator_hash = generator_hash

    def generate(self, anchor: ObservableAnchor) -> HybridCandidate:
        t0 = time.perf_counter()
        native = self.policy.predict_native(anchor.bundle)
        canonical = self.policy.canonicalize_native(native)
        elapsed = time.perf_counter() - t0
        return _candidate_with_provenance(
            source=self.source,
            safety_source=CandidateSource.VLA_FAST,
            anchor=anchor,
            trajectory=canonical.trajectory,
            canonical=canonical,
            generator_id=self.generator_id,
            generator_hash=self.generator_hash,
            generation_latency_s=elapsed,
            generated_wall_time_s=time.time(),
            uncertainty=0.15,
        )


def generate_hybrid_set(
    anchor: ObservableAnchor,
    expert: CandidateGenerator,
    vla: CandidateGenerator,
) -> HybridCandidateSet:
    """Run each independent source once against the same immutable anchor."""

    candidates: list[HybridCandidate] = []
    attempts: list[GenerationAttempt] = []
    for expected, generator in ((HybridSource.EXPERT, expert), (HybridSource.VLA, vla)):
        if generator.source is not expected:
            raise ValueError(f"generator_source_mismatch:{generator.source}!={expected}")
        started = time.perf_counter()
        try:
            candidate = generator.generate(anchor)
        except Exception as exc:  # source failure must not fabricate a candidate
            attempts.append(
                GenerationAttempt(
                    source=expected,
                    success=False,
                    generation_latency_s=time.perf_counter() - started,
                    error=f"{type(exc).__name__}:{exc}",
                )
            )
            continue
        candidates.append(candidate)
        attempts.append(
            GenerationAttempt(
                source=expected,
                success=True,
                generation_latency_s=candidate.provenance.generation_latency_s,
                candidate_id=candidate.candidate.candidate_id,
            )
        )
    return HybridCandidateSet(anchor=anchor, candidates=tuple(candidates), attempts=tuple(attempts))


__all__ = [
    "CandidateGenerator",
    "ClassicExpertGenerator",
    "NominalVLAGenerator",
    "combined_generator_hash",
    "generate_hybrid_set",
    "route_revision_sha256",
    "simlingo_generator_hash",
]
