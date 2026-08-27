"""Ordered per-candidate H1 Guard using observable inputs only."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Callable, Sequence

from classic_stack.control.config import ControlConfig, load_control_config
from classic_stack.control.controller import ControlLoop, EgoState
from classic_stack.geometry import ReferencePath
from driving_vla.hybrid.contracts import (
    GuardCheck,
    GuardResult,
    GuardVerdict,
    HybridCandidate,
    HybridCandidateSet,
    HybridSource,
)
from driving_vla.model.canonicalizer import stable_sha256
from driving_vla.schema.trajectory_contract import DT_S, HORIZON_S, T_STEPS
from driving_vla.runtime.safety_control_bind import safety_points_to_ctrl
from safety_kernel.config import SafetyKernelConfig, load_safety_config
from safety_kernel.contracts.schema import SCHEMA_VERSION
from safety_kernel.contracts.types import CandidateSource, ConstraintMargin
from safety_kernel.validator.checks import (
    check_collision,
    check_dynamics,
    check_freshness,
    check_numeric,
    check_road,
    check_rules,
    check_schema_fields,
    check_set_identity_and_contract,
    check_time_order,
    check_trackability,
    hard_violations,
)


GENERATION_DEADLINE_S = 2.5
IMMINENT_COLLISION_S = 0.75
OBVIOUS_COLLISION_PENETRATION_M = 0.50
GROSS_DYNAMICS_MARGIN = -5.0
GROSS_START_GAP_M = 2.0
GROSS_TRACKABILITY_GAP_M = 2.0


def _margin(name: str, ok: bool, message: str, value: float | None = None) -> ConstraintMargin:
    margin = (1.0 if ok else -1.0) if value is None else float(value)
    return ConstraintMargin(name=name, margin=margin, hard=True, message=message)


def _checks(stage: str, margins: Sequence[ConstraintMargin]) -> list[GuardCheck]:
    return [
        GuardCheck(
            stage=stage,
            name=margin.name,
            passed=not (margin.hard and margin.margin < 0.0),
            message=margin.message,
            margin=margin.margin,
        )
        for margin in margins
    ]


class CandidateGuard:
    """Three-state, observable-only candidate triage.

    PASS is clean, REVIEW is allowed to reach World and must be checked by
    final Safety, and REJECT is reserved for malformed/binding failures or an
    obvious, imminent, non-repairable problem.
    """

    def __init__(
        self,
        safety_config: SafetyKernelConfig | None = None,
        control_config: ControlConfig | None = None,
        *,
        generation_deadline_s: float = GENERATION_DEADLINE_S,
        control_factory: Callable[[], ControlLoop] | None = None,
    ) -> None:
        self.safety_config = safety_config or load_safety_config()
        self.control_config = control_config or load_control_config()
        self.generation_deadline_s = float(generation_deadline_s)
        self.control_factory = control_factory or (lambda: ControlLoop(self.control_config))

    def evaluate(self, candidate_set: HybridCandidateSet) -> HybridCandidateSet:
        guarded = tuple(
            item.with_guard(self.evaluate_candidate(candidate_set, item))
            for item in candidate_set.candidates
        )
        return replace(candidate_set, candidates=guarded)

    def evaluate_candidate(
        self, candidate_set: HybridCandidateSet, item: HybridCandidate
    ) -> GuardResult:
        started = time.perf_counter()
        all_checks: list[GuardCheck] = []
        all_margins: list[ConstraintMargin] = []
        controller_mode: str | None = None

        stages: tuple[tuple[str, Callable[[], list[ConstraintMargin]]], ...] = (
            ("contract", lambda: self._contract_checks(candidate_set, item)),
            ("binding_freshness", lambda: self._binding_checks(candidate_set, item)),
            ("route_navigation", lambda: self._route_checks(candidate_set, item)),
            ("dynamics_trackability", lambda: self._dynamics_checks(item)),
            ("observable_collision", lambda: self._collision_checks(candidate_set, item)),
        )
        rejects: list[str] = []
        reviews: list[str] = []
        for stage, function in stages:
            try:
                margins = function()
            except Exception as exc:  # malformed inputs fail at their current stage
                margins = [_margin(stage, False, f"exception:{type(exc).__name__}:{exc}")]
            all_margins.extend(margins)
            all_checks.extend(_checks(stage, margins))
            violations = hard_violations(margins)
            blocking = []
            for margin in violations:
                reason = f"{stage}:{margin.name}:{margin.message}"
                if self._is_hard_reject(stage, margin):
                    rejects.append(reason)
                    blocking.append(margin)
                else:
                    reviews.append(reason)
            if blocking:
                break
        else:
            controller_margins, controller_mode = self._controller_checks(candidate_set, item)
            all_margins.extend(controller_margins)
            all_checks.extend(_checks("controller_feasibility", controller_margins))
            violations = hard_violations(controller_margins)
            for margin in violations:
                reason = f"controller_feasibility:{margin.name}:{margin.message}"
                if self._is_hard_reject("controller_feasibility", margin):
                    rejects.append(reason)
                else:
                    reviews.append(reason)

        verdict = (
            GuardVerdict.REJECT
            if rejects
            else GuardVerdict.REVIEW
            if reviews
            else GuardVerdict.PASS
        )
        return GuardResult(
            candidate_id=item.candidate.candidate_id,
            verdict=verdict,
            checks=tuple(all_checks),
            reject_reasons=tuple(rejects),
            review_reasons=tuple(reviews),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            margins=tuple(all_margins),
            controller_mode=controller_mode,
        )

    @staticmethod
    def _is_hard_reject(stage: str, margin: ConstraintMargin) -> bool:
        """Keep only indisputable failures ahead of World.

        Thresholds are deliberately expressed in physical terms.  Borderline
        collision, road, rule, dynamics and controller findings are REVIEW so
        the richer World ranker and the final Safety repair chain can decide.
        """
        if stage in {"contract", "binding_freshness"}:
            return True
        if stage == "route_navigation":
            if margin.name == "road":
                # A finite off-corridor proposal is not itself an executable
                # trajectory, but it is still a useful World candidate: the
                # bounded RATO path can project it back into the observable
                # legal corridor and the final validator must then pass it.
                # Rejecting here prevented World and Safety from seeing the
                # very cases the repair path is designed to handle.
                return False
            if margin.name == "navigation_start":
                return margin.margin <= -GROSS_START_GAP_M
            return False
        if stage == "dynamics_trackability":
            if margin.name == "trackability":
                if margin.message in {"zero_dt", "too_short"}:
                    return True
                if margin.message == "teleport":
                    # The validator's label also covers small point-spacing /
                    # speed inconsistencies.  Those are finite and RATO can
                    # rebuild their geometry, so only an unmistakably large
                    # spatial jump is a Guard-level hard reject.
                    return margin.margin <= -GROSS_TRACKABILITY_GAP_M
                return False
            # Finite speed/accel/jerk/curvature findings can be reduced by the
            # bounded Safety repair chain.  Keep them visible to World as
            # REVIEW; final Safety still requires a full successful recheck.
            return False
        if stage == "observable_collision":
            if margin.message in {
                "non_finite_actor",
                "illegal_actor_size",
                "non_finite_distance",
                "non_finite_margin",
            }:
                return True
            first_t = margin.first_violation_time_s
            return bool(
                first_t is not None
                and first_t <= IMMINENT_COLLISION_S
                and margin.margin <= -OBVIOUS_COLLISION_PENETRATION_M
            )
        if stage == "controller_feasibility":
            return margin.name in {
                "minimum_execution_horizon",
                "controller_finite",
                "controller_bounds",
            }
        return True

    def _contract_checks(
        self, candidate_set: HybridCandidateSet, item: HybridCandidate
    ) -> list[ConstraintMargin]:
        candidate = item.candidate
        p = item.provenance
        safety_set = candidate_set.to_policy_candidate_set((candidate,))
        margins = check_set_identity_and_contract(
            safety_set,
            candidate_set.anchor.safety_snapshot,
            expected_schema_version=SCHEMA_VERSION,
        )
        expected_source = (
            CandidateSource.CLASSIC
            if p.source is HybridSource.EXPERT
            else CandidateSource.VLA_FAST
        )
        margins.extend(
            [
                _margin("h1_source", candidate.source is expected_source, "source_mapping"),
                _margin("h1_coordinate", p.coordinate_frame == "map", "map_required"),
                _margin("h1_t_steps", len(candidate.points) == T_STEPS, f"T={len(candidate.points)}"),
            ]
        )
        exact_times = len(candidate.points) == T_STEPS and all(
            abs(point.t - (i + 1) * DT_S) <= 1e-9
            for i, point in enumerate(candidate.points)
        )
        margins.append(_margin("h1_time_grid", exact_times, "t=(i+1)*0.25"))
        last_time = candidate.points[-1].t if candidate.points else 0.0
        margins.append(
            _margin(
                "h1_absolute_horizon",
                abs(last_time - HORIZON_S) <= 1e-9,
                f"last_t={last_time:.6f}",
            )
        )
        canonical_payload = tuple(
            (point.x, point.y, point.yaw, point.v, point.a, point.kappa)
            for point in candidate.points
        )
        margins.append(
            _margin(
                "h1_canonical_hash",
                stable_sha256(canonical_payload) == p.canonical_sha256,
                "canonical_hash_binding",
            )
        )
        margins.extend(
            [
                check_schema_fields(candidate, self.safety_config),
                check_numeric(candidate),
                check_time_order(candidate),
            ]
        )
        return margins

    def _binding_checks(
        self, candidate_set: HybridCandidateSet, item: HybridCandidate
    ) -> list[ConstraintMargin]:
        anchor, p, candidate = candidate_set.anchor, item.provenance, item.candidate
        bound = (
            p.observation_id == anchor.observation_id
            and p.frame_id == anchor.bundle.frame_id
            and p.carla_frame == anchor.bundle.carla_frame
            and abs(p.simulation_time_s - anchor.bundle.simulation_time_s) <= 1e-9
            and p.route_revision == anchor.route_revision
            and candidate.candidate_id == p.candidate_id
        )
        hashes_present = all(
            bool(value)
            for value in (
                p.generator_id,
                p.generator_hash,
                p.raw_sha256,
                p.canonical_sha256,
                p.canonicalizer_version,
            )
        )
        deadline_margin = self.generation_deadline_s - p.generation_latency_s
        return [
            _margin("observation_binding", bound, "anchor_binding"),
            _margin("provenance", hashes_present, "required_hashes"),
            _margin(
                "generation_deadline",
                deadline_margin >= 0.0,
                f"latency_s={p.generation_latency_s:.6f}",
                deadline_margin,
            ),
            check_freshness(
                candidate,
                now_s=anchor.bundle.simulation_time_s,
                cfg=self.safety_config,
            ),
        ]

    def _route_checks(
        self, candidate_set: HybridCandidateSet, item: HybridCandidate
    ) -> list[ConstraintMargin]:
        anchor, candidate = candidate_set.anchor, item.candidate
        margins = [
            check_road(candidate, anchor.safety_snapshot, self.safety_config),
            check_rules(candidate, anchor.safety_snapshot, self.safety_config),
        ]
        reference = ReferencePath.from_xy(
            [point[0] for point in anchor.bundle.route_xy],
            [point[1] for point in anchor.bundle.route_xy],
        )
        progress = [reference.project(point.x, point.y)[0] for point in candidate.points]
        route_forward = bool(progress) and progress[-1] + 0.5 >= progress[0]
        margins.append(_margin("navigation_progress", route_forward, "terminal_not_behind_start"))
        if candidate.points:
            first = candidate.points[0]
            distance = math.hypot(first.x - anchor.bundle.ego_x, first.y - anchor.bundle.ego_y)
            reach = (
                abs(anchor.bundle.ego_v) * DT_S
                + 0.5 * self.safety_config.max_accel_mps2 * DT_S * DT_S
                + 1.0
            )
            margins.append(
                _margin("navigation_start", distance <= reach, f"first_distance_m={distance:.6f}", reach - distance)
            )
        return margins

    def _dynamics_checks(self, item: HybridCandidate) -> list[ConstraintMargin]:
        return [
            check_dynamics(item.candidate, self.safety_config),
            check_trackability(item.candidate, self.safety_config),
        ]

    def _collision_checks(
        self, candidate_set: HybridCandidateSet, item: HybridCandidate
    ) -> list[ConstraintMargin]:
        return [
            check_collision(item.candidate, candidate_set.anchor.safety_snapshot, self.safety_config)
        ]

    def _controller_checks(
        self, candidate_set: HybridCandidateSet, item: HybridCandidate
    ) -> tuple[list[ConstraintMargin], str | None]:
        candidate, anchor = item.candidate, candidate_set.anchor
        span = candidate.points[-1].t - candidate.points[0].t if candidate.points else 0.0
        horizon = _margin(
            "minimum_execution_horizon",
            span + 1e-9 >= self.safety_config.min_horizon_s,
            f"span_s={span:.6f}",
            span - self.safety_config.min_horizon_s,
        )
        if horizon.margin < 0.0:
            return [horizon], None
        control = self.control_factory()
        control.set_trajectory(
            safety_points_to_ctrl(candidate.points), anchor.bundle.simulation_time_s
        )
        command = control.step(
            EgoState(
                x=anchor.bundle.ego_x,
                y=anchor.bundle.ego_y,
                yaw=anchor.bundle.ego_yaw,
                v=anchor.bundle.ego_v,
            ),
            anchor.bundle.simulation_time_s,
        )
        values = (command.steer, command.throttle, command.brake)
        finite = all(math.isfinite(float(value)) for value in values)
        bounded = (
            -1.0 <= float(command.steer) <= 1.0
            and 0.0 <= float(command.throttle) <= 1.0
            and 0.0 <= float(command.brake) <= 1.0
        )
        mode = str(command.mode)
        executable = mode != "brake"
        return (
            [
                horizon,
                _margin("controller_finite", finite, "finite_control"),
                _margin("controller_bounds", bounded, "bounded_control"),
                _margin("controller_mode", executable, f"mode={mode}"),
            ],
            mode,
        )


__all__ = [
    "CandidateGuard",
    "GENERATION_DEADLINE_S",
    "GROSS_DYNAMICS_MARGIN",
    "GROSS_TRACKABILITY_GAP_M",
    "IMMINENT_COLLISION_S",
    "OBVIOUS_COLLISION_PENETRATION_M",
]
