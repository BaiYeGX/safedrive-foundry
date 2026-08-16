"""H1 Guard → frozen route/defer → Safety integration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from driving_vla.hybrid.contracts import HybridCandidateSet, HybridSource, RoutingResult
from driving_vla.hybrid.guard import CandidateGuard
from driving_vla.hybrid.router import FrozenH1Router
from safety_kernel.contracts.types import ComponentAvailability
from safety_kernel.kernel import KernelTickResult, SafetyKernel


@dataclass(frozen=True)
class H1SafetyResult:
    guarded_set: HybridCandidateSet
    routing: RoutingResult
    safety_input_ids: tuple[str, ...]
    safety: KernelTickResult

    def to_dict(self) -> dict:
        decision = self.safety.decision
        return {
            "routing": self.routing.to_dict(),
            "safety_input_ids": list(self.safety_input_ids),
            "safety": {
                "decision_id": decision.decision_id,
                "decision_kind": decision.decision_kind.value,
                "final_candidate_id": decision.final_candidate_id,
                "pre_repair_trajectory_id": decision.pre_repair_trajectory_id,
                "post_repair_trajectory_id": decision.post_repair_trajectory_id,
                "executed_trajectory_id": decision.executed_trajectory_id,
                "reject_reasons": list(decision.reject_reasons),
                "fallback": (
                    None
                    if decision.fallback_request is None
                    else decision.fallback_request.to_dict()
                ),
            },
        }


def ego_history_entry(snapshot) -> dict:
    return {
        "ego_x": float(snapshot.ego_x),
        "ego_y": float(snapshot.ego_y),
        "ego_yaw": float(snapshot.ego_yaw),
        "ego_speed_mps": float(snapshot.ego_v),
        "ego_acceleration_mps2": float(snapshot.ego_a),
        "simulation_time_s": float(snapshot.simulation_time_s),
    }


class H1CandidatePipeline:
    def __init__(
        self,
        *,
        guard: CandidateGuard | None = None,
        router: FrozenH1Router | None = None,
        safety: SafetyKernel | None = None,
    ) -> None:
        self.safety = safety or SafetyKernel()
        self.guard = guard or CandidateGuard(self.safety.config)
        self.router = router or FrozenH1Router(self.safety.config)
        # Online World features must use a real sliding ego-history window.
        # Offline H3 context expects up to 20 ticks; the first online tick is
        # naturally shorter, but never an all-zero history.
        self._ego_history = deque(maxlen=20)

    def seed_ego_history(self, entries) -> None:
        """Preload observable ego history before the first online decision."""
        for entry in entries:
            self._ego_history.append(dict(entry))

    def decide(self, candidate_set: HybridCandidateSet) -> H1SafetyResult:
        guarded = self.guard.evaluate(candidate_set)
        features = None
        if getattr(self.router, "requires_features", False):
            from data_pipeline.h3.live_features import build_live_features
            snapshot = guarded.anchor.safety_snapshot
            self._ego_history.append(ego_history_entry(snapshot))
            features = build_live_features(
                guarded.anchor,
                list(self._ego_history),
                guarded.candidates,
            )
        if getattr(self.router, "requires_features", False):
            routing = self.router.route(guarded, features=features)
        else:
            routing = self.router.route(guarded)
        safety_input = self.router.safety_input(guarded, routing)
        successful_sources = {
            attempt.source for attempt in guarded.attempts if attempt.success
        }
        world_enabled = bool(getattr(self.router, "requires_features", False))
        availability = ComponentAvailability(
            classic=HybridSource.EXPERT in successful_sources,
            vla=HybridSource.VLA in successful_sources,
            world=world_enabled,
            safety=True,
            detail={"world": ("H5_WORLD_ROUTER" if world_enabled else "H1_NOT_IMPLEMENTED"), "selector": getattr(self.router, "selector_name", "")},
        )
        result = self.safety.tick(
            guarded.anchor.safety_snapshot,
            safety_input,
            now_s=guarded.anchor.simulation_time_s,
            availability=availability,
        )
        return H1SafetyResult(
            guarded_set=guarded,
            routing=routing,
            safety_input_ids=tuple(candidate.candidate_id for candidate in safety_input.candidates),
            safety=result,
        )


__all__ = ["H1CandidatePipeline", "H1SafetyResult"]
