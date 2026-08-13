"""H1 Guard → frozen route/defer → Safety integration."""

from __future__ import annotations

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

    def decide(self, candidate_set: HybridCandidateSet) -> H1SafetyResult:
        guarded = self.guard.evaluate(candidate_set)
        routing = self.router.route(guarded)
        safety_input = self.router.safety_input(guarded, routing)
        successful_sources = {
            attempt.source for attempt in guarded.attempts if attempt.success
        }
        availability = ComponentAvailability(
            classic=HybridSource.EXPERT in successful_sources,
            vla=HybridSource.VLA in successful_sources,
            world=False,
            safety=True,
            detail={"world": "H1_NOT_IMPLEMENTED"},
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
