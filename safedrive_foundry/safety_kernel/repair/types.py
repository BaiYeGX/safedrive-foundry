"""Longitudinal repair contracts: unified Raw/Rule/HardReject/Longitudinal."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from safety_kernel.contracts.types import ConstraintMargin, PolicyCandidate


class RepairMode(str, Enum):
    """Unified repair interface modes (CLAIMS C3 baselines + G2-03 RATO)."""

    RAW = "raw"
    RULE = "rule"
    HARD_REJECT = "hard_reject"
    LONGITUDINAL = "longitudinal"
    RATO = "rato"


class SolverStatus(str, Enum):
    SOLVED = "solved"
    SOLVED_INACCURATE = "solved_inaccurate"
    MAX_ITER = "max_iter"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    NUMERICAL_ERROR = "numerical_error"
    STALE_INPUT = "stale_input"
    DISABLED = "disabled"
    SKIPPED = "skipped"
    NOT_REPAIRABLE = "not_repairable"
    REVALIDATE_FAIL = "revalidate_fail"
    N_A = "n/a"


@dataclass(frozen=True)
class SolverTrace:
    status: SolverStatus
    iterations: int
    primal_residual: float
    dual_residual: float
    objective: float
    latency_ms: float
    warm_started: bool
    backend: str
    message: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "iterations": self.iterations,
            "primal_residual": self.primal_residual,
            "dual_residual": self.dual_residual,
            "objective": self.objective,
            "latency_ms": self.latency_ms,
            "warm_started": self.warm_started,
            "backend": self.backend,
            "message": self.message,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class RepairMetrics:
    """Report fields required by G2-02 completion criteria."""

    safety_margin_m: float
    modification_norm: float
    unjustified_stop: bool
    progress_ratio: float
    progress_loss: float
    comfort_jerk_rms: float
    comfort_accel_rms: float
    slack_used_max: float
    terminal_speed_mps: float
    terminal_s_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "safety_margin_m": self.safety_margin_m,
            "modification_norm": self.modification_norm,
            "unjustified_stop": self.unjustified_stop,
            "progress_ratio": self.progress_ratio,
            "progress_loss": self.progress_loss,
            "comfort_jerk_rms": self.comfort_jerk_rms,
            "comfort_accel_rms": self.comfort_accel_rms,
            "slack_used_max": self.slack_used_max,
            "terminal_speed_mps": self.terminal_speed_mps,
            "terminal_s_m": self.terminal_s_m,
        }


@dataclass(frozen=True)
class RepairResult:
    mode: RepairMode
    success: bool
    candidate: PolicyCandidate | None
    pre_repair_id: str | None
    post_repair_id: str | None
    solver_trace: SolverTrace
    metrics: RepairMetrics
    constraint_margins: tuple[ConstraintMargin, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "success": self.success,
            "pre_repair_id": self.pre_repair_id,
            "post_repair_id": self.post_repair_id,
            "solver_trace": self.solver_trace.to_dict(),
            "metrics": self.metrics.to_dict(),
            "reason": self.reason,
            "margins": [
                {"name": m.name, "margin": m.margin, "hard": m.hard, "message": m.message}
                for m in self.constraint_margins
            ],
        }
