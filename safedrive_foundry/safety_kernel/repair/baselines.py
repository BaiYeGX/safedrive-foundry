"""Raw / Rule / HardReject repair baselines for fair comparison with Longitudinal QP."""

from __future__ import annotations

import math
import time
from typing import Sequence

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import (
    ObservableSnapshot,
    PolicyCandidate,
    TrajectoryPoint,
)
from safety_kernel.repair.longitudinal_qp import _arc_lengths, _lead_s_profile, _stop_s
from safety_kernel.repair.types import (
    RepairMetrics,
    RepairMode,
    RepairResult,
    SolverStatus,
    SolverTrace,
)


def _empty_trace(status: SolverStatus, *, latency_ms: float = 0.0, message: str = "") -> SolverTrace:
    return SolverTrace(
        status=status,
        iterations=0,
        primal_residual=0.0,
        dual_residual=0.0,
        objective=0.0,
        latency_ms=latency_ms,
        warm_started=False,
        backend="baseline",
        message=message,
    )


def _metrics_from_candidate(
    original: PolicyCandidate,
    result: PolicyCandidate | None,
    *,
    obs: ObservableSnapshot,
    cfg: SafetyKernelConfig,
) -> RepairMetrics:
    if result is None or not result.points:
        return RepairMetrics(
            safety_margin_m=-1.0,
            modification_norm=0.0,
            unjustified_stop=False,
            progress_ratio=0.0,
            progress_loss=1.0,
            comfort_jerk_rms=0.0,
            comfort_accel_rms=0.0,
            slack_used_max=0.0,
            terminal_speed_mps=0.0,
            terminal_s_m=0.0,
        )
    import numpy as np

    v0 = np.array([p.v for p in original.points], dtype=float)
    v1 = np.array([p.v for p in result.points], dtype=float)
    n = min(len(v0), len(v1))
    mod = float(np.linalg.norm(v1[:n] - v0[:n]) / math.sqrt(max(n, 1)))
    s0 = _arc_lengths(original.points)
    s1 = _arc_lengths(result.points)
    progress_ratio = float(s1[-1] / max(s0[-1], 1e-3))
    jerk_rms = float(np.sqrt(np.mean(np.array([p.jerk for p in result.points]) ** 2)))
    accel_rms = float(np.sqrt(np.mean(np.array([p.a for p in result.points]) ** 2)))
    stop = _stop_s(obs, cfg, cfg.qp.stop_line_buffer_m)
    times = [p.t for p in result.points]
    s_lead = _lead_s_profile(
        obs,
        s1,
        result.points,
        __import__("numpy").array(times, dtype=float),
        cfg,
        min_gap_m=cfg.qp.min_gap_m,
        time_headway_s=cfg.qp.time_headway_s,
    )
    lead_margin = float(min(s_lead[i] - s1[i] for i in range(len(s1))))
    stop_margin = float(stop - s1[-1]) if stop is not None else 1.0
    mean_v = float(np.mean(v1))
    constrained = stop is not None or float(min(s_lead)) < s0[-1] - 1.0
    unjustified = mean_v < 0.3 and progress_ratio < cfg.qp.min_progress_ratio and not constrained
    return RepairMetrics(
        safety_margin_m=min(lead_margin, stop_margin),
        modification_norm=mod,
        unjustified_stop=unjustified,
        progress_ratio=progress_ratio,
        progress_loss=max(0.0, 1.0 - progress_ratio),
        comfort_jerk_rms=jerk_rms,
        comfort_accel_rms=accel_rms,
        slack_used_max=0.0,
        terminal_speed_mps=float(result.points[-1].v),
        terminal_s_m=float(s1[-1]),
    )


class RawPassThrough:
    """RAW baseline: return original trajectory unchanged."""

    mode = RepairMode.RAW

    def __init__(self, config: SafetyKernelConfig) -> None:
        self.config = config

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult:
        del now_s, reject_hints
        t0 = time.perf_counter()
        metrics = _metrics_from_candidate(candidate, candidate, obs=obs, cfg=self.config)
        return RepairResult(
            mode=RepairMode.RAW,
            success=True,
            candidate=candidate,
            pre_repair_id=candidate.candidate_id,
            post_repair_id=candidate.candidate_id,
            solver_trace=_empty_trace(SolverStatus.N_A, latency_ms=(time.perf_counter() - t0) * 1000.0),
            metrics=metrics,
            reason="raw_passthrough",
        )


class HardRejectBaseline:
    """HARD_REJECT baseline: never modifies; always reports failure."""

    mode = RepairMode.HARD_REJECT

    def __init__(self, config: SafetyKernelConfig) -> None:
        self.config = config

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult:
        del now_s
        t0 = time.perf_counter()
        metrics = _metrics_from_candidate(candidate, None, obs=obs, cfg=self.config)
        return RepairResult(
            mode=RepairMode.HARD_REJECT,
            success=False,
            candidate=None,
            pre_repair_id=candidate.candidate_id,
            post_repair_id=None,
            solver_trace=_empty_trace(
                SolverStatus.INFEASIBLE,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                message="hard_reject",
            ),
            metrics=metrics,
            reason=";".join(reject_hints) if reject_hints else "hard_reject",
        )


class RuleSlowdownBaseline:
    """RULE baseline: constant-deceleration slowdown for stop / lead / speed cap."""

    mode = RepairMode.RULE

    def __init__(self, config: SafetyKernelConfig) -> None:
        self.config = config

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult:
        del reject_hints
        t0 = time.perf_counter()
        cfg = self.config
        pts = candidate.points
        if len(pts) < 2:
            return RepairResult(
                mode=RepairMode.RULE,
                success=False,
                candidate=None,
                pre_repair_id=candidate.candidate_id,
                post_repair_id=None,
                solver_trace=_empty_trace(SolverStatus.NOT_REPAIRABLE, message="too_short"),
                metrics=_metrics_from_candidate(candidate, None, obs=obs, cfg=cfg),
                reason="too_short",
            )

        s = _arc_lengths(pts)
        stop = _stop_s(obs, cfg, cfg.qp.stop_line_buffer_m)
        times = [p.t for p in pts]
        import numpy as np

        s_lead = _lead_s_profile(
            obs,
            s,
            pts,
            np.array(times, dtype=float),
            cfg,
            min_gap_m=cfg.qp.min_gap_m,
            time_headway_s=cfg.qp.time_headway_s,
        )
        s_cap = float(min(s_lead))
        if stop is not None:
            s_cap = min(s_cap, stop)

        v0 = max(0.0, float(obs.ego_v if math.isfinite(obs.ego_v) else pts[0].v))
        # Plan constant decel to stop by s_cap if needed.
        if s_cap <= 0.05:
            a_cmd = -cfg.max_decel_mps2
        else:
            # v^2 = v0^2 + 2 a s → a = -v0^2 / (2 s)
            a_cmd = -min(cfg.max_decel_mps2, max(0.1, (v0 * v0) / (2.0 * max(s_cap, 0.1))))
            a_cmd = -abs(a_cmd)

        v_max = cfg.max_speed_mps
        if cfg.enforce_speed_limit and obs.speed_limit_mps is not None:
            v_max = min(v_max, obs.speed_limit_mps + cfg.speed_limit_margin_mps)

        new_pts: list[TrajectoryPoint] = []
        v = v0
        a_prev = float(obs.ego_a)
        for i, p in enumerate(pts):
            if i == 0:
                dt = 0.0
            else:
                dt = max(1e-3, p.t - pts[i - 1].t)
                v = max(0.0, min(v_max, v + a_cmd * dt))
            # If past cap along original path, force stop.
            if s[i] >= s_cap:
                v = 0.0
            a = a_cmd if v > 0.05 else 0.0
            jerk = (a - a_prev) / dt if dt > 1e-6 else 0.0
            a_prev = a
            # Keep geometric path; only change v,a,jerk (rule does not re-time positions).
            new_pts.append(
                TrajectoryPoint(
                    t=p.t,
                    x=p.x,
                    y=p.y,
                    yaw=p.yaw,
                    kappa=p.kappa,
                    v=v,
                    a=a,
                    jerk=jerk,
                )
            )

        post_id = f"{candidate.candidate_id}__rule"
        repaired = PolicyCandidate(
            candidate_id=post_id,
            source=candidate.source,
            generated_time_s=candidate.generated_time_s,
            valid_until_s=candidate.valid_until_s,
            probability=candidate.probability,
            points=tuple(new_pts),
            behavior="rule_slowdown",
            critical_actor=candidate.critical_actor,
            conflict_type=candidate.conflict_type,
            risk_horizon_s=candidate.risk_horizon_s,
            intended_action="rule_slowdown",
            uncertainty=candidate.uncertainty,
            availability=True,
            dynamics_meta={**dict(candidate.dynamics_meta), "repair": "rule_slowdown"},
        )
        metrics = _metrics_from_candidate(candidate, repaired, obs=obs, cfg=cfg)
        return RepairResult(
            mode=RepairMode.RULE,
            success=True,
            candidate=repaired,
            pre_repair_id=candidate.candidate_id,
            post_repair_id=post_id,
            solver_trace=_empty_trace(
                SolverStatus.SOLVED,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                message="rule_const_decel",
            ),
            metrics=metrics,
            reason="ok",
        )
