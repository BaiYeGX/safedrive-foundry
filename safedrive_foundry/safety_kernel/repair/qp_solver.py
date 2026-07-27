"""OSQP-form dense QP: minimize 1/2 x' P x + q' x  s.t.  l <= A x <= u.

Backends (first success wins):
1. python-osqp when importable (including tools/wsl_site_packages)
2. scipy SLSQP on the same OSQP-form problem
3. pure-numpy ADMM (last resort)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from safety_kernel.repair.types import SolverStatus, SolverTrace

# Optional local install: tools/ is gitignored; append so system numpy/scipy stay first.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_SITE = _REPO_ROOT / "tools" / "wsl_site_packages"
if _LOCAL_SITE.is_dir():
    _site = str(_LOCAL_SITE)
    if _site not in sys.path:
        sys.path.append(_site)

try:
    import osqp as _osqp  # type: ignore

    _HAS_OSQP = True
except Exception:  # pragma: no cover - optional dependency
    _osqp = None
    _HAS_OSQP = False


@dataclass
class QPProblem:
    P: np.ndarray  # (n, n) PSD  — cost uses 1/2 x' P x + q' x
    q: np.ndarray  # (n,)
    A: np.ndarray  # (m, n)
    l: np.ndarray  # (m,)
    u: np.ndarray  # (m,)


@dataclass
class WarmStart:
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None


class LongitudinalQPSolver:
    """Small dense QP solver with optional warm start and deadline."""

    def __init__(
        self,
        *,
        max_iter: int = 4000,
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-4,
        rho: float = 1.0,
        alpha: float = 1.6,
        deadline_ms: float = 50.0,
        prefer_osqp: bool = True,
        polish: bool = True,
    ) -> None:
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.rho = rho
        self.alpha = alpha
        self.deadline_ms = deadline_ms
        self.prefer_osqp = prefer_osqp
        self.polish = bool(polish)
        self._warm = WarmStart()

    def clear_warm_start(self) -> None:
        self._warm = WarmStart()

    def set_warm_start(self, x: np.ndarray, y: np.ndarray | None = None, z: np.ndarray | None = None) -> None:
        self._warm = WarmStart(
            x=np.asarray(x, dtype=float).copy(),
            y=None if y is None else np.asarray(y, dtype=float).copy(),
            z=None if z is None else np.asarray(z, dtype=float).copy(),
        )

    def _elapsed_ms(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000.0

    def _timeout_trace(
        self,
        *,
        t0: float,
        backend: str,
        message: str = "deadline_exceeded",
        warm_started: bool = False,
        iterations: int = 0,
        pri: float = float("inf"),
        dua: float = float("inf"),
        obj: float = float("inf"),
    ) -> SolverTrace:
        return SolverTrace(
            status=SolverStatus.TIMEOUT,
            iterations=iterations,
            primal_residual=pri,
            dual_residual=dua,
            objective=obj,
            latency_ms=self._elapsed_ms(t0),
            warm_started=warm_started,
            backend=backend,
            message=message,
        )

    def _enforce_deadline(
        self,
        x: np.ndarray | None,
        trace: SolverTrace,
        *,
        t0: float,
    ) -> tuple[np.ndarray | None, SolverTrace]:
        """Any backend over qp.deadline_ms is TIMEOUT; never return a late solution."""
        latency = self._elapsed_ms(t0)
        if latency > self.deadline_ms or trace.status is SolverStatus.TIMEOUT:
            return None, self._timeout_trace(
                t0=t0,
                backend=trace.backend,
                message="deadline_exceeded" if latency > self.deadline_ms else (trace.message or "timeout"),
                warm_started=trace.warm_started,
                iterations=trace.iterations,
                pri=trace.primal_residual,
                dua=trace.dual_residual,
                obj=trace.objective,
            )
        # Refresh latency on successful traces for accurate evidence.
        if x is not None and abs(trace.latency_ms - latency) > 1e-6:
            return x, SolverTrace(
                status=trace.status,
                iterations=trace.iterations,
                primal_residual=trace.primal_residual,
                dual_residual=trace.dual_residual,
                objective=trace.objective,
                latency_ms=latency,
                warm_started=trace.warm_started,
                backend=trace.backend,
                message=trace.message,
                extras=dict(trace.extras) if trace.extras else {},
            )
        return x, trace

    def solve(self, problem: QPProblem, *, warm_start: bool = True) -> tuple[np.ndarray | None, SolverTrace]:
        t0 = time.perf_counter()
        if not (
            np.all(np.isfinite(problem.P))
            and np.all(np.isfinite(problem.q))
            and np.all(np.isfinite(problem.A))
            and np.all(np.isfinite(problem.l))
            and np.all(np.isfinite(problem.u))
        ):
            return None, self._trace(
                SolverStatus.NUMERICAL_ERROR,
                t0=t0,
                message="non_finite_problem_data",
                backend="none",
            )

        # Zero / negative deadline: deterministic immediate TIMEOUT (no backend work).
        if self.deadline_ms <= 0.0:
            return None, self._timeout_trace(t0=t0, backend="none", message="deadline_non_positive")

        last_x: np.ndarray | None = None
        last_trace: SolverTrace | None = None

        if self.prefer_osqp and _HAS_OSQP:
            last_x, last_trace = self._solve_osqp(problem, warm_start=warm_start, t0=t0)
            last_x, last_trace = self._enforce_deadline(last_x, last_trace, t0=t0)
            if last_trace.status is SolverStatus.TIMEOUT:
                return None, last_trace
            # A convex QP reported primal/dual infeasible by OSQP does not
            # become feasible when handed to SLSQP.  Falling through only
            # burns the real-time budget and mislabels the condition as a
            # timeout.  Preserve the actual failure class so the caller can
            # apply its bounded fallback immediately.
            if last_trace.status is SolverStatus.INFEASIBLE:
                return None, last_trace
            if last_x is not None:
                return last_x, last_trace

        if self._elapsed_ms(t0) > self.deadline_ms:
            return None, self._timeout_trace(
                t0=t0,
                backend=last_trace.backend if last_trace else "none",
                message="deadline_before_scipy",
            )

        last_x, last_trace = self._solve_scipy(problem, warm_start=warm_start, t0=t0)
        last_x, last_trace = self._enforce_deadline(last_x, last_trace, t0=t0)
        if last_trace.status is SolverStatus.TIMEOUT:
            return None, last_trace
        if last_x is not None:
            return last_x, last_trace

        if self._elapsed_ms(t0) > self.deadline_ms * 0.6:
            # Not enough budget for ADMM fallback.
            return last_x, last_trace

        last_x, last_trace = self._solve_admm(problem, warm_start=warm_start, t0=t0)
        return self._enforce_deadline(last_x, last_trace, t0=t0)

    def _trace(
        self,
        status: SolverStatus,
        *,
        t0: float,
        message: str = "",
        backend: str = "none",
        iterations: int = 0,
        pri: float = float("inf"),
        dua: float = float("inf"),
        obj: float = float("inf"),
        warm_started: bool = False,
        extras: dict | None = None,
    ) -> SolverTrace:
        return SolverTrace(
            status=status,
            iterations=iterations,
            primal_residual=pri,
            dual_residual=dua,
            objective=obj,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            warm_started=warm_started,
            backend=backend,
            message=message,
            extras=extras or {},
        )

    def _primal_res(self, problem: QPProblem, x: np.ndarray) -> float:
        Ax = problem.A @ x
        viol = np.maximum(np.maximum(problem.l - Ax, 0.0), np.maximum(Ax - problem.u, 0.0))
        return float(np.linalg.norm(viol, ord=np.inf))

    def _solve_osqp(
        self,
        problem: QPProblem,
        *,
        warm_start: bool,
        t0: float,
    ) -> tuple[np.ndarray | None, SolverTrace]:
        assert _osqp is not None
        import scipy.sparse as sp

        n = problem.q.size
        P = sp.csc_matrix(0.5 * (problem.P + problem.P.T))
        A = sp.csc_matrix(problem.A)
        used_warm = False
        try:
            solver = _osqp.OSQP()
            # OSQP 1.x settings use warm_starting/polishing; fall back for older wheels.
            settings = {
                "verbose": False,
                "max_iter": int(max(self.max_iter, 4000)),
                "eps_abs": float(self.abs_tol),
                "eps_rel": float(self.rel_tol),
                "adaptive_rho": True,
            }
            try:
                solver.setup(
                    P=P,
                    q=problem.q.astype(float),
                    A=A,
                    l=problem.l.astype(float),
                    u=problem.u.astype(float),
                    warm_starting=True,
                    polishing=self.polish,
                    **settings,
                )
            except TypeError:
                solver.setup(
                    P=P,
                    q=problem.q.astype(float),
                    A=A,
                    l=problem.l.astype(float),
                    u=problem.u.astype(float),
                    warm_start=True,
                    polish=self.polish,
                    **settings,
                )
            if warm_start and self._warm.x is not None and self._warm.x.size == n:
                y0 = self._warm.y if self._warm.y is not None else np.zeros(problem.A.shape[0])
                try:
                    solver.warm_start(x=self._warm.x, y=y0)
                    used_warm = True
                except Exception:
                    used_warm = False
            # OSQP 1.x may warn about future raise_error default; keep non-raising for fallback chain.
            try:
                res = solver.solve(raise_error=False)
            except TypeError:
                res = solver.solve()
        except Exception as exc:
            return None, self._trace(
                SolverStatus.NUMERICAL_ERROR,
                t0=t0,
                message=f"osqp_exception:{exc}",
                backend="osqp",
                warm_started=used_warm,
            )

        status_str = str(getattr(res.info, "status", "unknown")).lower()
        x = np.asarray(res.x, dtype=float) if getattr(res, "x", None) is not None else None
        iters = int(getattr(res.info, "iter", 0) or 0)
        pri = float(getattr(res.info, "pri_res", 0.0) or 0.0)
        dua = float(getattr(res.info, "dua_res", 0.0) or 0.0)
        obj = float(getattr(res.info, "obj_val", 0.0) or 0.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if x is not None and np.all(np.isfinite(x)):
            pri_dense = self._primal_res(problem, x)
            pri = max(pri, pri_dense) if pri > 0 else pri_dense
            # Accept solved, solved inaccurate, or max-iter with tight residual.
            acceptable = (
                "solved" in status_str
                or (pri <= max(5e-3, 20 * self.abs_tol) and "infeas" not in status_str and "error" not in status_str)
            )
            if acceptable and pri <= max(5e-3, 20 * self.abs_tol):
                self._warm = WarmStart(
                    x=x.copy(),
                    y=np.asarray(res.y, dtype=float).copy() if getattr(res, "y", None) is not None else None,
                )
                if "solved" in status_str and "inaccurate" not in status_str:
                    st = SolverStatus.SOLVED
                else:
                    st = SolverStatus.SOLVED_INACCURATE
                return x, SolverTrace(
                    status=st,
                    iterations=iters,
                    primal_residual=pri,
                    dual_residual=dua,
                    objective=obj,
                    latency_ms=latency_ms,
                    warm_started=used_warm,
                    backend="osqp",
                    message=status_str,
                )

        if "infeas" in status_str:
            return None, SolverTrace(
                status=SolverStatus.INFEASIBLE,
                iterations=iters,
                primal_residual=pri,
                dual_residual=dua,
                objective=obj,
                latency_ms=latency_ms,
                warm_started=used_warm,
                backend="osqp",
                message=status_str,
            )
        return None, SolverTrace(
            status=SolverStatus.NUMERICAL_ERROR,
            iterations=iters,
            primal_residual=pri,
            dual_residual=dua,
            objective=obj,
            latency_ms=latency_ms,
            warm_started=used_warm,
            backend="osqp",
            message=status_str,
        )

    def _solve_scipy(
        self,
        problem: QPProblem,
        *,
        warm_start: bool,
        t0: float,
    ) -> tuple[np.ndarray | None, SolverTrace]:
        n = problem.q.size
        P = 0.5 * (problem.P + problem.P.T) + 1e-8 * np.eye(n)
        q = problem.q.astype(float)

        used_warm = False
        if warm_start and self._warm.x is not None and self._warm.x.size == n:
            x0 = self._warm.x.copy()
            used_warm = True
        else:
            x0 = np.zeros(n)

        # Split equality vs inequality for SLSQP efficiency.
        eq_A, eq_b = [], []
        ineq_A, ineq_lb, ineq_ub = [], [], []
        for i in range(problem.A.shape[0]):
            lo = float(problem.l[i])
            hi = float(problem.u[i])
            row = problem.A[i]
            if abs(hi - lo) <= 1e-9:
                eq_A.append(row)
                eq_b.append(0.5 * (lo + hi))
            else:
                ineq_A.append(row)
                ineq_lb.append(lo if np.isfinite(lo) else -1e6)
                ineq_ub.append(hi if np.isfinite(hi) else 1e6)

        constraints = []
        if eq_A:
            constraints.append(LinearConstraint(np.vstack(eq_A), np.array(eq_b), np.array(eq_b)))
        if ineq_A:
            constraints.append(
                LinearConstraint(np.vstack(ineq_A), np.array(ineq_lb), np.array(ineq_ub))
            )

        def fun(x: np.ndarray) -> float:
            return float(0.5 * x @ P @ x + q @ x)

        def jac(x: np.ndarray) -> np.ndarray:
            return P @ x + q

        maxiter = min(self.max_iter, max(60, 15 * n))
        try:
            res = minimize(
                fun,
                x0,
                method="SLSQP",
                jac=jac,
                constraints=constraints,
                options={"maxiter": maxiter, "ftol": self.abs_tol, "disp": False},
            )
        except Exception as exc:
            return None, self._trace(
                SolverStatus.NUMERICAL_ERROR,
                t0=t0,
                message=f"scipy_exception:{exc}",
                backend="scipy_slsqp",
                warm_started=used_warm,
            )

        x = np.asarray(res.x, dtype=float)
        obj = fun(x)
        pri = self._primal_res(problem, x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        iters = int(getattr(res, "nit", 0))

        if pri <= max(5e-3, 10 * self.abs_tol) and np.all(np.isfinite(x)):
            self._warm = WarmStart(x=x.copy())
            st = SolverStatus.SOLVED if res.success else SolverStatus.SOLVED_INACCURATE
            return x, SolverTrace(
                status=st,
                iterations=iters,
                primal_residual=pri,
                dual_residual=0.0,
                objective=obj,
                latency_ms=latency_ms,
                warm_started=used_warm,
                backend="scipy_slsqp",
                message=str(res.message),
            )

        if pri > 0.5 or "infeas" in str(res.message).lower():
            return None, SolverTrace(
                status=SolverStatus.INFEASIBLE,
                iterations=iters,
                primal_residual=pri,
                dual_residual=0.0,
                objective=obj,
                latency_ms=latency_ms,
                warm_started=used_warm,
                backend="scipy_slsqp",
                message=str(res.message),
            )

        status = SolverStatus.TIMEOUT if latency_ms > self.deadline_ms else SolverStatus.NUMERICAL_ERROR
        return None, SolverTrace(
            status=status,
            iterations=iters,
            primal_residual=pri,
            dual_residual=0.0,
            objective=obj,
            latency_ms=latency_ms,
            warm_started=used_warm,
            backend="scipy_slsqp",
            message=str(res.message),
        )

    def _solve_admm(
        self,
        problem: QPProblem,
        *,
        warm_start: bool,
        t0: float,
    ) -> tuple[np.ndarray | None, SolverTrace]:
        P = 0.5 * (problem.P + problem.P.T) + 1e-6 * np.eye(problem.q.size)
        q = problem.q.astype(float)
        A = problem.A.astype(float)
        l = problem.l.astype(float)
        u = problem.u.astype(float)
        n = q.size
        m = A.shape[0]
        rho = self.rho
        alpha = self.alpha

        try:
            K_factor = np.linalg.inv(P + rho * (A.T @ A))
        except np.linalg.LinAlgError:
            return None, self._trace(
                SolverStatus.NUMERICAL_ERROR,
                t0=t0,
                message="factorization_failed",
                backend="numpy_admm",
            )

        used_warm = False
        x = np.zeros(n)
        if warm_start and self._warm.x is not None and self._warm.x.size == n:
            x = self._warm.x.copy()
            used_warm = True
        z = np.clip(A @ x, l, u)
        y = np.zeros(m)
        pri_res = float("inf")
        dua_res = float("inf")
        iters = 0
        status = SolverStatus.MAX_ITER

        for k in range(1, min(self.max_iter, 2500) + 1):
            if (time.perf_counter() - t0) * 1000.0 > self.deadline_ms:
                status = SolverStatus.TIMEOUT
                iters = k
                break
            x_next = K_factor @ (-q + rho * A.T @ (z - y / rho))
            Ax_hat = alpha * (A @ x_next) + (1.0 - alpha) * z
            z_next = np.clip(Ax_hat + y / rho, l, u)
            y_next = y + rho * (Ax_hat - z_next)
            pri_res = float(np.linalg.norm(A @ x_next - z_next, ord=np.inf))
            dua_res = float(np.linalg.norm(rho * A.T @ (z_next - z), ord=np.inf))
            x, z, y = x_next, z_next, y_next
            iters = k
            if pri_res <= self.abs_tol + self.rel_tol and dua_res <= self.abs_tol + self.rel_tol:
                status = SolverStatus.SOLVED
                break
            if k % 40 == 0:
                if pri_res > 10 * max(dua_res, 1e-12):
                    rho = min(rho * 2.0, 1e3)
                elif dua_res > 10 * max(pri_res, 1e-12):
                    rho = max(rho / 2.0, 1e-4)
                try:
                    K_factor = np.linalg.inv(P + rho * (A.T @ A))
                except np.linalg.LinAlgError:
                    break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        obj = float(0.5 * x @ P @ x + q @ x)
        # TIMEOUT always wins: never execute a late ADMM iterate.
        if status is SolverStatus.TIMEOUT or latency_ms > self.deadline_ms:
            return None, SolverTrace(
                status=SolverStatus.TIMEOUT,
                iterations=iters,
                primal_residual=pri_res,
                dual_residual=dua_res,
                objective=obj,
                latency_ms=latency_ms,
                warm_started=used_warm,
                backend="numpy_admm",
                message="deadline_exceeded",
            )
        if status is SolverStatus.SOLVED or pri_res < 1e-3:
            self._warm = WarmStart(x=x.copy(), y=y.copy(), z=z.copy())
            return x, SolverTrace(
                status=SolverStatus.SOLVED if status is SolverStatus.SOLVED else SolverStatus.SOLVED_INACCURATE,
                iterations=iters,
                primal_residual=pri_res,
                dual_residual=dua_res,
                objective=obj,
                latency_ms=latency_ms,
                warm_started=used_warm,
                backend="numpy_admm",
                message="admm",
            )
        if pri_res > 1.0:
            return None, SolverTrace(
                status=SolverStatus.INFEASIBLE,
                iterations=iters,
                primal_residual=pri_res,
                dual_residual=dua_res,
                objective=obj,
                latency_ms=latency_ms,
                warm_started=used_warm,
                backend="numpy_admm",
                message="residual_large",
            )
        return None, SolverTrace(
            status=status,
            iterations=iters,
            primal_residual=pri_res,
            dual_residual=dua_res,
            objective=obj,
            latency_ms=latency_ms,
            warm_started=used_warm,
            backend="numpy_admm",
            message=status.value,
        )


def osqp_available() -> bool:
    return _HAS_OSQP


def osqp_local_site() -> str | None:
    return str(_LOCAL_SITE) if _LOCAL_SITE.is_dir() else None
