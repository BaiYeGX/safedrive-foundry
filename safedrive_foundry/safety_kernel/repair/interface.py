"""Unified Raw / Rule / HardReject / Longitudinal / RATO repair interface."""

from __future__ import annotations

from typing import Protocol, Sequence

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.types import ObservableSnapshot, PolicyCandidate
from safety_kernel.repair.baselines import HardRejectBaseline, RawPassThrough, RuleSlowdownBaseline
from safety_kernel.repair.longitudinal_qp import LongitudinalQPRepair
from safety_kernel.repair.rato_scp import RestrictedRatoScpRepair
from safety_kernel.repair.types import RepairMode, RepairResult


class RepairBackend(Protocol):
    mode: RepairMode

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult: ...


class _RatoBackendAdapter:
    """Adapter so RATO matches RepairBackend signature (force=False)."""

    mode = RepairMode.RATO

    def __init__(self, impl: RestrictedRatoScpRepair) -> None:
        self._impl = impl

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult:
        return self._impl.repair(candidate, obs, now_s=now_s, reject_hints=reject_hints)

    def clear_warm_start(self) -> None:
        self._impl.clear_warm_start()

    @property
    def solver(self) -> object:
        return self._impl.solver

    @property
    def _last_d(self) -> object:
        return self._impl._last_d


class RepairInterface:
    """Single entry for CLAIMS C3 baselines, longitudinal QP, and restricted RATO-SCP."""

    def __init__(self, config: SafetyKernelConfig) -> None:
        self.config = config
        self._rato_impl = RestrictedRatoScpRepair(config)
        self._backends: dict[RepairMode, RepairBackend] = {
            RepairMode.RAW: RawPassThrough(config),
            RepairMode.RULE: RuleSlowdownBaseline(config),
            RepairMode.HARD_REJECT: HardRejectBaseline(config),
            RepairMode.LONGITUDINAL: LongitudinalQPRepair(config),
            RepairMode.RATO: _RatoBackendAdapter(self._rato_impl),  # type: ignore[dict-item]
        }

    def available_modes(self) -> tuple[RepairMode, ...]:
        return (
            RepairMode.RAW,
            RepairMode.RULE,
            RepairMode.HARD_REJECT,
            RepairMode.LONGITUDINAL,
            RepairMode.RATO,
        )

    def repair(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        mode: RepairMode = RepairMode.LONGITUDINAL,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> RepairResult:
        backend = self._backends[mode]
        return backend.repair(candidate, obs, now_s=now_s, reject_hints=reject_hints)

    def compare_all(
        self,
        candidate: PolicyCandidate,
        obs: ObservableSnapshot,
        *,
        now_s: float | None = None,
        reject_hints: Sequence[str] = (),
    ) -> dict[str, RepairResult]:
        return {
            mode.value: self.repair(
                candidate,
                obs,
                mode=mode,
                now_s=now_s,
                reject_hints=reject_hints,
            )
            for mode in self.available_modes()
        }

    def clear_warm_starts(self) -> None:
        long_b = self._backends[RepairMode.LONGITUDINAL]
        if hasattr(long_b, "clear_warm_start"):
            long_b.clear_warm_start()  # type: ignore[attr-defined]
        self._rato_impl.clear_warm_start()
