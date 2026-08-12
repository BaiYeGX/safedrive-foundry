"""Action-conditioned signal and noise audits for R3 final-head data."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import ActionBranchSample, WorldContractError


def masked_actor_future_delta(sample: ActionBranchSample) -> float | None:
    """Return candidate-0/1 actor-future ADE on jointly valid slots."""
    if not bool(sample.candidate_mask.all()):
        return None
    valid = np.asarray(sample.actor_future_mask[0] & sample.actor_future_mask[1], dtype=bool)
    if not bool(valid.any()):
        return None
    # x/y are the first two future features.  Actor identity/order was bound by
    # the collector before this sample was built, so no outcome is consulted.
    delta = sample.actor_future[0, :, :, :2] - sample.actor_future[1, :, :, :2]
    distances = np.linalg.norm(delta, axis=-1)
    return float(distances[valid].mean())


def action_signal_report(
    samples: Sequence[ActionBranchSample],
    *,
    aa_p99: float,
    min_reactive_fraction: float = 0.25,
    min_reactive_decisive_fraction: float = 0.50,
) -> dict[str, Any]:
    """Audit whether candidate-conditioned actor futures exceed A/A noise."""
    if not np.isfinite(float(aa_p99)) or float(aa_p99) < 0.0:
        raise WorldContractError("aa_p99 must be finite and non-negative")
    reactive: list[float] = []
    reactive_decisive: list[float] = []
    by_family: dict[str, list[float]] = {}
    for sample in samples:
        if not bool(sample.audit.get("reactive_actor_present", False)):
            continue
        delta = masked_actor_future_delta(sample)
        if delta is None:
            continue
        reactive.append(delta)
        if sample.rank_mask and not sample.tie_target:
            reactive_decisive.append(delta)
        by_family.setdefault(sample.identity.family, []).append(delta)
    sensitive = [value for value in reactive if value > float(aa_p99)]
    sensitive_decisive = [value for value in reactive_decisive if value > float(aa_p99)]
    family_rates = {
        family: sum(value > float(aa_p99) for value in values) / max(len(values), 1)
        for family, values in sorted(by_family.items())
    }
    overall_rate = len(sensitive) / max(len(reactive), 1)
    decisive_rate = len(sensitive_decisive) / max(len(reactive_decisive), 1)
    return {
        "schema_version": "safedrive.r3.action_signal.v1",
        "aa_p99": float(aa_p99),
        "reactive_count": len(reactive),
        "reactive_decisive_count": len(reactive_decisive),
        "action_sensitive_count": len(sensitive),
        "action_sensitive_decisive_count": len(sensitive_decisive),
        "action_sensitive_reactive_fraction": overall_rate,
        "action_sensitive_reactive_decisive_fraction": decisive_rate,
        "family_action_sensitive_fraction": family_rates,
        "gate_reactive_fraction": overall_rate >= float(min_reactive_fraction),
        "gate_reactive_decisive_fraction": decisive_rate >= float(min_reactive_decisive_fraction),
        "gate_has_observable_signal": bool(reactive),
    }


def assert_final_head_namespace(
    samples: Iterable[ActionBranchSample],
    *,
    namespace: str,
    checkpoint_sha256: str,
) -> None:
    """Fail closed if samples are not bound to the final-head namespace/hash."""
    expected = str(checkpoint_sha256).lower()
    for sample in samples:
        actual_namespace = str(sample.audit.get("namespace", ""))
        actual_checkpoint = str(sample.audit.get("r2_checkpoint_sha256", "")).lower()
        if actual_namespace != str(namespace):
            raise WorldContractError(
                f"sample {sample.identity.sample_id} namespace mismatch: {actual_namespace}"
            )
        if actual_checkpoint != expected:
            raise WorldContractError(
                f"sample {sample.identity.sample_id} checkpoint binding mismatch"
            )


__all__ = [
    "action_signal_report",
    "assert_final_head_namespace",
    "masked_actor_future_delta",
]
