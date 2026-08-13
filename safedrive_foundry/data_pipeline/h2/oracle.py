"""Physically offline, source-neutral Oracle for completed H2 branches."""

from __future__ import annotations

from .contracts import BranchOutcome, OracleLabel, OracleVerdict


ORACLE_VERSION = "h2-offline-oracle-v1"
PROGRESS_MARGIN_M = 1.0
JERK_MARGIN_MPS3 = 1.0
COMFORT_PROGRESS_TOLERANCE_M = 0.25


def _winner(branch: BranchOutcome, reason: str) -> OracleLabel:
    return OracleLabel(
        verdict=OracleVerdict.CANDIDATE_WIN,
        winner_candidate_id=branch.candidate_id,
        winner_candidate_sha256=branch.candidate_sha256,
        reason=reason,
        oracle_version=ORACLE_VERSION,
    )


def label_pair(left: BranchOutcome, right: BranchOutcome) -> OracleLabel:
    """Apply the frozen H2 rules without consulting source, slot or branch order."""

    if left.candidate_id == right.candidate_id:
        raise ValueError("oracle requires two different candidate ids")
    if not left.complete or not right.complete:
        return OracleLabel(OracleVerdict.INVALID_PAIR, None, None, "incomplete_pair", ORACLE_VERSION)

    if left.hard_unsafe != right.hard_unsafe:
        return _winner(right if left.hard_unsafe else left, "single_hard_unsafe")
    if left.hard_unsafe and right.hard_unsafe:
        return OracleLabel(OracleVerdict.UNRESOLVED, None, None, "both_hard_unsafe", ORACLE_VERSION)

    if left.route_completed != right.route_completed:
        return _winner(left if left.route_completed else right, "route_completion")

    progress_delta = left.route_progress_m - right.route_progress_m
    if abs(progress_delta) >= PROGRESS_MARGIN_M - 1e-12:
        return _winner(left if progress_delta > 0 else right, "route_progress")

    jerk_delta = left.jerk_rms_mps3 - right.jerk_rms_mps3
    if abs(jerk_delta) >= JERK_MARGIN_MPS3 - 1e-12:
        comfortable, other = (left, right) if jerk_delta < 0 else (right, left)
        if comfortable.route_progress_m + COMFORT_PROGRESS_TOLERANCE_M + 1e-12 >= other.route_progress_m:
            return _winner(comfortable, "comfort_jerk")

    return OracleLabel(OracleVerdict.TIE, None, None, "within_frozen_margins", ORACLE_VERSION)


__all__ = ["COMFORT_PROGRESS_TOLERANCE_M", "JERK_MARGIN_MPS3", "ORACLE_VERSION", "PROGRESS_MARGIN_M", "label_pair"]
