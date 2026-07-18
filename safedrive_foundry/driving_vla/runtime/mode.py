"""VLA_SAFETY vs HYBRID candidate filtering."""

from __future__ import annotations

from enum import Enum

from safety_kernel.contracts.types import CandidateSource, PolicyCandidate, PolicyCandidateSet


class RuntimeMode(str, Enum):
    VLA_SAFETY = "VLA_SAFETY"
    HYBRID = "HYBRID"
    CLASSIC = "CLASSIC"


def filter_candidates_for_mode(cset: PolicyCandidateSet | None, mode: RuntimeMode) -> PolicyCandidateSet | None:
    if cset is None:
        return None
    if mode is RuntimeMode.HYBRID:
        return cset
    if mode is RuntimeMode.CLASSIC:
        kept = tuple(c for c in cset.candidates if c.source is CandidateSource.CLASSIC)
    else:  # VLA_SAFETY — no Classic current-frame candidates
        kept = tuple(
            c
            for c in cset.candidates
            if c.source in {CandidateSource.VLA_FAST, CandidateSource.VLA_SLOW}
        )
    return PolicyCandidateSet(
        run_id=cset.run_id,
        frame_id=cset.frame_id,
        scenario_id=cset.scenario_id,
        model_id=cset.model_id,
        carla_frame=cset.carla_frame,
        simulation_time_s=cset.simulation_time_s,
        wall_time_s=cset.wall_time_s,
        candidates=kept,
        schema_version=cset.schema_version,
        coordinate_frame=cset.coordinate_frame,
    )


def availability_for_mode(mode: RuntimeMode, *, vla_ok: bool = True):
    from safety_kernel.contracts.types import ComponentAvailability

    if mode is RuntimeMode.VLA_SAFETY:
        return ComponentAvailability(classic=False, vla=vla_ok, world=False, safety=True)
    if mode is RuntimeMode.HYBRID:
        return ComponentAvailability(classic=True, vla=vla_ok, world=False, safety=True)
    return ComponentAvailability(classic=True, vla=False, world=False, safety=True)
