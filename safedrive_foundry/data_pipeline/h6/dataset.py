"""Outcome-complete training view for World v3.

Source identity is retained only as an offline calibration/audit label.  It is
never included in context or candidate tensors passed to the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_pipeline.h3.dataset import (
    H3DatasetError,
    _candidate_tensor,
    _context_vector,
    read_h2_records,
)
from data_pipeline.h3.contracts import (
    H3_CANDIDATE_DIM,
    H3_CANDIDATE_STEPS,
    H3_CONTEXT_DIM,
    stable_sha256,
)
from data_pipeline.h6.matrix import H6_SEEDS, H6_TRAIN_SEEDS


@dataclass(frozen=True)
class OutcomeCandidateExample:
    candidate_key: str
    source: str
    context: tuple[float, ...]
    candidate: tuple[tuple[float, ...], ...]
    objective_target: float
    progress_m: float
    route_completed: bool
    collision: bool
    red_light_violation: bool
    offroad: bool
    jerk_rms_mps3: float
    acceleration_rms_mps2: float
    lateral_acceleration_rms_mps2: float
    repair_success: bool | None
    trust: bool
    # H2 counterfactual branches observe both candidates.  H6 closed loop only
    # observes the physical outcome of the candidate that actually drove that
    # tick.  Explicit masks prevent the other candidate receiving an invented
    # label while still allowing its independent Safety audit to supervise the
    # risk/trust heads.
    outcome_observed: bool = True
    safety_observed: bool = True
    # These fields are populated only when the closed-loop audit can bind the
    # candidate through Safety and the control adapter.  ``None`` is an honest
    # missing label and is masked by the executable head loss.
    executable: bool | None = None
    repair_method: str | None = None
    phase: str = "unknown"
    group_key: str = "unknown"

    @property
    def hard_unsafe(self) -> bool:
        return self.collision or self.red_light_violation or self.offroad


@dataclass(frozen=True)
class OutcomePairExample:
    pair_id: str
    map_name: str
    family: str
    seed: int
    weather: str
    split: str
    candidates: tuple[OutcomeCandidateExample, OutcomeCandidateExample]
    arm: str | None = None
    tick: int | None = None
    # The source that actually drove this decision tick, bound after the
    # Safety/control adapter.  It is an offline calibration/audit label only;
    # source identity is never included in the World feature tensor.
    executed_source: str | None = None


@dataclass(frozen=True)
class H6PolicyCalibrationExample:
    """Honest paired whole-policy outcome from exact-reset off/on arms."""

    pair_id: str
    map_name: str
    family: str
    seed: int
    weather: str
    candidates: tuple[OutcomeCandidateExample, OutcomeCandidateExample]
    actual_vla_coverage: float

    def as_outcome_pair(self, *, split: str) -> OutcomePairExample:
        """Expose the real paired branch as explicit ranking supervision."""

        return OutcomePairExample(
            pair_id=f"{self.pair_id}:whole-policy",
            map_name=self.map_name,
            family=self.family,
            seed=self.seed,
            weather=self.weather,
            split=split,
            candidates=self.candidates,
            arm="paired_policy",
            tick=0,
        )


def outcome_examples_lineage_sha256(
    examples: Sequence[OutcomePairExample],
) -> str:
    """Hash the complete evaluator/training input contract.

    The dataclass payload includes sample identity, split/seed/group/source
    audit labels, context features, trajectory tensors, every target and all
    supervision masks.  Ordering is preserved because it is part of the
    evaluator input identity.
    """

    if not examples:
        raise ValueError("outcome_examples_lineage_requires_rows")
    return stable_sha256(
        {
            "schema_version": "safedrive.world.outcome_input_lineage.v1",
            "rows": [asdict(example) for example in examples],
        }
    )


def objective_target(branch: Mapping[str, Any]) -> float:
    """Continuous objective outcome score, independent of candidate source."""
    collision = int(branch.get("collision_count", 0)) > 0
    red = bool(branch.get("red_light_violation", False))
    offroad_s = max(0.0, float(branch.get("off_corridor_duration_s", 0.0)))
    progress = max(0.0, float(branch.get("route_progress_m", 0.0)))
    completion = bool(branch.get("route_completed", False))
    jerk = max(0.0, float(branch.get("jerk_rms_mps3", 0.0)))
    accel = max(0.0, float(branch.get("acceleration_rms_mps2", 0.0)))
    lat_accel = max(0.0, float(branch.get("lateral_acceleration_rms_mps2", 0.0)))
    deadline = max(0, int(branch.get("deadline_misses", 0)))
    return (
        4.0 * float(completion)
        + 0.25 * min(progress, 40.0)
        - 12.0 * float(collision)
        - 7.0 * float(red)
        - 4.0 * min(offroad_s, 2.0)
        - 0.12 * math.log1p(jerk)
        - 0.04 * accel
        - 0.06 * lat_accel
        - 0.50 * deadline
    )


def _source(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"vla", "vla_fast", "vla_slow"}:
        return "vla"
    if text in {"expert", "classic"}:
        return "expert"
    return text or "unknown"


def _decision_source(decision: Mapping[str, Any]) -> str:
    """Prefer control-applied provenance, falling back to the legacy field."""

    return _source(
        decision.get("applied_source")
        or decision.get("applied_candidate_source")
        or decision.get("executed_source")
    )


def _repair_success(branch: Mapping[str, Any]) -> bool | None:
    value = branch.get("repair_success")
    if value is None:
        return None
    return bool(value)


def load_outcome_examples(
    roots: Path | Sequence[Path],
    split_manifest: Mapping[str, Any],
    *,
    splits: Sequence[str],
) -> list[OutcomePairExample]:
    wanted = set(str(item) for item in splits)
    if "test" in wanted:
        raise H3DatasetError("World v3 training/calibration cannot load locked test")
    split_rows = {
        str(item["pair_id"]): str(item["split"])
        for item in split_manifest.get("rows", ())
        if str(item.get("split")) in wanted and bool(item.get("valid_pair"))
    }
    records = {str(item["pair_id"]): item for item in read_h2_records(roots)}
    output: list[OutcomePairExample] = []
    for pair_id in sorted(split_rows):
        record = records.get(pair_id)
        if record is None:
            raise H3DatasetError(f"world_v3_pair_missing:{pair_id}")
        candidates = {str(item["candidate_id"]): item for item in record.get("candidates", ())}
        branches = {str(item["candidate_id"]): item for item in record.get("branches", ())}
        if len(candidates) != 2 or set(candidates) != set(branches):
            continue
        source_by_id = {
            candidate_id: _source(candidate.get("source"))
            for candidate_id, candidate in candidates.items()
        }
        expert_id = next((key for key, source in source_by_id.items() if source == "expert"), None)
        if expert_id is None:
            continue
        expert_progress = max(0.0, float(branches[expert_id].get("route_progress_m", 0.0)))
        rows: list[OutcomeCandidateExample] = []
        for candidate_id in sorted(candidates):
            candidate = candidates[candidate_id]
            branch = branches[candidate_id]
            collision = int(branch.get("collision_count", 0)) > 0
            red = bool(branch.get("red_light_violation", False))
            offroad = float(branch.get("off_corridor_duration_s", 0.0)) > 0.25
            progress = max(0.0, float(branch.get("route_progress_m", 0.0)))
            progress_ok = progress + 1e-9 >= 0.80 * expert_progress or progress + 0.50 >= expert_progress
            legal = not (collision or red or offroad)
            trust = bool(branch.get("safety_executed", False)) and legal and progress_ok
            rows.append(
                OutcomeCandidateExample(
                    candidate_key=candidate_id,
                    source=source_by_id[candidate_id],
                    context=tuple(_context_vector(record)),
                    candidate=tuple(tuple(row) for row in _candidate_tensor(record, candidate)),
                    objective_target=objective_target(branch),
                    progress_m=progress,
                    route_completed=bool(branch.get("route_completed", False)),
                    collision=collision,
                    red_light_violation=red,
                    offroad=offroad,
                    jerk_rms_mps3=max(0.0, float(branch.get("jerk_rms_mps3", 0.0))),
                    acceleration_rms_mps2=max(0.0, float(branch.get("acceleration_rms_mps2", 0.0))),
                    lateral_acceleration_rms_mps2=max(0.0, float(branch.get("lateral_acceleration_rms_mps2", 0.0))),
                    repair_success=_repair_success(branch),
                    trust=trust,
                )
            )
        scenario = record["scenario"]
        output.append(
            OutcomePairExample(
                pair_id=pair_id,
                map_name=str(scenario["map_name"]),
                family=str(scenario["family"]),
                seed=int(scenario["seed"]),
                weather=str(scenario["weather"]),
                split=split_rows[pair_id],
                candidates=(rows[0], rows[1]),
            )
        )
    return output


def _closed_loop_repair_success(run: Mapping[str, Any]) -> bool | None:
    attempted = False
    for decision in run.get("decisions", ()):
        repair = decision.get("repair")
        if repair is not None:
            attempted = True
            if bool(repair.get("success")):
                return True
        notes = set((decision.get("arbitration") or {}).get("notes", ()))
        if "preferred_vla_repair_failed_try_expert" in notes:
            attempted = True
    return False if attempted else None


def _closed_loop_features(decision: Mapping[str, Any], source: str):
    matches = [
        (candidate_id, value)
        for candidate_id, value in dict(decision.get("world_features") or {}).items()
        if _source(candidate_id.rsplit(":", 1)[-1]) == source
    ]
    if len(matches) != 1:
        raise H3DatasetError(f"h6_feature_source_count:{source}:{len(matches)}")
    candidate_id, value = matches[0]
    context = tuple(float(item) for item in value.get("context", ()))
    candidate = tuple(
        tuple(float(item) for item in row) for row in value.get("candidate", ())
    )
    if len(context) != H3_CONTEXT_DIM:
        raise H3DatasetError(f"h6_context_shape:{len(context)}")
    if len(candidate) != H3_CANDIDATE_STEPS or any(
        len(row) != H3_CANDIDATE_DIM for row in candidate
    ):
        raise H3DatasetError("h6_candidate_shape")
    if not all(math.isfinite(item) for item in context) or not all(
        math.isfinite(item) for row in candidate for item in row
    ):
        raise H3DatasetError("h6_non_finite_features")
    return candidate_id, context, candidate


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values) / max(1, len(values)))


def _repair_for_candidate(
    decision: Mapping[str, Any], candidate_id: str
) -> bool | None:
    repair = decision.get("repair")
    if not isinstance(repair, Mapping):
        return None
    if str(repair.get("pre_repair_id")) != candidate_id:
        return None
    return bool(repair.get("success"))


def _hazards_from_text(values: Sequence[Any]) -> tuple[bool, bool, bool]:
    text = " ".join(str(value).lower() for value in values)
    return (
        "collision" in text,
        "red_light" in text or "red light" in text,
        "offroad" in text or "off_corridor" in text or ":road:" in text,
    )


def _candidate_safety_label(
    decision: Mapping[str, Any], candidate_id: str, source: str
) -> tuple[bool, bool, bool, bool, bool, bool | None]:
    """Return observed/collision/red/offroad/trust/repair for one candidate."""

    repair = _repair_for_candidate(decision, candidate_id)
    guard = dict(decision.get("guard") or {}).get(candidate_id)
    if isinstance(guard, Mapping) and str(guard.get("verdict", "")).upper() == "REJECT":
        messages = [
            *(guard.get("reject_reasons") or ()),
            *(
                f"{item.get('name')}:{item.get('message')}"
                for item in guard.get("checks", ())
                if not bool(item.get("passed", True))
            ),
        ]
        collision, red, offroad = _hazards_from_text(messages)
        return True, collision, red, offroad, False, repair

    arbitration = decision.get("arbitration")
    audits = arbitration.get("audits", ()) if isinstance(arbitration, Mapping) else ()
    audit = next(
        (
            item
            for item in audits
            if str(item.get("candidate_id")) == candidate_id
            or _source(item.get("source")) == source
        ),
        None,
    )
    if not isinstance(audit, Mapping) or audit.get("final_ok") is None:
        return False, False, False, False, False, repair
    if repair is None and audit.get("repair_success") is not None:
        repair = bool(audit.get("repair_success"))
    final_ok = bool(audit.get("final_ok"))
    collision, red, offroad = _hazards_from_text(audit.get("reject_reasons") or ())
    # A candidate that passes as-is, or is successfully repaired by the
    # bounded Safety layer, is deployable.  Raw hazards remain separate labels.
    trust = final_ok or repair is True
    return True, collision, red, offroad, trust, repair


def _applied_binding(
    decision: Mapping[str, Any], candidate_id: str
) -> tuple[bool | None, str | None]:
    """Return control-bound executability and repair method for a candidate."""

    applied_id = decision.get("applied_candidate_id")
    if applied_id is None:
        applied_id = decision.get("applied_id")
    mode = str(decision.get("applied_mode") or "")
    if applied_id is None:
        return None, None
    if str(applied_id) != str(candidate_id):
        # A different candidate was selected; that does not prove this
        # candidate could not be executed after a different World/Safety
        # ranking.  Keep the executable head masked unless Guard/Safety has a
        # direct negative verdict for this id.
        return None, None
    return mode == "TRACK_APPROVED", str(
        (decision.get("repair") or {}).get("method") or ""
    ) or None


def _local_outcome(
    run: Mapping[str, Any], tick_index: int, source: str, *, window_ticks: int = 10
) -> Mapping[str, Any] | None:
    """Observed 2.5 s-equivalent outcome while the same source keeps control."""

    decisions = list(run.get("decisions", ()))
    timeline = list(run.get("timeline", ()))
    if tick_index >= len(decisions) or tick_index >= len(timeline):
        return None
    if _decision_source(decisions[tick_index]) != source:
        return None
    end = tick_index
    limit = min(len(decisions), len(timeline), tick_index + max(1, window_ticks))
    while (
        end + 1 < limit
        and _decision_source(decisions[end + 1]) == source
    ):
        end += 1
    segment = timeline[tick_index : end + 1]
    if not segment:
        return None

    if tick_index == 0:
        initial = run.get("initial_route_progress_m")
        start_progress = (
            float(initial)
            if initial is not None
            else float(segment[0].get("route_progress_m", 0.0))
        )
        start_time = float(decisions[0].get("simulation_time_s", 0.0))
        previous_accel = None
    else:
        start_progress = float(timeline[tick_index - 1].get("route_progress_m", 0.0))
        start_time = float(timeline[tick_index - 1].get("simulation_time_s", 0.0))
        previous_accel = float(timeline[tick_index - 1].get("acceleration_mps2", 0.0))
    end_progress = float(segment[-1].get("route_progress_m", start_progress))
    end_time = float(segment[-1].get("simulation_time_s", start_time + 0.05))
    elapsed = max(0.05, end_time - start_time)
    observed_progress = max(0.0, end_progress - start_progress)
    progress_2p5s = min(40.0, observed_progress * 2.5 / elapsed)

    accelerations = [float(item.get("acceleration_mps2", 0.0)) for item in segment]
    lateral = [float(item.get("lateral_acceleration_mps2", 0.0)) for item in segment]
    jerk = []
    prior = previous_accel
    prior_time = start_time
    for item, acceleration in zip(segment, accelerations):
        current_time = float(item.get("simulation_time_s", prior_time + 0.05))
        if prior is not None:
            jerk.append((acceleration - prior) / max(1e-3, current_time - prior_time))
        prior = acceleration
        prior_time = current_time

    start_frame = int(segment[0].get("carla_frame", -1))
    end_frame = int(segment[-1].get("carla_frame", start_frame))
    collision = any(
        str(event.get("event_type")) == "collision"
        and start_frame <= int(event.get("frame", -2)) <= end_frame
        for event in run.get("events", ())
    )
    offroad = 0.05 * sum(
        float(item.get("corridor_distance_m", 0.0)) > 2.0 for item in segment
    ) > 0.25
    stop_progress = run.get("red_light_stop_progress_m")
    red_active = any(
        bool(decisions[index].get("red_light_active"))
        for index in range(tick_index, end + 1)
    )
    red = bool(
        stop_progress is not None
        and red_active
        and start_progress <= float(stop_progress) + 1.0
        and end_progress > float(stop_progress) + 1.0
    )
    completed = bool(run.get("route_completed")) and end + 1 >= len(timeline)
    return {
        "route_progress_m": progress_2p5s,
        "route_completed": completed,
        "collision_count": int(collision),
        "red_light_violation": red,
        "off_corridor_duration_s": 0.30 if offroad else 0.0,
        "jerk_rms_mps3": _rms(jerk),
        "acceleration_rms_mps2": _rms(accelerations),
        "lateral_acceleration_rms_mps2": _rms(lateral),
        "deadline_misses": int(
            any(bool(item.get("deadline_miss")) for item in segment)
        ),
    }


def _read_h6_grouped_runs(
    roots: Path | Sequence[Path], wanted_seeds: set[int]
) -> dict[str, dict[str, Mapping[str, Any]]]:
    root_list = (roots,) if isinstance(roots, Path) else tuple(roots)
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for root in root_list:
        for path in sorted(Path(root).glob("runs/*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            expected = run.get("content_sha256")
            if expected is not None:
                verified = {
                    key: value for key, value in run.items() if key != "content_sha256"
                }
                if stable_sha256(verified) != expected:
                    raise H3DatasetError(f"h6_run_hash_mismatch:{path}")
            scenario = dict(run.get("scenario") or {})
            if int(scenario.get("seed", -1)) not in wanted_seeds:
                continue
            if run.get("manifest_kind") != "h6_fresh_training":
                raise H3DatasetError(f"h6_nontraining_manifest:{path}")
            grouped.setdefault(str(run.get("pair_id")), {})[
                str(run.get("arm"))
            ] = run
    return grouped


def _validate_h6_arms(
    pair_id: str, arms: Mapping[str, Mapping[str, Any]]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if set(arms) != {"off", "on"}:
        raise H3DatasetError(f"h6_missing_closed_loop_arm:{pair_id}")
    expert_run, vla_run = arms["off"], arms["on"]
    if not bool(expert_run.get("ok")) or not bool(vla_run.get("ok")):
        raise H3DatasetError(f"h6_training_run_not_ok:{pair_id}")
    comparison = dict(vla_run.get("reset_comparison") or {})
    if not bool(comparison.get("comparable")):
        raise H3DatasetError(f"h6_reset_not_comparable:{pair_id}")
    if expert_run.get("physical_sha256") != vla_run.get("physical_sha256"):
        raise H3DatasetError(f"h6_physical_mismatch:{pair_id}")
    if int(expert_run.get("vla_executed_ticks", 0)) > 0:
        raise H3DatasetError(f"h6_classic_baseline_contaminated:{pair_id}")
    return expert_run, vla_run


def load_h6_closed_loop_examples(
    roots: Path | Sequence[Path],
    *,
    seeds: Sequence[int],
    split: str,
) -> list[OutcomePairExample]:
    """Load honest per-tick H6 labels without invented counterfactual outcomes.

    Every row contains both source-blind candidates from the same observable
    anchor.  Physical outcome/comfort labels are enabled only for the source
    that actually drove.  Independent Guard/Safety audits may label either
    candidate's risk/trust heads even when that candidate was not executed.
    """

    if split == "test":
        raise H3DatasetError("H6 training cannot load locked test")
    wanted_seeds = {int(seed) for seed in seeds}
    if not wanted_seeds:
        raise H3DatasetError("h6_training_seeds_required")
    if not wanted_seeds.issubset(set(H6_TRAIN_SEEDS)):
        overlap = sorted(wanted_seeds.intersection(H6_SEEDS))
        raise H3DatasetError(f"h6_acceptance_seed_training_forbidden:{overlap}")
    grouped = _read_h6_grouped_runs(roots, wanted_seeds)

    output = []
    for pair_id in sorted(grouped):
        expert_run, vla_run = _validate_h6_arms(pair_id, grouped[pair_id])
        if not vla_run.get("decisions"):
            raise H3DatasetError(f"h6_decisions_missing:{pair_id}")
        scenario = dict(vla_run["scenario"])
        for run in (expert_run, vla_run):
            for tick_index, decision in enumerate(run.get("decisions", ())):
                if len(decision.get("world_features") or {}) != 2:
                    # Legacy off arms did not record features.  They remain
                    # valid evidence but cannot provide aligned model inputs.
                    continue
                executed = _decision_source(decision)
                phase = str(decision.get("phase") or "")
                if not phase:
                    ticks = max(1, int(run.get("ticks_executed", len(run.get("decisions", ())))))
                    third = max(1, ticks // 3)
                    phase = "early" if tick_index < third else "late" if tick_index >= 2 * third else "middle"
                group_key = f"{scenario.get('map_name','unknown')}|{scenario.get('family','unknown')}|{phase}"
                rows = []
                for source in ("expert", "vla"):
                    candidate_id, context, candidate = _closed_loop_features(
                        decision, source
                    )
                    local = _local_outcome(run, tick_index, source)
                    outcome_observed = local is not None and executed == source
                    outcome = local or {}
                    (
                        safety_observed,
                        audit_collision,
                        audit_red,
                        audit_offroad,
                        trust,
                        repair,
                    ) = _candidate_safety_label(decision, candidate_id, source)
                    executable, repair_method = _applied_binding(decision, candidate_id)
                    guard_row = dict(decision.get("guard") or {}).get(candidate_id)
                    if (
                        executable is None
                        and isinstance(guard_row, Mapping)
                        and str(guard_row.get("verdict", "")).upper() == "REJECT"
                    ):
                        executable = False
                    physical_collision = bool(outcome.get("collision_count", 0))
                    physical_red = bool(outcome.get("red_light_violation", False))
                    physical_offroad = (
                        float(outcome.get("off_corridor_duration_s", 0.0)) > 0.25
                    )
                    collision = physical_collision or audit_collision
                    red = physical_red or audit_red
                    offroad = physical_offroad or audit_offroad
                    rows.append(
                        OutcomeCandidateExample(
                            candidate_key=candidate_id,
                            source=source,
                            context=context,
                            candidate=candidate,
                            objective_target=(
                                objective_target(outcome)
                                if outcome_observed
                                else 0.0
                            ),
                            progress_m=max(
                                0.0, float(outcome.get("route_progress_m", 0.0))
                            ),
                            route_completed=bool(
                                outcome.get("route_completed", False)
                            ),
                            collision=collision,
                            red_light_violation=red,
                            offroad=offroad,
                            jerk_rms_mps3=max(
                                0.0, float(outcome.get("jerk_rms_mps3", 0.0))
                            ),
                            acceleration_rms_mps2=max(
                                0.0,
                                float(outcome.get("acceleration_rms_mps2", 0.0)),
                            ),
                            lateral_acceleration_rms_mps2=max(
                                0.0,
                                float(
                                    outcome.get(
                                        "lateral_acceleration_rms_mps2", 0.0
                                    )
                                ),
                            ),
                            repair_success=repair,
                            trust=bool(trust)
                            and not (
                                physical_collision
                                or physical_red
                                or physical_offroad
                            ),
                            outcome_observed=outcome_observed,
                            safety_observed=safety_observed or outcome_observed,
                            executable=executable,
                            repair_method=repair_method,
                            phase=phase,
                            group_key=group_key,
                        )
                    )
                output.append(
                    OutcomePairExample(
                        pair_id=pair_id,
                        map_name=str(scenario["map_name"]),
                        family=str(scenario["family"]),
                        seed=int(scenario["seed"]),
                        weather=str(scenario["weather"]),
                        split=split,
                        candidates=(rows[0], rows[1]),
                        arm=str(run.get("arm")),
                        tick=tick_index,
                        executed_source=executed,
                    )
                )
    if not output:
        raise H3DatasetError("h6_no_aligned_tick_examples")
    return output


def load_h6_policy_calibration_examples(
    roots: Path | Sequence[Path], *, seeds: Sequence[int]
) -> list[H6PolicyCalibrationExample]:
    """Load one honest Classic-vs-VLA-primary policy outcome per scenario.

    The caller may place the development-seed rows in the training split for
    whole-policy preference/executable supervision, while the held-out
    calibration-seed rows are used only to choose deployment thresholds.  No
    formal/consumed seed can reach this loader.
    """

    wanted_seeds = {int(seed) for seed in seeds}
    if not wanted_seeds or not wanted_seeds.issubset(set(H6_TRAIN_SEEDS)):
        overlap = sorted(wanted_seeds.intersection(H6_SEEDS))
        raise H3DatasetError(f"h6_acceptance_seed_training_forbidden:{overlap}")
    grouped = _read_h6_grouped_runs(roots, wanted_seeds)
    output = []
    for pair_id in sorted(grouped):
        expert_run, vla_run = _validate_h6_arms(pair_id, grouped[pair_id])
        decisions = list(vla_run.get("decisions", ()))
        if not decisions:
            raise H3DatasetError(f"h6_decisions_missing:{pair_id}")
        expert_id, expert_context, expert_candidate = _closed_loop_features(
            decisions[0], "expert"
        )
        vla_id, vla_context, vla_candidate = _closed_loop_features(
            decisions[0], "vla"
        )
        expert_progress = max(0.0, float(expert_run.get("route_progress_m", 0.0)))
        vla_progress = max(0.0, float(vla_run.get("route_progress_m", 0.0)))

        def row(candidate_id, source, context, candidate, run):
            collision = int(run.get("collision_count", 0)) > 0
            red = bool(run.get("red_light_violation"))
            offroad = float(run.get("off_corridor_duration_s", 0.0)) > 0.25
            return OutcomeCandidateExample(
                candidate_key=candidate_id,
                source=source,
                context=context,
                candidate=candidate,
                objective_target=objective_target(run),
                progress_m=max(0.0, float(run.get("route_progress_m", 0.0))),
                route_completed=bool(run.get("route_completed")),
                collision=collision,
                red_light_violation=red,
                offroad=offroad,
                jerk_rms_mps3=max(0.0, float(run.get("jerk_rms_mps3", 0.0))),
                acceleration_rms_mps2=max(
                    0.0, float(run.get("acceleration_rms_mps2", 0.0))
                ),
                lateral_acceleration_rms_mps2=max(
                    0.0, float(run.get("lateral_acceleration_rms_mps2", 0.0))
                ),
                repair_success=(
                    _closed_loop_repair_success(run) if source == "vla" else None
                ),
                trust=not (collision or red or offroad),
                executable=(
                    bool(run.get("arm") == "on" and source == "vla" and run.get("vla_executed_ticks", 0) > 0)
                    if source == "vla"
                    else bool(run.get("arm") == "off")
                ),
                phase="episode",
                group_key=f"{vla_run.get('scenario', {}).get('map_name', 'unknown')}|{vla_run.get('scenario', {}).get('family', 'unknown')}|episode",
            )

        ticks = max(1, int(vla_run.get("ticks_executed", 0)))
        output.append(
            H6PolicyCalibrationExample(
                pair_id=pair_id,
                map_name=str(vla_run["scenario"]["map_name"]),
                family=str(vla_run["scenario"]["family"]),
                seed=int(vla_run["scenario"]["seed"]),
                weather=str(vla_run["scenario"]["weather"]),
                candidates=(
                    row(
                        expert_id,
                        "expert",
                        expert_context,
                        expert_candidate,
                        expert_run,
                    ),
                    row(vla_id, "vla", vla_context, vla_candidate, vla_run),
                ),
                actual_vla_coverage=int(vla_run.get("vla_executed_ticks", 0)) / ticks,
            )
        )
    if not output:
        raise H3DatasetError("h6_policy_calibration_examples_missing")
    return output


__all__ = [
    "OutcomeCandidateExample",
    "OutcomePairExample",
    "H6PolicyCalibrationExample",
    "outcome_examples_lineage_sha256",
    "load_outcome_examples",
    "load_h6_closed_loop_examples",
    "load_h6_policy_calibration_examples",
    "objective_target",
]
