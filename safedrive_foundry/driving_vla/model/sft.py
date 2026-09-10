"""C3 SimLingo SFT data and differentiable training helpers.

The C2 release contains an observable anchor, a nominal expert proposal and a
canonical 10 point timed trajectory.  This module adapts that release to the
native SimLingo driving heads without using the C2 World labels as driving
supervision.  It deliberately keeps data loading and metric aggregation free
of the heavyweight model import so that the contract can be unit tested on
CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data_pipeline.h6.cora.loader import load_cora_roots
from data_pipeline.h6.cora.config import CORA_C2_CONFIG


ROUTE_STEPS = 20
SPEED_STEPS = 10
SPEED_DT_S = 0.25
EXECUTION_DT_S = float(CORA_C2_CONFIG["timing"]["fixed_delta_seconds"])
RELEASE_ID = "h6-cora-c2-devbaseline-20260907-v1"
QUALITY_PROFILE = "c2_dev_baseline_v1"
TARGET_SPLITS = ("train", "validation")
ROUTE_PROJECTION_MAX_DISTANCE_M = 2.5
ROUTE_PROJECTION_TIE_DISTANCE_M = 0.05
ROUTE_PROJECTION_TIE_PROGRESS_M = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def map_to_ego(
    points: Sequence[Sequence[float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
) -> np.ndarray:
    """Convert finite map-frame points to the anchor ego frame."""

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points_shape:{arr.shape}")
    dx = arr[:, 0] - float(ego_x)
    dy = arr[:, 1] - float(ego_y)
    c = math.cos(-float(ego_yaw))
    s = math.sin(-float(ego_yaw))
    return np.column_stack((c * dx - s * dy, s * dx + c * dy))


def cumulative_arclength(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError("polyline_requires_two_points")
    distances = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(distances)))


def sample_polyline(
    points: Sequence[Sequence[float]],
    distances: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a polyline without extrapolation and return values plus mask."""

    arr = np.asarray(points, dtype=np.float64)
    arc = cumulative_arclength(arr)
    query = np.asarray(distances, dtype=np.float64)
    result = np.zeros((query.size, 2), dtype=np.float64)
    mask = np.zeros(query.size, dtype=bool)
    for index, target in enumerate(query):
        if not math.isfinite(float(target)) or target < -1e-8 or target > arc[-1] + 1e-8:
            continue
        right = int(np.searchsorted(arc, target, side="left"))
        if right <= 0:
            result[index] = arr[0]
        elif right >= len(arc):
            result[index] = arr[-1]
        else:
            left = right - 1
            span = max(float(arc[right] - arc[left]), 1e-12)
            ratio = (float(target) - float(arc[left])) / span
            result[index] = arr[left] + ratio * (arr[right] - arr[left])
        mask[index] = True
    return result, mask


def _clean_consecutive_points(points: np.ndarray, *, tolerance_m: float = 1e-8) -> np.ndarray:
    """Remove only consecutive duplicate/non-finite route vertices.

    A route can legitimately self-intersect.  Cleaning is therefore deliberately
    local: it never sorts points, removes a later crossing, or drops a point just
    because one coordinate is negative in the ego frame.
    """

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"route_points_shape:{points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("route_points_non_finite")
    kept: list[np.ndarray] = []
    for point in points:
        if not kept or float(np.linalg.norm(point - kept[-1])) > tolerance_m:
            kept.append(point)
    if len(kept) < 2:
        raise ValueError("route_requires_two_distinct_points")
    return np.stack(kept, axis=0)


def native_route_target_with_report(
    route_map: Sequence[Sequence[float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    steps: int = ROUTE_STEPS,
    max_projection_distance_m: float = ROUTE_PROJECTION_MAX_DISTANCE_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Project the anchor onto the ordered native expert reference path.

    C2 stores a complete route from the scenario start.  An anchor may be in
    the middle of that route, and the first stored vertices can consequently be
    behind the vehicle.  The old adapter prepended the ego to the *untrimmed*
    route and taught those backward vertices as the native route head.  We first
    project onto the closest ordered segment, retain the forward suffix, and
    sample arc distances 0..19 m from the anchor.  The returned report records
    every decision needed to audit the projection and its remaining support.
    """

    if steps <= 0:
        raise ValueError("steps_must_be_positive")
    route = _clean_consecutive_points(np.asarray(route_map, dtype=np.float64))
    origin = np.asarray([float(ego_x), float(ego_y)], dtype=np.float64)
    if not np.isfinite(origin).all() or not math.isfinite(float(ego_yaw)):
        raise ValueError("route_anchor_non_finite")
    arc = cumulative_arclength(route)
    candidates: list[tuple[float, float, int, float, np.ndarray]] = []
    for index, (left, right) in enumerate(zip(route[:-1], route[1:])):
        vector = right - left
        squared = float(vector @ vector)
        if squared <= 1e-16:
            continue
        fraction = float(np.clip((origin - left) @ vector / squared, 0.0, 1.0))
        projected = left + fraction * vector
        distance = float(np.linalg.norm(origin - projected))
        progress = float(arc[index] + fraction * (arc[index + 1] - arc[index]))
        candidates.append((distance, progress, index, fraction, projected))
    if not candidates:
        raise ValueError("route_has_no_nonzero_segment")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_distance, best_progress, best_index, best_fraction, projected = candidates[0]
    ambiguous = False
    if len(candidates) > 1:
        second = candidates[1]
        ambiguous = bool(
            second[0] - best_distance <= ROUTE_PROJECTION_TIE_DISTANCE_M
            and abs(second[1] - best_progress) >= ROUTE_PROJECTION_TIE_PROGRESS_M
        )
    if ambiguous:
        raise ValueError(
            f"route_projection_ambiguous:{best_distance:.6f}:{candidates[1][0]:.6f}"
        )
    if best_distance > float(max_projection_distance_m):
        raise ValueError(f"route_projection_too_far:{best_distance:.6f}>{float(max_projection_distance_m):.6f}")

    # Keep the ordered suffix, including the fractional projection.  Adding the
    # origin makes point 0 exactly [0, 0] in the ego frame; no extrapolation or
    # navigation splice is used to fill a short suffix.
    suffix: list[np.ndarray] = [origin, projected]
    suffix.extend(route[best_index + 1 :])
    suffix_array = _clean_consecutive_points(np.asarray(suffix, dtype=np.float64))
    support_arc = cumulative_arclength(suffix_array)
    sampled_map, mask = sample_polyline(suffix_array, np.arange(steps, dtype=np.float64))
    report = {
        "source": "native_expert_reference_path",
        "projection_segment": int(best_index),
        "projection_fraction": float(best_fraction),
        "projection_s_m": float(best_progress),
        "projection_distance_m": float(best_distance),
        "projection_ambiguous": bool(ambiguous),
        "raw_point_count": int(route.shape[0]),
        "suffix_point_count": int(suffix_array.shape[0]),
        "support_m": float(support_arc[-1]),
        "remaining_route_m": float(arc[-1] - best_progress),
        "valid_points": int(mask.sum()),
        "steps": int(steps),
        "spacing_m": 1.0,
        "max_projection_distance_m": float(max_projection_distance_m),
    }
    return map_to_ego(sampled_map, ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw), mask, report


def native_route_target(
    route_map: Sequence[Sequence[float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    steps: int = ROUTE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the native 1 m-spaced route head target in ego coordinates."""
    target, mask, _ = native_route_target_with_report(
        route_map,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_yaw=ego_yaw,
        steps=steps,
    )
    return target, mask


def native_speed_target(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    steps: int = SPEED_STEPS,
    dt_s: float = SPEED_DT_S,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the native 2-D speed-waypoint target and preserve missingness."""

    values = np.zeros((steps, 2), dtype=np.float64)
    mask = np.zeros(steps, dtype=bool)
    failures: list[str] = []
    if len(trajectory) < steps:
        failures.append(f"trajectory_short:{len(trajectory)}<{steps}")
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in trajectory:
        try:
            t = float(row["t"])
            index = int(round(t / dt_s)) - 1
        except (KeyError, TypeError, ValueError):
            failures.append("time_invalid")
            continue
        expected = (index + 1) * dt_s
        if index < 0 or index >= steps or abs(t - expected) > 1e-5:
            failures.append(f"time_grid:{t}")
            continue
        if index in by_index:
            failures.append(f"time_duplicate:{index}")
        by_index[index] = row
    for index in range(steps):
        row = by_index.get(index)
        if row is None:
            failures.append(f"missing_step:{index}")
            continue
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"xy_invalid:{index}")
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            failures.append(f"xy_non_finite:{index}")
            continue
        values[index] = (x, y)
        mask[index] = True
    return map_to_ego(values, ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw), mask, failures


def native_speed_target_from_timeline(
    timeline: Sequence[Mapping[str, Any]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    execution_dt_s: float = EXECUTION_DT_S,
    steps: int = SPEED_STEPS,
    dt_s: float = SPEED_DT_S,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample actual expert execution states on the native 0.25 s grid.

    C2 timelines are recorded at ``execution_dt_s`` (0.05 s) and use tick 0
    for the first post-anchor state.  A terminal collision or MRM can end the
    file early; missing future grid points stay masked rather than being
    copied from the last state.  The canonical proposal is validated
    separately, but it is never used to fill an absent execution state.
    """

    values = np.zeros((steps, 2), dtype=np.float64)
    mask = np.zeros(steps, dtype=bool)
    failures: list[str] = []
    if not math.isfinite(float(execution_dt_s)) or execution_dt_s <= 0.0:
        raise ValueError("execution_dt_invalid")
    by_index: dict[int, Mapping[str, Any]] = {}
    ticks: list[int] = []
    for row in timeline:
        try:
            tick = int(row["tick"])
            x = float(row["x"])
            y = float(row["y"])
        except (KeyError, TypeError, ValueError):
            failures.append("timeline_state_invalid")
            continue
        ticks.append(tick)
        if not (math.isfinite(x) and math.isfinite(y)):
            failures.append(f"timeline_xy_non_finite:{tick}")
            continue
        # tick 0 is the first state after one execution interval.
        elapsed = (tick + 1) * float(execution_dt_s)
        index = int(round(elapsed / float(dt_s))) - 1
        expected = (index + 1) * float(dt_s)
        if index < 0 or index >= steps or abs(elapsed - expected) > max(1e-5, execution_dt_s * 0.51):
            continue
        if index in by_index:
            failures.append(f"timeline_duplicate_step:{index}")
        by_index[index] = {"x": x, "y": y}
    if ticks and ticks != list(range(min(ticks), min(ticks) + len(ticks))):
        failures.append("timeline_tick_sequence_invalid")
    for index in range(steps):
        row = by_index.get(index)
        if row is None:
            failures.append(f"missing_execution_step:{index}")
            continue
        values[index] = (float(row["x"]), float(row["y"]))
        mask[index] = True
    return map_to_ego(values, ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw), mask, failures


@dataclass(frozen=True)
class SFTSample:
    root_id: str
    pair_id: str
    split: str
    family: str
    map_name: str
    weather: str
    seed: int
    base_root: str
    anchor_path: str
    image_path: str
    image_sha256: str
    proposal_path: str
    proposal_sha256: str
    teacher_source: str
    teacher_generator: str
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_speed_mps: float
    target_ego_1: tuple[float, float]
    target_ego_2: tuple[float, float]
    route_target: tuple[tuple[float, float], ...]
    route_mask: tuple[bool, ...]
    speed_target: tuple[tuple[float, float], ...]
    speed_mask: tuple[bool, ...]
    route_source: str = "native_expert_reference_path_projected"
    speed_source: str = "expert_nominal_canonical_proposal"
    execution_timeline_path: str = ""
    execution_timeline_sha256: str = ""
    execution_timeline_rows: int = 0
    execution_timeline_dt_s: float | None = None
    route_missing_reasons: tuple[str, ...] = ()
    speed_missing_reasons: tuple[str, ...] = ()
    route_coordinate_frame: str = "ego"
    speed_coordinate_frame: str = "ego"
    route_spacing_m: float = 1.0
    speed_dt_s: float = SPEED_DT_S
    route_projection_s_m: float = 0.0
    route_projection_distance_m: float = 0.0
    route_support_m: float = 0.0
    route_projection_segment: int = -1
    route_projection_ambiguous: bool = False
    route_reference_revision: str = ""
    route_validity_basis: str = "native_expert_reference_path_audited"
    execution_quality: str = "audited"
    execution_failure_reasons: tuple[str, ...] = ()
    outcome_label_path: str = ""
    outcome_label_sha256: str = ""
    outcome_label_heads: tuple[tuple[str, Any], ...] = ()
    execution_timeline_start_offset_s: float | None = None
    execution_timeline_alignment: str = "unknown"
    execution_timeline_speed_usable: bool = False
    speed_input_source: str = "anchor.observable_snapshot.ego_v"
    anchor_history_speed_mps: float | None = None
    anchor_history_speed_disagreement_mps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _resolve_base_root(release_root: Path, release: Mapping[str, Any]) -> Path:
    raw = release.get("raw_data_references", {}).get("base")
    if not raw:
        raise ValueError("c3_release_base_reference_missing")
    base = Path(str(raw))
    return base if base.is_absolute() else release_root / base


def _find_expert_branch(pair: Mapping[str, Any]) -> Mapping[str, Any]:
    proposals = pair.get("proposals", ())
    for proposal in proposals:
        if proposal.get("kind") != "nominal" or proposal.get("audit_source") != "expert":
            continue
        proposal_id = proposal.get("proposal_id")
        branch = next((item for item in pair.get("branches", ()) if item.get("proposal_id") == proposal_id), None)
        if branch is not None:
            return {"proposal": proposal, "branch": branch}
    raise ValueError("c3_expert_nominal_branch_missing")


def _validate_expert_timeline(
    base_root: Path,
    branch: Mapping[str, Any],
) -> tuple[Path, str, int, float, tuple[dict[str, Any], ...]]:
    """Validate the separately stored expert execution timeline.

    The canonical proposal is checked separately below.  This timeline is
    bound here so a nominal proposal cannot be treated as a valid demonstration
    merely because it is labelled ``expert``; actual execution evidence must
    also be present and internally consistent.
    """

    artifact_paths = branch.get("artifact_paths", {})
    artifact_hashes = branch.get("artifact_sha256", {})
    raw_path = artifact_paths.get("timeline")
    expected_hash = str(artifact_hashes.get("timeline", ""))
    if not raw_path or not expected_hash:
        raise ValueError("expert_execution_timeline_reference_missing")
    timeline_path = base_root / str(raw_path)
    if not timeline_path.is_file():
        raise FileNotFoundError(timeline_path)
    actual_hash = sha256_file(timeline_path)
    if actual_hash != expected_hash:
        raise ValueError("expert_execution_timeline_hash_mismatch")
    expected_rows = int(branch.get("ticks_executed", 0))
    if expected_rows <= 0:
        raise ValueError("expert_execution_timeline_ticks_missing")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - the C3 runtime bundles pyarrow
        raise RuntimeError("expert_execution_timeline_reader_missing") from exc
    table = pq.read_table(
        timeline_path,
        columns=["tick", "simulation_time_s", "x", "y", "speed_mps", "control_mode", "deadline_miss"],
    )
    rows = int(table.num_rows)
    if rows != expected_rows:
        raise ValueError(f"expert_execution_timeline_rows:{rows}!={expected_rows}")
    times = np.asarray(table.column("simulation_time_s").to_numpy(), dtype=np.float64)
    if not np.isfinite(times).all() or (times.size >= 2 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("expert_execution_timeline_time_invalid")
    dt_s = float(np.median(np.diff(times))) if times.size >= 2 else EXECUTION_DT_S
    if times.size >= 2 and abs(dt_s - EXECUTION_DT_S) > 1e-3:
        raise ValueError(f"expert_execution_timeline_dt_mismatch:{dt_s}")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("expert_execution_timeline_dt_invalid")
    records = tuple(table.to_pylist())
    ticks = [int(row.get("tick", -1)) for row in records]
    if ticks != list(range(len(ticks))):
        raise ValueError("expert_execution_timeline_tick_sequence_invalid")
    return timeline_path, actual_hash, rows, dt_s, records


def _validate_expert_outcome(
    base_root: Path,
    pair_id: str,
    proposal_sha: str,
    branch: Mapping[str, Any],
) -> tuple[Path, str, str, tuple[str, ...], dict[str, Any]]:
    """Bind the independent outcome label and classify execution quality.

    ``expert`` and ``GUARD_ELIGIBLE`` are provenance facts, not proof that the
    executed trajectory is a usable demonstration.  The C2 label artifact and
    branch summary are checked for identity and for the failure flags that can
    invalidate an interval.  Invalid intervals are retained in the manifest and
    represented by masks rather than repaired into a normal driving target.
    """

    paths = branch.get("artifact_paths", {})
    hashes = branch.get("artifact_sha256", {})
    raw_path = str(paths.get("label", ""))
    expected_hash = str(hashes.get("label", ""))
    if not raw_path or not expected_hash:
        raise ValueError("expert_outcome_label_reference_missing")
    label_path = base_root / raw_path
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    actual_hash = sha256_file(label_path)
    if actual_hash != expected_hash:
        raise ValueError("expert_outcome_label_hash_mismatch")
    label = _load_json(label_path)
    if str(label.get("schema_version", "")) != "safedrive.cora.outcome_labels.v1":
        raise ValueError("expert_outcome_label_schema_mismatch")
    if str(label.get("root_id", "")) != pair_id or str(label.get("proposal_sha256", "")) != proposal_sha:
        raise ValueError("expert_outcome_label_identity_mismatch")
    label_heads = label.get("heads", {})
    branch_heads = branch.get("heads", {})
    required = ("collision", "offroad", "mrm", "fallback_needed", "safety_executable", "guard_eligible")
    reasons: list[str] = []
    for key in required:
        item = label_heads.get(key)
        if not isinstance(item, Mapping) or item.get("valid") is not True:
            raise ValueError(f"expert_outcome_label_head_invalid:{key}")
        branch_item = branch_heads.get(key)
        if not isinstance(branch_item, Mapping) or branch_item.get("valid") is not True:
            raise ValueError(f"expert_branch_head_invalid:{key}")
        if item.get("value") != branch_item.get("value"):
            raise ValueError(f"expert_outcome_branch_head_mismatch:{key}")
        if bool(item.get("value")) and key not in {"safety_executable", "guard_eligible"}:
            reasons.append(f"{key}=true")
    terminal = str(branch.get("terminal_reason", ""))
    if terminal and terminal != "HORIZON_COMPLETE":
        reasons.append(f"terminal:{terminal}")
    if not bool(branch.get("legal_terminal", False)):
        reasons.append("legal_terminal=false")
    if not bool(branch.get("cleanup_complete", False)):
        reasons.append("cleanup_complete=false")
    quality = "verified_nominal_execution" if not reasons else "terminated_or_intervened"
    return label_path, actual_hash, quality, tuple(sorted(set(reasons))), label_heads


def build_sft_manifest(
    release_root: Path | str,
    *,
    splits: Sequence[str] = TARGET_SPLITS,
) -> tuple[tuple[SFTSample, ...], dict[str, Any]]:
    """Build the immutable C3 SFT view from the existing C2 release."""

    release_root = Path(release_root)
    release_path = release_root / "release-index.json"
    if not release_path.is_file():
        raise FileNotFoundError(release_path)
    release = _load_json(release_path)
    if release.get("dataset_id") != RELEASE_ID or release.get("quality_profile") != QUALITY_PROFILE:
        raise ValueError("c3_release_identity_mismatch")
    if release.get("status") != "DEV_DATA_READY":
        raise ValueError("c3_release_not_ready")
    base_root = _resolve_base_root(release_root, release)
    if not base_root.is_dir():
        raise FileNotFoundError(base_root)
    wanted = tuple(str(item) for item in splits)
    # Ask the fail-closed C2 loader to resolve each permitted split first.  The
    # release index is still the immutable source of row order/metadata, while
    # this check prevents a caller from accidentally widening the training or
    # evaluation scope when constructing the SFT view.
    released_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for split in wanted:
        purpose = "training" if split == "train" else "evaluation" if split == "validation" else "audit"
        rows = tuple(
            load_cora_roots(
                release_root,
                splits=(split,),
                purpose=purpose,
                quality_profile=QUALITY_PROFILE,
            )
        )
        if len({str(row.get("pair_id", "")) for row in rows}) != len(rows):
            raise ValueError(f"c3_release_duplicate_pair_ids:{split}")
        released_rows[split] = rows
    release_ids = {
        split: {str(row.get("pair_id", "")) for row in rows}
        for split, rows in released_rows.items()
    }
    if len(set().union(*release_ids.values())) != sum(len(ids) for ids in release_ids.values()):
        raise ValueError("c3_release_cross_split_root_overlap")
    samples: list[SFTSample] = []
    failures: list[dict[str, str]] = []
    for item in release.get("samples", ()):
        split = str(item.get("split", ""))
        if split not in wanted:
            continue
        pair_id = str(item.get("pair_id", ""))
        if pair_id not in release_ids.get(split, set()):
            raise ValueError(f"c3_release_loader_row_mismatch:{split}:{pair_id}")
        try:
            anchor_path = base_root / "anchors" / f"{pair_id}.json"
            pair_path = base_root / "pairs" / f"{pair_id}.json"
            anchor = _load_json(anchor_path)
            pair = _load_json(pair_path)
            found = _find_expert_branch(pair)
            proposal = found["proposal"]
            branch = found["branch"]
            proposal_sha = str(proposal.get("proposal_sha256", ""))
            proposal_path = base_root / "proposals" / f"{pair_id}__{proposal_sha}.json"
            proposal_file = _load_json(proposal_path)
            snapshot = anchor.get("observable_snapshot", {})
            if str(proposal_file.get("audit_source")) != "expert":
                raise ValueError("teacher_source_not_expert")
            if str(proposal_file.get("status")) != "GUARD_ELIGIBLE":
                raise ValueError("expert_proposal_not_guard_eligible")
            if str(branch.get("guard_verdict")) not in {"PASS", "REVIEW"}:
                raise ValueError("expert_branch_not_eligible")
            if not branch.get("identity_valid") or not branch.get("outcome_valid"):
                raise ValueError("expert_branch_identity_or_outcome_invalid")
            if branch.get("auxiliary_only"):
                raise ValueError("expert_branch_auxiliary_only")
            if str(branch.get("proposal_sha256", "")) != proposal_sha:
                raise ValueError("expert_branch_proposal_binding_mismatch")
            outcome_label_path, outcome_label_sha, execution_quality, execution_failure_reasons, outcome_heads = _validate_expert_outcome(
                base_root, pair_id, proposal_sha, branch
            )
            timeline_path, timeline_sha, timeline_rows, timeline_dt_s, timeline_records = _validate_expert_timeline(base_root, branch)
            image_path = base_root / str(anchor.get("image_path", ""))
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            expected_image_hash = str(anchor.get("image_sha256", ""))
            actual_image_hash = sha256_file(image_path)
            if expected_image_hash and actual_image_hash != expected_image_hash:
                raise ValueError("anchor_image_hash_mismatch")
            ego_x = float(snapshot["ego_x"])
            ego_y = float(snapshot["ego_y"])
            ego_yaw = float(snapshot["ego_yaw"])
            if "ego_v" not in snapshot:
                raise ValueError("anchor_speed_missing")
            ego_speed_mps = float(snapshot["ego_v"])
            if not math.isfinite(ego_speed_mps) or ego_speed_mps < 0.0:
                raise ValueError("anchor_speed_invalid")
            route_target, route_mask, route_report = native_route_target_with_report(
                anchor.get("route", ()), ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw
            )
            # Validate the canonical proposal's declared ten-point time grid.
            # The execution timeline remains independent outcome evidence; its
            # first timestamp includes a C2 branch pre-roll and is not silently
            # treated as an anchor-relative speed label.
            _canonical_speed, canonical_mask, canonical_failures = native_speed_target(
                proposal_file.get("trajectory", ()), ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw
            )
            if canonical_failures or not canonical_mask.all():
                raise ValueError(";".join(canonical_failures or ["canonical_speed_target_incomplete"]))
            _timeline_speed, _timeline_mask, timeline_failures = native_speed_target_from_timeline(
                timeline_records,
                ego_x=ego_x,
                ego_y=ego_y,
                ego_yaw=ego_yaw,
                execution_dt_s=timeline_dt_s,
            )
            speed_target = _canonical_speed
            speed_mask = canonical_mask.copy()
            speed_failures = list(timeline_failures)
            if execution_failure_reasons:
                speed_mask[:] = False
                speed_failures.extend(f"execution_untrusted:{reason}" for reason in execution_failure_reasons)
            if not route_mask.any():
                raise ValueError("expert_route_target_empty")
            # Navigation target construction is the same implementation used by
            # the deployment policy.  Import lazily to keep CPU audits light.
            from driving_vla.model.simlingo_contract import SimLingoContractConfig, navigation_targets

            target = navigation_targets(
                anchor.get("route", ()),
                ego_x=ego_x,
                ego_y=ego_y,
                ego_yaw=ego_yaw,
                progress_hint_s=0.0,
                config=SimLingoContractConfig(official_contract=True),
            )
            provenance = proposal_file.get("provenance", {})
            teacher_generator = str(provenance.get("generator_id", ""))
            if teacher_generator != "classic-frenet-st@h1":
                raise ValueError(f"expert_generator_untrusted:{teacher_generator}")
            from driving_vla.hybrid.generators import route_revision_sha256

            route_revision = route_revision_sha256(
                tuple(tuple(float(v) for v in row) for row in anchor.get("route", ()))
            )
            if str(provenance.get("route_revision", "")) != route_revision:
                raise ValueError("expert_route_revision_mismatch")
            anchor_time = float(snapshot.get("simulation_time_s"))
            timeline_start_time = float(timeline_records[0].get("simulation_time_s")) if timeline_records else float("nan")
            timeline_offset = timeline_start_time - anchor_time
            timeline_alignment = (
                "aligned_first_post_anchor"
                if math.isfinite(timeline_offset) and abs(timeline_offset - timeline_dt_s) <= max(0.1, timeline_dt_s)
                else "unaligned_branch_preroll"
            )
            history = anchor.get("observable_history", ())
            history_speed = None
            if history:
                try:
                    history_speed = float(history[-1]["ego_speed_mps"])
                except (KeyError, TypeError, ValueError):
                    history_speed = None
            speed_disagreement = None if history_speed is None else float(history_speed - ego_speed_mps)
            missing_route_reasons: list[str] = []
            if not route_mask.all():
                missing_route_reasons.extend(
                    (
                        f"route_points_available:{int(route_mask.sum())}/{ROUTE_STEPS}",
                        f"route_support_shortfall_m:{max(0.0, ROUTE_STEPS - float(route_report['support_m'])):.6f}",
                    )
                )
            if timeline_alignment != "aligned_first_post_anchor":
                speed_failures.append(f"timeline_alignment:{timeline_alignment}:{timeline_offset:.6f}s")
            samples.append(
                SFTSample(
                    root_id=pair_id,
                    pair_id=pair_id,
                    split=split,
                    family=str(item.get("family", "")),
                    map_name=str(item.get("map", "")),
                    weather=str(item.get("weather", "")),
                    seed=int(item.get("seed", 0)),
                    base_root=str(base_root),
                    anchor_path=str(anchor_path),
                    image_path=str(image_path),
                    image_sha256=actual_image_hash,
                    proposal_path=str(proposal_path),
                    proposal_sha256=proposal_sha,
                    teacher_source=str(provenance.get("source", "expert")),
                    teacher_generator=teacher_generator,
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_yaw=ego_yaw,
                    ego_speed_mps=ego_speed_mps,
                    target_ego_1=tuple(float(v) for v in target.target_ego_1),
                    target_ego_2=tuple(float(v) for v in target.target_ego_2),
                    route_target=tuple(tuple(float(v) for v in row) for row in route_target),
                    route_mask=tuple(bool(v) for v in route_mask),
                    speed_target=tuple(tuple(float(v) for v in row) for row in speed_target),
                    speed_mask=tuple(bool(v) for v in speed_mask),
                    speed_source="expert_canonical_proposal",
                    execution_timeline_path=str(timeline_path),
                    execution_timeline_sha256=timeline_sha,
                    execution_timeline_rows=timeline_rows,
                    execution_timeline_dt_s=timeline_dt_s,
                    route_missing_reasons=tuple(missing_route_reasons),
                    speed_missing_reasons=tuple(speed_failures),
                    route_projection_s_m=float(route_report["projection_s_m"]),
                    route_projection_distance_m=float(route_report["projection_distance_m"]),
                    route_support_m=float(route_report["support_m"]),
                    route_projection_segment=int(route_report["projection_segment"]),
                    route_projection_ambiguous=bool(route_report["projection_ambiguous"]),
                    route_reference_revision=route_revision,
                    route_validity_basis="native_expert_reference_path_audited",
                    execution_quality=execution_quality,
                    execution_failure_reasons=execution_failure_reasons,
                    outcome_label_path=str(outcome_label_path),
                    outcome_label_sha256=outcome_label_sha,
                    outcome_label_heads=tuple((str(k), dict(v)) for k, v in sorted(outcome_heads.items())),
                    execution_timeline_start_offset_s=float(timeline_offset),
                    execution_timeline_alignment=timeline_alignment,
                    execution_timeline_speed_usable=False,
                    speed_input_source="anchor.observable_snapshot.ego_v",
                    anchor_history_speed_mps=history_speed,
                    anchor_history_speed_disagreement_mps=speed_disagreement,
                )
            )
        except Exception as exc:  # noqa: BLE001 - audit records each root failure
            failures.append({"pair_id": pair_id, "split": split, "reason": f"{type(exc).__name__}:{exc}"})
    samples.sort(key=lambda sample: (sample.split, sample.root_id))
    split_counts = {split: sum(sample.split == split for sample in samples) for split in wanted}
    manifest = {
        "schema_version": "safedrive.c3.sft_manifest.v2",
        "release_id": RELEASE_ID,
        "quality_profile": QUALITY_PROFILE,
        "base_root": str(base_root),
        "splits": list(wanted),
        "sample_count": len(samples),
        "split_counts": split_counts,
        "failure_count": len(failures),
        "failures": failures,
        "route_steps": ROUTE_STEPS,
        "route_spacing_m": 1.0,
        "speed_steps": SPEED_STEPS,
        "speed_dt_s": SPEED_DT_S,
        "teacher_contract": "native_expert_reference_path_plus_canonical_speed_proposal",
        "route_label_contract": {
            "source": "native_expert_reference_path_projected",
            "projection": "closest_ordered_segment_then_forward_suffix",
            "spacing_m": 1.0,
            "arc_queries_m": list(range(ROUTE_STEPS)),
            "no_navigation_splice": True,
            "no_extrapolation": True,
        },
        "speed_label_contract": {
            "source": "expert_canonical_proposal",
            "steps": SPEED_STEPS,
            "dt_s": SPEED_DT_S,
            "execution_timeline_audit_only": True,
            "unaligned_timeline_is_not_used_as_anchor_speed": True,
        },
        "anchor_speed_input_contract": "observable_snapshot.ego_v; missing_is_error; zero_is_preserved",
        "route_projection_max_distance_m": ROUTE_PROJECTION_MAX_DISTANCE_M,
        "future_labels_in_input": False,
        "world_labels_in_sft": False,
        "samples": [sample.to_dict() for sample in samples],
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return tuple(samples), manifest


def deterministic_order(samples: Sequence[SFTSample], seed: int = 17) -> list[SFTSample]:
    """Return a deterministic root order without replacement per epoch."""

    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    return ordered


def masked_smooth_l1(prediction: Any, target: Any, mask: Any) -> Any:
    """Torch-free contract helper; the training script supplies torch tensors."""

    import torch
    import torch.nn.functional as F

    pred = prediction
    truth = target
    valid = mask.to(dtype=torch.bool)
    if pred.shape != truth.shape or pred.shape[:2] != valid.shape:
        raise ValueError(f"masked_loss_shape:{tuple(pred.shape)}:{tuple(truth.shape)}:{tuple(valid.shape)}")
    element = F.smooth_l1_loss(pred, truth, reduction="none")
    expanded = valid.unsqueeze(-1).expand_as(element)
    count = expanded.sum()
    if int(count.item()) == 0:
        raise ValueError("masked_loss_no_valid_elements")
    return element.masked_select(expanded).mean()


def root_metrics(
    prediction: Mapping[str, Any],
    sample: SFTSample,
) -> dict[str, Any]:
    """Calculate one root's route and speed metrics from numpy predictions."""

    route = np.asarray(prediction.get("route"), dtype=np.float64)
    speed = np.asarray(prediction.get("speed"), dtype=np.float64)
    route_target = np.asarray(sample.route_target, dtype=np.float64)
    speed_target = np.asarray(sample.speed_target, dtype=np.float64)
    route_mask = np.asarray(sample.route_mask, dtype=bool)
    speed_mask = np.asarray(sample.speed_mask, dtype=bool)
    route_shape_valid = route.shape == route_target.shape
    speed_shape_valid = speed.shape == speed_target.shape
    finite_route = np.isfinite(route).all(axis=1) if route_shape_valid else np.zeros(ROUTE_STEPS, dtype=bool)
    finite_speed = np.isfinite(speed).all(axis=1) if speed_shape_valid else np.zeros(SPEED_STEPS, dtype=bool)
    route_valid = route_mask & finite_route
    speed_valid = speed_mask & finite_speed
    route_error = np.linalg.norm(route - route_target, axis=1) if route_shape_valid else np.array([])
    speed_error = np.linalg.norm(speed - speed_target, axis=1) if speed_shape_valid else np.array([])
    route_fde_target_valid = bool(route_mask.size == ROUTE_STEPS and route_mask[-1])
    route_fde_valid = bool(route_shape_valid and route_fde_target_valid and finite_route[-1])
    canonical_output_valid = bool(
        prediction.get("canonical_output_valid", route_shape_valid and speed_shape_valid and finite_route.all() and finite_speed.all())
    )
    failure_reasons: list[str] = []
    if not route_shape_valid:
        failure_reasons.append(f"route_shape:{route.shape}")
    if not speed_shape_valid:
        failure_reasons.append(f"speed_shape:{speed.shape}")
    if route_shape_valid and not finite_route.all():
        failure_reasons.append("route_non_finite")
    if speed_shape_valid and not finite_speed.all():
        failure_reasons.append("speed_non_finite")
    if not route_valid.any():
        failure_reasons.append("route_no_valid_points")
    if not speed_valid.any():
        failure_reasons.append("speed_no_valid_points")
    if not canonical_output_valid:
        failure_reasons.append("canonical_output_invalid")
    if not route_fde_valid:
        failure_reasons.append("route_fde_point20_unavailable")
    result: dict[str, Any] = {
        "root_id": sample.root_id,
        "split": sample.split,
        "route_valid_points": int(route_valid.sum()),
        "speed_valid_points": int(speed_valid.sum()),
        "route_shape_valid": route_shape_valid,
        "speed_shape_valid": speed_shape_valid,
        "route_finite": bool(route_shape_valid and finite_route.all()),
        "speed_finite": bool(speed_shape_valid and finite_speed.all()),
        "canonical_output_valid": canonical_output_valid,
        "route_full_valid": bool(route_mask.size == ROUTE_STEPS and route_mask.all() and route_shape_valid and finite_route.all()),
        "speed_full_valid": bool(speed_mask.size == SPEED_STEPS and speed_mask.all() and speed_shape_valid and finite_speed.all()),
        "route_fde_target_valid": route_fde_target_valid,
        "route_fde_valid": route_fde_valid,
        "route_ade_m": float(route_error[route_valid].mean()) if route_valid.any() else None,
        "route_fde_m": float(route_error[ROUTE_STEPS - 1]) if route_fde_valid else None,
        "speed_wp_ade_m": float(speed_error[speed_valid].mean()) if speed_valid.any() else None,
        "prediction_valid": bool(canonical_output_valid and route_valid.any() and speed_valid.any()),
        "failure_reasons": sorted(set(failure_reasons)),
        "route_missing_reasons": list(sample.route_missing_reasons),
        "speed_missing_reasons": list(sample.speed_missing_reasons),
        "route_source": sample.route_source,
        "speed_source": sample.speed_source,
        "execution_timeline_rows": int(sample.execution_timeline_rows),
        "execution_timeline_dt_s": sample.execution_timeline_dt_s,
    }
    return result


def aggregate_metrics(
    per_root: Sequence[Mapping[str, Any]],
    *,
    baseline_key: str = "m0",
    method_key: str = "m1",
    bootstrap_rounds: int = 1000,
    bootstrap_seed: int = 71,
) -> dict[str, Any]:
    """Root-equal aggregate and paired bootstrap for M0/M1 records."""

    groups: dict[str, list[Mapping[str, Any]]] = {baseline_key: [], method_key: []}
    for row in per_root:
        if row.get("model") in groups:
            groups[str(row["model"])].append(row)
    by_root: dict[str, dict[str, Mapping[str, Any]]] = {}
    for model, rows in groups.items():
        for row in rows:
            by_root.setdefault(str(row["root_id"]), {})[model] = row
    paired = [pair for pair in by_root.values() if baseline_key in pair and method_key in pair]
    rng = np.random.default_rng(bootstrap_seed)

    def mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        return float(np.mean(vals)) if vals else None

    metrics: dict[str, Any] = {}
    for key in ("route_ade_m", "route_fde_m", "speed_wp_ade_m"):
        base = mean_metric(groups[baseline_key], key)
        method = mean_metric(groups[method_key], key)
        delta = None if base is None or method is None else base - method
        relative = None if delta is None or abs(base or 0.0) < 1e-12 else 100.0 * delta / abs(base)
        diffs = np.asarray(
            [float(pair[baseline_key][key]) - float(pair[method_key][key]) for pair in paired
             if pair[baseline_key].get(key) is not None and pair[method_key].get(key) is not None],
            dtype=np.float64,
        )
        if diffs.size >= 2:
            draws = rng.integers(0, diffs.size, size=(bootstrap_rounds, diffs.size))
            boot = diffs[draws].mean(axis=1)
            ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        else:
            ci = None
        metrics[key] = {
            "m0": base,
            "m1": method,
            "delta_m0_minus_m1": delta,
            "relative_improvement_percent": relative,
            "paired_root_count": int(diffs.size),
            "bootstrap_ci95": ci,
        }
    metrics["root_count"] = {baseline_key: len(groups[baseline_key]), method_key: len(groups[method_key])}
    metrics["denominator_root_count"] = {model: len(rows) for model, rows in groups.items()}
    metrics["prediction_valid_count"] = {
        model: int(sum(bool(row.get("prediction_valid")) for row in rows)) for model, rows in groups.items()
    }
    metrics["prediction_valid_rate_percent"] = {
        model: (100.0 * sum(bool(row.get("prediction_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in groups.items()
    }
    metrics["prediction_fail_rate_percent"] = {
        model: (100.0 * sum(not bool(row.get("prediction_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in groups.items()
    }
    metrics["route_full_valid_rate_percent"] = {
        model: (100.0 * sum(bool(row.get("route_full_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in groups.items()
    }
    metrics["speed_full_valid_rate_percent"] = {
        model: (100.0 * sum(bool(row.get("speed_full_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in groups.items()
    }
    metrics["failure_reason_counts"] = {
        model: {
            str(reason): int(sum(str(reason) in row.get("failure_reasons", ()) for row in rows))
            for reason in sorted({str(item) for row in rows for item in row.get("failure_reasons", ())})
        }
        for model, rows in groups.items()
    }
    return {
        "schema_version": "safedrive.c3.metrics.v2",
        "aggregation_unit": "root_equal",
        "bootstrap_rounds": bootstrap_rounds,
        "bootstrap_seed": bootstrap_seed,
        "metrics": metrics,
        "paired_root_count": len(paired),
    }


__all__ = [
    "RELEASE_ID",
    "QUALITY_PROFILE",
    "ROUTE_STEPS",
    "ROUTE_PROJECTION_MAX_DISTANCE_M",
    "ROUTE_PROJECTION_TIE_DISTANCE_M",
    "ROUTE_PROJECTION_TIE_PROGRESS_M",
    "SPEED_STEPS",
    "SPEED_DT_S",
    "EXECUTION_DT_S",
    "SFTSample",
    "aggregate_metrics",
    "build_sft_manifest",
    "deterministic_order",
    "map_to_ego",
    "masked_smooth_l1",
    "native_route_target",
    "native_route_target_with_report",
    "native_speed_target",
    "native_speed_target_from_timeline",
    "root_metrics",
    "sample_polyline",
    "sha256_file",
]
