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


def native_route_target(
    route_map: Sequence[Sequence[float]],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    steps: int = ROUTE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the native 1 m-spaced route head target in ego coordinates."""

    if steps <= 0:
        raise ValueError("steps_must_be_positive")
    # SimLingo's native equal_spacing_route prepends the current ego origin
    # before interpolating the route.  Keeping that origin is material when a
    # recorded route starts a few metres behind the observation (the common
    # CARLA route representation); sampling the raw route directly would
    # shift every target and teach the wrong head contract.
    route = np.asarray(route_map, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2 or route.shape[0] < 1:
        raise ValueError("route_requires_one_point")
    origin = np.asarray([[float(ego_x), float(ego_y)]], dtype=np.float64)
    sampled_map, mask = sample_polyline(
        np.concatenate((origin, route), axis=0),
        np.arange(steps, dtype=np.float64),
    )
    return map_to_ego(sampled_map, ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw), mask


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
    route_source: str = "anchor.route_with_ego_origin"
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
    table = pq.read_table(timeline_path, columns=["tick", "simulation_time_s", "x", "y"])
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
            route_target, route_mask = native_route_target(
                anchor.get("route", ()), ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw
            )
            # Validate the canonical proposal's declared ten-point time grid,
            # but use the separately recorded execution timeline for actual
            # SFT speed labels.  Missing/early-terminal timeline rows remain
            # masked and do not discard the root.
            _canonical_speed, canonical_mask, canonical_failures = native_speed_target(
                proposal_file.get("trajectory", ()), ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw
            )
            if canonical_failures or not canonical_mask.all():
                raise ValueError(";".join(canonical_failures or ["canonical_speed_target_incomplete"]))
            speed_target, speed_mask, speed_failures = native_speed_target_from_timeline(
                timeline_records,
                ego_x=ego_x,
                ego_y=ego_y,
                ego_yaw=ego_yaw,
                execution_dt_s=timeline_dt_s,
            )
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
                    teacher_generator=str(provenance.get("generator_id", "")),
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_yaw=ego_yaw,
                    ego_speed_mps=max(0.0, float(snapshot.get("ego_v", 0.0))),
                    target_ego_1=tuple(float(v) for v in target.target_ego_1),
                    target_ego_2=tuple(float(v) for v in target.target_ego_2),
                    route_target=tuple(tuple(float(v) for v in row) for row in route_target),
                    route_mask=tuple(bool(v) for v in route_mask),
                    speed_target=tuple(tuple(float(v) for v in row) for row in speed_target),
                    speed_mask=tuple(bool(v) for v in speed_mask),
                    speed_source="expert_execution_timeline",
                    execution_timeline_path=str(timeline_path),
                    execution_timeline_sha256=timeline_sha,
                    execution_timeline_rows=timeline_rows,
                    execution_timeline_dt_s=timeline_dt_s,
                    route_missing_reasons=(
                        (f"route_points_available:{int(route_mask.sum())}/{ROUTE_STEPS}",)
                        if not route_mask.all()
                        else ()
                    ),
                    speed_missing_reasons=tuple(speed_failures),
                )
            )
        except Exception as exc:  # noqa: BLE001 - audit records each root failure
            failures.append({"pair_id": pair_id, "split": split, "reason": f"{type(exc).__name__}:{exc}"})
    samples.sort(key=lambda sample: (sample.split, sample.root_id))
    split_counts = {split: sum(sample.split == split for sample in samples) for split in wanted}
    manifest = {
        "schema_version": "safedrive.c3.sft_manifest.v1",
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
        "teacher_contract": "expert_nominal_proposal_only",
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
    finite_route = route.shape == route_target.shape and np.isfinite(route).all(axis=1)
    finite_speed = speed.shape == speed_target.shape and np.isfinite(speed).all(axis=1)
    route_valid = route_mask & finite_route
    speed_valid = speed_mask & finite_speed
    route_error = np.linalg.norm(route - route_target, axis=1) if route.shape == route_target.shape else np.array([])
    speed_error = np.linalg.norm(speed - speed_target, axis=1) if speed.shape == speed_target.shape else np.array([])
    result: dict[str, Any] = {
        "root_id": sample.root_id,
        "split": sample.split,
        "route_valid_points": int(route_valid.sum()),
        "speed_valid_points": int(speed_valid.sum()),
        "route_finite": bool(route.shape == route_target.shape and finite_route.all()),
        "speed_finite": bool(speed.shape == speed_target.shape and finite_speed.all()),
        "route_ade_m": float(route_error[route_valid].mean()) if route_valid.any() else None,
        "route_fde_m": float(route_error[np.flatnonzero(route_valid)[-1]]) if route_valid.any() else None,
        "speed_wp_ade_m": float(speed_error[speed_valid].mean()) if speed_valid.any() else None,
        "prediction_valid": bool(route_valid.any() and speed_valid.any()),
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
    metrics["prediction_fail_rate_percent"] = {
        model: (100.0 * sum(not bool(row.get("prediction_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in groups.items()
    }
    return {
        "schema_version": "safedrive.c3.metrics.v1",
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
    "native_speed_target",
    "native_speed_target_from_timeline",
    "root_metrics",
    "sample_polyline",
    "sha256_file",
]
