"""Observable-only H3v2 feature extraction, grouped split and leakage audits.

Feature tensors are constructed from an allow-list only.  The richer H2/Challenge
record (source, slot, future, labels) is used solely for labels and audit metrics
and is never serialized into the model input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safety_kernel.config import SafetyKernelConfig
from safety_kernel.contracts.serialize import _candidate_from_dict, _point_from_dict
from safety_kernel.contracts.types import (
    ObservableSnapshot,
    TrackedObject,
    TrafficLightObs,
)
from safety_kernel.arbitration.soft_score import score_candidate

from .contracts import (
    H3_ACTOR_SLOTS,
    H3_CONFIG_SHA256,
    H3_CONTEXT_DIM,
    H3_CANDIDATE_DIM,
    H3_CANDIDATE_STEPS,
    H3_HISTORY_TICKS,
    H3_LIGHT_SLOTS,
    H3_ROUTE_POINTS,
    H3_SCHEMA_VERSION,
    SplitRow,
    stable_sha256,
)


class H3DatasetError(ValueError):
    """Raised when an H3 input violates its immutable data boundary."""


FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "source", "slot", "guard", "provenance", "branch_order", "actor_future",
        "outcome", "oracle", "winner", "label", "regression", "future",
        "image_path", "image_sha256", "wall_time", "carla_frame",
        "observation_id", "candidate_id", "canonical_sha256",
    }
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def lineage_key(row: Mapping[str, Any]) -> str:
    scenario = row["scenario"]
    return f"{scenario['map_name']}|{scenario['family']}|{int(scenario['seed'])}"


def _split_lineages(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, str]]:
    lineages = sorted({lineage_key(row) for row in rows})
    grouped: dict[tuple[str, str], list[str]] = {}
    for lineage in lineages:
        map_name, family, _ = lineage.split("|")
        grouped.setdefault((map_name, family), []).append(lineage)
    result: dict[str, tuple[int, str]] = {}
    for group, items in sorted(grouped.items()):
        ranked = sorted(items, key=lambda value: _sha(f"sdf-h3-lineage-split-v2|{value}"))
        for rank, lineage in enumerate(ranked):
            result[lineage] = (rank, "test" if rank == 0 else f"dev_fold_{rank}")
    return result


def build_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    physical_manifest_sha256: str,
    store_manifest_sha256: str,
    challenge_physical_manifest_sha256: str | None = None,
    challenge_store_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    assignments = _split_lineages(records)
    # Exact numeric feature duplicates share a split (test has priority).
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    feature_owner: dict[str, str] = {}
    for record in records:
        lineage = lineage_key(record)
        find(lineage)
        for candidate in record.get("candidates", ()):
            digest = stable_sha256({"context": _context_vector(record), "candidate": _candidate_tensor(record, candidate)})
            if digest in feature_owner:
                union(lineage, feature_owner[digest])
            else:
                feature_owner[digest] = lineage
    components: dict[str, list[str]] = {}
    for lineage in {lineage_key(record) for record in records}:
        components.setdefault(find(lineage), []).append(lineage)
    duplicate_grouped = 0
    for component in components.values():
        if len(component) <= 1:
            continue
        ranks = [assignments[item][0] for item in component]
        target_rank = min(ranks)
        target_split = "test" if target_rank == 0 else f"dev_fold_{target_rank}"
        if len({assignments[item][1] for item in component}) > 1:
            duplicate_grouped += len(component)
        for item in component:
            assignments[item] = (target_rank, target_split)
    rows: list[SplitRow] = []
    for record in sorted(records, key=lambda item: str(item["pair_id"])):
        scenario = record["scenario"]
        lineage = lineage_key(record)
        rank, split = assignments[lineage]
        rows.append(
            SplitRow(
                pair_id=str(record["pair_id"]),
                map_name=str(scenario["map_name"]),
                family=str(scenario["family"]),
                seed=int(scenario["seed"]),
                weather=str(scenario["weather"]),
                lineage=lineage,
                lineage_rank=rank,
                split=split,
                valid_pair=str(record.get("terminal_status")) in {"VALID_PAIR", "COMPLETED"} and len(record.get("branches", ())) == 2,
            )
        )
    payload: dict[str, Any] = {
        "schema_version": H3_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "physical_manifest_sha256": physical_manifest_sha256,
        "store_manifest_sha256": store_manifest_sha256,
        "h3_config_sha256": H3_CONFIG_SHA256,
        "lineage_definition": ["map", "family", "seed"],
        "weather_together": True,
        "duplicate_lineages_grouped": duplicate_grouped,
        "rows": [row.to_dict() for row in rows],
    }
    if challenge_physical_manifest_sha256 is not None:
        payload["challenge_physical_manifest_sha256"] = challenge_physical_manifest_sha256
    if challenge_store_manifest_sha256 is not None:
        payload["challenge_store_manifest_sha256"] = challenge_store_manifest_sha256
    payload["manifest_sha256"] = stable_sha256(payload)
    return payload


def write_split_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_parquet_record(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise H3DatasetError("pyarrow_required") from exc
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1:
        raise H3DatasetError(f"pair_shard_not_single_row:{path.name}")
    row = rows[0]
    record = json.loads(row["record_json"])
    expected = str(row.get("record_sha256", ""))
    if expected and hashlib.sha256(row["record_json"].encode("utf-8")).hexdigest() != expected:
        raise H3DatasetError(f"record_sha_mismatch:{path.name}")
    if record.get("content_sha256"):
        body = {key: value for key, value in record.items() if key != "content_sha256"}
        if hashlib.sha256(json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() != record["content_sha256"]:
            raise H3DatasetError(f"content_sha_mismatch:{path.name}")
    return record


def _roots_to_list(root: Path | Sequence[Path]) -> list[Path]:
    if isinstance(root, (list, tuple, set)):
        return [Path(p) for p in root]
    return [Path(root)]


def read_h2_records(root: Path | Sequence[Path]) -> list[dict[str, Any]]:
    roots = _roots_to_list(root)
    records: list[dict[str, Any]] = []
    for r in roots:
        records.extend([_read_parquet_record(path) for path in sorted((r / "pairs").glob("*.parquet"))])
    if not records:
        raise H3DatasetError(f"no_h2_pair_shards:{roots}")
    return records


def read_h2_labels(root: Path | Sequence[Path], *, allowed_pair_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Read labels only for development pair ids; test labels are never opened."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise H3DatasetError("pyarrow_required") from exc
    roots = _roots_to_list(root)
    labels: dict[str, dict[str, Any]] = {}
    for r in roots:
        for path in sorted((r / "labels").glob("*.parquet")):
            pair_id = path.stem
            if pair_id not in allowed_pair_ids:
                continue
            rows = pq.read_table(path).to_pylist()
            if len(rows) != 1:
                raise H3DatasetError(f"label_shard_not_single_row:{path.name}")
            labels[pair_id] = json.loads(rows[0]["label_json"])
    missing = sorted(allowed_pair_ids - set(labels))
    if missing:
        raise H3DatasetError(f"missing_dev_labels:{missing[:4]}")
    return labels


def _rot_to_ego(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _candidate_tensor(record: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[list[float]]:
    snap = record["anchor"].get("observable_snapshot", {})
    ex, ey, eyaw = (_safe_float(snap.get("ego_x")), _safe_float(snap.get("ego_y")), _safe_float(snap.get("ego_yaw")))
    rows: list[list[float]] = []
    points = list(candidate.get("trajectory", ()))[: H3_CANDIDATE_STEPS]
    for point in points:
        dx, dy = _rot_to_ego(_safe_float(point.get("x")) - ex, _safe_float(point.get("y")) - ey, eyaw)
        rel_yaw = _safe_float(point.get("yaw")) - eyaw
        rows.append(
            [
                dx / 50.0,
                dy / 10.0,
                math.sin(rel_yaw),
                math.cos(rel_yaw),
                _safe_float(point.get("v")) / 10.0,
                _safe_float(point.get("a")) / 10.0,
                _safe_float(point.get("kappa")) / 0.5,
                _safe_float(point.get("t")) / 2.5,
            ]
        )
    while len(rows) < H3_CANDIDATE_STEPS:
        rows.append(list(rows[-1]) if rows else [0.0] * H3_CANDIDATE_DIM)
    return rows


def _context_vector(record: Mapping[str, Any]) -> list[float]:
    snap = record["anchor"].get("observable_snapshot", {})
    ex, ey, eyaw = (_safe_float(snap.get("ego_x")), _safe_float(snap.get("ego_y")), _safe_float(snap.get("ego_yaw")))
    values: list[float] = []

    # Ego history: 20 * 7.
    full_history = list(record.get("observable_history", ()))
    history = full_history[-H3_HISTORY_TICKS:] if len(full_history) > H3_HISTORY_TICKS else full_history
    for item in history:
        dx, dy = _rot_to_ego(_safe_float(item.get("ego_x")) - ex, _safe_float(item.get("ego_y")) - ey, eyaw)
        rel_yaw = _safe_float(item.get("ego_yaw")) - eyaw
        values.extend(
            [
                dx / 50.0,
                dy / 10.0,
                math.sin(rel_yaw),
                math.cos(rel_yaw),
                _safe_float(item.get("ego_speed_mps")) / 10.0,
                _safe_float(item.get("ego_acceleration_mps2")) / 10.0,
                (_safe_float(item.get("simulation_time_s")) - _safe_float(snap.get("simulation_time_s"))) / 2.0,
            ]
        )
    while len(history) < H3_HISTORY_TICKS:
        values.extend([0.0] * 7)
        history.append({})

    # Route: 51 * 4 = dx, dy, segment sin/cos.
    route = list(record.get("route", ()))[: H3_ROUTE_POINTS]
    for index, point in enumerate(route):
        dx, dy = _rot_to_ego(_safe_float(point[0]) - ex, _safe_float(point[1]) - ey, eyaw)
        if index == 0:
            seg_dx, seg_dy = 2.0, 0.0
        else:
            seg_dx = _safe_float(point[0]) - _safe_float(route[index - 1][0])
            seg_dy = _safe_float(point[1]) - _safe_float(route[index - 1][1])
        seg_len = max(1e-6, math.hypot(seg_dx, seg_dy))
        values.extend([dx / 100.0, dy / 20.0, seg_dx / seg_len, seg_dy / seg_len])
    while len(route) < H3_ROUTE_POINTS:
        values.extend([0.0, 0.0, 1.0, 0.0])
        route.append((0.0, 0.0))

    # Actors: 8 * 12.
    actors = list(snap.get("actors", ()))[: H3_ACTOR_SLOTS]
    for actor in actors:
        dx, dy = _rot_to_ego(_safe_float(actor.get("x")) - ex, _safe_float(actor.get("y")) - ey, eyaw)
        rel_yaw = _safe_float(actor.get("yaw")) - eyaw
        vx, vy = _safe_float(actor.get("vx")), _safe_float(actor.get("vy"))
        values.extend(
            [
                dx / 50.0,
                dy / 20.0,
                vx / 10.0,
                vy / 10.0,
                math.sin(rel_yaw),
                math.cos(rel_yaw),
                _safe_float(actor.get("length_m")) / 10.0,
                _safe_float(actor.get("width_m")) / 4.0,
                1.0 if actor.get("lost") else 0.0,
                (_safe_float(actor.get("observed_time_s")) - _safe_float(snap.get("simulation_time_s"))) / 5.0,
                math.hypot(vx, vy) / 10.0,
                math.hypot(dx, dy) / 50.0,
            ]
        )
    values.extend([0.0] * (H3_ACTOR_SLOTS - len(actors)) * 12)

    # Traffic lights: 6 * 9.
    states = {"red": (1.0, 0.0, 0.0, 0.0), "yellow": (0.0, 1.0, 0.0, 0.0), "green": (0.0, 0.0, 1.0, 0.0)}
    lights = list(snap.get("traffic_lights", ()))[: H3_LIGHT_SLOTS]
    for light in lights:
        state = states.get(str(light.get("state", "")).lower(), (0.0, 0.0, 0.0, 1.0))
        values.extend(
            [
                _safe_float(light.get("distance_m")) / 100.0,
                _safe_float(light.get("stop_line_distance_m")) / 100.0,
                *state,
                1.0 if light.get("controls_ego_lane") else 0.0,
                1.0 if str(light.get("state", "")).lower() == "red" else 0.0,
                _safe_float(light.get("stop_line_distance_m")) / 100.0,
            ]
        )
    values.extend([0.0] * (H3_LIGHT_SLOTS - len(lights)) * 9)

    # Ego current state.
    values.extend(
        [
            _safe_float(snap.get("ego_v")) / 10.0,
            _safe_float(snap.get("ego_a")) / 10.0,
            _safe_float(snap.get("speed_limit_mps")) / 20.0,
            _safe_float(snap.get("corridor_half_width_m")) / 5.0,
            (_safe_float(snap.get("freshness_s"), 0.0) - 0.25) / 1.0,
        ]
    )
    if len(values) != H3_CONTEXT_DIM:
        raise H3DatasetError(f"context_dimension:{len(values)}!={H3_CONTEXT_DIM}")
    return values


def _hard_unsafe_from_branch(branch: Mapping[str, Any]) -> bool:
    return bool(branch.get("collision_count", 0)) or bool(branch.get("red_light_violation", False)) or float(branch.get("off_corridor_duration_s", 0.0)) > 0.25


def _snapshot_from_record(record: Mapping[str, Any]) -> ObservableSnapshot:
    snap = record["anchor"].get("observable_snapshot", {})
    actors = tuple(
        TrackedObject(
            actor_id=str(item.get("actor_id", item.get("id", ""))),
            class_name=str(item.get("class_name", "")),
            x=_safe_float(item.get("x")), y=_safe_float(item.get("y")), yaw=_safe_float(item.get("yaw")),
            vx=_safe_float(item.get("vx")), vy=_safe_float(item.get("vy")),
            length_m=_safe_float(item.get("length_m"), 4.0), width_m=_safe_float(item.get("width_m"), 2.0),
            observed_time_s=_safe_float(item.get("observed_time_s")),
            lost=bool(item.get("lost", False)), source=str(item.get("source", "observable")),
        )
        for item in snap.get("actors", ())
    )
    lights = tuple(
        TrafficLightObs(
            light_id=str(item.get("light_id", item.get("id", ""))),
            state=str(item.get("state", "unknown")),
            distance_m=_safe_float(item.get("distance_m")),
            observed_time_s=_safe_float(item.get("observed_time_s")),
            stop_line_distance_m=_safe_float(item.get("stop_line_distance_m")),
            controls_ego_lane=bool(item.get("controls_ego_lane")) if item.get("controls_ego_lane") is not None else None,
        )
        for item in snap.get("traffic_lights", ())
    )
    corridor = tuple((float(p[0]), float(p[1])) for p in snap.get("corridor_centerline", ()))
    return ObservableSnapshot(
        run_id=str(snap.get("run_id", "")),
        frame_id=str(snap.get("frame_id", "")),
        scenario_id=str(snap.get("scenario_id", "")),
        simulation_time_s=_safe_float(snap.get("simulation_time_s")),
        wall_time_s=_safe_float(snap.get("wall_time_s")),
        ego_x=_safe_float(snap.get("ego_x")), ego_y=_safe_float(snap.get("ego_y")), ego_yaw=_safe_float(snap.get("ego_yaw")),
        ego_v=_safe_float(snap.get("ego_v")), ego_a=_safe_float(snap.get("ego_a")),
        observed_time_s=_safe_float(snap.get("observed_time_s")),
        freshness_s=_safe_float(snap.get("freshness_s")),
        speed_limit_mps=_safe_float(snap.get("speed_limit_mps"), 0.0),
        actors=actors, traffic_lights=lights,
        corridor_centerline=corridor,
        corridor_half_width_m=_safe_float(snap.get("corridor_half_width_m"), 1.75),
        coordinate_frame=str(snap.get("coordinate_frame", "map")),
    )


def _h1_soft_score(record: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    """Audit-only H1 soft score computed from the full H2/Challenge record.

    This value is attached to CandidateExample for the H1-soft-selector
    baseline, but is deliberately absent from context/candidate feature
    tensors.
    """
    try:
        raw = dict(candidate)
        raw["points"] = tuple(raw.get("trajectory", ()))
        policy = _candidate_from_dict(raw, "candidate")
        snapshot = _snapshot_from_record(record)
        return float(score_candidate(policy, snapshot, SafetyKernelConfig()).total)
    except Exception:
        # Older/partial records cannot reconstruct Safety input; fall back to
        # the same progress/comfort-only deterministic soft score used by
        # FrozenH1Router (source-neutral path length and planned jerk).
        return _planned_length_from_points(candidate.get("trajectory", ())) - 0.35 * _planned_jerk_from_points(candidate.get("trajectory", ()))


def _planned_length_from_points(points: Sequence[Mapping[str, Any]]) -> float:
    pts = list(points)
    return sum(math.hypot(_safe_float(pts[i]["x"]) - _safe_float(pts[i - 1]["x"]), _safe_float(pts[i]["y"]) - _safe_float(pts[i - 1]["y"])) for i in range(1, len(pts)))


def _planned_jerk_from_points(points: Sequence[Mapping[str, Any]]) -> float:
    pts = list(points)
    jerks = []
    for i in range(1, len(pts)):
        a_curr, a_prev = _safe_float(pts[i]["a"]), _safe_float(pts[i - 1]["a"])
        dt = max(0.01, _safe_float(pts[i]["t"]) - _safe_float(pts[i - 1]["t"]))
        jerks.append((a_curr - a_prev) / dt)
    return math.sqrt(sum(j * j for j in jerks) / len(jerks)) if jerks else 0.0


@dataclass(frozen=True)
class CandidateExample:
    candidate_key: str
    context: tuple[float, ...]
    candidate: tuple[tuple[float, ...], ...]
    progress_m: float
    jerk_rms_mps3: float
    risk: bool
    h1_soft_score: float


@dataclass(frozen=True)
class PairExample:
    pair_id: str
    map_name: str
    family: str
    seed: int
    weather: str
    split: str
    candidates: tuple[CandidateExample, CandidateExample]
    winner_index: int | None
    tie: bool

    @property
    def decisive(self) -> bool:
        return self.winner_index is not None


def _make_pair_example(record: Mapping[str, Any], label: Mapping[str, Any], split: str) -> PairExample:
    candidates = {str(item["candidate_id"]): item for item in record.get("candidates", ())}
    branches = {str(item["candidate_id"]): item for item in record.get("branches", ())}
    if len(candidates) != 2 or set(candidates) != set(branches):
        raise H3DatasetError(f"pair_not_two_complete_candidates:{record.get('pair_id')}")
    ordered = sorted(candidates.items(), key=lambda item: item[0])
    examples: list[CandidateExample] = []
    for key, candidate in ordered:
        branch = branches[key]
        examples.append(
            CandidateExample(
                candidate_key=key,
                context=tuple(_context_vector(record)),
                candidate=tuple(tuple(row) for row in _candidate_tensor(record, candidate)),
                progress_m=_safe_float(branch.get("route_progress_m")),
                jerk_rms_mps3=_safe_float(branch.get("jerk_rms_mps3")),
                risk=_hard_unsafe_from_branch(branch),
                h1_soft_score=_h1_soft_score(record, candidate),
            )
        )
    winner = label.get("winner_candidate_id") if label.get("verdict") == "CANDIDATE_WIN" else None
    winner_index = None if winner is None else next((index for index, item in enumerate(examples) if item.candidate_key == str(winner)), None)
    if winner is not None and winner_index is None:
        raise H3DatasetError(f"winner_not_in_candidates:{record.get('pair_id')}")
    scenario = record["scenario"]
    return PairExample(
        pair_id=str(record["pair_id"]),
        map_name=str(scenario["map_name"]),
        family=str(scenario["family"]),
        seed=int(scenario["seed"]),
        weather=str(scenario["weather"]),
        split=split,
        candidates=(examples[0], examples[1]),
        winner_index=winner_index,
        tie=winner_index is None and label.get("verdict") == "TIE",
    )


def load_examples(root: Path | Sequence[Path], split_manifest: Mapping[str, Any], *, split: str) -> list[PairExample]:
    if split == "test":
        raise H3DatasetError("H3 training code cannot load test labels")
    rows = {str(item["pair_id"]): item for item in split_manifest.get("rows", ()) if item.get("split") == split and item.get("valid_pair")}
    records = {str(record["pair_id"]): record for record in read_h2_records(root)}
    labels = read_h2_labels(root, allowed_pair_ids=set(rows))
    examples: list[PairExample] = []
    for pair_id in sorted(rows):
        record = records.get(pair_id)
        if record is None:
            raise H3DatasetError(f"split_pair_missing:{pair_id}")
        label = labels[pair_id]
        if label.get("verdict") not in {"CANDIDATE_WIN", "TIE"}:
            continue
        examples.append(_make_pair_example(record, label, split))
    return examples


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).lower()
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


def _forbidden_key_audit(records: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    rows = list(split_manifest.get("rows", ()))
    allowed = {str(item["pair_id"]) for item in rows if item.get("split") != "test" and item.get("valid_pair")}
    for record in records:
        if str(record["pair_id"]) not in allowed:
            continue
        for candidate in record.get("candidates", ()):
            payload = {"context": _context_vector(record), "candidate": _candidate_tensor(record, candidate)}
            tokens = set(_walk_keys(payload))
            if tokens & FORBIDDEN_FEATURE_TOKENS:
                failures.append(f"forbidden_feature_token:{record['pair_id']}")
    return failures


def leakage_audit(root: Path | Sequence[Path], split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = read_h2_records(root)
    rows = list(split_manifest.get("rows", ()))
    failures: list[str] = []
    if len(records) != len(rows) or len({str(item["pair_id"]) for item in rows}) != len(rows):
        failures.append("manifest_record_count_or_uniqueness")
    by_lineage: dict[str, set[str]] = {}
    for item in rows:
        by_lineage.setdefault(str(item["lineage"]), set()).add(str(item["split"]))
    if any(len(splits) != 1 for splits in by_lineage.values()):
        failures.append("lineage_split_mismatch")
    if any(len({str(item["split"]) for item in rows if item["lineage"] == lineage}) != 1 for lineage in by_lineage):
        failures.append("weather_cross_split")
    failures.extend(_forbidden_key_audit(records, split_manifest))

    seen_payloads: dict[str, str] = {}
    nonfinite = 0
    shape_failures = 0
    for split in ("dev_fold_1", "dev_fold_2", "dev_fold_3"):
        try:
            examples = load_examples(root, split_manifest, split=split)
        except H3DatasetError as exc:
            failures.append(f"loader:{split}:{exc}")
            continue
        for example in examples:
            for candidate in example.candidates:
                if len(candidate.context) != H3_CONTEXT_DIM or len(candidate.candidate) != H3_CANDIDATE_STEPS or any(len(row) != H3_CANDIDATE_DIM for row in candidate.candidate):
                    shape_failures += 1
                if not all(math.isfinite(float(value)) for value in candidate.context) or not all(math.isfinite(float(value)) for row in candidate.candidate for value in row):
                    nonfinite += 1
                payload = {"context": list(candidate.context), "candidate": [list(row) for row in candidate.candidate]}
                digest = stable_sha256(payload)
                split_name = next((str(item["split"]) for item in rows if item.get("pair_id") == example.pair_id), "unknown")
                if digest in seen_payloads and seen_payloads[digest] != split_name:
                    failures.append("cross_split_duplicate_observable_candidate_payload")
                seen_payloads[digest] = split_name
    if nonfinite:
        failures.append("nonfinite_tensor_values")
    if shape_failures:
        failures.append("tensor_shape")
    return {
        "schema_version": H3_SCHEMA_VERSION,
        "records": len(records),
        "manifest_rows": len(rows),
        "lineages": len(by_lineage),
        "test_lineages": sum(1 for splits in by_lineage.values() if splits == {"test"}),
        "feature_payloads": len(seen_payloads),
        "nonfinite": nonfinite,
        "shape_failures": shape_failures,
        "failures": sorted(set(failures)),
        "passed": not failures,
    }


__all__ = [
    "CandidateExample",
    "FORBIDDEN_FEATURE_TOKENS",
    "H3DatasetError",
    "PairExample",
    "build_split_manifest",
    "leakage_audit",
    "lineage_key",
    "load_examples",
    "read_h2_labels",
    "read_h2_records",
    "write_split_manifest",
]
