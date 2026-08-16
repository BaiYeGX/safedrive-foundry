"""H3v2 Challenge data-quality gate and offline-label audit."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..h2.contracts import OracleLabel, OracleVerdict, stable_sha256, branch_outcome_from_dict
from ..h2.oracle import label_pair as h2_label_pair
from ..h2.store import PairedOutcomeStore
from .contracts import H3_CONFIG

DEFAULT_DATASET_ROOT = Path("generated") / "h3" / "carla-challenge-v2"


def _read_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / "pairs").glob("*.parquet")):
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
        if len(rows) != 1:
            raise RuntimeError(f"pair_shard_not_single_row:{path.name}")
        records.append(json.loads(rows[0]["record_json"]))
    return records


def _candidate_source(record: Mapping[str, Any], candidate_id: str) -> str:
    for candidate in record.get("candidates", ()):
        if candidate.get("candidate_id") == candidate_id:
            return str(candidate.get("source", "unknown"))
    return "unknown"


def _distinct(record: Mapping[str, Any]) -> bool:
    candidates = list(record.get("candidates", ()))
    if len(candidates) != 2:
        return False
    return candidates[0].get("canonical_sha256") != candidates[1].get("canonical_sha256")


def audit_challenge_dataset(dataset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    store_root = root or (DEFAULT_DATASET_ROOT / dataset_id)
    store = PairedOutcomeStore(store_root.parent, store_root.name)
    manifest_ok, manifest_bad = store.verify_manifest()
    records = _read_records(store_root)
    by_map = collections.Counter()
    by_family = collections.Counter()
    by_weather = collections.Counter()
    valid_by_map = collections.Counter()
    valid_by_family = collections.Counter()
    valid_by_weather = collections.Counter()
    terminal = 0
    valid = 0
    distinct = 0
    decisive = 0
    hard_unsafe_branches = 0
    source_wins = collections.Counter()
    duplicate_pairs = 0
    vla_forward_ok = 0
    artifacts_ok = 0
    reset_comparable = 0
    errors: list[str] = []

    for record in records:
        terminal += 1
        by_map[record["scenario"]["map_name"]] += 1
        by_family[record["scenario"]["family"]] += 1
        by_weather[record["scenario"]["weather"]] += 1
        if int(record.get("vla_forward_count", -1)) == 1:
            vla_forward_ok += 1
        pair_is_valid_terminal = record.get("terminal_status") in {"VALID_PAIR", "COMPLETED"} and len(record.get("branches", ())) == 2
        if _distinct(record):
            distinct += 1
        elif pair_is_valid_terminal:
            duplicate_pairs += 1
        branches = record.get("branches", ())
        if record.get("terminal_status") not in {"VALID_PAIR", "COMPLETED"} or len(branches) != 2:
            continue
        outcomes = [branch_outcome_from_dict(branch) for branch in branches]
        if all(branch.complete for branch in outcomes):
            valid += 1
            valid_by_map[record["scenario"]["map_name"]] += 1
            valid_by_family[record["scenario"]["family"]] += 1
            valid_by_weather[record["scenario"]["weather"]] += 1
            hard_unsafe_branches += sum(branch.hard_unsafe for branch in outcomes)
            if all(branch.timeline_path and branch.actor_future_path and branch.event_path for branch in outcomes):
                artifacts_ok += 1
            if all(branch.reset.comparable for branch in outcomes):
                reset_comparable += 1
        label_path = store.labels_dir / f"{record['pair_id']}.parquet"
        if label_path.exists():
            import pyarrow.parquet as pq
            label = json.loads(pq.read_table(label_path).to_pylist()[0]["label_json"])
            if label.get("verdict") == "CANDIDATE_WIN":
                decisive += 1
                source_wins[_candidate_source(record, str(label.get("winner_candidate_id")))] += 1

    if decisive:
        source_only_baseline = max(source_wins.values()) / decisive if source_wins else 0.0
    else:
        source_only_baseline = 0.0
    gate = H3_CONFIG["acceptance"]
    matrix = H3_CONFIG["challenge"]
    checks = {
        "manifest_valid": manifest_ok,
        "terminal_all": terminal == matrix["anchors"],
        "valid_pairs": valid >= gate["min_valid_pairs"],
        "per_map_valid": all(valid_by_map[m] >= gate["min_per_map_valid"] for m in matrix["maps"]),
        "per_family_valid": all(valid_by_family[f] >= gate["min_per_family_valid"] for f in matrix["families"]),
        "per_weather_valid": all(valid_by_weather[w] >= gate["min_per_weather_valid"] for w in matrix["weathers"]),
        "decisive": decisive >= gate["min_decisive"],
        "source_wins": all(source_wins.get(s, 0) >= gate["min_source_wins"] for s in ("expert", "vla")),
        "source_only_baseline": source_only_baseline <= gate["max_source_only_baseline"],
        "no_duplicate_pairs": duplicate_pairs == 0,
        "hard_unsafe_branches": hard_unsafe_branches >= gate["min_hard_unsafe_branches"],
        "vla_forward_count": vla_forward_ok == terminal,
        "artifacts": artifacts_ok == valid,
        "reset_comparable": reset_comparable == valid,
    }
    return {
        "dataset_id": dataset_id,
        "manifest_ok": manifest_ok,
        "manifest_bad": list(manifest_bad),
        "counts": {
            "records": len(records), "terminal": terminal, "valid": valid, "distinct": distinct,
            "duplicate_pairs": duplicate_pairs, "decisive": decisive, "hard_unsafe_branches": hard_unsafe_branches,
            "vla_forward_ok": vla_forward_ok, "artifacts_ok": artifacts_ok, "reset_comparable": reset_comparable,
            "by_map": dict(by_map), "by_family": dict(by_family), "by_weather": dict(by_weather),
            "valid_by_map": dict(valid_by_map), "valid_by_family": dict(valid_by_family), "valid_by_weather": dict(valid_by_weather),
            "source_wins": dict(source_wins),
        },
        "source_only_baseline": source_only_baseline,
        "checks": checks,
        "passed": manifest_ok and all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
        "errors": errors,
    }


def audit_offline_labels(dataset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    store_root = root or (DEFAULT_DATASET_ROOT / dataset_id)
    records = _read_records(store_root)
    failures = []
    checked = 0
    for record in records:
        branches = record.get("branches", ())
        if record.get("terminal_status") not in {"VALID_PAIR", "COMPLETED"} or len(branches) != 2:
            continue
        left = branch_outcome_from_dict(branches[0])
        right = branch_outcome_from_dict(branches[1])
        forward = h2_label_pair(left, right)
        swapped = h2_label_pair(right, left)
        if forward.verdict != swapped.verdict or forward.winner_candidate_id != swapped.winner_candidate_id or forward.reason != swapped.reason:
            failures.append(f"swap_invariance:{record['pair_id']}")
        if forward.verdict == OracleVerdict.CANDIDATE_WIN and forward.winner_candidate_id not in {left.candidate_id, right.candidate_id}:
            failures.append(f"winner_identity:{record['pair_id']}")
        checked += 1
    return {
        "checked": checked,
        "failures": failures,
        "passed": not failures,
        "oracle_version": "h2-offline-oracle-v1",
    }


__all__ = ["audit_challenge_dataset", "audit_offline_labels"]
