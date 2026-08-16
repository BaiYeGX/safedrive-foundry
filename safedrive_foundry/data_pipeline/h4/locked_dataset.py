"""Locked H4 test loader and isolation audit.

The H4 loader is deliberately separate from H3's ``load_examples``: it is the
only component allowed to open test labels, and it does so only after the H4
evaluation script has been frozen and all input hashes verified.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data_pipeline.h3.dataset import (
    FORBIDDEN_FEATURE_TOKENS,
    H3DatasetError,
    PairExample,
    _candidate_tensor,
    _context_vector,
    _make_pair_example,
    _read_parquet_record,
    _walk_keys,
    read_h2_records,
)
from data_pipeline.h3.contracts import H3_CONFIG, H3_CONTEXT_DIM, H3_CANDIDATE_DIM, H3_CANDIDATE_STEPS, stable_sha256
from data_pipeline.h4.contracts import H4_CONFIG


@dataclass(frozen=True)
class LockedPairExample:
    """A test pair plus audit-only source labels.

    ``pair`` is the same immutable object used by H3 evaluation.  ``sources``
    is only for stratified reporting and is never part of model features.
    """

    pair: PairExample
    sources: tuple[str, str]

    @property
    def pair_id(self) -> str:
        return self.pair.pair_id

    @property
    def decisive(self) -> bool:
        return self.pair.decisive

    @property
    def winner_index(self) -> int | None:
        return self.pair.winner_index

    @property
    def winner_source(self) -> str | None:
        if self.pair.winner_index is None:
            return None
        return self.sources[self.pair.winner_index]


def _read_test_labels(root: Path | Sequence[Path], *, allowed_pair_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Read test label shards only for explicitly allowed test pair ids."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise H3DatasetError("pyarrow_required") from exc
    roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(p) for p in root]
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
        raise H3DatasetError(f"missing_locked_test_labels:{missing[:4]}")
    return labels


def _roots_to_list(root: Path | Sequence[Path]) -> list[Path]:
    if isinstance(root, (list, tuple, set)):
        return [Path(p) for p in root]
    return [Path(root)]


def load_locked_test_examples(
    root: Path | Sequence[Path],
    split_manifest: Mapping[str, Any],
    *,
    split: str = "test",
) -> list[LockedPairExample]:
    """Load test examples and their audit-only source mapping."""
    if split != "test":
        raise H3DatasetError("locked_test_loader_only_accepts_test")
    rows = {
        str(item["pair_id"]): item
        for item in split_manifest.get("rows", ())
        if item.get("split") == "test" and item.get("valid_pair")
    }
    records = {str(record["pair_id"]): record for record in read_h2_records(root)}
    labels = _read_test_labels(root, allowed_pair_ids=set(rows))
    examples: list[LockedPairExample] = []
    for pair_id in sorted(rows):
        record = records.get(pair_id)
        if record is None:
            raise H3DatasetError(f"test_pair_missing:{pair_id}")
        label = labels[pair_id]
        if label.get("verdict") not in {"CANDIDATE_WIN", "TIE"}:
            continue
        pair = _make_pair_example(record, label, split)
        source_by_candidate = {str(item["candidate_id"]): str(item.get("source", "")) for item in record.get("candidates", ())}
        sources = tuple(source_by_candidate.get(pair.candidates[i].candidate_key, "unknown") for i in range(2))
        examples.append(LockedPairExample(pair=pair, sources=sources))
    return examples


def _feature_payload(record: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {"context": _context_vector(record), "candidate": _candidate_tensor(record, candidate)}


def _forbidden_tokens_for_examples(examples: Sequence[LockedPairExample]) -> list[str]:
    failures: list[str] = []
    for example in examples:
        for candidate in example.pair.candidates:
            # The tensor is already a plain tuple; reconstruct the same payload
            # that H3 uses for feature hashing.
            payload = {
                "context": list(candidate.context),
                "candidate": [list(row) for row in candidate.candidate],
            }
            tokens = set(_walk_keys(payload))
            if tokens & FORBIDDEN_FEATURE_TOKENS:
                failures.append(f"forbidden_feature_token:{example.pair_id}")
    return failures


def audit_test_isolation(
    root: Path | Sequence[Path],
    split_manifest: Mapping[str, Any],
    dev_examples: Sequence[PairExample],
    test_examples: Sequence[LockedPairExample],
) -> dict[str, Any]:
    """Audit that the locked test split is lineage/feature isolated from dev."""
    rows = list(split_manifest.get("rows", ()))
    failures: list[str] = []
    by_lineage: dict[str, set[str]] = {}
    for item in rows:
        by_lineage.setdefault(str(item["lineage"]), set()).add(str(item["split"]))
    if any(len(splits) != 1 for splits in by_lineage.values()):
        failures.append("lineage_split_mismatch")

    test_ids = {str(item["pair_id"]) for item in rows if item.get("split") == "test" and item.get("valid_pair")}
    dev_ids = {str(item["pair_id"]) for item in rows if item.get("split") != "test" and item.get("valid_pair")}
    if test_ids & dev_ids:
        failures.append("test_dev_pair_overlap")

    seen_payloads: dict[str, str] = {}
    for example in dev_examples:
        for candidate in example.candidates:
            payload = {"context": list(candidate.context), "candidate": [list(row) for row in candidate.candidate]}
            digest = stable_sha256(payload)
            seen_payloads.setdefault(digest, "dev")
    for example in test_examples:
        for candidate in example.pair.candidates:
            payload = {"context": list(candidate.context), "candidate": [list(row) for row in candidate.candidate]}
            digest = stable_sha256(payload)
            if digest in seen_payloads and seen_payloads[digest] == "dev":
                failures.append(f"cross_split_duplicate_payload:{example.pair_id}")
            seen_payloads.setdefault(digest, "test")

    failures.extend(_forbidden_tokens_for_examples(test_examples))

    shape_failures = 0
    nonfinite = 0
    for example in test_examples:
        for candidate in example.pair.candidates:
            if (
                len(candidate.context) != H3_CONTEXT_DIM
                or len(candidate.candidate) != H3_CANDIDATE_STEPS
                or any(len(row) != H3_CANDIDATE_DIM for row in candidate.candidate)
            ):
                shape_failures += 1
            if not all(math.isfinite(float(value)) for value in candidate.context) or not all(
                math.isfinite(float(value)) for row in candidate.candidate for value in row
            ):
                nonfinite += 1
    if shape_failures:
        failures.append("test_tensor_shape")
    if nonfinite:
        failures.append("test_tensor_nonfinite")

    return {
        "schema_version": H4_CONFIG["schema_version"],
        "test_rows": len(test_ids),
        "test_examples": len(test_examples),
        "dev_examples": len(dev_examples),
        "test_lineages": sum(1 for splits in by_lineage.values() if splits == {"test"}),
        "failures": sorted(set(failures)),
        "passed": not failures,
    }


__all__ = [
    "LockedPairExample",
    "audit_test_isolation",
    "load_locked_test_examples",
]
