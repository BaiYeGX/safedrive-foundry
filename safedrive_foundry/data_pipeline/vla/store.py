"""Shard store with content-hash de-dup and resume (G3-01).

Authoritative on-disk format: Parquet (one file per split).
JSONL remains available as optional debug export only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from data_pipeline.vla.leakage import LeakageAuditor, LeakageError
from data_pipeline.vla.schema import (
    SampleRecord,
    content_hash,
    payload_hash,
    sample_from_dict,
    sample_to_dict,
)
from data_pipeline.vla.split import SplitAssigner, SplitName

STORAGE_FORMAT = "parquet"


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "G3-01 authoritative store requires pyarrow. "
            "Install into /home/sdf/.venvs/sdf: pip install pyarrow"
        ) from exc
    return pa, pq


class ShardStore:
    """Parquet shards per split; resume skips already-written content hashes."""

    def __init__(
        self,
        root: Path | str,
        *,
        assigner: SplitAssigner | None = None,
        auditor: LeakageAuditor | None = None,
        format: str = STORAGE_FORMAT,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.assigner = assigner or SplitAssigner()
        self.auditor = auditor or LeakageAuditor(self.assigner)
        self.format = format if format in {"parquet", "jsonl"} else STORAGE_FORMAT
        self._rows: dict[SplitName, list[dict[str, Any]]] = {s: [] for s in SplitName}
        self._load_existing()

    def _split_path(self, split: SplitName) -> Path:
        if self.format == "jsonl":
            return self.root / f"{split.value}.jsonl"
        return self.root / f"{split.value}.parquet"

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _load_existing(self) -> None:
        for split in SplitName:
            path = self._split_path(split)
            if not path.is_file():
                continue
            if self.format == "jsonl":
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        sample = sample_from_dict(json.loads(line))
                        self._index_existing(sample, split)
            else:
                pa, pq = _require_pyarrow()
                table = pq.read_table(path)
                for row in table.to_pylist():
                    payload = json.loads(row["record_json"])
                    sample = sample_from_dict(payload)
                    self._index_existing(sample, split)

    def _index_existing(self, sample: SampleRecord, split: SplitName) -> None:
        h = sample.content_hash or content_hash(sample)
        sample.content_hash = h
        ph = payload_hash(sample)
        self.auditor.committed[split].add(h)
        self.auditor.identity_index[sample.identity.key()] = split
        self.auditor.payload_to_split[ph] = split
        self.auditor.near_dup_prefix_index[self.auditor._near_prefix(sample)] = split
        self._rows[split].append(
            {
                "content_hash": h,
                "payload_hash": ph,
                "record_json": json.dumps(sample_to_dict(sample), ensure_ascii=True, sort_keys=True),
            }
        )

    def write_sample(self, sample: SampleRecord) -> SplitName:
        if not sample.parameter_hash:
            sample.recompute_parameter_hash()
        sample.content_hash = content_hash(sample)
        try:
            split = self.auditor.admit(sample)
        except LeakageError:
            raise
        ph = payload_hash(sample)
        row = {
            "content_hash": sample.content_hash,
            "payload_hash": ph,
            "record_json": json.dumps(sample_to_dict(sample), ensure_ascii=True, sort_keys=True),
        }
        self._rows[split].append(row)
        self._flush_split(split)
        self._update_manifest()
        return split

    def write_many(self, samples: list[SampleRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in samples:
            split = self.write_sample(sample)
            counts[split.value] = counts.get(split.value, 0) + 1
        return counts

    def _flush_split(self, split: SplitName) -> None:
        path = self._split_path(split)
        rows = self._rows[split]
        if self.format == "jsonl":
            with path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(row["record_json"] + "\n")
            return
        pa, pq = _require_pyarrow()
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)

    def _update_manifest(self) -> None:
        man = {
            "format": self.format,
            "storage_format_authoritative": STORAGE_FORMAT,
            "splits": {
                s.value: {
                    "path": str(self._split_path(s).name),
                    "n": len(self._rows[s]),
                }
                for s in SplitName
            },
        }
        self._manifest_path().write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")

    def iter_split(self, split: SplitName) -> Iterator[SampleRecord]:
        for row in self._rows[split]:
            yield sample_from_dict(json.loads(row["record_json"]))

    def export_jsonl(self, split: SplitName, path: Path | str) -> Path:
        """Optional debug export; not the authoritative store."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for sample in self.iter_split(split):
                fh.write(json.dumps(sample_to_dict(sample), ensure_ascii=True) + "\n")
        return path
