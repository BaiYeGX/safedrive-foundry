"""Crash-safe Parquet store for H2 pairs, timelines and content-addressed images."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .contracts import PairRecord, stable_json_bytes


STORE_VERSION = "h2-paired-outcome-store-v1"


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment admission
        raise RuntimeError("H2 authoritative store requires pyarrow") from exc
    return pa, pq


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    if not value or any(token in value for token in ("/", "\\", "..")):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


class PairedOutcomeStore:
    """One atomically committed Parquet file per pair plus immutable artifacts.

    Per-pair shards keep resume O(1) and avoid rewriting previous outcomes after a
    crash. The deterministic manifest is itself atomically replaced.
    """

    def __init__(self, root: Path | str, dataset_id: str) -> None:
        self.dataset_id = _safe_component(dataset_id)
        self.root = Path(root) / self.dataset_id
        self.pairs_dir = self.root / "pairs"
        self.timelines_dir = self.root / "timelines"
        self.images_dir = self.root / "images" / "sha256"
        self.events_dir = self.root / "events"
        self.actor_future_dir = self.root / "actor-future"
        self.labels_dir = self.root / "labels"
        for path in (
            self.pairs_dir, self.timelines_dir, self.images_dir, self.events_dir,
            self.actor_future_dir, self.labels_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _atomic_parquet(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        pa, pq = _require_pyarrow()
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            table = pa.Table.from_pylist(list(rows))
            pq.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _atomic_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8"))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def pair_path(self, pair_id: str) -> Path:
        return self.pairs_dir / f"{_safe_component(pair_id)}.parquet"

    def has_valid_pair(self, pair_id: str) -> bool:
        path = self.pair_path(pair_id)
        if not path.is_file():
            return False
        try:
            row = self._read_single_row(path)
            record = json.loads(row["record_json"])
            base_valid = (
                row["record_sha256"] == hashlib.sha256(row["record_json"].encode("utf-8")).hexdigest()
                and record["pair_id"] == pair_id
                and record["dataset_id"] == self.dataset_id
                and record["content_sha256"]
                == hashlib.sha256(stable_json_bytes({k: v for k, v in record.items() if k != "content_sha256"})).hexdigest()
            )
            if not base_valid:
                return False
            # Resume is safe only when every immutable artifact referenced by
            # the pair is still present and has the captured digest.
            for branch in record.get("branches", ()):
                candidate_id = str(branch.get("candidate_id", ""))
                for field, artifact_name in (
                    ("timeline_path", "timeline"),
                    ("actor_future_path", "actor_future"),
                    ("event_path", "events"),
                ):
                    relative = str(branch.get(field, ""))
                    if not relative:
                        continue
                    artifact = self.root / relative
                    expected = record.get("artifact_hashes", {}).get(f"{candidate_id}:{artifact_name}")
                    if not artifact.is_file() or not expected or file_sha256(artifact) != expected:
                        return False
            for item in record.get("observable_history", ()):
                relative = str(item.get("image_path", ""))
                digest = str(item.get("image_sha256", ""))
                if relative:
                    artifact = self.root / relative
                    if not artifact.is_file() or not digest or file_sha256(artifact) != digest:
                        return False
            return True
        except (KeyError, ValueError, OSError, RuntimeError):
            return False

    def write_pair(self, pair: PairRecord) -> Path:
        if pair.dataset_id != self.dataset_id:
            raise ValueError("pair dataset_id does not match store")
        path = self.pair_path(pair.pair_id)
        payload = pair.to_dict()
        record_json = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        record_sha256 = hashlib.sha256(record_json.encode("utf-8")).hexdigest()
        if path.exists():
            row = self._read_single_row(path)
            if row.get("record_sha256") == record_sha256:
                return path
            raise FileExistsError(f"immutable H2 pair already exists with different content: {pair.pair_id}")
        self._atomic_parquet(
            path,
            [{"pair_id": pair.pair_id, "record_sha256": record_sha256, "record_json": record_json}],
        )
        self.write_manifest()
        return path

    def _read_single_row(self, path: Path) -> dict[str, Any]:
        _, pq = _require_pyarrow()
        rows = pq.read_table(path).to_pylist()
        if len(rows) != 1:
            raise ValueError(f"expected one row in {path}")
        return dict(rows[0])

    def iter_pair_dicts(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self.pairs_dir.glob("*.parquet")):
            row = self._read_single_row(path)
            yield json.loads(row["record_json"])

    def write_timeline(self, pair_id: str, candidate_id: str, rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
        filename = f"{_safe_component(pair_id)}__{_safe_component(candidate_id)}.parquet"
        path = self.timelines_dir / filename
        materialized = list(rows)
        if not materialized:
            raise ValueError("timeline cannot be empty")
        self._atomic_parquet(path, materialized)
        return str(path.relative_to(self.root)), file_sha256(path)

    def write_event_rows(self, pair_id: str, candidate_id: str, rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
        filename = f"{_safe_component(pair_id)}__{_safe_component(candidate_id)}.parquet"
        path = self.events_dir / filename
        materialized = list(rows) or [{"event_type": "none", "frame": -1, "simulation_time_s": -1.0}]
        self._atomic_parquet(path, materialized)
        return str(path.relative_to(self.root)), file_sha256(path)

    def write_actor_future(self, pair_id: str, candidate_id: str, rows: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
        filename = f"{_safe_component(pair_id)}__{_safe_component(candidate_id)}.parquet"
        path = self.actor_future_dir / filename
        materialized = list(rows)
        if not materialized:
            raise ValueError("actor future cannot be empty")
        self._atomic_parquet(path, materialized)
        return str(path.relative_to(self.root)), file_sha256(path)

    def write_image(self, png_bytes: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(png_bytes).hexdigest()
        path = self.images_dir / digest[:2] / f"{digest}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as handle:
                    handle.write(png_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        elif file_sha256(path) != digest:
            raise RuntimeError(f"content-address collision: {digest}")
        return str(path.relative_to(self.root)), digest

    def write_label(
        self,
        pair_id: str,
        label: Mapping[str, Any],
        *,
        update_manifest: bool = True,
    ) -> Path:
        """Commit one immutable offline label, separate from live pair shards."""

        pair_id = _safe_component(pair_id)
        path = self.labels_dir / f"{pair_id}.parquet"
        label_json = json.dumps(dict(label), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(label_json.encode("utf-8")).hexdigest()
        if path.exists():
            row = self._read_single_row(path)
            if row.get("label_sha256") == digest:
                return path
            raise FileExistsError(f"immutable H2 label already exists with different content: {pair_id}")
        self._atomic_parquet(
            path,
            [{"pair_id": pair_id, "label_sha256": digest, "label_json": label_json}],
        )
        # Batch labelers can defer the full artifact scan until all labels are
        # committed.  The default remains eager for callers that expect every
        # single write to leave a current manifest.
        if update_manifest:
            self.write_manifest()
        return path

    def iter_labeled_pair_dicts(self) -> Iterator[dict[str, Any]]:
        labels: dict[str, dict[str, Any]] = {}
        for path in sorted(self.labels_dir.glob("*.parquet")):
            row = self._read_single_row(path)
            labels[str(row["pair_id"])] = json.loads(row["label_json"])
        for record in self.iter_pair_dicts():
            merged = dict(record)
            if record["pair_id"] in labels:
                merged["label"] = labels[record["pair_id"]]
            yield merged

    def artifact_manifest(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json" or ".tmp" in path.name:
                continue
            rows.append(
                {
                    "path": str(path.relative_to(self.root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        payload: dict[str, Any] = {
            "store_version": STORE_VERSION,
            "dataset_id": self.dataset_id,
            "pair_count": sum(1 for row in rows if row["path"].startswith("pairs/")),
            "artifact_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "artifacts": rows,
        }
        payload["manifest_sha256"] = hashlib.sha256(stable_json_bytes(payload)).hexdigest()
        return payload

    def write_manifest(self) -> Path:
        path = self.root / "manifest.json"
        self._atomic_json(path, self.artifact_manifest())
        return path

    def verify_manifest(self) -> tuple[bool, tuple[str, ...]]:
        path = self.root / "manifest.json"
        if not path.is_file():
            return False, ("manifest_missing",)
        stored = json.loads(path.read_text(encoding="utf-8"))
        expected_hash = stored.pop("manifest_sha256", None)
        reasons: list[str] = []
        if expected_hash != hashlib.sha256(stable_json_bytes(stored)).hexdigest():
            reasons.append("manifest_hash_mismatch")
        manifest_items = stored.get("artifacts", [])
        manifest_paths = [str(item.get("path", "")) for item in manifest_items]
        if len(manifest_paths) != len(set(manifest_paths)):
            reasons.append("manifest_duplicate_path")
        for item in manifest_items:
            artifact = self.root / item["path"]
            try:
                artifact.resolve().relative_to(self.root.resolve())
            except ValueError:
                reasons.append(f"artifact_path_escape:{item.get('path')}")
                continue
            if not artifact.is_file():
                reasons.append(f"artifact_missing:{item['path']}")
            elif artifact.stat().st_size != item["bytes"] or file_sha256(artifact) != item["sha256"]:
                reasons.append(f"artifact_hash_mismatch:{item['path']}")
        actual_paths = {
            str(path.relative_to(self.root))
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "manifest.json" and ".tmp" not in path.name
        }
        unexpected = sorted(actual_paths - set(manifest_paths))
        if unexpected:
            reasons.extend(f"unexpected_artifact:{path}" for path in unexpected)
        return not reasons, tuple(reasons)


__all__ = ["PairedOutcomeStore", "STORE_VERSION", "file_sha256"]
