"""Crash-safe immutable store for C2 anchors, branches, labels and features."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from data_pipeline.h2.contracts import stable_json_bytes, stable_sha256
from data_pipeline.h2.store import file_sha256

from .contracts import CoraRootRecord


STORE_VERSION = "safedrive.cora.store.v1"


def _safe_component(value: str) -> str:
    if not value or any(token in value for token in ("/", "\\", "..", ":")):
        raise ValueError(f"unsafe_cora_component:{value!r}")
    return value


class CoraDataStore:
    def __init__(self, root: Path | str, dataset_id: str) -> None:
        self.dataset_id = _safe_component(dataset_id)
        self.root = Path(root) / self.dataset_id
        self.anchors_dir = self.root / "anchors"
        self.images_dir = self.root / "images"
        self.proposals_dir = self.root / "proposals"
        self.pairs_dir = self.root / "pairs"
        self.features_dir = self.root / "features"
        self.timelines_dir = self.root / "timelines"
        self.actor_future_dir = self.root / "actor-future"
        self.events_dir = self.root / "events"
        self.interventions_dir = self.root / "interventions"
        self.labels_dir = self.root / "labels"
        for path in (
            self.anchors_dir,
            self.images_dir,
            self.proposals_dir,
            self.pairs_dir,
            self.features_dir,
            self.timelines_dir,
            self.actor_future_dir,
            self.events_dir,
            self.interventions_dir,
            self.labels_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _atomic_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def write_immutable_json(self, path: Path, payload: Mapping[str, Any]) -> tuple[str, str]:
        encoded = json.dumps(
            dict(payload), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if path.exists():
            if file_sha256(path) == digest:
                return str(path.relative_to(self.root)), digest
            raise FileExistsError(f"immutable_cora_artifact_conflict:{path}")
        self._atomic_bytes(path, encoded)
        return str(path.relative_to(self.root)), digest

    def write_anchor(self, root_id: str, payload: Mapping[str, Any]) -> tuple[str, str]:
        return self.write_immutable_json(self.anchors_dir / f"{_safe_component(root_id)}.json", payload)

    def write_image(self, encoded: bytes, *, suffix: str = ".png") -> tuple[str, str]:
        """Content-address and deduplicate an observable image."""

        if not encoded:
            raise ValueError("cora_store_empty_image")
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"cora_store_image_suffix:{suffix}")
        digest = hashlib.sha256(encoded).hexdigest()
        path = self.images_dir / f"{digest}{suffix}"
        if path.exists():
            if file_sha256(path) != digest:
                raise FileExistsError(f"immutable_cora_image_conflict:{path}")
        else:
            self._atomic_bytes(path, encoded)
        return str(path.relative_to(self.root)), digest

    def write_feature(
        self, root_id: str, proposal_sha256: str, payload: Mapping[str, Any]
    ) -> tuple[str, str]:
        if len(proposal_sha256) != 64:
            raise ValueError("cora_feature_proposal_hash")
        return self.write_immutable_json(
            self.features_dir / f"{_safe_component(root_id)}__{proposal_sha256}.json", payload
        )

    def write_proposal(
        self, root_id: str, proposal_sha256: str, payload: Mapping[str, Any], *, intervention: bool
    ) -> tuple[str, str]:
        directory = self.interventions_dir if intervention else self.proposals_dir
        return self.write_immutable_json(
            directory / f"{_safe_component(root_id)}__{proposal_sha256}.json", payload
        )

    def write_label(
        self, root_id: str, proposal_sha256: str, payload: Mapping[str, Any]
    ) -> tuple[str, str]:
        return self.write_immutable_json(
            self.labels_dir / f"{_safe_component(root_id)}__{proposal_sha256}.json", payload
        )

    def public_label_path(self, root_id: str, proposal_sha256: str) -> Path:
        if len(proposal_sha256) != 64:
            raise ValueError("cora_public_label_proposal_hash")
        return self.labels_dir / f"{_safe_component(root_id)}__{proposal_sha256}.public.json"

    def _atomic_parquet(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - live environment admission
            raise RuntimeError("cora_store_requires_pyarrow") from exc
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            table = pa.Table.from_pylist([dict(row) for row in rows])
            pq.write_table(table, temporary, compression="zstd")
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def write_rows(
        self,
        kind: str,
        root_id: str,
        proposal_sha256: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> tuple[str, str]:
        directories = {
            "timeline": self.timelines_dir,
            "actor_future": self.actor_future_dir,
            "events": self.events_dir,
        }
        if kind not in directories:
            raise ValueError(f"cora_store_row_kind:{kind}")
        materialized = [dict(row) for row in rows]
        if not materialized:
            if kind == "events":
                materialized = [{"event_type": "none", "frame": -1, "simulation_time_s": -1.0}]
            else:
                raise ValueError(f"cora_store_empty_rows:{kind}")
        path = directories[kind] / f"{_safe_component(root_id)}__{proposal_sha256}.parquet"
        if path.exists():
            raise FileExistsError(f"immutable_cora_rows_exist:{path}")
        self._atomic_parquet(path, materialized)
        return str(path.relative_to(self.root)), file_sha256(path)

    def write_root(
        self,
        record: CoraRootRecord | Mapping[str, Any],
        *,
        update_manifest: bool = True,
    ) -> tuple[str, str]:
        payload = record.to_dict() if isinstance(record, CoraRootRecord) else dict(record)
        if payload.get("dataset_id") != self.dataset_id:
            raise ValueError("cora_root_dataset_mismatch")
        root_id = str(payload.get("root_id", ""))
        path, digest = self.write_immutable_json(
            self.pairs_dir / f"{_safe_component(root_id)}.json", payload
        )
        if update_manifest:
            self.write_manifest()
        return path, digest

    def iter_roots(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self.pairs_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = payload.pop("content_sha256", None)
            if expected != stable_sha256(payload):
                raise ValueError(f"cora_root_content_hash:{path.name}")
            payload["content_sha256"] = expected
            yield payload

    def has_valid_root(self, root_id: str) -> bool:
        path = self.pairs_dir / f"{_safe_component(root_id)}.json"
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = payload.pop("content_sha256")
            if expected != stable_sha256(payload):
                return False
            for mapping_name in ("feature_paths", "feature_sha256"):
                if not isinstance(payload.get(mapping_name), Mapping):
                    return False
            anchor = self.root / str(payload.get("anchor_path", ""))
            if not anchor.is_file() or file_sha256(anchor) != payload.get("anchor_sha256"):
                return False
            for key, relative in payload["feature_paths"].items():
                artifact = self.root / str(relative)
                if not artifact.is_file() or file_sha256(artifact) != payload["feature_sha256"].get(key):
                    return False
            for branch in payload.get("branches", ()):
                for key, relative in branch.get("artifact_paths", {}).items():
                    artifact = self.root / str(relative)
                    if not artifact.is_file() or file_sha256(artifact) != branch.get("artifact_sha256", {}).get(key):
                        return False
            for proposal in payload.get("proposals", ()):
                digest = str(proposal.get("proposal_sha256", ""))
                directory = self.interventions_dir if proposal.get("kind") == "offline_intervention" else self.proposals_dir
                artifact = directory / f"{_safe_component(root_id)}__{digest}.json"
                if not artifact.is_file():
                    return False
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def artifact_manifest(self) -> dict[str, Any]:
        artifacts = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json" or ".tmp" in path.name:
                continue
            artifacts.append(
                {
                    "path": str(path.relative_to(self.root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        payload = {
            "schema_version": STORE_VERSION,
            "dataset_id": self.dataset_id,
            "root_count": sum(item["path"].startswith("pairs/") for item in artifacts),
            "artifact_count": len(artifacts),
            "total_bytes": sum(int(item["bytes"]) for item in artifacts),
            "artifacts": artifacts,
        }
        payload["manifest_sha256"] = stable_sha256(payload)
        return payload

    def write_manifest(self) -> Path:
        path = self.root / "manifest.json"
        payload = self.artifact_manifest()
        encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        self._atomic_bytes(path, encoded)
        return path

    def verify_manifest(self) -> tuple[bool, tuple[str, ...]]:
        path = self.root / "manifest.json"
        if not path.is_file():
            return False, ("manifest_missing",)
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = payload.pop("manifest_sha256", None)
        failures: list[str] = []
        if expected != stable_sha256(payload):
            failures.append("manifest_self_hash")
        rows = payload.get("artifacts", ())
        manifest_paths = [str(row.get("path", "")) for row in rows]
        if len(manifest_paths) != len(set(manifest_paths)):
            failures.append("manifest_duplicate_path")
        for row in rows:
            relative = str(row.get("path", ""))
            artifact = self.root / relative
            try:
                artifact.resolve().relative_to(self.root.resolve())
            except ValueError:
                failures.append(f"manifest_path_escape:{relative}")
                continue
            if not artifact.is_file():
                failures.append(f"manifest_missing_artifact:{relative}")
            elif artifact.stat().st_size != int(row.get("bytes", -1)) or file_sha256(artifact) != row.get("sha256"):
                failures.append(f"manifest_artifact_hash:{relative}")
        actual = {
            str(item.relative_to(self.root))
            for item in self.root.rglob("*")
            if item.is_file() and item.name != "manifest.json" and ".tmp" not in item.name
        }
        for relative in sorted(actual - set(manifest_paths)):
            failures.append(f"manifest_unexpected_artifact:{relative}")
        return not failures, tuple(failures)


def pending_collection_rows(
    store: CoraDataStore, rows: Sequence[Any]
) -> tuple[Any, ...]:
    """Return only rows that do not already have an immutable valid root.

    Resource admission for a resumed collection must project work that can
    still be attempted.  Charging every row in the requested scope again
    double-counts immutable roots and can falsely trip the branch-attempt
    ceiling even though the resume itself stays within the frozen budget.
    """

    return tuple(row for row in rows if not store.has_valid_root(str(row.root_id)))


__all__ = ["CoraDataStore", "STORE_VERSION", "pending_collection_rows"]
