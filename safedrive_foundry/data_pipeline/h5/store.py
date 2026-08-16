"""Atomic H5 evidence store and manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from data_pipeline.h3.contracts import stable_sha256


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class H5Store:
    """Minimal file store for H5 run records and manifest."""

    def __init__(self, root: Path, dataset_id: str) -> None:
        self.root = root
        self.dataset_id = dataset_id
        self.runs_dir = root / dataset_id / "runs"
        self.manifest_path = root / dataset_id / "manifest.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, pair_id: str, arm: str) -> Path:
        safe = pair_id.replace("__", "_").replace("/", "_")
        return self.runs_dir / f"{safe}__{arm}.json"

    def has_run(self, pair_id: str, arm: str) -> bool:
        path = self._run_path(pair_id, arm)
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(payload.get("ok", False))

    def write_run(self, run: Mapping[str, Any]) -> str:
        pair_id = str(run["pair_id"])
        arm = str(run["arm"])
        path = self._run_path(pair_id, arm)
        # Include content hash inside payload for self-verification.
        payload = dict(run)
        payload = _jsonable(payload)
        payload["content_sha256"] = stable_sha256(payload)
        _atomic_json(path, payload)
        return str(path)

    def read_run(self, pair_id: str, arm: str) -> dict[str, Any]:
        path = self._run_path(pair_id, arm)
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = payload.get("content_sha256")
        verify = {k: v for k, v in payload.items() if k != "content_sha256"}
        if stable_sha256(verify) != expected:
            raise ValueError(f"run_content_hash_mismatch:{path}")
        return payload

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.runs_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            out.append(payload)
        return out

    def write_manifest(self) -> dict[str, Any]:
        files = []
        for path in sorted(self.runs_dir.glob("*.json")):
            files.append(
                {
                    "path": str(path.relative_to(self.root / self.dataset_id)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
        payload = {
            "schema_version": "safedrive.h5.manifest.v1",
            "dataset_id": self.dataset_id,
            "files": files,
        }
        payload["manifest_sha256"] = stable_sha256(payload)
        _atomic_json(self.manifest_path, payload)
        return payload


__all__ = ["H5Store", "_atomic_json"]
