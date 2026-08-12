"""Simple baseline checkpoint save/restore with config hash (G3-02)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_checkpoint(path: Path | str, *, model_id: str, config: dict[str, Any], state: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "config": config,
        "config_hash": config_hash(config),
        "state": state,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return payload["config_hash"]


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = config_hash(data.get("config") or {})
    if data.get("config_hash") != expected:
        raise ValueError("checkpoint config_hash mismatch")
    return data
