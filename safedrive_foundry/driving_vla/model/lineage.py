"""Weight/code lineage freeze helpers (G3-03 F0)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LineageManifest:
    base_model: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    code_root: str
    license_scope: str
    deployment_scope: str
    precision: str
    model_id: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path | str, *, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_simlingo_manifest(
    *,
    ckpt: Path | str,
    code_root: Path | str,
    precision: str = "bf16",
) -> LineageManifest:
    ckpt = Path(ckpt)
    return LineageManifest(
        base_model="SimLingo/InternVL2-1B",
        checkpoint_path=str(ckpt),
        checkpoint_sha256=file_sha256(ckpt) if ckpt.is_file() else "",
        checkpoint_bytes=ckpt.stat().st_size if ckpt.is_file() else 0,
        code_root=str(code_root),
        license_scope="see simlingo + internvl notices",
        deployment_scope="simulation_research_only",
        precision=precision,
        model_id="sdf-vla-v0@0.0.1",
        notes=["F0 lineage freeze", "Do not use as real-vehicle deploy weights"],
    )


def write_manifest(path: Path | str, manifest: LineageManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
