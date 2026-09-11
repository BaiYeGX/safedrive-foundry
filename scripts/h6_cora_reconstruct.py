"""Reconstruct and audit native Classic planner supervision for C3."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "safedrive_foundry", REPO / "simlingo-main"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pipeline.h6.cora.c3_supervision import reconstruct_release  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args(argv)
    payload = reconstruct_release(args.release_root, splits=args.splits)
    payload["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    # The timestamp is part of the new artifact identity, while the replay
    # content digest remains the hash of the semantic reconstruction payload.
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "RECONSTRUCTION_MEASURED",
        "output": str(output),
        "sample_count": payload["sample_count"],
        "split_counts": payload["split_counts"],
        "failure_count": payload["failure_count"],
        "quality_counts": payload["quality_counts"],
        "native_generator_hash": payload["native_generator_hash"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
