#!/usr/bin/env python3
"""Estimate simulator A/A P99 from the frozen same-action repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.actor_future_collector import (  # noqa: E402
    load_actor_future_trace,
    resample_actor_frames,
)
from driving_vla.evaluation.paired_contract import content_hash  # noqa: E402


def _repeat_distance(report: dict[str, Any]) -> float:
    paths = [Path(value) for value in (report.get("actor_future_paths") or {}).values()]
    if len(paths) != 2 or any(not path.is_file() for path in paths):
        return float("nan")
    aligned = [resample_actor_frames(load_actor_future_trace(path)) for path in paths]
    keys = sorted(set(aligned[0]).intersection(aligned[1]))
    values: list[float] = []
    for key in keys:
        for first, second in zip(aligned[0][key], aligned[1][key]):
            if first is None or second is None:
                continue
            values.append(float(np.hypot(first.x - second.x, first.y - second.y)))
    return float(np.mean(values)) if values else float("nan")


def validate(root: Path, *, expected: int = 168) -> dict[str, Any]:
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("aa_report.json"))
    ]
    if len(reports) != expected:
        raise ValueError(f"A/A repeat coverage must be {expected}, got {len(reports)}")
    repeat_groups = [str(report.get("repeat_group") or "") for report in reports]
    identities = [str(report.get("aa_noise_identity") or "").lower() for report in reports]
    if any(not group for group in repeat_groups) or len(set(repeat_groups)) != expected:
        raise ValueError("A/A repeats require unique non-empty repeat_group bindings")
    expected_identities = [
        content_hash(
            {
                "namespace": "r3_aa_noise_probe",
                "repeat_group": group,
                "candidate_id": "v3_nominal_progress",
            }
        ).lower()
        for group in repeat_groups
    ]
    if identities != expected_identities:
        raise ValueError("A/A repeat identity binding mismatch")
    checkpoint_hashes = {str(report.get("checkpoint_sha256") or "").lower() for report in reports}
    if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("A/A repeats must bind one formal R2 checkpoint")
    values = [_repeat_distance(report) for report in reports]
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if len(finite) != expected:
        raise ValueError("A/A repeats missing finite actor-future comparison")
    p99 = float(np.percentile(finite, 99))
    result = {
        "schema_version": "safedrive.r3.aa_noise_report.v1",
        "repeat_count": len(reports),
        "repeat_groups": sorted(repeat_groups),
        "aa_noise_identities": sorted(identities),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "distance_mean": float(finite.mean()),
        "distance_p50": float(np.percentile(finite, 50)),
        "distance_p99": p99,
        "aa_p99": p99,
        "finite": bool(np.isfinite(finite).all()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=168)
    args = parser.parse_args()
    report = validate(Path(args.evidence_root), expected=int(args.expected))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
