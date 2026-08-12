#!/usr/bin/env python3
"""Build and freeze ActionBranchDatasetV0 from completed paired attempts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.world.dataset import sample_from_attempt, write_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--min-comparable", type=int, default=256)
    parser.add_argument("--min-decisive", type=int, default=64)
    parser.add_argument("--min-wins-per-slot", type=int, default=16)
    parser.add_argument("--min-future-coverage", type=float, default=0.95)
    parser.add_argument("--campaign-manifest", default="")
    parser.add_argument("--min-test-samples", type=int, default=0)
    parser.add_argument("--min-test-dual", type=int, default=0)
    parser.add_argument("--min-test-decisive", type=int, default=0)
    parser.add_argument("--min-test-wins-per-slot", type=int, default=0)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--r2-checkpoint-sha256", default="")
    args = parser.parse_args()
    attempts: list[Path] = []
    for raw in args.evidence_root:
        root = Path(raw)
        if "r2-spatial-k2-pilot-v5-blind" in str(root):
            raise SystemExit("frozen R2 blind Evidence is audit-only; refusing training dataset")
        for path in root.rglob("pair_manifest.json"):
            if not (path.parent / "branch-0/oracle/actor_future_trace.jsonl").is_file():
                continue
            try:
                status = str(json.loads(path.read_text(encoding="utf-8")).get("status"))
            except Exception:
                continue
            if status == "COMPLETED":
                attempts.append(path.parent)
    unique = sorted(set(attempts))
    if not unique:
        raise SystemExit("no completed R3 attempts with actor future traces found")
    samples = [sample_from_attempt(path) for path in unique]
    split_by_sample = None
    if args.campaign_manifest:
        campaign = json.loads(
            Path(args.campaign_manifest).read_text(encoding="utf-8")
        )
        scenario_splits = {
            str(row["scenario_id"]): str(row["split"])
            for row in campaign.get("slots", [])
        }
        missing = sorted(
            {
                sample.identity.scenario_id
                for sample in samples
                if sample.identity.scenario_id not in scenario_splits
            }
        )
        if missing:
            raise SystemExit(f"campaign split missing scenarios: {missing[:10]}")
        split_by_sample = {
            sample.identity.sample_id: scenario_splits[sample.identity.scenario_id]
            for sample in samples
        }
    manifest = write_dataset(
        samples,
        Path(args.output),
        split_by_sample=split_by_sample,
        shard_size=args.shard_size,
        quality_thresholds={
            "min_comparable": args.min_comparable,
            "min_decisive": args.min_decisive,
            "min_wins_per_slot": args.min_wins_per_slot,
            "min_future_coverage": args.min_future_coverage,
            "min_test_samples": args.min_test_samples,
            "min_test_dual": args.min_test_dual,
            "min_test_decisive": args.min_test_decisive,
            "min_test_wins_per_slot": args.min_test_wins_per_slot,
        },
        namespace=args.namespace,
        r2_checkpoint_sha256=args.r2_checkpoint_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
