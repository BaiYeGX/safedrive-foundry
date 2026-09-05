#!/usr/bin/env python3
"""Materialize and collect H6-CORA C2 potential outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from data_pipeline.h6.cora.config import CORA_C2_CONFIG  # noqa: E402
from data_pipeline.h6.cora.live import collect_map, freeze_manifest, materialize_map  # noqa: E402
from data_pipeline.h6.cora.live_repair import collect_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default=str(CORA_C2_CONFIG["dataset_id"]),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize-map")
    materialize.add_argument("--map", required=True, choices=tuple(CORA_C2_CONFIG["maps"]))
    commands.add_parser("freeze-manifest")
    commands.add_parser("repair-diagnostic")
    repair_batch = commands.add_parser("repair-batch")
    repair_batch.add_argument("--batch", type=int, default=0)
    collect = commands.add_parser("collect-map")
    collect.add_argument("--map", required=True, choices=tuple(CORA_C2_CONFIG["maps"]))
    collect.add_argument("--scope", required=True, choices=("smoke", "pilot", "development"))
    args = parser.parse_args()
    try:
        if args.command == "repair-diagnostic":
            result = collect_plan(ROOT / "safedrive_foundry/config/h6/cora_c2_repair.toml", diagnostic=True)
        elif args.command == "repair-batch":
            result = collect_plan(ROOT / "safedrive_foundry/config/h6/cora_c2_repair.toml", diagnostic=False, batch=args.batch)
        elif args.command == "materialize-map":
            result = materialize_map(args.dataset_id, args.map)
        elif args.command == "freeze-manifest":
            result = freeze_manifest(args.dataset_id)
        else:
            result = collect_map(args.dataset_id, args.map, args.scope)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}:{exc}"},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
