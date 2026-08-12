#!/usr/bin/env python3
"""Validate one captured 15–20 s R2 V3 trace without running CARLA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.long_horizon_observer import (  # noqa: E402
    LongHorizonObserver,
)
from driving_vla.model.navigation_contract import RouteContextV3  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-context", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--conflict-side", default="")
    parser.add_argument("--conflict-point-s-m", type=float)
    parser.add_argument("--final-actor-lon-m", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()

    route_value = json.loads(Path(args.route_context).read_text(encoding="utf-8"))
    route_context = RouteContextV3.from_mapping(route_value)
    observer = LongHorizonObserver(
        case_id=args.case_id,
        route_context=route_context,
        behavior=args.behavior,
        conflict_side=args.conflict_side,
        conflict_point_s_m=args.conflict_point_s_m,
    )
    with Path(args.trace).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                observer.observe(json.loads(line))
    report = observer.finalize(final_actor_lon_m=args.final_actor_lon_m).to_dict()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
