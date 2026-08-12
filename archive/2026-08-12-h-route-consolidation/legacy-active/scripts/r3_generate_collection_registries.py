#!/usr/bin/env python3
"""Generate deterministic R3 development variants from the non-blind V1 fixtures.

Geometry lineage remains explicit in the scenario suffix. Variants change
simulation-time actor controls and weather only; each generated registry still
requires the normal CARLA dry-run before it can be frozen or collected.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "safedrive_foundry/config/g4a/scenario_registry_v1.toml"
SCENARIOS = (
    "lead_brake_moderate",
    "lead_brake_hard",
    "cut_in_early",
    "cut_in_late",
    "cross_vehicle_clear",
    "cross_vehicle_tight",
)


def build_variant(base_text: str, index: int) -> str:
    prefix = f"r3v{index:02d}_"
    text = base_text
    text = text.replace(
        'registry_version = "v1"',
        f'registry_version = "r3-development-v{index:02d}-20260728"',
    )
    text = text.replace(
        'description = "R2 pilot: lead braking / cut-in / crossing on Town03 roads"',
        f'description = "R3 development actor-future collection variant {index:02d}"',
    )
    text = text.replace(
        'vla_config_ref = "config/vla/k2_v1.toml"',
        'vla_config_ref = "config/vla/k2_v2_spatial.toml"',
    )
    for scenario in SCENARIOS:
        text = text.replace(f"scenarios.{scenario}", f"scenarios.{prefix}{scenario}")
    # Keep exact snapped geometry, but vary observable conditions and actor scripts.
    cloudiness = min(70.0, 5.0 + 3.0 * index)
    wetness = min(40.0, 2.0 * (index % 10))
    text = re.sub(r"cloudiness = [0-9.]+", f"cloudiness = {cloudiness:.1f}", text)
    text = re.sub(r"wetness = [0-9.]+", f"wetness = {wetness:.1f}", text)
    brake_scale = 0.80 + 0.04 * (index % 9)
    throttle_scale = 0.85 + 0.03 * (index % 7)

    text = re.sub(
        r"(?m)^brake = ([0-9.]+)$",
        lambda m: f"brake = {min(1.0, float(m.group(1)) * brake_scale):.4f}",
        text,
    )
    text = re.sub(
        r"(?m)^throttle = ([0-9.]+)$",
        lambda m: f"throttle = {min(1.0, float(m.group(1)) * throttle_scale):.4f}",
        text,
    )
    # V1's hard-brake seed_b can receive one inconsistent initial-velocity
    # physics step during cold rebuild. For R3 variants it is intentionally a
    # stopped-lead case, which is both deterministic and decision-relevant.
    stopped_lead_header = (
        rf"(\[scenarios\.{prefix}lead_brake_hard\.seeds\.seed_b"
        rf"\.actors\.initial_velocity\]\n)vx = [^\n]+\nvy = [^\n]+"
    )
    text = re.sub(
        stopped_lead_header,
        r"\1vx = 0.000000\nvy = 0.000000",
        text,
    )
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = BASE.read_text(encoding="utf-8")
    for index in range(args.start_index, args.start_index + args.count):
        path = output / f"scenario_registry_r3_v{index:02d}.toml"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(build_variant(base, index), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
