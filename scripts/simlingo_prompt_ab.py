#!/usr/bin/env python3
"""Run one-image SimLingo navigation-conditioning A/B without CARLA.

The diagnostic holds image, speed, target points, checkpoint and preprocessing
fixed.  Only the released checkpoint's navigation prompt changes between
target-point, left-lane-change HLC and right-lane-change HLC.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT / "simlingo-main"))

from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-npy", type=Path, required=True)
    parser.add_argument("--speed-mps", type=float, default=3.0)
    parser.add_argument("--target1", type=float, nargs=2, default=(8.0, 0.0))
    parser.add_argument("--target2", type=float, nargs=2, default=(18.0, 0.0))
    parser.add_argument("--image-layout", choices=("bgr", "rgb"), default="bgr")
    parser.add_argument(
        "--probe-left-distances",
        type=int,
        nargs="*",
        default=(),
        help="also probe left-HLC commands at these integer distances",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-directional",
        action="store_true",
        help="exit 2 unless left shifts left and right shifts right versus target-point",
    )
    return parser


def _metrics(route_xy: np.ndarray) -> dict[str, object]:
    route = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    tail = route[max(0, len(route) // 2) :]
    delta = np.diff(route, axis=0)
    return {
        "route_xy": route.tolist(),
        "endpoint_x_m": float(route[-1, 0]),
        "endpoint_y_m": float(route[-1, 1]),
        "tail_mean_y_m": float(tail[:, 1].mean()),
        "min_y_m": float(route[:, 1].min()),
        "max_y_m": float(route[:, 1].max()),
        "arc_length_m": float(np.linalg.norm(delta, axis=1).sum()),
    }


def main() -> int:
    args = _parser().parse_args()
    image = np.load(args.image_npy, allow_pickle=False)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 HxWx3 image, got shape={image.shape} dtype={image.dtype}"
        )

    target1 = (float(args.target1[0]), float(args.target1[1]))
    target2 = (float(args.target2[0]), float(args.target2[1]))
    distance_m = max(0, int(math.hypot(*target1)))
    prompts = {
        "target_point": ("target_point", None),
        "left": (
            "command",
            f"Command: do a lane change to the left in {distance_m} meter "
            "then follow the road.",
        ),
        "right": (
            "command",
            f"Command: do a lane change to the right in {distance_m} meter "
            "then follow the road.",
        ),
    }
    for probe_distance in args.probe_left_distances:
        probe_distance = max(0, int(probe_distance))
        prompts[f"left_{probe_distance}m"] = (
            "command",
            f"Command: do a lane change to the left in {probe_distance} meter "
            "then follow the road.",
        )

    runtime = SimLingoNeuralRuntime()
    report = runtime.load()
    if not report.ok:
        raise RuntimeError(report.error)
    runtime.keep_model_on_gpu()
    results: dict[str, object] = {}
    try:
        for name, (mode, command_text) in prompts.items():
            forward = runtime.forward_numpy(
                image,
                speed_mps=float(args.speed_mps),
                target_point_xy=target1,
                target_point2_xy=target2,
                keep_on_gpu=True,
                borrow_gpu=False,
                image_layout=args.image_layout,
                official_contract=True,
                eval_route_as=mode,
                command_text=command_text,
            )
            results[name] = {
                "eval_route_as": mode,
                "command_text": command_text,
                "latency_ms": float(forward.latency_s * 1000.0),
                **_metrics(forward.route_xy),
            }
    finally:
        runtime.release_gpu_for_carla()

    tp_y = float(results["target_point"]["tail_mean_y_m"])  # type: ignore[index]
    left_y = float(results["left"]["tail_mean_y_m"])  # type: ignore[index]
    right_y = float(results["right"]["tail_mean_y_m"])  # type: ignore[index]
    left_shift = left_y - tp_y
    right_shift = right_y - tp_y
    directional = bool(left_shift < -0.25 and right_shift > 0.25)
    output = {
        "schema_version": "simlingo_prompt_ab.v1",
        "input": {
            "image_npy": str(args.image_npy.resolve()),
            "image_shape": list(image.shape),
            "image_layout": args.image_layout,
            "speed_mps": float(args.speed_mps),
            "target1": list(target1),
            "target2": list(target2),
        },
        "results": results,
        "comparison": {
            "coordinate_convention": "+y is CARLA-right; left is negative y",
            "left_tail_shift_vs_target_point_m": left_shift,
            "right_tail_shift_vs_target_point_m": right_shift,
            "right_minus_left_tail_separation_m": right_y - left_y,
            "directional_threshold_m": 0.25,
            "directional": directional,
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    return 0 if directional or not args.require_directional else 2


if __name__ == "__main__":
    raise SystemExit(main())
