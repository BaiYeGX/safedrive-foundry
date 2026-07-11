"""Minimal CARLA server smoke test for G0-03 (no ROS dependency)."""

from __future__ import annotations

import argparse
import json
import time

import carla


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--ticks", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    before_actor_ids = {actor.id for actor in world.get_actors()}
    frames = [world.wait_for_tick(args.timeout).frame for _ in range(args.ticks)]
    after_actor_ids = {actor.id for actor in world.get_actors()}
    result = {
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "map": world.get_map().name,
        "ticks_requested": args.ticks,
        "frames": frames,
        "frames_strictly_increasing": all(b > a for a, b in zip(frames, frames[1:])),
        "actors_before": len(before_actor_ids),
        "actors_after": len(after_actor_ids),
        "actor_ids_added_by_test": sorted(after_actor_ids - before_actor_ids),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["client_version"] != result["server_version"]:
        raise RuntimeError("CARLA client/server version mismatch")
    if not result["frames_strictly_increasing"]:
        raise RuntimeError("CARLA frames did not increase strictly")
    if result["actor_ids_added_by_test"]:
        raise RuntimeError("smoke test left actors behind")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
