# G1-03 Map, Lane Graph, Route Corridor and Behavior Layer

## Scope delivered

Pure-Python classic stack packages under `safedrive_foundry/classic_stack`:

| Package | Responsibility |
|---|---|
| `map` | OpenDRIVE subset parser, lane graph builder/cache/query, topology anomalies, Oracle/Observable field tags |
| `route` | Deterministic multi-objective A* route corridor over the lane graph |
| `behavior` | Auditable behavior state machine with enter/hold/exit/timeout/suppress |

No local trajectory generation and no vehicle control are produced in this task.

## Map sources

### Acceptance maps (real CARLA 0.9.16 OpenDRIVE)

Copied from the install tree (WSL path):

```text
/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/{Town01,Town03,Town10HD}.xodr
```

into:

```text
safedrive_foundry/classic_stack/map/fixtures/carla/
docs/architecture/evidence/g1-03/carla-opendrive/
```

| Map | SHA-256 (xodr) | Approx. graph |
|---|---|---|
| Town01 | `97a7f6ac67812567…` | ~422 nodes |
| Town03 | `abf177bcd9b66cfe…` | ~1960 nodes |
| Town10HD | `5d883b799f634030…` | ~509 nodes |

Load with `load_carla_map(name)`. Manifest: `fixtures/carla/manifest.json`.

**Do not** use `client.load_world` solely to export OpenDRIVE on this host: consecutive map switches have caused CARLA fatal errors. Packaged `.xodr` is the official road topology source and is preferred for offline G1-03 acceptance.

API cross-check: a successful `world.get_map().to_opendrive()` for Town01 produced the same byte size/hash as the packaged Town01.xodr before the later map-switch crash.

### Synthetic fixtures (unit regression only)

Under `fixtures/Town0*.xodr` (small hand-authored subsets) keep fixed node IDs for deterministic route/behavior unit tests. Load with `load_map_fixture(name)`.

## Route corridor

`RoutePlanner` expands lane-graph edges with costs for length, lane-change penalty, junction penalty and speed preference. Tie breaks are seed-stable. Corridor steps carry:

- maneuver: FOLLOW / LEFT / RIGHT / STRAIGHT / STOP
- semantic: road / lane_change / junction / regulated_road
- map_hash + route_id for audit

Failure paths report `UNREACHABLE` / `MAX_EXPANSIONS` / `START_FORBIDDEN` with failure node ids.

## Behavior layer

States: CRUISE, FOLLOW, STOP, YIELD, LANE_CHANGE, AVOID, MIN_RISK.

Transitions record phase and reason. Goals include Oracle vs Observable input lists and explicitly set `emits_controls=false` and `emits_local_trajectory=false`.

## ROS adapter

`safedrive_classic_stack` projects corridors/goals into `Route.msg`-shaped dictionaries without owning ticks or controls.

## Config

`safedrive_foundry/config/classic_stack/map_route_behavior.toml`

## Connection notes (G1-03 recovery / W0)

- Prefer WSL-native `python3 scripts/sdf.py sim preflight`
- Host candidates: loopback (mirrored networking) → non-proxy gateways → proxy `198.18.0.0/15` last
- CARLA Windows install path: `E:\CARLA_0.9.16` ↔ `/mnt/e/CARLA_0.9.16`
- `sdf sim ensure` launches `CarlaUE4.exe` via PowerShell from WSL

## Verification

```text
cd /mnt/e/autonomous\ driving
PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_03_map_route_behavior -v
PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_02_connection -v
source /opt/ros/jazzy/setup.bash
cd safedrive_foundry/ros_ws && colcon build --symlink-install --packages-select safedrive_classic_stack
```

Evidence: `docs/architecture/evidence/g1-03/`.
