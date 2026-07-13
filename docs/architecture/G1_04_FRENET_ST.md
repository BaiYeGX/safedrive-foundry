# G1-04 Frenet Lattice + S-T Speed (baseline freeze)

## Purpose

Provide a **fixed-sampling** Frenet lattice planner with S-T DP speed and jerk smoothing as the immutable fair baseline for G1-07 RACE-Plan.

## Packages

| Path | Role |
|---|---|
| `classic_stack/geometry/` | Reference path, Frenet frame, vehicle params |
| `classic_stack/planning/frenet/` | Lattice sampling, plan API, scenarios |
| `classic_stack/planning/speed/` | CV/CTRV/IDM prediction, ST occupancy, DP, jerk smooth |
| `config/classic_stack/frenet_st_baseline.toml` | Frozen sampling/costs/prediction/ST grid |

## Output schema

Each `TrajectoryPoint`: `t, x, y, yaw, kappa, v, a, jerk`.

## Baseline comparison

`CenterlineConstantSpeedPlanner` is the simple baseline (centerline, constant/linear stop speed).

## G1-07 rule

Do **not** silently edit `frenet_st_baseline.toml` to claim gains. Record and compare `config_hash` (SHA-256 of file bytes).

## Verification

```bash
cd "/mnt/e/autonomous driving"
PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_04_frenet -v
```

Evidence: `docs/architecture/evidence/g1-04/`.
