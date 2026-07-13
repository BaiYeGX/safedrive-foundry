# G1-05 Hybrid A* – Reeds–Shepp (baseline freeze)

## Purpose

Complex maneuver planner independent of Frenet road-following assumptions. Frozen basic search for G1-07 fair optimization.

## Packages

| Path | Role |
|---|---|
| `classic_stack/planning/hybrid_astar/` | Hybrid A*, RS expansion, Dubins grid baseline, selector |
| `config/classic_stack/hybrid_astar_baseline.toml` | Resolution, steer set, budgets |

## Output

Same `Trajectory` / `TrajectoryPoint` schema as G1-04 (`source=hybrid_astar`).

## Selector

`PlannerSelector` uses features only: `require_reverse`, `narrow`, `blocked` — **not** scenario name strings.

## G1-07 rule

Do not silently edit `hybrid_astar_baseline.toml`. Compare `config_hash`.

## Verification

```bash
PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_05_hybrid_astar -v
```

Evidence: `docs/architecture/evidence/g1-05/`.
