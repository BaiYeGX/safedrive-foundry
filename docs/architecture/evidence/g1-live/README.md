# G1 live classic expert

## Policy

- **Do not** call `client.load_world` during the run (avoids Unreal shader fatal).
- Stay on the map CARLA was started with (authoritative closeout: Town10HD_Opt).
- Stacks: `waypoint` (bypass) | `basic` | `full` (Frenet+ST approach, default dense U-turn, RaceControl).
- Default U-turn planner: **dense** (on-road). Hybrid is optional (`--uturn-planner hybrid`), not the closeout proof.

## Authoritative successful run (G1 closeout)

| Field | Value |
|---|---|
| Pointer | **`latest_success.json`** |
| File | `g1-live-full-11-1783946091.json` |
| schema | `safedrive.g1.live_stack.v4` |
| stack | `full` |
| uturn_planner | `dense` |
| Map | `Carla/Maps/Town10HD_Opt` |
| Seed | 11 |
| Route length | ~181.5 m |
| arrived | **true** |
| modules | FrenetPlanner, ST-DP, waypoint_dense_uturn |
| control | RaceControlLoop(full) + identify |
| claims | `frenet_in_loop=true`, `race_identification_active=true`, `uturn_dense_default=true`, **`hybrid_in_loop=false`** |
| traffic | none (`no_traffic`) |

Historical bypass / older JSON under this directory are **not** stage-close evidence.

## Commands

```bash
python3 scripts/sdf.py sim preflight   # must be READY
PYTHONPATH=safedrive_foundry python3 tests/g1/run_g1_classic_expert_live.py \
  --stack full --seed 11 --max-route-m 180 --hold-s 4 --uturn-planner dense
```

## Stage status

G1 is **`COMPLETED_WITH_LIMITS`**. See `PROGRESS.md` and `docs/architecture/G1_DEVELOPMENT_AND_ACCEPTANCE_REVIEW.md`.
