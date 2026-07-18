# INVALID — G3-05 neural_live (2026-07-14)

**Status**: `INVALID` for stage close and CLAIMS C1.

## Why

1. Live runner forced throttle / open-loop steer after Safety (bypass).
2. First-available candidate used when `executed_trajectory_id` missing.
3. Episode success defined as `steps >= 80` only.
4. Seed 11/13 `decision_tail` all `EMERGENCY` while distance ≈ 140–148 m.
5. Fault timeout: per-result `sources_seen=[]` with distance ≈ 138 m; top-level sources hard-coded `["vla_fast"]`.
6. `assert_g3_close` previously only trusted `all_ok`.

## Replacement

Authoritative live evidence path after safety-bind fix:

```text
docs/architecture/evidence/g3-05/neural_live_v2/
```

Do not cite files in this directory as G3 VERIFIED.
