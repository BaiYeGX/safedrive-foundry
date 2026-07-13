# G1-02 live acceptance summary

Date: 2026-07-12 (Asia/Singapore)

## Environment

- Connection entrypoint (historical for this run): `source safedrive_foundry/config/runtime/carla_environment.sh`
- **Current project entrypoint**: `python3 scripts/sdf.py sim preflight` / `ConnectionResolver` (do not treat the shell adapter as the only path)
- Host source (this run): current WSL default gateway (no copied IP); later hosts may be `127.0.0.1` under mirrored networking
- CARLA client/server: `0.9.16` / `0.9.16`
- Map: `Carla/Maps/Town10HD_Opt`
- Runtime profile: `throughput_20hz`

## Attempts

| Evidence directory | Last durable stage | Result |
|---|---|---|
| `live-20260712-acceptance-01` | empty scenario + unique lease | Failed before single-Ego spawn: CARLA `Transform` could not be deep-copied by config hashing |
| `live-20260712-acceptance-02` | single Ego completed | Harness cleanup check treated a stale destroyed proxy as alive |
| `live-20260712-acceptance-03` | single Ego completed | Same stale-proxy criterion; subsequent independent inventory showed no residue |
| `live-20260712-acceptance-04` | NPC+camera attempt 3 completed in SQLite | Process terminated with `SIGABRT (6)` during/after cleanup |
| `live-20260712-acceptance-05` | NPC+camera attempt 3 completed in SQLite | Same `SIGABRT (6)` after switching cleanup check to server actor inventory |
| `live-20260712-acceptance-06` | trace reached `repeat_spawned` for attempt 3; SQLite then recorded it `COMPLETED` | `SIGABRT (6)` inside `complete()/close()` before the post-close trace record |

The real-Transform hashing defect was fixed and covered by a unit regression.
The live harness cleanup predicate now uses the server actor inventory rather
than dereferencing destroyed CARLA proxies.

## Verified live facts

- The second runtime was rejected while the first held the tick lease.
- Empty and single-Ego attempts reached `COMPLETED` in SQLite.
- A real Ego accepted control and advanced a CARLA frame.
- The NPC+camera attempt spawned Ego `35`, NPC `36`, and camera `37`; its trace
  and SQLite records are preserved in `live-20260712-acceptance-06`.
- After the abort, a fresh CARLA inventory reported 23 baseline actors, none of
  IDs 35/36/37, asynchronous mode, and `fixed_delta_seconds=null`; no test actor
  or world setting remained.

## Blocking result

The initial in-process harness reproduced three native aborts. The preserved
faulthandler output reported:

```text
terminate called after throwing an instance of 'std::runtime_error'
what(): trying to operate on a destroyed actor; an actor's function was called, but the actor is already destroyed.
```

The cause was Traffic Manager still controlling an autopilot NPC while the
cleanup loop destroyed it. A second issue was a CARLA RPC propagation delay in
the immediate residue check. The runtime now withdraws NPC autopilot before
destroying the NPC, waits for bounded actor-list propagation, and writes
`COMPLETED` only after verification. The parent supervisor upgrades any child
non-zero exit to `CRASHED` and compensates cleanup.

## Final acceptance

- `live-isolated-crash-02`: intentional child `SIGABRT`, faulthandler thread
  dump captured, parent Registry `CRASHED`, compensating cleanup passed with no
  residue and restored settings.
- `live-isolated-10-03`: 10 consecutive real NPC+camera child processes,
  return code 0 each, no stderr/native signal, all Registry rows
  `COMPLETED`, no residue; final CARLA actor count 23 and asynchronous world
  settings restored.

The G1-02 live acceptance is therefore verified. No gdb/core run was needed
after the fix because the required 10-run native stability gate passed; the
pre-fix faulthandler/native message identified the destroyed-autopilot actor
call site sufficiently to make the minimal cleanup fix.
