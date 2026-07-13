# G1-02 Scenario Runtime

`ScenarioRuntime` is the only G1 component permitted to invoke `world.tick()`.
It acquires a cross-process lease before applying CARLA synchronous settings,
configures the Traffic Manager with an explicit port and seed, spawns actors and
sensors in stable `(spawn_order, name)` order, and destroys them in reverse
order. Every completed attempt is durable in the SQLite Run Registry; failures
and interruptions are terminal non-success states and must be retried with a
new `attempt_id`/`run_id`.

Finalization is a two-phase transition: `RUNNING -> FINALIZING -> COMPLETED`.
`COMPLETED` is written only after callback admission is closed, in-flight
callbacks reach zero, the queue is drained, the worker joins, sensors/NPC/Ego
are destroyed, Traffic Manager and WorldSettings are restored, and the actor
residue check passes. Cleanup errors become `CLEANUP_FAILED`; a parent process
upgrades any native child non-zero exit to `CRASHED`, never to success.

Autopilot is withdrawn from NPCs immediately before their destruction while
Traffic Manager is still available. This prevents TM from issuing a command to
an actor that is already being destroyed; TM synchronous mode is restored only
after the required sensor → NPC → Ego order.

`run_g1_02_live_isolated.py` is the real-CARLA supervisor. It pre-registers an
attempt, launches `run_g1_02_live_child.py` with `PYTHONFAULTHANDLER=1`, captures
stdout/stderr, marks signal exits `CRASHED`, and performs compensating cleanup
and verification from the saved actor IDs and original settings.

The adapter uses duck-typed CARLA client objects to remain testable without a
simulator. In production, its client must be verified as CARLA 0.9.16 before
construction. The sensor barrier requires every declared sensor to report the
same CARLA frame returned by the sole tick. The resulting `FrameHeader` is the
identity carried by Camera, Ego, Actor, Control and `/clock` adapters.

## CARLA connection (current)

Do **not** hardcode a Windows gateway IP. Before every real-CARLA command use the
unified resolver:

```bash
cd "/mnt/e/autonomous driving"
python3 scripts/sdf.py sim preflight   # must print status=READY
# optional one-shot start of the known Windows install:
python3 scripts/sdf.py sim ensure
```

`runtime.carla_connection.ConnectionResolver` ranks candidates (explicit
`CARLA_HOST`, loopback under WSL mirrored networking, non-proxy default
gateways, proxy-like `198.18.0.0/15` last) and requires an RPC handshake.
READY is never decided by TCP alone.

Optional compatibility export (not a business prerequisite):

```bash
source safedrive_foundry/config/runtime/carla_environment.sh
```

Historical G0 samples such as `172.30.80.1` are frozen evidence, not a permanent
host. Windows install path: `E:\CARLA_0.9.16` ↔ `/mnt/e/CARLA_0.9.16`.
