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

## WSL connection recovery

Before every CARLA, ROS bridge, or runtime command, source the environment
entrypoint rather than copying a previous Windows gateway:

```bash
source safedrive_foundry/config/runtime/carla_environment.sh
```

It resolves `CARLA_HOST` from the current WSL default route and exports RPC
port `2000`, expected version `0.9.16`, and a 10-second timeout.  A task must
then record the resolved endpoint and verify the client/server versions before
it creates a world or starts a tick owner.
