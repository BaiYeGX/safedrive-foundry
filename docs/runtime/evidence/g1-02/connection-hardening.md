# G1-02 post-completion connection hardening evidence

Date: 2026-07-12 (Asia/Singapore)

## Scope and entrypoint

- Unified module: `safedrive_foundry/runtime/carla_connection.py`
- CLI router: `safedrive_foundry/ros_ws/src/safedrive_carla_bridge/safedrive_carla_bridge/cli.py`
- Source shim: `scripts/sdf.py` / root `sdf`
- Compatibility shell adapter: `safedrive_foundry/config/runtime/carla_environment.sh`
- G0 files, configs, documents, and historical evidence were not modified by this hardening.

## Source-free real shell checks

Commands ran in a fresh process with `CARLA_HOST` and `CARLA_PORT` unset and
without sourcing the shell adapter:

```text
env -u CARLA_HOST -u CARLA_PORT python3 scripts/sdf.py --root . sim status --json
env -u CARLA_HOST -u CARLA_PORT python3 scripts/sdf.py --root . sim preflight --json
env -u CARLA_HOST -u CARLA_PORT python3 scripts/sdf.py --root . sim ensure --json
```

Observed result for all three: `status=READY`, `host_source=wsl_default_gateway`,
RPC port `2000`, TCP and RPC reachable, client/server `0.9.16`, map
`Carla/Maps/Town10HD_Opt`, WorldSettings readable, and
`tick_owner=free (configured=sdf.g0-05.sync)`. The gateway was resolved from
the current route at execution time; no address is embedded in code.

`ensure` returned `recovery_action=already_running` and the Windows CARLA
process count remained `2` before and after (`2 → 2`), proving no duplicate
instance was started.

## Error and recovery tests

- Connection unit tests cover explicit/environment/dynamic host precedence,
  port precedence, one changed-host retry, same-host no-retry, TCP vs RPC,
  client/server version mismatch, map/settings failures, tick-owner reporting,
  bounded ensure timeout, and `NEEDS_USER_ACTION`.
- New connection + existing G1 suite: `27/27 PASS`.
- CARLA-not-running automatic startup was not run against the real server,
  because stopping the validated active instance would disturb the current
  environment. The known-path bounded startup path is covered by injection
  tests and returns `RETRYABLE_FAILURE` on timeout or `NEEDS_USER_ACTION` when
  external interaction is required.

## Lifecycle regression after connection changes

`live-isolated-10-hardening-01/isolated_acceptance.json` records 10/10 real
NPC+camera child runs through the unified resolver: all return code 0, child
stderr empty, Registry `COMPLETED`, no actor/sensor residue, and the original
10/10 lifecycle standard remains intact.
