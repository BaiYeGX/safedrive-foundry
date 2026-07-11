# SafeDrive G0-05 environment doctor

- Overall status: **FAIL**
- Generated (UTC): `2026-07-11T18:41:32.780960+00:00`
- Project root: `/mnt/e/autonomous driving`

| Check | Status | Code | Message |
|---|---|---|---|
| `paths.project` | **PASS** | `project_path_ok` | project root and G0 skeleton are present |
| `sync.config` | **PASS** | `sync_config_ok` | fixed-step and frame contract configuration is valid |
| `versions.lock` | **PASS** | `versions_lock_ok` | authoritative version lock is readable |
| `python.version` | **PASS** | `python_version_ok` | Python major/minor matches the preferred frozen interpreter |
| `gpu.visible` | **PASS** | `gpu_visible` | NVIDIA GPU is visible to the diagnostic process |
| `wsl.available` | **PASS** | `wsl_ready` | WSL has at least one registered distribution |
| `ros2.available` | **FAIL** | `ros2_unavailable` | ROS 2 Jazzy command is not available in WSL |
| `paths.carla` | **FAIL** | `carla_installation_missing` | configured CARLA installation path does not exist |
| `disk.free` | **PASS** | `disk_space_ok` | free disk space is above the configured G0 safety floor |
| `carla.port` | **FAIL** | `carla_not_started` | CARLA RPC TCP endpoint 172.30.80.1:2000 is unreachable |
| `carla.rpc` | **FAIL** | `carla_not_started` | CARLA RPC endpoint is unreachable; server may be stopped or the endpoint is wrong |
| `carla.port.streaming` | **WARN** | `port_not_observed` | CARLA streaming endpoint 172.30.80.1:2001 is not reachable |
| `carla.port.traffic_manager` | **WARN** | `port_not_observed` | CARLA traffic_manager endpoint 172.30.80.1:2002 is not reachable |
| `ros.clock` | **BLOCKED** | `clock_observation_blocked` | live /clock observation is blocked because ROS 2 is unavailable |
| `sync.tick_master` | **PASS** | `single_tick_master` | configuration declares exactly one tick master |

## Details

### `paths.project`

```json
{
  "root": "/mnt/e/autonomous driving"
}
```

### `sync.config`

```json
{
  "clock_topic": "/clock",
  "fixed_delta_seconds": 0.05,
  "frame_contract": "episode_id+carla_frame",
  "max_frame_gap": 1,
  "max_substep_delta_time": 0.01,
  "max_substeps": 5,
  "snapshot_topic": "/safedrive/carla/status",
  "substepping": true,
  "synchronous_mode": true,
  "tick_master": "sdf.g0-05.sync",
  "timestamp_tolerance_seconds": 1e-06
}
```

### `versions.lock`

```json
{
  "carla": "0.9.16",
  "path": "/mnt/e/autonomous driving/versions.lock"
}
```

### `python.version`

```json
{
  "current": "3.12.3",
  "expected": "3.12.3"
}
```

### `gpu.visible`

```json
{
  "returncode": 0,
  "stderr": "",
  "stdout": "NVIDIA GeForce RTX 4080, 16376 MiB, 591.86"
}
```

### `wsl.available`

```json
{
  "distros": [
    "Ubuntu-24.04"
  ],
  "version": {
    "returncode": 0,
    "stderr": "",
    "stdout": "WSL Hr,g: 2.9.3.0\n\n�Q8hHr,g: 6.18.35.2-1\n\nWSLg Hr,g: 1.0.79\n\nMSRDC Hr,g: 1.2.7214\n\nDirect3D Hr,g: 1.611.1-81528511\n\nDXCore Hr,g: 10.0.26100.1-240331-1435.ge-release\n\nWindows Hr,g: 10.0.26200.8655"
  }
}
```

### `ros2.available`

```json
{
  "returncode": 2,
  "stderr": "usage: ros2 [-h] [--use-python-default-buffering]\n            Call `ros2 <command> -h` for more detailed usage. ...\nros2: error: unrecognized arguments: --version",
  "stdout": "/opt/ros/jazzy/bin/ros2"
}
```

### `paths.carla`

```json
{
  "path": "E:/CARLA_0.9.16"
}
```

### `disk.free`

```json
{
  "free_gib": 113.583,
  "minimum_free_gib": 20.0
}
```

### `carla.port`

```json
{
  "error": "TimeoutError: timed out",
  "host": "172.30.80.1",
  "port": 2000
}
```

### `carla.rpc`

```json
{
  "error": "TimeoutError: timed out"
}
```

### `carla.port.streaming`

```json
{
  "error": "TimeoutError: timed out",
  "host": "172.30.80.1",
  "port": 2001
}
```

### `carla.port.traffic_manager`

```json
{
  "error": "TimeoutError: timed out",
  "host": "172.30.80.1",
  "port": 2002
}
```

### `ros.clock`

```json
{
  "ros_status": "FAIL"
}
```

### `sync.tick_master`

```json
{
  "tick_master": "sdf.g0-05.sync"
}
```
