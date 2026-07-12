# SafeDrive G0-05 environment doctor

- Overall status: **WARN**
- Generated (UTC): `2026-07-11T19:24:06.476976+00:00`
- Project root: `/mnt/e/autonomous driving`

| Check | Status | Code | Message |
|---|---|---|---|
| `paths.project` | **PASS** | `project_path_ok` | project root and G0 skeleton are present |
| `sync.config` | **PASS** | `sync_config_ok` | fixed-step and frame contract configuration is valid |
| `versions.lock` | **PASS** | `versions_lock_ok` | authoritative version lock is readable |
| `python.version` | **PASS** | `python_version_ok` | Python major/minor matches the preferred frozen interpreter |
| `gpu.visible` | **PASS** | `gpu_visible` | NVIDIA GPU is visible to the diagnostic process |
| `wsl.available` | **PASS** | `wsl_ready` | WSL has at least one registered distribution |
| `ros2.available` | **PASS** | `ros2_ready` | ROS 2 Jazzy command is available in WSL |
| `paths.carla` | **PASS** | `carla_path_ok` | configured CARLA installation path exists |
| `disk.free` | **PASS** | `disk_space_ok` | free disk space is above the configured G0 safety floor |
| `carla.port` | **PASS** | `port_open` | CARLA RPC TCP endpoint 172.30.80.1:2000 is reachable |
| `carla.handshake` | **PASS** | `carla_ready` | CARLA RPC, client/server version and world handshake passed |
| `carla.port.streaming` | **PASS** | `port_open` | CARLA streaming endpoint 172.30.80.1:2001 is reachable |
| `carla.port.traffic_manager` | **PASS** | `port_open` | CARLA traffic_manager endpoint 172.30.80.1:2002 is reachable |
| `ros.clock` | **WARN** | `clock_topic_not_observed` | ROS 2 is available but /clock is not currently observable; start the sync driver |
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
  "returncode": 0,
  "stderr": "",
  "stdout": "/opt/ros/jazzy/bin/ros2\nusage: ros2 [-h] [--use-python-default-buffering]\n            Call `ros2 <command> -h` for more detailed usage. ...\n\nros2 is an extensible command-line tool for ROS 2.\n\noptions:\n  -h, --help            show this help message and exit\n  --use-python-default-buffering\n                        Do not force line buffering in stdout and instead use\n                        the python default buffering, which might be affected\n                        by PYTHONUNBUFFERED/-u and depends on whatever stdout\n                        is interactive or not\n\nCommands:\n  action     Various action related sub-commands\n  bag        Various rosbag related sub-commands\n  component  Various component related sub-commands\n  daemon     Various daemon related sub-commands\n  doctor     Check ROS setup and other potential issues\n  interface  Show information about ROS interfaces\n  launch     Run a launch file\n  lifecycle  Various lifecycle related sub-commands\n  multicast  Various multicast related sub-commands\n  node       Various node related sub-commands\n  param      Various param related sub-commands\n  pkg        Various package related sub-commands\n  plugin     Various plugin related sub-commands\n  run        Run a package specific executable\n  security   Various security related sub-commands\n  service    Various service related sub-commands\n  topic      Various topic related sub-commands\n  wtf        Use `wtf` as alias to `doctor`\n\n  Call `ros2 <command> -h` for more detailed usage."
}
```

### `paths.carla`

```json
{
  "path": "/mnt/e/CARLA_0.9.16"
}
```

### `disk.free`

```json
{
  "free_gib": 97.617,
  "minimum_free_gib": 20.0
}
```

### `carla.port`

```json
{
  "error": null,
  "host": "172.30.80.1",
  "port": 2000
}
```

### `carla.handshake`

```json
{
  "client": "0.9.16",
  "server": "0.9.16"
}
```

### `carla.port.streaming`

```json
{
  "error": null,
  "host": "172.30.80.1",
  "port": 2001
}
```

### `carla.port.traffic_manager`

```json
{
  "error": null,
  "host": "172.30.80.1",
  "port": 2002
}
```

### `ros.clock`

```json
{
  "returncode": 0,
  "stderr": "",
  "stdout": "/parameter_events\n/rosout"
}
```

### `sync.tick_master`

```json
{
  "tick_master": "sdf.g0-05.sync"
}
```
