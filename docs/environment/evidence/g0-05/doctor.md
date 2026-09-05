# SafeDrive G0-05 environment doctor

- Overall status: **FAIL**
- Generated (UTC): `2026-09-05T14:39:38.124674+00:00`
- Project root: `/mnt/e/autonomous driving`

| Check | Status | Code | Message |
|---|---|---|---|
| `paths.project` | **PASS** | `project_path_ok` | project root and G0 skeleton are present |
| `sync.config` | **PASS** | `sync_config_ok` | fixed-step and frame contract configuration is valid |
| `versions.lock` | **PASS** | `versions_lock_ok` | authoritative version lock is readable |
| `python.version` | **PASS** | `python_version_ok` | Python major/minor matches the preferred frozen interpreter |
| `gpu.visible` | **FAIL** | `gpu_not_visible` | NVIDIA GPU is not visible to the diagnostic process |
| `wsl.available` | **PASS** | `wsl_in_guest` | diagnostic process is already running inside WSL |
| `ros2.available` | **PASS** | `ros2_ready_in_guest` | ROS 2 Jazzy command is available in current WSL |
| `paths.carla` | **PASS** | `carla_path_ok` | configured CARLA installation path exists |
| `disk.free` | **PASS** | `disk_space_ok` | free disk space is above the configured G0 safety floor |
| `carla.port` | **FAIL** | `carla_not_started` | CARLA RPC TCP endpoint 172.30.80.1:2000 is unreachable |
| `carla.rpc` | **FAIL** | `carla_not_started` | CARLA RPC endpoint is unreachable; server may be stopped or the endpoint is wrong |
| `carla.port.streaming` | **WARN** | `port_not_observed` | CARLA streaming endpoint 172.30.80.1:2001 is not reachable |
| `carla.port.traffic_manager` | **WARN** | `port_not_observed` | CARLA traffic_manager endpoint 172.30.80.1:2002 is not reachable |
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
  "returncode": 255,
  "stderr": "",
  "stdout": "Failed to initialize NVML: GPU access blocked by the operating system\nFailed to properly shut down NVML: GPU access blocked by the operating system"
}
```

### `wsl.available`

```json
{
  "distro": "Ubuntu-24.04",
  "mode": "in_guest"
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
  "free_gib": 123.971,
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
  "returncode": 1,
  "stderr": "Failed opening file /home/sdf/.ros/log/python3_22_1788619178037.log for writing: Read-only file system",
  "stdout": ""
}
```

### `sync.tick_master`

```json
{
  "tick_master": "sdf.g0-05.sync"
}
```
