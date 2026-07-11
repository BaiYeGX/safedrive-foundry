# SafeDrive Foundry：G0-04/G0-05 工程骨架与确定性同步

本目录提供 Windows CARLA → WSL2 Python client → ROS 2 的最小状态链路，以及 G0-05 的固定步长同步契约。
它不包含规划、控制、VLA、世界模型或 Safety Kernel。

## 固定约定

- CARLA Server：Windows `E:\CARLA_0.9.16\CarlaUE4.exe`
- CARLA RPC：`2000`；streaming：`2001`；Traffic Manager：`2002`
- 当前 WSL NAT host gateway：`172.30.80.1`（以 `ip route` 的 default gateway 为准）
- WSL2：Ubuntu 24.04 + ROS 2 Jazzy
- ROS domain：`42`
- 状态 topic：`/safedrive/carla/status`，类型 `std_msgs/msg/String`
- 每条 JSON 消息包含 `episode_id`、`carla_frame`、仿真时间戳、地图和 endpoint
- G0-05 固定步长：`fixed_delta_seconds=0.05`、`max_substep_delta_time=0.01`、`max_substeps=5`
- 唯一 tick master：`sdf.g0-05.sync`；只允许它调用 `world.tick()`
- `/clock`、snapshot 和同步状态消息共享 `episode_id + carla_frame`，消息还记录 `snapshot_frame`、`message_frame`、`clock_frame` 以及 command 生成/计划/执行帧

## Windows 启动

```powershell
Start-Process -FilePath 'E:\CARLA_0.9.16\CarlaUE4.exe' `
  -ArgumentList '/Game/Carla/Maps/Town10HD','-windowed','-ResX=800','-ResY=600','-quality-level=Low','-nosound','-carla-rpc-port=2000'
```

不要在 WSL 中启动 Windows Server。防火墙只需允许用户明确选择的 CARLA TCP 端口；本骨架不自动修改防火墙规则。

## WSL 安装与启动

先确认发行版存在：

```bash
wsl -l -v
```

在 Ubuntu 24.04 中：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 -m pip install --user 'carla==0.9.16'
cd /mnt/e/autonomous\ driving/safedrive_foundry/ros_ws
colcon build --symlink-install
source install/setup.bash
ros2 run safedrive_carla_bridge carla_status_bridge
```

WSL 使用 `172.30.80.1` 这一当前 host gateway。WSL 重启后若 gateway 变化，使用路由表中的 default gateway 覆盖：

```bash
export CARLA_HOST="$(ip route | awk '/default/{print $3; exit}')"
```

可通过 `CARLA_HOST`、`CARLA_PORT`、`CARLA_EXPECTED_VERSION`、`CARLA_STATUS_TOPIC` 和 `CARLA_STATUS_HZ` 覆盖配置。默认期望版本为 `0.9.16`。

## ROS 验证

另一个 WSL shell 中：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic echo /safedrive/carla/status --once
```

消息中的 `carla_frame` 必须存在且为非负整数；只读 bridge 不调用 `world.tick()`。固定步长和唯一 tick master 由 G0-05 sync driver 管理。

## G0-05 确定性同步与诊断

在项目根目录可使用 `sdf doctor`（Windows 使用 `sdf.cmd`，WSL 源码树可使用 `sdf`）：

```powershell
.\sdf.cmd doctor
.\sdf.cmd validate-g0
```

`doctor` 会输出 `PASS/WARN/FAIL/BLOCKED`，并默认写入 `docs/environment/evidence/g0-05/doctor.json` 和 `doctor.md`。它检查版本锁、路径、GPU、WSL、ROS 2、CARLA RPC/版本/端口、`/clock`、同步参数和磁盘余量。

离线烟雾测试支持双跑比较和安全断点恢复：

```powershell
.\sdf.cmd sync-smoke --seed 2026 --steps 16 --run-id repeat-1
.\sdf.cmd sync-smoke --seed 2026 --steps 16 --run-id repeat-2
.\sdf.cmd compare docs/environment/evidence/g0-05/smoke/repeat-1/trace.json docs/environment/evidence/g0-05/smoke/repeat-2/trace.json

.\sdf.cmd sync-smoke --seed 303 --steps 10 --run-id recover --interrupt-after 4
.\sdf.cmd sync-smoke --seed 303 --steps 10 --run-id recover --resume
```

真实 CARLA/ROS 运行时，先确保没有其他 tick owner，再启动唯一 driver：

```bash
source /opt/ros/jazzy/setup.bash
source ros_ws/install/setup.bash
ros2 run safedrive_carla_bridge carla_sync_driver --steps 20
```

该 driver 设置同步模式和子步参数，每次 `world.tick()` 后读取同一 snapshot，同时发布 `/clock` 与带 frame 契约的状态消息；退出时恢复原 CARLA WorldSettings。

## 故障诊断

- `ModuleNotFoundError: carla`：使用了错误 Python，确认 `python3 -m pip show carla` 与 `python3 --version`。
- `CARLA client timeout ...:2000`：确认 Windows Server 已启动、端口和 `CARLA_HOST`；不要把端口错误当成 ROS 故障。
- `server/client version mismatch`：检查 WSL `carla` 包必须是 0.9.16，禁止混用 0.9.15 API。
- 版本诊断：可临时设置错误的 `CARLA_EXPECTED_VERSION`（例如字符串 `0.9.15`）验证 mismatch 分支；这只是负向测试，不需要安装或保留 0.9.15 文件，正常运行应保持 `0.9.16`。
- `ros2 topic list` 无 topic：确认两个 shell 的 `ROS_DOMAIN_ID=42`、ROS Jazzy setup 和 bridge 进程仍在运行。
- Server 未启动测试：停止 CARLA 后运行 bridge，预期在连接超时内给出 endpoint/端口错误并退出。
