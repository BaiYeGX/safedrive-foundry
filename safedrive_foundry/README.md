# SafeDrive Foundry：工程骨架与确定性同步

本目录提供 CARLA → WSL2 Python client → ROS 2 的最小状态链路、G0 确定性同步契约，以及 G1 起的 runtime / classic_stack 扩展。

它不包含 VLA、世界模型或 Safety Kernel。

## 固定约定

- CARLA Server 安装：Windows `E:\CARLA_0.9.16\CarlaUE4.exe`（WSL 路径 `/mnt/e/CARLA_0.9.16/CarlaUE4.exe`）
- VLA 本机资产与 venv：见 `docs/project/LOCAL_ASSETS.md` 与 `config/vla/local_assets.toml`  
  - 权重/代码路径已登记；**G3+ 必须** `source /home/sdf/.venvs/sdf/bin/activate`（torch 2.12.1+cu126）
- 官方 OpenDRIVE：`/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/*.xodr`
- CARLA RPC：`2000`；streaming：`2001`；Traffic Manager：`2002`
- **Host 不得写死**：由 `sdf sim preflight|status|ensure` → `runtime.carla_connection` 动态解析
- WSL2：Ubuntu 24.04 + ROS 2 Jazzy；项目代码权威环境为 **WSL 内原生 bash/python3**
- ROS domain：`42`
- 状态 topic（G0 兼容）：`/safedrive/carla/status`，类型 `std_msgs/msg/String`
- G0-05 固定步长：`fixed_delta_seconds=0.05`、`max_substep_delta_time=0.01`、`max_substeps=5`
- 唯一 tick master（业务）：只允许已登记 owner 调用 `world.tick()`（G1 为 ScenarioRuntime）

## 统一连接入口（现行）

在仓库根、**WSL** 中：

```bash
cd "/mnt/e/autonomous driving"
python3 scripts/sdf.py sim status
python3 scripts/sdf.py sim preflight   # READY 才能继续真实 CARLA 任务
# 可选一次启动（Windows CarlaUE4.exe，经 PowerShell 互操作）：
python3 scripts/sdf.py sim ensure
```

解析顺序（摘要）：

1. 显式 `CARLA_HOST` / CLI host
2. WSL 镜像网络下的 `127.0.0.1`（本机实测 Windows CARLA 常在此可达）
3. 非代理风格 default gateway
4. 代理段 `198.18.0.0/15` 等殿后（TCP 可能假连通，以 RPC handshake 为准）

兼容：`source safedrive_foundry/config/runtime/carla_environment.sh` 仅导出环境变量，**不是**业务前置；正式路径用 `sdf sim`。

历史 G0 文档里出现的 `172.30.80.1` 只是当时 NAT 网关采样，**不是**永久固定地址。

## 启动 CARLA Server

**推荐（Windows 安装包，从 WSL 也可 ensure）：**

```powershell
# Windows PowerShell
Start-Process -FilePath 'E:\CARLA_0.9.16\CarlaUE4.exe' `
  -ArgumentList '/Game/Carla/Maps/Town10HD','-windowed','-ResX=800','-ResY=600','-quality-level=Low','-nosound','-carla-rpc-port=2000'
```

或在 WSL：`python3 scripts/sdf.py sim ensure`（读取 `config/runtime/carla_start.toml`）。

说明：当前安装为 **Windows 构建**（`CarlaUE4.exe`），不是 Linux `CarlaUE4.sh`。Server 进程跑在 Windows；客户端在 WSL。
若将来提供 Linux 构建，同一 `ensure` 路径可选用 `linux_executable` / 旁路 `CarlaUE4.sh`。

防火墙只需放行用户选择的 CARLA TCP 端口；本仓库不自动改防火墙。

## WSL 客户端与 ROS

```bash
# 已在 Ubuntu-24.04 WSL 内
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# carla Python API 需已安装到当前 python3（0.9.16）
cd "/mnt/e/autonomous driving"
python3 scripts/sdf.py sim preflight
cd safedrive_foundry/ros_ws
colcon build --symlink-install
source install/setup.bash
ros2 run safedrive_carla_bridge carla_status_bridge
```

**禁止**：用 Windows Anaconda / embedded Python 跑依赖 `fcntl` 的 runtime，再据此断言“未安装 WSL”。

## ROS 验证

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic echo /safedrive/carla/status --once
```

只读 bridge 不调用 `world.tick()`。固定步长与唯一 tick master 由 sync driver / ScenarioRuntime 管理。

## 诊断与 G0 工具

```bash
# WSL 仓库根（推荐）
python3 scripts/sdf.py doctor
python3 scripts/sdf.py validate-g0
python3 scripts/sdf.py sim preflight

# Windows 仅作辅助
.\sdf.cmd doctor
```

## 故障诊断

| 现象 | 正确结论 |
|---|---|
| `IN_WSL=1` 且 `import fcntl, carla` 成功 | 工具链可用，不是“没装 WSL” |
| `process_state=NOT_RUNNING` | 启动 CARLA Server |
| `RPC_HANDSHAKE_FAILED` 且 TCP 通 | 代理假连通、Server 未就绪或 host 错；看 `sdf sim status` 的 host_source |
| `server/client version mismatch` | 统一到 0.9.16 |
| 连续 `load_world` 后 Fatal Error | 关闭残留进程；离线地图用官方 `.xodr`，勿为导出连切图 |
| `ModuleNotFoundError: carla` | 当前解释器未装 `carla==0.9.16` |
