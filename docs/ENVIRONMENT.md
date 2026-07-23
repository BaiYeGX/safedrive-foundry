# 环境与运行入口

本文只保留当前有效环境事实。历史 G0 安装记录和原始 Evidence 已移入 `archive/`。

## 1. 当前基线

```text
Windows 11 Pro 25H2
WSL2 Ubuntu 24.04
ROS 2 Jazzy
CARLA Server 0.9.16 on Windows
Python client/runtime on WSL
RTX 4080 16GB
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

版本锁以根目录 `versions.lock` 为准。

## 2. 目录

```text
Windows repo: E:\autonomous driving
WSL repo: /mnt/e/autonomous driving
CARLA: E:\CARLA_0.9.16
WSL CARLA view: /mnt/e/CARLA_0.9.16
VLA venv: /home/sdf/.venvs/sdf
```

仓库路径含空格，命令中必须加引号。

## 3. 统一 CARLA 入口

在 WSL 仓库根：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py sim status
python scripts/sdf.py sim preflight --json
```

CARLA 未运行且返回 `RETRYABLE_FAILURE` 时只执行一次：

```bash
python scripts/sdf.py sim ensure \
  --map Town03 \
  --rhi dx12 \
  --startup-timeout 180 \
  --json
```

随后只重新 preflight 一次。

处理规则：

- `READY`：继续；
- `RETRYABLE_FAILURE`：ensure 一次；
- `NEEDS_USER_ACTION`：记录并停止；
- version/tick owner/dependency conflict：`BLOCKED` 或 `DECISION_REQUIRED`；
- 不硬编码 CARLA host；
- TCP connect 不等于 RPC READY。

## 4. CARLA 启动边界

- 当前稳定默认 RHI 为 DX12；
- 地图/RHI 由 `carla_start.toml` 原子 pin；
- 不在运行中反复 `client.load_world()`；
- 正式业务只允许已登记 Runtime 成为 tick owner；
- 业务节点不得创建第二套 `carla.Client` 或直接 `world.tick()`；
- CARLA Server 在 Windows，客户端、ROS 和模型在 WSL。

手动启动只在自动 ensure 不适用时使用：

```powershell
Start-Process -FilePath 'E:\CARLA_0.9.16\CarlaUE4.exe' `
  -WorkingDirectory 'E:\CARLA_0.9.16'
```

## 5. ROS 2

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "/mnt/e/autonomous driving/safedrive_foundry/ros_ws"
colcon build --symlink-install
source install/setup.bash
```

高频传感器默认 best effort + volatile；控制/状态/事件默认 reliable + volatile。

## 6. 常见错误

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: torch` | 激活 `/home/sdf/.venvs/sdf` |
| `ModuleNotFoundError: carla` | 使用含 Linux carla 0.9.16 的 sdf venv |
| Windows Python 缺 `fcntl` | 不用 Windows Python 跑 runtime |
| TCP 通、RPC 失败 | 检查 Server 进程、host source、代理假连通 |
| version mismatch | 统一 Server/client 到 0.9.16 |
| DXGI/D3D failure | 使用冻结 DX12 配置，避免运行中换图 |
| shader 首次加载慢 | 等待编译，不循环强杀重开 |

## 7. 最小离线验证

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry
git diff --check
```

只报告实际运行过的命令和结果。
