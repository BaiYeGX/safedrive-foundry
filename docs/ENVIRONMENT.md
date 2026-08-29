# 环境与运行入口

本文只保留当前 H6-CORA 的有效环境事实和运行边界。历史安装/诊断位于 `archive/` 与
`docs/environment/evidence/`；历史诊断只描述当时执行上下文，不能覆盖当前资产事实。

## 1. 基线

```text
Windows 11 Pro 25H2（`versions.lock` 冻结 observed build 26200.8655；live task 仍需复查）
WSL2 Ubuntu 24.04
ROS 2 Jazzy
CARLA Server 0.9.16 on Windows
Python client/runtime/training on WSL
NVIDIA RTX 4080 16GB
Intel i5-13600KF
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

用户已确认 CARLA 与 GPU 可用。任何代理/终端仍需在实际任务上下文中重新 probe；一次
`CUDA_UNAVAILABLE`、RPC 不可达或 sandbox `PermissionError` 只证明该进程未成功访问，不能
推导物理资产不存在。

版本锁以 `versions.lock` 为准。

## 2. 目录

```text
Windows repo: E:\autonomous driving
WSL repo: /mnt/e/autonomous driving
CARLA: E:\CARLA_0.9.16
WSL CARLA view: /mnt/e/CARLA_0.9.16
VLA/World venv: /home/sdf/.venvs/sdf
```

仓库路径含空格，命令中必须引用完整路径。

## 3. 统一 CARLA 入口

```bash
cd "/mnt/e/autonomous driving"
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py sim status
python scripts/sdf.py sim preflight --json
```

状态处理：

- `READY`：本次 live task 可以继续；
- `RETRYABLE_FAILURE`：只执行一次 bounded ensure，再只复查一次；
- `NEEDS_USER_ACTION`：记录并停止；
- version/tick-owner/dependency/permission conflict：停止，不绕过；
- TCP connect 不等于 CARLA RPC READY；
- 不硬编码 host，统一通过 `scripts/sdf.py sim` 解析。

唯一允许的自动恢复形式：

```bash
python scripts/sdf.py sim ensure \
  --map Town03 \
  --rhi dx12 \
  --startup-timeout 180 \
  --json
```

## 4. Windows CARLA 边界

- 当前默认 RHI 为 DX12；
- 地图/RHI 由 `safedrive_foundry/config/runtime/carla_start.toml` 原子 pin；
- 不在正式 run 中临时反复 `client.load_world()`；
- Server 在 Windows，client/model/control 在 WSL；
- GUI/UAC 或 Windows 文件权限问题由用户接管；
- 不使用 force kill 清理不明进程。

手动启动只在用户明确接管时：

```powershell
Start-Process -FilePath 'E:\CARLA_0.9.16\CarlaUE4.exe' `
  -WorkingDirectory 'E:\CARLA_0.9.16'
```

## 5. Single tick owner

- 正式 collector/实验模式只允许登记的 `ScenarioRuntime` 成为 tick master；
- ROS G0/bridge bring-up 可单独使用 `carla_sync_driver` 作为 tick master，但这是互斥模式；
- `ScenarioRuntime` 与 `carla_sync_driver` 不得同时推进同一 CARLA endpoint；只读
  `carla_status_bridge` 不能获得 tick 权限；
- generator、World、Safety、controller、collector cleanup 不得直接调用 `world.tick()`；
- 业务脚本不得创建第二套 `carla.Client` 争抢 synchronous tick；
- 需要推进仿真必须通过 Runtime 的受控 tick API；
- cleanup 无法在唯一 owner 下完成时返回用户动作状态，不偷偷推进世界；
- 运行结束验证 asynchronous settings 恢复、tick owner free、registry terminal。

`single tick owner` 是运行期不变量，不是“仓库只能存在一个含 `world.tick()` 的文件”。合法
owner 实现可以有多个模式，但一次 run 只能激活一个；C1 必须让 preflight/lease 和 Evidence
明确解析当前 owner，而不是靠操作人员记忆避免冲突。

## 6. ROS 2

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "/mnt/e/autonomous driving/safedrive_foundry/ros_ws"
colcon build --symlink-install
source install/setup.bash
```

高频传感器默认 best effort + volatile；控制/状态/事件默认 reliable + volatile。

当前 ROS 2 能力主要是同步 tick/status bridge 和 frame/timestamp 合同；VLA→World→Safety→
Control 主链仍主要由 Python CARLA runtime 编排。文档、简历和演示不得写成完整量产 ROS 2
自动驾驶节点栈。

运行正式 Python collector 时不要同时启动 ROS tick-master launch；若只需要观测，启动只读
status bridge。需要切换模式时先让当前 owner 完整 cleanup/release，再重新 preflight。

## 7. Python/CUDA probe

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
python -c "import carla; print(carla.__file__)"
```

正式训练还记录 driver/runtime、PyTorch/CUDA、device name、available/total memory；正式在线
记录 whole-GPU peak。用户确认不能替代运行 artifact。

## 8. 常见错误

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: torch` | 激活 `/home/sdf/.venvs/sdf` |
| `ModuleNotFoundError: carla` | 使用含 Linux carla 0.9.16 的活动 venv |
| Windows Python 缺 `fcntl` | 不用 Windows Python 跑 WSL runtime |
| TCP 通、RPC 失败 | 检查 Server、host source、代理/权限和版本，不把 TCP 当 READY |
| CUDA 在一个进程不可见 | 在用户实际 WSL 终端复查驱动/权限；不直接否定 GPU |
| version mismatch | Server/client 统一到 0.9.16 |
| tick owner conflict | 停止并解析唯一 owner，不创建第二 client 绕过 |
| DXGI/D3D failure | 使用冻结 DX12 配置，避免 run 中换图 |
| shader 首次加载慢 | bounded 等待，不循环强杀重开 |

## 9. 验证集合

离线最小集合：

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry scripts tests
git diff --check
```

真实 CARLA/CUDA 任务在此基础上加 task-local probe/preflight。报告只写实际运行结果；测试
通过不等于真实 SimLingo forward、CARLA closed loop 或 formal Evidence 通过。
