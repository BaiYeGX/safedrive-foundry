# CARLA 0.9.16 Server 独立验证记录

日期：2026-07-11  
任务：G0-03  
结论：`COMPLETED`（`carla0916` 环境下 world/tick 与四轮 smoke 均通过）

## 已安装基线

- Server：`E:\CARLA_0.9.16\CarlaUE4.exe`（WSL：`/mnt/e/CARLA_0.9.16/CarlaUE4.exe`）
- 官方 OpenDRIVE（离线地图真源）：`E:\CARLA_0.9.16\CarlaUE4\Content\Carla\Maps\OpenDrive\` ↔ `/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/`
- Python API（Windows smoke）：`carla-0.9.16-cp312-cp312-win_amd64.whl`；**WSL 客户端**使用 Linux `carla==0.9.16` + 系统/venv Python 3.12
- 端口：RPC 2000、streaming 2001、Traffic Manager 2002
- 启动模式：Low quality、800×600、无声；测试过离屏与窗口模式
- 统一启动规格：`safedrive_foundry/config/runtime/carla_start.toml`（`sdf sim ensure`）
- 日志/证据：`docs/environment/evidence/g0-03/`；四轮正式结果亦记录于本文件与任务断点

现有 0.9.15 已由用户手动删除；0.9.16 位于 `E:\CARLA_0.9.16`。本任务未删除用户资产。

**注意**：G0-03 正文偏 Windows 客户端 smoke。G1 及以后客户端权威环境是 **WSL**；连接用 `python3 scripts/sdf.py sim preflight`，不要写死 host。

## 可复现命令

启动（PowerShell，Windows）：

```powershell
Start-Process -FilePath 'E:\CARLA_0.9.16\CarlaUE4.exe' `
  -ArgumentList '/Game/Carla/Maps/Town10HD','-windowed','-ResX=800','-ResY=600','-quality-level=Low','-nosound','-dx11','-carla-rpc-port=2000'
conda activate carla0916
```

### Shader fatal（`shader compilation failures are fatal`）

这是 **Unreal 在 Windows 上编译/加载着色器失败**，不是 ROS 或规划代码错误。常见诱因：首次启动、**运行中 `load_world` 切图**、显卡驱动/DX 后端不稳定。

处理建议（按顺序）：

1. 任务管理器结束所有 `CarlaUE4` / `CarlaUE4-Win64-Shipping` / `ShaderCompileWorker`。
2. 用 **DX11 + Low** 启动（见上，或 `sdf sim ensure` 读 `carla_start.toml`）。
3. **不要**在 live 脚本里频繁 `client.load_world`；要换图请改启动参数后重启 Server。
4. 首次启动若卡住，多等几分钟（后台在编 shader）；不要反复强制结束再开。
5. 仍失败：更新 NVIDIA 驱动；或清空 CARLA 安装目录下 `Engine/Saved` 中与 shader 缓存相关的目录后再冷启动（先备份）。
6. 确认本机用的是 **独立显卡** 跑 `CarlaUE4.exe`（笔记本核显有时会炸）。

独立客户端：

```powershell
python scripts\g0\carla_server_smoke.py --timeout 10 --ticks 10
```

错误路径：

```powershell
python scripts\g0\carla_server_smoke.py --port 2099 --timeout 2 --ticks 1
```

## 已验证结果

- Server 与 client 均报告 `0.9.16`。
- `client.get_available_maps()` 返回 Town01～Town15 等地图清单。
- 2000、2001、2002 均曾由 `CarlaUE4-Win64-Shipping.exe` 监听。
- Windows GUI 显示 CARLA 与地图成功加载；GPU 进程存在。
- 错误端口 2099 在 2.3 秒内退出，错误信息清晰可读。
- 每次测试后 CARLA/UE4 进程数为 0，2000～2002 残留监听数为 0。

四轮正式 smoke（`carla0916`）共同结果：

```text
client_version = 0.9.16
server_version = 0.9.16
map = Carla/Maps/Town10HD_Opt
ticks_requested = 10
frames_strictly_increasing = true
actors_before = 23
actors_after = 23
actor_ids_added_by_test = []
elapsed_seconds = 0.170–0.179
```

四轮 frame 序列分别为 `8295–8304`、`8347–8356`、`9280–9289`、`9330–9339`。

## 已修正的早期误判

早期使用未安装 CARLA wheel 的 `(base)` 环境时，曾得到 `ModuleNotFoundError`；另一次混用不完整客户端环境时出现 `UnicodeDecodeError`。这些结果不能归因于 CARLA 0.9.16 或中文 Windows。使用 `carla0916` 并安装匹配 `carla-0.9.16-cp312-cp312-win_amd64.whl` 后，`get_world()`、地图读取和 tick 均成功。无需开启全局 UTF-8，也无需回退 0.9.15。

## 复测与恢复步骤

1. 从 `E:\CARLA_0.9.16` 启动并等待地图完全加载。
2. `conda activate carla0916`，确认 `python -c "import carla; print(carla.__file__)"` 指向该环境。
3. 执行 smoke 两次或更多，确认版本、地图、严格递增 frame 和 actor 不变。
4. 关闭 server，核对 CARLA/UE4 进程和 2000～2002 监听均为 0。
5. 未启动 server 时使用端口 2099 等错误参数，确认错误快速可读。
