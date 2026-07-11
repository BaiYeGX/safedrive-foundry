# CARLA 0.9.16 Server 独立验证记录

日期：2026-07-11  
任务：G0-03  
结论：`COMPLETED`（`carla0916` 环境下 world/tick 与四轮 smoke 均通过）

## 已安装基线

- Server：`E:\CARLA_0.9.16\CarlaUE4.exe`
- Python API：`carla-0.9.16-cp312-cp312-win_amd64.whl`
- 首选客户端运行时：Conda 环境 `carla0916`，Python 3.12.x
- 端口：RPC 2000、streaming 2001、Traffic Manager 2002
- 启动模式：Low quality、800×600、无声；测试过离屏与窗口模式
- 日志/证据：`docs/environment/evidence/g0-03/`；四轮正式结果亦记录于本文件与任务断点

现有 0.9.15 已由用户手动删除；0.9.16 位于 `E:\CARLA_0.9.16`。本任务未删除用户资产。

## 可复现命令

启动（PowerShell，Windows）：

```powershell
Start-Process -FilePath 'E:\CARLA_0.9.16\CarlaUE4.exe' `
  -ArgumentList '/Game/Carla/Maps/Town10HD','-windowed','-ResX=800','-ResY=600','-quality-level=Low','-nosound','-carla-rpc-port=2000'
conda activate carla0916
```

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
