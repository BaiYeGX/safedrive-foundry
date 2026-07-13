# 本机资产路径与运行环境登记

> **状态**：`REGISTERED_AND_VERIFIED`（2026-07-14）  
> **用途**：G3+ VLA / 仿真的本机路径与 Python 环境权威说明；代码优先读环境变量与 `safedrive_foundry/config/vla/local_assets.toml`。  
> **范围**：本机开发机；换机器时改本文件与 toml，不改任务验收标准。

---

## 1. 路径总表

| 资产 | Windows 路径 | WSL 路径 | 用途 | 备注 |
|---|---|---|---|---|
| 仓库根 | `E:\autonomous driving` | `/mnt/e/autonomous driving` | SafeDrive Foundry 工程 | 路径含空格，脚本必须加引号 |
| SimLingo **代码** | `E:\autonomous driving\simlingo-main` | `/mnt/e/autonomous driving/simlingo-main` | 上游结构参考、模型/预处理移植源 | **不**作 tick master |
| SimLingo **权重** | `E:\autonomous driving\models\simlingo` | `/mnt/e/autonomous driving/models/simlingo` | G3-03 F0 主 checkpoint | 约 9.0 GB |
| InternVL2-1B **权重** | `E:\autonomous driving\models\InternVL2-1B` | `/mnt/e/autonomous driving/models/InternVL2-1B` | 后备底座 `SDF-VLA-1B-IVL` | 约 1.8 GB |
| CARLA 0.9.16 | `E:\CARLA_0.9.16` | `/mnt/e/CARLA_0.9.16` | Windows Server | `CarlaUE4.exe`；**host 动态解析** |

配置镜像：`safedrive_foundry/config/vla/local_assets.toml`。  
大体积目录已在根 `.gitignore`：`simlingo-main/`、`models/`。

---

## 2. 主 / 备权重策略

| 优先级 | 路径 | 角色 |
|---|---|---|
| **主** | `models/simlingo` | SimLingo research lineage；仿真研究默认 |
| **备** | `models/InternVL2-1B` | 仅当主权重 F0 兼容/显存/闭环无法解决时启用；**重跑 F0～F3**，不继承主权重成绩 |

### 本机 SimLingo 权重布局

```text
E:\autonomous driving\models\simlingo\
└── simlingo\checkpoints\epoch=013.ckpt\
    ├── pytorch_model.pt          # 推理优先
    ├── zero_to_fp32.py
    └── checkpoint\               # DeepSpeed/zero 分片
```

F0 加载顺序：`pytorch_model.pt` → 必要时按上游脚本从 zero 合并（合并产物另存，勿覆盖原目录）。精确 hash 在 G3-03 Evidence 冻结。

---

## 3. Python 环境分工（为什么不用 Windows `carla0916` 跑 G3）

本机实际有多套环境，**名字容易混**，职责不同：

| 环境 | 在哪 | 主要有什么 | 该干什么 | 不该干什么 |
|---|---|---|---|---|
| **`carla0916`（conda）** | **Windows** Anaconda | Windows 版 `carla` 0.9.16 wheel | G0-03 **Windows 侧** Server smoke（`scripts/g0/carla_server_smoke.py`） | **不要**跑 WSL Runtime / ROS / SimLingo / G3 VLA |
| **`/home/sdf/.venvs/sdf`** | **WSL** venv | `torch 2.12.1+cu126` + CUDA + Linux `carla` | **G3+ VLA 默认**；G1+ 权威客户端侧 | 勿与系统 python 混用 |
| **`/home/sdf/.venvs/carla_ros`** | **WSL** venv | 主要是 Linux `carla`（**无 torch**） | 早期 CARLA–ROS 连通/桥接 | **不要**跑 SimLingo |
| 系统 `/usr/bin/python3` | WSL | 系统包；PEP 668 | 系统工具 | **不要** `pip install` torch；**不要**当 G3 解释器 |
| Windows `(base)` / 任意 Anaconda | Windows | 可能无 carla 或环境不完整 | 一般不用 | Runtime 依赖 `fcntl`，**禁止**当 G1+ 客户端 |

### 为什么 G3 用 `sdf` venv，不用 `carla0916`

1. **操作系统边界**：G1 起权威客户端在 **WSL**（ROS 2、`fcntl`、`sdf sim`、ScenarioRuntime）。`carla0916` 是 **Windows conda**，跑不了这套 Linux 运行时。  
2. **能力边界**：`carla0916` 解决的是 **Windows 上 import carla + 连本机 Server 做 smoke**（见 `docs/environment/CARLA_SERVER_BASELINE.md`）。G3 需要 **CUDA PyTorch + 加载 SimLingo**，这些装在 WSL 的 `sdf` venv 里并已验证。  
3. **版本基线**：G0-02 冻结的是 WSL `/home/sdf/.venvs/sdf` + `torch 2.12.1+cu126`，不是把 Windows conda 当训练环境。  
4. **不是废弃 `carla0916`**：Windows 上仍可用它做 Server/API smoke；**只是不要拿它当 G3 主环境**。

文档出处：

- Windows + `carla0916`：`docs/environment/CARLA_SERVER_BASELINE.md`  
- WSL + `sdf` venv：`docs/environment/WSL_ROS2_BASELINE.md`  
- 勿用 Windows/Anaconda 跑 runtime：`docs/project/ENVIRONMENT_AND_DOC_CORRECTIONS.md`

### 3.1 G3+ 必须使用的 venv

| 项 | 值 |
|---|---|
| **项目 venv（G3+ VLA 唯一默认）** | `/home/sdf/.venvs/sdf` |
| Python | 3.12.3 |
| PyTorch（已验证 2026-07-14） | **`2.12.1+cu126`**，`torch.cuda.is_available() == True` |
| GPU | NVIDIA GeForce RTX 4080 |
| CUDA 实算抽查 | `torch.arange(1024, device="cuda").sum() == 523776` |
| `carla` | 该 venv 内 **可 import**（Linux 客户端） |

```bash
# 做 VLA / G3 训练与推理前必须激活
source /home/sdf/.venvs/sdf/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.12.1+cu126 True
```

PowerShell 一次性示例：

```powershell
wsl -d Ubuntu-24.04 -- /usr/bin/bash -lc "source /home/sdf/.venvs/sdf/bin/activate; cd '/mnt/e/autonomous driving'; python scripts/sdf.py doctor"
```

### 3.2 Windows `carla0916`（可选、仅 smoke）

```powershell
conda activate carla0916
python -c "import carla; print(carla.__file__)"
# 仅在 Windows 上连本机 CarlaUE4 做 G0 式 smoke 时使用
```

### 3.3 禁止 / 易错

1. Ubuntu 24.04 系统 Python 受 **PEP 668** 保护：禁止对系统 `pip install`；禁止 `--break-system-packages`。  
2. 用系统 `python3` 测 torch 会 `ModuleNotFoundError` → **应测 `sdf` venv**，不要重装整机。  
3. 用 Windows Anaconda / `carla0916` 跑 `scenario_runtime` 等会因 **`fcntl`** 等失败 → 换 WSL `sdf` venv。  
4. 安装/补齐 torch：`scripts/maintenance/install_torch_cu126.sh`（只写入 `sdf` venv）。  
5. 检查：`check_sdf_venv.sh`、`local_env_check.sh`。

### 3.3 2026-07-14 实测结论（摘要）

| 检查项 | 结果 |
|---|---|
| 资产 13 路径 | 全 OK |
| E 盘空闲 | ≥100 GB 级 |
| Windows / WSL GPU | RTX 4080 可见 |
| `sdf` venv torch | `2.12.1+cu126` + CUDA True |
| ROS 2 Jazzy | OK |
| task catalog | PASS |
| CARLA 未启动时 preflight | `RPC_HANDSHAKE_FAILED` / `NOT_RUNNING` → **预期**，非坏盘 |
| 端口 TCP 通但 RPC 失败 | 常见网关/代理假连通（如 `198.18.0.x`）；以 **进程在跑 + READY** 为准 |

**下载与 PyTorch：本机已齐，无需再装 torch。**  
live 任务前再起 CARLA，执行 `sdf sim preflight` 至 `READY`。

---

## 4. 使用边界（强制）

1. **CARLA host/IP 不得写死**；只登记安装根路径。连接走 `sdf sim preflight|ensure|status`。  
2. **不得**用 `simlingo-main` 内 Leaderboard/ScenarioRunner 替换本仓库唯一 tick owner。  
3. 上游可能写 CARLA **0.9.15**；本仓库仿真固定 **0.9.16**，只移植模型与预处理。  
4. 官方 **全量 dataset** 默认不下载；训练以项目自采为主。  
5. 仿真研究默认 `deployment_scope=simulation_research_only`；代码/权重/数据许可分开审计。  
6. G3+ 代理与脚本：默认解释器应为 **`/home/sdf/.venvs/sdf/bin/python`**（或激活后的 `python`），不得默认系统 `python3` 跑 torch。

---

## 5. 环境变量（可选覆盖）

```bash
export SDF_VENV="/home/sdf/.venvs/sdf"
export SDF_PYTHON="/home/sdf/.venvs/sdf/bin/python"
export SDF_SIMLINGO_CODE_ROOT="/mnt/e/autonomous driving/simlingo-main"
export SDF_SIMLINGO_CKPT_ROOT="/mnt/e/autonomous driving/models/simlingo"
export SDF_INTERNVL2_1B_ROOT="/mnt/e/autonomous driving/models/InternVL2-1B"
export CARLA_ROOT="/mnt/e/CARLA_0.9.16"
```

未设置时回退 `safedrive_foundry/config/vla/local_assets.toml`。

---

## 6. 与任务的关系

| 任务 | 如何使用本登记 |
|---|---|
| G3-01～02 | 可用 sdf venv；可不启 CARLA |
| G3-03 F0 | **必须** sdf venv；主路径 SimLingo；失败才 InternVL2-1B |
| G3-04/05 | 同一 venv + 权重根；live 前 preflight |
| 任意 live | CARLA 进程运行 + `READY`；忽略未开时的 handshake 失败 |

启动实现：`读取 START_TASK.md，启动 G3-01。`
