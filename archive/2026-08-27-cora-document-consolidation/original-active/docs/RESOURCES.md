# 本机资产与资源预算

本文是本机路径、Python 环境和单机资源边界的唯一活动说明。配置文件仍是机器可读
真源；换机器时改配置，不改研究验收标准。

## 1. 固定硬件

```text
GPU: NVIDIA RTX 4080 Desktop 16GB
CPU: Intel i5-13600KF, 14 cores / 20 threads
CARLA Server: Windows
ROS 2 / runtime / training: WSL2 Ubuntu 24.04
```

Windows CARLA 与 WSL CUDA 共享同一张物理 GPU。不得把服务器、第二张 GPU 或远程
训练服务当作必需依赖。

## 2. 本机路径

| 资产 | WSL 路径 | 用途 |
|---|---|---|
| 仓库 | `/mnt/e/autonomous driving` | 项目根 |
| SimLingo code | `/mnt/e/autonomous driving/simlingo-main` | 上游模型/预处理参考 |
| SimLingo weights | `/mnt/e/autonomous driving/models/simlingo` | 默认 checkpoint |
| InternVL2-1B | `/mnt/e/autonomous driving/models/InternVL2-1B` | 后备底座 |
| CARLA 0.9.16 | `/mnt/e/CARLA_0.9.16` | Windows Server |

机器可读配置：

```text
safedrive_foundry/config/vla/local_assets.toml
safedrive_foundry/config/runtime/carla_start.toml
```

`simlingo-main` 不能取代本项目 Runtime 或成为 tick master。

## 3. Python 环境

H 路线唯一默认：

```text
/home/sdf/.venvs/sdf
Python 3.12.3
PyTorch 2.12.1+cu126
CUDA available
Linux carla 0.9.16
```

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

禁止：

- 用系统 Python 安装/运行 SimLingo；
- `--break-system-packages`；
- 用 Windows Anaconda 跑 WSL runtime；
- 用 `/home/sdf/.venvs/carla_ros` 跑 VLA（该环境无 torch）。

Windows `carla0916` conda 仅用于 Windows 侧 Server/API smoke。

## 4. Workload profile

| Profile | GPU 主任务 | 必须关闭 |
|---|---|---|
| `vla_eval` | nominal VLA 离线/在线评估 | World train |
| `world_train` | H World scorer | CARLA、VLA eval |
| `online_eval` | CARLA + VLA + World | 所有训练优化器 |
| `data_collect` | CARLA + 轻量采集 | 训练 |
| `regression` | CARLA + 单配置 | 其他配置与训练 |

总控应拒绝冲突 profile，不等待 OOM 后恢复。

## 5. 在线显存目标

| 占用 | 初始目标 |
|---|---:|
| CARLA Low/No Rendering | 约 4–5GB，必须实测 |
| 量化 VLA | 约 5–6GB |
| H World 增量 | ≤约 1.5GB |
| context/cache/余量 | ≥约 2GB |
| whole-GPU peak | ≤约 14–14.5GB |

这些是 admission target，不是已经验证的结果。资源不足时依次降低渲染质量、图像
分辨率、历史长度和 World batch；不得改变 H candidate contract。

## 6. 训练预算

VLA：

- H 主线使用冻结 nominal SimLingo，不训练候选生成模块；
- BF16/FP16，必要时量化，仅优化推理资源；
- 如未来研究 VLA 微调，必须作为独立任务授权，不能改变 H1 候选合同。

World：

- 两候选、T10；history/actor 数按显存预算冻结；
- 4M–8M object/vector model；
- mixed precision；
- 小样本过拟合、action permutation 和恢复 smoke 后再正式训练；
- 稳定峰值目标约 12–14GB。

## 7. CPU、磁盘和数据

- DataLoader/编码 worker 初始总数 4–6；
- 不占满 20 线程影响 CARLA tick/ROS callback；
- 图像、actor、轨迹和事件分片去重；
- Regression/Evidence/冻结数据只读；
- World 保存结构化 future，不保存像素视频作为核心训练数据。

活动工作集计划：

| 资产池 | 范围 |
|---|---:|
| 共享 Observation/split/Evidence | 25–35GB |
| VLA 独占 | 55–90GB |
| H World 独占 | 25–45GB |
| 有 World 常态目标 | 105–170GB |

200GB 是软上限。达到配额先停止采集、去重和清理可重建 cache，不自动删除冻结证据。

## 8. 运行前检查

```bash
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py doctor
python scripts/sdf.py sim preflight --json
```

只有需要真实 CARLA 的任务才要求 preflight。`RETRYABLE_FAILURE` 只允许 ensure 一次；
需要 GUI/UAC/登录时停止并交给用户。
