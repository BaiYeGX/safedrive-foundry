# 本机资产与 H6-CORA 资源预算

本文是本机资产和单机资源边界的活动说明。路径/版本配置是机器可读真源；任何一次任务的
GPU/CARLA 可用性仍由该任务实际 probe/Evidence 证明。

全文区分三类口径：`configured` 是仓库/用户确认的资产，`measured` 必须指向运行 artifact，
`budget` 是后续任务的停止上限。预算数字不能写进简历成为实测性能。

## 1. 固定硬件与运行分工

```text
GPU: NVIDIA RTX 4080 Desktop 16GB
CPU: Intel i5-13600KF, 14 cores / 20 threads
Host: Windows 11 + WSL2 Ubuntu 24.04
CARLA Server: Windows
ROS 2 / client / VLA / World / training: WSL2
```

用户已确认本机 GPU 与 CARLA 可用。某个受限代理进程无法访问 CUDA/RPC，只说明该进程的
权限/网络上下文，不能否定资产存在；正式运行仍必须保存实际 `torch.cuda`、preflight、
CARLA version 和 GPU resource Evidence。

Windows CARLA 与 WSL CUDA 共享同一物理 GPU。不得假设第二张 GPU、远程服务器或云训练。

## 2. 本机路径

| 资产 | WSL 路径 | 用途 |
|---|---|---|
| 仓库 | `/mnt/e/autonomous driving` | 项目根 |
| SimLingo code | `/mnt/e/autonomous driving/simlingo-main` | 上游模型/预处理参考 |
| SimLingo weights | `/mnt/e/autonomous driving/models/simlingo` | nominal VLA checkpoint |
| InternVL2-1B | `/mnt/e/autonomous driving/models/InternVL2-1B` | VLM 底座资产 |
| CARLA 0.9.16 | `/mnt/e/CARLA_0.9.16` | Windows Server 文件 |
| default venv | `/home/sdf/.venvs/sdf` | Python/CUDA/CARLA client |

机器可读配置：

```text
versions.lock
safedrive_foundry/config/vla/local_assets.toml
safedrive_foundry/config/runtime/carla_start.toml
```

`simlingo-main` 是第三方上游参考，不得替代本项目 Runtime、Guard、Safety 或 tick master。

## 3. Python 基线

活动默认：

```text
/home/sdf/.venvs/sdf
Python 3.12.3
PyTorch CUDA build
Linux carla 0.9.16 client
```

每次需要 GPU 的任务先实际运行：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"
```

禁止使用系统 Python、`--break-system-packages`、Windows Anaconda 执行 WSL runtime，或在
没有 torch 的历史 venv 中运行 VLA/World。

## 4. 独占 workload profiles

| Profile | GPU/系统主任务 | 必须关闭或暂停 |
|---|---|---|
| `cora_data` | CARLA + nominal VLA branch collection | World optimizer、其他 CARLA client |
| `cora_train` | CORA World/ensemble training | CARLA Server 渲染负载、VLA live eval |
| `cora_calibrate` | frozen checkpoints + calibration | training optimizer、formal CARLA |
| `cora_online` | CARLA + VLA + World + Safety | 所有训练、数据预处理大任务 |
| `regression` | 专项 CARLA/单元测试 | formal collector |
| `showcase` | 固定 checkpoint live/replay demo | training、实验 collector |

总控应显式拒绝冲突 profile，不依赖 OOM 后恢复。

## 5. 在线显存预算

下表全部是 `budget`，不是既有测量：

| 占用 | 目标 |
|---|---:|
| CARLA Low/No Rendering | 约 4–5 GiB，按 map/RHI 实测 |
| nominal VLA | 约 5–6 GiB，按 checkpoint/precision 实测 |
| CORA ensemble/scorer | 尽量 ≤1.5 GiB |
| CUDA/context/cache 余量 | ≥2 GiB |
| whole-GPU peak | ≤14–14.5 GiB |

正式在线 Evidence 同时记录 whole-GPU peak 和 World incremental peak。不能把独立离线
microbenchmark 的低显存/延迟直接写成完整在线结果。

## 6. CORA 模型预算

- 首版继续使用 object/vector context 和候选 trajectory；
- 允许冻结视觉特征，但不在结题阶段训练视频生成器；
- shared candidate encoder、outcome heads、pair head 和 3-seed ensemble；
- mixed precision；
- 单模型参数量/hidden size 由 C3 小样本过拟合与 latency smoke 冻结；
- scorer P99、ensemble latency 和 online deadline 分开报告；
- 资源不足先减 batch、history/token 数或冻结特征分辨率，不改 label/candidate/Safety 合同。

VLA 在 H6-CORA 保持冻结 nominal proposal。任何 LoRA fine-tune 必须等 CORA 正式结题后作为
新项目授权，避免同时改变 generator 与 selector 而无法归因。

## 7. 数据与磁盘

CORA development 预计包含：

- 240–360 个有效 root anchors（单机预算假设，C2 前按事件覆盖重新冻结）；
- 每 anchor 至少两个真实 50-tick branches；
- 部分 offline-only intervention branches；
- images、timelines、actor future、events、labels 和 manifest。

存储原则：

- 图像 content-addressed 去重；
- timeline/events 分片压缩；
- frozen dataset/Evidence 只读；
- 同 ID 不同 hash 拒绝覆盖；
- 可重建 cache 与不可变 Evidence 分开；
- formal 前检查剩余磁盘和预估增长。

统计与存储都按 root anchor 计数；两个 branch、50 个 tick 和同 anchor interventions 不能被
写成独立样本放大数据量。C2 分 smoke、coverage pilot、frozen development 三段，每段达到
磁盘/时间上限或事件覆盖失败都应停止。

活动工作集软预算（规划包络，未做本轮磁盘实测）：

| 资产池 | 范围 |
|---|---:|
| observation/split/evidence | 25–35GB |
| VLA/model assets | 55–90GB |
| World/data/checkpoints | 25–45GB |
| 正常总量 | 105–170GB |
| 软上限 | 200GB |

达到上限先停止采集并审计；只清理可重建 cache，不删除 frozen Evidence、失败数据或用户文件。

正式采集任务开始前必须把 `df`、预计每 anchor bytes、剩余空间、最大 anchor 数写入 run-lock；
不能因为已经启动 CARLA 就无限追加数据。

## 8. CPU 与实时性

- DataLoader/编码 worker 初始总数 4–6；
- 不占满 20 线程影响 CARLA tick、sensor barrier 或 ROS callback；
- collector、spectator 和 writer 使用有界队列；
- 每阶段记录 wall latency 与 simulation time；
- online 目标为 20Hz，但具体 scorer/全链 deadline 在 C3/C5 前冻结；
- P99 和 deadline miss 是 gate，平均延迟不能代替尾延迟。

## 9. 任务前检查

离线：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py doctor
```

真实 CARLA：

```bash
python scripts/sdf.py sim preflight --json
```

只有该次执行返回 `READY` 才继续 live task。`RETRYABLE_FAILURE` 只允许一次 ensure 和一次
复查；GUI/UAC、版本、tick owner、依赖或权限冲突立即停止并交给用户。
