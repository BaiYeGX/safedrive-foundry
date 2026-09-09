# 本周资源与预算

硬件为 RTX 4080 16GB、i5-13600KF；Windows CARLA 与 WSL2 CUDA 共用同一张卡。
资源是实验设置，不是论文贡献。以下是规划上限，实际性能必须记录，不能借用历史 probe。

## 本地资产

| 资产 | 路径 |
|---|---|
| 项目 | /mnt/e/autonomous driving |
| SimLingo 参考代码 | /mnt/e/autonomous driving/simlingo-main |
| VLA 权重 | /mnt/e/autonomous driving/models/simlingo |
| VLM 底座 | /mnt/e/autonomous driving/models/InternVL2-1B |
| Python 环境 | /home/sdf/.venvs/sdf |
| CARLA | /mnt/e/CARLA_0.9.16 |

机器可读真源：versions.lock、config/vla/local_assets.toml 与 config/runtime/carla_start.toml
（后两者位于 safedrive_foundry）。不自动装包、换 CUDA 或下载公开大数据。

## 只保留三种工作负载

- 训练：C3/C4 GPU 优化，CARLA 渲染与其他 optimizer 不并发。
- 在线：C5 CARLA + 一个 VLA + 小型 World + Safety，不训练。
- 离线整理：数据检查、评估、C6 重放材料，不启动模拟器。

这些是任务模式，不宣称已有同名 CLI 或配置实现。统一入口见 [ENVIRONMENT](ENVIRONMENT.md)。

## 上限与日程

| 项目 | 上限 |
|---|---|
| C3 常规训练 | 完整优化最多 4 小时 |
| C4 联合训练 | 完整优化最多 4 小时 |
| 全部 GPU 优化 | 含 smoke/失败累计最多 10 小时 |
| 训练数据补采 | 仅必要时一次，最多 12 新 roots / 24 attempts / 2 小时 CARLA 含启动恢复，先到即停 |
| C5 开发闭环 | 6 roots × 3 臂 = 18 runs，每 run 最多 60 s 仿真 |
| C5 CARLA wall | 含启动、失败、恢复最多 4 小时 |
| whole-GPU peak | <=14.5 GiB |
| 采集前空闲磁盘 | >=60 GiB，并容纳预计新增量 |
| 新公开数据下载 | 本周不做 |

C3 第 1–2 天，C4 第 3–4 天，C5 第 5 天，C6 第 6 天，第 7 天缓冲。
第一天真实监督/GPU 不通就报告阻塞；不拖到最后一天才发现训练不可行。
达到上限保留实际结果，不自动增样、加 seed、加模型或删失败。
M1/M2 步数在预算内按真实吞吐冻结，避免把额外训练时间混成方法优势。

microbatch=1 起步、梯度累积、原生 LoRA 与实际可用 mixed precision；具体配置经真实 batch 冻结。
不能照搬上游多 GPU/batch/DeepSpeed 学习率。减显存不能 detach 联合梯度后冒称联合学习。
20Hz 是 Runtime tick 目标；VLA cadence、chunk 与超时策略在 C5 前绑定并实测。

## 历史与记录

C2 CPU baseline 为当时四线程/30 分钟预算，三 seed MLP 已完成；这不等于 GPU 微调测量。
2026-09-08 设备只读 probe 当时为总量 16376 MiB、已用 1073 MiB，不是可用资源承诺。
记录 wall time、优化步数、allocated/reserved/整卡峰值、P50/P95/P99 与 deadline miss。
清理只针对可重建缓存，不删除旧模型、冻结 Evidence 或失败数据；不重扫旧 Hash。

C3 实测资源账本已写入
`generated/h6/cora/c3-vla-sft-20260909-final-v3/resource-ledger.json`：RTX 4080 CUDA/BF16
正式 M1 共 200 updates，wall `267.697 s`，峰值 allocated/reserved 为 `4.259/4.398 GiB`，
adapter `72,156,073 bytes`，完整恢复 checkpoint `216,521,989 bytes`。本次低于 4 h 单次
训练和 14.5 GiB 峰值上限；C3/C4 合计 10 h 优化总账仍按合同保留，C4 尚未消耗新预算。

## 固定训练与运行口径

C3 的完整任务文档登记 M1/M2 共用的初始 optimizer/seed/batch 与最多 200 updates；
这些是起始工程默认值，须在正式 M1 前依据真实 smoke 将共同步数一次冻结到上述预算内。
C4 不额外增加数据暴露或续训预算。C5 的 18 次上限按 episode attempts 计，失败占次数，
不自动另补 root 或重跑；C6 只离线加载/指标重算/replay，不增加训练与 CARLA 预算。


## 本次提高稳定性的预算安排

C3/M1 与 C4/M2 共用短 head 预热、低 LoRA 学习率和同 T；预热包含在最多 200 updates 内。
M2 增加残差 head 与同 accumulation window 的两类共享梯度计算，可能接近额外一次 backward，
不承诺零算力开销。必须在正式 M1 前用真实 smoke 测出成本后冻结共同 T。
固定 b 的 CPU 拟合/导出计入阶段 wall，M2 在线只增加小型 b+R 运算，不加载第二个 VLA。
不增加 teacher 网络、SAM、ensemble、多 seed、完整消融或新 CARLA 臂；
原 4h/4h 完整优化、10h GPU 总账与 18 attempts/4h CARLA 上限不变。

## 论文资源指标

执行 [PROJECT 论文量化合同](PROJECT.md#论文量化合同c3_c6_metrics_v1planned)：M1/M2
分别记录实际 wall hours、updates、峰值 allocated/reserved 显存 GiB、可训练参数及 adapter MiB。
在线记录全决策与 World 部分时延 P50/P95/P99、deadline miss n/N 和输出新鲜度。
异步 GPU 的 enqueue 时间不能当计算完成时间；计时方法、warmup 和样本数进入 run config。
仅复用已有 smoke/预测/trace 和预算，不新增计时训练或闭环；缺仪器数据填 N/A。
20 Hz 仿真 tick 不等于 VLA 达到 50 ms 实时推理；不得宣称未实测的加速比或成本节省。
所有原资源硬上限保持，论文中的百分比参数量是实验设置，不作为独立方法创新。

本次研究后的适配检查复用原 smoke 预算：native cumsum/空间时间标签、输入一致性、
干净 M0 重载和两组梯度裁剪。仍两次训练、两类共享梯度；不新增 teacher 前向、SVD 初始化、
视频 World、RL 或候选模型搜索。纯读取源码/小型 CPU 数值检查不计为模型训练结果。
