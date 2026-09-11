# C4 执行单：后果辅助联合微调与现场接入准备

更新：2026-09-11。当前状态：`PLANNED / NOT_RUN`。C3 按学校项目的离线基线范围收尾，
后续复用现有 M1、数据与评估入口。本文件负责执行顺序和交付范围，
[WORLD_MODEL](WORLD_MODEL.md) 负责模型、损失和梯度公式。

目标：完成一份从原始 M0 开始的 M2，提供同数据的 M2/M1 对照和可供 C5 加载的 World。
老师要求实时 CARLA 闭环；截止为周日 2026-09-13。现场链路检查前置，周日中午为建议
版本冻结点，下午留给排练、报告和 GitHub 同步。

## 接续资产与当前缺口

| 项目 | 已有内容 | C4 的处理 |
|---|---|---|
| 数据 | 冻结 train 158 / validation 53；原生 route/speed 部分监督 | 从 manifest 读取清单、mask 与身份，沿用 SFT 暴露顺序 |
| 对照模型 | C3 M1 已完成 200 次更新，有 adapter、驾驶头与预测 | 保留作为基线；M2 重新加载原始 M0 |
| 评估 | 保存预测、独立计算、实际 canonical 转换检查 | 扩展到 M2；正式比较读取原始预测和冻结真值 |
| World 标签 | 旧候选及短时分支后果 | 核验同初态、候选身份、时间与 Safety/controller 记录 |
| 联合训练 | C3 可微 SFT 与部分辅助通路代码 | 尚无已登记的完整 M2 CLI；实现真实 b+R 联合更新 |
| 现场 | CARLA 安装、统一 sim 入口、已有 Runtime | 修复启动、接入当前 M1，再复用接口加载 M2 |
| 资源 | 历史运行及 GPU 记录 | 在新增优化前核对 smoke/失败耗时与剩余预算 |

基线 run：`generated/h6/cora/c3-repair-20260910T165902Z/`。读取其中的
`manifest.json`、`run_config.json`、`metrics-spec.json`、`m1_adapter.pt`、
`m1_training_summary.json`、`training.jsonl` 与 M0/M1 预测。
M0 路径由该配置解析；已登记文件为
`models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`。
独立评估实现为 [h6_cora_independent_analysis.py](../scripts/h6_cora_independent_analysis.py)。

## 顺序与完成节点

### 1. 先确认现场链路与实验输入

从 [ENVIRONMENT](ENVIRONMENT.md) 的统一入口定位 CARLA 启动故障，核对实际地图和本次
RPC READY；由 ScenarioRuntime 唯一持有 tick。实际安装为
`E:\CARLA_0.9.16\CarlaUE4.exe`。现有 live demo 使用旧 scorer，接入时显式加载原始 M0
加当前 M1 adapter/heads，并在 trace 保存模型身份。

先在一个允许、预登记的低复杂度场景中运行 M1，验证图像/状态、模型候选、Guard、Safety、
控制和车辆反馈。记录模型实际参与控制的区间、延迟、回退与重置结果。现场检查复用 C5
场景与总预算，启动、恢复和试跑都入账。CARLA 尚未恢复时可继续 CPU 数据/代码准备，
现场缺项仍单列；训练阶段关闭渲染。

同时建立 World 标签视图：SFT 标签保持原清单；每个辅助候选绑定它自己的真实分支。
单头不可信则关闭该头 mask，配对损失仅使用双方均可信的头。没有辅助标签的 root
仍按原顺序参加 SFT；不得把旧分支后果配给新模型刚预测出的不同候选。
保存每头有效 root/候选/配对数、支持区间与缺失原因。

完成节点：当前 M1 的现场链路记录、可核验的辅助 manifest，以及可用优化预算。
预算核对未完成时先做不涉及 GPU 优化的准备，不把历史未知消耗记零。

### 2. 实现一份固定配方并做真实联合检查

共享表示 h 由同一次可微 VLA forward 的当前观察 token 提取；两候选共用 h。
保留原驾驶头，增加冻结 candidate ridge b 与 `concat(h, candidate) → 128 → 64 → 4`
残差 head R，ReLU、无新增 dropout、末层零初始化。四个目标为 progress、acceleration RMS、
jerk RMS、lateral acceleration RMS。b、逐头尺度只在允许的 train 数据上拟合。

| 配方项 | 固定设置 |
|---|---|
| 原始起点、seed、更新数 | 与 M1 同一个 M0；seed 17；T=200 |
| 采样 | root 每轮无放回；root 内 anchor 确定性轮换；微批量 1、累积 4 |
| SFT | route/speed 各自有效坐标 SmoothL1，beta=1、权重各 1 |
| 辅助损失 | `0.1 × L_outcome + 0.05 × L_pair`；有效头/root 内先平均 |
| 优化器 | AdamW，weight decay=0.01；LoRA 2e-5，驾驶头 1e-4 |
| 新 R 的优化 | 默认 lr=1e-4，沿用驾驶头倍率日程；在正式 M2 前写入配置 |
| 参数范围 | 原 LoRA、route/speed 驾驶头、新 R；基座、视觉投影、query、b 冻结 |
| 精度 | 复用 C3 实测 BF16 设置；新通路经真实 batch 确认 |
| 预热 | 第 1–10 步 LoRA 冻结；第 11–20 步 LoRA 线性预热；之后沿用 C3 余弦日程 |
| 裁剪 | 合成后的 LoRA＋驾驶头共同 clip=1；R 独立 clip=1 |
| checkpoint | 最后一个有限有效更新；保存 optimizer、RNG、采样游标、日志边界与累计耗时 |

日程以 C3 `_set_schedule` 的实际实现为准，余弦倍率下限 0.1；新 R 的初始化使用独立
随机上下文，避免扰动与 M1 对齐的 SFT 随机序列。尾窗口按实际样本数归一化。
共享梯度在完整累积窗口上计算和门控，具体 w/q 公式见 WORLD_MODEL。

正式训练前只运行直接相关的检查：

- b 的离线/Torch 输出一致；R 初始为零；有效标签下 R 能学习。
- 当前输入与未来/标签隔离；候选交换后预测跟随候选；缺头与无 pair 正确 mask。
- 预热后真实辅助梯度到达 LoRA；驾驶与 World 两组裁剪互不串扰。
- 累积尾窗口、共享梯度门控、一次 cumsum 和部署输入/输出合同正确。
- 保存/重载与一次中断恢复可用；冻结参数和新增参数变化有证据。

完成节点：上述直接检查通过、真实完整窗口成功、整卡显存与优化耗时可用。
现有 C3 成本 smoke 可供诊断参考，完整联合训练须通过本轮实际检查。

### 3. 运行一次 M2 并独立评估

丢弃 smoke 更新，重新加载 M0，重置正式训练 RNG/optimizer，冻结配置与指标后训练。
新增产物使用唯一 `generated/h6/cora/c4-<UTC>/`，阶段目录分开，禁止覆盖旧运行。
单次 M2 优化最多 4 小时，C3/C4 含历史与失败累计最多 10 小时；整卡采样峰值最多
14.5 GiB，采样间隔 0.5 s，记录 GPU UUID、采样缺口和进程 allocated/reserved。
从实际吞吐和剩余预算检查 T=200 可完成性；不足时记录具体差额，不能把少步数称同预算完成。

| 比较 | 主结果与并列项 |
|---|---|
| M2 / M1 | 主指标 P-WP；并列 P-ADE、真实 canonical P-VALID、P-FAIL、逐 root 差值 |
| M1 / M0 | 使用现有离线基线，说明微调适配与联合方法分别带来的变化 |
| b+R / b | 四头物理单位 MAE/RMSE、有效配对误差、进度排序/regret 与各自有效数 |
| 工程成本 | 更新数、耗时、显存、参数变化、共享辅助梯度参与情况 |

M0/M1/M2 使用相同的完整冻结 validation 清单、输入和真值 mask。先样本→root 聚合，
再 root 等权；配对 bootstrap 1000 次、seed 71；改善/持平/退化的阈值在本轮预测前登记。
P-WP 为米；P-FDE 仅固定第 20 点有效时计算，P-SPEED 仅有可信速度真值时计算，否则 N/A。
预测失败保留分母与原因，World 只在对应头的共同有效支持上比较。

独立进程重载 M2，从保存预测重新计算指标。完成节点：200 个有效更新、真实共享辅助
参与、可重载模型，以及完整可复算的 M2/M1 和 b+R/b 结果；收益方向按实际结果记录。

### 4. 交接 C5 与 C6

提供一个明确的加载配置，绑定 M0、M2 adapter/驾驶头、R、b、normalization、candidate
编码、h 提取与预处理身份。在线复用一个 VLA forward，World 只为 Guard eligible 候选
排序或 defer，选择后继续经过同一 Safety/controller。

C5 当前以预登记 1–2 roots 的 A=M1 固定规则、B=M2 固定规则、C=M2+World 功能对照为
学校项目范围，按实际执行量报告。试跑/失败在原 18 attempts、4 小时 CARLA 上限内计数。
完整三臂对照及现场排练属于 C5；C4 交付模型、可加载配置和接口检查，不据文档宣称已运行。

C6 提前准备已有数据与模型说明；追加 M2 结果后汇总报告、现场操作步骤、备用录像和
本机大产物索引。提交、普通合并 main、推送和远端 SHA 核对沿用已授权流程。

## 实现入口与验证登记

已有可复用入口：`scripts/h6_cora_sft.py`、`scripts/h6_cora_independent_analysis.py`、
`scripts/carla_live_world_demo.py`、`scripts/sdf.py sim`。旧 demo 的模型绑定需适配。
拟新增 `scripts/h6_cora_joint.py`，职责为 audit、smoke、train/resume、evaluate、verify；
该文件和子命令目前是待实现设计，开发后以实际 `--help`、直接测试和运行记录登记命令。

以下是已有入口的只读检查命令，本次文档编辑未执行 CARLA 检查：

```bash
cd "/mnt/e/autonomous driving"
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py sim status
python scripts/sdf.py sim preflight --json
```

最小交付目录：冻结配置与 manifest 引用、b/尺度、smoke 与恢复证据、M2 adapter/heads/R、
训练日志/checkpoint、预测/逐 root 结果、资源账本、独立重算、C5 加载配置和 stage-summary。
summary 分开记录 `training_complete`、`evaluation_complete`、`live_readiness` 与
`algorithm_gain_status`。C4 完成后更新 PROGRESS/Evidence/START_TASK 指向 C5。
