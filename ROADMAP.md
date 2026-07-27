# VLA + World 最短路线

本文只定义当前之后的最短依赖顺序。每次只执行 `START_TASK.md` 中的一项；该项关闭后
才把下一项写入 START_TASK。

## 1. 总路径

```text
已有 K1 pure VLA+MPC
  ↓
R1 真实 K2
  ↓
R2 G4A paired pilot + oracle（纵向 K2，已关：NO_SELECTION_SPACE）
  ↓
R2-X Spatial/Semantic K2（已选定的 R3 前扩展；新 Evidence，不覆盖旧 R2）
  ↓ 若有稳定选择空间
R3 World 数据与简单基线
  ↓
R4 World-V0
  ↓
R5 World runtime + 核心 A/B
  ├─ VLA 候选限制上限 → R6 条件式 G6 → 重跑 R5
  └─ 候选已足够 → R7 核心 Evidence/准出

Safety / G4B Search / Agent：After Core / Optional
```

核心完成标签：`VLA_WORLD_RESEARCH_COMPLETE`。

## 2. R1：真实 VLA K2（已完成）

详细代码分析、算法、文件级改动和验收见
[R1 实施任务](docs/R1_REAL_K2.md)。

目标：

- `K=2/T=10/dt=0.25s/horizon=2.5s`；
- candidate 0/1 同一 Observation；
- 可执行、可区分、运动学一致；
- 可分别强制执行；
- K1 baseline 不回归。

最短实现先做 nominal + conservative 时间分支，但必须重新参数化 T10 位置。只有
G4A 证明纵向空间不足时才训练空间 residual/双头。

状态：`COMPLETED_WITH_LIMITS`。最新复验见 `PROGRESS.md`。

## 3. R2：极简 G4A（已关闭：`COMPLETED_WITH_LIMITS`）

**终态**：`COMPLETED_WITH_LIMITS`；pilot **`NO_SELECTION_SPACE`**（11/12 TIE；
1 pair CARLA timeout 进分母）。repeat 2/2 标签一致。
详见 `PROGRESS.md` 与 `docs/runtime-evidence/r2-g4a-paired-pilot/r2_closure_report.json`。

**冻结**：该结论针对**纵向** K2（同 path + 不同 speed），**不得**被 R2-X 覆盖或改写。

详细代码审计、实验单位、registry、comparability、oracle 和文件级验收见
[R2 Paired Outcome + Oracle](docs/R2_PAIRED_ORACLE.md)。

## 3b. R2-X：Spatial / Semantic K2（已选定；在 R3 前）

**实施权威**：[R2-X Spatial/Semantic K2](docs/R2X_SPATIAL_K2.md)。

因纵向 R2 证明选择空间不足，当前路线已选择在进入 World 前扩展真正的
**语义双头空间 K2**：

- 一次 backbone forward + nominal/defensive mode query + Frenet residual；
- Guard V2（per-candidate path）；新 Evidence 目录 `r2-spatial-k2-pilot-v2/`；
- **不**固定左右偏、**不**只改 Oracle deadband、**不**首版 Diffusion。

阶段：R2-F 诊断 → R2-G 合同 → R2-H 数据训练 → R2-I 离线验收 →
R2-J force smoke → R2-K 新 12-pair pilot。

**不得**自动启动；需用户把具体子阶段写入 `START_TASK.md` 并明确授权。
默认只有 R2-X 得到稳定选择空间后才申请进入 R3；若用户明确提前终止 R2-X，
必须保留负结果并重新决定是否仍值得实现“预期无增益”的 World 对照。

### R2.1 Registry + Replay

- 6–8 个困难场景；
- 至少 3 类决策差异；
- 每场景至少 2 seed；
- 固定 spawn/route/initial-state hash；
- 可恢复、幂等、不可比分支显式标记。

### R2.2 Paired Oracle

同一起点分别执行 candidate 0/1，输出：

- VLA top-1；
- oracle best-of-K；
- collapse/同步失败；
- 选择空间标签。

pilot 有信号才在看结果前冻结扩大后的 final set。MAP-Elites、自动最小化和大覆盖
不是核心前置。

退出条件：paired replay 可复现，oracle 表和原始 run 可查询。

## 4. R3：World 数据与简单基线

详细实施权威：
[R3/R4 World 数据与模型规格](docs/R3_R4_WORLD_DATA_MODEL.md)。

当前 R2-X 只允许 World development：nominal 长跑可用，但 defensive availability
不可靠，正式门 `WORLD_GATE_NOT_MET`。R3 必须原生处理 `K_eff=1` 和
`NO_ALTERNATIVE → nominal`；这不等于已授权启动 R3。

- 冻结 `ActionBranchDatasetV0`；
- 保存 Observable ego、≤8 actors、road、K2 和 paired outcome；
- 新增 oracle-only、逐 actor、逐 candidate 的 2.5 s future trace；现有聚合
  outcome 不能反推 future；
- 实现至少 persistence + CV/CTRV + Reward/规则中的两类；
- 完成 action swap、无动作条件和泄漏审计；
- 冻结 group split；旧 R2 blind Evidence 只作 holdout/audit，不进训练；
- 记录 R2 标签；弱选择空间允许受限开发，但不得写成 World-ready。

退出条件：数据可重建、oracle/runtime 零泄漏、split 零重叠、基线可复现，并明确
记录 `R3_DATA_READY`、`R3_DATA_READY_WITH_WEAK_ACTION_SIGNAL` 或阻塞标签。

## 5. R4：World-V0

固定第一版：

```text
K2/T10/2.5s
N<=8
M=1
4M–8M parameters
object/vector latent
```

采用共享 scene encoder + candidate query，同一次 forward 评分有效 K2；输出
candidate-conditioned actor future、collision、TTC、off-road 和 pairwise score。
先做 32-sample 过拟合、candidate swap/permutation、no-action 和 action sensitivity，
再正式训练。禁止像素视频生成、第二视觉主干和在线控车分叉。

退出条件：checkpoint 可恢复、不同 candidate 产生可区分 future/risk、异常状态显式；
若 no-action 或简单基线不弱于 World，保留
`R4_WORLD_ACTION_SIGNAL_NOT_PROVEN` 负结论，不调 Oracle 掩盖。

## 6. R5：World Runtime 与核心 A/B

Runtime：

```text
VLA K2 → Guard → World batch rank on/off → selected → MPC
```

要求：

- World off/failure → VLA 原始 top-1；
- 同一候选 batch 评分；
- 不阻塞控制、不抢 tick；
- 可查询 selected/executed candidate；
- 目标约 5Hz，实际值必须实测。

核心比较：

1. `VLA-Top1`；
2. `VLA+World`；
3. `Oracle-Best-of-K`；
4. CV/CTRV/Reward/no-action。

World 正、零、负收益均可关闭 G5，但模块、开关、fallback 和原始证据必须真实。

## 7. R6：条件式 G6

只有以下情况才启动：

- 两条候选经常双双失败；
- collapse；
- oracle 上限被 VLA 质量限制；
- World 排序正确但没有好候选。

只做一轮：

```text
失败前 2–4s 窗口
→ Classic/Safety/人工规则 corrective label
→ 单个 adapter/少量 LoRA
→ K2/oracle/World/正常集前后回归
```

禁止 PPO/GRPO、多 preference、多轮自动飞轮和同时训练 VLA/World。

若问题是 World 误排、校准、数据或执行器，返回对应阶段修复，不启动 G6。

## 8. R7：核心 Evidence 与准出

冻结：

- VLA-Top1、VLA+World、Oracle；
- G6 已执行时增加 PostTrained；
- code/config/model/data/scenario hash；
- 资源、失败和限制；
- 冷启动复现步骤。

完成条件见 `docs/PROJECT.md`。最终标记
`VLA_WORLD_RESEARCH_COMPLETE`，不上实车。

## 9. Optional

| 模块 | 何时做 | 是否阻塞核心 |
|---|---|---|
| Safety live | 核心效果后，需要工程护栏演示时 | 否 |
| G4B Search | 固定 paired set 无法覆盖研究问题时 | 否 |
| Agent | 确定性流程已成立且需研究效率对照时 | 否 |
| VLA-V2 reasoning | K2 稳定且有明确复杂场景收益假设时 | 否 |

Safety 扩展全部关闭后，可额外标记
`FULL_PROJECT_COMPLETE_WITH_SAFETY`。

## 10. 停止规则

- R1 不通过，不进入 R2；
- paired replay 不可比，不声称 World 效果；
- World 不得因预期收益低而跳过；
- 负结果保留，不移动 split/阈值换结论；
- 每一阶段通过后停止，由下一轮更新 START_TASK。
