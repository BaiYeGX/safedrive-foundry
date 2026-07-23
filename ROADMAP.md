# VLA + World 最短路线

本文只定义当前之后的最短依赖顺序。每次只执行 `START_TASK.md` 中的一项；该项关闭后
才把下一项写入 START_TASK。

## 1. 总路径

```text
已有 K1 pure VLA+MPC
  ↓
R1 真实 K2
  ↓
R2 G4A paired pilot + oracle
  ↓
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

## 2. R1：真实 VLA K2（当前）

目标：

- `K=2/T=10/dt=0.25s/horizon=2.5s`；
- candidate 0/1 同一 Observation；
- 可执行、可区分、运动学一致；
- 可分别强制执行；
- K1 baseline 不回归。

最短实现先做 nominal + conservative 时间分支，但必须重新参数化 T10 位置。只有
G4A 证明纵向空间不足时才训练空间 residual/双头。

退出条件：`START_TASK.md` 的 K2 测试和 smoke 全部通过。通过后停止。

## 3. R2：极简 G4A

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

- 冻结 `ActionBranchDatasetV0`；
- 保存 Observable ego、≤8 actors、road、K2 和 paired outcome；
- 实现至少 persistence + CV/CTRV + Reward/规则中的两类；
- 完成 action swap、无动作条件和泄漏审计；
- 记录 G4 标签，但任何标签都不能跳过 World。

退出条件：数据可重建、基线可运行、简单方法结果可复现。

## 5. R4：World-V0

固定第一版：

```text
K2/T10/2.5s
N<=8
M=1
4M–8M parameters
object/vector latent
```

输出 actor future、collision、TTC、off-road 和 pairwise score。先小样本过拟合与
action sensitivity，再正式训练。禁止像素视频生成、第二视觉主干和在线控车分叉。

退出条件：checkpoint 可恢复，不同 candidate 产生可区分 future，异常状态显式。

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
