# World-V0 与最小效果实验

本文定义 G4A 选择空间量尺和 G5 World-V0。World 无论预期收益强弱都必须真实实现；
标签只决定实验规模和结论强度。

## 1. 前置条件

进入 G4A 前必须已有真实 VLA K2：

- 同一 Observation；
- candidate 0/1 可分别执行；
- 运动学一致；
- 可追溯；
- 不坍塌。

没有真实 K2，World 没有可排序的动作空间，不得用两份复制轨迹继续。

## 2. G4A 最小量尺

首轮只冻结：

- 6–8 个困难场景；
- 至少 3 类决策差异；
- 每场景至少 2 个 seed；
- spawn、route、initial-state hash；
- candidate 0/1 paired branch；
- top-1 与 oracle best-of-K。

优先场景：切入/急刹、换道冲突、无保护转向、VRU/遮挡。可复现和可比性优先于
数量；MAP-Elites、自动搜索、最小化和大覆盖属于 G4B optional。

pilot 输出：

| 标签 | 含义 | 后续 |
|---|---|---|
| `ENTER_WORLD` | 有稳定选择空间 | 完成 World，可扩 final set |
| `WEAK_SELECTION_SPACE` | 空间弱或不稳定 | 完成 World，结论谨慎 |
| `IMPROVE_VLA` | 两条候选经常都差/坍塌 | 完成 World；优先修 K2 或条件式 G6 |
| `NO_SELECTION_SPACE` | oracle 接近 top-1 | 完成 World；C2 可为无增益/负 |

无论标签如何都进入 G5。

## 3. ActionBranchDatasetV0

每个样本包含：

```text
scene/frame/run/scenario identity
Observable ego + <=8 actors + simple road context
VLA candidate 0/1
paired CARLA outcome
collision/TTC/off-road/progress/comfort
comparability flag and failure reason
```

训练必须包含非专家、保守、扰动和危险候选，不能只有 expert action。Oracle 字段与
runtime Observable 字段物理隔离，Regression 不进入训练。

## 4. World-V0

固定首版：

| 项 | 值 |
|---|---|
| K/T/dt/horizon | 2 / 10 / 0.25s / 2.5s |
| actors | N≤8 |
| modes | M=1 |
| 参数目标 | 4M–8M，实施后登记实数 |
| 表示 | object/vector latent，不生成高清像素视频 |

输入：

```text
ego history + actor states + simple road + candidate trajectory
```

输出：

```text
actor future
collision probability / first-conflict time
TTC
off-road
pairwise candidate score
invalid/unavailable/timeout/calibration state
```

World 不能生成并验证自己的安全标签，也不能修改 Safety 硬阈值或执行控制。

## 5. 必做基线和因果检查

至少实现：

- persistence；
- CV 或 CTRV；
- 规则/IDM 或 Reward MLP；
- no-action model；
- action shuffle/swap。

核心指标：

- actor ADE/FDE 仅作辅助；
- pairwise accuracy；
- ranking regret；
- top-1/oracle gap recovery；
- collision/TTC/off-road calibration；
- action sensitivity；
- 闭环碰撞、进度、舒适和资源。

若 no-action 或简单运动学更强，必须记录负结论，不能隐藏。

## 6. Runtime

```text
VLA K2
  → Contract Guard
  → World batch score (on/off)
  → deterministic tie-break
  → selected candidate
  → MPC/PID
```

要求：

- World on/off 独立配置；
- 目标约 5Hz，不阻塞控制；
- 同一 K2 一次 batch 评分；
- feature cache 只读；
- 不创建第二 CARLA client/tick owner；
- 不做当前帧在线 CARLA 分叉；
- timeout/OOM/NaN/invalid 直接跳过，执行 VLA 原始 top-1。

## 7. G5 最小 A/B

冻结以下共同变量：

```text
VLA checkpoint
K2 candidates
scenario/seed/initial-state
MPC/config
hardware/profile
```

比较：

1. `VLA-Top1`；
2. `VLA+World`；
3. `Oracle-Best-of-K`；
4. CV/CTRV/Reward/no-action；
5. 若 G6 已执行，再比较 `PostTrained VLA+World`。

G5 完成不要求 World 获得正收益，但要求实现真实、开关真实、fallback 真实、原始证据
可追溯。负收益时允许 default-off，不允许删除实现。

## 8. 何时执行 G6

只有以下归因才启动 G6：

- candidate collapse；
- candidate 0/1 经常双双失败；
- oracle 上限被 VLA 候选质量限制；
- World 排序正确但没有好候选可选。

如果问题是 World 误排、校准差、数据泄漏或执行器失败，不得用 G6 掩盖。
