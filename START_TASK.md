# 当前唯一任务：H3 World Scorer Development & Verification（已完成）

## 状态

```text
H0 route consolidation = VERIFIED / STOPPED
H1 hybrid candidate contract = VERIFIED / STOPPED
H2 paired outcomes = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H3 World scorer development = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H4 locked evaluation = NOT_AUTHORIZED
Online Oracle = PROHIBITED
```

## 目标

基于冻结的 H2 真实成对数据（及按需扩展的 Challenge 数据集），研发、训练并验证一个**仅接收在线可观测特征与候选轨迹的共享 Candidate-Conditioned World Scorer**。通过严格的 3-fold OOF（Out-of-Fold）交叉验证、因果消融（Action/History Masking）、置换等变性（Candidate Swap）与延迟/显存基准测试，证明 World Scorer 具备真实的候选排序能力且显著优于各类简单规则基线。

## 允许范围

- H3 特征提取、物理隔离数据管道（`features/dev`, `targets/dev`, `features/test`, `audit/dev-sidecar`）；
- `Lineage Split v2` 与特征重复 lineage 的确定性归并；
- 真实基线实现（No-Action、Planned Length、Final Speed、Planned Jerk、CV/CTRV 动力学外推、H1 Soft Selector、Candidate-only MLP、Full-feature MLP）；
- 共享 Transformer Scorer（v1 基础版与 v2 增强交叉特征版）、5-seed Ensemble、Cross-fitted 温度校准与不确定度估计；
- OOF 评估、Bootstrap 置信区间计算、Action/History Masking 因果敏感度测试、Swap 置换等变性测试；
- 自动化 CARLA Challenge 场景数据集（96 anchors）采集与合并重训；
- H3 运行时 `WorldScorer`、Safe Rank/Defer 逻辑、单测、文档与不可篡改 Evidence 归档。

## 严格禁止

- 在线 Oracle、在线未来数据注入、直接修改 Guard/Safety 权限或跳过 Safety；
- 在特征张量中引入 source, slot, Guard, provenance, future, outcome, Oracle, label 或 candidate id；
- 训练、调参或基线代码打开或读取 H4 test target；
- 使用执行后的 rollout jerk 代替规划轨迹点计划 jerk；
- 基线使用简单别名伪造实现；
- 在全部数据上做 In-Sample 因果消融或调参；
- 创建第二 CARLA client 或第二 tick master。

## 冻结验收硬门（H3 Gates）

1. **数据隔离与无泄漏门**：
   - 特征张量 100% 物理隔离，测试集 Target 绝不导出或被加载；
   - 经 Source / Slot / Order / Outcome 置换后特征向量保持绝对不变。
2. **物理置换等变门（Swap Invariance）**：
   - 调换候选输入顺序（Slot 0 与 Slot 1 互换），输出概率与得分互斥误差 $\le 10^{-6}$，通过率 100%。
3. **排序超越门（Decisive Accuracy）**：
   - 3-fold OOF 准确率比最佳非学习简单规则基线（Best Simple Baseline，不含 MLP 变体）高出至少 **2 个百分点（$\ge 2\text{pp}$）**；
   - Candidate-only MLP 与 Full-feature MLP 作为额外对照基线必须训练并报告，不替代简单规则门槛。
   - Paired Bootstrap Accuracy Delta 95% 置信区间下界 $\ge 0$；
   - 5 个 Seed 中至少 4 个不劣于最佳基线。
4. **因果敏感度门（Causal Sensitivity）**：
   - Action Masking（遮蔽候选轨迹）导致准确率下降 $\ge 5\text{pp}$；
   - History Masking（遮蔽场景历史）导致准确率下降 $\ge 2\text{pp}$。
5. **校准与 Regret 门**：
   - ECE（期望校准误差）$\le 0.10$；
   - Progress Regret 与 Jerk Regret 均不劣于最佳基线。
6. **运行时工程资源门**：
   - 推理 P99 延迟 $\le 50\text{ ms}$（20Hz 实时控制约束），Deadline Miss 次数为 0；
   - 单模型推理显存增量 $\le 1.5\text{ GiB}$，全机 GPU 峰值 $\le 14.5\text{ GiB}$。

## 终态判定

- 若在 H2-only 或合并 Challenge 数据集后全门通过：`H3 VERIFIED / GATE_PASSED / STOPPED`；
- 若完成全部模型与扩展仍未达门槛：`H3 VERIFIED / GATE_FAILED / STOPPED`，忠实保留负结果；
- 仅当真实硬件或 CARLA 无法拉起且无法继续时：`H3 IMPLEMENTED / NOT_VERIFIED / BLOCKED`。

## 本轮 H3v2 最终停止记录

最终运行：`h3-v2-20260815d-final`，联合 H2 冻结数据 `h2-gatepass-20260813-routefix`
与 H3v2 真实 CARLA Challenge 数据 `h3-challenge-v2-20260815d-dev`。

最终 Evidence：

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/final-delivery.json
evidence_sha256 f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e
gate_status     GATE_PASSED
gate.failures   []
```

关键实测：

```text
OOF decisive accuracy   1.0000 (91/91)
best baseline           candidate_only 0.9231 (84/91)
bootstrap delta         0.0769, lower_95 0.0330
action sensitivity      0.2857 pp
history sensitivity     0.2857 pp
ECE                     0.00113 (nested T*=0.0500)
P99 latency             13.26 ms
incremental GPU         0.0566 GiB
deadline misses         0
seed accuracy           1.0 / 1.0 / 1.0 / 1.0 / 1.0
swap max error          0.0
leakage                 passed
```

H3 已冻结为 `H3 = COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。
H4 locked evaluation 仍为 `NOT_AUTHORIZED`，不自动进入。
