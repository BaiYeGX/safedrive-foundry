# 已完成任务：H6-CORA C1 正确性加固

## 1. 状态

```text
H0 = VERIFIED / STOPPED
H1 = VERIFIED / STOPPED
H2 = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H3 = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H4 = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H5 = COMPLETED / VERIFIED / GATE_FAILED / STOPPED
H6 v1/v2 = IMPLEMENTED / MEASURED / NOT_VERIFIED
H6-CORA C0 document consolidation and QA = COMPLETED
H6-CORA C1 correctness hardening = COMPLETED / STOPPED
H6-CORA C2 counterfactual data = AWAITING SEPARATE AUTHORIZATION / NOT_STARTED
Online Oracle = PROHIBITED
```

H5/H6 的历史 gate 和 Evidence 已冻结，不能改写。H6-CORA 是 H6 的结题修订，不新增 H7，
不改变 `AGENTS.md` 规定的唯一 H0→H6 研究链。后续不再把“World 在 90% tick 偏好 VLA”
或“VLA 使用率达到某个比例”作为主要优化目标；source usage 只做诊断。

## 2. 完成结果与边界

在任何新数据采集、GPU 训练或 CARLA formal 之前，C1 已修复代码审计确认的 correctness
问题，使下一阶段的反事实数据和 CORA World 不再建立在虚假 validation、错误 loss、
离线/在线语义漂移或 tick 所有权违规之上。

本任务只完成 C1；没有采集 CORA 数据、训练项目 checkpoint、运行 formal seed 或修改 Safety
硬权限。代码、测试和 schema Evidence 已达到 `IMPLEMENTED`，程序在此停止，等待 C2 数据任务
单独授权。

C1 只把旧实现变成可信的开发底座；虽然专项和全量单元测试已经通过，H6-CORA 的算法
Evidence 仍为 `PLANNED`，直到 C2/C3 实际产生数据和模型 artifact。

## 3. 已完成的 correctness 范围

### C1.1 真实 validation 指标

- 删除/替换 H6 checkpoint selection 中硬编码的 masking、swap、latency、GPU `pass/0`。
- 未测量字段必须是 `NOT_MEASURED`/`null`，不能写成通过。
- checkpoint selection 只能使用真实 evaluator 产生并绑定数据 lineage 的指标。
- readiness 必须拒绝缺失、常量占位或未绑定 evaluator artifact 的正式 summary。

### C1.2 正确的 per-sample multi-task / Group-DRO loss

- 每个 outcome head 按自身目标、单位和 mask 计算 per-sample loss。
- 先按冻结权重合成 per-sample multi-task loss，再按 map/family/weather 等 group 聚合。
- 不得用一个标量 objective 与异质输出向量做绝对差来伪造 group loss。
- 报告每个 head/group 的有效样本数；空 mask 不得当作零风险通过。

### C1.3 唯一 temporal selector 语义

- EMA/hold/switch 状态使用 episode/route revision 作用域内的稳定 source key，不用
  frame-scoped candidate id。
- hysteresis、minimum hold 与 emergency override 分开定义，不能由布尔条件相互吞掉。
- offline calibration replay 与 live router 必须调用同一个核心状态机。
- 给定同一输入 trace，逐 tick 的 raw score、EMA、defer、selected source、hold/switch reason
  必须一致。
- `hold` 只保持 source 决策，不能重用过期轨迹；每 tick 仍选择该 source 的当前 fresh candidate。
- `defer` 必须调用冻结的非学习回退规则，不得静默等价为“总选 Expert”、在线 Oracle 或无控制。

### C1.4 single tick owner

- 移除采集/清理脚本中绕过 Runtime 直接调用 `world.tick()` 的路径。
- 清理需要推进仿真时必须通过唯一 tick owner；无法安全恢复时返回用户动作状态并停止。
- 正式 collector 的 owner 是 `ScenarioRuntime`；ROS `carla_sync_driver` 是互斥 bring-up 模式，
  不能与 collector 同时推进同一 endpoint。

### C1.5 Evidence/readiness 真实性

- readiness/summary 必须区分 `IMPLEMENTED`、`MEASURED`、`VERIFIED`。
- 关键 summary/run-lock/evaluator artifact 必须有稳定自哈希和输入哈希。
- 失败、缺测和 skipped live checks 不得汇总成 `ALL_TARGETS_ACHIEVED`。
- benchmark 随机/未训练模型时必须在名称和输出中明确标为 smoke/latency-only。

## 4. 允许修改路径

只允许为上述 C1 问题修改必要文件：

```text
safedrive_foundry/data_pipeline/h5/
safedrive_foundry/data_pipeline/h6/
scripts/h5_collect.py
scripts/h6_readiness.py
scripts/train_world_v3.py
scripts/h5_ultimate_benchmark.py
tests/hybrid/
START_TASK.md
PROGRESS.md
docs/WORLD_MODEL.md
docs/EVIDENCE.md
```

如果真实调用链要求修改另一个非冻结文件，先在 `PROGRESS.md` 记录原因；不得借机重构
VLA、Classic、Safety、controller、H2/H3 frozen data 或 ROS 2。

## 5. 冻结边界

- 不修改 H2–H6 已冻结 dataset、split、seed、threshold、checkpoint 或 Evidence。
- 不复用已消费 formal seed 101。
- 不把 source、slot、branch order、Oracle、rollout future 或 outcome 放入 World 在线输入。
- 不降低 Guard/Safety 硬检查，不让 World 生成轨迹或控制底盘。
- 不删除失败测试、失败 Evidence 或用户无关工作区文件。
- `test_registry.sqlite3` 是任务开始前已存在的未跟踪文件；本任务不得删除或纳入正式 run-lock。

## 6. 已落实的测试要求

已新增或强化：

1. 缺真实 evaluator 指标时 checkpoint/readiness 失败；
2. 不同 outcome head 的 per-sample loss 与 mask 独立正确；
3. Group-DRO 对真实 group loss 更新，而不是对输出向量做伪回归；
4. frame id 每 tick 改变时 EMA 仍按 source 延续；
5. hysteresis、hold、emergency 三种边界行为；
6. offline/live selector 完整 trace parity；
7. collector 不直接拥有或推进第二 tick；
8. latency-only/random-model benchmark 不能输出质量 gate 通过。

验收时每项必须同时给出“实现位置、直接测试、失败行为、产出 artifact”。只有测试名称、
注释、空 schema 或把缺测值改名为 `NOT_MEASURED` 而没有 fail-closed 消费者，都不算完成。

## 7. 验收命令

优先运行专项测试，再运行全量：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -m unittest tests.hybrid.test_world_v3 -v
python -m unittest tests.hybrid.test_vla75_hardening -v
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry scripts tests
git diff --check
```

本任务是离线正确性加固，不强制启动 CARLA。任何声称 GPU latency/memory 或 CARLA 行为的
新数字都必须来自实际运行；单元测试不能升级成 `MEASURED`。

## 8. 已达到的停止点

以下条件已经满足，C1 必须保持停止：

- C1.1–C1.5 均有实现和直接测试；
- 全量测试和 compileall 通过，或失败被完整记录；
- `git diff --check` 通过；
- `PROGRESS.md` 记录修改、实测命令、未解决问题和 C2 接管条件；
- 没有采新数据、训练 CORA、运行 CARLA pilot/formal 或自动进入 C2。

没有遇到需要两次修复仍无进展、未知重叠修改或冻结 Evidence 冲突。实际测试和剩余限制见
`PROGRESS.md`。

## 9. C1 完成后的下一任务

C2 将按 [COUNTERFACTUAL_DATA](docs/COUNTERFACTUAL_DATA.md) 复用 H2/H3 exact-reset 能力，
构建双候选均有真实 short-horizon outcome 的 development 数据。C2 当前是
`AWAITING SEPARATE AUTHORIZATION / NOT_STARTED`；必须由用户另行授权并更新本文件后才能开始。
