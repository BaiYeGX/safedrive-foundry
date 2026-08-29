# SafeDrive Foundry 已确认进度

本文只保留当前可操作事实、冻结结果和下一接管点。2026-08-27 以前的逐轮日志已保存在
[`archive/2026-08-27-cora-document-consolidation/`](archive/2026-08-27-cora-document-consolidation/README.md)
的原始快照和阶段文档中，不能作为活动任务或阈值来源。

## 2026-08-29 — 活动文档收敛与第二轮全量 QA 完成

状态：

```text
H0-H4 = VERIFIED / STOPPED
H5 = VERIFIED / GATE_FAILED / STOPPED
H6 v1/v2 = IMPLEMENTED / MEASURED / NOT_VERIFIED
H6-CORA C0 document consolidation and QA = COMPLETED
H6-CORA C1 correctness hardening = CURRENT TASK / NOT_STARTED_THIS_TURN
```

本轮只做文档和可恢复归档：

- 根 README、START_TASK、ROADMAP、PROGRESS 和活动设计文档已统一为 CORA 结题主线；
- H3/H4 交付报告、H5 进入/矩阵、旧 H6 VLA75 handoff、旧 H2 paired contract 移入
  `archive/2026-08-27-cora-document-consolidation/historical-stage-docs/`；
- 收敛前的根/设计文档逐字快照保存在同一归档的 `original-active/`；
- 冻结的 `docs/runtime-evidence/` 和 `generated/` 数据没有改写或删除；
- H6-CORA 保持在 H6 内，不创建 H7；
- 90% VLA preference / 75% VLA usage 不再是 CORA 主要优化目标，历史 H6 仍按原门解释；
- 当前唯一下一任务是 C1 正确性加固，禁止自动采数据、训练或运行 formal。

用户要求重新检查后，第二轮逐份重读全部活动权威/入口文档、冻结 doctor 记录、关键代码
常量与调用点、Evidence 路径和相关工作的一手来源。修正的实质问题：

1. 分开 `program status` 与 `algorithm Evidence`：C0 完成、C1 当前不等于 CORA 算法已实现；
2. 把 outcome estimand 定义为 Guard eligible proposal 在冻结 single-candidate Safety/controller
   下的 CARLA intervention，采集 branch 禁止跨候选 fallback；
3. 定义 `CHOOSE/HOLD/SWITCH/DEFER`，其中 hold 使用 fresh candidate，defer 交回冻结非学习
   fallback，不是 Oracle、人工接管或无控制；
4. 修正 pair head：普通 MLP 接收差分不能保证反对称，必须用效用差或显式反对称化；
5. 区分 metadata-source-blind 与 trajectory-to-source 可预测性，避免伪称物理 planner 风格
   可以从 feature 中消失；
6. 修正 C2 选择偏差：Guard eligible 不等于预先 Safety 可执行；两个 branch 可按同一规则在
   不同 tick/reason terminal；missingness 和 root-anchor cluster 必须报告；
7. 明确 `ScenarioRuntime` collector 与 ROS `carla_sync_driver` 是互斥 tick-owner 模式；
8. 重写 C5 A–D 对照，使 selector 效果不被 generator workload 混淆，并要求 paired unsafe
   non-inferiority 口径而非只比原始事件数；
9. SHOWCASE 分开“现在可讲”和“C6 formal 后可讲”，移除当前枝干中的 LoRA 暗示；
10. RELATED_WORK 复核论文/官方来源，加入 CVPR 2026 Latent-CoT-Drive，并把 novelty 收敛为
    可证伪的组合贡献而非单概念首创。

C0 第一轮开始时：

```text
branch = main
tracked worktree changes = none
pre-existing untracked = test_registry.sqlite3
```

`test_registry.sqlite3` 没有被删除或纳入正式身份。

第二轮 QA 在上述未提交的 C0 文档工作树上继续；没有发现新的来源不明重叠修改，
`test_registry.sqlite3` 仍未触碰。

本轮文档验证：

```text
active Markdown local-link check = 16 files / 0 missing
active docs top-level set = 9 expected files / passed
cross-document semantic assertions = passed
frozen H1-H6 Evidence value assertions = passed
configured asset/path existence check = passed
archive original-to-HEAD byte comparison = passed
archive sha256 manifest = 19 files / passed
git diff --check = passed
```

本轮没有运行代码测试、CARLA 或训练；它们不属于 C0 文档收敛的验收范围。上述验证只证明
活动文档可导航、关键数字仍匹配冻结 artifact、归档可恢复且文本差异无格式错误，不能升级
任何算法 Evidence 状态。RELATED_WORK 的外部资料只用于定位，已按论文/官方页面复核，不是
本项目运行 Evidence。

## 冻结阶段结果

### H0 / H1

- H0 完成 H-only 路线收敛、旧路线归档和单机边界冻结；
- H1 在 Town03 完成真实 Classic + SimLingo 双候选 smoke；
- 两来源绑定同一 frame，VLA forward count 为 1；
- 两候选均 Guard PASS 且为 DISTINCT；
- selected/final/executed/applied identity 连贯；
- H1 Evidence 只证明候选与执行合同，不是驾驶性能结论。

### H2 paired outcomes

最终 gate-pass dataset：`h2-gatepass-20260813-routefix`。

```text
terminal = 120/120
valid/distinct = 108
decisive = 83
Expert wins = 51
VLA wins = 32
source-only baseline = 0.6144578313
dataset bytes = 1,480,172,014
whole-GPU peak = 8.3720703125 GiB
status = VERIFIED / GATE_PASSED / STOPPED
```

该数据证明两个独立来源存在真实选择空间。旧 gate-failed H2 dataset 同样保留，不能删除。

### H3 World development

最终运行：`h3-v2-20260815d-final`。

```text
OOF decisive = 91/91
reported best non-learning simple baseline = 84/91 = 0.9231
ECE = 0.00113
P99 = 13.89 ms
status = VERIFIED / GATE_PASSED / STOPPED
```

限制必须同时保留：

- learned candidate-only MLP 也达到 91/91；
- full-feature MLP 为 87/91；
- history sensitivity 部分由 hard scene gate 的结构保证；
- 因此 H3 证明小型开发集可区分，不证明上下文模型普遍优于所有 learned candidate-only
  baseline。

### H4 locked evaluation

最终运行：`h4-locked-20260816-final`。

```text
locked decisive = 64
World = 64/64
best simple = 57/64 = 0.890625
defer coverage = 63/64
status = VERIFIED / GATE_PASSED / STOPPED
```

限制必须同时保留：

- test set 小；
- simple baseline 已为 89.06%；
- temperature 到达 0.05 下边界，存在过度自信信号；
- microbenchmark 不能代表完整闭环尾延迟；
- 场景只覆盖三地图、有限 family/weather，不等于真实 OOD。

### H5 World on/off closed loop

最终运行：`h5-pilot-all2`，222/222 runs 完成。

```text
paired scenario roots = 74
World ON unsafe = 4
World OFF unsafe = 6
ON-only unsafe = 0
paired progress mean = +0.2549 m
bootstrap lower-95 = -0.0709 m
World ON switches = 14
World OFF switches = 0
scorer P99 = 47.2159 ms
deadline miss = 1
reset mismatch = 1
status = VERIFIED / GATE_FAILED / STOPPED
```

正式结论：World 没有在冻结 gate 上证明可复现闭环净收益。样本内没有 ON-only unsafe，但
未做出统计安全优越/非劣证明；进度置信区间跨 0，切换、资源和完整性门也未全部通过。不得
用 H3/H4 离线准确率改写该结论。

### H6 VLA-primary v1/v2

旧 seed 101 正式 pilot：`h6-vla90-formal-pilot-20260820-v1`。

```text
ticks = 600
World pair scored = 590/600
strict World VLA preference = 131/600 = 21.83%
VLA applied = 285/600 = 47.50%
Expert applied = 315/600
MRM applied = 0
VLA Guard = PASS 453 / REVIEW 147 / REJECT 0
World/Safety fallback to Expert = 18 ticks
RATO repair = 42
QP repair = 26
unsafe delta vs Classic = 0
paired progress lower-95 = +0.629 m
paired scenario roots = 12
scorer P99 = 9.12 ms
deadline miss = 0
switches = 31
ping-pong scenarios = 5
provenance failure = 10 missing World pair-score ticks（同一 Town01 aggressive-cut-in run）
status = MEASURED / GATE_FAILED / NOT_VERIFIED
```

正式低 VLA 占比的主因是 World 逐 tick 排名/校准，不是 Guard 或 Safety 大量拒绝 VLA。
Town03 free-flow 存在静态物体碰撞，Town03/Town05 存在红灯违规。seed 101 已消费，不能
调参后复用。

冻结 acceptance 的失败项为 actual VLA coverage、World VLA preference、switch rate、
ping-pong 和 provenance；paired progress 下界为正、deadline 为 0 不能覆盖这些失败。

H6 VLA75 v2 已实现 14-output 模型、A/B/C lineage、run-lock、acceptance 和 collector
hardening，但没有形成新 CUDA checkpoint、CORA 数据或正式 CARLA Evidence，因此仍是
`IMPLEMENTED / NOT_VERIFIED`。

## 2026-08-27 代码与数据审计确认的问题

这些问题是 C1/C2/C3 的依据，不是已经完成的修复。

### 1. 旧 H6 tickwise 反事实监督缺失

对本机 development dataset `h6-vla90-train-pilot-20260820-v2` 的 loader 审计：

| seed | tick rows | 双 outcome | 单 outcome | 双 executable | whole-policy pairs |
|---:|---:|---:|---:|---:|---:|
| 89 | 1200 | 0 | 1197 | 0 | 12 |
| 97 | 1200 | 0 | 1198 | 0 | 12 |

因此逐 tick pairwise mask 全为 0；主要比较监督来自每 seed 12 条 whole-policy row。旧
policy calibration 还使用第一条 decision 的特征监督整段 episode outcome。CORA 不允许
继续用该口径。

### 2. checkpoint validation 有常量占位

H6 `_vla75_validation_metrics` 存在 hardcoded masking/source-swap pass、`swap_error=0`、
`p99_ms=0`、`gpu_gib=0`，且参与 checkpoint selection。训练脚本后段虽计算部分真实指标，
readiness/selection 仍未完整绑定。C1 必须修复。

### 3. Group-DRO 语义错误

当前实现用标量 objective 与前 12 个异质输出做绝对差来构造 group loss，混合 progress、
variance、hazards、comfort、trust 等不同单位。C1 必须改为正确 per-head/per-sample loss。

### 4. temporal calibration/runtime 不一致

offline calibration 用稳定 source key；live EMA 使用 frame-scoped candidate id，跨 tick
可能重置。部分 hysteresis/emergency 条件会退化。C1 必须统一状态机并增加 trace parity。

### 5. single tick owner 违规

`scripts/h5_collect.py` 的清理路径存在直接 `world.tick()`。C1 必须移除或通过唯一 Runtime
推进。

### 6. demo/蒸馏宣传边界

- `carla_live_world_demo.py` 使用旧 H5 scorer，`--map` 未真正切图，并在 Safety 后强制
  最小 throttle；不能作为正式 Evidence；
- distilled scorer 的 risk loss 未加入总 loss，`val_data` 未实际用于验证；
- 某 latency 输出用 `<10ms` 判断却打印 `<4ms PASS`；
- `h5_ultimate_benchmark.py` 对随机未训练 World 做 latency 时仍输出质量式成功文案。

这些在 C1/C6 分别修复；当前不能作为简历正式结果。

## 最近一次实际离线验证

2026-08-27 全仓审计期间实际运行：

```text
python -m unittest discover -s tests -t . -v
436 tests: 435 passed, 1 skipped, 0 failed

python -m compileall -q safedrive_foundry scripts tests
passed

git diff --check
passed
```

跳过项是需要真实 GPU/checkpoint 的多次 SimLingo live forward；当前没有安装 coverage 模块，
因此 436 tests 不能解释为量化覆盖率。

该受限代理进程运行 `doctor/preflight` 时未访问到 GPU/CARLA。用户已明确确认本机 GPU 和
CARLA 可用；这次失败只代表该进程访问范围。任何后续真实任务仍必须在实际执行上下文中
重新记录 CUDA probe、preflight 和 CARLA Evidence，不能直接借用用户口头确认升级状态。

## 当前接管点

下一轮严格执行 [START_TASK.md](START_TASK.md) 的 C1：

1. 先检查分支和工作区；
2. 只修 validation/loss/temporal/tick-owner/readiness 真实性；
3. 新行为加直接测试；
4. 跑专项与全量离线回归；
5. 更新本文后停止；
6. 不采 CORA 数据、不训练、不运行 CARLA formal、不自动进入 C2。
