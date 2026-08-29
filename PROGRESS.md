# SafeDrive Foundry 已确认进度

本文只保留当前可操作事实、冻结结果和下一接管点。2026-08-27 以前的逐轮日志已保存在
[`archive/2026-08-27-cora-document-consolidation/`](archive/2026-08-27-cora-document-consolidation/README.md)
的原始快照和阶段文档中，不能作为活动任务或阈值来源。

## 2026-08-30 — H6-CORA C1 正确性加固完成并停止

状态：

```text
H6-CORA C0 document consolidation and QA = COMPLETED
H6-CORA C1 correctness hardening = IMPLEMENTED / COMPLETED / STOPPED
H6-CORA algorithm Evidence = PLANNED / NOT_MEASURED / NOT_VERIFIED
H6-CORA C2 counterfactual data = AWAITING SEPARATE AUTHORIZATION / NOT_STARTED
```

实施边界与 Git：

- 开始时只发现已知 C0 文档改动和既有未跟踪 `test_registry.sqlite3`，没有新的来源不明重叠修改；
- C0 文档经 staged diff/check 后提交为 `b8f3707 docs: consolidate H6-CORA active contracts`；
- C1 在 `codex/h6-cora-c1` 分支实施，默认保持未提交、未 push；
- `test_registry.sqlite3` 始终未跟踪、未删除、未暂存，并已从 C1 worktree/run-lock identity
  中显式排除；
- 没有采集 CORA 数据、训练项目/CUDA checkpoint、启动 CARLA、消费 formal seed、修改冻结
  dataset/split/seed/threshold 或改写 `docs/runtime-evidence/`；
- `scripts/h6_run_lock.py` 不在 C1 原允许路径表内，但用户本轮计划明确要求 C1 后 run-lock
  绑定 evaluator、validation lineage 和训练输入哈希，因此只做该必要调用链修改，没有扩展重构。

### C1.1 validation、evaluator、readiness

- `safedrive_foundry/data_pipeline/h6/dataset.py` 对 validation row 的 feature、trajectory、target、
  mask、split、seed、group 和样本身份生成稳定 lineage hash；
- `safedrive_foundry/data_pipeline/h6/model.py` 的 checkpoint selection 只接收 evaluator 实际产生
  的 loss、per-head count/loss、pair accuracy/regret、worst-group 和 candidate swap；缺字段、
  非有限值、零有效样本或无 lineage 均 fail closed，source/VLA usage 只保留诊断意义；
- 新增 `safedrive_foundry/data_pipeline/h6/evaluator.py`，定义
  `safedrive.world.vla75.evaluator.v1`，绑定 checkpoint/seed、validation/config/code/worktree/input
  hash、per-head/group/pair/probe、实测 latency、资源状态、Evidence 状态和自哈希；
- `scripts/train_world_v3.py` 产生 `safedrive.world.vla75.training_summary.v2`，绑定 train/validation
  lineage、三个 checkpoint 和三个 evaluator，并明确 artifact `VERIFIED` 不等于 CORA algorithm
  `VERIFIED`；
- `scripts/h6_readiness.py` 只接受完整 C1 v2 summary，识别但拒绝 v1，验证 summary/evaluator/
  checkpoint/input/self hash、三 seed 顺序、有效计数、probe、latency 和 GPU 实测状态，并输出
  `readiness_sha256`；
- `run_lock.py` 与 `scripts/h6_run_lock.py` 将新建 C1 lock 升级为
  `safedrive.h6.vla75.run_lock.v2`：calibration payload 必须带 evaluator、validation lineage
  和训练输入绑定，并在落盘前验证；显式 v1 只保留历史只读兼容。

失败行为：缺 evaluator/metric/lineage、`NOT_MEASURED`、零样本、零占位资源、未观察到
action/context sensitivity、swap 不变量失败或任一 hash/顺序不一致时 readiness 失败；CPU evaluator
的 incremental GPU peak 正确记录为 `NOT_MEASURED/value=null`，不能获得正式 readiness。

### C1.2 per-sample multi-task 与 Group-DRO

- `model.py` 新增逐样本 head report：objective、progress、completion、collision、red-light、
  offroad、comfort、repair、trust、pair preference 和 executable 各有独立 unreduced loss/mask；
- candidate head 先在样本内有效候选聚合，pair head 只在双 outcome 有效时启用，再按冻结权重
  计算“有效 head 加权和/有效权重和”；无有效 head 的 row 排除，整 batch 无有效监督时直接失败；
- `world_v3_loss`/`world_vla75_loss` 保持既有返回形状兼容，内部统一使用逐样本 reducer 并报告
  每个 head 有效计数；
- 持久 `GroupDROState` 预登记 map/family/weather/group，按 detached 真实多任务 group mean
  指数更新并应用 floor；当前 batch 未出现的 group 保留历史，空 group 记录
  `NOT_MEASURED/count=0/loss=null`；coverage/temporal penalty 不进入 Group-DRO 风险。

失败行为：mask 外 target 变化不影响 loss，repair/executable 独立 mask；空监督 batch 不可优化，
空 group 不会伪装成零风险或 gate pass。

### C1.3 唯一 temporal selector

- 新增 `safedrive_foundry/data_pipeline/h6/temporal.py` 的纯状态机，状态仅保存作用域内
  `expert`/`vla` source 和 source EMA，不保存 candidate ID；HOLD 总是返回本 tick fresh ID；
- scope 固定由 run/episode identity 与 route revision 组成，变化时重置 EMA/hold/history；
- 顺序固定为 scope/eligibility、EMA、held unavailable/emergency risk、emergency margin、minimum
  hold、hysteresis、普通 switch/choose；disposition/reason code 与 C1 合同一致；
- VLA75 live `H5WorldRouter` 和 offline `select_vla75_router_config` 调用同一核心并输出同一 trace；
  旧 H5 historical 路径保持原行为；single candidate、feature/deadline/low-confidence/forced defer
  也进入统一 defer core，按 held→Expert→VLA→MRM，非 MRM 结果仍交给 Safety。

失败行为：scope 不匹配会重置，held source 不可用或 unsafe 不会复活，Guard REJECT 不会重新
进入候选；无 eligible source 返回 `DEFER_SINGLE_CANDIDATE`/MRM 稳定原因。

### C1.4 single tick owner 与 cleanup

- `scripts/h5_collect.py` 删除 runtime 外直接 `world.tick()` 和强制推进/强杀恢复；正常运行仍只
  通过 `ScenarioRuntime.tick_controls()`；
- infrastructure retry 前只读检查 scene/settings/tick owner：clean scene 允许既有的一次 bounded
  retry；存在 vehicle/walker residue 或 ownership/settings 不可确认时，写
  `NEEDS_USER_ACTION/CLEANUP_RESIDUE` artifact 后停止；
- failure artifact 记录 residue IDs、tick owner、`tick_advanced=false`、cleanup/retry 状态和自哈希。

失败行为：residue 永不调用 fake/world tick；clean scene 也不推进 world，只允许一次既有 retry。

### C1.5 benchmark 与 Evidence 真实性

- `scripts/h5_ultimate_benchmark.py` 现在只产生
  `benchmark_scope=latency_only_smoke`、`model_state=random_untrained`、
  `quality_gate_eligible=false` 的 self-hashed artifact；
- 默认写入非冻结 `generated/h6/c1-smoke/ultimate-latency-smoke.json`；总状态只能是
  `SMOKE_COMPLETED` 或 `SMOKE_FAILED`，CPU/GPU 使用各自 latency threshold，局部 PASS/FAIL
  只描述 latency，不能解释为模型质量；
- C1 工程 Evidence 更新为 `IMPLEMENTED`；没有新增 GPU/CARLA 数字，CORA algorithm 仍为
  `PLANNED / NOT_MEASURED / NOT_VERIFIED`。

### 实际验证

第一次在未激活虚拟环境的 shell 直接运行 `python` 返回 `command not found`（exit 127）；随后按
任务文件激活 `/home/sdf/.venvs/sdf` 并重跑。最终实际结果：

```text
python -m unittest tests.hybrid.test_world_v3 -v
  24 tests / OK

python -m unittest tests.hybrid.test_vla75_hardening -v
  32 tests / OK

python -m unittest discover -s tests -t . -v
  456 tests / OK / skipped=1
  skipped 项是既有 SDF_CONTRACT_LIVE_FORWARD GPU+checkpoint 条件测试

python -m compileall -q safedrive_foundry scripts tests
  passed

git diff --check
  passed

git diff --stat / 完整 git diff 人工核对
  passed；17 个 tracked 文件有 C1 改动，另有 evaluator.py、temporal.py 两个新实现文件
  test_registry.sqlite3 仍是唯一与 C1 无关的未跟踪文件
```

补充审计期间曾两次把单个测试指定到不存在的类名，unittest loader 各返回 2 个加载错误；改用
文件内真实类名后对应 Group-DRO/selection/readiness 定向测试均通过。这是测试命令定位错误，
未触发产品代码修复或删除测试。最终专项和全量结果以上述完整模块/发现命令为准。

专项 evaluator 测试只在临时目录以微型 CPU 模型生成测试 checkpoint/evaluator，用于验证真实
计算与 hash/fail-closed 消费；它不是项目/CORA 训练结果，不升级 algorithm Evidence。GPU peak
在该测试中为 `NOT_MEASURED`，没有 CUDA/CARLA 实测数字。

剩余接管条件：C2 必须由用户单独授权、更新 `START_TASK.md`，并按
`docs/COUNTERFACTUAL_DATA.md` 冻结 development collection contract；C1 完成后不得自动采集、
训练、运行 pilot/formal 或进入 C2。

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
