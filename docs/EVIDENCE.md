# SafeDrive Foundry Evidence 与归档索引

## 1. 状态与引用规则

Evidence 状态只允许：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

- `PLANNED`：只有合同/计划，没有实现或测量；
- `IMPLEMENTED`：代码/测试存在，不代表 GPU/CARLA 运行；
- `MEASURED`：实际运行并保存 artifact，但尚未完成冻结复核；gate pass/fail 是独立维度；
- `VERIFIED`：冻结配置、数据、哈希、评估和审计完成。

只有 `VERIFIED` 数字可在不附“开发/未验证”限定时引用。`GATE_FAILED` 的 VERIFIED Evidence
表示负结果本身可信，不表示研究目标成功。失败、负收益、尾延迟、deadline miss、资源与
reset/provenance 缺陷必须与正结果同等保留。

正式 artifact 至少绑定：

```text
commit and full dirty-worktree identity
config/schema/matrix/split/seed lineage
CARLA/Python/PyTorch/CUDA/model/checkpoint versions
observable/candidate/Guard/World/router/Safety/executable/applied chain
raw outcomes and aggregate metrics
P50/P95/P99, deadline miss, GPU peak
dataset/evaluator/summary/run-lock self hashes
cleanup and terminal status
```

代码存在、单元测试、开发强制采样、单帧 calibration、随机模型 latency 或用户口头确认均
不能升级为正式 CARLA/模型 Evidence。

阶段执行状态与 Evidence 成熟度分开：`C1 COMPLETED / STOPPED` 只说明 correctness 工程代码
和测试已完成，不能据此把 CORA algorithm 从 `PLANNED` 升级。Gate 也是独立维度；
`VERIFIED / GATE_FAILED` 表示负结果可信，不是成功。

## 2. 当前状态总表

| 阶段 | Evidence 状态 | Gate | 可引用结论 |
|---|---|---|---|
| H0 | VERIFIED | — | 活动路线和归档边界完成 |
| H1 | VERIFIED | contract passed | 真实 VLA/Expert 双候选与执行身份链成立 |
| H2 | VERIFIED | GATE_PASSED | paired outcome 数据存在真实选择空间 |
| H3 | VERIFIED | GATE_PASSED | 小型开发/OOF 上 candidate scorer 达到冻结门 |
| H4 | VERIFIED | GATE_PASSED | 小型 locked set 上 World 超过 simple baseline |
| H5 | VERIFIED | GATE_FAILED | 未证明 World closed-loop 可复现净收益 |
| H6 v1 | MEASURED | GATE_FAILED | seed 101 pilot 未达 World/VLA-primary gate |
| H6 v2 | IMPLEMENTED | NOT_RUN | 新代码存在，无新 checkpoint/CARLA formal |
| H6-CORA C1 engineering | IMPLEMENTED | PASSED | evaluator/loss/selector/tick-owner/benchmark 加固与离线测试完成 |
| H6-CORA C2 data | MEASURED | GATE_FAILED | 351 valid paired roots；真实覆盖不足，已冻结并停止 |
| H6-CORA C2 repair v2 | MEASURED | GATE_FAILED | 351 base roots / 1295 branches 的 v3 更正与 root 去重完成；Town03 diagnostic 12 roots / 36 branches，repair-failure 0/2，未进入正式批次 |
| H6-CORA C3+ algorithm | PLANNED | NOT_AUTHORIZED | 无 checkpoint、calibrated router、formal 或闭环结果 |

## 3. 指标口径

| 名称 | 本项目固定解释 |
|---|---|
| decisive | 两候选按冻结 outcome/utility 规则存在可判定 winner 的 pair |
| pairwise coverage | 两个 eligible candidate 都有原始 World 输出且 selector 可比较的比例 |
| defer coverage | 旧阶段定义下非 defer 的可判定覆盖；引用时必须带对应阶段 schema |
| unsafe | 按对应阶段冻结的 collision/red-light/offroad 聚合；跨阶段不能默认同定义 |
| progress delta | 同 scenario/root 的处理臂减对照臂 route progress，单位 m |
| lower-95 | 对应 artifact 冻结方法得到的 95% 下界；必须说明 bootstrap/cluster unit |
| P99 | artifact 中指定测量边界的 99 分位；scorer microbenchmark 不等于全链 tick latency |
| source usage | applied identity 的诊断分布，不是 World quality 或安全指标 |

后续 CORA 必须把统计单位冻结为 root anchor/scenario；branch、intervention 和 tick 不能当独立
样本缩窄置信区间。

## 4. H1 contract Evidence

成功 run：

```text
docs/runtime-evidence/h1/h1-smoke-20260812T161321Z/h1_smoke.json
sha256 2be0a5171856848bf52fb1ac48bbc88e714d65b8ff7c1811b89baae0bc857db7
```

已验证：

- CARLA 0.9.16 / Town03 / RTX 4080；
- anchor、front camera、VLA/Expert 绑定同一 frame；
- VLA forward count 为 1；
- 两候选 Guard PASS 且 DISTINCT；
- selected/final/executed/applied ID 连贯；
- Safety ACCEPT，控制为 TRACK_APPROVED；
- 运行完成后 settings/tick owner 恢复。

两次 camera barrier timeout 失败 run 同样保留，不能删除。H1 只证明合同，不证明驾驶性能。

## 5. H2 paired outcomes

最终 gate-pass dataset：

```text
generated/h2/paired-outcomes/h2-gatepass-20260813-routefix/
docs/runtime-evidence/h2/h2-gatepass-20260813-routefix/final-delivery.json
```

冻结身份：

```text
physical_manifest_sha256  6e74a789647182d9333cd99a69305bc2700a95216ebc7f34d2af21024a6d48ed
store_manifest_sha256     22d11961c74509843a1df6ea453794fad2519fcc42077540c33ce46e9f3c3524
config_sha256             70996b2b2a0d88cd02c210e75206cc1be1f189fae249979d14c417c866092043
offline_audit_sha256      3dc0573b5fe7a80fc3358f1e11d1c981d1fbe900f07357acc30a7b40d389b585
```

Verified 结果：

```text
120/120 terminal
108 valid/distinct
83 decisive
Expert/VLA wins = 51/32
source-only baseline = 0.6144578313
whole-GPU peak = 8.3720703125 GiB
dataset = 1,480,172,014 bytes
status = GATE_PASSED / STOPPED
```

旧 `h2-final-20260813-scenariov2-cleanup` 为 VERIFIED/GATE_FAILED，Expert 单边胜出、配额与
source-only 门失败。它仍是有效负 Evidence，不参与 CORA 新标签或 formal。

## 6. H3 development Evidence

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/final-delivery.json
evidence_sha256 f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e
```

Verified 结果：

```text
OOF decisive = 91/91
best frozen non-learning simple baseline = 84/91 = 0.9231
bootstrap lower-95 = 0.0330
ECE = 0.00113
P99 = 13.89 ms
deadline miss = 0
status = GATE_PASSED / STOPPED
```

限制：learned candidate-only MLP 同样达到 91/91；full-feature MLP 87/91；hard scene gate
结构性影响 history masking。H3 不能无保留证明上下文 World 优于所有 learned baseline。

阶段详细报告归档在：

```text
archive/2026-08-27-cora-document-consolidation/historical-stage-docs/H3_DELIVERY_REPORT.md
```

## 7. H4 locked Evidence

```text
docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json
evidence_sha256 35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4
split_manifest_sha256 17dedd305aaf2933266a15345926f035aa7ebcd3210b6c636cc92d99e676b08c
```

Verified 结果：

```text
locked decisive = 64
World = 64/64
best simple = 57/64 = 0.890625
defer coverage = 63/64
P99 = 15.23 ms
GPU peak = 0.03125 GiB
deadline miss = 0
status = GATE_PASSED / STOPPED
```

限制：test 小、simple baseline 已高、temperature 到 0.05 下边界、微基准不是完整闭环尾
延迟、地图/family/weather 覆盖有限。

详细报告归档在：

```text
archive/2026-08-27-cora-document-consolidation/historical-stage-docs/H4_DELIVERY_REPORT.md
```

## 8. H5 closed-loop 最终负结果

```text
docs/runtime-evidence/h5/h5-pilot-all2/final-delivery.json
evidence_sha256 846ef8a6f5ff6b3ca330a55ba53f69849f043346b74910f6a405eb95f5543517
```

Verified 结果：

```text
runs = 222/222
paired scenario roots = 74
World ON unsafe = 4
World OFF unsafe = 6
ON-only unsafe = 0
paired progress mean = +0.2549 m
bootstrap lower-95 = -0.0709 m
World ON/OFF switches = 14/0
scorer P99 = 47.2159 ms
deadline miss = 1
reset mismatch = 1
status = GATE_FAILED / STOPPED
```

正式解释：样本内没有 ON-only unsafe，但未做出统计安全优越/非劣证明；route-progress 没有
统计稳定正收益，切换、资源与完整性也未全部过门。不能用 H3/H4 离线结果改写。

H5 进入/矩阵文档已经执行完毕并归档，不得复用其 seed 或根据结果修改后重跑。

## 9. H6 VLA-primary Evidence

旧正式 pilot：

```text
docs/runtime-evidence/h6/h6-vla90-formal-pilot-20260820-v1/final-delivery.json
evidence_sha256 8dae5c2e661abafc1dceab633d3338201a7fe1e6b50ecd5e334641aa68223194
```

Measured 结果：

```text
ticks = 600
World pair scored = 590/600
strict World VLA preference = 131/600 = 21.83%
VLA applied = 285/600 = 47.50%
Expert applied = 315/600
MRM = 0
VLA Guard = PASS 453 / REVIEW 147 / REJECT 0
Safety fallback to Expert = 18
RATO/QP repair = 42/26
unsafe delta = 0
paired progress lower-95 = +0.629 m
paired scenario roots = 12
scorer P99 = 9.12 ms
deadline miss = 0
switches = 31
ping-pong scenarios = 5
provenance failure = 10 missing World pair-score ticks（同一 Town01 aggressive-cut-in run）
status = GATE_FAILED / NOT_VERIFIED
```

冻结 gate failures：actual VLA coverage、World VLA preference、switch rate、ping-pong、
provenance。progress lower-95 为正、deadline miss 为 0，不会覆盖其他 gate failure。

seed 101 已消费。历史 90% preference/usage gate 和 2026-08-27 的 75% usage 修订只用于解释
旧 H6，不是 CORA 当前优化目标，也不能回写历史结果。

H6 VLA75 v2 的 14-output、A/B/C lineage、run-lock、acceptance 和 hardening 是
`IMPLEMENTED`；没有新 GPU checkpoint、CORA 数据、held-out pilot/full，不能升级。

旧 H6 handoff 已归档：

```text
archive/2026-08-27-cora-document-consolidation/historical-stage-docs/H6_VLA75_HANDOFF.md
```

## 10. H6-CORA Evidence 合同

当前 program：`C0 COMPLETED / C1 COMPLETED / C2 COMPLETED / DATA MEASURED / GATE_FAILED /
STOPPED`。C3 仍为 `NOT_AUTHORIZED / NOT_STARTED`。已有 CORA paired data，但没有项目
checkpoint、calibrated router、formal 或闭环数字。

### C1 正确性

C1 只产生代码、测试和 artifact schema Evidence，工程状态为 `IMPLEMENTED`：

```text
safedrive.world.vla75.evaluator.v1
safedrive.world.vla75.training_summary.v2
safedrive.h6.vla75.run_lock.v2（C1 bindings required）
self-hashed C1 readiness
self-hashed cleanup failure artifact
self-hashed latency-only random-model smoke artifact
```

离线测试证明缺失/占位 evaluator fail closed、per-sample loss/mask/Group-DRO 合同、offline/live
共享 selector、collector 不直接 tick、artifact hash/tamper 和随机 benchmark 不可获得质量 gate。
微型 CPU evaluator 测试只产生临时测试 artifact，不是项目模型 Evidence；未测 GPU/CARLA 时
latency/显存/闭环均不能升级为正式 `MEASURED`。因此 C1 engineering 是 `IMPLEMENTED`，CORA
algorithm 保持 `PLANNED / NOT_MEASURED / NOT_VERIFIED`。

### C2 数据

最终数据与 Evidence：

```text
generated/h6/cora/h6-cora-c2-dev-20260830-v1/
docs/runtime-evidence/h6/h6-cora-c2-dev-20260830-v1/final-delivery.json
terminal roots = 351/351
valid nominal pairs = 351/351（development 324/324）
branch outcomes = 1295
nominal VLA forwards = 351
aggregate collector wall = 41184.8297303014 s
whole-GPU peak = 9.9462890625 GiB
minimum observed free disk = 124.15421295166016 GiB
dataset = 292689739 bytes
status = DATA MEASURED / GATE_FAILED / STOPPED
```

Pilot gate 通过。Development 的 manifest/run-lock、29-head public labels、feature reproduction、
inventory、reset/identity/cleanup/cross-fallback 和资源审计均通过。冻结 gate 失败只来自真实覆盖：
locked-development offroad 正例 1（门为 2）；repair success 负例四个 split 均为 0；executable
负例 train/validation/calibration/locked-development 为 9/1/0/2（门为 12/3/3/3）。三次外部
collector/server failure 已保存且失败耗时计入资源，immutable resume 后完成矩阵。Formal 未采集，
C3 未授权。

每个 dataset 至少保存：

```text
frozen matrix and split lineage
anchor/reset/candidate manifests
two branch outcomes per pair
invalid-pair reasons
pair label coverage by map/family/weather/risk
source/slot/branch permutation audit
feature leakage audit
artifact/self hashes
```

双 outcome 不完整的 row 不进入 pairwise training。

同时记录 proposal→Safety repair/executable→applied 的 intervention identity，以及按 source、
Guard、risk、branch order 的 missingness。CORA outcome 只解释冻结 CARLA/Safety/controller 下
的 proposal intervention，不升级为现实世界因果真值。

### C3/C4 模型与 router

保存：

```text
three-seed checkpoints
per-head metrics and valid masks
pairwise/selective regret
NLL/Brier/ECE/AUPRC/unsafe recall
source/candidate/action/context probes
worst-group report
calibration-only artifact and assumptions
risk-coverage/defer curves
offline/live trace parity
measured latency/VRAM when actually run
```

source-blind claim 只指 metadata schema；trajectory-to-source predictability 必须单独报告。
coverage 必须说明 per-head/per-candidate marginal 还是 joint，不能把 marginal 指标包装成系统级
同时覆盖。

checkpoint summary 中未测量字段为 `NOT_MEASURED`，不能用 0 或 `pass=true` 占位。

### C5 closed loop

pilot/full 每臂保存相同 candidate/Guard/Safety/controller/reset 条件下的 raw World、router、
Safety、executed/applied chain。pilot 失败不运行 full；formal 无论正负都冻结并关闭。

## 10A. C2 repair v2（2026-09-05）

修复版 Evidence 位于：

```text
generated/h6/cora/h6-cora-c2-repair-20260905-v2/
docs/runtime-evidence/h6/h6-cora-c2-repair-20260905-v2/
```

已实际生成并审计：351 个 base root、1295 个 base branch 的 v3 更正标签；648 行 train/Town03
Guard+Safety 离线筛选；root-cluster 去重统计；`test-report.json`（491 passed、1 skipped）；
12 个 Town03 diagnostic root、36 个 branch；`final-delivery.json`（`GATE_FAILED`, `DATA MEASURED`）。
旧数据和旧 Hash 状态按 `REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN` 复用，没有重扫旧文件
或模型。

CARLA admission Evidence 为 `admission.json`：显式 URL/`-ini` 参数在本机 Shipping 包中仍回落
Town05，随后使用可恢复的 Windows-side `DefaultEngine.ini` 临时覆盖启动 Town03 并通过 READY；
原配置已备份并在采集后恢复，不进入 Git。诊断采集 elapsed 256.77 s、RTX 4080 峰值约 8.35 GiB，
加上此前启动/恢复 188.46 s，aggregate CARLA wall 为 445.23 s。诊断得到 3 个有效 offroad root，
但 0 个实际尝试且全部修复失败的 root，未满足 2 个 repair-failure diagnostic gate，故不执行正式
批次。最终报告保留 10 项 coverage 缺口和 diagnostic gate 缺口，CARLA 已关闭、tick owner free，
不把工程修复或诊断采集写成数据门通过。

## 11. 环境诊断边界

跟踪的历史诊断：

```text
docs/environment/evidence/g0-05/doctor.json
docs/environment/evidence/g0-05/doctor.md
```

它只证明对应 run/进程当时的检查结果。2026-08-27 受限代理进程未访问到 GPU/CARLA，用户
随后明确确认本机资产可用；两者不冲突。后续 live task 必须在实际执行上下文重新产生
task-local CUDA/preflight Evidence。

## 12. 归档索引

2026-08-27 开始、2026-08-29 完成的文档收敛归档：

```text
archive/2026-08-27-cora-document-consolidation/README.md
```

该目录保存收敛前权威文档快照和被移出的阶段文档，记录原路径、原因与恢复方式。归档只读，
不得自动恢复旧任务、seed、阈值或 handoff。

更早的仓库/H-route 历史：

```text
archive/2026-07-23-repository-consolidation/
archive/2026-08-12-h-route-consolidation/
archive/legacy_project_2024_2025/
```

## 13. 引用检查

任何 README、简历、报告、视频字幕或面试 slide 在引用数字前必须回答：

1. 是 development、pilot 还是 formal？
2. 状态是 MEASURED 还是 VERIFIED？
3. gate passed 还是 failed？
4. 样本量、split 和主要限制是否同时展示？
5. 是否来自实际 artifact 而不是单元测试/随机 benchmark？
6. 是否把 CARLA SIL 错写成实车或量产安全？

不满足任一项时，数字不得无保留公开。
