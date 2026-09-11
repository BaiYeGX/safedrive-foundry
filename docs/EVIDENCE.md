# SafeDrive Foundry Evidence 与归档索引

2026-09-11 复核说明：C3 已有可用的 200-step M1 和全部 53-root 离线结果。本轮修正
独立有效率、P-WP 单位及导航共同支持，新增 10 项直接回归通过，完整预测复算一致。
见 [C3 学校项目结果页](C3_SCHOOL_RESULTS.md) 与 [周日交付安排](C3_C4_FINISH_PLAN.md)。
历史完整验收措辞由此收窄：smoke/失败资源总账和真实 C4 联合 smoke 尚未完成，旧报告
保留原样；不能用旧 verify 的通过字符串代替这些检查。当前可引用范围是已有开发数据上的
离线适配效果；实时 CARLA 与 C4/C5 的效果须由后续实际运行证明。

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
| H6-CORA C2 repair v2/v3 | MEASURED | GATE_FAILED | v2 保留历史失败证据；v3 修通 recipe/trace/预算链，Town03 diagnostic 12 roots / 34 branches，repair-failure 1/2、offroad 5/1，正式批次被诊断门阻断 |
| H6-CORA C2 dev baseline | MEASURED | DEV_BASELINE_GATE_PASSED | 现有数据 release 340 usable roots；World 三 seed 已跑完，`NO_DEMONSTRATED_GAIN`；原 coverage gate 仍失败 |
| H6-CORA C3 repair | MEASURED | OFFLINE_BASELINE_USABLE / RESOURCE_REVIEW_PENDING | 200-step M1、53-root 主误差与实际 canonical 有效率复算一致；范围为 `partial_native_support`；历史资源与完整联合 smoke 仍有缺口，C4 未执行 |
| H6-CORA C4–C6 | PLANNED | NOT_RUN | C4 联合微调、C5 闭环和 C6 交付尚未启动 |

## 3. 指标口径

当前权威记录为 [C3 修复最终验收](runtime-evidence/h6/c3-final-repair-20260910T165902Z.md)，
身份索引为 [c3-repair-20260910T165902Z-index.json](runtime-evidence/h6/c3-repair-20260910T165902Z-index.json)。
此前勘误仍解释旧运行为何撤回；旧运行不覆盖当前修复版。当前指标是带
`partial_native_support` 限制的开发集测量，不能引用为普遍驾驶能力。

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

当前 program：C0/C1 完成；C2 原覆盖门 GATE_FAILED，C2 dev baseline 的工程门
DEV_BASELINE_GATE_PASSED、学习结果 NO_DEMONSTRATED_GAIN。C2 已有三 seed MLP checkpoint。
C3 修复版已于 2026-09-10 完成并验证；C4–C6 仍是待实施证据合同。C3 的权威记录见下方
`C3 修复版 VERIFIED` 段落，初版和修复尝试的旧记录保留但已明确撤回。

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
C3 当时未授权；当前后续入口见 START_TASK。

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

### C3 — 常规 VLA 微调（原计划合同；已由下述 VERIFIED 记录实现）

保存 SFT root manifest、teacher 身份、原生 target 对齐、M0 离线预测、M1 adapter/heads、
超参/seed/实际步数、梯度与 round-trip、逐 root 预测、wall/显存和全部失败。
真实 batch smoke 不等于完整训练；未测字段为 NOT_MEASURED/null。
不重扫旧模型/数据 Hash，新权重必须有新身份。

### C3 — 常规 VLA 微调（当前修复版 VERIFIED，2026-09-11）

当前权威 run：

```text
generated/h6/cora/c3-repair-20260910T165902Z/
```

当前 run 的逐项数据、监督来源、补采阻断、M0/M1 数字、简单基线、资源、独立复算和六类
验收攻击见 [C3 修复最终验收](runtime-evidence/h6/c3-final-repair-20260910T165902Z.md)。
`verify.json` 为 `VERIFIED` 且 `errors=[]`；`deep-self-check.json` 为 `PASSED`，六类篡改
均被拒绝。route ADE 虽相对 M0 降低，但略差于 train-mean，且 29/53 个 root 退化；speed
waypoint ADE 改善。因此这里的 `ALGORITHM_MEASURED` 只表示固定离线任务的实测结果，
不表示算法泛化或闭环安全成立。

### C3 — 常规 VLA 微调（早期修复 run，SUPERSEDED）

权威 run：

```text
generated/h6/cora/c3-repair-20260910T100651Z/
```

修复版撤回初版 `c3-vla-sft-20260909-final-v3` 的监督、验收和 90.77% 路线结论；完整
原因与历史路径见 [C3 SFT 勘误](runtime-evidence/h6/c3-sft-repair-erratum.md)，可提交的
身份索引为 [`c3-repair-20260910T100651Z-index.json`](runtime-evidence/h6/c3-repair-20260910T100651Z-index.json)。
初版和此前 repair attempts 继续保留，但不能用于模型选择或活动比较。

修复版读取冻结 C2 dev release，保留 train 158 / validation 53 和 211 个 root，manifest
SHA-256 为 `54281a1de6207b4e0c553ee8456de40f43131a6ef30b434e7bd1064da776c2fd`。route 先按
当前位置投影到有序 native expert reference path、裁去已通过前缀，再按 1 m 采样 20 点
(0…19 m)；speed 独立使用 canonical expert proposal 的 10 点、0.25 s 合同。导航仅作输入/
诊断，不能冒充专家标签。缺失、碰撞终止和未对齐 timeline 按逐头 mask 保留，3 个碰撞终止
train root 的 speed 监督无效；anchor speed 的 211 个零值和 history disagreement 全部披露。
无补采、无 future/World label 泄漏，validation 输入与 mask 在预测前冻结。

真实 RTX 4080 CUDA/BF16 smoke 通过，LoRA/驾驶 head 共 18,417,664 参数更新，冻结视觉、
基座和 query 指纹不变；原生 forward/backward、adapter 和独立 checkpoint reload 通过。
正式 M1 从原始 M0 开始，seed 17、AdamW、LoRA `2e-5`、驾驶 head `1e-4`、weight decay
`0.01`、microbatch 1、累积 4，完成 200/200 updates；5 轮共 790 个样本暴露，root exposure
均为 5，尾部窗口按 39×4 + 1×2 保留。C4 成本 smoke 的 zero residual 为 0、shared LoRA
gradient 非零、两组 clip 后范数均 ≤1.0；C4 尚未训练，M2 不能从 M1 续训。

固定 53 个 validation root 的 root-equal 指标（bootstrap 1000、seed 71，`eps_ADE=1e-5 m`）为：

| 指标 | M0 | M1 | Δ（M0−M1） | 相对改善 | 95% 配对 CI |
|---|---:|---:|---:|---:|---:|
| P-ADE route (m) | 0.210514 | 0.232211 | -0.021698 m | -10.3069% | [-0.159394, 0.103600] |
| P-FDE route (m) | 0.735295 | 0.806690 | -0.071395 m | -9.7097% | [-0.625551, 0.471091] |
| P-WP speed (m) | 2.178557 | 0.336781 | 1.841776 m | 84.5411% | [1.570158, 2.128422] |
| P-VALID | 100.0% (53/53) | 100.0% (53/53) | — | — | — |
| P-FAIL | 0.0% (0/53) | 0.0% (0/53) | — | — | — |

P-SPEED 为 `N/A_without_trusted_time_speed_ground_truth`。route 为负、speed waypoint 为正，
不称全面改善；工程状态与算法收益分开。优化总账（含历史失败/撤回运行的保守上界）为
`8.622365 h`，观测整卡 allocated/reserved 峰值为 `9.889132/10.845703 GiB`，低于既定
10 h / 14.5 GiB 限制。`verify.json` 为 `VERIFIED`、`errors=[]`，独立重算和篡改拦截证据已
留在该 run。实际验证命令为
`/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests -t . -v`（509 tests、1 skipped、
`OK`，75.094 s）、`/home/sdf/.venvs/sdf/bin/python -m compileall -q safedrive_foundry
scripts/h6_cora_sft.py` 和 `git diff --check`；篡改案例摘要见
[`c3-repair-20260910T100651Z-verify-attacks.json`](runtime-evidence/h6/c3-repair-20260910T100651Z-verify-attacks.json)。

### C3 — 常规 VLA 微调（初版 VERIFIED，已撤回）

> 以下段落是 2026-09-09 的历史记录。其监督、manifest、指标和 `VERIFIED` 状态均已被上方
> 修复版取代；其中的 90.77% 路线改善不能引用。

权威 run：

```text
generated/h6/cora/c3-vla-sft-20260909-final-v3/
```

可提交的产物身份索引：
[`docs/runtime-evidence/h6/h6-cora-c3-sft-20260909-final-v3/c3-sft-index.json`](runtime-evidence/h6/h6-cora-c3-sft-20260909-final-v3/c3-sft-index.json)。
大型权重和逐 root 原始 JSON 仍按资源规则保留在上述本机目录。

该 run 使用冻结 C2 dev release `h6-cora-c2-devbaseline-20260907-v1`，manifest SHA-256
为 `e5116aedd2b79e1512f4f0567f1b740880623fce4f977e9470bf6786f21c1bb3`，release index SHA-256
为 `0c867199d9b6682648471b21a2ab850c86bf8f1eb4d8eb99d73abe3e7c8789b8`。审计通过 211 个
sample：train 158 roots、validation 53 roots；route 原生 20 点共有效 3160/1060 点，
speed 执行时间线原生 10 点共有效 1550/530 点，3 个提前终止 train sample 的 speed 缺失
mask 被保留。输入仅含部署时可用的图像、navigation、ego/history；World/future/source
字段均未进入 SFT，重复 root 和 split 交叉污染检查通过。执行时间线行数为 1…50，dt 为
0.05 s；没有执行补采，因为 C2 release 已含完整可信 expert timeline。

真实 CUDA batch smoke 使用 RTX 4080、BF16，336 个 LoRA 与 10 个驾驶 head checkpoint
key 均匹配，missing/unexpected 均为 0。LoRA/驾驶 head 共 18,417,664 个参数更新，
视觉投影、基座和原生 query embeddings 保持冻结；forward/backward、保存和独立 fresh
reload round-trip 最大绝对差 0（容差 `1e-5`），峰值 allocated/reserved 为
`4.319/4.914 GiB`。C4 共享 hidden 梯度诊断为非零 auxiliary path（hidden 896，SFT
norm 21.1636，aux norm 2.9575，cosine -0.3814，门控权重 0），未混入 C3 loss。

M1 按 seed 17、AdamW、LoRA `2e-5`、驾驶 head `1e-4`、weight decay `0.01`、microbatch 1、
累积 4、200 更新和 10 步 head-only warmup 完成；LoRA 从第 11 步解冻，checkpoint 可独立
重载且冻结指纹不变。训练耗时 267.697 s，峰值 allocated/reserved `4.259/4.398 GiB`，
低于 14.5 GiB 限制。M0/M1 在相同 53 validation root 上 root 等权，bootstrap 1000 次、
seed 71，`eps_ADE=1e-5 m`：

| 指标 | M0 | M1 | Δ（M0−M1） | 相对改善 | 95% CI |
|---|---:|---:|---:|---:|---:|
| P-ADE route (m) | 5.032139 | 0.464337 | 4.567802 m | 90.7726% | [4.204761, 4.863890] m |
| P-FDE route (m) | 5.576035 | 0.585869 | 4.990166 m | 89.4931% | [4.614955, 5.325819] m |
| P-WP speed (m) | 1.984764 | 0.531818 | 1.452946 m | 73.2050% | [1.240386, 1.675059] m |

M0/M1 root count 均为 53，prediction fail rate 均为 0%。P-SPEED 为 `N/A`，因为没有可信
真实速度换算 ground truth。点估计是一次 seed 的 validation 开发结果，不能代替独立测试或
稳定性结论；逐 root 明细、有效点数、mask、失败原因、预测和资源账本均已保存。`verify.json`
状态为 `VERIFIED`，并绑定以下核心身份：代码 SHA-256
`a41d4740993b3d3d67950940abb5b7c2ff4a1d9b0a42e66e57f912a4c862bf81`，运行时 Git HEAD
`598308fd4cb57df94f02f784404635e942c3ff9c`，模型 SHA-256
`ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28`。大型 checkpoint 和
原始运行数据保留在本机 ignored 目录；Git 提交只包含代码、测试、文档和可审阅的结果身份。

实际验证命令：

```text
/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests -t . -v
# Ran 506 tests in 73.859s; OK (skipped=1)
/home/sdf/.venvs/sdf/bin/python -m compileall -q safedrive_foundry scripts
git diff --check
```

C3 验收后入口已更新为 C4；本轮没有自动启动 C4，也没有启动 CARLA 补采。

### C4 — 后果辅助联合微调（PLANNED）

保存 M2 与 M1 同起点/同数据/同 seed/同更新预算的记录、四头 target/mask、
World-only LoRA 梯度、候选交换检查、策略/World 开发误差及既有 ridge 对照。
配对差分是已有方法，不作为新发明。无执行中介模型、无 ensemble/新风险头要求。

### C5 — 小型开发闭环（PLANNED）

A=M1 固定 eligible VLA→Expert；B=M2 相同规则；C=M2 World rank/defer。
6 roots / 18 runs，每 run 最多 60 s 仿真。新 lineage/recipe/规则先登记；
不用旧 seed 101、reserved 173/179 或旧保留集。shadow 同计算负载与真实部署成本分开记录。
保存所有 raw/selected/repaired/executable/applied、事件、逐 root 进度、干预/defer、延迟与失败。
无独立校准，明确 UNCALIBRATED；这是开发结果，不是 formal、安全非劣或全域保证。
首个登记 root 检查工程链，失败停止；不新增独立 pilot/正式双矩阵。

### C6 — 最小可复现交付（PLANNED）

两份新权重、配置、结果表、典型 replay、架构图、方法/实验/限制草稿与真实复现入口。
训练/闭环没做则 PARTIAL；负收益可冻结结题，不能制造正结果。
三份研究备忘录是来源与说明，不再承担独立排期、预算或验收。

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

## 10B. C2 repair v3（2026-09-06）

修复版 Evidence 位于：

```text
generated/h6/cora/h6-cora-c2-repair-20260906-v3/
docs/runtime-evidence/h6/h6-cora-c2-repair-20260906-v3/
generated/h6/cora/h6-cora-c2-repair-20260906-v3/budget-ledger.json
```

本版沿用 351 个 base root / 1295 个 base branch，新增 12 个 Town03 diagnostic root、34 条执行
branch；合并后为 363 roots / 1329 branches。采集器读取冻结 screening recipe，保留实际
source/operator/multiplier、Safety trace、parent identity 和不适用记录；旧文件和模型没有进行
Hash 扫描或重算。预算账本记录本版已记录 CARLA 工作时间 118.4057 s，CARLA 已关闭且 tick owner
已释放。

v3 诊断门实际结果为 1/2 个独立 `repair_attempted=true, repair_success=false` root 和 5/1 个
offroad root。由于修复失败门未达到 2，正式 batch collector 在读取 v3 labels 后拒绝启动；没有
使用诊断或 branch 数量补齐 root，也没有进入 C3。全量回归为 492 tests run、1 skipped、OK。
最终 `final-delivery.json` 与 `data-quality.json` 保留 `DATA MEASURED / GATE_FAILED / STOPPED`
及 10 个 coverage 缺口；本版不能写成 `GATE_PASSED`。

## 10C. C2 dev baseline（2026-09-07）

新合同的 release 与 Evidence：

```text
generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1/
docs/runtime-evidence/h6/h6-cora-c2-devbaseline-20260907-v1/
safedrive_foundry/config/h6/cora_c2_devbaseline_v1.toml
```

这次不启动 CARLA，也不重新验证旧文件/数据/模型 Hash；release 只引用 base 数据、既有 v3
corrected-label sidecar 和登记身份。全部 351 个 root 被盘点，按物理初态与 capture 条件隔离
11 个重复后保留 340 个 usable roots：

```text
coverage_pilot 25   train 158   validation 53   calibration 52   locked_development 52
```

训练只允许 train，开发评估只允许 validation；intervention、diagnostic、repair 和稀有事件
head 不参与本轮优化。29 个公开 head 仍逐 head 保存 value/unit/mask/derivation；World 只学习四个
连续 head。旧身份政策固定为 `REUSE_RECORDED_IDENTITIES_NO_OLD_FILE_HASH_SCAN`。

World 是 CPU 四线程、共享 128/64 MLP，三个固定 seed 17/29/43，输入为 499-D context＋`10x8`
candidate。对照包括 train mean、context-only ridge、candidate-only ridge 和 Expert/VLA 固定
选择。每个模型和对照在相同 validation roots 上报告四个 head 的原单位 MAE/RMSE、进度差、
1,000 次 root bootstrap 及 map/family/weather 分组。候选交换检查通过（最大输出差约 `2.4e-7`），
source/slot 元数据不进入输入；candidate-only ridge 的 progress MAE 约 `0.302 m`，MLP 三 seed
约 `0.555/0.617/0.555 m`，因此 `gain_status=NO_DEMONSTRATED_GAIN`。该结果是当前采样分布的
validation 开发结果，不是安全概率、闭环收益或通用策略价值。

验收：

```text
data release = DEV_DATA_READY
final delivery = DEV_BASELINE_GATE_PASSED
original coverage gate = GATE_FAILED
tests = 495 run / 1 skipped / OK
closed_loop = NOT_MEASURED
vla_finetune = NOT_RUN
```

`final-delivery.json`、`audit-report.json`、`baseline-report.json` 和 `test-report.json` 是本次
可复核入口。由于 baseline 只是开发基线，不升级本 Evidence 为 `VERIFIED`，也不自动启动
calibration、C3 或 VLA 微调。

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


## 单阶段 goal 交付摘要

C3/C4/C5/C6 都必须在自己的新 run-id 下保存 stage-summary.json，关联实际输入身份、
产物路径、固定 config、真实命令、测试结果、资源与失败、剩余问题及下一阶段。
不能把上游存在的文件列表当成本阶段真实产物；NOT_RUN/null 不得写为 0 或通过。
摘要用于下一 goal 核对前置，PLANNED 文档不是前置验收证明。
C3 配方见 VLA_FINETUNE_WEEK_PLAN；C4 见 WORLD_MODEL；C5 见 HYBRID_CANDIDATES；
C6 见 SHOWCASE。负收益与工程完成分开，预算不匹配或缺 runs 必须明确限制。


## 本轮固定配方与正向证据

新 C3/C4 摘要额外绑定共同预热/学习率日程、b 拟合子集/系数、零残差等价检查、
梯度余弦/门控/范数及激活比例。初始等于 b 不算学习收益，必须报告最终 b+R 对 b。
主要比较、eps_ADE、root CI 与 C/B 事件/进度/延迟判断以 PROJECT 为准，正式训练前登记。
小幅点估计优于对照不等于统计证实；辅助指标不能替换失败的主要指标。
保留全部历史报告及失败，不修改旧阈值；没有新训练/闭环时所有新收益为 NOT_MEASURED。

## C3–C6 论文指标证据合同（2026-09-08，PLANNED）

新实验依据 [PROJECT 论文量化合同](PROJECT.md#论文量化合同c3_c6_metrics_v1planned)，
在既有 run-id 目录保存 metrics-spec.json、逐 root 明细与 metrics-summary.json；名称是计划，
当前不代表这些产物已经存在。记录公式/单位/mask/聚合/version/hash、模型身份、分母、
失败/缺失、绝对/相对差、CI、目标和证据路径；null 必须附 reason。
C6 从该明细复算表格；工程通过、规划达标与测得优势分别验收。
保留旧 C2 数字及其旧权重/平局规则，不覆盖历史 Evidence，不把 NOT_MEASURED 填为 0。

## C3/C4 原生适配与优化干扰审计（2026-09-08，PLANNED）

计划记录原生空间/时间采样、cumsum、future 缺失 mask、部署输入一致性、
smoke 后 M0 重载身份、参数白名单/query 冻结、root 暴露和驾驶/World 各自裁剪系数。
来源为本地 SimLingo adaptor/dataset、运行器和旧 ridge 拟合源码检查；属于待实现合同，
未修改运行代码、冻结模型或旧 release。简单 CPU 范数算例只验证裁剪耦合的代数，
不是训练测试或模型收益。保留文献启发与本项目验证的区别，不向运行报告预填 2%–3%。
