# CORA 反事实 potential-outcome 数据合同

## 1. 要解决的问题

普通 closed-loop 日志在一个 tick 只能观察实际执行候选的结果：

\[
Y(\tau_{\text{executed}})
\]

未执行候选的结果缺失。把同一 episode 的总体结果贴给两个候选，或用第一 tick 特征监督
整段 outcome，会把策略选择、后续状态分布和候选真实作用混在一起。

CORA 的训练单位必须是同一 observable anchor 上两个候选各自的真实 short-horizon
potential outcome：

\[
(O_t,\tau_V,Y_V,\tau_E,Y_E)
\]

其中 treatment 是“把 Guard eligible proposal 交给冻结下游栈”，不是绕过 Safety 强行跟踪
原始轨迹：

\[
Y_i = Y\!\left(do(\text{proposal}=\tau_i);\pi_{Guard},\pi_{Safety},\pi_{control}\right)
\]

`τ_i` 始终指原 proposal；同时保存 Safety 输出的 `τ_i^exec`、repair/MRM 和 applied
identity。World 用 `O_t, τ_i` 预测这条 proposal 在冻结部署栈下的后果。若 Safety 拒绝并进入
MRM，该 feasibility/terminal 结果本身就是 label，不能改成“无 outcome”。

采集 branch 禁止跨候选 fallback；否则 `Y_i` 会依赖另一个候选和 fallback 顺序，不再是
per-candidate estimand。在线运行仍可按冻结顺序检查另一候选，但该 transition 作为 router/
Safety Evidence 单独记录，不写回 `Y_i`。

该合同复用 H2/H3 已验证的 capture、exact reset、单 tick owner、forced-single-candidate
Safety binding、50-tick control 和不可变存储；旧 H2 阶段文档已归档，不作为新矩阵来源。

## 1A. C2 repair v2 更正合同（2026-09-05）

修复版本 `h6-cora-c2-repair-20260905-v2` 采用 base 引用加 delta，不覆盖
`h6-cora-c2-dev-20260830-v1`。旧 branch 没有可追溯 repair trace 时，只有完整执行绑定的 QP/RATO
成功可以保留成功标签；无法判断是否尝试的 `repair_attempted`、`repair_success`、`repair_mode`
必须为独立缺测 mask，禁止从决策名称猜测失败。新 branch 保存
`safedrive.cora.safety_trace.v1`，逐次记录 proposal parent、QP/RATO mode、solver status、
reason、post-repair 与最终 executable。

标签使用独立 `safedrive.cora.outcome_labels.v3` sidecar，29 个 head 各自保存
`value/unit/valid/derivation_version`。route completion 使用绝对 route projection 与剩余路线
长度；red-light 只使用穿越停止线 tick 的灯态，穿越证据缺失时仅该 head 无效。统计按
`root_cluster_id` 集合去重：branch、intervention、edge、tick 和重试不能膨胀 root；Guard
REJECT、auxiliary-only、invalid branch 不进入核心 coverage。

修复执行状态为 `DATA MEASURED / GATE_FAILED`：base 351 roots、1295 branches 已更正，离线
train/Town03 screening 完成；通过可恢复的 Windows-side `DefaultEngine.ini` 临时覆盖后，
Town03 诊断实际生成 12 roots、36 branches。诊断有 3 个 offroad root，但没有实际尝试且
`repair_success=false` 的 repair-failure root（要求至少 2 个），因此不进入正式批次。该事实不
改变原始数据或门槛，也不允许把修复完成或诊断采集宣称为 coverage gate 通过；临时配置已恢复且
不进入仓库。

## 2. 已确认的旧数据缺口

对 `h6-vla90-train-pilot-20260820-v2` 的 loader 审计：

| seed | tick rows | 双 outcome | 单 outcome | 双 executable | whole-policy pairs |
|---:|---:|---:|---:|---:|---:|
| 89 | 1200 | 0 | 1197 | 0 | 12 |
| 97 | 1200 | 0 | 1198 | 0 | 12 |

因此旧 tickwise rows 不得作为 CORA pairwise supervision。旧数据仍可用于事实 outcome
baseline、回归和失败复现，但必须标记 `factual_single_outcome`，不能通过 mask/复制/episode
majority 补成反事实 pair。

## 3. Anchor 合同

每个 anchor 至少绑定：

```text
dataset_id / pair_id / anchor_id
map / route / scenario family / weather / seed lineage
CARLA version / world settings / fixed_delta_seconds
ego pose, yaw, speed and observable history
actor roles, transforms, velocities and scripted controls
traffic-light state, elapsed time and phase script
route polyline and navigation command
camera/content hashes and sensor timestamps
VLA/Expert raw and canonical trajectory hashes
Guard result at capture time
code/config/model/worktree hashes
```

anchor feature view 与 label store 物理分离。在线/训练 feature object 不能包含 branch order、
future actor states、outcome、winner、source answer、Oracle 或 formal split answer。

## 4. 双分支流程

1. 在清洁场景中预滚出冻结 observable history；
2. 同一 frame 分别生成一次 VLA 和 Expert；
3. 两条候选分别为 Guard `PASS/REVIEW`；不预先要求 Safety 可执行，否则 feasibility/repair
   负例会被选择性删除；
4. 捕获 anchor/reset signature 和原始 candidate object/hash；
5. 清理并按冻结脚本重建初态；
6. 只把候选 A 交给 forced-single-candidate Safety，禁止跨 source fallback；Safety 可按冻结
   合同 accept/repair/MRM，所有变换绑定输入/输出 hash；
7. 唯一 Runtime 以 20Hz 执行 50 ticks（2.5s）；
8. 保存 trajectory、control、ego timeline、actor future、events、latency 和 cleanup；
9. 再次重建完全相同初态，执行候选 B；
10. 两分支均完整、reset comparable、identity 通过后才形成有效 pair。

branch order 由冻结 hash 决定并做平衡，但 label 只引用 candidate ID/hash，不使用先后顺序。

这里得到的是 CARLA 内、冻结 reset/actor policy/Safety/controller 下的 interventional outcome，
不是现实世界个体反事实真值。NPC 对 ego 行为产生的反应可以在两个 branch 中不同；必须相同
的是初态、外生脚本和反应 policy，而不是强迫两条 future 时间线逐帧相同。

## 5. Reset 可比性

必须完全一致：

```text
map / route / weather / world settings
actor role set and blueprint
traffic-light script/phase
NPC control script
sensor configuration
candidate raw/canonical points
```

位置、yaw、speed 误差阈值由 C2 任务在采集前冻结；可继承已验证 H2 边界，但不能在看到
新结果后修改。reset 不可比时两个 branch 都保留为失败 Evidence，该 pair 不进入训练。

## 6. Outcome 向量

每个 branch 保存原始时间线，并由 offline-only labeler 计算：

```text
route_progress_m
route_completed
collision_count / collision severity / other actor id
red_light_violation
off_corridor_duration_s / max corridor deviation
minimum TTC / clearance where observable
mean and tail acceleration / jerk / lateral acceleration
controller deadline miss
safety decision / repair / MRM / would_require_cross-candidate-fallback
ticks executed / cleanup status
```

训练目标不把所有维度压成一个不透明 scalar。每个 head 有独立 target、mask、单位和 loss；
utility 只在离线 evaluator 中按预注册规则组合，硬风险不能被 progress 抵消。

`route_completed` 在 2.5s 短时域通常稀疏，必须同时报告其正例数；若几乎全为 0/1 常量，C3
不得把该 head 的低 loss 当能力证明，应降为审计字段或改用预注册的局部 goal/branch-terminal
定义。

## 7. Offline-only intervention curriculum

为避免 World 只看安全/正常数据，可对 capture candidate 构造物理受限干预：

- speed scaling；
- delayed braking；
- stop-line crossing；
- lateral corridor offset；
- curvature/yaw-rate increase；
- obstacle-envelope penetration；
- shortened stopping margin。

约束：

- provenance 必须写 `offline_intervention`、base candidate hash、operator 和 magnitude；
- 不冒充 nominal VLA/Expert，不进入 live candidate set 或正式 source usage；
- 先通过 finite/kinematic/canonical checks，并离线记录完整 Guard 结果；
- outcome 必须来自真实 CARLA branch，不用手写 outcome 伪造；
- base 与 interventions 按 root anchor 分组，不能跨 split；
- 不修改 formal scenario 或在看过 formal 结果后补 intervention。

只有 Guard `PASS/REVIEW` intervention 可进入 CORA 的核心 deployed-distribution outcome/
pairwise training。Guard `REJECT` intervention 只用于 auxiliary risk-foreseeing benchmark 或
明确隔离的预训练消融，不能混入 online router 的 pair coverage、calibration 或正式收益
claim。这样既能检查 optimistic bias，又不改变“World 在线不可见 REJECT”的权力边界。

该 curriculum 用于检验 World 是否能“诚实预测危险”，不是让危险轨迹绕过 Guard/Safety。

## 8. 数据划分

C2 必须在采集前冻结 mutually exclusive：

```text
train
validation
calibration
locked development test
pilot
formal
```

隔离键至少包括 root anchor、map、family、weather、route lineage、seed 和 intervention family。
calibration 不参与 checkpoint selection；formal 不参与任何训练、阈值、temperature、conformal
quantile、router parameter 或 failure diagnosis 后重跑。

已消费 seed 101 永远不进入新 formal。具体新 lineage 只能由 C2/C5 `START_TASK.md` 预注册。

统计单位是 root anchor。两个 branch、同 anchor 的多个 intervention 和时间序列 tick 都属于
同一 cluster；train/test 隔离、bootstrap 和有效样本量不能把它们当独立观测膨胀置信度。

## 9. 数据质量 gate

### Pair 完整性

- 两条候选各自有完整 50-tick outcome，或按同一冻结 early-terminal 判定规则合法结束；两个
  branch 的实际 terminal tick/reason 可以不同，这正是 outcome 的一部分；
- `pair_outcome_mask = 1` 才进入 pairwise loss；
- 不允许复制执行候选 outcome 给未执行候选；
- 不允许用 whole-policy majority 填补 tick outcome。

### 身份与泄漏

- pre/post binding trajectory hash 一致；
- selected/final/executed/applied ID 可解析；
- World feature schema 物理无 source/slot/order/future/outcome/winner；
- metadata-only probe 从实际 feature schema 构造并应确认字段不存在；
- trajectory-to-source probe 只从隔离审计副本计算，不把 source 回灌模型。

### 对称性

- branch order 交换后 label/reason 不变；
- slot/source metadata 交换不改变 outcome 绑定；
- intervention label 仍绑定实际 trajectory hash。

### 覆盖与尾部

必须按 map/family/weather/source winner/risk event 报告计数，不只给总样本数。collision、
red-light、offroad 等稀有 target 样本不足时，停止并重新设计 development curriculum；不能
用 class weight 隐藏零正样本。

还必须报告 candidate/branch 缺失机制：按 source、Guard 状态、risk family 和 branch order
统计生成失败、Safety early-terminal、cleanup 失败与 reset mismatch。只有完整 pair 进入
pairwise loss，但不能只展示 complete subset 而隐藏系统性 missingness。

## 10. 存储与 Evidence

建议延续不可变布局：

```text
generated/h6/cora/<dataset-id>/
  anchors/
  pairs/
  timelines/
  actor-future/
  events/
  interventions/
  labels/
  manifest.json

docs/runtime-evidence/h6/<dataset-id>/
  collection-summary.json
  data-quality.json
  final-delivery.json
```

每个 artifact 原子写入、内容哈希、同 ID 不同内容拒绝覆盖。正式 Evidence 绑定 HEAD、完整
dirty-worktree identity、config、matrix、model/checkpoint、CARLA、seed lineage 和资源。

## 11. C2 停止点

实际 C2 已在 2026-09-05 到达停止点：固定数据集
`h6-cora-c2-dev-20260830-v1` 完成 351/351 terminal roots、351/351 valid nominal pairs、1295 个
真实 branch outcomes 和 351 次 nominal VLA forwards。Pilot gate 通过；development gate 因
locked-development offroad 正例仅 1 个，以及 repair_success/executable 负类不足而失败。完整性、
29-head public labels、feature reproduction、inventory 与资源审计通过；formal 未采集，C3 未授权。
该结果冻结为 `DATA MEASURED / GATE_FAILED / STOPPED`。

C2 只完成 development paired data 与质量审计：

- 先小规模 smoke；
- pilot 通过才扩大到预注册 development matrix；
- 数据质量通过后冻结 manifest；
- 更新 `PROGRESS.md` 后停止；
- 不在同一任务自动训练 CORA 或查看 formal。

`240–360` root anchors 只是当前单机预算假设。C2 应先冻结 smoke、coverage pilot、development
三段上限和稀有事件下限；若预算内仍没有足够 Guard-eligible hazard outcomes，冻结数据不足的
负结论，不通过复制 intervention/tick 或放宽 Guard 来制造样本量。
