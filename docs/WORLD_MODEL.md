# CORA Counterfactual Outcome World 与选择性路由合同

## 1. 准确定义

CORA World 是 candidate-conditioned trajectory outcome model，不是像素视频生成器、轨迹
generator、Safety 或 controller。它学习：

\[
p_\theta(Y_i\mid O_t,\tau_i)
\]

其中 `O_t` 是当前可观测 history/route/actors/lights，`τ_i` 是 Guard eligible 候选，`Y_i`
是执行该候选后的 short-horizon progress、risk、comfort 和 feasibility。

World 只对候选预测、排序并提供 uncertainty；calibrated router 可 choose、hold 或 defer；
Safety 保留最终批准、repair、fallback 和 MRM 权限。

预测 estimand 固定为：

\[
p_\theta\!\left(Y_i\mid O_t,\tau_i;\pi_{Safety},\pi_{control}\right)
\]

即 Guard eligible proposal `τ_i` 被交给冻结 single-candidate Safety/controller 后的 outcome。
label 必须保留 proposal→repaired/MRM executable→applied 关系；采集时禁止跨候选 fallback，
否则 `Y_i` 会依赖 pair。CORA 不预测绕过下游安全栈的“裸轨迹世界”。

## 2. 当前实现与结题目标

现有 H3/H4/H6 代码已经具备的 scaffolding 与 CORA 目标必须分开：

- 499-D vector observable context；
- `10×8` candidate tensor；
- shared candidate-conditioned Transformer/MLP scorer；
- 多 outcome heads；
- temperature/calibration、router、temporal state 和三 seed scaffolding；
- locked evaluation、readiness、run-lock 和 closed-loop collector。

但当前正式结论仍是 H5 gate failed、H6 not verified。CORA 在现有基础上重点修复标签、
对称性、loss、uncertainty 和 selector，而不是从头训练大型视觉生成模型。

| 能力 | 当前状态 |
|---|---|
| 499-D context、10×8 candidate、旧多头 scorer | 已实现并有 H3–H6 历史代码/Evidence |
| H6 v2 14-output、lineage/readiness scaffolding | `IMPLEMENTED / NOT_VERIFIED` |
| 正确 per-sample Group-DRO、真实 validation、统一 temporal state | `IMPLEMENTED / NOT_MEASURED / NOT_VERIFIED`（C1 工程） |
| 同 anchor 双 potential outcomes | C2 待采 |
| 本文 CORA 模型、joint calibration、formal 闭环结论 | `PLANNED` |

## 3. 输入与禁止项

允许：

- ego history、当前 ego state；
- route polyline、navigation command；
- 当前可观测 actors/lights/lane/topology；
- 当前/历史图像的冻结特征（若在阶段任务中预注册）；
- candidate trajectory 及其可重算 kinematics。

禁止：

- source、slot、branch order、Guard verdict、provenance reason；
- actor future、rollout event、真实 outcome、Oracle/winner；
- Regression/故障答案、scenario family answer、formal label；
- World 输出修改 Guard/Safety/repair/MRM/controller 权限。

feature object 应从允许字段新建，不从完整记录事后删除禁止字段。

source-blind 只承诺元数据不可见。轨迹几何可能暴露 planner 风格，所以要同时做两种不同
检查：metadata-only/source-swap 用于验证 schema；trajectory-to-source probe 用于量化行为
捷径风险。后者高于随机是诊断信号，不等价于数据泄漏，也不能被当作结果标签输入。

## 4. 模型结构

建议使用共享权重、候选交换等变结构：

\[
z_O=f_O(O_t),\qquad z_i=f_\tau(\tau_i)
\]

\[
h_i=\operatorname{CrossAttention}(z_i,z_O)
\]

\[
\hat Y_i=g_Y(h_i)
\]

每个 candidate 共享 `f_τ` 与 `g_Y`。pair difference 必须由结构保证反对称，例如：

\[
\widehat{\Delta U}=u_\theta(h_A,z_O)-u_\theta(h_B,z_O)
\]

或对任意 pair 网络显式反对称化：

\[
\widehat{\Delta U}=\tfrac12\left[f_\theta(h_A,h_B,z_O)-f_\theta(h_B,h_A,z_O)\right]
\]

交换 A/B 后：

```text
absolute outcome A/B 跟随 trajectory 交换
pair utility difference 变号
uncertainty 跟随对应 candidate
```

候选 identity 只用于把输出绑定回 trajectory，不作为 embedding。

仅把 `h_A-h_B` 输入普通 MLP、或只做 swap augmentation，都不能数学保证输出变号；它们只能
作为额外训练/测试，不能替代结构约束。

## 5. Outcome heads

最低输出：

| head | 类型 | 说明 |
|---|---|---|
| progress | mean + scale | 2.5s route progress |
| completion | Bernoulli | 预注册 local-goal/route completion；2.5s 内正例不足则只作审计 |
| collision | Bernoulli / severity | 碰撞与严重度 |
| red-light | Bernoulli | stop-line/red-light violation |
| offroad | Bernoulli + duration | corridor/drivable-area |
| comfort | mean + scale | acceleration/jerk/lateral acceleration |
| feasibility | Bernoulli | controller/Safety executability |
| repairability | Bernoulli | bounded repair 后通过 final validation |
| epistemic | ensemble disagreement | 分布外/模型不确定性 |
| pair difference | real/logit | 两候选 utility 差或 dominance |

progress/comfort 等连续量使用合适的 NLL/Huber；hazard 使用 BCE/focal/asymmetric loss；每个
head 的 mask、单位和有效计数独立。

## 6. Loss 合同

\[
\mathcal L=
\mathcal L_{outcome}
+\lambda_{pair}\mathcal L_{pair}
+\lambda_{swap}\mathcal L_{equiv}
+\lambda_{cons}\mathcal L_{consistency}
+\lambda_{tail}\mathcal L_{tail}
\]

- `L_outcome`：各候选真实 potential outcome；
- `L_pair`：同 anchor 两候选真实 utility/outcome difference；
- `L_equiv`：candidate swap 一致性；
- `L_consistency`：pair head 与 absolute outcome 组合方向一致；
- `L_tail`：collision/red-light/offroad hard cases，不允许零正样本伪通过。

pair utility 不直接覆盖各 outcome head。风险先做约束/支配关系，舒适和 progress 只在风险
可比的候选之间组合；utility 权重、归一化、风险阈值和 tie/defer margin 必须在对应阶段冻结。
这样不能用少量 progress 抵消碰撞或红灯风险。

### Per-sample 与 Group-DRO

正确顺序：

1. 每个 head 按自身 mask 计算 per-sample loss；
2. 只在有效 head 上按冻结权重合成 per-sample multi-task loss；
3. 按 map/family/weather/route/risk-event 分组；
4. Group-DRO 对真实 group loss 更新权重；
5. 报告每组有效计数、原始 loss 和权重。

禁止用一个 scalar objective 与异质输出向量做绝对差来构造 group loss。空 mask 必须是
`NOT_MEASURED`，不能当作 0 loss/pass。

C1 实现固定在 `safedrive_foundry/data_pipeline/h6/model.py`：objective、progress、completion、
collision、red-light、offroad、comfort、repair、trust、pair preference 和 executable 各自保留
unreduced loss 与独立 mask。candidate head 先在单个样本的有效候选内聚合，pair head 只在双
outcome 有效时启用；逐样本总 loss 是“有效 head 的冻结权重加权和 / 有效权重和”。无有效
head 的样本不进入优化，整 batch 无有效监督时 fail closed。

持久 `GroupDROState` 在训练开始前登记 train 数据中的 map/family/weather/group key，只以真实
target/mask 的逐样本多任务监督计算 group mean 并更新指数权重/floor。当前 batch 未出现的
group 保留历史权重，空 group 输出 `NOT_MEASURED/count=0/loss=null`；coverage、temporal 等
非监督 penalty 不构造 Group-DRO 风险。

## 7. 基线与因果检查

| 检查 | 要回答的问题 |
|---|---|
| no-action | history 是否已解释标签，模型没有使用 trajectory？ |
| candidate-only MLP | context 是否提供额外价值？ |
| CV/CTRV / hand reward | 复杂模型是否超过简单动力学/规则？ |
| frozen factual H6 World | 反事实监督是否修复旧标签问题？ |
| metadata-only/source swap | schema 是否真的不含 source/slot/order？ |
| trajectory-to-source probe | 合法轨迹风格能多大程度预测来源，模型是否只靠风格？ |
| candidate swap | 是否依赖 slot/绑定？ |
| source metadata swap | trajectory 不变时预测是否不变？ |
| action intervention | risk/outcome 是否随轨迹物理变化？ |
| context intervention | risk/outcome 是否随相关场景变化？ |
| history masking | 模型是否使用时序状态？ |

对 intervention 既报告分类准确率，也报告物理方向单调性，例如更晚制动不应降低 stop-line
risk，轨迹更接近障碍物不应降低 collision risk。

## 8. 不确定性与 calibration

区分：

- aleatoric：outcome 本身噪声，由 distribution head 表示；
- epistemic：训练覆盖不足，由独立 seed ensemble/模型分歧表示；
- calibration：在独立 calibration split 上估计 residual/temperature/quantile；
- selection：根据 outcome/utility bounds 选择或 defer。

示意：

\[
LCB(U_i)=w_p(\hat p_i-q_{p,i})-w_r(\hat r_i+q_{r,i})-w_c(\hat c_i+q_{c,i})
\]

只有：

\[
LCB(U_i)>UCB(U_j)+\delta
\]

才允许选择 `i`。否则 hold 或 defer。

风险比较先于软 utility：若某候选的 risk upper bound 超过冻结上限，它不能靠 progress LCB
进入占优；两个候选都不满足 learned risk 条件时直接 defer 给非学习 fallback/Safety。

conformal/coverage 的有效范围必须说明 calibration distribution 和 exchangeability 假设；
不能写成任意 OOD、任意闭环状态或实车环境的全域安全保证。

必须明确 coverage 单位。逐 outcome、逐 candidate 的 marginal coverage 不能直接宣传为“两
候选所有风险头同时覆盖”；系统级 claim 需要对 candidate/head 的 joint residual 或预注册的
multiple-comparison 校正，并按 root anchor 聚类评估。

## 9. Temporal selector

唯一状态机：

```text
raw outcomes/scores
→ calibrated utility intervals
→ dominance / ambiguity
→ minimum hold / hysteresis
→ choose / switch / defer
→ Safety
```

要求：

- temporal state 按 episode/route revision 作用域内的稳定 source key，而不是 frame candidate ID；
- emergency override、hysteresis 和 minimum hold 独立；
- raw、EMA、interval、selected source、defer/switch reason 每 tick 记录；
- offline calibrator 与 live router 调同一实现；
- replay trace 必须逐 tick 完全一致；
- router 不能通过 EMA/hold 提高 raw model coverage 指标。

`hold` 只延续 source 决策，当前 tick 仍使用 fresh candidate；`defer` 调用冻结非学习规则：
held source 仍 eligible 时优先，否则 Expert→VLA，最终都经过 Safety，均失败才 MRM。defer
不是在线 Oracle、人工接管或无控制。

C1 唯一实现位于 `safedrive_foundry/data_pipeline/h6/temporal.py`。状态只保存稳定
`expert`/`vla` source 与 source EMA，不保存 frame-scoped candidate ID；scope 由
run/episode identity 加 route revision 定义，变化时清空 EMA、hold 和历史。状态机固定依次执行
scope/eligibility、EMA、held unavailable/emergency risk、emergency margin、minimum hold、
hysteresis、普通 choose/switch，并输出统一 raw/EMA/margin/hold/switch/defer trace 与稳定 reason。
VLA75 offline calibration 和 live router 调用同一核心；旧 H5 historical 模式保持旧行为。

## 10. Checkpoint selection/readiness

checkpoint selection 只能使用实际 evaluator 输出：

```text
outcome NLL/Brier/ECE
unsafe recall/AUPRC
pair accuracy/regret
swap/source/action/context probes
worst-group metrics
measured P50/P95/P99
measured GPU peak
```

硬编码 `pass=True`、`swap_error=0`、`p99_ms=0`、`gpu_gib=0` 均无效。未运行就是
`NOT_MEASURED`，readiness 必须拒绝缺项。summary 绑定 dataset manifest、split、seed、
checkpoint ensemble、evaluator artifact、config、code/worktree 和自哈希。

C1 evaluator schema 为 `safedrive.world.vla75.evaluator.v1`，记录 checkpoint/seed、validation/
config/code/worktree/input lineage、per-head loss/count/hazard positives、pair accuracy/regret、
group loss/count/weight、candidate/source/action/context/history probes、实测 latency 和 incremental
GPU peak 状态，并使用排除自身字段计算的 `evaluator_sha256`。checkpoint metadata 同时绑定
validation lineage、selection metrics 与其 hash；最终 evaluator 必须重算并验证一致。

training summary schema 为 `safedrive.world.vla75.training_summary.v2`，固定绑定三个有序 seed、
checkpoint、evaluator 和输入 lineage，并使用 `summary_sha256`。readiness 识别但拒绝 v1 summary，
对 summary/evaluator/checkpoint/input/self hash、有效计数、probe、latency/GPU 状态和顺序
fail closed，输出 self-hashed readiness。CPU 或未执行 CUDA 时 GPU peak 必须是
`NOT_MEASURED/value=null`，因此不具备正式 readiness；artifact 验证状态和 CORA algorithm
验证状态始终分开。

C1 后新建 formal run-lock 固定为 `safedrive.h6.vla75.run_lock.v2`，calibration payload
必须绑定三个 evaluator hash、validation lineage 和 training input lineage；创建脚本在写盘前
调用验证器。`run_lock.v1` 仅保留历史 artifact 的只读兼容，不能作为 C1 后新建的 formal lock。

## 11. 评估

### 离线

- per-head NLL/Brier/ECE/AUROC/AUPRC；
- pairwise accuracy 和 regret；
- selective regret / risk-coverage / defer-rate；
- source/map/family/weather/risk-event worst group；
- swap/intervention/consistency；
- three-seed mean、spread 和方向；
- P50/P95/P99、deadline miss、GPU peak。

### 在线

使用相同 candidates、Guard、Safety、controller、scenario/reset 比较：

```text
Classic selection baseline（dual-generator shadow load）
factual World
CORA without abstention
full CORA
```

四臂都运行双 generator 以保持 selector 对照的 workload；Classic 臂将 VLA 置为 shadow 且
始终请求 eligible Expert。真正关闭 VLA 的 profile 另做资源对照，不能与效果臂混算。

主要 gate 见 [PROJECT](PROJECT.md) 与对应阶段 `START_TASK.md`。VLA/Classic/MRM usage、
repair/fallback transition 只做诊断，不是 source quota。

## 12. 资源边界

首版继续使用 object/vector context 或冻结视觉特征，模型规模服从 RTX 4080 16GB 上
CARLA + VLA + World 同时在线的预算。资源不足时优先减少 batch/history/feature 分辨率，
不改变 candidate、label、Safety 或 formal gate。

前沿方法定位与为什么不复制大型视频 World 见 [RELATED_WORK](RELATED_WORK.md)。
