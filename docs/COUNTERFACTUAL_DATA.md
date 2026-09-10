# 训练与反事实数据合同

## 本周只准备两种视图

状态：C3 修复版数据适配 `VERIFIED`（2026-09-10）；C4 World 视图仍 `PLANNED / NOT_RUN`。
C3 用经过当前位置投影和语义审计的可信专家轨迹微调 VLA；C4 用已有真实四维后果联合学习。
权威 C3 manifest 位于
`generated/h6/cora/c3-repair-20260910T100651Z/manifest.json`。初版 route 前缀监督和 90.77%
结果已撤回，详见 [C3 SFT 勘误](runtime-evidence/h6/c3-sft-repair-erratum.md)。
数据先用 C2 release 的 train 158 / validation 53，不重开稀有事件覆盖任务。

| 视图 | 输入 | 标签 |
|---|---|---|
| SFT | 当前图像、导航、ego/history | 可信 expert 原生 route/speed |
| World | 相同当前观察 + 原始 candidate | progress、acceleration RMS、jerk RMS、lateral acceleration RMS |

每个目标独立 mask；World 不要求本周新增逐时刻控制/未来视频/actor sequence。
SFT teacher 不等于 branch outcome；不能复制 Expert 后果给 VLA。
失败分支可用于真实后果监督，但不能自动成为正确驾驶示范。
配对损失只用同 root 两个真实有效分支；缺一个 outcome 不得补造。

修复版 route 监督来自有序 native expert reference path 的当前位置投影后前向片段，按 1 m
采样 20 点；speed 监督独立来自 10 点、0.25 s 的 canonical expert proposal。导航只作为
部署输入或明确诊断来源，不能替代专家标签；未知、提前终止、碰撞和时间未对齐区间保留
原因并逐头 mask，不用零填充或导航拼接。

## C3 一次完成数据适配

核对图像可读取、时间与坐标、原生驾驶 target 时域、可信 teacher 来源和物理 root 隔离。
canonical 保持 T=10、dt=0.25 s。当前 input 与未来 label 物理分离；
source/slot/order/provenance/Guard verdict 不进入 World。
数据不足先检查已有 timeline；必要时只做一次正常专家补采，见 [RESOURCES](RESOURCES.md)。
补采前登记新 train/dev roots、采集清单和预算；不能用原 VLA 自标冒充专家。

旧 release 的 calibration 52、locked_development 52、coverage_pilot 25 仍仅审计，
11 个重复 root 继续隔离。不能用改名、相邻 frame 或同 root 分支扩大独立样本量。
第一天结束仍没有有效 SFT 监督则报告阻塞，不把“四个 World 标签存在”当作 SFT 数据可用。

## C5 一次开发闭环

新预登记 6 个 roots，每 root 三臂；正常行驶/候选分歧各 3 roots。
recipe、seed、初始化、cadence、终止与排除规则先冻结，不能看结果后挑场景。
不用 seed 101、reserved 173/179 或旧保留集。全部实际失败与资源记录保留。
这是新开发评估，不替代原正式矩阵；本周无新 calibration、无安全非劣证明。

下文 C2 原始合同与失败记录仅解释历史数据。原 split、阈值、outcome 身份规则不变；
旧“不能进入 C3”等描述对应当时阶段，新的执行范围以本文及 ROADMAP 为准。

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

## 1B. C2 repair v3 收尾合同（2026-09-06）

`h6-cora-c2-repair-20260906-v3` 继续引用同一 base 数据，新增 delta 不覆盖 v1/v2 Evidence。采集
器已改为读取冻结 screening recipe，严格保存 source/operator/multiplier；不适用不再静默替换。
诊断与 batch 使用独立 manifest/run-lock，root target 按 split-local 序号分配，物理去重排除 ID、
seed 和时间戳，预算账本跨启动、采集和恢复累计。实际 Safety trace 写入 repair heads，未尝试
修复保持 success 缺测。

本版 Town03 实际生成 12 个诊断 root、34 条执行 branch。诊断得到 1 个独立
`repair_attempted=true, repair_success=false` root 和 5 个 offroad root；要求分别为 2 和 1，
因此 batch-1 collector 在读取 v3 labels 后直接阻断。原 351 root/1295 branch 保留，合并报告为
363 root/1329 branch；全量回归 492 tests run、1 skipped、OK。CARLA 已关闭，预算账本记录 118.4057
秒的已记录工作时间；本版仍为 `DATA MEASURED / GATE_FAILED / STOPPED`，不能进入 C3。

## 1C. C2 dev baseline 现有数据 release（2026-09-07）

本次调整不新增 CARLA branch，也不把 repair v2/v3 的诊断或 intervention 当作训练样本。新
`h6-cora-c2-devbaseline-20260907-v1` 通过 `c2_dev_baseline_v1` profile 读取 base pair index
和 v3 corrected-label sidecar，保留原 root/branch 身份；旧文件、旧数据和旧模型 Hash 不重扫。

审计粒度仍是 physical root/capture cluster。351 个原 root 中，按规范化路线、ego 初态、NPC/灯控、
天气和 capture 条件检查出 11 个重复：跨 split 的重复簇整簇隔离，同 split 的重复按 root 名称排序
保留一个。最终 340 个 root 的 split 为 coverage_pilot 25、train 158、validation 53、
calibration 52、locked_development 52。训练只使用 train 的 nominal Expert/VLA 两候选，开发
评估只使用 validation；其他 split 继续保留作审计，不能被 loader 当作 held-out training 输入。

29 个 public outcome heads 仍独立保存 value、unit、valid mask 和 derivation version。baseline 只
学习 `route_progress_m`、`acceleration_rms_mps2`、`jerk_rms_mps3`、
`lateral_acceleration_rms_mps2` 四个连续 head；collision、red-light、offroad、executable 和
repair 只报告真实分布和缺测。未知 repair trace 不补成负类，单个辅助 head 缺测不删除其他有效
监督。新 loader 必须同时看到 `DEV_DATA_READY`、正确 profile、training/evaluation split 和
完整 release index，否则拒绝读取。

这个 release 的用途是可复现的短时 outcome baseline，不再声称本轮已完成完整 counterfactual
coverage gate。后续若要重新采集或改变候选生成器，必须创建新的数据合同，并重新检查物理去重、
reset、proposal/execution 绑定和 World 对新版候选的适用性。

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

已消费 seed 101 永远不进入新 formal。具体新 lineage 由对应 C3/C4/C5 的 `START_TASK.md`
在采集前预注册；C2 原 lineage 与保留用途不变。

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
red-light、offroad 等稀有 target 在 C2 原质量门下样本不足时，停止对应覆盖任务并冻结缺口；不能
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
29-head public labels、feature reproduction、inventory 与资源审计通过；formal 未采集；C3 当时未授权，当前后续路线见本文第 0 节。
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


## 残差辅助的数据口径

C4 固定 b 与残差 R 使用同一允许 train 子集，b 只读 canonical candidate；
标签归一化与 b 系数不拟合 validation。不能因某头在 validation 上更好而临时换基线。
World mask 缺失不删除任一策略的 SFT 样本；teacher 质量按真实字段/时间/坐标与行为规则判定，
不按 M0/M1/M2 误差筛选。无 SFT 的 World-only roots 本周不额外采样。
root 指标与固定比较规则见 PROJECT，旧 split/labels/阈值与饱和排序报告保持不变。

## 量化分母与标签合同

遵守 [PROJECT 论文量化合同](PROJECT.md#论文量化合同c3_c6_metrics_v1planned)。manifest
记录 root、样本、候选、原生有效点、四头有效 mask 和各 split 计数；root/lineage 交集必须为 0。
缺 label 与推理失败分开存储；不能通过预测是否成功决定真值 mask。
同锚点 pair 的差分指标必须两边真值都有效，单边有效只进入对应逐候选指标。
每头 ridge/normalizer 仅使用有效 train label；val 不参与拟合、标准化或目标幅度选择。
保存原始单位、dt、路线投影、信号差分/滤波版本；新 root 等权与旧 C2 候选聚合口径分别命名。

## C3 原生监督审计细节

本地上游 route label 是弧长 0…19 m 的空间采样，speed-waypoint 是时间采样；
分别保存有效空间长度/点 mask 与时间戳/dt，不能以 canonical 2.5 s 轨迹长度冒充全部原生真值。
原生 head 为增量后 cumsum 的累计位置预测，label 坐标应与最终累计输出一致。
原始记录缺帧时不得复制上一帧后标有效；短路线不得把 padding 当真实未来监督。
专家当前规划路线与执行后位置序列分开标 teacher 类型；名义导航折线不能自动充当避障 expert。
标签所需的隐藏状态仍只离线使用，训练输入严格复用部署的导航/速度/图像约定。
记录 resolved speed 及 startup assist/override 配置，不能训练喂真实速度、部署却喂修改速度而不说明。
训练根按无放回轮转、同根 anchor 轮换；有效集合先冻结，所有分布诊断不改变验证分母。
