# H6 VLA 75% 主驾：现状、迭代方案与新对话交接

更新时间：2026-08-27

## 1. 先说结论

H6 还没有完成，但方向已经比较清楚。

三态 Guard、World v3、VLA 优先后由同拍 Expert 兜底、最终 Safety 重验、跟车镜头和正式
验收脚本都已经实现。正式 seed 101 pilot 失败后，可以确定：当前 VLA 占比低的主要原因
不是 Guard，也不是 Safety，而是 World 的逐时刻判断泛化很差。

用户本轮修订后的前向目标是：

```text
World 严格 VLA 高分比例       >= 90%
VLA 最终实际执行比例          >= 75%
Classic Expert + MRM          <= 25%
相对纯 Classic 不安全率增量   <= 1 个百分点
paired progress 95% 下界       >= 0
0 scorer deadline miss
切换率、ping-pong、provenance  全部通过
```

这里只下调“最终实际执行”门：从原来的 90% 改为 75%。用户此前要求的 World 90% 高分目标
没有被撤回，因此仍保留。这样可以区分两件事：World 是否真正看好 VLA，以及 VLA 在经过
Safety 后是否真的执行。不能靠路由强制锁定 VLA 来伪造 World 的判断。

历史 `h6-vla90-*` 数据和 2026-08-20 的正式失败仍按原 90% 合同解释，不得回写成“按新
75% 已经通过”。下一轮必须使用新 schema、配置 hash、数据集 id 和新的 held-out seed
lineage。

## 2. 当前代码与证据状态

当前项目总状态：

```text
H0 VERIFIED / STOPPED
H1 VERIFIED / STOPPED
H2 VERIFIED / GATE_PASSED / STOPPED
H3 VERIFIED / GATE_PASSED / STOPPED
H4 VERIFIED / GATE_PASSED / STOPPED
H5 VERIFIED / GATE_FAILED / STOPPED
H6 IMPLEMENTED / MEASURED / NOT_VERIFIED
```

已经完成的工程能力：

- Guard 是 `PASS / REVIEW / REJECT` 三态；`PASS` 和 `REVIEW` 都能进入 World；
- World v3 使用同一个 source-blind scorer 分别评价 Expert/VLA，并分开输出任务效果、
  进度、完成、碰撞、红灯、越界、舒适性、修复成功率、可信度与风险；
- World 只能排序或 defer，不能复活 Guard `REJECT`，也不能覆盖 Safety；
- World 首选 VLA 后，Safety 先尝试一次有边界修复；修复仍失败才在同一 tick 使用 Expert；
- 两个候选都失败才进入 MRM；
- 开发 seed 89/97 与正式 seed 101/103 隔离；
- 验收直接审计 World 原始双评分和最终实际执行来源；
- 真实 CARLA 运行使用独立 20Hz 跟车镜头。

2026-08-27 重新运行的当前工作树验证：

```text
436 tests: 435 passed, 1 skipped, 0 failed
compileall: passed
git diff --check: passed
```

跳过项需要真实 GPU checkpoint forward。当前任务环境的 readiness 显示 `cuda=false`；此前
GPU 重训练还被工作区以 `credits exhausted` 拒绝。旧 World v3 summary 也会被新版
readiness 以 `h6_policy_calibration_scope_missing` 正确拒绝。

当前工作树已按本轮计划切换到 `codex/h6-vla75-completion`，保留大量未提交修改和新增
文件；在 Git-ignored 的 `generated/h6/h6-vla75-worktree-baseline/` 留有 tracked binary
patch、任务相关 untracked 快照、状态和 SHA256 manifest。它是可测试的开发工作树，不是
已冻结的最终交付。v2 配置、矩阵、World 14-output scorer、逐 tick acceptance、实际
applied binding、repair final validation、训练器实际 event-aware temporal loss、固定
字典序 checkpoint 选择、锁定 router calibration、temporal router、run-lock 和 CLI 已实现；
collector/readiness 还会在正式运行前绑定 summary 的 checkpoint 顺序、逐文件 hash、ensemble
model hash 和 deployment/router/temperature calibration；没有
CUDA/CARLA 运行，因此仍不能把 H6 标成 `VERIFIED`。
正式编排器还会写入不可覆盖的 A/B/C lineage 状态账本：pilot 失败或 full 失败冻结该
lineage，full 只能在同一 run-lock 的 pilot 通过后启动；pilot 历史 hash 会随 full 状态保留，
三条 lineage 全失败时显式报告整体关闭。

## 3. 正式 600 tick 到底发生了什么

正式数据：

```text
dataset  h6-vla90-formal-pilot-20260820-v1
scope    12 paired scenarios / 24 runs / 600 on-arm ticks
seed     101（已消耗，不得再次冒充 held-out）
evidence docs/runtime-evidence/h6/h6-vla90-formal-pilot-20260820-v1/final-delivery.json
payload  8dae5c2e661abafc1dceab633d3338201a7fe1e6b50ecd5e334641aa68223194
```

### 3.1 从 47.5% 到 75% 差多少

```text
当前 VLA 实际执行       285 / 600 = 47.50%
75% 最低要求            450 / 600 = 75.00%
还差                    165 个 VLA tick

当前 World 选 VLA       303 / 600 = 50.50%
Safety 换成 Expert       18 / 303 = 5.94%（占全部 tick 的 3%）
当前实际 VLA            303 - 18 = 285
```

如果 Safety fallback 仍固定为 18 次，World 至少要把 VLA 排第一 `468/600 = 78%`，才能
得到 450 次实际 VLA。如果 fallback 比例随新增 VLA 选择保持约 5.94%，World 需要选择 VLA
约 479 次，也就是约 80%。所以工程上应把“World 路由 VLA”开发目标设在 80% 以上，并保留
90% 原始严格高分目标作为更高一层约束。

当前 World 只选择了 303 次，所以达到 75% 实际主驾至少要把 165 个当前 Expert tick 转成
安全可执行的 VLA tick。继续削 Guard 无法提供这些 tick。

### 3.2 Guard 不是瓶颈

正式 600 个 VLA 候选：

```text
PASS    453
REVIEW  147
REJECT    0
```

Expert 为 `PASS 588 / REVIEW 2 / REJECT 10`。VLA 没有一次被 Guard 硬杀。那 10 个 Expert
REJECT 来自 Town01 aggressive 最后 10 tick 的 navigation-start 距离异常，同时导致
正式 Evidence 出现 10 个 missing-pair provenance failure。

结论：Guard 已经足够弱。再删除 Guard 条件不会解决 165 tick 缺口，只会把坏数据或明显
不可执行轨迹送得更深。

### 3.3 Safety 不是主要瓶颈，但新增 VLA 选择仍必须重新验证

World 选择 VLA 的 303 个 tick 中，Safety 只在 18 个 tick 回到 Expert。18 次全部涉及
碰撞包络，7 次还伴随横向加速度，1 次伴随 offroad；修复失败集中在 RATO 不可行或最终
重验仍有碰撞、yaw-rate/teleport 问题。

18 次 fallback 分布在：

- Town01 aggressive：1 次；
- Town01 emergency：11 次；
- Town03 emergency：6 次。

这说明当前已被 World 选中的 VLA 大多能过 Safety。但不能简单推断另外 297 个未选 VLA
也有相同通过率；新 World 放开选择后，必须用 shadow final-validation 或真实开发闭环统计
新增区域的 Safety 通过率。

### 3.4 World 的失败不是单一阈值问题

正式严格 World 高分只有 `131/600 = 21.83%`。590 个双候选有完整评分的 tick 可拆成：

| 情况 | tick | 含义 |
|---|---:|---|
| VLA 分数不低、可信通过、风险通过 | 131 | 真正严格高分 |
| VLA 分数不低，但可信失败，风险通过 | 74 | 可能可用校准改善一部分 |
| VLA 分数低，但可信与风险通过 | 87 | 排名头判断错，不是调可信阈值能解决 |
| VLA 分数低、可信失败、风险通过 | 64 | 排名和可信同时有问题 |
| VLA 分数低、可信失败、风险也失败 | 234 | 最严重的泛化失败 |
| 缺 Expert/VLA 双评分 | 10 | provenance/Expert 候选问题 |

因此不能只把 `trust_threshold` 调低。即使把 74 个“只差可信”的 tick 全放过，严格高分也
只有 205/600，仍远低于 90%。至少 385 个已完整评分 tick 的综合分、可信或风险预测需要
真正改善。

### 3.5 地图和场景差异很大

实际 VLA 执行率按场景族：

| 场景族 | VLA tick | 比例 |
|---|---:|---:|
| aggressive cut-in | 51/150 | 34.00% |
| emergency lead brake | 91/150 | 60.67% |
| free-flow | 43/150 | 28.67% |
| red-light dilemma | 100/150 | 66.67% |

按地图：

| 地图 | VLA tick | 比例 |
|---|---:|---:|
| Town01 | 87/200 | 43.50% |
| Town03 | 131/200 | 65.50% |
| Town05 | 67/200 | 33.50% |

不能被 red-light 的 66.67% 迷惑：Town03/Town05 的两个红灯 VLA-on 运行都发生红灯违规。
当前 World/VLA 在“占比高的地方”不一定更安全。因此下一轮不能只优化平均覆盖率，必须按
`map × family × phase` 看最差组，并给红灯、紧急制动、急弯单独的安全门。

### 3.6 已通过与未通过的正式门

已通过：

- on/off 不安全运行均为 `3/12`，相对增量为 0；
- route progress mean `+2.455m`；bootstrap lower-95 `+0.629m`；
- scorer P99 `9.12ms`；
- 0 deadline miss；
- VLA Guard REJECT 0；
- 跟车镜头每条运行持续更新，错误 0。

未通过：

- World 严格 VLA 高分；
- VLA 实际执行；
- 31 次来源切换，约 `1.033 次/秒`；
- 5 个 ping-pong 场景；
- 10 个 missing pair provenance tick；
- Town03 free-flow 静态物体碰撞；
- Town03/Town05 红灯违规；
- 完整 108-pair 开发训练和正式 full 均未执行。

## 4. 为什么开发集看起来很好，正式却崩了

开发 seed 89/97 的 VLA 实际执行分别为 94.33% 和 90.67%，但那是开发态 VLA-primary
采集，用来获得 VLA outcome，不能证明 World 学会了选择。

旧校准又只拿每个场景第一拍的 12 个点算覆盖，得到 `11/12 = 91.67%`。正式测试看全部
600 tick 后只有 21.83%。这说明三个问题：

1. 校准单位错了：场景第一拍不能代表整段；
2. 闭环分布漂移：VLA/Expert 的选择会改变后续状态，独立同分布假设不成立；
3. 平均训练掩盖最差组：Town01/Town05、free-flow/aggressive 的表现明显更差。

校准单位已经在代码中改成每个 on-arm tick，但新训练还没有运行，所以现在只有“修好了
尺子”，没有“训练出能过新尺子的模型”。

## 5. 搜索到的方法，怎样落到本项目

以下来源只支持方法选择，不是 SafeDrive 已经验证的结果。

### 5.1 把覆盖率直接放进训练，不要训练后硬调阈值

[SelectiveNet（ICML 2019）](https://proceedings.mlr.press/v97/geifman19a.html)把“预测”和
“是否接管”一起训练，并用覆盖约束优化 risk-coverage，而不是只在已训练模型上切一个
置信阈值。本项目可以把它改造成 VLA/Classic 选择头：

```text
L = L_outcome
  + λ_rank * L_pairwise_rank
  + λ_coverage * max(0, 0.90 - mean(p_world_prefers_vla))^2
  + λ_safety * L_predicted_unsafe
  + λ_actual * max(0, 0.75 - mean(p_vla_executable))^2
```

覆盖约束不能单独使用：只有离线 paired outcome 显示 VLA 非劣且可执行的 tick 才能贡献
正向覆盖监督。否则模型只会学会“偏爱 VLA”，而不是学会“VLA 更好”。source 可以用于
离线绑定标签和计算 loss，但仍不得成为 World 在线输入。

### 5.2 把 Classic 兜底看成有预算的 learning-to-defer

[Consistent Estimators for Learning to Defer（ICML 2020）](https://proceedings.mlr.press/v119/mozannar20b.html)
把模型与下游 Expert 作为一个整体来优化，而不是只优化模型自身。更新的
[Probabilistic Learning to Defer（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/78df0f831fbe5854349dbdfccde7ee5d-Abstract-Conference.html)
还显式控制模型与 Expert 的工作量分配。

本项目对应关系：

```text
学习系统     = VLA + World
下游 Expert  = Classic Expert
defer 预算   = Classic + MRM <= 25%
```

建议新增一个 source-blind、pair-conditioned 的 defer/selection head，训练时同时最小化最终
系统损失和 Classic 使用预算。它不能获得 source id 作为输入，不能绕过 Guard/Safety，且
MRM 必须单独强惩罚，不能把 MRM 当普通 Classic defer。

### 5.3 用闭环到达的状态反复聚合数据

[DAgger（AISTATS 2011）](https://proceedings.mlr.press/v15/ross11a.html)指出序列决策中，模型
动作会改变后续输入分布；只在 Expert 到达的状态训练会产生累积误差。
[SafeDAgger](https://arxiv.org/abs/1605.06450)进一步用安全策略只查询重要/危险状态，减少
不必要的 Expert 查询。

对应到本项目，下一轮不应只把现有 24 pair 重训很多遍，而应进行开发态数据聚合：

1. 用当前 World 在 seed 89/97 开发矩阵闭环运行；
2. 记录低 margin、低 trust、高 risk、World/实际来源不一致和 Safety fallback 的 tick；
3. 对这些状态所在 episode 做 exact-reset Classic-off/VLA-primary-on 配对 rollout；
4. Oracle 仍只离线生成 outcome/排序标签；
5. 加入训练集后重训，再运行下一轮开发闭环。

禁止把 seed 101 正式失败直接混入训练后继续宣称同一 lineage 的盲测。它只能保留为失败
诊断 Evidence。

### 5.4 优化最差地图/场景组，不要只看平均 loss

[Group DRO（ICLR 2020）](https://arxiv.org/abs/1911.08731)针对平均表现很好、少数组很差的
问题，优化最差组损失；论文也强调必须配合正则化或 early stopping，否则过参数模型仍会
过拟合。

本项目建议用以下开发分组计算 group loss：

```text
map × family × phase
Town01/Town03/Town05
× free/emergency/aggressive/red
× approach/conflict/recovery
```

训练时优先提高最差组的 pairwise ranking、risk calibration 和 repair-success 预测；模型
选择看最差组验证结果，不看平均验证 loss。map/family/phase 只用于离线分组和 loss 权重，
不能作为在线“场景答案”输入 World。

### 5.5 增加时间上下文和一致性，但不能用锁定掩盖原始分数

[CarPlanner（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_CarPlanner_Consistent_Auto-regressive_Trajectory_Planning_for_Large-Scale_Reinforcement_Learning_in_CVPR_2025_paper.html)
通过跨时间保持一致的 mode 信息改善序列一致性。本项目不需要照搬其 RL 规划器，但可以
借用“跨 tick 一致性”的思想：

- World 输入增加最近 8–12 个可观测 tick 的紧凑历史；
- 训练 `Δscore = score_vla - score_expert` 的时间一致性；
- 仅在没有新危险事件时惩罚不必要的分数翻转；
- 新 actor、红灯变相、碰撞风险突变时必须允许立即切换；
- router 使用小幅 EMA/最短保持和紧急 break；
- 验收同时报告 raw World preference 和 routed selection，不能用 hysteresis 把低 raw score
  锁成 VLA 高分。

### 5.6 重新校准风险，但先修排名再校准

[On Calibration of Modern Neural Networks（ICML 2017）](https://proceedings.mlr.press/v70/guo17a.html)
说明神经网络置信度常常失准，temperature scaling 是实用的后处理基线。
[Conformal Risk Control（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
给出了用校准集控制单调损失期望风险的方法，并讨论了分布漂移扩展。

本项目的使用边界：

- 碰撞、红灯、越界、trust 分头校准，不共享一个温度；
- 按完整 episode 做 block/bootstrap，不把 600 个相关 tick 当 600 个独立样本；
- calibration split 只能用开发 seed；
- 可用 conformal 风格选择风险阈值，但 CARLA map/family shift 不满足简单交换性时，不能
  宣称无条件安全保证；
- 当前有 385 个 tick 不只是 trust 阈值问题，所以校准不能替代重新训练排名和风险头。

### 5.7 保留 Classic 与硬 Safety

[PDM / tuPlan Garage（CoRL 2023）](https://arxiv.org/abs/2306.07962)强调闭环评价和简单规则
先验的价值。[World4Drive（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.html)
则提供了用 world-model selector 评价多条规划轨迹的直接先例。

本项目应继续保持当前分工：VLA/Expert 产生候选，World 评价，Safety 最终硬重验，Classic
只做兜底。可以进一步加入轻量 future-latent/object-state auxiliary loss，帮助 World 学到
“候选执行后会发生什么”，但 RTX 4080 16GB 下先用冻结视觉/对象向量，不优先做像素视频
生成。

## 6. 推荐的下一版训练与路由设计

### 6.1 模型输出

保留现有 12 项 World v3 输出，再新增两个训练/路由量：

```text
pair_preference_logit  = VLA 相对 Expert 的综合优势
vla_executable_logit   = VLA 经过当前 Safety/一次修复后可实际执行的概率
```

两个量都由同一 source-blind candidate encoder 的双候选差分得到。不能简单加固定
`vla_bonus`；否则会在 swap/source masking 中暴露捷径。

### 6.2 训练损失

建议按优先级加入：

1. 现有 outcome/trust/risk 多任务损失；
2. 基于真实 paired policy outcome 的 VLA-vs-Expert ranking loss；
3. VLA 可执行/修复成功监督；
4. 90% World preference 约束，但只在 VLA outcome 非劣的开发样本上计算；
5. 75% 可执行覆盖约束；
6. map×family×phase 的 group-DRO/worst-group loss；
7. 有事件 mask 的时间一致性 loss；
8. 适度 L2、dropout、early stopping，checkpoint 由最差组验证门选出。

### 6.3 路由顺序

```text
Guard PASS/REVIEW
    -> World 原始双候选评分
    -> risk/trust 校准
    -> VLA/Expert preference + defer
    -> event-aware temporal stabilizer
    -> Safety final validate/repair
    -> VLA 或同拍 Expert 或 MRM
```

raw preference、stabilized preference、selected、Safety executed、applied 五个 id/来源都要
保存。正式 World 90% 只看 raw 双候选评分；实际 75% 只看 applied/executed source。

### 6.4 VLA 候选质量专项

World 提高 VLA 选择后，Safety fallback 很可能上升，所以同步做三项定点改进：

- emergency：让 VLA smoother 更早使用前车长度、相对速度和可停车距离做纵向限速；
- aggressive/急弯：降低高曲率区速度，减少 lat-accel/yaw-rate 和 RATO 最终重验失败；
- red-light：修复 Town03/Town05 停车线与 VLA 停车意图，不能把当前闯灯的 100 个 VLA tick
  当成覆盖率成功。

不删除最终碰撞、红灯、越界、状态机检查；修复结果仍须重新跑完整 Safety。

## 7. 验收代码需要先改什么

当前 `evaluate_vla90_gate` 用同一个 `target_vla_coverage=0.90` 同时检查 World 和实际执行。
下一对话应先做一个文档/合同可审计的 v2 改造：

```text
target_world_vla_preference = 0.90
target_actual_vla_coverage  = 0.75
max_classic_mrm_share       = 0.25
```

建议：

- 新 schema，例如 `safedrive.vla75.acceptance.v2`；
- 新 config hash，不修改旧 `H6_VLA90_CONFIG_SHA256`；
- readiness 分开读取 World 90% 和 actual 75%；
- 训练 episode outcome 归因纯度仍保持 90%；
- 新 dataset id 使用 `h6-vla75-*`；
- 回归覆盖 75% 边界、90% World 边界、Classic+MRM 预算和历史 v1 兼容；
- `START_TASK.md`、`PROJECT.md`、`WORLD_MODEL.md`、`ROADMAP.md` 保持一致。

## 8. 推荐执行顺序

### A. 不需要 CARLA/GPU 的合同改造

1. 新增 v2 acceptance/config/schema，不改旧 Evidence；
2. 把 World/actual 两个阈值拆开；
3. 增加按 map/family/phase、World gate reason、selected→executed fallback 的报告；
4. 增加 raw vs stabilized preference；
5. 全量单测、compileall、diff check。

### B. GPU 恢复后的离线训练

1. 只用 seed 89/97 开发数据；
2. 先重跑现有 tick-wise calibration，确认旧模型被拒绝；
3. 训练 coverage-aware + group-aware + temporal World；
4. 至少 3 个训练 seed；
5. checkpoint 选择看 worst-group、raw World 90%、actual 75% proxy、安全/进度，不看平均
   val loss；
6. 做 swap、source masking、action/history masking、校准和延迟测试。

### C. 开发 CARLA 数据聚合

1. 先跑 24-pair seed 89/97 pilot；
2. 所有真实运行强制跟车镜头；
3. 收集 World 低分、低 trust、高 risk、fallback、红灯/碰撞/急弯 hard cases；
4. 需要时进行一至两轮 DAgger-style 开发数据聚合；
5. 通过后采完整 108-pair 开发矩阵；
6. full readiness 必须先通过。

### D. 新正式 lineage

1. 用户明确授权并预注册新的 formal seed；
2. seed 101 不复用，103 不与已看过的 101 拼成旧 full；
3. 冻结模型、阈值、配置、代码/worktree hash；
4. 先跑 12-pair formal pilot；
5. pilot 任一硬门失败就停止；
6. pilot 全过后才跑 108-pair full；
7. full 结果无论正负都冻结。

## 9. 新正式 gate 清单

| 门 | 新要求 |
|---|---|
| World 原始 VLA 高分 | >=90%，分母为全部 decision tick |
| 双候选评分完整 | 缺 pair 计失败，不能缩分母 |
| VLA 实际执行 | >=75% |
| Classic + MRM | <=25% |
| Guard | REJECT 不得进入 World；不设人为 VLA REJECT 配额 |
| Safety | 相对 Classic unsafe delta <=1pp；不得有 target-only 新 unsafe |
| 进度 | paired bootstrap lower-95 >=0 |
| 资源 | scorer P99 <=50ms，deadline miss=0 |
| 稳定 | <=2 次/30s，ping-pong=0 |
| provenance | selected/executed/applied/repair/hash 全完整 |
| 隔离 | formal seed/checkpoint/threshold 不进入训练调参 |
| 镜头 | spectator follow 开启，updates>0，error=null |

此外必须分组报告绝对碰撞、红灯、越界。相对安全门通过不代表绝对安全好：当前 on/off
各 3/12 unsafe 就是例子。

## 10. 新对话第一轮建议直接这样开始

```text
请继续 H6 VLA 75% 主驾任务。先完整读取 AGENTS.md、START_TASK.md、PROGRESS.md、
docs/H6_VLA75_HANDOFF.md、docs/PROJECT.md、docs/WORLD_MODEL.md，并检查 git 分支和
status。不要复用 seed 101，不要进入 formal，不要削弱 Guard REJECT 或 Safety 硬检查。

第一步只做 acceptance/config/schema v2：保持 World 原始高分门 >=90%，把 VLA 实际执行
门改为 >=75%，Classic+MRM <=25%，保留历史 vla90 Evidence 不变；补齐边界测试、按组
诊断、raw/stabilized/selected/executed provenance。完成最小验证并更新 PROGRESS 后停止。
```

## 11. 关键本地入口

- 当前任务：[`../START_TASK.md`](../START_TASK.md)
- 动态事实：[`../PROGRESS.md`](../PROGRESS.md)
- 项目边界：[`PROJECT.md`](PROJECT.md)
- World 合同：[`WORLD_MODEL.md`](WORLD_MODEL.md)
- Guard/候选合同：[`HYBRID_CANDIDATES.md`](HYBRID_CANDIDATES.md)
- H6 Evidence：[`EVIDENCE.md`](EVIDENCE.md)
- 正式失败：[`runtime-evidence/h6/h6-vla90-formal-pilot-20260820-v1/final-delivery.json`](runtime-evidence/h6/h6-vla90-formal-pilot-20260820-v1/final-delivery.json)
- H6 acceptance：[`../safedrive_foundry/data_pipeline/h6/acceptance.py`](../safedrive_foundry/data_pipeline/h6/acceptance.py)
- H6 calibration：[`../safedrive_foundry/data_pipeline/h6/calibration.py`](../safedrive_foundry/data_pipeline/h6/calibration.py)
- World v3 训练：[`../scripts/train_world_v3.py`](../scripts/train_world_v3.py)
- readiness：[`../scripts/h6_readiness.py`](../scripts/h6_readiness.py)
- 真实采集与跟车：[`../scripts/h5_collect.py`](../scripts/h5_collect.py)

## 12. 研究来源

- [SelectiveNet: A Deep Neural Network with an Integrated Reject Option, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html)
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning / DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)
- [Query-Efficient Imitation Learning for End-to-End Autonomous Driving / SafeDAgger](https://arxiv.org/abs/1605.06450)
- [Distributionally Robust Neural Networks for Group Shifts, ICLR 2020](https://arxiv.org/abs/1911.08731)
- [Consistent Estimators for Learning to Defer to an Expert, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html)
- [Probabilistic Learning to Defer, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/78df0f831fbe5854349dbdfccde7ee5d-Abstract-Conference.html)
- [On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)
- [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
- [CarPlanner: Consistent Auto-regressive Trajectory Planning, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_CarPlanner_Consistent_Auto-regressive_Trajectory_Planning_for_Large-Scale_Reinforcement_Learning_in_CVPR_2025_paper.html)
- [Parting with Misconceptions about Learning-based Vehicle Motion Planning / PDM, CoRL 2023](https://arxiv.org/abs/2306.07962)
- [World4Drive, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.html)

这些论文支持迭代方法，不证明 SafeDrive 的任何数字。SafeDrive 的结论只以冻结 CARLA
Evidence 为准。
