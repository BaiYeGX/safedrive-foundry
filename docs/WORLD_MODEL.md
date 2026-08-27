# H World：候选条件化结果预测与排序

## 1. 定位

H World 是 selector，不是 generator、Safety 或 controller。它只接收逐候选 Guard 判为
`PASS` 或 `REVIEW` 的 Expert/VLA 轨迹，并对每条轨迹在当前 observable history 下的未来
结果打分；`REJECT` 永远不可见。

```text
z_t = Encoder(observable_history, route)
e_i = CandidateEncoder(trajectory_i)
y_i = OutcomeHead(z_t, e_i)
s_i = RankHead(y_i)
```

同一组参数应用于所有候选，排序必须对 candidate slot 置换等变。

## 2. 输入与禁止项

允许：

- 当前/历史图像或其冻结特征；
- ego history、route、导航命令；
- 当前可观测 actor、traffic-light、lane/topology；
- candidate trajectory 及由该 trajectory 本身可重算的动力学量。

禁止：

- rollout future、碰撞结果、Oracle winner；
- 场景 family 答案、Regression 注入、测试标签；
- Guard 结果、provenance、source、候选槽位或 branch order；
- World 输出修改 Guard/Safety 权限。

source id 只保留在 H2 审计和分层指标中，不能进入 H3 scorer feature view。source-only
baseline 只能从隔离的审计表计算，不能成为训练特征或模型变体。

## 3. 预测目标

H3/H4 的首版使用结构化 outcome，不做像素生成：

- collision/violation probability；
- route progress 与 completion；
- minimum clearance / TTC risk；
- comfort（accel、jerk、lateral accel）；
- infeasible/deadline risk；
- uncertainty 与 defer probability。

排序采用约束优先：硬风险不能被效率 reward 抵消。具体权重和阈值必须在 H3 开始前
冻结，并与手写 reward baseline 共用相同定义。

H5 正式闭环给出负结果后，H6 World v3 将“任务效果”和“这条候选是否值得信任”拆开，
共享模型对每条候选输出 12 项：

```text
综合任务效果
进度均值 + 方差
完成概率
碰撞 / 红灯 / 越界概率（分别预测）
jerk / 加速度 / 横向加速度
Safety 修复成功概率
可信概率
```

在线 `deployment_score` 综合任务效果、完成、可信和三类风险。VLA 只有在综合分不低于
Expert、可信度达到开发集冻结门槛、风险不超过冻结上限时，才成为主候选；“高可信”本身
不能给 VLA 人为加成到超过 Expert。

## 4. 训练样本

H2/H3 训练单位是同一 anchor 的 paired candidates：

```text
(observable_history, route, candidate_i, actual_outcome_i, pair_id)
```

split 按 root lineage/map/scenario family 分组，不能把同一 anchor 或近重复轨迹拆到
train/test。对称增强可交换 candidate slot，但不能交换 trajectory 与其 outcome 的绑定。

H6 重新训练使用真实闭环配对结果：同一物理场景的 `off` arm 只执行 Classic，开发态
`on` arm 用于采 VLA-primary 结果。训练样本仍只取 source-blind 的 observable/candidate
特征；source 只用于把实际 outcome 绑定回正确候选。训练 seed 固定为 `89/97`，正式验收
seed 固定为 `101/103`，loader 和 readiness 都拒绝把正式 seed 用于训练。某一来源必须
实际执行该 episode 至少 90% 才能贴该来源 outcome；readiness 还要求完整 108-pair 训练
矩阵，12-pair pilot 只能检查采集链。

## 5. 必做基线与因果检查

| 检查 | 失败含义 |
|---|---|
| no-action scorer | history 本身已解释标签，candidate conditioning 未证明 |
| candidate-only scorer | 模型可能忽略场景交互 |
| CV/CTRV rollout | 复杂 World 没有超过简单动力学 |
| frozen hand reward | 学习排序没有超过确定性规则 |
| source-only classifier | 数据存在来源捷径 |
| candidate swap / slot permutation | scorer 依赖槽位或绑定错误 |
| action masking | score 不随动作改变 |
| history masking | scorer 不使用场景状态 |

ACT-Bench 指出视觉质量不能替代 action fidelity，因此 candidate swap、masking 和
permutation 是进入 locked evaluation 的硬门，而非可选消融。

## 6. defer

World 在以下情况必须 defer：

- 仅一条候选通过 Guard；
- 输入/候选 provenance 不完整；
- uncertainty 超阈值或 score margin 不足；
- 推理超时或模型不可用。

defer 回到冻结的非学习 selector。World 失败不能跳过 Safety，也不能将已拒绝候选复活。

“两候选近重复就 defer”只保留给冻结的 H1/H3/H5 v1 行为。H6 World v3 仍给近重复候选
打分，因为用户目标包含“World 在 90% 决策时刻真正给 VLA 更高分”，不能用近重复门绕开
评分。只有一条 eligible 时仍不构成 World 的双候选高分证据。

## 7. 评估

离线：outcome calibration、pairwise accuracy、AUROC、NLL/Brier、regret、defer-risk、
两来源分层胜率、swap consistency、P50/P95/P99 延迟与显存。

在线：World on/off 使用相同 candidates、Guard、Safety、controller、seed 和场景；报告
碰撞/违规、完成/进度、舒适、回退、deadline miss 和资源。World 只有在安全不退化且
任务收益可复现时才算有效。

H6 的正式验收额外同时要求：

- 所有决策 tick 中至少 90% 同时存在 Expert/VLA 原始评分，且 VLA 综合分不低于 Expert、
  可信门和风险门都通过；只剩 VLA 一条候选不能计数；
- VLA 最终实际执行占全部决策 tick 至少 75%，Classic + MRM 合计不超过 25%，不能只看
  World 选择；
- 对纯 Classic paired baseline 的不安全率增量不超过 1 个百分点；
- paired route progress bootstrap 95% 下界不小于 0；
- scorer 无 deadline miss，切换率受限且无短窗 Expert↔VLA ping-pong，完整 provenance 可审计。

H6 当前是 `IMPLEMENTED / MEASURED / NOT_VERIFIED`：已完成 24-pair 平衡开发训练 pilot、
旧第一拍口径的 World v3 训练和一次 seed 101 正式 pilot。正式逐 tick World 高分只有
21.83%、VLA 实际执行 47.50%，已经失败；新的 tick-wise 重训练、完整 108-pair 开发矩阵、
新 held-out pilot/full 尚未完成。

2026-08-27 的前向验收口径只下调“实际执行”硬门；World 原始双候选高分门仍为 90%。
训练 episode 的 90% outcome 归因纯度门也保持不变，它用于保证标签可信，不能和 75%
正式执行目标混为一谈。历史 `vla90` 运行继续按原合同解释。

## 8. 单机预算

首版优先 object/vector 或冻结视觉特征，参数量与历史长度以 RTX 4080 16GB 在线并行
CARLA+VLA 可运行为约束。先跑小样本过拟合与 action-sensitivity smoke，再扩大训练。
资源不足时减少视觉分辨率、history 或 batch，不改变候选合同和验证门。

## 9. 外部依据

- [World4Drive](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.html)：
  用 latent world representation 评价并选择 trajectory modality。
- [ACT-Bench](https://arxiv.org/abs/2412.05337)：单独评估 action controllability/fidelity。
- [PDM](https://arxiv.org/abs/2306.07962)：规则闭环 proposal 与学习模块分工，并强调
  闭环评价。

这些是设计依据，不是本仓库的实测证据。
