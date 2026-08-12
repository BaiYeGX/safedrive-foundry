# H World：候选条件化结果预测与排序

## 1. 定位

H World 是 selector，不是 generator、Safety 或 controller。它只接收已经通过逐候选
Guard 的 Expert/VLA 轨迹，并对每条轨迹在当前 observable history 下的未来结果打分。

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
- candidate trajectory、动力学量与 Guard 结果摘要。

禁止：

- rollout future、碰撞结果、Oracle winner；
- 场景 family 答案、Regression 注入、测试标签；
- 候选槽位作为语义；
- World 输出修改 Guard/Safety 权限。

source id 可用于审计和分层指标，但默认不能进入主 scorer。若实验使用 source embedding，
必须单独做 source-only baseline 并证明模型不是在背候选来源先验。

## 3. 预测目标

首版使用结构化 outcome，不做像素生成：

- collision/violation probability；
- route progress 与 completion；
- minimum clearance / TTC risk；
- comfort（accel、jerk、lateral accel）；
- infeasible/deadline risk；
- uncertainty 与 defer probability。

排序采用约束优先：硬风险不能被效率 reward 抵消。具体权重和阈值必须在 H3 开始前
冻结，并与手写 reward baseline 共用相同定义。

## 4. 训练样本

训练单位是同一 anchor 的 paired candidates：

```text
(observable_history, route, candidate_i, actual_outcome_i, pair_id)
```

split 按 root lineage/map/scenario family 分组，不能把同一 anchor 或近重复轨迹拆到
train/test。对称增强可交换 candidate slot，但不能交换 trajectory 与其 outcome 的绑定。

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
- 两候选近重复，无实际选择空间；
- uncertainty 超阈值或 score margin 不足；
- 推理超时或模型不可用。

defer 回到冻结的非学习 selector。World 失败不能跳过 Safety，也不能将已拒绝候选复活。

## 7. 评估

离线：outcome calibration、pairwise accuracy、AUROC、NLL/Brier、regret、defer-risk、
两来源分层胜率、swap consistency、P50/P95/P99 延迟与显存。

在线：World on/off 使用相同 candidates、Guard、Safety、controller、seed 和场景；报告
碰撞/违规、完成/进度、舒适、回退、deadline miss 和资源。World 只有在安全不退化且
任务收益可复现时才算有效。

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
