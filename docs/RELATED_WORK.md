# CORA-Drive 前沿技术定位

更新日期：2026-08-29。本文只使用论文、官方项目页或官方代码作为活动技术依据。外部结果
用于定义问题和对比方法，不是本仓库的 Evidence，也不能替代本项目冻结实验。

## 1. Vision-Language-Action 驾驶

### SimLingo

[SimLingo（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Renz_SimLingo_Vision-Only_Closed-Loop_Autonomous_Driving_with_Language-Action_Alignment_CVPR_2025_paper.pdf)
同时研究 closed-loop driving、vision-language understanding 和 language-action alignment。
其关键观点是：模型能在语言中识别红灯/障碍，不代表动作真的与理解一致；Action Dreaming
等任务用于加强语言与 path/speed 行为的对齐。

本项目复用其预训练 nominal VLA 作为真实候选来源，但不把语言解释或 VLA confidence 当作
安全证书。CORA 检查的是轨迹条件后果与实际执行。

### Alpamayo-R1（2025 预印本）

[NVIDIA Alpamayo-R1](https://research.nvidia.com/labs/avg/publication/wang.luo.etal.arxiv2025/)
使用 Chain of Causation 数据、reasoning VLM 和 diffusion trajectory decoder，把因果推理与
轨迹规划对齐，并报告 reasoning-action consistency。

CORA 不复制其大规模模型/RL 路线；可借鉴的评估思想是：解释必须通过 action/outcome
干预验证，不能只检查文本是否通顺。

### Latent-CoT-Drive

[Latent Chain-of-Thought World Modeling（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Latent_Chain-of-Thought_World_Modeling_for_End-to-End_Autonomous_Driving_CVPR_2026_paper.html)
把 action-proposal token 与表达候选未来后果的 world-model token 交错在 latent reasoning
空间，并使用 future rollout supervision 与 closed-loop RL。它强化了“reasoning 必须和 action
outcome 对齐”的趋势。

CORA 不训练 latent CoT 或 RL policy；它把 proposal generation、outcome prediction、拒绝和
Safety 权力拆开，以便在单机系统里做可证伪归因。

## 2. 驾驶 World Model 的三种含义

### 2.1 像素/视频生成式世界模型

[GAIA-1](https://wayve.ai/wp-content/uploads/2024/04/2309.17080.pdf) 以 video、text、action
生成未来驾驶视频，目标接近 learned simulator、场景生成与表征学习。其规模和数据远超
单张 RTX 4080 可从头复制的范围。

[Drive-WM（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Driving_into_the_Future_Multiview_Visual_Forecasting_and_Planning_with_CVPR_2024_paper.html)
生成 action/trajectory-conditioned multiview future，并探索用 image-based reward 评价多种
驾驶 future。

本项目不把“能生成逼真视频”等价为“action-conditioned 后果忠实”，也不把像素生成作为
结题目标。

### 2.2 BEV/latent dynamics model

[WoTE（ICCV 2025）](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Li_End-to-End_Driving_with_Online_Trajectory_Evaluation_via_BEV_World_Model_ICCV_2025_paper.pdf)
让 planner 产生多条轨迹，BEV World 为各候选预测 imagined future，reward model 再选择。
它说明候选条件 future evaluation 可以比只看当前状态更适合 trajectory selection。

[World4Drive（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.pdf)
在 latent space 中预测 intention-conditioned future states 并选择多模态轨迹。

[ProDrive（2026 预印本）](https://arxiv.org/abs/2604.25329) 强调 planner candidate 与
BEV future 的 ego-environment co-evolution；[DriveWorld-VLA（2026 预印本）](https://arxiv.org/abs/2602.06521)
和 [DriveLaW（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_DriveLaW_Unifying_Planning_and_Video_Generation_in_a_Latent_Driving_CVPR_2026_paper.html)
进一步研究 world latent 与 VLA/planner 的统一。

这些近期工作说明“候选动作必须影响预测 future”是主流方向，但它们不自动证明任意 World
在 closed loop 有用。

### 2.3 方法谱系对照

| 路线 | 典型输出 | action-conditioned | 主要用途 | CORA 取舍 |
|---|---|---:|---|---|
| GAIA-1 / Drive-WM | 视频/多视角 future | 是 | 生成、仿真、image reward | 不从头复制，资源和忠实度验证成本过高 |
| WoTE / World4Drive / ProDrive | BEV/latent future | 是 | 多候选生成与评价 | 借鉴 candidate-conditioned evaluation |
| DriveLaW / DriveWorld-VLA / LDrive | unified latent world-action | 是 | planning 与 imagination 联训 | 不联训 VLA，保留归因和 Safety 分离 |
| CORA | 结构化 outcome distribution | 是 | 异构候选排序、校准拒绝 | 实时、可审计，但不生成完整 future state |

### 2.4 Outcome/value world model

CORA 属于轻量 outcome model：不重建像素/BEV，而直接预测
`p(outcome | observation, trajectory)`。优势是实时、结构化、可校准、适合异构 planner 和
独立 Safety；限制是不能声称学习了完整视觉世界生成规律。

简历和报告使用：

```text
candidate-conditioned trajectory outcome model
counterfactual outcome World
trajectory consequence scorer
```

不使用“9B generative world simulator”等不符合实现的描述。

## 3. 反事实与闭环分布漂移

[Model-Based Policy Adaptation](https://arxiv.org/abs/2511.21584) 将 end-to-end driving 的
closed-loop 下降归因于 observation/objective mismatch，使用 counterfactual trajectories、
policy adapter 和 multi-step Q model 适配闭环目标。

[AD-R1](https://arxiv.org/abs/2511.20325) 指出标准 World 可能存在 optimistic bias：对危险
action 幻想安全 future。其 Counterfactual Synthesis 主动加入 collision/offroad 等危险结果，
训练更“诚实”的 critic。

CORA 与它们的共同点：

- action-conditioned outcome；
- counterfactual/hard-risk data；
- 不以普通安全日志中的相关性替代因果动作差异；
- 最终用 closed-loop 而不是 open-loop imitation metric 判定。

CORA 的差异：

- 两个在线候选来自异构 VLA/Classic，而不是同一 policy 的采样；
- 使用 CARLA 同锚点 exact-reset 获得冻结 simulator/policy 下的双分支 interventional outcomes；
- World 在线 schema 保持 metadata-source-blind，并额外审计轨迹风格捷径；
- 独立 Guard/Safety 不参与 learned critic 权力扩张；
- 单机 4080 上优先结构化 outcome，而不是大规模 occupancy/video generation 或 RL。

## 4. 选择性决策与安全边界

[SafePath](https://arxiv.org/abs/2505.09427) 将 conformal prediction 用于候选路径集合和委托，
展示 uncertainty-aware selection 与 autonomy/safety coverage 的权衡。

CORA 借鉴的是独立 calibration 和 abstention 思想：候选 utility bounds 未明确分离时 defer。
必须同时声明：conformal coverage 依赖 calibration 分布与 exchangeability 假设，不等于实车
全域安全保证；Safety Kernel 仍需独立验证最终轨迹。

SafePath 的 candidate generator、LLM formulation 和高不确定性时的人类 delegation 与本项目
不同。CORA 的 defer 是软件在环中的冻结非学习 fallback/Safety，不得把 SafePath 的理论主张
直接移植为 CORA 的系统级安全保证。

## 5. 本项目的合理创新主张

公开工作已经分别存在多候选 planning、World future evaluation、counterfactual synthesis、
selective prediction 和 safety shield。因此在没有系统文献检索前，不能声称单个概念世界首创。

本项目可以验证并主张的组合贡献是：

1. 同一 CARLA anchor 上对异构 VLA/Classic 候选执行 exact-reset 双分支 potential outcomes；
2. metadata-source-blind、candidate-swap-equivariant 的结构化后果模型；
3. risk/outcome calibrated choose/hold/defer，而不是固定 VLA source quota；
4. learned World 与 Guard/Safety/MPC 权力分离；
5. factual World、counterfactual World、no-abstention 和 full CORA 的冻结多臂 closed-loop
   可证伪对照。

最接近的公开思路已经覆盖了单个组成部分，因此真正需要 Evidence 支撑的 novelty 不是
“第一次使用 VLA/World/conformal”，而是下面这条组合链是否在当前约束下成立：

```text
heterogeneous independent proposals
→ same-anchor simulator interventions
→ metadata-source-blind outcome estimation
→ calibrated non-learning handback
→ authority-separated closed-loop evaluation
```

只有本项目 Evidence 通过后，才能声称这些组合在当前 CARLA matrix 上降低 regret 或改善
闭环效用；否则应报告为负结果和瓶颈诊断。

## 6. 面试解释边界

| 可以说 | 不能说 |
|---|---|
| 集成预训练 SimLingo VLA | 从头训练了 SimLingo/VLA foundation model |
| 构建 candidate-conditioned outcome World | 构建 GAIA 类生成式视频 World |
| 用 exact-reset 获得双候选 potential outcomes | 在实车上获得因果安全保证 |
| 使用 calibration/abstention 管理统计不确定性 | conformal 已证明任意 OOD 全域安全 |
| 独立 Safety Kernel 提供 fail-closed 边界 | 达到量产功能安全认证 |
| 完成 CARLA/ROS 2 同步 SIL | 完成全车量产 ROS 2 自动驾驶栈 |
