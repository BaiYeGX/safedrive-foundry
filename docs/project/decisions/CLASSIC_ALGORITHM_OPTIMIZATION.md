# 经典专家算法优化选型（RACE）

## 1. 结论

经典专家不对每个公开算法平均“加料”，而围绕一个统一问题形成 **RACE（Risk-Adaptive Classic Expert）**：在动态不确定环境和单机实时预算下，让规划搜索更有效、轨迹更可跟踪、控制在模型失配和 deadline 下可降级，并为 VLA/世界模型提供可信教师与安全回退。

入选的三项核心算法贡献：

1. **RACE-Plan**：统一 `RiskField` + Frenet 风险自适应粗到细采样 + Hybrid A* 多启发式/支配剪枝/解析扩展门控。
2. **RACE-Control**：有界在线参数辨识 + 误差包络约束收紧 + deadline-aware warm-start MPC。
3. **RATO-SCP**：在 G2 对学习/经典候选做最小干预安全修复；它是独立安全创新，不塞入 G1。

未入选为核心的方向：神经启发式 Hybrid A*、纯 RL 局部规划、神经残差动力学、像素风险地图、完整 Branch MPC。它们会增加训练/OOD/验证负担，与 G3/G5 重复，或难以在 RTX 4080 单机条件下建立更清晰的净收益证据。

## 2. 研究与工业依据

- Autoware规划架构将行为路径、可行驶区域、障碍预测与优化轨迹分层，并明确动态障碍的时间维度和预测稳定性仍是困难点：<https://autowarefoundation.github.io/autoware-documentation/main/design/autoware-architecture-v1/components/planning/>。
- Autoware Obstacle Avoidance Planner同时强调运动学可行、道路边界、计算时间、鲁棒性和可诊断调参：<https://autowarefoundation.github.io/autoware.universe_planning/pr-5562/planning/obstacle_avoidance_planner/>。
- Feedback Enhanced Lattice Planner和Adaptive Sampling Control支持优先优化采样维度、响应式风险和粗到细预算：<https://arxiv.org/abs/2007.05794>、<https://arxiv.org/abs/2307.00482>。
- Stanford Junior Hybrid A*与Multi-Heuristic搜索工作支持多启发式、解析扩展和状态支配，而非首先训练不可审计的神经启发式：<https://ai.stanford.edu/~ddolgov/dolgov08gppSTAIR.html>、<https://arxiv.org/abs/2307.07857>。
- 风险Branch MPC和不确定性感知预测—规划支持将预测不确定性送入风险场和约束收紧，但G1不承担重型分支优化：<https://arxiv.org/abs/2403.18695>、<https://arxiv.org/abs/2403.02297>。
- Waymo Field of Safe Motion支持评估安全逃逸、可恢复性与净空，而不只检查碰撞：<https://waymo.com/research/field-of-safe-motion/>。
- Tube/contraction安全规划支持在模型误差下使用安全裕度；本项目采用可完成的有界辨识和误差包络，不声称完整形式化证明：<https://research.google/pubs/safe-motion-planning-with-tubes-and-contraction-metrics/>。
- Autoware MPC显式建模一阶转向延迟，OSQP官方MPC示例支持warm start，支持优先估计少量执行器参数和设计deadline回退：<https://autowarefoundation.github.io/autoware_universe/latest/control/autoware_mpc_lateral_controller/>、<https://osqp.org/docs/examples/mpc.html>。

## 3. 统一接口

```text
Observable Actor/Map/Ego histories
              ↓
CV/CTRV/IDM prediction + uncertainty envelope
              ↓
RiskField(t, s/d or x/y)
├─ collision probability / TTC / THW
├─ clearance and road/rule margin
├─ prediction uncertainty
└─ trackability / recoverability
              ↓
Frenet budget | Hybrid heuristic/cost | MPC tightening/weights | RATO constraints
```

G1风险场必须由可解释基线预测独立成立。G5世界模型只能通过版本化adapter提供额外预测，不允许覆盖G1基线证据。

## 4. 公平消融与准入

| RACE-Plan版本 | 自适应采样 | 粗到细 | 多启发式/剪枝 | 风险/可跟踪性 |
|---|---:|---:|---:|---:|
| Basic | 否 | 否 | 基础 | 几何 |
| P1 | 是 | 否 | 基础 | 几何 |
| P2 | 是 | 是 | 基础 | 风险 |
| Full | 是 | 是 | 是 | 风险+可跟踪性 |

| RACE-Control版本 | Warm start | 在线辨识 | 约束收紧 | Deadline回退 |
|---|---:|---:|---:|---:|
| Fixed | 可选基线 | 否 | 否 | 基础超时 |
| C1 | 是 | 否 | 否 | 是 |
| C2 | 是 | 是 | 否 | 是 |
| Full | 是 | 是 | 是 | 是 |

改进进入默认配置必须满足：固定协议、多seed配对场景下产生稳定净收益；不以安全、进度或可行率换取平均时延；报告尾延迟和deadline miss；Oracle与Observable分开；负结果和适用区间完整保留。

## 5. 简历边界

完成并验证前只能写“复现/集成”。只有对应Claim达到`VERIFIED`才能写“设计风险自适应采样”“改进Hybrid A*搜索效率”“构建鲁棒自适应MPC”。不得把统一集成描述成全新理论算法，也不得预填性能提升百分比。
