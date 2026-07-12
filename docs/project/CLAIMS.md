# 可证伪主张与证据矩阵

本文件只定义项目最终需要证明的研究主张。通用证据状态、完成规则和简历表述边界见根目录 `AGENTS.md`。

## 核心主张

| Claim | 要回答的问题 | 必须对照 | 主要证据阶段 |
|---|---|---|---|
| C0 | RACE 是否比公开基础专家在同安全/计算预算下改善规划效率、可跟踪性和扰动鲁棒性 | 固定 Frenet、基础 Hybrid A*、固定 MPC、各单项消融 | G1、G8 |
| C1 | 轻量 VLA 是否比非语言模仿基线产生更好的可执行候选 | Classic-Observable、视觉轨迹基线、无语言 VLA | G3、G8 |
| C2 | 世界模型是否改善候选排序并减少昂贵 CARLA 分支 | CV/CTRV/IDM、Reward MLP、无动作条件模型 | G5、G8 |
| C3 | RATO 是否优于 Hard Reject/Rule Slowdown 且不过度保守 | Raw、Validator、Hard Fallback、Longitudinal QP | G2、G8 |
| C4 | 主动搜索是否在同预算下发现更多独立可复现失败 | Manual、Random、LHS、CMA-ES | G4、G8 |
| C5 | 失败驱动后训练是否降低重复失败且保持旧能力 | 训练前模型、随机采样后训练 | G6、G8 |
| C6 | Agent 是否相对固定脚本产生可量化净收益 | 固定模板、人工脚本、无 Agent | G7、G8 |

## 统一指标维度

- 安全：碰撞、TTC、规则、接管和最小风险成功率；
- 效率：Route Completion、进度损失、无故停车；
- 舒适：加速度、横摆、jerk；
- 泛化：未见 Town、天气、布局、行为和故障；
- 实时性：P50/P95/P99、deadline miss、降级率；
- 资源：CPU、内存、GPU、显存、磁盘和 wall time；
- 可追溯：run/model/data/scenario/evidence ID。

## C0专项证据

- RACE-Plan：候选、节点和碰撞检查数，可行率、最小净空、曲率变化、P50/P95/P99及deadline miss；比较固定采样、基础启发式、单项与Full。
- RACE-Control：横纵向误差、约束违反、参数估计稳定性、收紧量、舒适性、P50/P95/P99、deadline miss和回退率；比较Fixed、Warm-start、Adaptive与Full。
- 所有比较固定场景、seed、硬件、输入轨道和预算；平均时延下降不能掩盖尾延迟、安全或可行率退化。
- 没有稳定净收益时，C0降级为“完成系统化复现与消融”，不得写成性能提升。
