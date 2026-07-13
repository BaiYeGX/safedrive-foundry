# 可证伪主张与证据矩阵

本文件只定义项目最终需要证明的研究主张。通用证据状态、完成规则和简历表述边界见根目录 `AGENTS.md`。

## 核心主张

| Claim | 要回答的问题 | 必须对照 | 主要证据阶段 |
|---|---|---|---|
| C0 | RACE 是否比公开基础专家在同安全/计算预算下改善规划效率、可跟踪性和扰动鲁棒性 | 固定 Frenet、基础 Hybrid A*、固定 MPC、各单项消融 | G1、G8 |
| C1 | 轻量 Fast/Slow VLA 是否比非语言多候选基线产生更可执行、可校准的候选 | Classic-Observable、单轨迹视觉基线、非语言多候选、无 Slow 分支 | G3、G8 |
| C2 | 动作条件结构化世界模型是否改善 VLA 候选排序并减少昂贵 CARLA 分支 | CV/CTRV/IDM、Reward MLP、无动作条件、无 World | G5、G8 |
| C3 | 两级 Safety 修复是否优于 Hard Reject/Rule Slowdown 且不过度停车 | Raw、Validator、Hard Fallback、Longitudinal QP、受限 RATO-SCP | G2、G8 |
| C4 | 覆盖引导搜索是否在同预算下发现更多独立可复现失败 | Random、LHS、无 archive 搜索 | G4、G8 |
| C5 | CARLA 验证的失败驱动后训练是否降低重复失败且保持旧能力 | 训练前模型、随机/周期采样后训练 | G6、G8 |
| C6 | 受控研发助手是否相对确定性脚本提高场景提议或诊断效率且不破坏治理 | 固定模板、确定性脚本、无 Agent | G7、G8 |

## 统一指标维度

- 安全：碰撞、TTC、规则、接管和最小风险成功率；
- 效率：Route Completion、进度损失、无故停车；
- 舒适：加速度、横摆、jerk；
- 泛化：未见 Town、天气、布局、行为和故障；
- 实时性：P50/P95/P99、deadline miss、降级率；
- 资源：CPU、内存、GPU、显存、磁盘和 wall time；
- 可追溯：run/model/data/scenario/evidence ID。

## 核心发布配置

最终矩阵只冻结四个可解释配置，避免模块笛卡尔积：

1. `Classic`：经典专家与控制；
2. `VLA+Safety`：轻量 VLA、多候选轨迹与独立 Safety；
3. `VLA+World+Safety`：加入动作条件世界模型排序；
4. `PostTrained-Full`：加入一轮失败驱动后训练。

Agent、Active CARLA 和单个算法增强以消融开关评价，不新增不可维护的发布栈。

## C0专项证据

- RACE-Plan：候选、节点和碰撞检查数，可行率、最小净空、曲率变化、P50/P95/P99及deadline miss；比较固定采样、基础启发式、单项与Full。
- RACE-Control：横纵向误差、约束违反、参数估计稳定性、收紧量、舒适性、P50/P95/P99、deadline miss和回退率；比较Fixed、Warm-start、Adaptive与Full。
- 所有比较固定场景、seed、硬件、输入轨道和预算；平均时延下降不能掩盖尾延迟、安全或可行率退化。
- 没有稳定净收益时，C0降级为“完成系统化复现与消融”，不得写成性能提升。
