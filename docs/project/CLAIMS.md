# 可证伪主张与证据矩阵

本文件只定义项目最终需要证明的研究主张。通用证据状态、完成规则和简历表述边界见根目录 `AGENTS.md`。  
作品级成功口径（稳定可跑、VLA+World 必接入、效果可负、不上实车）见 `docs/project/PROJECT_SUCCESS_PROFILE.md`。

**重要：** 主张可以为负或“无稳定净收益”，但不得用负结论作为“删除 World/VLA 模块”的借口；模块接入与科学增益分开记录。

## 核心主张

| Claim | 要回答的问题 | 必须对照 | 主要证据阶段 |
|---|---|---|---|
| C0 | RACE 是否比公开基础专家在同安全/计算预算下改善规划效率、可跟踪性和扰动鲁棒性 | 固定 Frenet、基础 Hybrid A*、固定 MPC、各单项消融 | G1、G8 |
| C1 | VLA-V0/V1是否能在单机产生完整K1/K2轨迹并完成无Classic当前帧候选闭环 | Classic-Observable、上游anchor、V0 K1、V1 K2 | G3、G8 |
| C2 | 在oracle best-of-K证明存在选择空间后，World-V0是否改善VLA候选排序 | VLA top-1、oracle best-of-K、CV/CTRV、Reward MLP、无动作条件、无World | G4A、条件式G5、G8 |
| C3 | 两级 Safety 修复是否优于 Hard Reject/Rule Slowdown 且不过度停车 | Raw、Validator、Hard Fallback、Longitudinal QP、受限 RATO-SCP | G2、G8 |
| C4 | G4A固定困难场景和可比K2分支能否复现并量化top-1与oracle best-of-K差距 | 不可比分支、候选同步失败、不同seed replay | G4A、G8；主动搜索C4B optional |
| C5 | 一轮困难样本监督adapter/LoRA是否改善预登记切片且保持正常能力 | 同一冻结集训练前模型 | G6、G8 |
| C6 | Optional Agent是否相对确定性脚本提高研发效率且不破坏治理 | 固定模板、确定性脚本、无Agent | Optional G7；不属于发布门禁 |

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
3. `VLA+World+Safety`：仅World门禁通过后在线；否则标Shadow/SKIPPED/negative；
4. `PostTrained VLA+World+Safety`：加入一轮监督后训练；World未通过时对应`PostTrained VLA+Safety`。

Hybrid单独报告工程稳健性。G4B、Agent、Active CARLA和单个算法增强均为optional，不新增不可维护的发布栈。

## C0专项证据

- RACE-Plan：候选、节点和碰撞检查数，可行率、最小净空、曲率变化、P50/P95/P99及deadline miss；比较固定采样、基础启发式、单项与Full。
- RACE-Control：横纵向误差、约束违反、参数估计稳定性、收紧量、舒适性、P50/P95/P99、deadline miss和回退率；比较Fixed、Warm-start、Adaptive与Full。
- 所有比较固定场景、seed、硬件、输入轨道和预算；平均时延下降不能掩盖尾延迟、安全或可行率退化。
- 没有稳定净收益时，C0降级为“完成系统化复现与消融”，不得写成性能提升。
