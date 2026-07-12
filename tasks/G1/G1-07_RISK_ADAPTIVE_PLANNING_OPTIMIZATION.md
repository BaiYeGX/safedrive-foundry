# G1-07：风险自适应规划搜索优化（RACE-Plan）

**状态**：PENDING  
**依赖**：G1-04、G1-05

## 目标与为什么现在做

在已经可运行的 Frenet/ST 与 Hybrid A*–Reeds–Shepp 基线上，建立统一的可解释风险场，并针对候选爆炸、无效扩展、动态不确定性和规划—控制脱节进行算法优化。该任务是经典专家的主要个人算法贡献之一，不把公开算法复现冒充创新。

## 输入、范围与交付物

- 统一 `RiskField`：几何距离、TTC/THW、动态占用概率、预测协方差、道路/规则裕度和可跟踪性；G1 使用 CV/CTRV/IDM 等可解释预测，G5 可通过同一接口接入世界模型但不得改变基线定义。
- Frenet：风险驱动采样预算、低/中/高风险档、粗到细 Top-K 精化、候选缓存与稳定性滞回；保留固定均匀采样基线。
- Hybrid A*：无碰撞 2D 启发式 + 非完整 Reeds–Shepp 启发式、解析扩展门控、状态支配/重复抑制、可跟踪性与净空代价、deadline 下可审计部分解；保留基础启发式版本。
- 输出逐候选风险分项、采样/扩展预算、拒绝原因、最优性或次优性说明、P50/P95/P99 和 deadline miss。

## 明确不做与允许修改

不训练风险网络，不使用 CARLA 隐藏未来，不把世界模型作为本任务依赖，不修改控制器或 Safety Kernel。允许修改 `safedrive_foundry/classic_stack/risk/**`、`safedrive_foundry/classic_stack/planning/frenet/**`、`safedrive_foundry/classic_stack/planning/hybrid_astar/**`、`safedrive_foundry/classic_stack/geometry/**`、`safedrive_foundry/config/planning/**`、`tests/g1/**`、`safedrive_foundry/validation/g1/**`、`safedrive_foundry/scenario/**`、文档、本任务和 `PROGRESS.md`。

## 完成标准

- 在相同场景、seed、硬件、deadline 与安全约束下完成 Baseline、单项改进和 Full RACE-Plan 消融。
- Frenet Full 相比固定采样不得以明显降低可行率/安全裕度换取时延；报告候选数、碰撞检查数与尾延迟变化，不预设必须提升的百分比。
- Hybrid Full 报告节点扩展、重开节点、解析扩展命中、路径质量、最小净空、曲率变化和超时部分解有效率。
- 风险场仅使用 Observable 输入的闭环结果与 Oracle 上界分开；预测不确定性增大时安全裕度变化具有单调性属性测试。
- 所有未带来稳定收益的改进必须保留负结果，不得合并进默认配置。

## 验证方法

运行离线地图属性测试、固定动态场景多 seed、同预算 A/B、风险单调性、采样稳定性和 20/50Hz Profile benchmark；对跟车切入、遮挡横穿、动态绕行、道路阻断、狭窄掉头和脱困分别报告安全—效率—舒适—计算 Pareto。

## 资源与断点记录

纯 CPU/少量 CARLA 验证，不训练模型。中断时保存 config hash、场景/seed、Baseline 与消融完成矩阵、最近失败候选和 profiler 输出。

| 字段 | 内容 |
|---|---|
| 最后状态 | PENDING |
| 已完成/修改文件/验证 | 无 |
| 阻塞 | 无 |
| 恢复步骤 | 核对 G1-04/G1-05 基线冻结版本，从缺失消融单元继续 |
| 下一条建议命令 | 待执行时填写 |
