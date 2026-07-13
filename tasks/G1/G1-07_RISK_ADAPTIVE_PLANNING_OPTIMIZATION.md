# G1-07：风险自适应规划搜索优化（RACE-Plan）

**状态**：`COMPLETED_WITH_LIMITS`
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

---

## 验收结果与证据映射（追加，不替代上文标准）

| 完成标准条款 | 结果 | 权威证据 / 测试 |
|---|---|---|
| Baseline / 单项 / Full 消融 | **通过（offline）** | `tests.g1.test_g1_07_race_plan`；`docs/architecture/evidence/g1-07/repair-20260713/summary.json` schema `safedrive.g1_07.ablation.repair.v2` |
| 候选数 / 尾延迟诚实 | **通过** | `candidates_raw` / `nodes_raw`；已删除 `*0.7`/`*0.85` 事后缩放 |
| 风险单调性 / Oracle 分轨 | **通过（属性）** | `risk_monotonicity_ok`；`evaluate_risk_field` oracle/observable 分轨 |
| 无稳定收益不进默认 | **通过（负结论）** | `default_admission.promote_full_to_default=false`，`recommended_default=p1`，`work_ratio_full_over_basic≈1.24` |
| coarse_to_fine / multi_heuristic 等 | **有限制** | flags 标为 `experimental_partial_*`（独立重跑/预算/stats）；**非**完整算法交付 |
| CARLA 动态 Pareto 全场景 | **未 VERIFIED** | 本任务以 offline 消融为准；live 见 G1-09 限制清单 |
| 基线 hash 保护 | **通过** | evidence 记录 frenet/hybrid config SHA-256 |

### 验证与断点

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-13 |
| 最后状态 | `COMPLETED_WITH_LIMITS` |
| 验证 | `PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_07_race_plan -v`（2/2 PASS） |
| 权威证据 | `docs/architecture/evidence/g1-07/repair-20260713/`（含 `manifest.json`） |
| 旧证据 | 根目录 `summary.json` 可为 pointer；以 `repair-20260713` 为准；见 `SUPERSEDED.md` |
| 限制 | full 无净收益不 promote；部分 RACE flags 为 experimental/partial；无 CARLA A/B 收益证明 |
| 停止 | **任务关闭**；阶段已 `COMPLETED_WITH_LIMITS`（见 `PROGRESS.md` / G1-09）；不自动 G2 |
