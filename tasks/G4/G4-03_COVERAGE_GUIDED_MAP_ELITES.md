# G4-03：覆盖引导 MAP-Elites 主动失败搜索

**状态**：PENDING
**依赖**：G4-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、6～8 节；
3. docs/project/CLAIMS.md C4 与统一指标；
4. G4-01 参数域/门禁和 G4-02 同预算执行器、基线结果、checkpoint 与断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

只实现一个面向不同失败覆盖的 MAP-Elites/archive，不同时维护 CMA-ES 与多套 QD；风险严重度用于 cell 内质量，行为描述符用于覆盖。

## 实现范围与边界

- 预登记 descriptor、bin、cell、quality 和 novelty；
- 由合法 Random/LHS 初始化，mutation 后重新过有效性/可解性；
- 重复失败、INVALID 和仿真异常不更新 archive；
- checkpoint、预算、seed、重复和 provenance 可恢复；

## 完成标准与验证

- 与 Random/LHS 同 rollout 预算公平比较。
- 报告独立失败、覆盖、严重度、INVALID、复现和成本。
- 未优于基线时保留负结果并关闭默认采样。
- archive 空洞和代表性 cell 可解释/重放。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `scenario_search/qd、archive、config/search、tests/g4、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
