# G4-02：Random/LHS 同预算搜索基线与可恢复执行器

**状态**：PENDING
**依赖**：G4-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、6～8 节；
3. docs/project/CLAIMS.md C4 与统一指标；
4. G4-01 场景 schema、合法性/可解性门禁、suite manifest 和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

在相同参数域、rollout 预算、seed、有效性和终止规则下实现透明的 Random/LHS 基线；人工场景只作为 Engineered Suite。

## 实现范围与边界

- 可暂停恢复队列、幂等 run_id、预算和重试分类；
- 风险分项不压成不可解释单分数；
- INVALID/仿真异常不污染搜索但计入成本；
- 统一首次失败、独立失败、覆盖、严重度、复现率和资源；

## 完成标准与验证

- 固定 seed 重放和断点恢复不重复 rollout。
- 样本数、有效数和预算核对一致。
- 失败簇/连续帧不虚增独立失败。
- 冻结 baseline manifest、原始 run 和负结果。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `scenario_search/baselines、checkpoint、executor、config/search、tests/g4、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
