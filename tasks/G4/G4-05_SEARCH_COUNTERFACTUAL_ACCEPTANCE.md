# G4-05：搜索、反事实与复现性阶段验收

**状态**：PENDING
**依赖**：G4-01～G4-04

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C4 与统一指标；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md G4 与资源字段；
4. G4-01～G4-04 最终产物、断点和失败清单；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

冻结协议比较 Random、LHS 和 MAP-Elites，验收有效场景、独立失败、可比反事实、最小反例和资源成本。

## 实现范围与边界

- 固定 rollout 预算、seed、参数域、有效性和终止规则；
- 报告首次失败、独立失败、覆盖、严重度、INVALID 和复现率；
- 稀有事件区间、代表性失败人工复核和 wall-time/资源；
- 搜索结果进入 Counterexample/Regression Registry；

## 完成标准与验证

- MAP-Elites 无收益可关闭，但 Random/LHS、重建和最小化必须通过。
- C4 对照、统计单位和负结果完整。
- Evidence hash/link/schema 与原始 run 一致。
- 完成后停止，不自动开始 G5。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

