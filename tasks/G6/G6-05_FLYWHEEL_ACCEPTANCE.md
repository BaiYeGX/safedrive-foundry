# G6-05：失败驱动数据飞轮阶段验收

**状态**：PENDING
**依赖**：G6-01～G6-04

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C5、核心发布配置与统一指标；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10、12～13 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md G6 与资源字段；
4. G6-01～G6-04 最终产物、A/B 矩阵和缺失 run；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

验收一轮发现失败→窗口/门禁→DAgger/Preference→后训练→抗遗忘回归是否产生可信净收益或可解释负结论。

## 实现范围与边界

- 冻结 target failure、正常/OOD/实时性护栏和统计规则；
- 比较 hard-case 与随机/周期采样的数据效率；
- 报告 CARLA、训练、存储和人工成本；
- 确认 Regression/Oracle 泄漏和 World-only 标签均被阻断；

## 完成标准与验证

- 完整数据/模型/场景/Evidence 谱系可重建。
- 目标改善不以正常能力、OOD、实时性或停车退化换取。
- C5 可为负但飞轮全过程必须真实完成。
- Evidence 自检通过且不自动开始 G7。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

