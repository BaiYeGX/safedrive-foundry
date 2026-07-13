# G8-03：核心消融、统计、异质性与 Pareto

**状态**：PENDING
**依赖**：G8-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md 全文；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 10～14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 全文；
4. G8-01 冻结协议与 G8-02 完整运行矩阵；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

围绕 VLA、World、Safety、搜索、后训练和 Agent 的核心因果问题做配对消融，避免所有小组件笛卡尔积。

## 实现范围与边界

- VLA single/multi、language/Slow；World 无/简单/Interactive/Active；
- Safety Validator/QP/RATO/fallback；posttrain before/after/采样；
- Search Random/LHS/MAP-Elites；Agent script/template/model；
- 效应大小、bootstrap/区间、稀有失败、异质性和 Pareto；

## 完成标准与验证

- 聚合层级、seed、统计单位和多重比较处理正确。
- 表图与 run registry 一致。
- 负贡献、局部贡献和无结论显式标记。
- 不从次要 slice 反推全局主张。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

