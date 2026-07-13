# G3-02：非语言单轨迹与多候选基线

**状态**：PENDING
**依赖**：G3-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3～5、12 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2～4、7、8 节；
3. docs/project/CLAIMS.md C1 与统一指标；
4. G3-01 数据 schema、冻结 split、泄漏审计和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

用 Route/Ego MLP、时序视觉单轨迹、时序视觉多候选三个基线隔离多候选轨迹本身的价值，避免把轨迹头收益误归因于语言。

## 实现范围与边界

- 统一 Policy Adapter、数据/split、encoder 预算、轨迹格式和 seed；
- Classic-Observable 作为系统参照，不获取额外 Observable 信息；
- checkpoint 精确恢复和 model/data/config hash；
- 短闭环与 open-loop 同时评价；

## 完成标准与验证

- 报告 ADE/FDE/NLL、候选覆盖/多样性、可执行率、Safety 拒绝、闭环和资源。
- 单轨迹/多候选固定 encoder、预算和 seed 公平比较。
- 基础模型选择不能只依赖单一 open-loop 指标。
- 负结果和失败 slice 完整保留。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `driving_vla/baselines、training、adapter、config、tests/g3、validation/g3、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
