# G6-01：Safety Event、Hard Case Mining、谱系与数据门禁

**状态**：PENDING
**依赖**：G5-05

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 身份、时间、Oracle/Observable 与 Registry 章节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3、8、10、12、13 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2～4、6～8 节；
4. G5-05 发布模型、Safety/World 事件 schema、split 和失败 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

把安全/接管/分歧/World 误排/OOD/超时切为 normal/pre-event/intervention/recovery 窗口，并按风险、分歧、不确定、误差、覆盖和学习价值挖掘困难样本。

## 实现范围与边界

- 窗口边界、因果顺序、重叠优先级、provenance 和 hash；
- 失败簇/连续帧去重、场景平衡、正常样本配比和预算；
- 同步、完整性、泄漏、冲突、许可、分布和 Regression 隔离；
- World 只能提议，未经 CARLA/专家验证不能产安全偏好；

## 完成标准与验证

- 同一日志产生相同窗口 ID/manifest，拒绝理由稳定。
- 高价值与随机/周期采样同预算比较。
- Regression 写入、泄漏和不可比反事实被拒绝。
- 数据版本可重建且不覆盖旧版本。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `data_pipeline/events、lineage、hard_case_miner、gates、query、registry、tests/g6` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
