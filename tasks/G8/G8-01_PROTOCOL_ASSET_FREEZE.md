# G8-01：核心主张、最终协议与资产冻结

**状态**：PENDING
**依赖**：G7-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md 全文；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 10～14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 全文；
4. G2-05、G3-05、G4-05、G5-05、G6-05、G7-03 的发布资产、限制和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

冻结 C1～C6、指标/护栏/阈值/对照/统计/缺失规则和代码/环境/模型/数据/地图/场景/seed/config hash，隔离训练、调参和最终测试。

## 实现范围与边界

- 四配置：Classic、VLA+Safety、VLA+World+Safety、PostTrained-Full；
- Agent/Active CARLA/二级 RATO/Slow 是消融开关；
- World/Agent 负结果对应 shadow/offline/disabled 配置；
- 单机预算和串行加载策略预演；

## 完成标准与验证

- 每项主张有唯一对照、主指标、护栏、统计单位和停止规则。
- 控制/Safety 不依赖 GPU 可用性。
- 干净工作区重建 smoke 并核对全部 hash。
- 最终标签对训练/调参不可见。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `validation/g8/protocol、registry/evidence、artifacts/g8/freeze、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
