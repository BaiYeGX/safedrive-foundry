# G3-01：场景身份、VLA 数据契约、冻结划分与泄漏审计

**状态**：PENDING
**依赖**：G2-05

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 第 2～8 节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3、5、8、12 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1～4、6～8 节；
4. G2-05 发布接口、SafetyEvent/执行结果 schema、最终断点和 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

在训练前统一 run/frame/scenario 身份、策略输入、特权标签和评测隔离，并前移最小 scenario_id/family/parameter_hash，避免后续场景注册返工。

## 实现范围与边界

- 前视图像、Ego 历史、Route、语言、Classic 候选、Safety/执行结果对齐；
- policy_input、privileged_label、evaluation_only、regression_frozen 四层；
- Parquet/DuckDB 分片、hash、断点续采、容量和重放；
- 按 Town/Route/场景族/天气/失败簇冻结 ID/OOD/Regression；

## 完成标准与验证

- 固定 seed 数据摘要/hash 一致，中断不重复分片。
- 故意泄漏、错帧、错误 grounding、近重复和回归集写入被拒绝。
- 语言标签限定为 behavior/critical_actor/conflict/risk_horizon/intended_action。
- 数据卡记录来源、许可、Oracle 字段、失败比例和容量。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `data_pipeline/vla、collector、schema、最小 scenario/schema、registry、config、tests/g3、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
