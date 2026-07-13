# G5-05：VLA+World+Safety 闭环验收

**状态**：PENDING
**依赖**：G5-01～G5-04

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C1～C3 与统一指标；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～7、9～14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、5、7、8 节；
4. G5-01～G5-04 最终产物、模型卡、降级矩阵和失败 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

比较 VLA+Safety、VLA+运动学预测、VLA+Interactive World、VLA+World+Active CARLA，验证预测是否转化为候选排序和闭环净收益。

## 实现范围与边界

- 冻结数据、模型、候选、场景和协议；
- 同时评价预测、动作敏感、ranking/regret、校准/OOD 和闭环；
- 常规、交互长尾、错误筛选与模型失效分开报告；
- 模型卡与运行时默认开关绑定证据状态；

## 完成标准与验证

- 报告碰撞/TTC、完成、舒适、时延、显存和 CARLA 成本。
- 至少一个预登记 slice 验证稳定净收益，或如实给负结论。
- C2 可为负但 World 实现、消融、闭环和 Evidence 必须完成。
- 无收益时 World 保持 Shadow/离线，系统仍可 VLA+Safety 发布。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `G5 小型缺陷、validation/g5、registry、artifacts/g5、tests/g5、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
