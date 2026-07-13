# G8-05：最终审计、限制登记与版本准出

**状态**：PENDING
**依赖**：G8-04

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md 全文；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 全文；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 全文；
4. G8-01～G8-04 最终产物、限制和失败门禁；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

输出可复现、可演示、可答辩且不夸大的发布候选，冻结四配置、资源预算、已知限制、发布清单和项目叙事。

## 实现范围与边界

- 需求→任务→测试→Evidence 追踪；
- VLA、World、Safety 的贡献与边界分别解释；
- CARLA 结果不宣称真实道路安全证明；
- 只修复发布阻塞，不扩大实验或移动阈值；

## 完成标准与验证

- G0～G8 条件满足，C1～C6 有证据或降级/负结论。
- 最终清单、回归索引、链接/hash 和从零抽检通过。
- 单机复现、演示、失败回放和限制完整。
- 不自动 commit/push/merge main，完成后停止。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

