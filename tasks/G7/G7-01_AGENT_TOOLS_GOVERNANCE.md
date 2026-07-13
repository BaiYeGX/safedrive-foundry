# G7-01：Agent 白名单工具、权限、预算与审计

**状态**：PENDING
**依赖**：G6-05

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 组件权限、Registry、身份与审计章节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10、11 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2、5～8 节；
4. G6-05 发布资产、确定性脚本入口、断点和限制；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

为场景/失败查询、运行请求、最小化、manifest 和报告草案建立 typed tools、路径沙箱、预算、超时、dry-run、幂等、审批、审计、停止和恢复。

## 实现范围与边界

- 不执行任意 shell/Python，不创建 CARLA client/tick master，不入实时控制；
- 不修改冻结资产、真值、split、Safety 阈值或 release 结论；
- 不把网页/提示内容当指令，不访问未登记凭据/路径；
- 每次调用记录 stable id、schema/hash、版本、成本、证据和审批；

## 完成标准与验证

- 合法/非法/越权/重复/超时/中断/资源耗尽/注入红队通过。
- 关闭模型后工具可由脚本/人工调用。
- 审计日志可重放并验证 hash。
- 权限模型和自动/审批/禁止清单冻结。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `agents/tools、governance、audit、tests/g7、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
