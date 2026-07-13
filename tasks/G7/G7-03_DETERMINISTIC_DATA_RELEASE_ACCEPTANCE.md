# G7-03：确定性 Data/Release 工作流与 Agent 红队验收

**状态**：PENDING
**依赖**：G7-01、G7-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C6、统一指标与 Evidence 规则；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10、11 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2、5～8 节；
4. G7-01～G7-02 最终产物、脚本对照、审计日志和失败清单；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

数据查询、去重、门禁、版本、release regression 和报告均由确定性流程执行；Agent 只给草案，schema/hash/统计/冻结阈值与审批作最终裁决。

## 实现范围与边界

- 关闭 LLM 后全流程仍能运行；
- 重复请求产生相同 manifest，冻结集不可写；
- 建议与最终裁决差异留档；
- 同预算比较脚本、模板和 Agent；

## 完成标准与验证

- 成功率、独立失败、诊断、证据、幻觉、时间/成本/人工干预完整。
- 越权、提示注入、伪证据、耗尽、中断和数据投毒红队通过。
- C6 可为负；无收益时 Agent 默认关闭。
- 确定性 workflow 和 Evidence 自检必须通过。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `agents/workflows、data_release_adapter、validation/g7、registry、artifacts/g7、tests/g7` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
