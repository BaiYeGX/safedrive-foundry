# G7-02：Scenario/Failure Research Assistant

**状态**：PENDING
**依赖**：G7-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10、11 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2、5～8 节；
3. G7-01 typed tools、权限/预算 schema、审计和红队结果；
4. G4 场景 schema、G5 World 证据、G6 失败簇的最终只读接口；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

用一个受控助手根据覆盖空洞、World 不确定和失败簇提出可证伪场景，并对齐 Runtime/VLA/World/Safety/规划/控制/CARLA 证据生成根因候选与复现实验。

## 实现范围与边界

- 场景草案必须满足 G4 schema 并经确定性门禁；
- 诊断引用 run/frame/candidate/evidence id，区分事实/推断/未知；
- 缺证据时 abstain，不自造真值、不改标签/代码；
- 只发工具请求，执行由治理层控制；

## 完成标准与验证

- 与固定模板同预算比较有效率、独立失败、覆盖和成本。
- 已知根因盲测准确、证据率、幻觉、abstention 和成本。
- 提示注入、伪证据、缺/冲突日志和越权被检测。
- Agent 负收益不影响确定性工作流。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `agents/research_assistant、evidence_retriever、validation/g7、tests/g7、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
