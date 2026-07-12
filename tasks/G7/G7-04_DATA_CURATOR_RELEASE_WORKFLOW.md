# G7-04：确定性 Data Curator 与 Release Gate 工作流

**状态**：PENDING  
**依赖**：G7-03

## 目标与范围

把数据查询、去重、门禁、版本登记和 release regression 设计为确定性工作流；Agent 只能提出查询/结论草案，由 schema、hash、统计和阈值程序裁决。

允许修改 `safedrive_foundry/agents/workflows/**`、`safedrive_foundry/agents/data_release_adapter/**`、`tests/g7/**` 和报告。

## 完成标准与验证

重复请求产生同一 manifest；冻结集不可写；准出结论由机器检查项产生；关闭 LLM 后流程仍可手工/脚本运行；所有 Agent 建议与最终裁决差异留档。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录 workflow 版本、manifest、准出检查和待审批项。
