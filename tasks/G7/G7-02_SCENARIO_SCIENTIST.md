# G7-02：Scenario Scientist Agent

**状态**：PENDING  
**依赖**：G7-01

## 目标与范围

实现把覆盖缺口/失败簇/世界模型不确定区转为可检验假设，调用 schema validator、搜索、复现和最小化工具，输出 hypothesis→experiment→observation→conclusion→evidence。

允许修改 `safedrive_foundry/agents/scenario_scientist/**`、`safedrive_foundry/agents/prompt_policy/**`、`safedrive_foundry/validation/g7/**` 和 `docs/architecture/**`；不得直接写 OpenSCENARIO 绕过合法性检查。

## 完成标准与验证

至少三类风险假设闭环；已知反例召回、错误结论、无效场景、复现和成本可测；相同任务核心结论稳定，所有引用可追溯。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录 hypothesis 队列、Agent 配置、tool trace 和 counterexample ID。
