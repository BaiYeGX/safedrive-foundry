# G7-03：Failure Analyst Agent

**状态**：PENDING  
**依赖**：G7-02

## 目标与范围

对齐 Runtime、VLA、World、Safety、Planner、Controller 和 CARLA 日志，生成根因假设、支持/反驳证据、置信度与复现实验建议；未知时必须 abstain。

允许修改 `safedrive_foundry/agents/failure_analyst/**`、`safedrive_foundry/agents/evidence_retriever/**`、`safedrive_foundry/validation/g7/**` 和 `docs/architecture/**`；不得自行改标签真值。

## 完成标准与验证

在带已知根因的故障注入集盲测定位准确率、证据率、幻觉率、abstention 和成本；伪造证据/提示注入/缺日志红队可检测。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取 Agent 版本、评测集、错误样例和待验证根因。
