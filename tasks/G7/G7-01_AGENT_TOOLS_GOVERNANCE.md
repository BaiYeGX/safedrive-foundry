# G7-01：Agent 工具、权限与审计治理

**状态**：PENDING  
**依赖**：G6-06

## 目标与范围

为场景查询/生成/运行/分析/最小化/数据登记/报告建立 typed tool schema、白名单、路径沙箱、资源预算、超时、dry-run、幂等、审批边界、审计和紧急停止。

允许修改 `safedrive_foundry/agents/tools/**`、`safedrive_foundry/agents/governance/**`、`safedrive_foundry/agents/audit/**`、`tests/g7/**` 和 `docs/architecture/**`；不实现具体 Agent 策略。

## 完成标准与验证

Agent 不能执行任意命令、修改冻结资产/真值/阈值；每次调用有稳定 ID、输入输出、版本、成本和证据；合法/非法/越权/重复/超时/中断红队通过。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录工具版本、权限策略、威胁项和审计样例。
