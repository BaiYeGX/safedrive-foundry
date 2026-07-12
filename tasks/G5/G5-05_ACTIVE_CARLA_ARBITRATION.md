# G5-05：候选排序与 Active CARLA Verification

**状态**：PENDING  
**依赖**：G5-04

## 目标与范围

把 WorldRollout 接入候选仲裁；对高风险、高不确定、候选分歧或 OOD 请求昂贵 CARLA 分支，缓存结果并反馈误差。超时/失效时降级到 G2 Safety+Classic。

允许修改 `safedrive_foundry/world_model/runtime/**`、`safedrive_foundry/runtime/**` 中的 arbitration adapter、`safedrive_foundry/world_model/verification_queue/**`、`safedrive_foundry/world_model/cache/**`、`tests/g5/**` 和报告。

## 完成标准与验证

- 世界模型不能覆盖 Validator 硬约束；CARLA verify 有预算、超时和可恢复队列。
- 报告排序收益、CARLA rollout 节省率、错误筛选率、延迟和降级率。
- 注入 model unavailable/overconfident/timeout 验证回退。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取队列、缓存 hash、预算和失败 candidate。
