# G4-03：Manual/Random/LHS/CMA-ES 搜索基线

**状态**：PENDING  
**依赖**：G4-02

## 目标与范围

在完全相同参数空间、rollout 预算、seed 和有效性门禁下实现 Manual、Random、LHS 与 CMA-ES。目标保留碰撞、TTC、违规、接管、失败和进度等原始分项，不用单一分数隐藏结果。

允许修改 `safedrive_foundry/scenario_search/baselines/**`、`safedrive_foundry/scenario_search/checkpoint/**`、`safedrive_foundry/config/search/**`、`tests/g4/**` 和报告。

## 完成标准与验证

- 搜索可暂停恢复，仿真异常/INVALID 不污染优化器。
- 报告首次失败、独立失败、严重度、覆盖、INVALID、复现率与成本。
- CMA-ES 未优于随机时明确记录负结果。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从 optimizer checkpoint、已消费预算和 pending case 继续。
