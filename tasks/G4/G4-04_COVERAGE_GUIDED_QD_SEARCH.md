# G4-04：覆盖引导 Quality-Diversity 搜索

**状态**：PENDING  
**依赖**：G4-03

## 目标与范围

实现面向“更多不同失败而非单个最严重失败”的覆盖引导 Quality-Diversity 搜索，可采用 MAP-Elites/novelty archive 等单机可执行方法，并与 CMA-ES/Random 同预算比较。

允许修改 `safedrive_foundry/scenario_search/qd/**`、`safedrive_foundry/scenario_search/archive/**`、`safedrive_foundry/config/search/**`、`tests/g4/**` 和报告；不引入 Agent。

## 完成标准与验证

- 行为描述符、archive cell、novelty 和风险分开记录。
- 重复失败不会虚增独立失败数；覆盖空洞可解释。
- 中断恢复不重复 rollout，多 seed 同预算结果可统计。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取 archive hash、预算、未完成 cell 和失败日志。
