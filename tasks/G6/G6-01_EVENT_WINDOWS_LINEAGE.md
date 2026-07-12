# G6-01：Safety Event 窗口与数据谱系

**状态**：PENDING  
**依赖**：G5-06

## 目标与范围

把碰撞、低 TTC、接管、RATO 修复、策略分歧、OOD、超时和恢复事件切成 normal/pre-event/intervention/recovery 窗口，并关联策略、世界、专家、安全、控制和 CARLA 真值。

允许修改 `safedrive_foundry/data_pipeline/events/**`、`safedrive_foundry/data_pipeline/lineage/**`、`safedrive_foundry/data_pipeline/schema/**`、`safedrive_foundry/registry/**`、`tests/g6/**` 和 `docs/architecture/**`；不筛选训练样本。

## 完成标准与验证

窗口边界、优先级、重叠、事件因果顺序和 provenance 可重放；同一日志重复处理 ID/hash 一致；Regression 数据不可被训练写入。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录已处理 run、窗口版本、冲突事件和下一分片。
