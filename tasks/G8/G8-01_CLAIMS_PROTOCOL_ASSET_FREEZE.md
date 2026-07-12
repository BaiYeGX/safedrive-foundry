# G8-01：主张、实验协议与资产冻结

**状态**：PENDING  
**依赖**：G7-05

## 目标与范围

在最终运行前冻结 C1～C6、主/护栏指标、阈值、对照、重复、统计、异常/缺失规则，以及代码、环境、模型、数据、地图、场景、seed 和配置 hash。

允许修改 `safedrive_foundry/validation/g8/protocol/**`、`safedrive_foundry/registry/evidence/**`、`safedrive_foundry/artifacts/g8/freeze/**` 和文档；不运行全量实验。

## 完成标准与验证

判定规则无歧义且不可结果后修改；训练/调参/最终测试隔离；单机预算可执行；从干净工作区重建 smoke 并核对 hash。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录协议版本、冻结清单、资源估算和演练结果。
