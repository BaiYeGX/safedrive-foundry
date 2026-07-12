# G6-05：历史失败、正常能力与抗遗忘回归

**状态**：PENDING  
**依赖**：G6-04

## 目标与范围

按配对 seed 比较训练前后策略在目标失败、相邻失败簇、正常能力、未见 ODD、故障和实时性上的变化，定义允许提升、不可接受回归与无结论区间。

允许修改 `safedrive_foundry/validation/g6/regression/**`、`safedrive_foundry/validation/statistics/**`、`safedrive_foundry/config/thresholds/**` 和报告；不得根据结果后改阈值。

## 完成标准与验证

报告效应大小、置信区间、重复失败、正常指标、灾难性遗忘和资源变化；失败迁移到新场景必须登记；不以平均分掩盖关键退化。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取冻结阈值、A/B 矩阵、缺失 run 和回归项。
