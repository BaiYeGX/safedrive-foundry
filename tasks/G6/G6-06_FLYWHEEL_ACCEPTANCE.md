# G6-06：失败驱动数据飞轮验收

**状态**：PENDING  
**依赖**：G6-01～G6-05

## 目标与完成标准

证明“发现失败→窗口→门禁→DAgger/Preference→后训练→回归”至少完成一轮并保留全谱系。目标长尾改善必须统计可信，正常/OOD/实时性不突破冻结护栏；数据效率和 CARLA/训练成本需报告；未提升时输出负结论。Evidence Bundle 完整。

允许修改 G6 小缺陷、`safedrive_foundry/validation/g6/**`、`safedrive_foundry/registry/**`、报告、本任务和 `PROGRESS.md`。最后状态：PENDING；恢复时从 A/B 完成度、模型/数据版本和缺失 run 继续，不自动开始 G7。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取 A/B 完成度、模型/数据版本、缺失 run 与最近验证结果。
