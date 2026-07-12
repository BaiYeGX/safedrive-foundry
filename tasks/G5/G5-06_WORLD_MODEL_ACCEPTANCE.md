# G5-06：世界模型闭环验收

**状态**：PENDING  
**依赖**：G5-01～G5-05

## 目标与完成标准

冻结数据、模型、候选与协议，比较无世界、简单运动学、Reward MLP、Interactive World 和 Active Verify。报告预测、因果敏感性、排序、校准、OOD、闭环收益、时延、显存与 CARLA 成本；至少明确一类稳定净收益或如实给出负结论；常规能力不得被隐藏退化。Evidence 可追溯。

允许修改 G5 小缺陷、`safedrive_foundry/validation/g5/**`、`safedrive_foundry/registry/**`、报告、本任务和 `PROGRESS.md`。最后状态：PENDING；恢复时从验收矩阵与缺失 run 继续，不自动开始 G6。

## 验证方法

执行冻结切分的开环预测、候选排序、校准、OOD 和闭环主动验证矩阵，并复核资源与 CARLA 成本。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取验收矩阵、模型哈希与缺失 run。
