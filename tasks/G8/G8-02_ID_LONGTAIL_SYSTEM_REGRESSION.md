# G8-02：ID、常规、长尾与历史失败回归

**状态**：PENDING  
**依赖**：G8-01

## 目标与范围

按冻结矩阵运行 Classic、VLA、World、Safety、Full Hybrid 在常规城市、长尾、历史失败和最小反例上的闭环回归，记录安全、完成、舒适、效率和模块事件。

允许修改 `safedrive_foundry/validation/g8/regression/**`、`safedrive_foundry/validation/g8/collector/**`、`safedrive_foundry/artifacts/g8/**` 和故障修复所需最小范围；协议不可更改。

## 完成标准与验证

矩阵完整或按协议标缺失；失败定位到 run/config/scenario/model；关键 case 跨 seed 复跑并抽查视频/轨迹；产物损坏与重复检测通过。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从最后 run、矩阵 cell、失败队列和资源状态继续。
