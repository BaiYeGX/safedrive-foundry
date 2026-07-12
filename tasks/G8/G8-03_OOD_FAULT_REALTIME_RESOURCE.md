# G8-03：OOD、故障、20ms 实时性与资源长稳

**状态**：PENDING  
**依赖**：G8-01

## 目标与范围

运行 paired Town/天气/布局/行为/视觉 shift、传感器/通信/模型/执行器故障、`control_50hz` deadline、GPU/CPU/内存/磁盘/温度和长稳压力测试。

## 完成标准与验证

报告 paired degradation、OOD/abstain、故障降级、20ms deadline miss、P50/P95/P99、资源峰值、OOM/断连恢复和连续运行结果；任何失败可追溯。

允许修改 `safedrive_foundry/validation/g8/stress/**`、`safedrive_foundry/validation/g8/collector/**`、报告和最小缺陷修复。最后状态：PENDING；恢复时记录 profile、最后场景、监控与失败队列。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取运行 profile、最后场景、资源监控快照、失败队列与最近命令。
