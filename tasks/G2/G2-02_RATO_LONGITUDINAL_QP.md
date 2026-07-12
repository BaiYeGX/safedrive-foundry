# G2-02：RATO 纵向 QP 基线

**状态**：PENDING  
**依赖**：G2-01

## 目标与范围

在不改变横向路径的前提下，以 OSQP 实现最小必要减速/停车修复，作为 Rule Slowdown、Hard Reject 与二维 RATO-SCP 之间的可解释基线。建模速度、加速度、jerk、停止线、前车/动态占用、slack、进度和舒适代价。

## 不做与允许修改

不修改横向轨迹。允许修改 `safedrive_foundry/safety_kernel/rato/longitudinal/**`、`safedrive_foundry/safety_kernel/solver/**`、`safedrive_foundry/config/**`、`tests/g2/**`、`safedrive_foundry/validation/g2/**` 和 `docs/architecture/**`。

## 完成标准与验证

- Raw/Rule/HardReject/Longitudinal 使用统一接口。
- 保存激活约束、slack、修改范数、进度损失、P50/P95/P99 和无解原因。
- 防止持续零速取得虚假安全；无解进入规定回退。
- 用红灯、急刹、切入和跟车场景多 seed 对照。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取最近 solver 日志、场景 ID、OSQP warm-start 与未满足约束。
