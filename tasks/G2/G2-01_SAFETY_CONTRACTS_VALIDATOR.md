# G2-01：Safety 契约、Validator 与状态机

**状态**：PENDING  
**依赖**：G1-09

## 目标与范围

冻结 Policy/Safety 契约，实现对数值、时间、连续性、道路、动力学、静动态碰撞、交通规则与可跟踪性的 Trajectory Validator，以及 NORMAL/DEGRADED/MINIMAL_RISK/EMERGENCY 状态机。每项检查输出严重度、首次违反时间、Actor/规则、裕度和恢复条件。

## 不做与允许修改

不做轨迹优化。允许修改 `safedrive_foundry/safety_kernel/contracts/**`、`safedrive_foundry/safety_kernel/validator/**`、`safedrive_foundry/safety_kernel/state_machine/**`、`safedrive_foundry/ros_ws/src/**` 中的接口 adapter、`safedrive_foundry/config/**`、`tests/g2/**`、`docs/architecture/**`、本任务和 `PROGRESS.md`。

## 完成标准与验证

- 非法、NaN、过期、缺失、越界和极端轨迹均被确定性拒绝。
- 道路/动力学/碰撞/规则检查具有属性测试与边界用例。
- 状态切换含 debounce、持续时间、恢复和不可静默吞错日志。
- 运行合成轨迹、G1 真实轨迹与故意违规注入，保存 SafetyEvent schema 和覆盖率。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

| 字段 | 内容 |
|---|---|
| 最后状态 | PENDING |
| 已完成/验证 | 无 |
| 恢复步骤 | 读取 G1-09 接口、RiskField 与失败轨迹库 |
