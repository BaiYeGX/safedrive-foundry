# G2-04：Shadow 仲裁、回退与最小风险

**状态**：PENDING  
**依赖**：G2-03

## 目标与范围

统一 Raw/修复/Classic 候选，建立可审计仲裁、风险门控、Classic Shadow、Minimal Risk 与 Emergency Brake 链。首版只使用确定性风险和 Validator 结果，为后续学习风险保留接口。

## 不做与允许修改

不接入 VLA 或世界模型。允许修改 `safedrive_foundry/safety_kernel/arbitration/**`、`safedrive_foundry/safety_kernel/fallback/**`、`safedrive_foundry/runtime/**`、`safedrive_foundry/ros_ws/src/**`、`safedrive_foundry/config/**`、`tests/g2/**` 和 `docs/architecture/**`。

## 完成标准与验证

- 每次选择记录候选集、拒绝原因、风险、修改量与最终动作。
- 连续异常进入 Minimal Risk，迫近碰撞进入 Emergency，恢复有滞回。
- Shadow 不争夺控制权；Classic 回退使用明确版本。
- 在正常/边界/无解/超时 case 核对状态与动作时间线。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从最近 SafetyEvent 时间线、候选 ID 和未通过状态转移继续。
