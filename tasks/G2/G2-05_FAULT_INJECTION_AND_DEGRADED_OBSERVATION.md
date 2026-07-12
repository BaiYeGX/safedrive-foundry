# G2-05：故障注入与退化观测安全

**状态**：PENDING  
**依赖**：G2-04

## 目标与范围

建立状态延迟、丢包/乱序、控制保持、定位偏移、相机退化、对象漏检/偏移、低附着、执行器饱和和模块超时故障矩阵，并验证 Observable 轨道下的预期降级。

## 不做与允许修改

不把 Oracle 结果作为运行时结果。允许修改 `safedrive_foundry/fault_injector/**`、`safedrive_foundry/runtime/**` 中的 observation adapter、`safedrive_foundry/scenario/**`、`safedrive_foundry/config/**`、`safedrive_foundry/validation/g2/**`、`tests/g2/**` 和 `docs/architecture/**`。

## 完成标准与验证

- 故障具有开始、持续、严重度、恢复、seed 和预期安全动作。
- 故障注入本身不破坏 frame identity；INVALID 注入单独记录。
- Oracle/Observable 安全差距、检测延迟和恢复时间可计算。
- 每族运行正常/边界/严重级别并保存可复现 case。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取故障矩阵、最后 case 和未满足预期动作。
