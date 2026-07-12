# G4-02：场景套件与可解性验证

**状态**：PENDING  
**依赖**：G4-01

## 目标与范围

实现代表性切入/切出、急刹、换道冲突、无保护左转、VRU 横穿、遮挡、道路封闭、低附着与通信故障套件，并用 Classic-Oracle/可达性检查验证场景不是先天不可解。

允许修改 `safedrive_foundry/scenario/**`、`safedrive_foundry/scenario/solvability/**`、`safedrive_foundry/config/scenarios/**`、`tests/g4/**`、`safedrive_foundry/registry/**` 和 `docs/architecture/**`；不做黑盒优化。

## 完成标准与验证

- 每个场景有 base case、参数边界、预期事件和 termination。
- 固定 seed 可复现；无 Ego、错误 spawn、不可达路线和物理重叠判 INVALID。
- 正常能力、长尾、故障和 Regression 套件分离冻结。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从未通过场景、地图和可解性报告继续。
