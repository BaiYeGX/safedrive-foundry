# G4-01：场景注册表、参数域与覆盖模型

**状态**：PENDING  
**依赖**：G3-07

## 目标与范围

建立 Functional→Logical→Concrete→Regression→Minimal Counterexample 层级，统一 Engineered、Failure-derived、System-generated 三路来源。定义地图/Actor/天气/故障参数域、约束、有效性、覆盖维度、风险目标和场景版本。

允许修改 `safedrive_foundry/scenario/**`、`safedrive_foundry/registry/scenario/**`、`safedrive_foundry/scenario/schema/**`、`safedrive_foundry/config/**`、`tests/g4/**` 和 `docs/architecture/**`；不实现搜索算法。

## 完成标准与验证

- 场景 schema 可表达核心城市、VRU、遮挡、组合故障和语言条件。
- INVALID 原因、coverage bin、solvability 状态和 provenance 可查询。
- 合法/非法/边界参数属性测试及 schema migration 测试通过。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取 registry 版本、未完成场景族和 schema 失败。
