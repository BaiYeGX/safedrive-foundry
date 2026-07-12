# G5-01：动作分支数据与简单世界基线

**状态**：PENDING  
**依赖**：G4-06

## 目标与范围

从同一可比场景起点执行专家、非专家、危险和扰动 Ego action chunk，构建 `P(other future | scene, ego action)` 数据；实现 persistence、CV/CTRV、IDM/规则和 Reward MLP 基线。

允许修改 `safedrive_foundry/world_model/data/**`、`safedrive_foundry/world_model/baselines/**`、`safedrive_foundry/world_model/schema/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/config/**`、`tests/g5/**` 和 `docs/architecture/**`；不训练正式世界模型。

## 完成标准与验证

- action/future 严格对齐并覆盖非专家动作；不可比分支隔离。
- ID/OOD/Regression 划分与 VLA 测试集无泄漏。
- 基线报告 actor/occupancy、collision/action ranking、校准与时延。
- 固定 seed 小数据重复 hash 一致并可断点续采。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录数据版本、分支队列、基线结果和失败样本。
