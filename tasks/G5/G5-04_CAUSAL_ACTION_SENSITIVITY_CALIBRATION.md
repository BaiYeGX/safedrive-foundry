# G5-04：动作敏感性、因果 Actor 与校准

**状态**：PENDING  
**依赖**：G5-03

## 目标与范围

验证世界模型是否真正使用 Ego action 和关键 Actor，而非拟合场景先验。执行 action swap、critical actor removal/perturbation、无动作条件对照、ranking consistency、OOD 与风险校准。

允许修改 `safedrive_foundry/world_model/evaluation/**`、`safedrive_foundry/world_model/interventions/**`、`safedrive_foundry/world_model/calibration/**`、`tests/g5/**` 和报告；不接入正式仲裁。

## 完成标准与验证

- action sensitivity、causal actor response、collision/action ranking 和 ECE/Brier 可计算。
- paired CARLA rollout 作为真值对照，模型错误有类别归因。
- 无动作条件或简单基线更好时不得进入有效性声明。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从 intervention matrix、缺失 run 和 calibration artifact 继续。
