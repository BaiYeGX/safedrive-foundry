# G6-04：反事实 Preference 与风险前瞻后训练

**状态**：PENDING  
**依赖**：G6-03

## 目标与范围

用 CARLA 验证的反事实候选、专家反馈和接管前窗口构造 pairwise/ranking 数据，采用 LoRA/QLoRA 的 SFT、ranking/DPO 类轻量后训练和 Risk Anticipation 辅助目标。世界模型只可提议样本，未经 CARLA 验证不得作为安全真值。

允许修改 `safedrive_foundry/driving_vla/posttrain/**`、`safedrive_foundry/driving_vla/losses/**`、`safedrive_foundry/config/posttrain/**`、`safedrive_foundry/registry/**`、`tests/g6/**` 和报告。

## 完成标准与验证

数据、权重和超参可追溯；训练可恢复且适配 16GB；独立 split 的偏好正确率、风险提前量和校准优于训练前；无数据泄漏。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录 base model、adapter、step、数据版本、checkpoint 和失败。
