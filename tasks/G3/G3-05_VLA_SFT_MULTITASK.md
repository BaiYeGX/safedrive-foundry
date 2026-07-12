# G3-05：VLA 多任务 SFT/VLT

**状态**：PENDING  
**依赖**：G3-04

## 目标与范围

训练轨迹 NLL/多模态、行为、风险、关键 Actor、结构化证据与平滑目标，完成资源稳定、断点续训和 open-loop 消融。与 G3-03 基线、无语言、单轨迹、无辅助头公平对照。

允许修改 `safedrive_foundry/driving_vla/training/**`、`safedrive_foundry/driving_vla/losses/**`、`safedrive_foundry/config/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/validation/g3/open_loop/**` 和报告；不做后训练或正式闭环。

## 完成标准与验证

- 训练可从合法 checkpoint 精确恢复，NaN/OOM 有自动降级和审计。
- 独立 split 报告 ADE/FDE/NLL、行为、风险、Actor、grounding、资源和置信区间。
- 选择候选 checkpoint 的规则预先登记；负消融如实保存。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录 epoch/step、数据与配置 hash、最佳指标和下一实验。
