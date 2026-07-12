# G5-02：轻量 Object-Centric 交互世界模型

**状态**：PENDING  
**依赖**：G5-01

## 目标与范围

训练 action-conditioned object/vector/BEV latent dynamics，以场景 token、Actor 交互和 Ego 候选为条件预测 1～5 秒多 Actor 状态/占用。核心不生成像素视频。

允许修改 `safedrive_foundry/world_model/model/**`、`safedrive_foundry/world_model/training/**`、`safedrive_foundry/config/**`、`tests/g5/**` 和 `safedrive_foundry/registry/**`。

## 完成标准与验证

- 单张 4080 可训练/推理，支持候选批量 rollout、mixed precision 和 checkpoint。
- 不同 Ego 候选产生可区分 future；小样本过拟合和无动作条件消融通过。
- 报告显存、吞吐、数值稳定、actor error 和 interaction slice。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录模型/data/config hash、checkpoint、显存和异常日志。
