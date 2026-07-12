# G5-03：多模态未来、风险与不确定性

**状态**：PENDING  
**依赖**：G5-02

## 目标与范围

为世界模型加入多模态响应、碰撞/规则/舒适/终止/奖励输出和 epistemic/aleatoric 不确定性，形成批量 `WorldRollout`。

允许修改 `safedrive_foundry/world_model/heads/**`、`safedrive_foundry/world_model/losses/**`、`safedrive_foundry/world_model/calibration/**`、`safedrive_foundry/config/**`、`tests/g5/**` 和报告。

## 完成标准与验证

- 已知安全/擦碰/碰撞/违规样例输出分项风险与置信度。
- 不确定性与误差相关，模式坍塌和过度自信有检测。
- 风险接口可评价 K 个候选，失败时返回明确不可用状态。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取风险头版本、阈值、checkpoint 和失败 slice。
