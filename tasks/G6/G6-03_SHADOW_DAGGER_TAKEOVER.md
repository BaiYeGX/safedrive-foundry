# G6-03：Shadow DAgger、Takeover 与 Pre-intervention

**状态**：PENDING  
**依赖**：G6-02

## 目标与范围

在不改变基线车辆控制权时并行运行 VLA、Classic 与 Safety，以分歧、风险和不确定性触发专家查询，采集 state/policy/expert/safety action 与接管前风险标签。

允许修改 `safedrive_foundry/data_pipeline/dagger/**`、`safedrive_foundry/runtime/**` 中的 shadow adapter、`safedrive_foundry/config/collection/**`、`tests/g6/**` 和报告；不训练新模型。

## 完成标准与验证

执行动作与影子动作严格隔离；监督可追溯到专家版本；事件触发样本价值高于随机；查询预算、重复率和场景平衡可审计。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录采集轮次、版本、样本、触发统计和异常。
