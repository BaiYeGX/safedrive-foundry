# G6-02：Hard Case Mining 与数据门禁

**状态**：PENDING  
**依赖**：G6-01

## 目标与范围

按风险、分歧、不确定性、重复失败、覆盖稀缺性和学习价值挖掘、去重、分层和平衡困难样本；执行同步、完整性、泄漏、标签冲突、分布和许可门禁。

允许修改 `safedrive_foundry/data_pipeline/hard_case_miner/**`、`safedrive_foundry/data_pipeline/gates/**`、`safedrive_foundry/data_pipeline/query/**`、数据卡、`tests/g6/**` 和报告。

## 完成标准与验证

拒绝样本有原因；失败簇与连续帧不会虚增数量；高价值采样与随机/周期采样同预算比较；数据版本可重建且不覆盖基准集。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取数据版本、门禁统计、争议样本和未处理 run。
