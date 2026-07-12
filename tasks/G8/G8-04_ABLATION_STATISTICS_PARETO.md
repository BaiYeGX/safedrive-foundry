# G8-04：消融、统计、异质性与 Pareto

**状态**：PENDING  
**依赖**：G8-02、G8-03

## 目标与范围

对 RACE-Plan、RACE-Control、VLA language/evidence、Fast/Slow、World、Active Verify、RATO、后训练与 Agent 做配对消融，计算效应大小、bootstrap/置信区间、稀有失败不确定性、场景子组异质性和安全—效率—舒适—成本 Pareto。

## 完成标准与验证

每个 C0～C6 有对照；RACE 必须比较基础、单项与 Full 且固定场景/seed/deadline；统计聚合层级/seed/配对正确；表图与 run registry 一致；无支持、负贡献和局部贡献明确标记。

允许修改 `safedrive_foundry/validation/g8/statistics/**`、`safedrive_foundry/artifacts/g8/figures/**` 和报告。最后状态：PENDING；恢复时从消融矩阵、统计版本和缺失数据继续。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取消融矩阵、统计脚本版本、缺失数据和最近验证结果。
