# G4-06：搜索与反事实阶段验收

**状态**：PENDING  
**依赖**：G4-01～G4-05

## 目标与完成标准

冻结协议后公平比较 Manual/Random/LHS/CMA-ES/QD，报告首次失败、固定预算独立失败、覆盖、严重度、INVALID、复现、最小反例距离、wall time 和 CPU/GPU 成本。稀有事件使用置信区间；代表性失败人工复核；Counterexample/Regression Registry 与 Evidence Bundle 完整。

允许修改 G4 小缺陷、`safedrive_foundry/validation/g4/**`、`safedrive_foundry/registry/**`、报告、本任务和 `PROGRESS.md`。最后状态：PENDING；恢复时从缺失实验单元与 run_id 继续，不自动开始 G5。

## 验证方法

执行冻结预算的多基线重复实验、失败复现抽检、registry 完整性与证据哈希检查。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取缺失实验单元、run_id 与最近失败命令。
