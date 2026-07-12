# G4-05：反事实重建、分支与失败最小化

**状态**：PENDING  
**依赖**：G4-04

## 目标与范围

通过 scenario seed+warm-up 重建危险起点，验证 Ego/Actor/信号/路线/事件进度容差后，执行 Raw、RATO、Classic、Brake 等分支，并用二分、坐标下降或 delta debugging 最小化与聚类失败。

允许修改 `safedrive_foundry/counterfactual/**`、`safedrive_foundry/counterfactual/replay/**`、`safedrive_foundry/counterfactual/minimizer/**`、`safedrive_foundry/counterfactual/cluster/**`、`tests/g4/**` 和 `docs/architecture/**`；不得把 CARLA Recorder 称为无损状态恢复。

## 完成标准与验证

- 超容差分支标记不可比较，不生成因果结论。
- 保存起点差异、分支动作、碰撞/TTC/进度/舒适和轨迹差异。
- 最小反例可重复，失败簇有稳定 ID 与去重依据。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从 failure run_id、branch frame、容差报告和 minimizer checkpoint 继续。
