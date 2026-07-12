# G2-06：Safety Kernel 消融与阶段验收

**状态**：PENDING  
**依赖**：G2-01～G2-05

## 目标与范围

按冻结协议比较 Raw、Validator、Hard Fallback、Rule Slowdown、Longitudinal QP、RATO-SCP 和 Classic Fallback，证明安全改善没有被无故停车、进度损失或 Oracle 特权掩盖。

允许修改 G2 小型缺陷、`safedrive_foundry/validation/g2/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/artifacts/g2/**`、报告、Evidence、本任务和 `PROGRESS.md`；实质缺陷退回原任务。

## 完成标准与验证

- 多 seed 报告碰撞/TTC、规则、完成率、无故停车、舒适、修改量、solver 成功率、时延和降级率。
- Oracle/Observable、正常/故障分别报告并给出 Pareto 前沿。
- 历史 G1 轨迹和故障回归通过；负结果如实登记。
- G2 Evidence hash/link/schema 自检通过，不自动开始 G3。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时检查消融矩阵、缺失 run_id 和未关闭限制。
