# G1-09：Classic Expert 双轨集成、算法消融与阶段验收

**状态**：PENDING  
**依赖**：G1-01～G1-08

## 目标

集成 Runtime、Route、Behavior、Frenet/ST、Hybrid A*、RACE-Plan、基础/自适应 MPC 和 PID，形成 Classic-Oracle 教师与 Classic-Observable 公平基线；冻结专家数据质量门禁，并判断哪些算法优化有真实净收益。

## 范围与允许修改

只做 G1 集成、小型缺陷修复、场景套件、算法配对消融、指标、Run/Data Registry 和 Evidence Bundle；实质缺陷退回对应任务。允许修改 `safedrive_foundry/runtime/**`、`safedrive_foundry/classic_stack/**`、`safedrive_foundry/ros_ws/src/**` 中的 G1 adapter、`safedrive_foundry/validation/g1/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/artifacts/g1/**`、文档、本任务和 `PROGRESS.md`。

## 完成标准

- 城市路线、红灯、跟车、切入、换道、绕障、路口让行和复杂机动闭环可重复运行。
- 公布 Basic Classic、RACE-Plan、RACE-Control 和 Full RACE 的配对结果，分别报告安全、进度、舒适、规划/控制尾延迟、资源和失败类型。
- Oracle/Observable 结果分开；正专家数据必须无碰撞/严重违规、动力学可行、舒适合格且无连续超时，失败进入失败库。
- 只有跨 seed/场景产生稳定净收益的改进进入默认专家；局部有效或负贡献保留适用边界。
- 两个运行 Profile、重复性、资源和阶段 Evidence hash/link/schema 检查通过；形成 G2 所需 CandidateTrajectory、RiskField、ActorObservation 和控制基线。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 验证方法与断点

从干净终端执行固定 G1 suite、多 seed 配对消融和算法退化测试；中断记录缺失矩阵单元、run_id、默认配置选择依据和负结果，不自动开始 G2。

| 字段 | 内容 |
|---|---|
| 最后状态 | PENDING |
| 已完成/修改文件/验证 | 无 |
| 阻塞 | 无 |
| 恢复步骤 | 汇总 G1-01～08 状态，从未通过用例或缺失消融继续 |
| 下一条建议命令 | 待执行时填写 |
