# G1-06：MPC/PID 多速率控制与 20ms 门禁

**状态**：PENDING  
**依赖**：G1-04、G1-05

## 目标

统一跟踪两类规划器轨迹，在 `control_50hz` Profile 下实现固定模型 MPC/PID、基础超时降级和真实 20ms deadline 证据；冻结 G1-08 的控制基线。

## 范围

实现运动学自行车模型、LTV 或 SQP-RTI MPC、OSQP warm start、steer/throttle/brake 映射、执行器饱和/死区、PID/Pure Pursuit/LQR 基线、多速率 trajectory buffer 和 stale command 防护。

## 不做与允许修改

不训练模型，不实现 RATO。允许修改 `safedrive_foundry/classic_stack/control/**`、`safedrive_foundry/ros_ws/src/**` 中的控制 ROS 节点、`safedrive_foundry/config/control/**`、`tests/g1/**`、`safedrive_foundry/validation/g1/**` 与 `docs/architecture/**`。

## 完成标准

- 直线、连续弯道、停车、跟车急刹、换道和倒车闭环运行。
- 20ms Profile 报告 solver 与端到端控制 P50/P95/P99、deadline miss 和 jitter。
- 超时/不可行按 MPC→LQR/PurePursuit→PID/Brake 规定降级。
- 增益、权重和约束全部配置化，结果含跟踪误差与舒适性。
- 固定模型、warm-start 开关、solver tolerance、迭代/墙钟预算和回退条件完整登记，供 G1-08 配对消融。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 验证与断点

运行离线轨迹回放、CARLA 50Hz closed-loop、执行器故障和超时测试；中断记录 run_id、warm-start 状态和最近控制帧。

| 字段 | 内容 |
|---|---|
| 最后状态 | PENDING |
| 已完成/验证 | 无 |
| 恢复步骤 | 复核轨迹 schema 与 control Profile 后继续 |
