# G1-06：MPC/PID 多速率控制与 20ms 门禁

**状态**：`COMPLETED_WITH_LIMITS`
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

## 验证与断点

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-13 |
| 最后状态 | `COMPLETED_WITH_LIMITS` |
| 已完成 | ControlLoop + Watchdog（每 tick 单次 record）+ 缓冲 + 降级链；有符号 reverse；e2e inject；六类离线闭环；配置冻结 |
| 验证 | `unittest tests.g1.test_g1_06_control` PASS；阶段全量 `tests/g1` **78 OK** |
| 权威证据 | `docs/architecture/evidence/g1-06/repair-20260713/`（含 `manifest.json`） |
| 限制 | `solver_type=constrained_gradient_ltv_bicycle`（**非** OSQP/SQP-RTI）；闭环以离线 plant 为主；**无** live 50Hz VERIFIED 门禁 |
| 停止 | **任务关闭**；阶段状态见 `PROGRESS.md` / G1-09 |
