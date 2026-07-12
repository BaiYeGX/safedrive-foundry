# G1-08：鲁棒自适应与 Deadline-Aware MPC（RACE-Control）

**状态**：PENDING  
**依赖**：G1-06、G1-07

## 目标与为什么现在做

在 G1-06 可运行的 LTV-MPC/PID 基线上，解决执行器时延、转向增益和模型失配导致的跟踪退化，以及 20ms deadline 下求解不可预测的问题。优化保持灰盒、可关闭、可回退，避免用黑盒残差网络制造新的安全债。

## 输入、范围与交付物

- 用递推最小二乘或等价轻量辨识在线估计转向增益、执行器一阶时延等可观测参数；设投影边界、遗忘因子、激励不足检测和冻结策略。
- 根据辨识残差/扰动界进行 tube-like 约束收紧或离线误差包络收紧；风险场只调整预先允许的权重/裕度范围。
- 复用上一周期轨迹和 primal/dual 解 warm start；固定稀疏结构、迭代/墙钟预算、可行 incumbent、超时接受条件和 MPC→LQR/PurePursuit→PID/Brake 回退。
- 输出参数估计、残差界、收紧量、KKT/solver 状态、求解分位数、deadline miss、回退原因和闭环误差。

## 明确不做与允许修改

第一版不训练神经残差动力学，不声称形式化鲁棒性证明，不在线修改硬安全阈值。允许修改 `safedrive_foundry/classic_stack/control/adaptation/**`、`safedrive_foundry/classic_stack/control/mpc/**`、`safedrive_foundry/classic_stack/control/solver/**`、`safedrive_foundry/config/control/**`、`tests/g1/**`、`safedrive_foundry/validation/g1/**`、`safedrive_foundry/scenario/**`、文档、本任务和 `PROGRESS.md`。

## 完成标准

- 完成 Fixed MPC、Warm-start MPC、Adaptive MPC、Adaptive+Tightening+Deadline Full 的配对消融。
- 在转向延迟、增益漂移、低附着代理、质量/速度变化与噪声场景报告横纵向误差、约束违反、舒适性、P50/P95/P99、deadline miss 和回退率。
- 参数不可观或估计异常时自动冻结/重置，不得比固定 MPC 更危险；收紧量随误差界单调变化。
- 超时只允许执行已通过约束检查的 incumbent；否则使用冻结回退链。
- 若自适应或收紧没有稳定净收益，默认仍使用 G1-06 基线并记录负结论。

## 验证方法

执行辨识合成测试、离线轨迹回放、CARLA 50Hz A/B、执行器延迟/增益/噪声故障注入、solver 超时与不可行测试，并复核独立 watchdog 记录的真实墙钟时延。

## 资源与断点记录

不训练模型；以 CPU deadline 为核心。中断保存参数估计状态、误差包络版本、OSQP warm-start、run_id、最后控制帧和未完成消融矩阵。

| 字段 | 内容 |
|---|---|
| 最后状态 | PENDING |
| 已完成/修改文件/验证 | 无 |
| 阻塞 | 无 |
| 恢复步骤 | 校验 G1-06 固定 MPC 与 G1-07 RiskField 版本，从最近 A/B run 继续 |
| 下一条建议命令 | 待执行时填写 |
