# G1-08：鲁棒自适应与 Deadline-Aware MPC（RACE-Control）

**状态**：`COMPLETED_WITH_LIMITS`
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

---

## 验收结果与证据映射（追加，不替代上文标准）

| 完成标准条款 | 结果 | 权威证据 / 测试 |
|---|---|---|
| Fixed / Warm / Adaptive / Full 配对消融 | **通过（offline plant）** | `tests.g1.test_g1_08_race_control`；`docs/architecture/evidence/g1-08/repair-20260713/` |
| 扰动表（delay / gain / noise） | **通过（plant 注入）** | disturbances 含 `raw_runs` 与实测 `lateral_err`（非硬编码 0.4/0.3） |
| 稳定 seed | **通过** | `stable_seed` = SHA-256（非 Python `hash()`） |
| Watchdog 真实墙钟 | **通过（offline）** | 与 G1-06 一致：每 tick 单次 `record`；`steps==ticks` |
| 无净收益保留基线 | **通过（admission）** | evidence `default_admission` 相对 fixed 记录 |
| Solver / warm-start 表述 | **有限制** | `solver_type=constrained_gradient_ltv_bicycle`（**非** OSQP/SQP-RTI） |
| CARLA 50Hz A/B | **未 VERIFIED** | 验证方法要求项；本轮以 offline 消融 + live identify 样本为限，**不**声称 CARLA 控制 A/B 收益 |

### 验证与断点

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-13 |
| 最后状态 | `COMPLETED_WITH_LIMITS` |
| 验证 | `PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_08_race_control -v` |
| 权威证据 | `docs/architecture/evidence/g1-08/repair-20260713/`（含 `manifest.json`） |
| 限制 | 非 OSQP；无 CARLA 50Hz A/B VERIFIED；自适应为轻量在线估计 + 消融矩阵 |
| 停止 | **任务关闭**；阶段已 `COMPLETED_WITH_LIMITS`（见 `PROGRESS.md` / G1-09）；不自动 G2 |
