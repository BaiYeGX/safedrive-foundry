# G1～G8 跨阶段执行架构

## 1. 本文定位

本文把 G0 已验证的环境链路转化为后续算法研发必须遵守的系统边界。它不修改 G0，也不替代 `FUTURE_PROJECT_VISION.md`。

## 2. 运行主链

```text
CARLA Server (Windows)
  ↕ TCP
Simulation Runtime (WSL2, unique tick owner)
  ├─ Scenario/Actor/Sensor lifecycle
  ├─ Frame identity and /clock
  ├─ ObservationProvider
  ├─ Policy candidates
  ├─ Safety Kernel
  ├─ Controller
  └─ Run/Event/Evidence writer
```

正式业务节点不得自行调用 `world.tick()`。需要 CARLA truth 的模块通过显式 `OracleObservation` 获得，不能从普通策略输入旁路读取。

## 3. 身份与时间

所有消息、Parquet、日志、模型样本和证据至少包含：

```text
experiment_id
run_id
scenario_id
attempt_id
server_epoch
carla_frame
simulation_time
wall_time
producer_version
schema_version
```

`episode_id` 不再由 bridge 随机生成后作为跨系统权威身份；兼容字段可保留，但正式主键由 Runtime 生成。

## 4. 多速率 Profile

| Profile | CARLA 固定步长 | 主要用途 | 核心门禁 |
|---|---:|---|---|
| `throughput_20hz` | 0.05 s | 数据、搜索、一般闭环 | 帧一致、吞吐和稳定性 |
| `control_50hz` | 0.02 s | MPC/PID/Safety 实时性 | 20ms deadline、抖动和降级 |
| `training_offline` | 无 CARLA | VLA/World 训练 | 显存、checkpoint、数据版本 |
| `counterfactual_serial` | 0.05/0.02 s | 短时分支 | 起点可比、预算和恢复 |
| `agent_offline` | 无实时控制 | 假设、诊断、数据与报告 | 白名单、审计和成本 |

不得把 `throughput_20hz` 的结果写成 20ms 控制周期结果，也不得把 VLA 端到端推理延迟混同为底层控制 deadline。

## 5. Oracle/Observable 双轨

| 轨道 | 输入 | 用途 | 可对外表述 |
|---|---|---|---|
| Oracle | CARLA 真值 Actor/地图/信号 | 教师、上界、安全诊断、标签 | 只能称仿真特权上界 |
| Observable | 图像、Ego、导航、经噪声/延迟的对象接口 | 公平运行时比较 | 可用于策略闭环结论 |

Classic Expert 至少提供 `Classic-Oracle` 和 `Classic-Observable` 两个适配器。Safety Kernel 同时报告 Oracle safety upper bound 与 degraded-observation robustness。

## 6. 模型边界

- VLA 输出候选轨迹、行为、风险、不确定性和结构化证据，不直接输出未约束控制。
- 世界模型只做动作条件 object/vector/BEV latent rollout，不以像素视频生成作为核心交付。
- Agent 不进入实时控制环，不修改真值、冻结测试集或安全阈值。
- 大型公开 VLA 可作为离线教师或架构参考，不是 16GB 单机运行依赖。

## 6.1 RACE 经典专家优化边界

- `RiskField` 是规划、控制和 Safety 共用的版本化风险描述，不是共享写权限；规划可使用软代价，Safety 独占硬阈值和 slack 上限。
- G1 的风险场必须由 CV/CTRV/IDM 与显式不确定性包络独立运行；G5 世界模型只能通过 adapter 增强，不能成为 Classic 基线依赖。
- RACE-Plan 只优化采样/搜索效率、风险与可跟踪性；RACE-Control 只优化可观测参数适应、误差收紧和 deadline 行为；RATO-SCP 负责候选最小安全修复。
- 神经启发式、神经残差动力学和 Branch MPC 不属于核心完成条件，除非基础三项已验证且另立任务。
- 详细选型、文献依据和消融见 `docs/project/decisions/CLASSIC_ALGORITHM_OPTIMIZATION.md`。

## 7. Registry 最小集合

- Run Registry：运行身份、场景、seed、版本、状态、资源、指标。
- Scenario Registry：场景族、参数域、合法性、可解性和失败历史。
- Data Registry：来源、许可、切分、hash、门禁和泄漏审计。
- Model Registry：架构、权重 hash、训练数据、配置和评测。
- Evidence Registry：主张与可验证产物的映射。

Registry 可先以 SQLite/DuckDB + Parquet 实现，不因未来存储后端变化修改上层契约。

## 8. 自动恢复

每项长任务保存 config hash、输入版本、已消费场景/epoch、checkpoint、资源峰值和失败类别。恢复前必须验证这些外部状态仍匹配；不允许仅凭文件存在跳过验证。
