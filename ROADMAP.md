# SafeDrive Foundry 总路线

总需求见 [FUTURE_PROJECT_VISION.md](./FUTURE_PROJECT_VISION.md)。本路线只定义阶段顺序、依赖和门禁，不替代总需求。

## 阶段依赖

```text
G0 环境与确定性
 ↓
G1 经典专家
 ↓
G2 Safety Kernel
 ↓
G3 VLA 基线
 ↓
G4 场景搜索与反事实
 ↓
G5 世界模型
 ↓
G6 安全后训练
 ↓
G7 Agentic Loop
 ↓
G8 版本准出
```

阶段必须按依赖推进。前置门禁未通过，不得用新增热点模块绕过基础问题。

## G0：环境与确定性

目标：建立可重复的 Windows/WSL2/CARLA/ROS 2 软件在环基础。

完成条件：

- Windows、WSL2、GPU、CARLA、ROS 2 连接烟雾测试通过；
- 固定步长、唯一 tick master 与 frame 对齐稳定；
- `versions.lock` 与环境诊断入口完成；
- G0 证据包可以复现环境和测试结果。

详细任务位于 `tasks/G0/`，共6项，必须按编号和依赖执行。

## G1：经典专家

目标：完成 Route、行为、Frenet/Hybrid A*、S-T、MPC 闭环，以及专家质量门禁和指标。

| 任务 | 能力包 |
|---|---|
| G1-01 | 地图、Lane Graph、全局路由与行为层 |
| G1-02 | Frenet Lattice与Hybrid A*–Reeds–Shepp双局部规划器 |
| G1-03 | S-T速度规划、MPC与PID控制闭环 |
| G1-04 | 经典专家集成、质量门禁与基线报告 |

## G2：Safety Kernel

目标：完成 Validator、状态机、RATO、Classic回退和故障降级。

| 任务 | 能力包 |
|---|---|
| G2-01 | Safety契约、Trajectory Validator与安全状态机 |
| G2-02 | RATO纵向QP与RATO-SCP轨迹修复 |
| G2-03 | 风险门控、Shadow仲裁、回退与故障注入 |
| G2-04 | Safety Kernel集成与安全—效率评测 |

## G3：VLA基线

目标：完成数据隔离、特权信息审计、轻量VLA的open-loop与closed-loop基线。

| 任务 | 能力包 |
|---|---|
| G3-01 | VLA数据管线、划分、语言标签与特权审计 |
| G3-02 | 轻量驾驶VLA架构与训练接口 |
| G3-03 | SFT/VLT训练与open-loop评测 |
| G3-04 | VLA闭环部署、不确定性与grounding基线 |

## G4：场景搜索与反事实

目标：完成Random/LHS/CMA-ES公平对照、失败复现、最小化和分支一致性。

| 任务 | 能力包 |
|---|---|
| G4-01 | 场景注册、故障schema与Random/LHS基线 |
| G4-02 | CMA-ES风险场景搜索与有效性门禁 |
| G4-03 | 反事实分支、状态一致性、失败最小化与聚类 |
| G4-04 | 场景搜索与反事实综合评测 |

## G5：世界模型

目标：完成动作条件交互响应建模、简单基线对照和不确定性触发CARLA验证。

| 任务 | 能力包 |
|---|---|
| G5-01 | 动作条件rollout数据集与简单世界模型基线 |
| G5-02 | Latent Interactive World Model |
| G5-03 | 风险/奖励/终止预测与Active CARLA Verification |
| G5-04 | 世界模型因果性、排序、校准与效率评测 |

## G6：安全后训练

目标：完成DAgger、Preference、Risk数据门禁和新旧策略回归。

| 任务 | 能力包 |
|---|---|
| G6-01 | 事件窗口、Hard Case Mining与后训练数据门禁 |
| G6-02 | Shadow Expert DAgger闭环 |
| G6-03 | 反事实Preference与风险预判后训练 |
| G6-04 | 后训练版本A/B与正常能力回归 |

## G7：Agentic Loop

目标：完成白名单工具Agent、固定脚本对照、grounding与幻觉评测。

| 任务 | 能力包 |
|---|---|
| G7-01 | Agent白名单工具、schema、审计与沙箱 |
| G7-02 | Scenario Scientist与Counterexample Agent |
| G7-03 | Failure Analyst与Data Curator Agent |
| G7-04 | Release Gate Agent与Agent净收益评测 |

## G8：版本准出

目标：完成全系统消融、未见ODD、故障、实时性、资源评测和可追溯准出报告。

| 任务 | 能力包 |
|---|---|
| G8-01 | 最终实验协议、版本与数据冻结 |
| G8-02 | 全量回归、OOD、故障、实时性与资源测试 |
| G8-03 | 消融、统计检验与Pareto分析 |
| G8-04 | Evidence Bundle、发布报告、演示与最终验收 |

## 阶段关闭规则

每个阶段关闭前必须：

1. 所有阶段任务均为 `COMPLETED`；
2. 阶段门禁验证通过；
3. 失败、限制与外部依赖已记录；
4. `PROGRESS.md` 已更新；
5. 用户明确批准开始下一阶段的第一个任务。
