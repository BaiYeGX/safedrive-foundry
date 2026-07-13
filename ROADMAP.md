# SafeDrive Foundry 总路线（G0 冻结后重构版）

> 总需求以 [FUTURE_PROJECT_VISION.md](./FUTURE_PROJECT_VISION.md) 为准。本文件只定义 G0～G8 的执行顺序、阶段依赖与关闭门禁。G0 已验收冻结，本次重构不修改 G0 产物。

## 1. 项目主线

项目围绕一个问题展开：在单张 RTX 4080 16GB 上，能否通过“轻量驾驶 VLA + 动作条件世界模型 + 独立 Safety Kernel + 反事实失败闭环”，改善长尾驾驶，同时保持常规能力、实时性和可追溯安全下限。

```text
G0 环境与确定性（已冻结）
  ↓
G1 统一运行时与经典专家
  ↓
G2 Independent Safety Kernel
  ↓
G3 轻量可验证驾驶 VLA
  ↓
G4 场景覆盖、主动搜索与反事实
  ↓
G5 动作条件交互世界模型
  ↓
G6 失败驱动安全后训练
  ↓
G7 可审计 Agentic Research Loop
  ↓
G8 系统准出与成果证据
```

阶段保持线性关闭，但阶段内部允许任务声明非线性直接依赖。任何学习模块未优于简单基线时，必须记录负结果，不得靠增加新模块绕过。

跨阶段时间、身份、接口、Oracle/Observable、Profile、权限和Registry约束只在[执行架构](./docs/project/EXECUTION_ARCHITECTURE.md)维护；通用完成规则只在`AGENTS.md`维护；最终研究主张见[CLAIMS](./docs/project/CLAIMS.md)。

## 2. 阶段与任务数量

| 阶段 | 任务数 | 阶段目标 | 关闭条件摘要 |
|---|---:|---|---|
| G0 | 6 | 环境、连接、同步与证据基线 | 已验收冻结 |
| G1 | 9 | 统一运行时、经典专家、风险自适应规划与鲁棒实时控制 | 基础/优化专家在 Oracle/Observable 轨道均可闭环，算法净收益与数据门禁可审计 |
| G2 | 4 | 独立 Validator、两级修复、仲裁回退与故障验收 | 学习模块全失效时 Classic+Safety 仍可安全闭环 |
| G3 | 5 | 统一数据、非语言基线、轻量 Fast/Slow VLA 与闭环 | 无特权泄漏，多候选轨迹、选择性驾驶和资源边界可量化 |
| G4 | 4 | 场景注册、同预算搜索、反事实重建与最小化 | 失败有效、可解、可复现、可去重、可最小化 |
| G5 | 5 | 动作条件结构化世界模型、因果校准与主动验证 | 世界模型真实使用 Ego 动作并改善至少一个候选排序切片，失效可降级 |
| G6 | 4 | 事件数据、Shadow DAgger、偏好后训练与抗遗忘 | 完成一轮可信飞轮且正常/OOD/实时性不过线退化 |
| G7 | 3 | 受控研发助手与确定性数据/准出工作流 | Agent 可关闭、不可越权、建议与机器裁决分离 |
| G8 | 4 | 协议冻结、统一回归、统计证据和最终准出 | 四个发布配置可复现，所有对外主张可追溯 |

G1～G8 共 38 项；全项目共 44 项。任务按可形成独立运行能力和证据的垂直切片拆分，避免为每个模型头、搜索器或报告重复建立任务。

## 3. G0：环境与确定性（已冻结）

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G0-01 | 环境盘点与版本冻结 | 无 |
| G0-02 | WSL2、GPU 与 ROS 2 基础环境 | G0-01 |
| G0-03 | CARLA Server 安装与独立验证 | G0-01、G0-02 |
| G0-04 | 工程骨架与 CARLA–ROS 跨系统连通 | G0-02、G0-03 |
| G0-05 | 确定性同步与环境诊断 | G0-04 |
| G0-06 | G0 集成验收与证据包 | G0-01～G0-05 |

## 4. G1：统一运行时与经典专家

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G1-01 | 正式数据契约、稳定运行身份与 20Hz/50Hz Profile | G0-06 |
| G1-02 | Simulation Runtime、ScenarioRunner、Actor/Sensor/Control 生命周期 | G1-01 |
| G1-03 | OpenDRIVE、Lane Graph、全局路由与行为层 | G1-02 |
| G1-04 | Frenet Lattice、动态占用与 S-T 速度规划 | G1-03 |
| G1-05 | Hybrid A*–Reeds–Shepp 复杂机动规划基线 | G1-03 |
| G1-06 | MPC/PID 多速率控制、20ms deadline 与降级 | G1-04、G1-05 |
| G1-07 | RACE-Plan：风险场、自适应 Frenet 与 Hybrid A* 搜索优化 | G1-04、G1-05 |
| G1-08 | RACE-Control：鲁棒自适应与 Deadline-Aware MPC | G1-06、G1-07 |
| G1-09 | Classic-Oracle/Observable 集成、算法消融、数据门禁与阶段验收 | G1-01～G1-08 |

## 5. G2：Independent Safety Kernel

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G2-01 | Safety 契约、Trajectory Validator 与安全状态机 | G1-09 |
| G2-02 | 纵向 QP + 受限二维 RATO-SCP 两级最小修复 | G2-01 |
| G2-03 | 候选预筛、World-ready 仲裁、Shadow 与回退链 | G2-02 |
| G2-04 | 故障注入、Observable 退化与 Safety 阶段验收 | G2-01～G2-03 |

## 6. G3：轻量可验证驾驶 VLA

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G3-01 | 场景身份、VLA 数据契约、冻结划分与泄漏审计 | G2-04 |
| G3-02 | Route/Ego、单轨迹与非语言多候选基线 | G3-01 |
| G3-03 | 4080 约束下的轻量 Fast/Slow VLA 与并行轨迹头 | G3-02 |
| G3-04 | 多任务训练、动作 grounding、校准、OOD 与资源稳定性 | G3-03 |
| G3-05 | VLA+Safety 闭环、选择性驾驶与阶段验收 | G3-01～G3-04 |

## 7. G4：场景覆盖、主动搜索与反事实

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G4-01 | 场景注册表、核心长尾套件、覆盖与可解性 | G3-05 |
| G4-02 | Random/LHS 同预算基线与可恢复执行器 | G4-01 |
| G4-03 | 覆盖引导 MAP-Elites 主动失败搜索 | G4-02 |
| G4-04 | 可比重建、反事实分支、最小化、聚类与阶段验收 | G4-01～G4-03 |

## 8. G5：动作条件交互世界模型

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G5-01 | 同场景多动作分支数据与 CV/CTRV/IDM/Reward 基线 | G4-06 |
| G5-02 | 轻量 object-centric/vector latent 交互动力学 | G5-01 |
| G5-03 | 多模态未来、风险/奖励、不确定性、动作敏感性与校准 | G5-02 |
| G5-04 | VLA 候选排序、Active CARLA Verification 与安全降级 | G5-03 |
| G5-05 | VLA+World+Safety 精度—收益—成本闭环验收 | G5-01～G5-04 |

## 9. G6：失败驱动安全后训练

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G6-01 | Safety Event 窗口、Hard Case Mining、谱系与数据门禁 | G5-05 |
| G6-02 | Shadow DAgger、Takeover 与 Pre-intervention 采集 | G6-01 |
| G6-03 | CARLA 验证偏好、Risk Anticipation 与轻量后训练 | G6-02 |
| G6-04 | 历史失败/正常/OOD 抗遗忘回归与飞轮验收 | G6-01～G6-03 |

## 10. G7：可审计 Agentic Research Loop

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G7-01 | 白名单工具、权限、预算、审计与中断恢复 | G6-04 |
| G7-02 | Scenario/Failure Research Assistant 与证据 grounding | G7-01 |
| G7-03 | 确定性 Data/Release 工作流、脚本对照与红队验收 | G7-01、G7-02 |

## 11. G8：系统准出与成果证据

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G8-01 | 核心主张、最终协议、代码/模型/数据/场景资产冻结 | G7-03 |
| G8-02 | 四发布配置的 ID/长尾/OOD/故障/实时性统一回归 | G8-01 |
| G8-03 | 核心消融、统计、Pareto、Evidence 与冷启动复现 | G8-02 |
| G8-04 | 最终审计、限制登记、演示和版本准出 | G8-03 |

## 12. 阶段关闭规则

每个阶段关闭前必须同时满足：

1. 阶段内所有任务为 `COMPLETED`；
2. 阶段验收任务运行真实测试，而不是只检查文件存在；
3. 任务输出已登记代码、配置、数据、run、指标、资源和失败；
4. 所有未达标方法明确记录为负结果或适用边界；
5. Evidence Bundle hash/link/schema 自检通过；
6. `PROGRESS.md` 已更新；
7. 用户明确批准后，才可把下一阶段第一项设为 `CURRENT`。
