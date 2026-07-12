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
| G2 | 6 | Validator、RATO、仲裁、故障与最小风险 | 安全改善不以无故停车和进度损失掩盖 |
| G3 | 7 | 数据、基线、轻量 VLA、校准、OOD 与闭环 | 无特权泄漏，闭环收益与失效边界可量化 |
| G4 | 6 | 场景覆盖、基线搜索、QD 搜索、反事实最小化 | 同预算搜索公平，失败可复现、可去重、可最小化 |
| G5 | 6 | 动作条件世界模型、因果评价与主动 CARLA 验证 | 候选排序优于简单基线，错误筛选有降级 |
| G6 | 6 | Hard Case、DAgger、偏好后训练与抗遗忘 | 至少一轮飞轮产生可信净收益且旧能力不过线退化 |
| G7 | 5 | 两类受控 Agent 与确定性数据/准出工作流 | Agent 优于固定脚本且通过治理红队 |
| G8 | 6 | 冻结协议、系统回归、统计、证据和最终准出 | 所有对外主张可追溯并完成冷启动复现 |

G1～G8 共 51 项；全项目共 57 项。任务不是按文件或类拆分，而是按“可在一个 Codex 对话中实现、测试、形成证据并停下”的垂直切片拆分。

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
| G2-02 | RATO 纵向 QP 简单修复基线 | G2-01 |
| G2-03 | 二维 RATO-SCP 与可行走廊 | G2-02 |
| G2-04 | Shadow、仲裁、回退、Minimal Risk 与 Emergency | G2-03 |
| G2-05 | 故障注入与退化观测安全评测 | G2-04 |
| G2-06 | Safety 消融、Pareto 与阶段验收 | G2-01～G2-05 |

## 6. G3：轻量可验证驾驶 VLA

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G3-01 | VLA/VLT 数据契约、采集与版本登记 | G2-06 |
| G3-02 | ID/OOD/Regression 划分、泄漏审计与语言 grounding | G3-01 |
| G3-03 | 非语言模仿、行为与视觉轨迹基线 | G3-02 |
| G3-04 | 4080 约束下的轻量 Fast/Slow VLA 架构 | G3-03 |
| G3-05 | 多任务 SFT/VLT、checkpoint 与资源稳定性 | G3-04 |
| G3-06 | 风险校准、OOD、选择性驾驶和证据一致性 | G3-05 |
| G3-07 | VLA 闭环、Safety 接入、消融与阶段验收 | G3-01～G3-06 |

## 7. G4：场景覆盖、主动搜索与反事实

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G4-01 | 场景注册表、参数域、覆盖模型与有效性规则 | G3-07 |
| G4-02 | 工程/历史/系统生成场景套件与可解性验证 | G4-01 |
| G4-03 | Manual/Random/LHS/CMA-ES 同预算基线 | G4-02 |
| G4-04 | 覆盖引导 Quality-Diversity 主动搜索 | G4-03 |
| G4-05 | 可比重建、反事实分支、最小化与失败聚类 | G4-04 |
| G4-06 | 搜索净收益、复现性与阶段验收 | G4-01～G4-05 |

## 8. G5：动作条件交互世界模型

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G5-01 | 同场景多动作分支数据与 CV/CTRV/IDM/Reward 基线 | G4-06 |
| G5-02 | 轻量 object-centric/vector latent 交互动力学 | G5-01 |
| G5-03 | 多模态未来、风险/奖励/终止与不确定性 | G5-02 |
| G5-04 | 动作敏感性、因果 Actor、校准与 OOD 评价 | G5-03 |
| G5-05 | 候选排序、Active CARLA Verification 与安全降级 | G5-04 |
| G5-06 | 精度—收益—成本闭环验收 | G5-01～G5-05 |

## 9. G6：失败驱动安全后训练

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G6-01 | Safety Event 窗口、全链路数据谱系与隔离 | G5-06 |
| G6-02 | Hard Case Mining、去重、平衡与数据门禁 | G6-01 |
| G6-03 | Shadow DAgger、Takeover 与 Pre-intervention 数据 | G6-02 |
| G6-04 | 反事实 Preference、Risk Anticipation 与轻量后训练 | G6-03 |
| G6-05 | 历史失败、正常能力、OOD 与抗遗忘回归 | G6-04 |
| G6-06 | 发现—反事实—后训练—准出飞轮验收 | G6-01～G6-05 |

## 10. G7：可审计 Agentic Research Loop

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G7-01 | 白名单工具、权限、预算、审计与中断恢复 | G6-06 |
| G7-02 | Scenario Scientist Agent | G7-01 |
| G7-03 | Failure Analyst Agent | G7-02 |
| G7-04 | 确定性 Data Curator 与 Release Gate 工作流 | G7-03 |
| G7-05 | 固定脚本对照、grounding、红队与阶段验收 | G7-01～G7-04 |

## 11. G8：系统准出与成果证据

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G8-01 | 主张、最终协议、模型/数据/场景/代码资产冻结 | G7-05 |
| G8-02 | ID、常规、长尾与历史失败全系统回归 | G8-01 |
| G8-03 | OOD、故障、20ms 实时性、资源与长稳测试 | G8-01 |
| G8-04 | 消融、统计、异质性与 Pareto 分析 | G8-02、G8-03 |
| G8-05 | Evidence Bundle、演示、冷启动复现与简历追溯 | G8-02、G8-03、G8-04 |
| G8-06 | 最终审计、限制登记与版本准出 | G8-05 |

## 12. 阶段关闭规则

每个阶段关闭前必须同时满足：

1. 阶段内所有任务为 `COMPLETED`；
2. 阶段验收任务运行真实测试，而不是只检查文件存在；
3. 任务输出已登记代码、配置、数据、run、指标、资源和失败；
4. 所有未达标方法明确记录为负结果或适用边界；
5. Evidence Bundle hash/link/schema 自检通过；
6. `PROGRESS.md` 已更新；
7. 用户明确批准后，才可把下一阶段第一项设为 `CURRENT`。
