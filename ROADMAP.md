# SafeDrive Foundry 总路线（G0 冻结后重构版）

> 总需求以 [FUTURE_PROJECT_VISION.md](./FUTURE_PROJECT_VISION.md) 为准。本文件只定义 G0～G8 的执行顺序、阶段依赖与关闭门禁。G0 已验收冻结，本次重构不修改 G0 产物。

## 1. 项目主线

项目围绕一个问题展开：一个人能否在单张 RTX 4080 16GB 上，本地完成“轻量驾驶 VLA + 独立 Safety Kernel + 动作条件 World Model + 一轮困难样本后训练”，并在 **SIL 仿真** 中稳定演示。

作品路径成功口径（详见 `docs/project/PROJECT_SUCCESS_PROFILE.md`）：**VLA 与 World 都必须真实接入**；**稳定可跑优先于指标**；效果可负但须诚实；**不上实车**。

```text
G0 环境与确定性（已冻结）
  ↓
G1 统一运行时与经典专家
  ↓
G2 Independent Safety Kernel
  ↓
G3 轻量可验证驾驶 VLA
  ↓
G4A 固定场景、可比动作分支与 oracle best-of-K（科学标注选择空间）
  ↓
G5 World-V0（本项目必做接入；弱选择空间仍实现，C2 可负）
  ↓
G6 一轮困难样本监督适配（增益可负）
  ↓
G8 系统准出与成果证据（演示配置含 VLA+World+Safety）

G4B 主动搜索/最小化、G7 Agentic Research Loop：Optional / After Release
```

G0～G6、G8 是发布主线；G4B/G7 不阻塞发布。任何学习模块未优于简单基线时，必须记录负结果，不得靠删除 World 或伪造指标绕过。

跨阶段时间、身份、接口、Oracle/Observable、Profile、权限和Registry约束只在[执行架构](./docs/project/EXECUTION_ARCHITECTURE.md)维护；G2～G8 的 VLA/World/Safety 运行链、资源与降级边界见[工业化架构规范](./docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md)；具体实现分别见[SDF-VLA-1B 设计](./docs/project/SDF_VLA_1B_DESIGN.md)、[World-V0 门禁与设计](./docs/project/SDF_WORLD_MODEL_DESIGN.md)和[VLA主线+可选World总体架构](./docs/project/SDF_VLA_WORLD_SYSTEM_ARCHITECTURE.md)；Windows CARLA 与 WSL 模型共享 4080 的准入、互斥 workload 和降级规则见[单机执行预算](./docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md)；通用完成规则只在`AGENTS.md`维护；最终研究主张见[CLAIMS](./docs/project/CLAIMS.md)。

VLA 是必须完成的主线：G3 依次完成 V0/V1 和无 Classic 当前帧候选闭环，G4A 再比较 top-1 与 oracle best-of-K。G4A 输出的选择空间标签决定 World 收益主张强弱，但不取消作品要求的 G5 World-V0 实现；没有稳定选择空间时仍接入并保存负结论。二者始终位于 Validator/Safety Kernel 约束内。

## 2. 阶段与任务数量

| 阶段 | 任务数 | 阶段目标 | 关闭条件摘要 |
|---|---:|---|---|
| G0 | 6 | 环境、连接、同步与证据基线 | 已验收冻结 |
| G1 | 9 | 统一运行时、经典专家、风险自适应规划与鲁棒实时控制 | 基础/优化专家在 Oracle/Observable 轨道均可闭环，算法净收益与数据门禁可审计 |
| G2 | 5 | 独立 Validator、两级修复、仲裁回退与故障验收 | 学习模块全失效时 Classic+Safety 仍可安全闭环 |
| G3 | 5 | VLA-V0、VLA-V1、可选 V2 与闭环 | K1/K2、2.5s 合同；VLA+Safety 无 Classic 当前帧候选闭环 |
| G4 | 5 | G4A 固定场景/replay/可比分支；G4B 可选搜索 | G4A 与 oracle best-of-K 必须；G4B 不阻塞 |
| G5 | 5 | 4M～8M World-V0 必做接入 | 科学标签记录选择空间强弱；无净收益仍保留模块与 on/off；C2 可负 |
| G6 | 5 | 一轮困难样本监督 adapter/LoRA | 困难切片改善且正常集/Safety/时延无明显退化 |
| G7 | 3 | Optional / After Release | 不作为项目完成或 G8 依赖 |
| G8 | 5 | 协议冻结、统一回归、统计证据和最终准出 | 四个发布配置可复现，所有对外主张可追溯 |

G2～G8 共 33 项，G1～G8 共 42 项，全项目共 48 项。任务按可形成独立运行能力和证据的垂直切片拆分；训练、在线评测与大规模回归按单机 workload 串行化，不把多 GPU 或外部服务器当作隐含条件。

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
| G2-02 | RATO 纵向 QP 最小修复基线 | G2-01 |
| G2-03 | 受限二维 RATO-SCP 与可行走廊 | G2-02 |
| G2-04 | 候选预筛、World-ready 仲裁、Shadow 与回退链 | G2-03 |
| G2-05 | 故障注入、Observable 退化与 Safety 阶段验收 | G2-01～G2-04 |

## 6. G3：轻量可验证驾驶 VLA

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G3-01 | 场景身份、VLA 数据契约、冻结划分与泄漏审计 | G2-05 |
| G3-02 | Route/Ego、V0 K1 与 V1 K2 公平基线 | G3-01 |
| G3-03 | VLA-V0：F0 checkpoint 门禁、上游接入、canonicalizer 与最小闭环 | G3-02 |
| G3-04 | VLA-V1 最小增强；V2 FAST/REASON 为递进 optional | G3-03 |
| G3-05 | VLA+Safety 无 Classic闭环与 Hybrid 工程验收 | G3-01～G3-04 |

### 2026-07-19 发布检查点

pure SimLingo VLA path/speed → constrained MPC → CARLA 已达到
`MEASURED_WITH_LIMITS`，并完成官方输入契约、DX12 D3D 隔离、自动冷启动和长测证据。
这只是 G3 核心驾驶能力检查点；正式 G3-05 仍须补 VLA+Safety 的新 live evidence，不能把
stable runner 的 `DEMO_PASS` 写成阶段 `VERIFIED`。当前统一入口见
`docs/architecture/G3_VLA_MPC_RELEASE_GUIDE.md`。

用户若希望先准备下一阶段，只能明确授权 G4-01/G4-02 以 `PRE_G3_CLOSE` 实现场景
Registry/固定 replay；它不改变 G4 对 G3-05 的正式依赖。

## 7. G4：场景覆盖、主动搜索与反事实

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G4-01 | 场景注册表、核心长尾套件、覆盖与可解性 | G3-05 |
| G4-02 | G4A 固定 seed/replay、20～40 困难场景与可恢复执行 | G4-01 |
| G4-03 | G4B：MAP-Elites/自动搜索（Optional） | G4-02 |
| G4-04 | G4A 可比 K2 动作分支与 oracle best-of-K；最小化/聚类 optional | G4-02 |
| G4-05 | G4A/World 入口门禁验收；G4B 单独标记 | G4-01、G4-02、G4-04 |

## 8. G5：动作条件交互世界模型

本项目 G5 **必做**（作品完整性）。G4-05 输出的选择空间标签只约束 C2 表述强度，不取消 G5 实现。

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G5-01 | 登记 WE0/科学标签；冻结 K2 branches 与 CV/CTRV/Reward 基线 | G4-05 |
| G5-02 | World-V0：K2/T10/N8/M1、4M～8M | G5-01 |
| G5-03 | actor future、collision/TTC/off-road、action/no-action 检验 | G5-02 |
| G5-04 | 两候选软排序、5Hz worker 与退化 VLA+Safety | G5-03 |
| G5-05 | 相同 VLA/Safety 的 World 开关闭环 A/B | G5-01～G5-04 |

## 9. G6：失败驱动安全后训练

G6 硬依赖 G4-05。G5 若已 `ENTER_WORLD` 并完成，其结论可作为 hard-case 附加输入；若 `SKIPPED_BY_GATE` 或未执行，G6 仍可独立完成一轮监督适配。第一轮只允许单个监督 adapter/少量 LoRA，禁止 PPO/GRPO/多种 preference/多轮自动飞轮。

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G6-01 | VLA 失败前 2～4 秒窗口、谱系与数据门禁 | G4-05 |
| G6-02 | Shadow 专家修正采集与 corrective 谱系 | G6-01 |
| G6-03 | 单个监督 adapter 或少量 LoRA checkpoint | G6-02 |
| G6-04 | 历史失败/正常/OOD 抗遗忘回归 | G6-03 |
| G6-05 | 失败驱动数据飞轮阶段验收 | G6-01～G6-04 |

## 10. G7：可审计 Agentic Research Loop（Optional / After Release）

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G7-01 | 白名单工具、权限、预算、审计与中断恢复 | G6-05 |
| G7-02 | Scenario/Failure Research Assistant 与证据 grounding | G7-01 |
| G7-03 | 确定性 Data/Release 工作流、脚本对照与红队验收 | G7-01、G7-02 |

## 11. G8：系统准出与成果证据

| ID | 能力包 | 直接依赖 |
|---|---|---|
| G8-01 | 核心主张、最终协议、代码/模型/数据/场景资产冻结 | G6-05；不依赖 G7 |
| G8-02 | 四发布配置的 ID/长尾/OOD/故障/实时性统一回归 | G8-01 |
| G8-03 | 核心消融、统计、异质性与 Pareto | G8-02 |
| G8-04 | Evidence Bundle、演示与冷启动复现 | G8-02、G8-03 |
| G8-05 | 最终审计、限制登记和版本准出 | G8-04 |

## 12. 阶段关闭规则

每个阶段关闭前必须同时满足：

1. 主线阶段内所有必做任务为 `COMPLETED` 或 `COMPLETED_WITH_LIMITS`；G5 不得以 `SKIPPED_BY_GATE` 逃避实现；G4B/G7 可为 `OPTIONAL_NOT_RUN`；
2. 阶段验收任务运行真实测试，而不是只检查文件存在；
3. 任务输出已登记代码、配置、数据、run、指标、资源和失败；
4. 所有未达标方法明确记录为负结果或适用边界；
5. Evidence Bundle hash/link/schema 自检通过；
6. `PROGRESS.md` 已更新；
7. 用户明确批准后，才可把下一阶段第一项设为 `CURRENT`。
