# 项目统一进度

> 本文件是当前执行状态的唯一入口。每次任务结束必须更新；任何时刻只能有一个 `CURRENT` 任务。

## 当前状态

| 字段 | 当前值 |
|---|---|
| 当前阶段 | G0：环境与确定性 |
| 当前任务 | G0-05 确定性同步与环境诊断 |
| 当前任务状态 | BLOCKED_EXTERNAL |
| 阶段状态 | IN_PROGRESS |
| 最近更新 | 2026-07-12 |
| 推荐下一任务 | 不推荐；先完成 G0-05 的 WSL/CARLA/ROS 现场门禁并恢复本任务 |

## 状态枚举

- `PENDING`：尚未开始。
- `CURRENT`：当前唯一允许执行的任务。
- `IN_PROGRESS`：已开始但因对话或额度中断，必须按断点继续。
- `BLOCKED`：存在明确阻塞，需先解决阻塞。
- `BLOCKED_EXTERNAL`：需要用户完成管理员、GUI、重启、登录或其他外部步骤后才能继续。
- `COMPLETED`：完成标准和验证方法均已通过。

## G0 环境与确定性

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G0-01 | 环境盘点与版本冻结 | COMPLETED | 无 |
| G0-02 | WSL2、GPU 与 ROS 2 基础环境 | COMPLETED | G0-01 |
| G0-03 | CARLA Server 安装与独立验证 | COMPLETED | G0-01, G0-02 |
| G0-04 | 工程骨架与 CARLA–ROS 跨系统连通 | COMPLETED | G0-02, G0-03 |
| G0-05 | 确定性同步与环境诊断 | BLOCKED_EXTERNAL | G0-04 |
| G0-06 | G0 集成验收与证据包 | PENDING | G0-01～G0-05 |

## G1 经典闭环专家

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G1-01 | 地图、全局路线与行为层 | PENDING | G0-06 |
| G1-02 | 双局部规划器 | PENDING | G1-01 |
| G1-03 | 速度规划与底层控制 | PENDING | G1-02 |
| G1-04 | 经典专家集成验收 | PENDING | G1-03 |

## G2 安全优化与故障治理

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G2-01 | 安全契约与轨迹校验器 | PENDING | G1-04 |
| G2-02 | RATO 风险感知轨迹优化器 | PENDING | G2-01 |
| G2-03 | 安全门控、仲裁与故障降级 | PENDING | G2-02 |
| G2-04 | 安全层闭环验收 | PENDING | G2-03 |

## G3 轻量 VLA 语义驾驶

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G3-01 | VLA 数据与评测流水线 | PENDING | G2-04 |
| G3-02 | 轻量 VLA 架构与接口 | PENDING | G3-01 |
| G3-03 | VLA 监督微调与开环评测 | PENDING | G3-02 |
| G3-04 | VLA 闭环接入与验收 | PENDING | G3-03 |

## G4 反事实场景搜索

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G4-01 | 场景注册表与搜索基线 | PENDING | G3-04 |
| G4-02 | CMA-ES 反例搜索 | PENDING | G4-01 |
| G4-03 | 反事实验证与失败最小化 | PENDING | G4-02 |
| G4-04 | 搜索系统闭环验收 | PENDING | G4-03 |

## G5 世界模型风险预测器

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G5-01 | 动作条件数据集与世界模型基线 | PENDING | G4-04 |
| G5-02 | 交互式潜空间世界模型 | PENDING | G5-01 |
| G5-03 | 风险头与主动仿真验证 | PENDING | G5-02 |
| G5-04 | 世界模型闭环验收 | PENDING | G5-03 |

## G6 反事实数据飞轮与策略改进

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G6-01 | 困难样本窗口与数据门禁 | PENDING | G5-04 |
| G6-02 | 影子专家与 DAgger 聚合 | PENDING | G6-01 |
| G6-03 | 反事实偏好与风险前瞻后训练 | PENDING | G6-02 |
| G6-04 | 数据飞轮闭环验收 | PENDING | G6-03 |

## G7 Agent 自动化与可审计研究闭环

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G7-01 | Agent 工具、权限与审计治理 | PENDING | G6-04 |
| G7-02 | 场景科学家与反例搜索 Agent | PENDING | G7-01 |
| G7-03 | 失败分析与数据策展 Agent | PENDING | G7-02 |
| G7-04 | Agent 净收益与发布门禁 | PENDING | G7-03 |

## G8 系统级验证与成果交付

| ID | 任务 | 状态 | 依赖 |
|---|---|---|---|
| G8-01 | 最终实验协议与资产冻结 | PENDING | G7-04 |
| G8-02 | 全系统回归、OOD 与实时性评测 | PENDING | G8-01 |
| G8-03 | 消融、统计与 Pareto 分析 | PENDING | G8-02 |
| G8-04 | 证据包、演示与最终验收 | PENDING | G8-03 |

## 阻塞问题

- G0-03 已完成：`E:\CARLA_0.9.16` 与 `carla0916` Conda/Python 3.12 匹配 wheel 下，四轮独立 smoke、world/tick、错误端口和进程清理均通过；证据见 `docs/environment/CARLA_SERVER_BASELINE.md`。
- G0-04 已完成：Ubuntu-24.04、ROS 2 Jazzy、`carla==0.9.16` Linux client 和 `safedrive_carla_bridge` 已构建；通过 `172.30.80.1:2000` 读取 Windows CARLA world，并两次收到 `/safedrive/carla/status` 消息。错误端口、版本不匹配和清理门禁均已验证；证据见 `docs/environment/CARLA_ROS_CONNECTIVITY.md`。
- G0-01 无阻塞，已完成。
- G0-02 已完成：WSL 2.9.3.0、Ubuntu 24.04.4、RTX 4080、PyTorch CUDA、ROS 2 Jazzy、talker/listener、`ros2 doctor` 和最小 colcon 工作区均已验证；证据见 `docs/environment/WSL_ROS2_BASELINE.md`。
- 非阻塞环境项：WSL 的 `systemd-binfmt.service` 失败；Mihomo/TUN Fake-IP 曾使 GitHub 返回 403；`sdf` 密码保持锁定。它们不影响已验证的 GPU、ROS 2 和构建基线。
- ROS 2 Jazzy 与 CARLA ROS 接口缺少逐版本官方矩阵，须在 G0-04 用真实连通与同步测试验证。
- `E:` 当前可用约 102.67 GiB，安装和数据生成前需设置容量门禁；不得删除用户文件。
- G0-05 代码与离线验证已完成：固定步长/子步、唯一 tick master、frame contract、重复/缺帧/过期检测、双跑比较、checkpoint 恢复、doctor 报告和 CARLA/ROS sync driver 均已实现；`validate-g0` 14/14 通过。
- G0-05 当前外部阻塞：`sdf doctor` 于 2026-07-12 识别 `wsl_distribution_unavailable`（无可用注册发行版）、`ros2_blocked_by_wsl`、`clock_observation_blocked`，以及 `172.30.80.1:2000` `carla_not_started`；现场 `/clock` 与 snapshot/ROS 消息对齐尚未验证，不能标记 `COMPLETED`。

## 已完成决策

- 硬件基线：RTX 4080 Desktop 16GB、i5-13600KF，单机无集群。
- 项目形态：纯软件在环 SIL。
- 原始总需求 `FUTURE_PROJECT_VISION.md` 保持不随意修改。
- G0～G8 均按适合单次 Codex 对话的宏任务预拆分，共 38 项；执行时不跨任务、不自动开始下一项。
- 后续任务可在其正式开始前依据已完成阶段的证据做范围校准，但不得静默改写总需求或验收目标。
- G0 首选版本组合冻结为 Windows 11 + WSL2 Ubuntu 24.04 + ROS 2 Jazzy + CARLA/ScenarioRunner 0.9.16 + Python 3.12 + PyTorch 2.12.1/cu126。
- G0 回退版本组合冻结为 WSL2 Ubuntu 22.04 + ROS 2 Humble + CARLA/ScenarioRunner 0.9.15 + Python 3.10 + PyTorch 2.7.1/cu126；现有 `E:\CARLA` 0.9.15 保留不覆盖。

## 更新日志

| 日期 | 变更 |
|---|---|
| 2026-07-12 | 执行 G0-05：实现固定步长/子步与唯一 tick master、`episode_id+carla_frame` 契约、重复/缺帧/过期检测、可恢复离线烟雾、CARLA/ROS sync driver、`sdf doctor` 及 JSON/Markdown 证据；离线验证 14/14、双跑和恢复通过。现场 doctor 发现 WSL 无可用发行版且 CARLA 未启动，按规则标记 `BLOCKED_EXTERNAL`，等待外部门禁后恢复；未开始 G0-06。 |
| 2026-07-12 | 完成 G0-04：创建 `safedrive_foundry` 骨架与最小 bridge；WSL Ubuntu 24.04/ROS 2 Jazzy 使用 `172.30.80.1` 连接 Windows CARLA 0.9.16，ROS topic 重复收发、错误端口、版本不匹配和清理验证通过；未开始 G0-05。 |
| 2026-07-12 | 根据四轮 `carla0916` smoke 证据，将 G0-03 更正为 `COMPLETED`；早期错误 Python 环境导致的 `UnicodeDecodeError` 不再作为 CARLA 兼容性结论。用户明确要求继续 G0-04，完成骨架后因当前会话无注册 WSL 发行版暂停为 `BLOCKED_EXTERNAL`。 |
| 2026-07-11 | G0-03 纯 ASCII 路径复测：CARLA 移至 `E:\CARLA_0.9.16` 后版本和地图 RPC 仍通过，`get_world` 仍报相同非 UTF-8 错误；已排除安装路径空格，关闭后无进程/端口残留，保持 `BLOCKED_EXTERNAL`。 |
| 2026-07-11 | G0-03 暂停为 `BLOCKED_EXTERNAL`：安装并核验 CARLA/Python API 0.9.16，版本、地图清单、端口、负向错误和进程清理通过；`get_world/load_world` 被非 UTF-8 服务端错误阻断，未完成 tick 与两轮 smoke，未开始 G0-04。 |
| 2026-07-11 | 完成 G0-02：安装并复测 WSL2/Ubuntu、RTX 4080 CUDA、PyTorch、ROS 2 Jazzy 与 colcon；重启 WSL 后 GPU 和 ROS 通信再次通过；记录 DDS、域 ID、QoS、Python 和工作区约定。未开始 G0-03。 |
| 2026-07-11 | G0-02 恢复：安装并验证 WSL 2.9.3、Ubuntu 24.04.4、Python 3.12.3 和 RTX 4080 透传；ROS 官方引导包下载因平台 approval credits 耗尽被拒，保持 `BLOCKED_EXTERNAL`。 |
| 2026-07-11 | G0-02 外部步骤更新：管理员 PowerShell 的 Ubuntu 24.04 下载返回 HTTP 403；按 Microsoft 官方路径改用 `--web-download`，任务继续保持 `BLOCKED_EXTERNAL`。 |
| 2026-07-11 | 启动 G0-02 preflight；确认固件虚拟化/SLAT 已启用，但 WSL 功能查询与安装需要当前会话无法获得的管理员令牌，任务标记 `BLOCKED_EXTERNAL`，等待用户完成最小外部步骤后恢复。 |
| 2026-07-11 | 完成 G0-01：只读主机盘点、官方兼容性核对、首选/回退版本冻结；等待用户明确指定 G0-02。 |
| 2026-07-11 | 根据粒度反馈，将 G0 由 10 项合并为 6 项，并指定 G0-01 为当前任务。 |
| 2026-07-11 | 按最新指令预拆分 G1～G8，每阶段 4 项；全项目形成 38 个可续接宏任务。 |
