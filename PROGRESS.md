# 项目当前进度

> 本文件只记录动态执行状态。任务编号、名称和依赖以 `ROADMAP.md` 为唯一来源；任务范围、验收和断点以对应任务文件为准。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前阶段 | G2：**COMPLETED_WITH_LIMITS（offline）** |
| 当前任务 | **G2-05 COMPLETED_WITH_LIMITS**；G2 离线阶段关闭 |
| 当前状态 | **G2-01～G2-04 COMPLETED**；**G2-05 COMPLETED_WITH_LIMITS**（无 live CARLA 短闭环） |
| 最近完成 | Codex G2 复核修复（NaN actor / identity / deadline / RATO timeout / 故障矩阵 / evidence 门控）；`tests/g2` **111/111** |
| 推荐下一任务 | 仅当用户明确指令：`读取 START_TASK.md，启动 G3-01。`（或授权 live 补验 G2-05） |
| 最近更新 | 2026-07-13 |
| 工作分支 | `grok/g2-01-safety-contracts`（未 merge main；未 push） |

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| G0 | `COMPLETED / FROZEN` | 任务/代码/历史 evidence 冻结 |
| G1 | **`COMPLETED_WITH_LIMITS`** | 正式关闭 |
| G2 | **`COMPLETED_WITH_LIMITS`** | 离线 Safety Kernel 全链路完成；无 live 50Hz/短闭环 VERIFIED |
| G3 | `PENDING` | 5 项；统一数据、非语言多候选、轻量 Fast/Slow VLA 与闭环 |
| G4 | `PENDING` | 5 项 |
| G5 | `PENDING` | 5 项 |
| G6 | `PENDING` | 5 项 |
| G7 | `PENDING` | 3 项 |
| G8 | `PENDING` | 5 项 |

## G2 任务状态一览

| 任务 | 状态 |
|---|---|
| G2-01 | **`COMPLETED`** |
| G2-02 | **`COMPLETED`** |
| G2-03 | **`COMPLETED`** |
| G2-04 | **`COMPLETED`** |
| G2-05 | **`COMPLETED_WITH_LIMITS`**（offline fault matrix；无 live 短闭环） |

## 当前阻塞与决策

**无阻塞（offline 范围）。** Live CARLA 短闭环未跑 → G2-05 限制保留。

**不**自动启动 G3、**不**自动 merge/push main。

### G2 阶段限制（诚实）

1. 全阶段验收为 **offline CPU regression**，**不是** CARLA live 50Hz / 短闭环 VERIFIED。
2. G2-05 CLAIMS C3 为 offline MEASURED；live 补验需用户授权 + `sdf sim preflight`。
3. Missed-actor Observable 漏检为已登记负结果。
4. Shadow 仅对比，不控制车辆、不占 tick。
5. 碰撞为 CV+圆包络；红灯为近距速度门（非完整停线规划器）。
6. 状态硬故障 **state lock**：不得在不可信观测上 ACCEPT/QP/RATO。

### 证据指针

| 轨 | 路径 |
|---|---|
| G2-01 | `docs/architecture/evidence/g2-01/` |
| G2-02 | `docs/architecture/evidence/g2-02/` |
| G2-03 | `docs/architecture/evidence/g2-03/` |
| G2-04 | `docs/architecture/evidence/g2-04/` |
| G2-05 | `docs/architecture/evidence/g2-05/` |

### 实测摘要

| 项 | 结果 |
|---|---|
| **`unittest discover -s tests/g2`** | **111/111 OK** |
| G2-02 QP latency | P50/P95 ≈ **8.41 / 10.20 ms**（n=24，deadline 50ms，miss=0） |
| G2-03 RATO latency | P50/P95 ≈ **40.90 / 73.91 ms**（n=8，deadline 100ms，miss=0） |
| G2-04 arbitration | P50/P95 ≈ **3.12 / 12.50 ms**（n=5；fresh kernel/场景；degradation audit） |
| G2-05 fault matrix | **13** 类故障 + expected_action 校验；tick P50/P95 ≈ 14.0 / 51.5 ms（n=13）；live `NOT_RUN` |

## 最近更新

| 日期 | 变更 |
|---|---|
| 2026-07-13 | **Codex G2 复核修复**：P0 NaN actor state-lock、identity 契约硬校验、QP deadline→TIMEOUT；P1 RATO timeout 不执行、故障矩阵补齐、g2-04 场景隔离；P2 evidence 资源/hash/git/门控；`tests/g2` **111/111**；G2-05 仍 `COMPLETED_WITH_LIMITS`。 |
| 2026-07-13 | **G2 语义修复**：state floor lock、soft_stale 仅 VLA、final 扫全 ranked、ROS conf、公开 event/latency API、VISION_SOFT_DEGRADE 入矩阵。 |
| 2026-07-13 | **G2 离线收口**：G2-04 仲裁/Shadow/回退 + G2-05 故障矩阵；阶段 `COMPLETED_WITH_LIMITS`。 |
| 2026-07-13 | **G2-03 COMPLETED**：受限 RATO-SCP + Frenet 走廊 + Kernel 级联；evidence `g2-03/`。 |
| 2026-07-13 | **G2-02 COMPLETED**：纵向 QP 修复；evidence `g2-02/`。 |
| 2026-07-13 | **G2-01 COMPLETED**：契约/Validator/状态机；分支 `grok/g2-01-safety-contracts`。 |
| 2026-07-13 | **G1 正式结束**：`COMPLETED_WITH_LIMITS`。 |

## 下一动作

- G2 offline 停止点已到。
- 下一任务口令：`读取 START_TASK.md，启动 G3-01。`
- 若需 G2-05 live 短闭环补验：先启动 CARLA，再 `sdf sim preflight` 后明确授权。
