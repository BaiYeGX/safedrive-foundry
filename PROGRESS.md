# 项目当前进度

> 本文件只记录动态执行状态。任务编号、名称和依赖以 `ROADMAP.md` 为唯一来源；任务范围、验收和断点以对应任务文件为准。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前阶段 | G1：**已关闭**（`COMPLETED_WITH_LIMITS`） |
| 当前任务 | **无**（G1-01～G1-09 全部关闭；不自动开始 G2） |
| 当前状态 | **`COMPLETED_WITH_LIMITS`**（正式结束） |
| 最近完成 | G1 阶段收口：算法验收修复 + 短 full dense live v4 + 任务标准治理 |
| 推荐下一任务 | 仅当用户明确指令：`执行G2-01，读取START_TASK.md。` |
| 最近更新 | 2026-07-13 |
| 工作分支 | `main`（已 fast-forward 合并 `grok/g1-acceptance-repair`，tip `f711b8b`，已 push `origin/main`） |

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| G0 | `COMPLETED / FROZEN` | 任务/代码/历史 evidence 冻结；工作区或有历史 env 文档 diff（不纳入 G1 关闭条件） |
| G1 | **`COMPLETED_WITH_LIMITS`** | **正式关闭**；见下方限制与证据指针 |
| G2 | `PENDING` | 6 项；**未**启动 |
| G3 | `PENDING` | 7 项 |
| G4 | `PENDING` | 6 项 |
| G5 | `PENDING` | 6 项 |
| G6 | `PENDING` | 6 项 |
| G7 | `PENDING` | 5 项 |
| G8 | `PENDING` | 6 项 |

## G1 任务状态一览

| 任务 | 状态 |
|---|---|
| G1-01 | `COMPLETED` |
| G1-02 | `COMPLETED` |
| G1-03 | `COMPLETED` |
| G1-04 | `COMPLETED_WITH_LIMITS` |
| G1-05 | `COMPLETED_WITH_LIMITS` |
| G1-06 | `COMPLETED_WITH_LIMITS` |
| G1-07 | `COMPLETED_WITH_LIMITS` |
| G1-08 | `COMPLETED_WITH_LIMITS` |
| G1-09 | `COMPLETED_WITH_LIMITS`（阶段验收） |

## 当前阻塞与决策

**无阻塞。** G1 已正式结束。本轮已 commit 并 push 至 GitHub；**不**自动启动 G2、**不**自动 merge `main`。

### G1 限制（诚实，关闭后仍有效）

1. 控制 solver 为 **constrained gradient LTV bicycle**，**不是** OSQP/SQP-RTI。
2. **无** live 50Hz P50/P95/P99 正式 VERIFIED 门禁。
3. RACE admission **仅 offline**；无 CARLA A/B 收益证明。
4. 长 600m full **零碰撞非 VERIFIED**（权威 live 为短 ~182m + dense uturn）。
5. 权威 live **`hybrid_in_loop=false`**（dense uturn 默认；**不**作 Hybrid/RS live 证明）。
6. 权威 live **`no_traffic` / `junctions=0`**；红灯/切入/路口让行等 **未 live VERIFIED**（见 G1-09 映射表）。
7. 工作区含历史 `docs/environment/*` 等 diff → **披露、不纳入关闭条件**。

### 证据指针

| 轨 | 路径 |
|---|---|
| Offline repair | `docs/architecture/evidence/g1-0{4..9}/repair-20260713/`（含 manifest） |
| Live 权威 v4 | `docs/architecture/evidence/g1-live/latest_success.json`（`g1-live-full-11-1783946091`） |
| 阶段 Review | `docs/architecture/G1_DEVELOPMENT_AND_ACCEPTANCE_REVIEW.md` |
| 阶段验收任务 | `tasks/G1/G1-09_CLASSIC_EXPERT_ACCEPTANCE.md` |

### 实测摘要（关闭依据）

| 项 | 结果 |
|---|---|
| `unittest discover -s tests/g1` | **78/78 OK** |
| Live | `stack=full`，`uturn_planner=dense`，seed=11，route≈181.5m，**arrived**，schema **v4** |
| 任务标准 | G1-07～09 原文保留；未达标项写入限制，非删标准 |

## 最近更新

| 日期 | 变更 |
|---|---|
| 2026-07-13 | **G1 正式结束**：状态统一为 `COMPLETED_WITH_LIMITS`；指针清空；文档对齐。 |
| 2026-07-13 | 任务标准治理：恢复 G1-07～09 原文 + 证据映射；Review 修正；g1-04 manifest。 |
| 2026-07-13 | 短 full dense live v4 arrived；P0/P1 收口；78 tests 绿。 |
| 2026-07-13 | G1-04～09 严格验收修复（分支 `grok/g1-acceptance-repair`）。 |
| 2026-07-12 | G1-01/G1-02 完成；G0 冻结。 |

## 下一动作

- **G1 已关闭**，无进行中任务。
- GitHub：`origin/main` @ `f711b8b`（G1 已合入 main）；功能分支 `origin/grok/g1-acceptance-repair` 仍保留。
- 下一阶段：仅当用户指令 `执行G2-01，读取START_TASK.md。`
