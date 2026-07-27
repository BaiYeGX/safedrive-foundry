# 当前任务：修复 Spatial R2-X 的真实选择空间

**状态指针（2026-07-27）**：

```text
R2  = COMPLETED_WITH_LIMITS / NO_SELECTION_SPACE          # 纵向，只读
R2X = COMPLETED_WITH_LIMITS
      / NOMINAL_TOWN12_60S_PASS
      / DEFENSIVE_AVAILABILITY_UNRELIABLE
      / WORLD_DEVELOPMENT_ONLY
      / WORLD_GATE_NOT_MET
R3  = NOT_AUTHORIZED
```

## 本轮目标

修复造成 spatial candidate 1 全部 `NO_ALTERNATIVE` 的标签、模型和
R2-K provenance 根因；不改写冻结的 v4 pilot，不靠降低 0.50 m 门或 runtime
模板制造选择空间。

## 已完成

| 项 | 当前事实 |
|---|---|
| R2-K provenance | 新 manifest 可绑定 spatial policy/head/config；失败行不再伪装成 V1 candidate |
| Teacher Guard | defensive candidate 必须通过 exact Guard；Guard reject 不再 fail-open |
| Teacher legal pool | 先逐候选执行 0.50 m/progress/comfort/direction/improvement 合同，再在合法集合内排序 |
| Teacher identity | `safedrive.k2_spatial_teacher.v4` / `spatial_defensive_lattice_v4_legal_pool` |
| v9 development | 74 条；train/val=52/22；available=29；subfloor available=0；episode leakage=false |
| v9 head | Guard 8/8、eligible sep 7/8、availability recall 7/8；specificity 11/14，未过 0.80 |
| proposal semantics | learned confidence 非阻塞；candidate validity 由 diversity + Guard + PM/MPC 决定 |
| offline v3 | eligible proposal valid 7/8；eligible Guard 8/8；`head_status=OK` |
| development live | 3/3 PASS；cut-in/crossing dual-force 2/3；所有 branch MPC solved、0 collision |
| Guard diversity repair | 只用相同 Frenet-s 上的 `max\|d0-d1\|`；native excursion 独立诊断 |
| nominal slot | candidate 0 绑定 exact native anchor；learned nominal geometry 仅审计 |
| v10 development head | eligible Guard 8/8；sep/proposal-valid 7/8；defensive speed gap 加强 |
| paired cold start | SpeedPlanner 从实测 ego speed 初始化，不再从 0 m/s 把两分支锁成满刹 |
| MPC feasibility | 速度收紧 `a_y` 转角门时保留物理可达 recovery envelope；OSQP infeasible 不再落入慢 SLSQP |
| MPC tail/验收 | tracker OSQP 禁用 polishing；development smoke 必须 `timeout=0`，不再漏计 |
| v12 development head | diversity floor 改为逐样本 hinge；低学习率续训，eligible Guard/sep/proposal-valid = 8/8、7/8、7/8 |
| post-repair CARLA | lead/crossing 2/2 dual-force PASS；4 branch 共 MPC 200/200，0 timeout/fallback/collision |

## 当前结论

1. v4 formal 的 `IMPROVE_VLA` 仍是有效冻结结果。
2. 旧 teacher 会让 d=0/低于 0.50 m/Guard reject candidate 污染监督；已修复且版本化。
3. v9 已证明 head 可以输出可执行的合法左右 residual；learned confidence 的空路
   specificity=0.786 继续保留为非阻塞诊断。
4. v12 checkpoint 仅允许 `development_live_smoke`，仍禁止 formal/X5H/R2-K。
5. lead-brake 与 crossing 已证明真实闭环选择空间；这不能覆盖冻结 v5，也不能单独授权 R3。
6. crossing 原 timeout 已定位为瞬时不可达硬约束 + fallback 误分类；修复后同一 Evidence
   离线重放 100/100 OSQP solved，CARLA v12 smoke 100/100 solved。
7. 正式下一步是建立全新 blind registry；
   不能复用已看过结果的 v2 blind registry。
8. 2026-07-27 复验已修复正式入口：post-v1 registry 可使用全新 6 个
   scenario ID；promotion 必须绑定 frozen blind registry hash、offline v3
   gates，并验证训练集与 blind `(scenario_id, seed_id)` 零重叠；formal X5H
   必须显式传 frozen registry manifest 与 1–3 个预声明 pair。
9. CARLA 一次 `ensure` 后恢复 `READY`；本轮只创建 development smoke
   Evidence，没有创建或消耗新的 blind Evidence。

## 冻结 v5 正式终态（不得改写）

1. 新 blind registry 已 dry-run 12/12 并冻结，训练 pair 重叠为 0。
2. formal X5H v8：`PASS_WITH_LIMITS`，2/3 dual-force；唯一失败为 Guard
   `SPATIAL_COLLAPSE_ELIGIBLE`。
3. formal R2-K v5：12/12 已执行；8 comparable、8 TIE、4 incomparable。
4. 18 个实际 branch × 50 ticks；MPC 900/900，0 fallback、0 collision、
   0 offroad。
5. 冻结门 `comparable>=10` 未过，且 decisive=0，因此 `WORLD_GATE_NOT_MET`。
6. 按“获得一个可用模型、不继续过度强化”的用户目标，R2-X 工程以
   `COMPLETED_WITH_LIMITS` 结束；formal pilot 的 `PILOT_INCONCLUSIVE /
   REPAIR_REQUIRED` 原始报告保持不改写。

## 停止点

- R2/R2-X 工程按“获得一个可用模型、不继续过度强化”目标正式停止；
  nominal 可用，defensive availability 限制必须保留。
- 不降低 0.50 m；不改 Oracle；不使用 runtime rescue。
- 不覆盖 `r2-spatial-k2-pilot-v4-formal/`。
- 当前 development smoke 不得冒充 formal X5H/R2-K。
- 不启动 R3；不 commit/push（未要求）。
- 当前停止于 post-v5 development repair；v12 不是 formal checkpoint。
- 下一轮用全新 blind registry 做 formal X5H/R2-K；不得复用 v2 blind outcome。
- 不启动 R3；lead-brake 的单个 decisive development pair 只能证明修复方向，
  不能单独推导 World 有效。
