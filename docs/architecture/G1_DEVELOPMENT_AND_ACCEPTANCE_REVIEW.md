# G1 完整开发与验收 Review（Codex 复核版）

> **文档用途**：G1 阶段正式关闭说明；后续 G2 开发查阅。
> **证据原则**：只写已实现、已跑过、有磁盘证据的事实；**offline 与 live 分轨**；禁止假数据/硬编码收益。
> **修订日期**：2026-07-13（**正式关闭** v4.2）
> **工作分支**：`grok/g1-acceptance-repair`（**未** commit / **未** push）
> **阶段状态（`PROGRESS.md`）**：G1 **正式结束 `COMPLETED_WITH_LIMITS`**；G0 `FROZEN`；**未**启动 G2。

---

## 0. 阶段关闭摘要（先读）

### 0.1 一句话

G1 曾在弱测试下被标为 `COMPLETED`；经严格复核后修复 **ST-DP / RACE / reverse·deadline / 计量 / 任务标准** 等问题，并完成 **短 full dense live v4**。阶段已 **正式结束为 `COMPLETED_WITH_LIMITS`**，不是无限制的 VERIFIED 全绿。

### 0.2 判定表（关闭依据；勿沿用旧无限制 COMPLETED 叙事）

| 项 | 结论 | 权威证据 |
|---|---|---|
| G0 冻结 | **保持**，本阶段未改 G0 | — |
| 离线 `tests/g1` | **78/78 PASS**（含 live robustness helpers） | 本地 unittest 输出 |
| G1-04 ST-DP | **已修**：time-layered DP；禁强制 `v=0` 终点 | `evidence/g1-04/repair-20260713/` + `SUPERSEDED.md` |
| G1-05 RS/Hybrid | **已修**：端点校验、禁 snap、partial `ok=False` | `evidence/g1-05/repair-20260713/` |
| G1-06 控制 | **已修**：有符号 reverse；deadline/e2e 注入可 miss | `evidence/g1-06/repair-20260713/` |
| G1-07 RACE-Plan | **已修**：删除 `*0.7/*0.85`；raw 计数；flags 真改行为 | `evidence/g1-07/repair-20260713/` |
| G1-08 RACE-Control | **已修**：四 variant 真闭环；扰动从轨迹测得 | `evidence/g1-08/repair-20260713/` |
| G1-09 门禁 | **已修**：expert gate 真断言；repair summary | `evidence/g1-09/repair-20260713/` |
| Live **旁路** waypoint | 仍可用（`--stack waypoint`） | 历史 `g1-live-full-11-1783921932.json` 等 |
| Live **全链路** full | **已测通过（有限）**：Frenet+ST approach + **dense U-turn** + RaceControl+identify；**非** Hybrid/RS live | `g1-live/latest_success.json` v4（见 §5 / §11.4） |
| 「live 50Hz P99&lt;20ms VERIFIED」 | **否** | offline only |
| 「RACE 在 CARLA 有稳定 A/B 收益」 | **否** | offline admission only |
| Solver 是否 OSQP/SQP-RTI | **否** | 诚实标为 `constrained_gradient_ltv_bicycle` |
| 是否可开 G2 | 需**用户明确指令**；本 review 不自动开 | `START_TASK.md` |

### 0.3 两套测试对象（Codex 原核心观点 — 仍然成立，但 live 侧已升级）

| 对象 | 内容 | 状态 |
|---|---|---|
| **A. 集成 live** | CARLA 连接 + Runtime + 控车 | **通过**；现分 `waypoint` / `basic` / `full` |
| **B. 任务正文算法/消融** | 真 DP、真 RS、真 reverse/deadline、真 RACE | **已做验收修复**；以 `repair-20260713` 为准 |

旧错误：用 A（旁路）冒充 B。
现纠正：B 靠 offline repair；A 的 full stack 有独立 evidence，且 **claims 字段自报 modules**。

---

## 1. 阶段目标与边界

### 1.1 G1 目标

可审计经典专家：契约 → Runtime → 地图/路由/行为 → Frenet/ST → Hybrid/RS → 控制 → RACE → 双轨门禁；为 G2 导出轨迹/风险/控制基线。

### 1.2 硬边界

- CARLA Windows / 客户端 WSL2；固定单机资源。
- 禁止第二套 `carla.Client`/tick master、业务直接 `world.tick()`。
- G0 冻结。
- 禁止假数据、禁止用弱化测试掩盖失败。
- Oracle / Observable 分轨。

### 1.3 本轮授权

用户批准 plan 后：创建分支 `grok/g1-acceptance-repair`，连续执行 G1-04→09 验收修复；随后实现 full-stack live 并 CARLA 实测。默认不 commit/push。

---

## 2. 时间线（含修复）

| 阶段 | 内容 |
|---|---|
| T0–T3 | G1-01～03 契约、Runtime、地图/路由（大体保留） |
| T4–T6 | G1-04～09 初版实现（后经复核发现证据/算法缺陷） |
| T7 | CARLA **旁路** live：`plan_along_nodes` + ControlLoop（~603m 成功） |
| T8 | Codex 严格复核：指出 live≠全验收、RACE 假指标等 |
| T9 | **验收修复** G1-04～09 + 新 evidence `repair-20260713` |
| T10 | Live **`--stack full`**：Frenet/Hybrid/RaceControl 进环，~161m 到终点 |

---

## 3. 验收修复明细（B 轨）

### 3.1 G1-04 Frenet / ST-DP

| 项 | 内容 |
|---|---|
| 问题 | `solve_st_dp` 为 greedy；停车失败时 `append(v=0)` 造成功 |
| 修复 | `(t,s)` 时间层 DP，节点存连续 `v`；占用段检查；`ST_INFEASIBLE_STOP`；平滑后重积分；禁假终点 |
| 测试 | `tests/g1/test_g1_04_frenet.py`（含 ST 单测、运动学、障碍） |
| 证据 | `docs/architecture/evidence/g1-04/repair-20260713/` |
| config_hash | `cd31014def5f566d8be3beb99ea9d25620c46a2a688299d6fcd7b0604429f1ee`（toml 未改；实现已变） |

### 3.2 G1-05 Hybrid / Reeds–Shepp

| 项 | 内容 |
|---|---|
| 问题 | RS 扩展 snap 到 goal；partial 标 `ok=True` |
| 修复 | sample 端点 xy/yaw 容差校验；失败则丢弃 family；碰撞含末点；partial → `ok=False` + `PARTIAL_SOLUTION` |
| 测试 | endpoint property、no-snap、maneuver 套件 |
| 证据 | `docs/architecture/evidence/g1-05/repair-20260713/` |

### 3.3 G1-06 Control

| 项 | 内容 |
|---|---|
| 问题 | plant `v=max(0)`；deadline 只改 solver 字段；reverse 测不了 |
| 修复 | 有符号 plant + `ControlCommand.reverse`；`inject_solver_ms` / `inject_e2e_extra_ms` 进 e2e；watchdog steps==ticks |
| 诚实标注 | `solver_type=constrained_gradient_ltv_bicycle`（**非** OSQP） |
| 证据 | `docs/architecture/evidence/g1-06/repair-20260713/` |

### 3.4 G1-07 RACE-Plan + RiskField

| 项 | 内容 |
|---|---|
| 问题 | `candidates*0.7`、`nodes*0.85` 事后改写；风险可发散负值 |
| 修复 | **删除**假缩放；adaptive/coarse/multi_heuristic **真改**配置后重跑；`candidates_raw`/`nodes_raw`；RiskField 非负有限 + track 分轨 |
| Admission | 成功率 + work_ratio；可负结论 |
| 证据 | `docs/architecture/evidence/g1-07/repair-20260713/` |

### 3.5 G1-08 RACE-Control

| 项 | 内容 |
|---|---|
| 问题 | 四 variant 共用结果 + 硬编码 lateral/miss |
| 修复 | `RaceControlLoop` 分 variant；扰动 `steer_delay/gain_drift/noise` plant 注入；指标从轨迹计算 |
| 证据 | `docs/architecture/evidence/g1-08/repair-20260713/` |

### 3.6 G1-09 阶段门禁

| 项 | 内容 |
|---|---|
| 问题 | gate 只数 `ok`，不断言 comfort/timeout |
| 修复 | 真断言：规划 positives≥5、CTE&lt;5、非全 brake、oracle 非负分轨 |
| 证据 | `docs/architecture/evidence/g1-09/repair-20260713/summary.json` |

### 3.7 旧证据策略

```text
docs/architecture/evidence/g1-0X/
  SUPERSEDED.md              # 说明旧 summary 为何作废
  repair-20260713/           # 权威新证据
  summary.json               # 可指向 repair 的 pointer 副本
```

**禁止删除**历史 JSON；以 `repair-20260713` 为验收主路径。

---

## 4. 离线验证命令（Codex 可复跑）

```bash
cd "/mnt/e/autonomous driving"   # 或 WSL 等价路径
git branch --show-current        # 期望：grok/g1-acceptance-repair

PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g1 -t . -v
# 期望：OK（修复后曾记录 73 tests）

python3 -m compileall -q safedrive_foundry/classic_stack
```

抽查假数据是否已死：

```bash
rg "candidates \* 0\.7|nodes \* 0\.85|lateral_err.: 0\.4" safedrive_foundry/classic_stack || true
# 期望：无匹配（或仅 SUPERSEDED/文档提及）
```

---

## 5. CARLA Live（A 轨）— 分 stack 说明

### 5.1 入口

```bash
python3 scripts/sdf.py sim preflight   # 或 ensure
PYTHONPATH=safedrive_foundry python3 tests/g1/run_g1_classic_expert_live.py \
  --stack full --seed 11 --max-route-m 160 --hold-s 4
```

| `--stack` | 实际模块 | 用途 |
|---|---|---|
| `waypoint` | dense 航点定时 | 旧旁路演示；**不得**写成 Frenet/RACE 验收 |
| `basic` | Frenet+ST → ControlLoop | 基础全链路规划+控制 |
| **`full`** | Frenet+ST；U-turn 段 Hybrid+RS；**RaceControlLoop(full)** | 当前推荐的全链路演示 |

### 5.2 已测成功：full stack（权威 live 证据）

| 字段 | 值 |
|---|---|
| 文件 | `docs/architecture/evidence/g1-live/g1-live-full-11-1783946091.json` |
| 指针 | `docs/architecture/evidence/g1-live/latest_success.json` |
| schema | **`safedrive.g1.live_stack.v4`** |
| stack | **`full`**；`uturn_planner=dense`（默认贴路） |
| map | Town10HD_Opt |
| route | ≈ **181.5 m**（`--max-route-m 180` 短场景） |
| modules_used | `FrenetPlanner`, `ST-DP`, `waypoint_dense_uturn` |
| control | RaceControlLoop(full) → mpc；identify samples=2462 |
| claims | `frenet_in_loop=true`, `race_identification_active=true`, `uturn_dense_default=true`, `hybrid_in_loop=false`（诚实） |
| arrived | **true**；uturn_seen **true**；prog ≈ 174 m |
| 初始 plan | `planner=frenet_st`, `ok=true`, candidates=120（**非** 全程 waypoint bypass） |

### 5.3 历史旁路成功（对照，勿混用）

| 文件 | 说明 |
|---|---|
| `g1-live-full-11-1783921932.json` | 约 603 m；**waypoint dense + ControlLoop**；证明集成旁路 |
| 其它 `g1-live-complex-*` / `g1-live-expert-*` | 早期迭代 |

### 5.4 Live 限制（仍成立）

1. **无** live 50Hz P50/P95/P99 / deadline miss 正式门禁面板。
2. 单图 Town10HD；**禁止**默认 `load_world` 切图（shader fatal 风险）。
3. full 场景可截断（`--max-route-m`）；长跑去掉该参数即可，但不是已测 VERIFIED 全长。
4. Hybrid 主要在 **uturn replan** 进环；approach 以 Frenet 为主（claims 已记录 modules）。
5. 静态路边车/灯杆无法当 NPC 删除；贴路 + 跟踪质量规避。

---

## 6. 代码与配置地图

```text
safedrive_foundry/classic_stack/
  planning/speed/st_dp.py          # 真 DP（repair）
  planning/frenet/**               # 接 ST 失败码
  planning/hybrid_astar/**         # RS 端点 / partial
  planning/race/plan.py            # 无假缩放
  control/controller.py            # reverse + deadline e2e
  control/adaptation.py            # 真 variant 闭环
  risk/field.py                    # 非负有限风险
tests/g1/
  test_g1_04_frenet.py … test_g1_09_*.py
  run_g1_classic_expert_live.py    # --stack waypoint|basic|full
docs/architecture/evidence/
  g1-04..09/repair-20260713/
  g1-live/latest_success.json
```

---

## 7. 问题手册（高价值，供回归）

| 问题 | 解决 |
|---|---|
| 旁路 live 冒充全 G1 验收 | stack 分轨 + claims 字段 |
| RACE 事后改计数 | 删除；只报 raw |
| ST 假停车 | DP + 不可行失败码 |
| RS snap | sample 校验端点 |
| reverse plant | 有符号速度 + reverse 标志 |
| deadline 注入无效 | e2e 计入 inject |
| Risk 负发散 | gap≤0 高风险、分数 clamp |
| run_id 重复 / 有线无车 | 唯一 run_id；spawn 后画线 |
| load_world 卡死 | 禁止默认切图 |
| 轨迹 0.2s stale | 每 tick re-stamp |
| latest 被失败覆盖 | `latest_attempt` vs `latest_success` |

---

## 8. 已知限制（阶段关闭条件）

下列**不得**写成 VERIFIED 简历数字或“无限制完成”：

1. Solver **不是** OSQP/SQP-RTI。
2. **无** CARLA live 控制周期统计门禁。
3. RACE admission 来自 **offline** 配对场景，**无** live A/B 收益证明。
4. full live 已测 **短路线**；长 600m full stack 未作为本证据强制项。
5. 工作区大量改动在分支上，**未**合入 main / 未 push。

---

## 9. Codex 复核清单（请逐项勾）

### 9.1 立场

- [ ] 不以旧 `COMPLETED` 标签为准；以代码 + `repair-20260713` + 测试为准
- [ ] 区分 offline 修复 vs live full stack vs 旧 waypoint 旁路

### 9.2 假数据已死

- [ ] `rg` 无 `candidates * 0.7` / `nodes * 0.85`
- [ ] G1-08 disturbances 含 `raw_runs` / 实测 `lateral_err`（非写死 0.4/0.3）
- [ ] `st_dp.py` 无 “Force terminal stop samples”

### 9.3 离线

- [ ] `unittest discover -s tests/g1` 全绿
- [ ] G1-06 reverse：`signed_progress_x < 0`（evidence 或测试）
- [ ] G1-06 deadline：`inject_e2e_extra_ms` 路径 miss≥1
- [ ] G1-05 partial：`ok=False`

### 9.4 Live

- [ ] 读 `latest_success.json`：`stack=full`，`frenet_in_loop=true`
- [ ] 初始 plan_log 含 `planner=frenet_st` 而非仅 `waypoint_dense`
- [ ] 不把旁路 603m 文件当作 full stack 证据

### 9.5 冻结与流程

- [ ] G0 未改
- [ ] 无自动 G2
- [ ] 分支 `grok/g1-acceptance-repair`

---

## 10. 建议后续（非本交付必做）

1. 用户确认后可选：full stack **长路线**（去掉 `--max-route-m`）+ 多 seed。
2. 专项：live 50Hz watchdog 汇总 → 才可谈 live VERIFIED。
3. 用户指令：`执行G2-01，读取START_TASK.md。`
4. 入库：用户明确要求时再 commit（本 review 默认不提交）。

---

## 11. 收口修复（Codex P0/P1 + 撞墙/Stuck + 任务标准治理）

### 11.1 优先级与状态

| ID | 优先级 | 问题 | 状态 |
|---|---|---|---|
| A1 | P0 | Watchdog 每 tick 单次 `record` | **已修** + 测试钉 `steps==ticks` |
| A2 | P0 | full 无净收益不 promote | **已修**；g1-07 `promote_full=false` recommend p1 |
| A3 | P0 | `stable_seed` 替 `hash()` | **已修** |
| A4 | P0 | g1-05 partial ⇒ `hybrid_ok=false` | **已重生** repair evidence |
| A5 | P1 | live claims v4 + identify + RS 计数 | **已交付**；权威 `latest_success` = **v4**（`g1-live-full-11-1783946091`） |
| A6 | P1 | manifest / 披露 env diff | **g1-04～09 均含 manifest**；env 历史 diff 仅披露 |
| A7 | P1 | G1-09 单测不写正式目录 | **已修**（tempfile；`SDF_WRITE_G1_09_EVIDENCE=1`） |
| A8 | P1 | 任务文件删减原始完成标准 | **已恢复** G1-07～09 原文 + 追加证据映射（不得再以摘要替代标准） |
| B1 | P1 | uturn dense + 限速 + CTE 降油 | **已修** + live 验证 |
| B2 | P1 | StuckWatch 干净退出 | **已修** + 离线单测 |
| B3 | P2 | mesh 壳 / live 50Hz VERIFIED | **不做** |

### 11.2 离线验证（本轮实测）

```bash
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g1 -t . -v
# 2026-07-13：78 tests OK（含 3× StuckWatch/cruise helpers）
python3 -m compileall -q safedrive_foundry/classic_stack
```

### 11.3 Live 默认策略（诚实）

- `--uturn-planner dense`（默认）：贴路 U-turn，**不**伪装全程 Hybrid。
- `--stack full`：approach Frenet；RaceControlLoop(full)+identify；可选 hybrid uturn。
- Stuck → `failure=STUCK_NO_PROGRESS`，禁止无 failure 空转满 tick。

### 11.4 CARLA 复验（已通过）

| 项 | 值 |
|---|---|
| 恢复 | 代理结束僵死 `CarlaUE4` → `sdf sim ensure` **READY** → preflight **READY** |
| 命令 | `--stack full --seed 11 --max-route-m 180 --hold-s 4 --uturn-planner dense` |
| 结果 | **arrived=true**，uturn_seen=true，prog≈174m，route≈181.5m |
| 权威文件 | `docs/architecture/evidence/g1-live/g1-live-full-11-1783946091.json` |
| 指针 | `docs/architecture/evidence/g1-live/latest_success.json` |
| schema | **`safedrive.g1.live_stack.v4`** |
| run_id | `g1-live-full-11-1783946091` |
| modules | FrenetPlanner, ST-DP, waypoint_dense_uturn |
| control | RaceControlLoop(full)；`race_identification_samples=2462` |
| claims | `frenet_in_loop=true`，`race_identification_active=true`，`uturn_dense_default=true`，**`hybrid_in_loop=false`**（诚实：默认 dense uturn） |
| failure | null |

```bash
PYTHONPATH=safedrive_foundry python3 tests/g1/run_g1_classic_expert_live.py \
  --stack full --seed 11 --max-route-m 180 --hold-s 4 --uturn-planner dense
```

### 11.5 仍禁止写成 VERIFIED 的项

同 §8：非 OSQP；无 live 周期门禁；RACE 无 CARLA A/B；600m 零碰撞非强制；env 历史 diff 披露。

---

## 12. 结论

| 层级 | 判定 |
|---|---|
| Codex 原核心观点「旁路≠任务验收」 | **正确**；已用 repair + stack 分轨回应 |
| 离线算法/消融 + P0 计量 | **已交付**（`repair-20260713` + **78** tests） |
| CARLA full-stack 短场景 v4 | **已实测通过**（dense uturn 默认；RaceControl+identify） |
| 撞墙/Stuck 最小补丁 | **已落地并 live 验证**（本跑未 stuck；路径可 STUCK 干净退出） |
| 阶段关闭 | **正式结束：`COMPLETED_WITH_LIMITS`** |
| 无限制 / live 50Hz VERIFIED | **否** |
| G2 | **未启动**；需用户明确 `执行G2-01，读取START_TASK.md` |

**关闭声明**：G1 以限制清单正式结束。离线权威 `repair-20260713`；live 权威 v4 `latest_success`。任务 G1-01～03 为 `COMPLETED`；G1-04～09 为 `COMPLETED_WITH_LIMITS`。查阅清单见 §9 / §11；**无需**再为收口等待外部复核。

---

## 13. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-07-13 | 初版：开发过程 + 旁路 live（后被严格复核纠正） |
| 2026-07-13 | **v2 Codex 复核版**：验收修复明细、假数据清除、repair 证据索引、full-stack live 成功结果、限制与勾选清单 |
| 2026-07-13 | **v3 收口**：P0/P1 + stuck/uturn；78 tests；曾 CARLA BLOCKED_EXTERNAL |
| 2026-07-13 | **v4 关闭**：杀僵死 CARLA；short full dense live arrived；`COMPLETED_WITH_LIMITS` |
| 2026-07-13 | **v4.1 治理**：恢复 G1-07～09 原始完成标准并追加证据映射；修正 Hybrid live 不实表述与 v3 残留；补 g1-04 manifest |
| 2026-07-13 | **v4.2 正式结束**：PROGRESS/任务状态统一；去掉“待确认”；阶段指针清空 |

---

*本文件不替代任务原文与 evidence JSON；验收以可运行命令 + 磁盘文件为准。*
