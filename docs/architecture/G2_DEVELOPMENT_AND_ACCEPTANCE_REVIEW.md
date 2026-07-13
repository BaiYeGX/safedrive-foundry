# G2 完整开发与验收 Review（Safety Kernel）

> **文档用途**：G2 阶段（Independent Safety Kernel）开发过程、问题手册、验收与限制的正式说明；供后续 G3+ 与人工复核查阅。  
> **证据原则**：只写已实现、已跑过、有磁盘证据的事实；**offline 与 live 分轨**；禁止假数据/硬编码收益；负结果保留。  
> **修订日期**：2026-07-13  
> **工作分支**：`grok/g2-01-safety-contracts`（默认 **未** merge main / **未** push；以本地工作区为准）  
> **阶段状态（`PROGRESS.md`）**：G2 **`COMPLETED_WITH_LIMITS`（offline）**；G0 `FROZEN`；G1 `COMPLETED_WITH_LIMITS`；**不**自动启动 G3。

---

## 0. 阶段关闭摘要（先读）

### 0.1 一句话

G2 交付了**可独立运行的 Safety Kernel 库**：契约 → Validator → 状态机 → 纵向 QP → 受限 RATO-SCP → 仲裁/Shadow/降级 → 故障矩阵与阶段 Evidence。验收以 **offline CPU regression** 为主（`tests/g2` **111/111**）。收口后做了语义加固与 **Codex 复核修复**（NaN actor state-lock、identity 契约、QP/RATO deadline、故障矩阵补齐、evidence 门控）。**无** CARLA live 短闭环 VERIFIED，**无** 多节点 ROS 2 控制图接线。

### 0.2 判定表（关闭依据）

| 项 | 结论 | 权威证据 |
|---|---|---|
| G0 / G1 冻结边界 | 未改 G0；G1 算法本体未重构 | — |
| 离线 `tests/g2` | **111/111 PASS** | 本地 `unittest discover -s tests/g2` |
| G2-01 契约/Validator/SM | **COMPLETED** | `evidence/g2-01/` |
| G2-02 纵向 QP | **COMPLETED** | `evidence/g2-02/`（P50/P95 ≈ 8.4/10.2 ms） |
| G2-03 受限 RATO-SCP | **COMPLETED** | `evidence/g2-03/`（RATO P50/P95 ≈ 40.9/73.9 ms） |
| G2-04 仲裁/Shadow/回退 | **COMPLETED** | `evidence/g2-04/`（fresh kernel/场景） |
| G2-05 故障与阶段验收 | **COMPLETED_WITH_LIMITS** | `evidence/g2-05/`（13 故障；live `NOT_RUN`） |
| State hard 故障可否被 QP 覆盖 | **已修：state lock** | `test_stale_obs_full_tick_state_lock` |
| NaN/Inf Actor 可否 ACCEPT | **已修：actor_numeric state-lock** | `test_g2_codex_regressions` |
| Identity 错配可否 ACCEPT | **已修：HARD_REJECT** | pipeline + validator |
| soft_stale 是否误伤 Classic | **已修：仅学习源** | `test_classic_not_soft_stale` 等 |
| Final 是否只验 top-K | **已修：扫全 ranked** | `test_final_sweeps_beyond_topk` |
| Live CARLA 短闭环 | **未跑** | g2-05 `live_carla_short_loop=NOT_RUN` |
| 完整 ROS 2 多节点 | **未接线** | 仅 dict adapter 形状对齐 `sdf_interfaces` |
| CLAIMS C3 | offline **MEASURED**，非 live VERIFIED | g2-05 `claim_c3_offline` |
| 是否可开 G3 | 需**用户明确指令** | `START_TASK.md` |

### 0.3 两套对象（不要混）

| 对象 | 内容 | 状态 |
|---|---|---|
| **A. Offline Safety 语义** | 契约、硬校验、QP/RATO、仲裁、故障矩阵 | **已验收**（111 tests + evidence） |
| **B. Live / ROS 集成** | CARLA 闭环 + Safety 进控制环 + ROS 图 | **未交付**；G1 live 存在但**未**调用 `SafetyKernel` |

旧错误（审查发现）：用 A 的弱测试掩盖状态锁失败。  
现纠正：A 已加固；B 不得用 A 冒充。

---

## 1. 阶段目标与边界

### 1.1 G2 目标

独立 Safety Kernel：学习模块**不能**覆盖硬约束、slack 上界、回退权限或紧急动作；输出裁决/修复/回退，而不是直接无约束控车。为 G3 VLA 候选预筛与最终裁决留接口。

任务链（`START_TASK.md` / `tasks/G2/`）：

| ID | 名称 | 主交付 |
|---|---|---|
| G2-01 | Safety contracts + Validator + 状态机 | 类型契约、预筛/全检、NORMAL→…→EMERGENCY |
| G2-02 | Longitudinal QP repair | path 固定的速度/加速度修复 |
| G2-03 | Restricted RATO-SCP | 有走廊时的横向二次修复 |
| G2-04 | Arbitration / Shadow / Fallback | hard→soft→final→repair→fallback→shadow |
| G2-05 | Fault matrix + stage acceptance | 故障注入、模式对照、Evidence |

### 1.2 硬边界

- 固定单机：RTX 4080 16GB / i5-13600KF；训练与 CARLA 不按双卡算。
- CARLA Windows / ROS·客户端 WSL2；业务禁止第二 `carla.Client`/tick master、禁止直接 `world.tick()`。
- Runtime **只吃 Observable**；Oracle 仅 offline 标签路径。
- Classic + Safety + MPC/PID 在学习全关时仍须可闭环（逻辑层已保证；live 未接）。
- G0 冻结；G1 只读除非明确修复。

### 1.3 本轮授权与实现形态

- 分支：`grok/g2-01-safety-contracts`。
- 实现形态：**纯 Python 库** `safedrive_foundry/safety_kernel/` + 版本化 `config/safety_kernel/baseline.toml` + `tests/g2/`。
- ROS：仅 **msg 形状 dict adapter**（不建 Node、不 `rclpy` spin）。
- 默认不 commit/push，以用户 Git 策略为准。

---

## 2. 时间线

| 阶段 | 内容 |
|---|---|
| T0 | G1 正式关闭 `COMPLETED_WITH_LIMITS`；分支准备 G2 |
| T1 | **G2-01**：契约/schema hash、Validator、状态机、Kernel facade、G1 轨迹回放、ROS dict adapter |
| T2 | **G2-02**：纵向 reduced-space QP；OSQP 优先 + scipy SLSQP/ADMM 回退；统一 Raw/Rule/HardReject/Longitudinal 接口 |
| T3 | **G2-03**：Frenet 走廊 + 受限 SCP；QP 优先、RATO 二级触发；禁用 RATO 时 QP 独立可用 |
| T4 | **G2-04**：仲裁管线、soft score、降级门、Classic Shadow（compare-only）、fallback |
| T5 | **G2-05**：故障矩阵、学习关闭回归、阶段 Evidence；阶段标 `COMPLETED_WITH_LIMITS` |
| T6 | **严格审查**：发现 state 硬故障可被 QP 覆盖、soft_stale 误伤 Classic、final 仅 top-K、弱测试等 |
| T7 | **语义修复 + 回归加严** → **98 tests**；g2-05 evidence 重测；本 Review 文档 |

---

## 3. 分任务交付明细

### 3.1 G2-01 — 契约、Validator、状态机

| 项 | 内容 |
|---|---|
| 交付 | `contracts/`（类型+序列化+schema hash）、`validator/`（prefilter/full/state）、`state_machine/`、`kernel.py` facade、`oracle_offline.py`、G1 轨迹 adapter |
| 硬检查 | privilege、schema、numeric、freshness、time_order、road、dynamics、collision、rules、trackability |
| 状态 | NORMAL / DEGRADED / MINIMAL_RISK / EMERGENCY；EMERGENCY 立即升级；其余 debounce/dwell/hysteresis |
| 测试 | `test_g2_01_*.py` |
| 证据 | `docs/architecture/evidence/g2-01/` |
| 时延（MEASURED） | state P50/P95 ≈ 0.011 / 0.014 ms（n=220）；cand P50/P95 ≈ 1.79 / 1.91 ms（n=221）；deadline miss=0（`evidence/g2-01/summary.json`） |

### 3.2 G2-02 — 纵向 QP

| 项 | 内容 |
|---|---|
| 交付 | `repair/longitudinal_qp.py`、`qp_solver.py`、`baselines.py`、`interface.py` |
| 自由度 | path `(x,y,yaw,κ)` 固定；优化 `v/a/jerk` + 有界 slack |
| 场景 | 红灯、跟车、cut-in、急刹动力学、stale/不可行诚实失败 |
| 后端 | `osqp_auto`：python-osqp（含 `tools/wsl_site_packages`）→ scipy SLSQP → numpy ADMM |
| 测试 | `test_g2_02_*.py` |
| 证据 | `docs/architecture/evidence/g2-02/` |
| 时延（MEASURED） | QP P50/P95 ≈ **8.41 / 10.20 ms**（n=24，deadline 50 ms，miss=0；`evidence/g2-02/summary.json`） |

### 3.3 G2-03 — 受限 RATO-SCP

| 项 | 内容 |
|---|---|
| 交付 | `repair/rato_scp.py`、`corridor.py`；Kernel 级联 QP→RATO |
| 触发 | 合法横向走廊 +（侧向相关 reject **或** QP 失败/进度过低） |
| 门控 | 无走廊 / 纯红灯 hints → 不直接 RATO；`rato.enabled=false` 时 QP 独立 |
| 测试 | `test_g2_03_*.py` |
| 证据 | `docs/architecture/evidence/g2-03/` |
| 时延（MEASURED） | RATO P50/P95 ≈ **40.90 / 73.91 ms**（n=8，deadline 100 ms，miss=0；`evidence/g2-03/summary.json`） |

### 3.4 G2-04 — 仲裁 / Shadow / 回退

| 项 | 内容 |
|---|---|
| 交付 | `arbitration/{pipeline,soft_score,degradation,shadow,types}.py` |
| 顺序 | state → degrade → prefilter → soft score → rank → **final（全 ranked）** → repair → fallback → shadow |
| Soft 规则 | 永不覆盖 hard reject / Emergency / state lock |
| Shadow | `claims_control=False`，`claims_tick_ownership=False` |
| 测试 | `test_g2_04_*.py` |
| 证据 | `docs/architecture/evidence/g2-04/` |

### 3.5 G2-05 — 故障矩阵与阶段验收

| 项 | 内容 |
|---|---|
| 交付 | `faults/matrix.py`；阶段 Evidence 写盘测试 |
| 矩阵（9 类） | stale_obs、packet_drop、out_of_order、localization_bias、missed_actor、actor_offset、solver_stale、numeric_nan、**vision_soft_degrade** |
| 每故障字段 | start/duration/severity/seed/expected_action/recovery |
| 模式对照 | Raw / Rule / HardReject / Longitudinal / RATO |
| 测试 | `test_g2_05_*.py` |
| 证据 | `docs/architecture/evidence/g2-05/`（含负结果） |

**故障矩阵实测摘要（g2-05/summary.json，offline）：**

| fault_id | decision_kind | 备注 |
|---|---|---|
| stale_obs | **MINIMAL_RISK** | state lock；禁止 QP 覆盖（修复后） |
| packet_drop_actor | ACCEPT | lost track 允许无碰撞时通过 |
| out_of_order_time | HARD_REJECT | |
| localization_bias | QP | 可能修复/硬失败路径 |
| missed_actor | ACCEPT | **负结果**：Observable 漏检可漏风险 |
| actor_offset | QP | |
| solver_stale_candidate | HARD_REJECT | 修复器拒 stale |
| numeric_nan | HARD_REJECT | |
| vision_soft_degrade | QP 等 | meta/cov，无 Oracle |

---

## 4. 架构与数据流

### 4.1 逻辑闭环（设计）

```text
ObservationBundle (Observable only)
  → Classic / VLA 产生 PolicyCandidateSet
  → [可选] prefilter
  → Soft score + 仲裁排序（软，可关）
  → Final Validator 硬检
  → 失败则 Longitudinal QP →（可选）RATO-SCP
  → Fallback / MINIMAL_RISK / EMERGENCY
  → Classic Shadow（只对比）
  → （目标）MPC/PID → Vehicle
```

当前实现：`SafetyKernel.tick(obs, candidate_set)` **库内**完成上图至 Shadow；**未**接到 G1 live 控制环或 ROS 节点。

### 4.2 State Floor Lock（语义加固后）

```text
check_state → hard_violations?
  YES 且 floor ∈ {MINIMAL_RISK, EMERGENCY}:
      不跑候选校验 / soft / QP / RATO
      decision = state decision + state_locked_no_candidate_override
      无 executed_trajectory_id
  NO:
      正常仲裁管线
state_machine.step(..., state_floor=...)  # 纵深防御，防止 ACCEPT/QP 把 mode 拉回 NORMAL
```

### 4.3 模块地图

```text
safedrive_foundry/safety_kernel/
  contracts/          # types, schema, serialize
  validator/          # checks, engine
  state_machine/      # mode machine
  repair/             # QP, RATO, corridor, baselines, solver
  arbitration/        # pipeline, soft_score, degradation, shadow
  faults/             # offline fault matrix
  adapters/           # G1 trajectory + ROS msg-shaped dicts
  kernel.py           # SafetyKernel facade
  config.py           # TOML loader
  oracle_offline.py   # Oracle 仅 offline
  metrics.py
safedrive_foundry/config/safety_kernel/baseline.toml
tests/g2/             # 98 offline tests
docs/architecture/evidence/g2-01 .. g2-05/
```

### 4.4 配置冻结

- 路径：`safedrive_foundry/config/safety_kernel/baseline.toml`
- 学习模块**不可**运行时改硬约束 / slack 上界 / deadline
- soft_stale：`soft_stale_age_s` **仅 VLA**；Classic 用 `max_candidate_age_s` 硬门槛

---

## 5. 问题手册（开发与审查中遇到的问题 → 解法）

### 5.1 安全语义（P0，审查后修复）

| 问题 | 现象 | 解法 |
|---|---|---|
| **State 硬故障被 QP 覆盖** | stale obs 时 state 已 MR，但仲裁仍 QP 成功，mode 回 NORMAL | `kernel.tick` **state floor lock**；硬 margin 时禁止候选路径与修复 |
| **soft_stale 误伤 Classic** | age∈(0.20,0.25] Classic 被 degrade，再被 RATO「救回」 | `apply_quality_gates`：soft 门仅 `VLA_*`；Classic 只走硬 freshness |
| **弱测试掩盖问题** | `test_stale_obs` 含 `decision_kind is not None` 恒真 | 改为 full-tick 断言：禁止 ACCEPT/QP/RATO；要求 lock reason |
| **Evidence 与 expected 不一致** | g2-05 写 stale→QP | 修代码后重跑写盘；assert evidence 行 |

### 5.2 仲裁与修复（P1）

| 问题 | 现象 | 解法 |
|---|---|---|
| **Final 仅 top-K** | 高分非法占满 K 后先昂贵 repair，合法低分延后 | Final **按 soft 序扫全 ranked**；K 仅审计「主窗口」；note `final_sweep_beyond_topk` |
| **EMERGENCY 仍 validate 候选** | 浪费时延、污染 latency 统计 | state lock / EMERGENCY 短路，不调 `validate_candidates` |
| **RATO 与 QP 职责不清** | 怕变成第二规划器 | 二级触发 + 走廊门控 + 纯纵向不抢 RATO；可关闭 |

### 5.3 工程与求解（实现期）

| 问题 | 现象 | 解法 |
|---|---|---|
| **无系统 OSQP** | 环境可能无 `osqp` | 多层后端 + `tools/wsl_site_packages` 可选；后端标签诚实 |
| **QP 维数/等式** | 完整动力学等式难解 | reduced-space：`v0 + a[]` 映射 `s,v`，减少约束块 |
| **假成功停车** | 无约束时无意义刹停 | `unjustified_stop` + `min_progress_ratio` 判失败 |
| **Slack 无限** | 软约束假装安全 | `slack_*_max` 上界写死配置 |
| **Oracle 泄漏** | runtime 误用特权 | privilege 硬拒；`oracle_offline` 独立路径 |
| **G1 轨迹 horizon** | 样本可 >5s | config `max_horizon_s=6` 兼容 G1 样本（注释标明） |
| **ROS conf=0 on QP** | 有可执行修复轨迹却 conf=0 | ACCEPT=1.0 / QP=0.9 / RATO=0.85 |
| **私有 API 耦合** | pipeline 写 `_latencies_*` | 公开 `emit_decision_event` / `record_candidate_latency_ms` |

### 5.4 建模与验收边界（诚实保留，非「修没」）

| 问题 | 处理 |
|---|---|
| 碰撞 = 等速预测 + 圆包络 | docstring + g2-05 limits 登记；**不**伪称 polygon 碰撞 |
| 红灯 = 近距接近速度门 | 与 Validator/QP 对齐；非完整停线规划 |
| Missed-actor | Observable 可漏 → **负结果**登记，不引入 Oracle 补检 |
| 无 live 50Hz 面板 | 限制写清；不得写 VERIFIED 控制周期 |
| 无 ROS 控制图 | adapter only；完整 ROS 属后续集成 |

### 5.5 与 G1 的接口问题

| 问题 | 解法 |
|---|---|
| G1 轨迹 JSON → Safety | `adapters/g1_trajectory.py`：`g1_plan_result_to_candidate_set` |
| Live runner 未调用 Safety | **已知缺口**；同进程 glue / ROS 节点为后续工作（见 §9） |
| 导入路径 | 测试 `sys.path.insert(safedrive_foundry)`，包名 `safety_kernel`（与 G1 `classic_stack` 同模式） |

---

## 6. 语义修复清单（T7，对应审查 Plan）

| ID | 严重度 | 改动落点 | 回归测试 |
|---|---|---|---|
| S1/K1 | P0 | `kernel.py` state lock；`state_machine.step(state_floor=)` | `test_stale_obs_full_tick_state_lock`；`test_state_emergency_skips_candidates` |
| S2 | P0 | `arbitration/degradation.py` | `test_classic_not_soft_stale`；`test_learning_soft_stale`；`test_classic_age_022_accepts_without_repair` |
| A1 | P1 | `arbitration/pipeline.py` final 全 ranked | `test_final_sweeps_beyond_topk` |
| T1 | P1 | 删除恒真断言；加 full-tick | 上列 + g2-05 suite |
| R1 | P2 | `adapters/ros_safety_status.py` | `test_qp_ros_confidence_positive` |
| V1 | P2 | `validator/engine.py` 公开 API | 调用方无 SLF001 写 latency |
| F1 | P2 | `faults/matrix.py` VISION_SOFT_DEGRADE | matrix 9 项 |
| E1 | P2 | 重写 `evidence/g2-05/` | evidence 内 assert stale 非 QP |
| L1 | P2 | docstring / README / PROGRESS | 限制字段 |

---

## 7. 离线验证命令（可复跑）

```bash
cd "/mnt/e/autonomous driving"   # 或 WSL 等价路径

# 全量 G2
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g2 -t . -v
# 期望：Ran 98 tests ... OK

python3 -m compileall -q safedrive_foundry/safety_kernel tests/g2

# 语义修复抽查
PYTHONPATH=safedrive_foundry python3 -m unittest \
  tests.g2.test_g2_05_fault_matrix.G205FaultMatrixTests.test_stale_obs_full_tick_state_lock \
  tests.g2.test_g2_04_arbitration.G204DegradationTests.test_classic_not_soft_stale \
  tests.g2.test_g2_04_arbitration.G204KernelPipelineTests.test_final_sweeps_beyond_topk \
  tests.g2.test_g2_01_kernel_integration.G201KernelIntegrationTests.test_state_emergency_skips_candidates \
  -v
```

抽查 evidence 中 stale 不得再被 QP 盖掉：

```bash
python3 -c "
import json
from pathlib import Path
s=json.loads(Path('docs/architecture/evidence/g2-05/summary.json').read_text())
row=next(r for r in s['fault_matrix'] if r['fault_id']=='stale_obs')
assert row['decision_kind'] in ('MINIMAL_RISK','EMERGENCY','HARD_REJECT'), row
assert row['decision_kind'] not in ('ACCEPT','QP','RATO'), row
print('stale_obs OK', row['decision_kind'])
"
```

**不需要** CARLA / ROS 2 即可完成当前 G2 offline 验收。

---

## 8. Evidence 索引

| 轨 | 路径 | schema（summary） | 状态 |
|---|---|---|---|
| G2-01 | `docs/architecture/evidence/g2-01/` | `safedrive.g2_01.evidence.v1` | MEASURED offline |
| G2-02 | `docs/architecture/evidence/g2-02/` | `safedrive.g2_02.evidence.v1` | MEASURED offline |
| G2-03 | `docs/architecture/evidence/g2-03/` | `safedrive.g2_03.evidence.v1` | MEASURED offline |
| G2-04 | `docs/architecture/evidence/g2-04/` | `safedrive.g2_04.evidence.v1` | MEASURED offline |
| G2-05 | `docs/architecture/evidence/g2-05/` | `safedrive.g2_05.evidence.v1` | MEASURED offline + limits |

各目录通常含：`summary.json`、`manifest.json`、`README.md`。  
G2-01 summary 的 limits 仍可写「当时尚无 QP/RATO」——属**阶段快照**；全阶段以 g2-05 + 本 Review 为准。

**配置/契约 hash（evidence 重测后一致样本）：**

- `contracts_schema_hash`：`8e97e3f10e9e7004d2b1b9d16da40233c901a5805f171fe7cbcda02b0fca2d72`
- `config_hash`：以最新 `baseline.toml` 的 sha256 为准（evidence summary 内字段）

---

## 9. Live / ROS：现状与后续怎么做（非本阶段完成项）

### 9.1 现状

| 能力 | 状态 |
|---|---|
| G1 CARLA live | **已有**（`tests/g1/run_g1_classic_expert_live.py`） |
| Live 内调用 `SafetyKernel` | **无** |
| ROS bridge 状态话题 | 骨架（JSON String） |
| Classic/Safety **ROS 控制节点** | **无** |

### 9.2 若要「看 G2 上真车流」（建议最小路径）

**同进程 glue（优先于完整 ROS 图）：**

```text
G1 live 控制环
  → 候选轨迹转 PolicyCandidateSet
  → CARLA ObservableSnapshot（非 Oracle）
  → SafetyKernel.tick
  → ACCEPT/QP/RATO 轨迹进 ControlLoop
  → MR/EMERGENCY → 停车/最小风险，不发无约束控制
  → 写 docs/architecture/evidence/g2-live/
```

完整 ROS 多节点属于更大集成任务，**不能**用 offline 98 tests 冒充。

### 9.3 当前验收口径

- **需要 CARLA？** 验 offline G2：**不需要**。  
- **需要 ROS 2？** 验 offline G2：**不需要**。

---

## 10. 已知限制（阶段关闭条件）

下列**不得**写成无限制完成或 live VERIFIED 简历数字：

1. 全阶段为 **offline CPU**；**无** CARLA live 50Hz / 短闭环 VERIFIED。  
2. CLAIMS C3 仅 offline MEASURED。  
3. 碰撞 CV+圆包络；红灯近距速度门。  
4. Missed-actor Observable 漏检为负结果。  
5. RATO/仲裁个别样本可能 deadline miss（以 evidence 为准，如实保留）。  
6. Shadow 仅对比，不控车、不占 tick。  
7. 未合入 main / 未 push（以实际 Git 状态为准）。  
8. 未接 ROS 控制图；未进 G1 live runner。

---

## 11. 复核清单（人工 / 代理）

### 11.1 立场

- [ ] 不以「有 90+ tests」代替语义审查  
- [ ] 区分 offline Safety vs G1 live vs 未来 ROS 图  

### 11.2 语义

- [ ] stale obs full tick **非** ACCEPT/QP/RATO  
- [ ] Classic age 0.22 **不被** soft_stale  
- [ ] max_final=1 时 hard-legal 低分仍可在 **repair 前** 被接受  
- [ ] Shadow `claims_control=False`  

### 11.3 命令

- [ ] `unittest discover -s tests/g2` → **111/111 OK**  
- [ ] g2-05 summary `stale_obs.decision_kind` ∈ {MINIMAL_RISK, EMERGENCY, HARD_REJECT}  

### 11.4 冻结与流程

- [ ] G0 未改  
- [ ] 无自动 G3  
- [ ] 负结果未删除  

---

## 12. 与 G1 Review 的衔接

| G1 提供 | G2 如何用 |
|---|---|
| Frenet/ST、Hybrid、Control、Risk 轨迹 | 适配为 `PolicyCandidate`；回放测 ACCEPT |
| Live full stack | **尚未**接入 Safety；是下一集成目标 |
| 诚实限制文化 | G2 同样 offline/live 分轨、负结果登记 |

G1 权威文档：`docs/architecture/G1_DEVELOPMENT_AND_ACCEPTANCE_REVIEW.md`。

---

## 13. Codex 独立复核修复（2026-07-13）

Codex 在 `tests/g2` 全绿后仍判验收未通过。本轮**只修 G2**，未启动 G3，未改 G0/G1 业务代码，未 commit/push。

| ID | 问题 | 修复 | 回归 |
|---|---|---|---|
| P0-1 | `TrackedObject` NaN/Inf 绕过 collision → ACCEPT | `actor_numeric` 状态硬门 + state-lock；collision 防御 | `test_g2_codex_regressions` |
| P0-2 | set/obs 身份字段错配仍 ACCEPT | Validator + 仲裁 pipeline 入口硬校验 | 同上 |
| P0-3 | OSQP/后端超 deadline 仍 success | 全 backend `_enforce_deadline` → TIMEOUT、不返回解 | 极小 deadline 测试 |
| P1-1 | RATO 部分迭代后 TIMEOUT→SOLVED_INACCURATE | 任意 timeout `success=False` | RATO timeout 测试 |
| P1-2 | 故障矩阵缺低附着/饱和/solver/model timeout | `DEFAULT_MATRIX` 13 项 + `expected_action_holds` | `test_g2_05_fault_matrix` |
| P1-3 | g2-04 多场景共用 kernel 污染 | 每场景新 `SafetyKernel` + degradation audit 断言 | `test_g2_04_latency_evidence` |
| P2 | Evidence 笼统 n/a、无 hash/git | `evidence_util` + `SDF_WRITE_G2_EVIDENCE=1` 门控重建 | g2-01～05 evidence |

**命令结果**：`unittest discover -s tests/g2` → **111/111 OK**；`compileall` OK；`git diff --check` OK。  
**Evidence**：已用门控重建 g2-01～05（manifest 含 artifact/config/contracts hash、command、git）。  
**仍未完成**：live CARLA 短闭环、C3 live VERIFIED、多节点 ROS 控制图。

---

## 14. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-07-13 | 初版：G2-01～05 交付、问题手册、语义修复 T7、证据索引、限制与复核清单 |
| 2026-07-13 | Codex 复核修复：P0/P1/P2；tests **111/111**；evidence 门控重建；G2-05 仍 `COMPLETED_WITH_LIMITS` |

---

*本文档不构成启动 G3 的授权。启动下一阶段请使用：`读取 START_TASK.md，启动 G3-01。`*
