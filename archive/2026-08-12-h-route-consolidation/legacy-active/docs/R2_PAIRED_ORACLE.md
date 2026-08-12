# R2 / G4A 实施方案：Paired Outcome + Oracle Best-of-K

**状态**：`CLOSED / COMPLETED_WITH_LIMITS`（2026-07-24；纵向 K2 最终结论）

12-pair live pilot comparable = **11/12**；pilot 主标签 **`NO_SELECTION_SPACE`**
（11×TIE）。repeat 2-pair 标签一致。实验、审计与关闭文档已完成。

本结论仅针对**纵向** K2（同 native path + 不同 speed）。
空间/语义双头扩展见 **`docs/R2X_SPATIAL_K2.md`（R2-X）**，使用**新** Evidence
目录，**禁止**覆盖或重新解释本文关闭结论。

**不得**自动开始 R3、World 数据或 World runtime。

**前置状态**：R1 真实 K2 已 `COMPLETED_WITH_LIMITS`。最新 R1 Evidence：
`docs/runtime-evidence/r1-real-k2-guard-fix2-2026-07-24/`。

R2 Evidence：`docs/runtime-evidence/r2-g4a-paired-pilot/`（含 immutable
`run_set_manifest`、`repeat_audit_plan` 冻结于 outcome 前、`r2_closure_report`）。

本文是 R2 的实施权威。项目总合同见 `PROJECT.md`，长期 World 设计见
`WORLD_MODEL.md`，当前停止点仍以根目录 `START_TASK.md` 为准。

## 1. R2 要回答的问题

R2 只回答：

> 对同一个 Observable Observation、同一次真实 SimLingo forward 产生的
> `v1_nominal / v1_conservative`，在可复现的相同 CARLA 初始状态下分别执行，
> 两条候选是否产生可比较、可重复、足以定义 oracle best-of-K 的闭环结果差异？

R2 不回答：

- World 是否能预测或选择该差异；
- 哪个候选普遍更安全；
- learned K2 是否优于 deterministic longitudinal K2；
- Safety live、15 m/s 高速或实车安全；
- World 的训练集规模、网络结构或 runtime latency。

R2 的成功不是“candidate 1 获胜”。以下结果都允许：

- candidate 0/1 在不同场景各有优势；
- 选择空间弱；
- candidate 0 几乎总是最优；
- 两候选经常都失败；
- 部分 pair 因 CARLA/同步/初始状态不一致而不可比。

负结果和不可比结果必须保留。

## 2. 当前代码能力与缺口

### 2.1 可直接复用

R1 已提供：

```text
NeuralV1Policy.predict_bundle()
  → 一次真实 forward
  → K2PredictionBundle
  → validate_k2_bundle()
  → select_k2(top1|force 0|1)
  → VLASpeedPlanner + VLAPathManager + ConstrainedVLAMPC
```

`tests/g3/run_g3_vla_mpc_stable.py` 已实测并可复用：

- 单一 CARLA client/tick owner；
- official camera / SimLingo 输入；
- force candidate 0/1；
- generated/selected/executed/source_id 绑定；
- collision episode；
- lane invasion；
- off-road fraction；
- route progress；
- 20 Hz control/MPC solver trace；
- latency、VRAM、failure 和 cleanup Evidence。

`safedrive_foundry/runtime/scenario_runtime.py` 已有：

- `ScenarioSpec / ActorSpec / SensorSpec`；
- 显式 spawn 顺序；
- `RunIdentity / RunRegistry`；
- config hash；
- 单 tick lease、同步帧和 cleanup 语义。

R2 应复用其身份、hash、registry 和生命周期思想，但第一版不直接替换已经实测的
stable VLA 控制循环。若强行切换到另一套 live runner，会把 R2 同时变成执行器迁移
任务。

### 2.2 不能直接拿来做 paired oracle

当前 stable runner 有四个关键缺口：

1. `--seed` 主要用于 `--map random`，不固定 ego spawn、NPC、weather 或 actor script；
2. `_spawn_ego()` 使用 `free_spawn()`，失败时遍历其他 spawn，不能保证两分支同起点；
3. runner 只创建 ego，没有 6–8 个困难场景的 actor fixture；
4. `--force-candidate-index` 会在每个 VLA 更新重新 forward。两分支第一次动作后
   Observation 已不同，后续 K2 也不同，因此“整段 force-0 vs force-1”是两种策略
   rollout，不是同一 K2 的 candidate outcome。

R2 禁止把现有 R1 force smoke 简单批量运行后称为 oracle。

### 2.3 CARLA 没有可信的完整 world snapshot restore

R2 不假设 CARLA 提供可移植的物理状态快照/回滚。`world.get_snapshot()` 只有当前帧
信息，不是可恢复 checkpoint。

因此 paired replay 定义为：

```text
同一冻结 ScenarioFixture
  → 冷重建 anchor
  → 生成并保存一次 K2AnchorArtifact
  → cleanup
  → 冷重建 branch-0，加载同一 anchor bundle，执行 candidate 0
  → cleanup
  → 冷重建 branch-1，加载同一 anchor bundle，执行 candidate 1
```

每次重建都验证 measured initial state；不是在运行中复制 world，也不创建第二 tick
master。

## 3. 冻结的实验单位

### 3.1 三层 identity

```text
scenario_id
  └─ seed_id
      └─ pair_id
          ├─ anchor_run_id
          ├─ branch_0_run_id
          └─ branch_1_run_id
```

定义：

```text
pair_id = sha256(
  scenario_registry_hash
  + scenario_id
  + seed_id
  + model/checkpoint/config/retimer hash
  + executor config hash
)[:20]
```

attempt 重试不得覆盖原 run。使用 `attempt_id` 递增并保留失败。

### 3.2 一个 pair 的唯一动作集合

每个 pair 只有一个 `K2AnchorArtifactV1`：

```text
schema_version
pair_id / scenario_id / seed_id
anchor_run_id / anchor CARLA frame / simulation time
requested_initial_state_hash
measured_initial_state_hash
observation fingerprint
model/checkpoint/config/retimer hash
native_path_xy / native_path_hash
candidate 0/1 full T10
execution_spec 0/1
top1_index
guard status/metrics
artifact content hash
```

branch-0 和 branch-1 必须加载同一 artifact content hash。分支阶段不得重新 forward、
重建 candidate、修改 probability、重定时或重新选择空间路径。

### 3.3 主 outcome 时域

冻结：

```text
primary outcome horizon = 2.50 s
control dt = 0.05 s
expected control ticks = 50
K2 horizon = 2.50 s
```

主 oracle 只使用 2.5 s 内结果，与未来 World-V0 的预测时域一致。

可额外保存：

```text
diagnostic tail = 2.50 s
```

tail 只观察延迟碰撞/恢复，不参与 R2 主 oracle，不得用 tail 结果事后改主标签。

### 3.4 冻结候选的执行

branch 开始时只调用一次：

```text
select_k2(anchor_bundle, mode="force", force_index=0|1)
apply_k2_to_executors(...)
```

之后由同一个 PathManager、VLASpeedPlanner、ConstrainedVLAMPC 跟踪冻结 execution
spec 2.5 s。不得为保持 freshness 伪造新 VLA forward 或新 candidate ID。

现有 MPC freshness 为：

```text
soft=1.0s, hard=2.5s, zero=5.0s
```

R2 保持该配置不变。2.5 s 内的 soft ramp 是当前真实 executor 行为，不能为了提高
候选差异而放宽。

## 4. Scenario Registry V1

### 4.1 首版规模

冻结 6 个场景 × 2 个 seed，共：

```text
12 pairs
12 anchor forwards
24 primary branch runs
```

若某 pair 不可比，只允许按同一 registry 重试至最多 2 个新 attempt；不得替换为更
好看的场景或 seed。

### 4.2 场景族

由于 R1 K2 只有纵向 timing 差异，R2 首版使用三类纵向选择真正可能影响 outcome 的
场景。每类两个 fixture：

| family | scenario_id | 目的 |
|---|---|---|
| lead braking | `lead_brake_moderate` | 前车中等制动，比较 nominal progress 与 conservative gap |
| lead braking | `lead_brake_hard` | 较短 headway/较强制动，测试碰撞或近碰差异 |
| cut-in | `cut_in_early` | 邻车较早进入 ego path |
| cut-in | `cut_in_late` | 邻车较晚进入，窗口更窄 |
| crossing | `cross_vehicle_clear` | 横穿车辆较大余量，防止只收集失败 |
| crossing | `cross_vehicle_tight` | 横穿车辆较小余量，测试 timing 选择 |

VRU/遮挡可以在 registry dry-run 证明 walker 控制、遮挡物和复现稳定后替换
`cross_vehicle_clear`，但替换必须在看 candidate outcome 前完成并冻结。不能根据
oracle 结果挑场景。

### 4.3 第一版地图

优先全部使用已经通过 R1 的 `Town03`：

- 避免把 R2 同时变成多地图/DX12 稳定性实验；
- 减少 Windows CARLA 冷启动和 shader 变量；
- 允许按 road/lane/waypoint 选择直道、并线和路口。

多地图不是 R2 完成条件。若 Town03 无法稳定构造三类 fixture，才加入一个已 preflight
通过的普通 Town，并在 registry 中冻结。

### 4.4 Registry 必填字段

机器可读文件：

```text
safedrive_foundry/config/g4a/scenario_registry_v1.toml
```

每个场景/seed 必须展开为显式配置：

```text
schema_version
registry_version
scenario_id
family
seed_id
map_name
weather preset + sun/cloud/wetness exact values
sim_dt_s
duration_s
ego blueprint
ego spawn transform
ego initial linear/angular velocity
route waypoint/segment identity
actor list:
  name, role, blueprint, transform, initial velocity, bounding box expectation
actor script:
  script type, simulation-time knots, throttle/brake/steer/walker control
traffic light initial state/freeze policy
sensor contract
VLA/MPC/config references
expected decision anchor time
```

禁止：

- “任意可用 spawn”；
- spawn 失败后换下一个点；
- Traffic Manager 随机行为作为核心 actor 行为；
- actor script 读取 candidate_id 或 ego future 后改变计划；
- 根据 branch 结果改变 weather、timing、gap 或 route。

### 4.5 Fixture dry-run 与冻结

R2-A 先做 registry dry-run，只验证：

- 所有 actor 可在精确 transform spawn；
- 互不重叠；
- route 可用且长度足够；
- actor script 在无 VLA 的固定控制下按预期发生；
- 2 个 seed 确实改变预登记 gap/timing，而不是只改无效 RNG；
- cleanup 后无残留 actor/sensor；
- requested/measured initial state 可在容差内重复。

dry-run 不计算 candidate outcome，不产生 oracle。

registry 经 dry-run 后生成：

```text
registry_manifest.json
registry_sha256
scenario visual/contact sheet（仅人工核对布局）
```

此后不得因 outcome 结果改 registry。必要修复必须提升 registry version，并使旧 run
保持可查询。

## 5. Deterministic actor script

### 5.1 原则

核心 NPC 使用 simulation-time 驱动脚本，不依赖 wall time，不读取 branch candidate，
不使用在线 oracle 决策。

车辆 actor 第一版只允许：

```text
hold
constant_throttle
constant_brake
piecewise_vehicle_control
```

walker 第一版只允许显式 `WalkerControl`，若复现不稳定则不进入冻结 6 场景。

### 5.2 同 branch 行为

actor script 输入：

```text
scenario_id / seed_id / simulation_time_since_anchor
```

输出：

```text
carla.VehicleControl or carla.WalkerControl
```

脚本不基于 ego 距离做触发。否则 candidate 0/1 速度差会反过来改变 hazard timing，
难以区分“动作条件交互”与“fixture 不一致”。交互响应属于后续扩展，不是首版 pilot。

### 5.3 初始化边界

允许在第一 tick 前设置冻结的初始 transform/velocity，作为 ScenarioFixture 初始化；
运行开始后不得用 `set_transform()` 拖动 actor。运动由 CARLA physics + control script
产生。

## 6. Initial State 与 Observation 可比性

### 6.1 两个 hash

`requested_initial_state_hash`：

- registry canonical JSON 的精确 hash；
- 表示请求的 fixture 完全相同。

`measured_initial_state_hash`：

- decision anchor 时实际 world state 的 canonical/量化 hash；
- 表示 CARLA 实际建立的状态。

量化仅用于跨 run hash：

```text
position: 0.01 m
rotation: 0.1 deg
linear/angular velocity: 0.01 SI unit
simulation time: 0.001 s
```

原始未量化 float 同时保存，不能只保留 hash。

### 6.2 Measured state 内容

至少包括：

- map/OpenDRIVE identity；
- world settings；
- weather；
- ego/NPC blueprint、transform、velocity、angular velocity、control；
- traffic light state；
- route anchor；
- sensor calibration；
- CARLA server epoch/version；
- actor script phase；
- anchor simulation frame/time。

### 6.3 Observation fingerprint

anchor 保存：

```text
front RGB raw sha256
image dimensions/layout
ego observable state
route targets
camera frame
K2 bundle hash
```

branch 不重新调用 VLA，但必须采一帧 branch-start RGB 用于可比性诊断。像素 hash 不同
不自动判不可比（渲染可能存在微小差异）；同时保存像素 MAE/P99 和感知 hash距离。

### 6.4 Pair 硬可比条件

以下全部成立才是 `COMPARABLE`：

1. registry/scenario/seed/config/model/executor hash 相同；
2. branch 0/1 加载同一个 anchor artifact hash；
3. Guard 在 anchor 为 `OK`；
4. measured ego/NPC transform 均在 0.02 m / 0.2 deg 内；
5. velocity 差均在 0.05 m/s 内；
6. anchor frame phase、actor script phase、traffic light state一致；
7. 两分支完成 50 个 primary tick；
8. 无 sensor barrier/sync/version/tick-owner/cleanup failure；
9. candidate selected/executed/source_id 全程与 forced ID 一致；
10. 每分支 MPC 至少 48/50 tick 为 `solved`，其余必须是显式 bounded fallback。

不满足时标记 `INCOMPARABLE`，保留原因，不进入 oracle accuracy/gap 分母。

### 6.5 失败分类

固定：

```text
INITIAL_STATE_MISMATCH
ANCHOR_BUNDLE_MISMATCH
SPAWN_FAILED
SCRIPT_PHASE_MISMATCH
SENSOR_SYNC_FAILURE
TICK_OWNER_CONFLICT
SERVER_CRASH_OR_HANG
MPC_DEADLINE_UNRELIABLE
EXECUTION_BINDING_FAILURE
GUARD_REJECT
CLEANUP_FAILURE
```

同一 pair 最多一次初始运行和两次有实质差异的重试。外部服务器失败只能按
AGENTS.md ensure 一次；不得无限重跑直到结果好看。

## 7. Outcome Trace V1

### 7.1 Runtime Observable 与 Oracle 隔离

分支控制只能读取 R1 Observable、冻结 K2 和当前 executor 状态。

以下写入独立 `oracle_trace.jsonl`，只供离线评价：

- 所有 actor 的真实 transform/velocity；
- CARLA collision/接触；
- 真实未来最小间距；
- lane/off-road ground truth；
- first-conflict time；
- branch 结束后的 outcome。

Oracle trace 对象必须带：

```text
oracle_only=true
consumed_by_control=false
```

测试必须证明该 namespace 未进入 VLA、selector、PathManager、SpeedPlanner 或 MPC。

### 7.2 每 tick 保存

```text
pair/run/scenario/seed/candidate identity
carla frame + simulation time
ego state
scripted actor states + script phase
selected/executed/source_id
path age/freshness regime
MPC mode/status/latency
collision/lane/offroad
route progress
actor clearance / TTC diagnostic
control and comfort signals
```

### 7.3 分支聚合指标

主 2.5 s：

```text
completed_primary_horizon
collision_episode_count
first_collision_time_s
collision_impulse_sum
offroad_fraction / first_offroad_time_s
lane_invasion_episode_count
minimum_actor_clearance_m
minimum_ttc_s
route_progress_delta_m
distance_m
longitudinal_accel abs P95/max
jerk abs P95/max
lateral_accel abs P95/max
steer_rate abs P95/max
MPC solved/timeout/fallback counts
```

TTC 为 oracle diagnostic，使用真实 actor state计算；它不是 VLA/World runtime 输入。
无 closing actor 时为 `null`，不能编码成“无限安全分数”。

## 8. Oracle Best-of-K V1

### 8.1 不使用伪概率

R1 probability 固定 `[0.5, 0.5]`。Oracle 不读取 probability 排序，也不使用
`oracle_best_of_k(..., expert=None)` 的旧 probability top-1 语义。

### 8.2 预登记的 lexicographic comparator

只有 `COMPARABLE` pair 进入比较。依次比较：

1. **collision**：无 collision 优于有 collision；都有时 first collision 更晚；
2. **off-road**：offroad fraction 差超过 0.02 时更低者优；
3. **near conflict**：TTC risk bucket 更低者优：
   `<0.5s`、`0.5–1.0s`、`1.0–1.5s`、`>=1.5s/null`；
4. **minimum clearance**：差超过 0.50 m 时更大者优；
5. **route progress**：差超过 0.50 m 时更大者优；
6. **comfort**：jerk P95 差超过 1.0 m/s³ 时更低者优；
7. 仍无差异：`TIE`，确定性回到 VLA top-1（candidate 0）。

collision/off-road 优先级不可被 progress 或 comfort 抵消。R2 不在看结果后调权重。

### 8.3 Pair 标签

```text
TOP1_BEST
CANDIDATE_1_BEST
TIE
BOTH_BAD
INCOMPARABLE
```

`BOTH_BAD`：

- 两分支都 collision；或
- 两分支 offroad fraction ≥0.10；或
- 两分支均未完成 primary horizon 且原因不是外部不可比故障。

`BOTH_BAD` 仍可有相对 oracle winner，但必须同时保留 `both_bad=true`，后续用于判断
是否触发 G6，而不能把“较不坏”写成安全成功。

### 8.4 Oracle 输出

每 pair：

```text
top1_candidate_id
oracle_candidate_id or null
oracle_decision_level
decision_reason
outcome_delta
both_bad
comparable
failure reasons
```

聚合：

```text
comparable rate
top1 best / candidate1 best / tie / both bad counts
candidate1 win by scenario family and seed
top1→oracle safety-event recovery
progress/comfort tradeoff
collapse/no-space rate
MPC deadline miss and external failure rate
```

R2 的“oracle gap”优先报告计数和分母，不在 12 pairs 上伪造高精度百分比。

## 9. Pilot 结果标签

按以下顺序赋一个主标签：

1. `IMPROVE_VLA`：
   - comparable pairs 中 `BOTH_BAD >= 50%`，或
   - anchor numeric collapse/Guard reject 使可执行 pair 少于 50%；
2. `ENTER_WORLD`：
   - 至少 4 个 decisive（非 tie）comparable pair；
   - candidate 1 至少在 2 个 pair 获胜；
   - candidate 1 的胜利覆盖至少 2 个 scenario family；
3. `WEAK_SELECTION_SPACE`：
   - 有 1–3 个 candidate 1 win；或
   - decisive 结果只在一个 family/单一 seed 出现；
4. `NO_SELECTION_SPACE`：
   - comparable rate ≥80%；
   - candidate 1 win=0；
   - 至少 80% comparable pairs 为 `TOP1_BEST` 或 `TIE`；
5. 其他：`PILOT_INCONCLUSIVE`。

无论标签为何，ROADMAP 仍进入 World 主线；标签只约束 R3/R5 的结论强度和是否需要
条件式 G6。

## 10. Evidence 目录

```text
docs/runtime-evidence/r2-g4a-paired-pilot/
  registry/
    scenario_registry_v1.toml
    registry_manifest.json
    dry_run_report.json
  pairs/
    <pair_id>/
      pair_manifest.json
      anchor/
        anchor_bundle.json
        anchor_front_rgb.npy
        initial_state_raw.json
        run_config.json
        failure.json (if any)
      branch-0/
        run_config.json
        initial_state_raw.json
        outcome_trace.jsonl
        oracle_trace.jsonl
        control_seq.json
        collision_episodes.json
        branch_summary.json
        failure.json (if any)
      branch-1/
        ...
      pair_comparability.json
      pair_oracle.json
  paired_outcomes.jsonl
  oracle_table.json
  pilot_summary.json
  lineage_manifest.json
  failure_ledger.jsonl
```

大图像/trace 可由 hash 引用；pair manifest 和失败原因不可省略。

Evidence 状态：

```text
registry dry-run          → IMPLEMENTED
single pair smoke         → MEASURED
12 pair pilot             → MEASURED
重复/审计通过             → VERIFIED_WITH_LIMITS
```

R2 不把 12 pairs 外推为统计显著安全结论。

## 11. 文件级实施计划

### 11.1 新增

| 文件 | 职责 |
|---|---|
| `safedrive_foundry/config/g4a/scenario_registry_v1.toml` | 6 场景 × 2 seed 冻结 registry |
| `safedrive_foundry/driving_vla/evaluation/paired_contract.py` | anchor/branch/pair 数据对象、canonical hash |
| `safedrive_foundry/driving_vla/evaluation/scenario_registry.py` | TOML 解析、schema、seed 展开、freeze manifest |
| `safedrive_foundry/driving_vla/evaluation/comparability.py` | initial-state tolerance、pair failure 分类 |
| `safedrive_foundry/driving_vla/evaluation/outcome_metrics.py` | collision/offroad/clearance/TTC/progress/comfort 聚合 |
| `safedrive_foundry/driving_vla/evaluation/oracle.py` | 冻结 comparator、pair/pilot 标签 |
| `tests/g4/run_g4a_paired.py` | anchor → branch 0/1 的唯一 live orchestrator |
| `tests/g4/test_g4a_registry.py` | registry/hash/freeze/invalid 测试 |
| `tests/g4/test_g4a_comparability.py` | 初始状态、同步、失败分类测试 |
| `tests/g4/test_g4a_oracle.py` | comparator、tie、both-bad、不可比测试 |
| `tests/g4/test_g4a_anchor_execution.py` | 同一 artifact、无第二 forward、ID bind 测试 |

新增 package 时补齐 `__init__.py`，不把 oracle helper 导出到 runtime 控制 namespace。

### 11.2 最小修改

| 文件 | 修改 |
|---|---|
| `tests/g3/run_g3_vla_mpc_stable.py` | 只抽取/公开已验证的 ego exact-spawn、metrics 和 cleanup helper；默认 G3 CLI 行为不变 |
| `safedrive_foundry/runtime/scenario_runtime.py` | 如确有必要，仅增加 measured-state/actor access 的只读接口；不创建新 tick 路径 |
| `README.md` | 增加 R2 权威入口 |
| `ROADMAP.md` | R2 链接到本文 |
| `PROGRESS.md` | 只登记实际实施/测量事实 |
| `START_TASK.md` | R2 关闭后更新状态并停止 |

若必须复制 stable runner 超过约 200 行控制逻辑，应先抽取无 CARLA 副作用的 helper；
不得维护第二套 PathManager/MPC 参数。

## 12. 实施顺序

### R2-A：纯合同、registry 与 oracle

1. 定义 dataclass/schema/hash；
2. 实现 registry loader 和 canonical freeze；
3. 实现 comparability；
4. 实现 outcome metrics；
5. 实现 lexicographic oracle；
6. 用合成 outcome 测试所有 tie/both-bad/invalid 分支。

此阶段不启动 CARLA。

### R2-B：确定性 fixture dry-run

1. 为 Town03 选择精确 spawn/waypoint；
2. 实现 time-based actor script；
3. 每场景/seed 重建两次；
4. 验证 measured initial state 容差与 cleanup；
5. 看 outcome 前冻结 registry v1。

若 6 个场景无法稳定重建，不得进入 paired outcome。

### R2-C：单 pair end-to-end smoke

使用一个 `lead_brake_moderate` pair：

1. anchor 真实 forward 一次；
2. Guard OK；
3. serialize/deserialize bundle bitwise/hash 一致；
4. branch 0/1 均加载同一 artifact；
5. 各完成 50 ticks；
6. 生成 comparability 和 oracle 文件；
7. 证明 oracle trace 未进入控制。

### R2-D：冻结 pilot

按 registry 顺序运行 12 pairs。branch 顺序 counterbalance：

```text
seed A: candidate 0 → candidate 1
seed B: candidate 1 → candidate 0
```

避免固定 run order 与 GPU/server warm-state 混淆。顺序本身写入 pair manifest。

### R2-E：聚合、复验和关闭

1. 聚合所有成功/失败 attempt；
2. 计算 comparability；
3. 生成 oracle table；
4. 随机抽 2 个 pair 做同 registry 重复；
5. 保留不一致和 deadline miss；
6. 赋 pilot 主标签；
7. 更新 `PROGRESS.md` 后停止。

不得自动创建 R3 数据集或训练 World。

## 13. 必做离线测试

至少覆盖：

1. registry 6 场景、3 family、每场景 2 seed；
2. canonical hash 固定且字段变化必改 hash；
3. duplicate scenario/seed 拒绝；
4. spawn fallback/随机 actor script 拒绝；
5. anchor artifact K=2/T=10/Guard OK；
6. serialize round-trip hash 一致；
7. branch 0/1 使用同一 artifact；
8. branch 模式不允许第二次 VLA forward；
9. candidate/source_id mismatch fail closed；
10. initial position/yaw/velocity 超容差 → incomparable；
11. script/weather/light phase mismatch → incomparable；
12. runtime/sensor/tick/cleanup failure → incomparable；
13. collision 优先于 progress；
14. offroad 优先于 progress；
15. TTC/clearance/progress/comfort tie-break；
16. exact tie 回到 top-1 但标签仍为 `TIE`；
17. both-bad 保留相对 winner；
18. oracle fields 不进入 control namespace；
19. failed attempt 不被后续成功覆盖；
20. pilot label 五种分支均覆盖；
21. R1/G3 全量测试不回归。

## 14. 验证命令

### 14.1 离线

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

python -m unittest discover -s tests/g4 -t . -v
python -m unittest discover -s tests/g3 -t . -v
python -m compileall -q safedrive_foundry
git diff --check
```

### 14.2 Registry dry-run

只有离线通过后：

```bash
python scripts/sdf.py sim preflight --json

python tests/g4/run_g4a_paired.py registry-dry-run \
  --registry safedrive_foundry/config/g4a/scenario_registry_v1.toml \
  --evidence-dir docs/runtime-evidence/r2-g4a-paired-pilot/registry
```

preflight 非 READY 时按 `AGENTS.md` 停止。

### 14.3 单 pair smoke

```bash
python tests/g4/run_g4a_paired.py run-pair \
  --registry safedrive_foundry/config/g4a/scenario_registry_v1.toml \
  --scenario-id lead_brake_moderate \
  --seed-id seed_a \
  --evidence-dir docs/runtime-evidence/r2-g4a-paired-pilot/pairs
```

### 14.4 冻结 pilot

```bash
python tests/g4/run_g4a_paired.py run-set \
  --registry safedrive_foundry/config/g4a/scenario_registry_v1.toml \
  --evidence-dir docs/runtime-evidence/r2-g4a-paired-pilot

python tests/g4/run_g4a_paired.py aggregate \
  --registry safedrive_foundry/config/g4a/scenario_registry_v1.toml \
  --evidence-dir docs/runtime-evidence/r2-g4a-paired-pilot
```

runner 必须支持幂等：已完成且 hash 匹配的 run 只读取，不重写；失败 attempt 另建目录。

## 15. R2 关闭标准

只有全部成立才可标 `COMPLETED_WITH_LIMITS`：

1. registry 在看 outcome 前冻结；
2. 6 个场景、3 个 family、每场景 2 seed；
3. 每 pair 只有一次 anchor forward 和一个 K2 artifact；
4. branch 0/1 加载同一 artifact；
5. exact spawn、actor script、initial-state hash 可审计；
6. 至少 10/12 pairs 为 comparable；否则 `PILOT_INCONCLUSIVE`，不得正常关闭；
7. 每个 comparable pair 具有完整 2.5 s outcome trace；
8. generated/selected/executed/source_id 绑定正确；
9. oracle comparator 预登记且测试覆盖；
10. top-1/oracle/both-bad/tie/incomparable 可查询；
11. 至少 2 个 pair 重复运行后 oracle 标签一致；数值差异保留；
12. R1/G3 不回归；
13. latency P50/P95/P99、MPC deadline miss、VRAM/RAM 和失败全部记录；
14. `git diff --check` 通过；
15. 实际结果与限制进入 `PROGRESS.md`。

若 comparable 少于 10：

- 状态为 `BLOCKED_EXTERNAL`（CARLA/同步为主）或
  `REPAIR_REQUIRED`（fixture/runner 为主）；
- 不得用替换 seed/场景凑足分母。

## 16. WITH_LIMITS 必须写明

- 只有 6 场景 × 2 seed 的 pilot，不具统计代表性；
- 全部为 CARLA SIL；
- R1 候选只在纵向 timing 上不同；
- actor script 是固定、非交互首版；
- oracle 使用 privileged future，只能离线评价；
- probability 仍未训练/未校准；
- 未实现 World，未证明 World 可恢复 oracle gap；
- 2.5 s 主时域不代表长期驾驶安全；
- server/GPU deadline miss 与不可比率必须保留。

## 17. 实测关闭记录（2026-07-24）

| 项 | 结果 |
|---|---|
| registry 冻结 | 是（R2-B） |
| R2-C smoke | COMPARABLE / TIE |
| 早期失败 live | 归档 `failed_live_blocked_external_*`（0/12；RPC 挂死） |
| 修复后 live | 11× TIE COMPLETED + 1× FAILED timeout |
| comparable | **11/12** |
| pilot | **NO_SELECTION_SPACE** |
| repeat | 2/2 标签一致（TIE/TIE） |
| latency P50/P95/P99 | ~207 / ~858 / ~1276 ms |
| VRAM peak | ~2218 MB |
| MPC | solved 1100 / timeout 0 / fallback 0 |
| `r2_status` | **`COMPLETED_WITH_LIMITS`** |

早期“CARLA 已开却 0 comparable”原因：tick 中途进程挂死 + no_auto_retry 密封
attempt，同 manifest 不会重跑。加固后：soft cleanup、purge、pair 间 ensure、
shared policy、60s RPC。未启动 R3/World；未 commit/push。

## 18. 立即停止条件

- 无法从同一 anchor artifact执行两个分支；
- 为配对需要第二 CARLA client/tick owner；
- branch 必须重新 forward 才能继续；
- exact spawn 只能靠失败后随机 fallback；
- actor script 依赖 candidate ID 或 oracle future；
- measured initial state 频繁超容差；
- 需要放宽 R1 Guard、PathManager 或 MPC 才能跑通；
- outcome comparator 需要看结果后改权重；
- oracle 数据进入 runtime Observable/control；
- CARLA preflight/版本/tick conflict；
- 两次修复无实质进展。

## 18. 本任务明确不做

- World dataset 正式冻结；
- persistence/CV/CTRV/Reward baseline；
- World-V0 训练或 runtime；
- online oracle；
- MAP-Elites/G4B 自动搜索；
- learned spatial K2、LoRA 或 G6；
- Safety live；
- 多机并行、第二 GPU 或第二 CARLA server；
- 实车/公共道路结论。

R2 的最短正确产物是：一个真实 SimLingo anchor K2，在相同可复现初始状态下分别执行
candidate 0/1，得到诚实的 comparable/incomparable paired outcome 和离线 oracle
best-of-K 表，然后停止。
