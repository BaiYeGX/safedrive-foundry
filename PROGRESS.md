# 项目当前进度

本文只记录已确认动态事实。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前任务 | Spatial R2-X 已按可用模型目标收尾 |
| R2 纵向 | `COMPLETED_WITH_LIMITS / NO_SELECTION_SPACE`（只读） |
| R2-X 正式 v4 | `R2K_V4_FORMAL_CLOSED / IMPROVE_VLA / REPAIR_REQUIRED` |
| R2-X | `COMPLETED_WITH_LIMITS / NOMINAL_60S_PASS / DEFENSIVE_UNRELIABLE` |
| Formal blind pilot | `12/12 EXECUTED / 8 COMPARABLE / 8 TIE / PILOT_INCONCLUSIVE` |
| 外部状态 | CARLA `READY / async / tick owner free`（pilot 后已复核） |
| R3 | `NOT_AUTHORIZED` |
| 更新日期 | 2026-07-28 |

## 2026-07-28 R3/R4 实施文档细化

- 新增 `docs/R3_R4_WORLD_DATA_MODEL.md`，把 R3/R4 从路线摘要展开为：
  - candidate-conditioned World 的代码映射与外部方法依据；
  - `ActionBranchDatasetV0` 的 Observable 输入、candidate、label、mask 和坐标合同；
  - oracle-only actor future collector、跨 cold rebuild actor identity；
  - group split、blind Evidence 隔离、样本规模和数据质量门；
  - persistence/CV/CTRV/rule/no-action/action-swap 基线；
  - 4M–8M object/vector World-V0 架构、loss、训练、指标和关闭标签；
  - `K_eff=1 / NO_ALTERNATIVE → nominal` 的原生 fallback 语义。
- 明确现有 R2 aggregate 没有逐 actor future，不能直接当 World 训练集；正式
  12-pair blind Evidence 只作冻结 audit/holdout。
- 同步更新 `docs/WORLD_MODEL.md`、`ROADMAP.md` 和 `README.md`。
- 本轮仅修改文档，没有启动 R3 采集、R4 训练、CARLA、commit 或 push；
  `R3=NOT_AUTHORIZED` 保持不变。

## 冻结结论

- 纵向 Evidence：`docs/runtime-evidence/r2-g4a-paired-pilot/`
  - 11 comparable TIE；共享空间路径；`NO_SELECTION_SPACE`。
- Spatial v4 Evidence：
  `docs/runtime-evidence/r2-spatial-k2-pilot-v4-formal/`
  - 0/12 comparable；candidate 1 全部 `NO_ALTERNATIVE`；
    `IMPROVE_VLA / REPAIR_REQUIRED`。
- 上述两项没有被本轮开发结果覆盖或改写。

## 本轮根因与修复

### 1. R2-K provenance

- future run-set manifest 可选绑定：
  `policy_type`、`policy_model_id`、`spatial_head_checkpoint_hash`、
  `spatial_k2_config_hash`。
- pair identity / model-retimer hash 绑定实际 Spatial head 与 config。
- failed V2 row 使用实际 V2 candidate identity；不再默认 `v1_nominal`。
- 旧 longitudinal v1 manifest 仍可验证。

### 2. Teacher fail-open

确认旧 production teacher 有两个标签错误：

1. Guard reject 只对少数 reason fail，导致低于 0.50 m 的 candidate 仍可能标
   `alternative_available=true`。
2. d=0 lattice 被当成 nominal 绕过 diversity；选择器先 argmin、后检查
   separation/progress，使非法候选遮蔽合法 0.7/1.0 m 候选。

修复后：

- defensive candidate 统一走 exact Guard，只有真实 nominal identity 可绕过 set diversity；
- 0.50 m、progress、comfort、direction、strict improvement 先定义 legal pool；
- 只在 legal pool 内排序；
- identity 升为：
  - schema `safedrive.k2_spatial_teacher.v4`
  - id `spatial_defensive_lattice_v4_legal_pool`

### 3. Development 数据

| 数据 | 用途 | 结果 |
|---|---|---|
| v8 | 中间诊断 | 证明 exact Guard 后旧选择顺序只剩 5/74 available；`INVALID_TEACHER_SELECTION_ORDER` |
| v9 | 当前 development | 74；train/val=52/22；available=29；available subfloor=0 |

v9：

- episode leakage=false；
- train available=21，val available=8；
- 左右 defensive direction 均有 train/val 支持；
- 旧 v4 exam 明确 retired to development；
- `formal_train_candidate=false`、`r2k_pilot_allowed=false`；
- 正式关闭仍需要全新的 blind exam registry。

Evidence：

```text
docs/runtime-evidence/r2x-training/dataset-v8-development/
docs/runtime-evidence/r2x-training/dataset-v9-development/
docs/runtime-evidence/r2x-training/feature_probe_v9_development.json
```

### 4. v9 head 与 proposal-validity 结果

训练：

```text
run = v9_development_teacher_v4
steps = 2000
device = cpu
checkpoint sha256 = f99557566cf0c942dab91a43ecdb1cf8cc7234a2f2666d58dcb837bc7db2fbd3
status = HEAD_TRAINED_NOT_FORMAL
```

验证集 n=22、eligible=8：

| 门 | 实测 | 阈值 | 状态 |
|---|---:|---:|---|
| eligible Guard | 8/8 = 1.000 | ≥0.90 | PASS |
| eligible spatial sep | 7/8 = 0.875 | ≥0.70 | PASS |
| availability recall | 7/8 = 0.875 | ≥0.80 | PASS |
| learned confidence specificity | 11/14 = 0.786 | 诊断 | 非阻塞 |
| eligible proposal valid | 7/8 = 0.875 | ≥0.70 | PASS |

3 个 learned-confidence false positive 全部来自 held-out empty episode，继续原样
报告。候选是否能执行不再由该分类器否决：

```text
learned confidence → 排序诊断
proposal valid     → residual diversity + Guard + PM/MPC
```

checkpoint 允许 `development_live_smoke`，仍禁止 `formal_offline / X5H / R2-K`。

Evidence：

```text
docs/runtime-evidence/r2x-training/checkpoints/v9_development_teacher_v4/
docs/runtime-evidence/r2x-training/offline_exec_report_v9_teacher_v4.json
docs/runtime-evidence/r2x-training/offline_exec_report_v10_executability_semantics.json
```

### 5. CARLA development smoke

Evidence：
`docs/runtime-evidence/r2x-live-force-smoke-v6-development-executability/`

| pair | 结果 |
|---|---|
| cut_in_early/seed_a | PASS；force 0/1；committed path + control 分叉 |
| lead_brake_hard/seed_a | PASS；无合法 spatial proposal，force 0 only |
| cross_vehicle_clear/seed_a | PASS；force 0/1；committed path + control 分叉 |

汇总：

- 3/3 PASS，dual-force=2；
- 每个 anchor 恰好一次 SimLingo forward，branch 不 re-forward；
- 5 个 branch × 50 ticks = 250 ticks；
- MPC solved 250/250，fallback=0，collision=0；
- `formal_acceptance=false`、`r2k_authorized=false`。
- smoke 后 CARLA 已恢复 async；最终 preflight `READY`、
  `synchronous_mode=false`。

## 特征诊断

- mean64 对左右 defensive direction 可预测（v9 probe balanced accuracy=1.0）。
- learned confidence 对 empty/no-conflict 仍弱，但不再拥有执行否决权。
- 当前模型已达到 development 可用；下一步是新 blind exam 下的 formal X5H/R2-K，
  不是继续扩大网络。

## 本轮实际验证

```text
TeacherSkeletonTest: Ran 11, OK
R2-X feature/development/training + spatial/runner contracts:
  Ran 87, OK
proposal/checkpoint/live directed regression:
  Ran 53, OK
v9 offline eval:
  Guard PASS; spatial PASS; proposal validity PASS
CARLA development smoke:
  3/3 PASS; dual-force=2; MPC 250/250; collision=0
```

## 禁止

- 未经新 blind exam，把 v9 development checkpoint promote 为 formal。
- 把 v4 exam 再作为 blind exam。
- 降低 0.50 m 或缩小 Oracle TIE 带。
- runtime fixed bias / lattice rescue。
- 自动启动 R3。

## 2026-07-27 正式关闭复验

结论：R2 尚不能诚实关闭；没有复用旧 v1 或 development smoke 冒充 blind
formal Evidence。已完成不依赖 CARLA 的必要修复：

- `scenario_registry.py`
  - v1 继续严格要求原 6 个 ID；
  - post-v1 registry 允许全新 6 个 scenario ID；
  - 仍硬要求 12 pair、每场景 `seed_a/seed_b`、三类 family、Town03、
    exact spawn 与 time-script 合同。
- `r2x_promote_formal_checkpoint.py`
  - 改读 `safedrive.r2x.offline_exec.v3`；
  - Guard/spatial/proposal-valid 三门分别要求 0.90/0.70/0.70；
  - 必须传 frozen blind registry + manifest；
  - 拒绝 registry v1；
  - 逐行读取训练数据并要求 blind 与 train `(scenario_id, seed_id)` 零重叠；
  - 正式 checkpoint manifest 绑定 blind registry SHA256/version/audit。
- `r2x_live_force_smoke_v2.py`
  - formal X5H 不再写死旧 registry；
  - 必须显式传 frozen manifest；
  - 1–3 个 pair 必须在 CLI 中预声明并存在于 frozen registry；
  - report 记录 registry audit 与预声明 pair。

验证：

```text
latest registry/promotion/checkpoint/spatial directed tests: Ran 68, OK
compileall: OK
git diff --check: OK（仅两个既有 Windows 文件的 LF/CRLF warning）
```

CARLA：

```text
preflight #1: RETRYABLE_FAILURE / RPC_HANDSHAKE_FAILED
ensure: 唯一一次恢复动作长期无返回，已终止客户端等待
preflight #2: RETRYABLE_FAILURE / RPC_HANDSHAKE_FAILED
TCP reachable=true; RPC reachable=false; process_state=NOT_RUNNING
```

因此未运行 registry dry-run、formal promotion、X5H 或 R2-K，未创建 blind
Evidence，R3 继续 `NOT_AUTHORIZED`。精确恢复顺序：

1. Windows CARLA 恢复到 `preflight READY`；
2. author/freeze 新 `registry_version != v1` 的 6×2 blind registry（仅做
   spawn/script/rebuild dry-run，不看 candidate outcome）；
3. formal promote 同一 v9 checkpoint，绑定 registry hash；
4. frozen pair 中预声明 1–3 pair 跑 X5H；
5. X5H 过门后新 Evidence 目录跑 formal R2-K 12 pair + repeat + close；
6. 只有 closure gate 通过才把 R2 关闭；本任务不自动进入 R3。

## 2026-07-27 R2-X 最终 blind 验收

### Blind registry 与 checkpoint

- registry：`scenario_registry_v2_blind.toml`
- version：`v2-blind-20260727`
- SHA256：`01da3aba4d9c68581a0204b489b214ee578f13099e5e9436a797bacb6719b559`
- dry-run：12/12 spawn/script/cold-rebuild PASS 后冻结
- 与 dataset-v9 `(scenario_id, seed_id)` overlap：0
- head SHA256：
  `f99557566cf0c942dab91a43ecdb1cf8cc7234a2f2666d58dcb837bc7db2fbd3`
- formal manifest 绑定上述 registry hash、offline v3 report 与数据 hash

### Formal X5H v8

Evidence：`docs/runtime-evidence/r2x-live-force-smoke-v8-blind-formal/`

- `X5H_PASS_WITH_LIMITS`
- 3 个预声明 pair 中 2 PASS、2 dual-force
- crossing 与 lead 的 committed path/control 均分叉
- cut-in 为 `SPATIAL_COLLAPSE_ELIGIBLE`，Guard fail-closed
- 未降低 0.50 m；未使用 rescue

### Formal R2-K v5

Evidence：`docs/runtime-evidence/r2-spatial-k2-pilot-v5-blind/`

```text
denominator = 12
executed = 12
comparable = 8
TIE = 8
incomparable/failed = 4
candidate1 wins = 0
decisive = 0
MPC solved = 900
MPC timeout/fallback = 0/0
collision/offroad = 0/0
latency P50/P95/P99 ≈ 189/909/1319 ms
peak VRAM ≈ 2218 MB
```

三项不可比来自 Guard `SPATIAL_COLLAPSE_ELIGIBLE`；一项来自 actor
position/velocity cold-rebuild comparability。冻结的 `comparable>=10` 与
selection-space 门未过，正式原始 closure 保持：

```text
PILOT_INCONCLUSIVE / REPAIR_REQUIRED / WORLD_GATE_NOT_MET
```

工程终态按用户“可用即可、不继续强化”目标记为：

```text
R2X = COMPLETED_WITH_LIMITS
    / USABLE_FAIL_CLOSED_SPATIAL_K2
    / WORLD_GATE_NOT_MET
R3  = NOT_AUTHORIZED
```

这里的“可用”仅表示 CARLA SIL 中候选生成、Guard、PathManager、MPC 执行链
已实测，非法候选会安全拒绝；不表示 candidate 1 更优、World-ready、实车安全
或公共道路可用。

## 2026-07-27 post-v5 selection-space repair

冻结的 v5 结果保持不改写。本轮修复了两个会系统性制造 TIE 的根因：

1. Guard/runtime diversity 只认相同 Frenet-s 上的候选间
   `max |d0(s)-d1(s)|`；candidate 相对 native 的绝对偏移改为独立诊断。
2. paired cold branch 的 `VLASpeedPlanner` 从实测 ego speed 初始化；旧逻辑从
   0 m/s 只更新一次，把 path target 锁到约 0.1 m/s，导致所有候选满刹。

候选语义同时固定为：

```text
candidate 0 = exact SimLingo/native nominal anchor
candidate 1 = learned defensive residual + learned speed scale
```

v10 heads-only development checkpoint：

```text
run = v10_anchor_speed_contract_development
sha256 = e133815932e7dad680871a4750556df28aa3623c127cb3bd76fae246cb4dfef1
device/steps = CUDA / 2000
formal = false
```

离线 val=22、eligible=8：

```text
eligible Guard = 8/8
eligible sep>=0.5m = 7/8
eligible proposal-valid = 7/8
head_status = OK
```

Evidence：

```text
docs/runtime-evidence/r2x-training/checkpoints/v10_anchor_speed_contract_development/
docs/runtime-evidence/r2x-training/offline_exec_report_v11_anchor_speed_contract_development.json
```

开发 CARLA smoke（不作 formal acceptance）：

```text
docs/runtime-evidence/r2x-live-force-smoke-v10-branch-speed-init-development/
```

- cut-in：candidate 1 unavailable，force 0 正常执行；
- lead-brake：dual-force、MPC 50/50 + 50/50、0 collision；
  - nominal：progress≈12.85 m、clearance≈2.32 m、TTC≈1.43 s；
  - defensive：progress≈6.38 m、clearance≈8.35 m、TTC≈5.12 s；
  - 首次得到真实闭环可排序结果，不再是数值噪声 TIE；
- crossing：路线/控制分叉，但 solver deadline timeout，保持 FAIL。

结论：

```text
R2X = POST_V5_REPAIR_IMPLEMENTED
    / LEAD_SELECTION_SPACE_MEASURED
    / CROSS_MPC_DEADLINE_REPAIR_REQUIRED
    / NEW_BLIND_NOT_RUN
R3 = NOT_AUTHORIZED
```

v10 仅是 development 模型。正式下一步必须先处理 crossing executability，
再冻结一个未看过候选结果的新 blind registry；禁止复用 v2 blind 直接 promotion。

本轮实际验证：

```text
定向 Spatial/Guard/executability：Ran 67, OK
unittest discover -s tests/g4 -t . -q：PASS（loader count=189）
compileall safedrive_foundry scripts tests/g4：PASS
git diff --check：PASS（仅既有 LF/CRLF warning）
CARLA preflight：READY / async / tick owner free
development smoke：2/3 PASS，dual-force=1；crossing timeout 保留
```

最终回归：

```text
R2-X / registry / runner / promotion / X5H directed: Ran 86, OK
R1 contract + execution: Ran 24, OK
compileall: OK
git diff --check: OK（仅既有 Windows LF/CRLF warning）
final CARLA preflight: READY / synchronous_mode=false / tick owner free
```

## 2026-07-28 crossing MPC 修复与 v12 head 优化

### MPC 根因与修复

- 旧 crossing 的“timeout”不是 GPU/CARLA 变慢：速度恢复后，横向加速度
  转角上限收紧得比当前 steering state 的物理恢复速度更快，QP 瞬时不可行。
- MPC 现在构造一个受 steering rate/acceleration 限制的 zero-seeking recovery
  sequence，只在硬上限暂时不可达时最小扩展 angle envelope，保证有一条回到
  `v²κ` 门内的可行路径。
- OSQP 明确 `INFEASIBLE` 时立即返回，不再进入 SLSQP 消耗 deadline 并误记 timeout。
- 20 Hz tracker 禁用 OSQP polishing，避免连续 live pair 下 30 ms 以上尾延迟。
- X5H/development smoke 的 branch gate 现在显式要求 `mpc_timeout=0`；
  旧逻辑只看 solved≥40 与 fallback=0，可能把 46 solved + 4 timeout 错标 PASS。

同一 v10 crossing Evidence 逐 tick 离线重放：

```text
branch 0: OSQP solved 50/50, median/max ≈ 3.54/5.50 ms
branch 1: OSQP solved 50/50, median/max ≈ 3.62/4.06 ms
```

### v12 小 head

- 发现旧 diversity loss 对 batch mean 做 hinge，强分离样本会掩盖同 batch 的
  collapse 样本；已改为逐样本 hinge 后再平均。
- 新增 runtime-decoded Frenet lateral second-difference smoothness loss；
  不注入 runtime template，不降低 0.50 m。
- v11 smooth-only 消融虽将 eligible smoothness mean 从约 0.169 降至 0.097，
  但 lead live collapse，故不采用。
- 采用 v12：

```text
run = v12_per_sample_diversity_development
checkpoint sha256 = 3601441168b8225d670077c03169e46229aca66f6bbd2f4d7fa049b0131b4953
steps/device = 800 / CUDA
diversity target = 0.75 m
smoothness weight = 0.10
formal = false
```

离线 val=22、eligible=8：

```text
eligible Guard = 8/8
eligible sep>=0.5m = 7/8
eligible proposal-valid = 7/8
head_status = OK
```

Development live Evidence：
`docs/runtime-evidence/r2x-live-force-smoke-v13-v12-head-development/`

```text
lead_brake_hard: dual-force PASS
  nominal progress≈12.80m, clearance≈2.37m, TTC≈1.45s
  defensive progress≈6.33m, clearance≈8.35m, TTC≈5.13s
cross_vehicle_clear: dual-force PASS
4 branches total: MPC 200/200 solved, timeout/fallback=0/0, collision=0
```

当前状态：

```text
R2X = MPC_FEASIBILITY_AND_DEADLINE_REPAIRED
    / V12_PER_SAMPLE_DIVERSITY_DEVELOPMENT_PASS
    / LEAD_AND_CROSS_SELECTION_SPACE_MEASURED
    / NEW_BLIND_PENDING
R3 = NOT_AUTHORIZED
```

开发结果不覆盖冻结 v5，不授权 R3。正式下一步仍需全新 blind registry、
formal promotion、X5H 和 R2-K。

R1 首轮曾因测试 mock 缺少新增可选 `driving_feature` 字段出现 20 errors；已将
读取改为向后兼容 `getattr(..., ())`，随后 24/24 PASS。

## 2026-07-28 CARLA spectator 跟随复验

- paired runner 现通过 CARLA Python API 在 anchor 出车时锁定 spectator，
  并在 branch 每个 tick 把镜头更新到 ego 后方 8 m、上方 5.5 m。
- spectator 只改变 CARLA 主窗口观察视角，不进入 observation、VLA、Guard、
  PathManager、MPC、Oracle 或控制 payload。
- watchable development smoke：
  `docs/runtime-evidence/r2x-live-force-smoke-v15-v12-spectator-watchable-development/`
  - `cross_vehicle_clear/seed_a`：`PASS`，dual-force；
  - branch 0/1 均 50/50 MPC solved，0 timeout，0 fallback；
  - branch 0/1 均 51 次 spectator update，0 failure；
  - `spectator_wall_pace_s=0.20` 仅放慢墙钟观察，不改变 0.05 s 仿真步长。
- 全量 `tests/g4`：195 tests，PASS。
- 该结果仍是 `development_live_smoke`，不改变 formal blind / R3 停止点。

## 2026-07-28 Town12 rolling Spatial V2 60 秒复验

- stable rolling runner 新增显式 `v2` development 模式：
  - 周期性 SimLingo forward → Spatial K2 → Guard → PathManager → MPC；
  - 本次固定 candidate 0，不引入 World；
  - defensive-only `SPATIAL_COLLAPSE_ELIGIBLE` 可审计降级为
    `degraded_nominal_only`，不执行 candidate 1；
  - 其他新 bundle Guard reject 时不执行拒绝轨迹，保留上一条合法 committed
    path 并按周期重试；原有 5 s stale-stop 仍有效。
- Town03 失败诊断保留：
  - 首轮约 4.5 s 因 defensive collapse 严格中止；
  - 第二轮约 31.5 s 因 nominal `CURVATURE_ENVELOPE` 严格中止；
  - 未覆盖失败目录。
- Town12 watchable Evidence：
  `docs/runtime-evidence/r2x-rolling-v2-nominal-town12-60s-development-v2/`
  - `DEMO_PASS`，60.0 s / 1200 ticks / 243.12 m；
  - VLA updates 40，PathManager accepted 40；
  - Guard OK 34；6 次仅 defensive collapse，均 nominal-only 审计降级；
  - MPC 1200/1200 solved；
  - collision/lane-invasion/offroad = 0/0/0；
  - CTE RMS ≈0.034 m，route progress ≈243.19 m；
  - inference P50/P95 ≈181.6/193.4 ms，peak VRAM ≈2218 MB。
- 运行后 CARLA：Town12 `READY / async / tick owner free`。
- 该长跑是 development endurance evidence，不替代 formal blind R2-K，
  不授权 R3。

## 2026-07-28 R2-X 最终工程收尾

- 用户目标：获得一个能用的 VLA 模型，不继续为提高 dual-force 比例过度强化。
- 最终 candidate 1 对照：
  `docs/runtime-evidence/r2x-rolling-v2-defensive-town12-60s-development/`
  - 同一 Town12 / 60 s / 6 m/s / v12 head 配置；
  - 第一次 forward 即 `HEAD_COLLAPSE_SEP`；
  - force candidate 1 按合同 fail-closed，未执行拒绝轨迹；
  - 未通过重复运行挑选有利开局，失败 Evidence 保留。
- 最终判断：
  - nominal：Town12 60 s / 243 m / 1200 MPC solved，满足“可用模型”目标；
  - defensive：在部分场景/帧可提供不同路线，但 availability 不可靠；
  - R2-X 工程状态为
    `COMPLETED_WITH_LIMITS / NOMINAL_TOWN12_60S_PASS /
    DEFENSIVE_AVAILABILITY_UNRELIABLE`；
  - World 只允许 development integration，并必须处理
    `NO_ALTERNATIVE → nominal fallback`；
  - formal `WORLD_GATE_NOT_MET`、旧 blind 8/12 TIE 与 R3
    `NOT_AUTHORIZED` 均保持不变。
