# 当前唯一任务：H6-CORA C2 修复版（已授权实施）

## 2026-09-05 本轮优先合同

用户明确授权执行 C2 最快修复计划，本节优先于下方保留的 v1 冻结合同。
修复版 `h6-cora-c2-repair-20260905-v2` 采用 v1 原数据引用＋v3 更正标签＋delta，
不改写 v1 数据或 Evidence。修复实际 repair traces、按 root 去重、路线/红灯标签、
loader 隔离与质量门。只使用 train 做离线筛选；Town03 诊断最多 12 roots，
诊断出现至少 2 个有效 repair-failure roots 和 1 个 offroad root 才正式补采。
正式最多两批，每批 train/validation/calibration/locked 为 24/8/8/8。
新增上限 108 roots、432 branch attempts、14400 秒 CARLA 工作（含启动/恢复），
显存峰值 <=14.5 GiB、空闲磁盘 >=60 GiB。预算不足冻结准确负结果。
跳过旧文件、模型、旧数据 Hash 重算；允许新轨迹/新记录内部标识计算。
质量阈值不变；禁止 C3、calibration 执行、formal、commit/push。
验收覆盖 trace/identity/mask/root 去重/坐标/loader/resume/resource，最后运行相关及全量测试、
compileall、git diff --check，更新五份活动文档并停止。

本轮执行结果：修复代码、v3 labels、offline screening、loader/audit/finalizer 和测试已完成；
通过可恢复的 Windows-side `DefaultEngine.ini` 临时覆盖解决 Town03 admission（原文件先备份，
采集后恢复，安装配置不提交）。单 CARLA、单 tick owner 已完成冻结的 12 个诊断 root、36 个
branch；offroad 有效 root 为 3，但 repair-failure 有效 root 为 0/2，因此不得进入正式批次，
最终保持 `GATE_FAILED`。不得切换 Town05 采集、放宽门槛、修改已冻结 recipe/seed，或把诊断采集
误写成 coverage gate 通过。

以下为保留的 v1 历史冻结合同，不代表本轮仍禁止已授权的独立修复批次。

## 1. 状态与目标

```text
H0-H4 = COMPLETED / VERIFIED / STOPPED
H5 = COMPLETED / VERIFIED / GATE_FAILED / STOPPED
H6 v1/v2 = IMPLEMENTED / MEASURED / NOT_VERIFIED
H6-CORA C0 = COMPLETED
H6-CORA C1 correctness hardening = COMPLETED / STOPPED
H6-CORA C2 counterfactual data = COMPLETED / DATA MEASURED / GATE_FAILED / STOPPED
H6-CORA C3 = NOT_AUTHORIZED / NOT_STARTED
Online Oracle = PROHIBITED
```

C2 已按冻结矩阵完成并在停止点关闭。最终数据包含 351/351 terminal roots、351/351 valid
nominal pairs、1295 个真实 branch outcomes 和 351 次 nominal VLA forwards；pilot gate 通过，
development gate 因真实覆盖不足失败。完整终态见
`docs/runtime-evidence/h6/h6-cora-c2-dev-20260830-v1/final-delivery.json`。不得在本任务内补样、
改阈值、训练 C3、运行 calibration 或消费 reserved formal seeds。

C2 必须交付可运行、可恢复、可审计并带真实 CARLA 数据的完整成品，而不是只交付 schema、
测试或空 collector。固定数据集为：

```text
dataset_id = h6-cora-c2-dev-20260830-v1
data = generated/h6/cora/h6-cora-c2-dev-20260830-v1/
evidence = docs/runtime-evidence/h6/h6-cora-c2-dev-20260830-v1/
```

估计对象是同一 observable anchor 上，把 Guard eligible proposal 单独交给冻结
Safety/repair/MRM/controller 后的 CARLA short-horizon outcome。branch 禁止跨候选 fallback；
World、Oracle、actor future、outcome 和 formal answer 不进入在线/训练 feature view。

## 2. 实施范围

实现并实际运行：

```text
safedrive_foundry/config/h6/cora_c2.toml
safedrive_foundry/data_pipeline/h6/cora/
scripts/h6_cora_collect.py
scripts/h6_cora_audit.py
scripts/h6_cora_finalize.py
scripts/h6_cora_run.py
tests/hybrid/test_cora_counterfactual_data.py
```

允许为复用活动 H2/H3/H6 场景、唯一 Runtime、GPU sampler 或 Safety binding 修改必要的非冻结
共享代码，但不得改写冻结数据、历史 Evidence、Safety 权限、VLA/Classic generator 语义或 ROS 2。
任何额外调用链修改在 `PROGRESS.md` 记录原因。

公开 schema：

```text
safedrive.cora.data_config.v1
safedrive.cora.run_lock.v1
safedrive.cora.root_anchor.v1
safedrive.cora.proposal.v1
safedrive.cora.branch_outcome.v1
safedrive.cora.outcome_labels.v2
safedrive.cora.pair_index.v1
safedrive.cora.feature_view.v1
safedrive.cora.collection_summary.v1
safedrive.cora.data_quality.v1
safedrive.cora.final_delivery.v1
```

## 3. 冻结矩阵与 split

```text
maps = Town01, Town03, Town05
families = free_flow, slow_lead, stopped_lead, cut_in, red_light_hold,
           emergency_lead_brake, aggressive_cut_in, red_light_dilemma,
           cross_traffic_conflict
weather = ClearNoon, CloudyNoon

coverage_pilot = seed 137 / ClearNoon / 27 roots
train = seeds 139,149,151 / both weather / 162 roots
validation = seed 157 / both weather / 54 roots
calibration = seed 163 / both weather / 54 roots
locked_development = seed 167 / both weather / 54 roots
reserved_formal = seeds 173,179 / both weather / 108 roots / NOT COLLECTED IN C2
```

总计 351 个 terminal root attempts，其中 development 324；最少 240 个有效 development nominal
pairs。smoke 是 pilot 内固定三项：Town01/free_flow、Town03/red_light_dilemma、
Town05/aggressive_cut_in，均为 seed 137/ClearNoon。

branch order、slot 和 intervention→base source 用冻结 stable hash 分配。development Expert slot
精确 162/162。每 root 只允许一次 nominal VLA forward；intervention 不重新运行 generator。

## 4. Reset、branch 与 outcome

reset 边界冻结为：position `<=0.05m`、yaw `<=0.5deg`、speed `<=0.10m/s`，同时 route、weather、
world settings、actors、light/script、sensor 和 proposal hash 必须一致。

- nominal Guard `PASS/REVIEW` 才运行 forced-single-candidate branch；`REJECT` 只记录 missingness；
- Safety candidate set 只能含当前 proposal；不能尝试另一个候选；
- ACCEPT/QP/RATO 跟踪 Safety 批准的 executable；MRM/EMERGENCY/HARD_REJECT 执行冻结 braking；
- 合法 terminal 是 50 ticks、MRM 后 `v<=0.05m/s` 连续 10 ticks、route/local goal 完成或碰撞
  当前 tick 已记录；Runtime/actor/reset/identity/cleanup/hash 失败不合法；
- proposal→Safety input→repair executable→executed→applied 的 id/hash 全链必须保存；repair 可以
  改变 trajectory hash，但必须绑定 parent proposal；
- nominal pair 只有两条真实 branch outcome 都有效时 `pair_outcome_mask=1`。

每个 outcome head 独立保存 value、unit、valid mask 和 derivation version：progress/completion、
collision/severity、red-light、off-corridor、TTC/clearance、acceleration/jerk/lateral acceleration、
deadline、Guard/Safety/repair/executable/MRM/fallback-needed、ticks/terminal/cleanup。缺测使用
`null + valid=false`，不能填零。

## 5. 冻结 intervention

每 root 最多两条，一条派生自 Expert、一条派生自 VLA；不适用时记录 `NOT_APPLICABLE`，不得
按 outcome 换 operator。family→operator：

```text
free_flow: speed_scale_up, curvature_bump
slow_lead: delayed_brake, shortened_stopping_margin
stopped_lead: shortened_stopping_margin, obstacle_envelope_approach
cut_in: speed_scale_up, lateral_offset_toward_conflict
red_light_hold: delayed_brake, stop_line_crossing
emergency_lead_brake: delayed_brake, obstacle_envelope_approach
aggressive_cut_in: lateral_offset_toward_conflict, obstacle_envelope_approach
red_light_dilemma: delayed_brake, stop_line_crossing
cross_traffic_conflict: speed_scale_up, obstacle_envelope_approach
```

幅度冻结为：speed `x1.20` 且 `<=15m/s`；brake 延迟 `0.50s`；stopping margin 缩短 `1.0m`；
stop-line 末端越过 `1.5m`；lateral offset `0.60m`；curvature S-bend 最大横移 `0.45m`；
obstacle clearance 减少 `0.30m` 且只在 `t>=1.0s`。

转换后重算 yaw/v/a/kappa 并重新做 T10/dt/horizon/canonical/kinematic/Guard。PASS/REVIEW
进入 core；REJECT 只能进入 `auxiliary_only` Safety/MRM audit，不能进入 core pair、训练或校准。

## 6. 资源与运行边界

```text
root attempts <= 351
branch attempts <= 1404
dataset <= 20 GiB
whole GPU peak <= 14.5 GiB
aggregate collector wall <= 12 h
free disk before/during collection >= 60 GiB
profile = cora_data
```

正式 collector 只能由 `ScenarioRuntime` 持有 tick；不得硬编码 host、直接 `carla.Client()`、
`load_world()`、`world.tick()`、启动第二 tick master 或在 cleanup 中推进 world。ROS
`carla_sync_driver` 与 collector 互斥。无法确认 owner/residue 时写 `NEEDS_USER_ACTION` 并停止。

run-lock 绑定 dataset/config/matrix/formal、代码/worktree、VLA model、Classic/Guard/Safety/control、
CARLA/CUDA/GPU/disk/resource/tick-owner。运行增长的 generated/Evidence 不进入 source identity；
既有 `test_registry.sqlite3` 单独记录并排除，不删除、不修改、不暂存。

## 7. Pilot 与 development gate

pilot 只有以下全部满足才扩展：

```text
terminal roots = 27
valid nominal pairs >= 20
each map valid >= 6
each family valid >= 1
VLA forward count = 27
all 7 operators have >=1 finite canonical proposal
Guard-eligible valid intervention branches >= 9
```

同时 manifest/config/matrix/run-lock/reset/identity/permutation/cleanup/resource 全通过，无
cross-candidate fallback 或第二 tick owner。pilot 失败冻结负结果并停止。

development gate：

```text
terminal roots = 324
valid nominal pairs >= 240
train >=120; validation >=40; calibration >=40; locked_development >=40
each map >=70; each family >=20; each weather >=110
each intervention operator: terminal >=12; Guard-eligible valid core >=6
```

collision、red-light、offroad 的 Guard-eligible core positives 各要求 train `>=8`，validation/
calibration/locked development 各 `>=2`。repairability/executability 每个正负类要求 train
`>=12`，其他三个 development split 各 `>=3`。completion 只报告，稀疏不单独导致 C2 失败。

统计单位始终是 root；branch/intervention/edge/tick 不能膨胀样本量。覆盖不足时冻结
`GATE_FAILED`，不改矩阵、阈值、Guard 或 outcome。

## 8. 验收与停止

离线门：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -m unittest tests.hybrid.test_cora_counterfactual_data -v
python -m unittest tests.hybrid.test_h2_paired_outcomes -v
python -m unittest tests.hybrid.test_h3_challenge_contract -v
python -m unittest tests.hybrid.test_world_v3 -v
python -m unittest tests.hybrid.test_vla75_hardening -v
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry scripts tests
git diff --check
```

真实采集前必须实际运行 doctor、CUDA probe、`sim preflight --json` 和 `df`。只有 `READY` 继续；
`RETRYABLE_FAILURE` 只 ensure 一次再复查一次；`NEEDS_USER_ACTION`、版本/tick/权限/GUI/CUDA 冲突
停止。顺序固定为 materialize/freeze→3-root smoke→27-root pilot/audit→324-root development/audit。

finalization 必须输出 self-hashed `run-lock.json`、`collection-summary.json`、`data-quality.json`、
`final-delivery.json`，记录真实数字、失败、missingness、资源和 C3 `NOT_AUTHORIZED`。

终态为 `C2 COMPLETED / DATA VERIFIED / GATE_PASSED / STOPPED` 或诚实的
`C2 COMPLETED / DATA MEASURED / GATE_FAILED / STOPPED`。更新 `START_TASK.md`、`PROGRESS.md`、
`ROADMAP.md`、`docs/EVIDENCE.md`、必要的 `docs/COUNTERFACTUAL_DATA.md` 后停止；不得自动训练 C3、
运行 calibration 或 formal。
