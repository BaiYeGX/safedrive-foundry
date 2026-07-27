# R1 实施任务：真实 SimLingo K2

**状态**：`COMPLETED_WITH_LIMITS`（2026-07-24 Guard 首步积分 + 有符号进度复验通过）

**任务边界**：R1 已关闭；不得自动开始 R2/G4A。

**最新复验 Evidence**：`docs/runtime-evidence/r1-real-k2-guard-fix2-2026-07-24/`

**历史 Evidence（冻结，不覆盖）**：

- `docs/runtime-evidence/r1-real-k2/`
- `docs/runtime-evidence/r1-real-k2-guard-fix-2026-07-24/`

本文是 R1 的文件级实施与验收说明。项目总合同见 `PROJECT.md`，VLA 长期设计见
`VLA.md`，当前停止点仍以根目录 `START_TASK.md` 为准。

## 1. R1 最终效果

R1 关闭后，真实链路必须是：

```text
同一 Observation
  → 一次真实 SimLingo CUDA forward
  → native pred_route + pred_speed_wps
  → nominal / conservative 两个纵向动作分支
  → 分别沿同一 native path 重新时间参数化
  → K2 Contract Guard
  → top-1 或强制 candidate 0/1
  → 同一个 VLAPathManager
  → 同一个 VLASpeedPlanner + ConstrainedVLAMPC
  → CARLA
```

冻结输出：

```text
K=2
T=10
time=0.25, 0.50, ... 2.50s
fields=x,y,yaw,v,a,kappa
candidate 0=v1_nominal
candidate 1=v1_conservative
top1_index=0
branch_type=longitudinal_temporal
```

R1 第一版是“真实 SimLingo 输出锚定的 deterministic temporal K2”，不是训练得到的
多模态双头。这个表述必须进入 manifest 和 Evidence，不能写成 learned K2。

## 2. 代码状态

### 2.1 已经可复用的主链（K1，仍有效）

经过真实 CARLA 测量的 K1 链路在 `tests/g3/run_g3_vla_mpc_stable.py`：

```text
NeuralV0Policy.predict_native()
  → NativePathPrediction.path_map_xy       # 原生 20 点空间路径
  → NativePathPrediction.speed_mps         # speed head 转换结果
  → VLASpeedPlanner
  → VLAPathManager
  → ConstrainedVLAMPC
```

R1 在该链上扩展 V1/K2；不以 `run_g3_vla_safety_live.py` 的
`Safety → classic ControlLoop` 作为 pure VLA 验收入口。

### 2.2–2.5 R1 修复前基线 / 历史问题（已关闭，勿当现状）

> **说明**：下列 2.2–2.5 描述的是 **R1 实施前** 的缺陷与验收动机。
> 现状见 **2.2a 当前实现状态** 与最新 Evidence
> `docs/runtime-evidence/r1-real-k2-guard-fix2-2026-07-24/`。

#### 2.2（历史）V1 曾有两个不同实现

| 实现 | 实际来源 | 当时用途 | 历史问题 |
|---|---|---|---|
| `model/v1_policy.py::V1Policy` | fingerprint + 几何锚点 | 旧 G3-04 单测 | 非真实 neural；0.6m lateral bias |
| `model/neural_policy.py::NeuralV1Policy` | 真实 SimLingo forward | 旧 V1 residual | 两候选同 `x/y`，只缩放 `v` |

当时 `test_g3_04_v1.py` 通过不能证明真实 Neural K2；正式路径现已改为
`NeuralV1Policy` + fake/real runtime + R1 合同测试。

#### 2.3（历史）已复现的坍塌和运动学错误

旧 `NeuralV1Policy` + residual 固定输入曾复现：

```text
K=2
max_xy_separation=0.0m
nominal xy == conservative xy
conservative first spatial step 与速度积分矛盾
```

根因是 `_apply_residual()` 只改速度不按新速度重积分弧长。R1 用 `K2Builder`
路径 retime 替代该路径。

#### 2.4（历史）旧 Guard 拦不住矛盾候选

旧 `validate_trajectory_array()` 只查 T/probability/finite；G2 Safety 也不做
`Δs ≈ ½(v_prev+v)dt` 级一致性。R1 已增加 set-level `validate_k2_bundle`
（残差重算 + path 内容 hash + 有符号进度 + 首步积分）。

#### 2.5（历史）旧执行证据不够

当时 stable runner 仅 V0、无 force index、无
`generated → selected → executed → source_id` 闭环。R1 已在 pure stable runner
上增加 `--vla-version v1` / `--force-candidate-index` 与 ID 绑定。

### 2.2a 当前实现状态（R1 关闭后）

正式 K2 链路：

```text
NeuralV0Policy.predict_native()          # 一次 forward
  → K2Builder.build / NeuralV1Policy.predict_bundle
  → validate_k2_bundle (Contract Guard)
  → select_k2(top1|force 0|1)
  → VLASpeedPlanner + VLAPathManager + ConstrainedVLAMPC
```

| 组件 | 路径 | 当前行为 |
|---|---|---|
| K2 构建 | `model/k2_builder.py` | 纵向 temporal retime；`branch_type=longitudinal_temporal`；短 path → `PATH_HORIZON_EXHAUSTED`（不合成假路径） |
| Guard | `validate_k2_bundle` | 从 T10 + `ego_v` 重算残差；首步 `s[0]≈0.5*(ego_v+v0)*dt`；有符号 `ds`；execution `spatial_path_xy` 内容 hash |
| 选择/执行 | `runtime/k2_execution.py` | top1/force fail-closed；`source_id=frame:candidate_id` |
| 正式 Neural V1 | `NeuralV1Policy` | 一次 `predict_native` → K2Builder；`predict_arrays` 兼容包装 |
| Fingerprint V1 | `v1_policy.V1Policy` | **debug only**，非 R1 验收对象 |
| 验收入口 | `run_g3_vla_mpc_stable.py` | `--vla-version v0\|v1`，`--force-candidate-index 0\|1` |

### 2.6 当前概率和 oracle 语义不成立

- `0.62/0.38` 是常量，不是训练或校准得到的概率；
- `oracle_best_of_k(..., expert=None)` 实际返回 probability top-1，不是 oracle；
- 当前 position ADE oracle 会奖励人为 lateral bias，不等价于 G4A 的闭环 outcome
  oracle。

R1 不使用伪置信度和 lateral bias 制造“可分”。真正的 G4A oracle 留给 R2。

### 2.7 speed head 接口还有一个 shape 错配

`NeuralForwardResult.speed_mps` 注释写 `(10,)`，但 official runtime 实际调用
`speed_wps_to_planner_samples(..., use_official_scalar=True)`，返回同一个官方
desired-speed 标量重复 5 次。它是给现有 `VLASpeedPlanner` 取中位数的 planner
samples，不是已经定义好时间语义的 T10 profile。

R1 必须显式区分：

```text
speed_wps_xy: 原始 neural speed waypoint head，10x2，仅保留证据/诊断
speed_mps: K1 planner samples，official 模式当前长度 5
k2 target_speed_profile: R1 明确定义的 T10 目标序列
```

不能依靠当前 canonicalizer 的 `_speed_at()` 猜测 index/time 语义；该函数在可变速度
输入上会跳过 index 0。

## 3. 冻结的最短技术决策

### 3.1 只做一次真实 forward

`NeuralV1Policy` 不再先调用 `NeuralV0Policy.predict_arrays()` 再改数组，而应：

```text
native = NeuralV0Policy.predict_native(obs)  # 只调用一次
bundle = K2Builder.build(native, obs)
```

以下内容必须在一次 forward 后共享：

- 图像 backbone；
- target point；
- native 20-point path；
- checkpoint/model identity；
- source frame 和 Observation identity。

K2 生成不得再次运行视觉主干，不得用不同图像帧拼成两个 candidate。

### 3.2 两个动作分支

第一版配置：

```toml
k = 2
t_steps = 10
dt_s = 0.25
horizon_s = 2.5
conservative_speed_ratio = 0.65
max_accel_mps2 = 2.5
max_decel_mps2 = 3.0
stop_threshold_mps = 0.35
probability_prior = [0.5, 0.5]
top1_index = 0
```

其中：

```text
v_official = finite non-negative official desired-speed scalar
u_nom[i]  = v_official, repeated to an explicit T10 target profile
u_cons[i] = min(u_nom[i], 0.65 * u_nom[i])
```

official 模式从 5 个重复 planner samples 取一致标量，再明确扩展为 10 个 target；
legacy mode 若输入 10 个有限差分样本，也必须通过有版本的 normalize 函数，不得共享
隐式索引约定。

若 nominal 已请求停车，conservative 不得为了制造差异而重新加速。

### 3.3 动力学投影

两分支都从当前可观测 `obs.ego_v` 开始，用同一个 deterministic projector 生成可行动力学
速度序列。对每个 `i=0..9`：

```text
a_des[i] = clip((u[i] - v_prev) / dt, -max_decel, max_accel)
v[i]     = max(0, v_prev + a_des[i] * dt)
a[i]     = (v[i] - v_prev) / dt
s[i]     = s_prev + 0.5 * (v_prev + v[i]) * dt
```

然后更新 `v_prev/s_prev`。第一点的 acceleration 必须相对 `obs.ego_v` 计算，不能像当前
canonicalizer 一样把第一点 acceleration 固定为 0。

R1 不增加 learned jerk head。若输出没有 jerk 字段，不能在 Evidence 中声称 jerk
一致或已校准。

### 3.4 沿原生路径重新时间参数化

先对 `native.path_map_xy` 建立累计弧长表，再分别用 nominal/conservative 的 `s[i]`
插值得到：

```text
x[i], y[i] = interp(native_path, s[i])
yaw[i]      = native path tangent at s[i]
kappa[i]    = wrapped Δyaw / max(Δs, epsilon)
```

这样两候选可以共享空间几何，但在同一时间点位于不同弧长位置，形成真实的 action
condition：

```text
same path geometry
different time-indexed x/y
different v/a
internally consistent yaw/kappa
```

禁止重新引入固定 lateral bias、随机噪声或地图中心线作为第二条候选。

### 3.5 原生路径长度不足

当前原生 path 通常约 19m，而 2.5s 高速轨迹可能需要更长距离。R1 必须显式处理，禁止
把位置钳在 path 末端但保留正速度。

规则：

1. 使用剩余路径和 `max_decel` 计算 stopping-distance envelope；
2. envelope 生效时记录 `path_speed_cap_active=true`；
3. 若当前 ego speed 已无法在剩余 path 内形成一致 T10，返回
   `PATH_HORIZON_EXHAUSTED`，不发布非法 K2；
4. R1 live smoke 固定 `--max-speed 6`，不借本任务宣称 15m/s K2；
5. 不用 HD-map 或 route centerline 延长 VLA 几何。

高速 K2 需要更长 neural path 或专门的可信外推设计，属于 R1 已知限制，不得静默解决。

### 3.6 概率、top-1 和 margin

R1 没有训练 probability head，所以必须使用诚实语义：

```text
probability=[0.5, 0.5]
probability_source=fixed_equal_prior_unscaled
probability_margin=0.0
top1_index=0
top1_rule=explicit_nominal_not_probability_argmax
```

确定性 top-1 不依赖 Python 排序、candidate_id 字典序或虚构的 0.62/0.38 置信度。
World 接入后产生的是独立 `world_score_margin`，不能覆盖 VLA prior 字段。

## 4. R1 数据对象

新增本地 `K2PredictionBundle`，它位于 VLA 层，不修改冻结的 Safety 公共 schema：

```text
observation_identity
model_id
model/checkpoint/config hash
retimer_version
native_path_xy
native_path_hash
candidates[2]                  # TrajectoryArray
execution_specs[candidate_id]  # full spatial path + speed samples
top1_index
diagnostics
```

`execution_specs` 的作用是保留完整 native 20-point path。World 看 T10 timed trajectory；
PathManager 继续看完整空间路径；两者通过 candidate_id、native_path_hash 和
retiming residual 绑定，避免低速 T10 只有 2–4m 时被 PathManager 当成过短路径。

每个 execution spec：

```text
candidate_id
spatial_path_xy
speed_samples_mps
timed_trajectory_hash
native_path_hash
branch_type
```

未来若升级为空间双头，每个 candidate 可以携带不同的 `spatial_path_xy`，执行接口
不需要再次拆房。

## 5. K2 Contract Guard

Guard 在进入 World、force selector 或执行器前运行，返回结构化状态，不只抛出模糊
`ValueError`。

### 5.1 硬检查

- K 必须等于 2；
- 每条 T=10，implicit time 对应 0.25..2.50s；
- 第一点绝对时间为 0.25s、末点为 2.50s，point span 明确为 2.25s；
- candidate_id 非空且唯一；
- probability 有限、范围正确且和为 1；
- 两候选绑定同一 Observation/model/config/native path；
- 所有 `x/y/yaw/v/a/kappa` 有限；
- `v >= 0`；
- acceleration 在配置 envelope 内；
- path 未静默耗尽；
- execution spec 与 candidate_id/hash 一致。

### 5.2 运动学残差

至少输出：

```text
position_integration_error_max_m
acceleration_error_max_mps2
yaw_tangent_error_max_rad
curvature_error_max_per_m
native_path_cross_track_error_max_m
```

首版预登记容差：

```text
position_integration_error_max_m <= 0.05
acceleration_error_max_mps2 <= 1e-6
yaw_tangent_error_max_rad <= 0.05
native_path_cross_track_error_max_m <= 0.05
```

curvature 同时报告 max/P95；若 native path 点噪声导致单点 spike，不能通过扩大所有
阈值掩盖，应使用与 PathManager 一致的 robust 统计并保留原始 max。

### 5.3 差异和 collapse

每帧输出：

```text
mean_speed_gap_mps
final_progress_gap_m
max_position_separation_m
mean_position_separation_m
collapsed
collapse_reason
selection_space_eligible
```

moving frame 的资格：

```text
max nominal speed > 0.75m/s
nominal final progress > 1.5m
```

eligible frame 至少满足：

```text
mean_speed_gap_mps >= 0.25
final_progress_gap_m >= 0.50
max_position_separation_m >= 0.50
```

以下情况可以显式无选择空间，但不能伪装 non-collapse：

- `SEMANTIC_STOP_NO_SPACE`：nominal 和 ego 都处于停车语义；
- `PATH_LIMIT_NO_SPACE`：path envelope 使两分支相同；
- `NUMERIC_COLLAPSE`：本应可分但生成器失败，属于 R1 硬失败。

R1 real-forward/live smoke 必须至少出现一个 eligible frame，且 eligible frame 不得出现
`NUMERIC_COLLAPSE`。不能只用停车画面通过 K2。

## 6. 执行器接入

### 6.1 唯一选择接口

新增纯函数：

```text
select_k2(bundle, mode=top1|force, force_index=None)
  → selected_candidate_id
  → selected execution spec
```

约束：

- top1 永远返回 bundle 的显式 `top1_index`；
- force 只接受 0 或 1；
- 缺失 ID、hash mismatch、Guard 非 OK 时 fail closed；
- 禁止找不到 ID 时执行“第一个可用候选”。

### 6.2 复用 K1 PathManager/MPC

选中后：

```text
speed_decision = VLASpeedPlanner.update(
    selected.execution_spec.speed_samples_mps
)
VLAPathManager.update(
    selected.execution_spec.spatial_path_xy,
    target_speed_mps=speed_decision.target_speed_mps,
    source_id=f"{frame_id}:{candidate_id}"
)
→ ConstrainedVLAMPC.step(...)
```

不创建第二套 PathManager、MPC、CARLA client 或 tick owner。R1 Evidence 必须能对齐：

```text
generated_candidate_ids
selected_candidate_id
executed_candidate_id
PathManager source_id
CARLA frame
```

### 6.3 R1 不以 Safety runner 验收

`run_g3_vla_safety_live.py` 可以复用新的 `NeuralV1Policy`，但它不是 R1 的完成门槛。
R1 核心 smoke 使用 pure VLA stable runner，符合项目当前效果主轴。

## 7. 文件级实施计划

| 文件 | 必须改动 |
|---|---|
| `driving_vla/model/k2_builder.py` | 新增 config、动力学投影、路径 retiming、bundle 和 diagnostics |
| `driving_vla/model/neural_policy.py` | `NeuralV1Policy` 改为一次 `predict_native()` 后构建 bundle；保留 `predict_arrays()` 兼容包装 |
| `driving_vla/model/v1_policy.py` | 移除正式验收对 `_apply_residual`/lateral bias 的依赖；几何锚点明确标成 debug/non-neural |
| `driving_vla/model/canonicalizer.py` | 复用弧长/插值工具；明确 speed sample 时间语义，禁止跳过 index 0 |
| `driving_vla/model/simlingo_runtime.py` | 修正 `speed_mps` shape 注释；不改变已验证的 official K1 scalar 语义 |
| `driving_vla/model/speed_convert.py` | 新增有版本的 K2 T10 target normalization，不把 raw speed waypoints偷换为 K1 速度语义 |
| `driving_vla/adapter/policy_adapter.py` | 增加 K2 set-level Guard 入口和 candidate-specific dynamics metadata |
| `driving_vla/runtime/k2_execution.py` | 新增 top1/force selector 和 PathManager execution binding |
| `config/vla/k2_v1.toml` | 冻结 ratio、动力学 envelope、collapse 阈值和版本 |
| `tests/g3/test_g3_04_v1.py` | 验收对象改为 `NeuralV1Policy` + fake neural runtime，不再用 fingerprint anchor 代表真实 V1 |
| `tests/g3/test_g3_r1_contract.py` | K2 identity、determinism、retiming、collapse、path exhaustion、invalid 测试 |
| `tests/g3/test_g3_r1_execution.py` | force 0/1、ID/hash bind、同一 PathManager/MPC、orphan fail-closed |
| `tests/g3/run_g3_r1_forward_smoke.py` | 一次真实 CUDA forward 的 K2/VRAM/NaN/lineage smoke |
| `tests/g3/run_g3_vla_mpc_stable.py` | 增加 `--vla-version v0|v1` 与 `--force-candidate-index 0|1`，记录 generated/selected/executed |

若实现证明 `TrajectoryArray` 无法携带必要的 hash/metadata，优先放入
`K2PredictionBundle` 和 `PolicyCandidate.dynamics_meta`，不为 R1 破坏 Safety 公共
schema。

## 8. 实施顺序

### R1-A：纯函数和假 runtime

1. 冻结 `k2_v1.toml`；
2. 实现 speed projector；
3. 实现 path retiming；
4. 实现 bundle/diagnostics/Guard；
5. 用 fake neural runtime 证明只 forward 一次。

通过后才能改 runner。

### R1-B：选择与执行绑定

1. 实现 top1/force selector；
2. 将 execution spec 接入现有 PathManager/SpeedPlanner/MPC；
3. 增加 orphan/hash mismatch/collapse fail-closed；
4. 保留 `--vla-version v0` 原路径。

### R1-C：真实 forward smoke

使用真实 SimLingo checkpoint：

- 一次 CUDA forward；
- K2 从同一 native result 派生；
- 无第二次视觉 forward；
- 无 OOM/NaN；
- 记录 latency/VRAM 和 lineage；
- 不使用 CARLA outcome 宣称效果。

### R1-D：CARLA 强制执行 smoke

固定 Town03、seed、spawn/profile 和 `max-speed=6`，分别运行 candidate 0/1。这里只证明
两条都能进入相同执行器，不比较谁更好；paired outcome 比较属于 R2。

### R1-E：K1 回归与关闭

运行全部 G3 单测，并用 `--vla-version v0` 做一次短 smoke，确认 stable K1 入口、
PathManager、speed semantics 和 MPC 没被替换。

## 9. 必做测试

### 9.1 离线单元测试

1. `K=2/T=10/time=0.25..2.50`；
2. 同一 Observation/model/native path identity；
3. fake runtime forward counter 等于 1；
4. fixed input bitwise deterministic；
5. nominal/conservative 顺序和显式 top1；
6. official 5 planner samples 被显式归一化为 T10，K1 official scalar 不变；
7. probability `[0.5,0.5]`、margin 0、来源诚实；
8. moving input 满足差异阈值；
9. semantic stop 显式标记 no-space；
10. position/acceleration/yaw/path residual 通过；
11. path exhaustion 不产生末端正速度复制点；
12. NaN/Inf、wrong T、duplicate ID、hash mismatch 拒绝；
13. `force=0/1` 都解析到正确 execution spec；
14. orphan ID 不执行；
15. candidate 0/1 使用同一 PathManager/MPC 类型和配置；
16. V0/K1 原测试保持通过。

### 9.2 真实模型 smoke

必须记录：

```text
forward_count
latency_ms
peak_vram_mb
candidate_count
candidate_ids
guard_status
collapse diagnostics
checkpoint/config/retimer hash
```

成功标准：

- `forward_count=1`；
- candidate_count=2；
- Guard OK 或显式 semantic-stop no-space；
- 至少一个 moving eligible 样本 non-collapse；
- 无 OOM、NaN、Inf；
- K2 构建不触发第二次模型加载。

### 9.3 CARLA force smoke

candidate 0 和 candidate 1 各自必须：

- 至少一次真实 SimLingo forward；
- 至少 5 次 PathManager accepted update；
- 至少 5 个 MPC tracking tick；
- `selected_id == executed_id == forced_id`；
- `PathManager source_id` 含同一 forced ID；
- 无 silent switch、orphan、NaN 或 OOM；
- 车辆产生可测执行行为，失败也必须保留原因。

不要求 candidate 1 比 candidate 0 更安全或进度更高；该结论属于 R2/R5。

## 10. Evidence 最小结构

R1 Evidence 写入：

```text
docs/runtime-evidence/r1-real-k2/
  config_snapshot.toml
  lineage_manifest.json
  offline_contract_report.json
  real_forward_report.json
  force_0_summary.json
  force_1_summary.json
  k1_regression_summary.json
```

每个 frame/run 至少保存：

```text
run/frame/scenario/observation identity
model/checkpoint/config/retimer hash
native_path_hash
generated candidate ids
top1/selected/executed id
K2 diagnostics
PathManager accept/reason/source_id
MPC mode/status
latency/VRAM
failure classification
```

大数组可单独保存并由 hash 引用，不能只保留聚合 summary。

## 11. 关闭标准

只有以下全部成立才能把 R1 标为 `COMPLETED_WITH_LIMITS`：

1. 正式测试覆盖真实 `NeuralV1Policy` 路径；
2. 一次真实 forward 产生同 Observation K2；
3. 两分支重新时间参数化，不是只改速度字段；
4. moving eligible frame 不坍塌；
5. semantic stop/no-space 显式标记；
6. Guard 能拒绝当前已复现的旧错误；
7. candidate 0/1 可在 pure stable runner 中分别强制执行；
8. generated/selected/executed/PathManager ID 可追溯；
9. K1 单测和短 smoke 不回归；
10. 真实 forward/CARLA smoke 无 OOM、NaN；
11. 实际命令和结果写入 `PROGRESS.md`；
12. `git diff --check` 通过。

`WITH_LIMITS` 的限制必须写明：

- 当前是 deterministic longitudinal temporal branch；
- probability 未训练/未校准；
- 未证明 G4A 选择空间和 World 收益；
- 未完成高速、空间双路径或 Safety live；
- 无实车含义。

## 12. 决策点和停止条件

立即停止并报告，不得静默扩架构：

- 实际 speed head 语义无法支持 T10 retiming；
- native path 在 `max-speed=6` 下仍频繁 horizon exhausted；
- eligible moving frame 仍 collapse；
- force 0/1 不能绑定同一 PathManager/MPC；
- 需要使用 HD-map 中心线或随机 lateral noise 才能通过；
- 真实 forward 需要第二模型副本或发生 OOM；
- 修改 K1 executor 才能让 K2 工作且 K1 无法回归。

此时只允许提出两个后续选项：

1. 延长/升级 VLA spatial path head；
2. 训练明确的 spatial residual/双轨迹 head。

未经上述证据，不启动 LoRA、G4A、World 或 Safety 补课。

## 13. 本任务明确不做

- learned probability head；
- 固定 lateral 0.6m 分支；
- spatial dual-head；
- LoRA/adapter 训练；
- K4/3s；
- G4A paired outcome；
- World Model；
- Safety live；
- 15m/s 高速结论；
- VLA-V2 reasoning。

R1 的最短正确产物是：真实 SimLingo 一次 forward 后，得到两条诚实、可行动力学、
可强制执行的纵向定时轨迹，并完整接入现有 pure VLA PathManager/MPC。

## 14. 实现完成后的验证命令

以下文件和 CLI 参数目前尚未实现；它们是 R1 实现后的固定验证入口，不得在实现前
报告为已运行。

离线：

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

python -m unittest \
  tests.g3.test_g3_04_v1 \
  tests.g3.test_g3_r1_contract \
  tests.g3.test_g3_r1_execution -v

python tests/g3/run_g3_r1_forward_smoke.py \
  --evidence-dir docs/runtime-evidence/r1-real-k2/forward

python -m unittest discover -s tests/g3 -t . -v
python -m compileall -q safedrive_foundry/driving_vla
git diff --check
```

只有离线和真实 forward 通过后才运行 CARLA：

```bash
python scripts/sdf.py sim preflight --json

python tests/g3/run_g3_vla_mpc_stable.py \
  --map Town03 --duration-s 30 --max-speed 6 --seed 11 \
  --vla-version v1 --force-candidate-index 0 \
  --evidence-dir docs/runtime-evidence/r1-real-k2/force-0

python tests/g3/run_g3_vla_mpc_stable.py \
  --map Town03 --duration-s 30 --max-speed 6 --seed 11 \
  --vla-version v1 --force-candidate-index 1 \
  --evidence-dir docs/runtime-evidence/r1-real-k2/force-1

python tests/g3/run_g3_vla_mpc_stable.py \
  --map Town03 --duration-s 30 --max-speed 6 --seed 11 \
  --vla-version v0 \
  --evidence-dir docs/runtime-evidence/r1-real-k2/k1-regression
```

若 preflight 不是 `READY`，按 `AGENTS.md` 的外部阻塞协议停止；不得用 baseline、
fingerprint anchor 或旧 Evidence 代替真实 neural/CARLA smoke。
