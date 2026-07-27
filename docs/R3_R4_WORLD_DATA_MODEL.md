# R3/R4：动作条件 World 数据与 World-V0 实施规格

本文把 `ROADMAP.md` 中的 R3/R4 展开为可直接实施、可测试、可停止的工程规格。
它服从 `docs/PROJECT.md` 的系统边界，不授权当前轮启动采集、训练或 CARLA live。

当前前置事实（2026-07-28）：

```text
R2 longitudinal = COMPLETED_WITH_LIMITS / NO_SELECTION_SPACE
R2-X             = COMPLETED_WITH_LIMITS
                   / NOMINAL_TOWN12_60S_PASS
                   / DEFENSIVE_AVAILABILITY_UNRELIABLE
                   / WORLD_DEVELOPMENT_ONLY
                   / WORLD_GATE_NOT_MET
R3/R4            = PLANNED / NOT_AUTHORIZED
```

因此，R3/R4 的目标不是宣称当前 Spatial K2 已有稳定收益，而是回答一个更窄、可证伪
的问题：

> 在 candidate 1 合法可用时，一个只读 Observable、以 ego candidate 为条件的轻量
> World，能否比 VLA top-1 或简单运动学/规则基线更好地预测闭环结果并排序 K2？

若数据最终没有足够的 candidate 条件信号，允许以负结果关闭；禁止制造候选、缩小
Oracle TIE 带或把缺失 candidate 当作低风险。

---

## 1. 为什么采用“candidate-conditioned World”

当前架构中，World 不生成轨迹，只排序 Guard 已接受的 K2。最匹配的建模问题是：

```text
共享 scene context
  + 查询轨迹 candidate k
  → 在“ego 执行 candidate k”的条件下预测 actor future / risk / utility
```

这与 Conditional Behavior Prediction 的核心形式一致：输入 ego 的查询未来轨迹，
预测其他 actor 在该查询条件下的未来分布。Scene Transformer 也使用可条件化的查询
统一不同预测任务；MultiPath++ 说明稀疏 polyline + actor state 是高效的场景表示；
GameFormer 则证明共享场景编码和交互解码可以联合处理预测与规划。

本项目只采用其中与现有合同相容的最小部分：

- 共享 object/vector scene encoder；
- candidate 轨迹作为显式 query；
- 同一 forward 批量评分 K=2；
- N≤8、M=1、2.5 s、小模型；
- 不复制 UniAD/GameFormer 的完整视觉或分层规划系统。

外部方法只能支持设计选择，不能替代本仓库的 paired CARLA Evidence。

---

## 2. 现有代码能复用什么、还缺什么

### 2.1 可直接复用

| 现有模块 | R3/R4 用途 |
|---|---|
| `evaluation/paired_contract.py` | pair/run/anchor identity、初始 actor 状态、canonical hash |
| `evaluation/comparability.py` | 只让严格可比 branch 进入 pairwise 监督 |
| `evaluation/outcome_metrics.py` | collision/off-road/TTC/progress/comfort 聚合标签 |
| `evaluation/oracle.py` | 冻结的 pair label、winner、TIE、BOTH_BAD |
| `evaluation/k2_spatial_artifact.py` | candidate_id、T10、execution path 与 artifact 绑定 |
| `model/k2_spatial_guard.py` | candidate eligibility；World 不得绕过 Guard |
| `evaluation/runner_contract.py` | manifest、attempt、恢复、不可覆盖 Evidence |

### 2.2 不能直接拿现有 R2 aggregate 当训练集

现有 `BranchOutcomeMetrics` 只有分支级聚合值；`TickRecord` 主要记录 ego/control，
oracle trace 只有 clearance/TTC 等诊断。它没有保存：

- 每个 actor 在 2.5 s 内的逐时刻真值状态；
- actor track 与 anchor observation 的稳定绑定；
- simple road/vector context 的正式 runtime tensor；
- candidate-conditioned actor future label mask。

因此不能从 `minimum_ttc_s`、`clearance` 或 `pair_label` 反推 actor future，也不能用
registry 的 scripted future 代替 CARLA 实际测量。R3 必须先新增独立的
**oracle-only actor future trace**，再构建数据集。

### 2.3 当前 R2 Evidence 的正确用途

正式 Spatial R2-K v5 的 12 pair：

- 8 comparable；
- 8 TIE；
- 4 incomparable/failed；
- decisive=0。

它们适合：

- schema/loader 回归；
- TIE、缺 candidate、incomparable 的合同测试；
- frozen holdout audit。

它们不适合：

- 作为 World 正式训练集；
- 证明 candidate-conditioned signal 已存在；
- 调阈值后重新标注；
- 拆成 50 ticks 当作 50 个独立 pair，虚增样本数。

Town12 60 秒 nominal run 是执行耐久 Evidence，不是 paired action label，也不能直接
当 World 排序训练集。

---

## 3. R3 的交付物与停止点

R3 只负责数据和非神经/弱神经基线，不训练正式 World-V0。

### 3.1 必须交付

```text
ActionBranchDatasetV0 schema + validator
Observable scene encoder input builder
oracle-only actor future collector
immutable dataset shards + manifest + split manifest
data-quality / leakage / action-sensitivity report
persistence + CV + CTRV baselines
observable-only rule/reward baseline
no-action baseline contract
```

### 3.2 R3 关闭标签

| 标签 | 含义 | 是否可进 R4 |
|---|---|---|
| `R3_DATA_READY` | 合同、分割、基线、动作信号均过门 | 是 |
| `R3_DATA_READY_WITH_WEAK_ACTION_SIGNAL` | 可训练，但 decisive/action signal 弱 | 可做受限 R4 |
| `R3_DATA_LIMITED` | 可比或有效 candidate 太少 | 否，先补采集/K2 |
| `R3_LABEL_TRACE_MISSING` | 没有真实 actor future | 否 |
| `R3_LEAKAGE_BLOCKED` | split/oracle 泄漏 | 否 |

达到任一终态后停止，不自动进入 R4。

---

## 4. ActionBranchDatasetV0

### 4.1 样本单位

一个样本对应一个 **anchor/pair**，不是一个 tick，也不是两个互不关联的 branch：

```text
sample
├── identity
├── scene_context(t0 and observable history)
├── candidate[0..1]
├── branch_label[0..1]
├── pair_label
└── masks / audit
```

两条 candidate 必须绑定同一：

- observation fingerprint；
- anchor artifact hash；
- scenario/seed/initial-state identity；
- VLA/head/config/Guard/executor hash。

branch 0/1 冷重建的绝对 CARLA frame 可不同；comparability 继续由现有 pose/yaw/velocity/
phase 合同判断。

### 4.2 坐标系与采样

统一使用 anchor 时刻 ego-local 右手坐标：

```text
origin = ego rear-axle/reference pose at t0
+x     = ego heading forward
+y     = ego left
yaw    = wrap_to_pi(actor_yaw - ego_yaw_t0)
```

所有 ego、actor、road polyline、candidate T10 和 future label 都先变换到该坐标系，
同时保留 world-frame hash 供审计。禁止一个 tensor 混用 CARLA world yaw degree 和
local yaw radian。

固定时间：

```text
history: current + up to 4 past points
future : T=10, dt=0.25 s, horizon=2.5 s
control outcome: existing 50 × 0.05 s primary ticks
```

历史缺失用 mask 和 `time_since_seen` 表示，不复制最后点假装完整观测。

### 4.3 scene_context：仅 Observable

#### Ego

每个历史点建议字段：

```text
x, y, sin(yaw), cos(yaw), vx, vy, ax, ay, yaw_rate, valid, dt
```

#### Actors

最多 N=8，每个 actor 每个历史点：

```text
track_id_hash
type
x, y, sin(yaw), cos(yaw)
vx, vy, ax, ay
length, width
observation covariance summary
time_since_seen
valid / missing
```

actor 选择必须只用 t≤0 的 Observable。默认：

1. 50 m 范围内有效 track；
2. 当前 ego-route corridor 内或距离最近者优先；
3. 用当前相对位置/速度的确定性 risk proxy 排序；
4. 稳定 `track_id_hash` 作 tie-break；
5. 不使用真实未来 TTC、碰撞结果、oracle winner 或 scripted intention。

如果感知链当前没有 actor covariance/track history，R3 必须显式标记
`observable_source=carla_sil_actor_proxy_v0`；不得把它写成量产感知输入。

#### Road/navigation

首版只保存小型 vector context：

```text
route centerline: <=16 points
left/right drivable boundary: each <=16 points
current lane tangent/width/speed limit
traffic-light observable state + relative stop-line
route command/target
polyline/type/valid masks
```

不保存高清 raster，不读取测试时 privileged HD future occupancy。

### 4.4 candidate tensor

每条 candidate 固定 T=10：

```text
x, y, sin(yaw), cos(yaw), v, a, kappa, relative_time
candidate_id
source/proposal/path/timed hash
Guard status
available mask
```

候选轨迹必须是 Guard 后、与 execution spec 完全绑定的最终 T10。World 不允许读取：

- VLA 文字推理或隐藏 teacher cost；
- oracle outcome；
- candidate 的未来 CARLA 执行结果；
- 为某个 split 特制的 availability override。

Guard reject 的 candidate 不进入模型评分。它保留在 dataset audit 中：

```text
candidate_mask=0
unavailable_reason=<exact reason>
ranking_mask=0
```

不得填零轨迹后把它当安全 candidate。

### 4.5 branch labels：严格 oracle-only

每个可执行 candidate 的标签包括：

```text
actor_future[N,T]:
  x, y, sin(yaw), cos(yaw), vx, vy, valid

outcome:
  collision
  first_collision_time / censored
  offroad_fraction
  minimum_ttc / censored
  minimum_clearance
  route_progress
  comfort
  mpc reliability
```

actor future 来自 branch 中 CARLA 实际 actor state，按 future target time
`0.25...2.50 s` 插值/最近邻对齐，并记录原 frame/time。actor 在未来消失时设
`future_valid=0`，不能外推成真值。

这些标签必须：

- 写在 `oracle/` 或 dataset-label 命名空间；
- 带 `oracle_only=true, consumed_by_control=false`；
- 不进入 runtime feature cache；
- 由测试证明 runtime module 无法 import/读取。

### 4.6 pair label 与 loss mask

| R2 状态 | R3 保存 | R4 监督 |
|---|---|---|
| comparable + decisive | 全部 | forecast + risk + pair rank |
| comparable + TIE | 全部 | forecast + risk；rank 使用 tie target/mask |
| BOTH_BAD | 全部 + flag | forecast/risk；relative rank 降权 |
| incomparable | 失败审计 | 不进入监督 loss |
| candidate 1 unavailable | singleton audit | 不伪造 pair rank |
| incomplete/timeout/fallback | 原始失败 | 默认不进正式监督 |

TIE 不等于两条 future 完全一致。其 pairwise target 为等效区间，不能强制 candidate 0
获胜；Oracle 为执行语义返回 top-1 的行为不能被误当作 winner label。

---

## 5. Actor future collector

### 5.1 采集边界

collector 只能挂在 evaluation paired branch，不能进入 runtime 控制路径：

```text
run_anchor
  → freeze observation + K2 artifact
  → branch k cold rebuild
  → at 20 Hz record actual actor state
  → resample oracle label to T10 timestamps
  → finalize branch report
```

不新增 tick owner，不直接从独立 client 调 `world.tick()`。使用现有 paired runner 的
唯一 tick 路径。

### 5.2 稳定 actor identity

CARLA actor id 在冷重建后会变化，不能作为跨 branch identity。使用：

```text
scenario_id + seed_id + registry actor name + role + blueprint
```

并在 t0 验证 bounding box、pose、velocity、script phase。未知动态 actor 采用
episode-local track identity，只在分支内监督，不参与 branch 间逐 actor 差分。

### 5.3 原始 Evidence

每个 attempt 至少新增：

```text
oracle/actor_future_trace.jsonl
oracle/actor_future_trace_manifest.json
```

manifest 绑定：

```text
pair_id / attempt_id / branch
anchor_artifact_hash
registry/model/config/executor hashes
raw trace SHA256
sampling rate / interpolation rule
actor identity table
missing/duplicate/out-of-order counts
```

原始 trace 先于 dataset builder 冻结；builder 只读，不修改 branch Evidence。

---

## 6. 采集 registry 与 split

### 6.1 禁止复用 blind exam 作训练

已有 R2 formal blind registry 和 12-pair结果只能作为冻结审计/holdout。R3 必须创建
新的 development collection registry；训练、验证、测试按 **scenario lineage group**
分割，不能随机拆 frame。

group key 至少包含：

```text
map + family + fixture geometry lineage + actor script lineage + route lineage
```

同一 episode 的多个 anchor、左右镜像/速度扰动、相同 base fixture 的 seed 必须在
同一 split。

### 6.2 分阶段规模

样本目标是工程门，不是论文规模承诺：

| 阶段 | 建议规模 | 用途 |
|---|---:|---|
| schema probe | 24 pairs | collector/loader/hash/mask |
| development set | ≥256 comparable pairs | 基线与小样本 World |
| decisive subset | ≥64，且两候选各有 ≥16 wins | 最低 ranking signal |
| frozen test | ≥72 grouped pairs、≥3 family | 只做最终 R4/R5 评价 |
| formal expansion | 目标 ≥1,000 comparable | 仅开发门有信号后追加 |

若 256 comparable 后仍没有 64 decisive，不继续靠同质 TIE 堆数量；标记
`R3_DATA_READY_WITH_WEAK_ACTION_SIGNAL` 或 `R3_DATA_LIMITED`，检查 candidate、
scenario/horizon，而不是改 Oracle。

### 6.3 场景配比

训练必须包含：

- cut-in / merge；
- lead brake / stop-and-go；
- crossing / yield / unprotected turn；
- clear-road / no-alternative 负例；
- nominal、defensive、轻扰动和风险 candidate；
- 两个候选各有合理赢例，避免学成固定 slot 偏好。

所有 candidate 仍须通过 Guard；训练危险候选不等于允许非法几何。

---

## 7. Dataset 存储、冻结和质量门

### 7.1 建议格式

为避免引入大型数据库依赖：

```text
dataset_root/
├── dataset_manifest.json
├── split_manifest.json
├── quality_report.json
├── shards/
│   ├── shard_00000.npz
│   └── ...
└── source_index.jsonl
```

- arrays 存 compressed NPZ；
- identity/source/provenance 存 canonical JSON；
- 每 shard 和 manifest 都有 SHA256；
- 首次写 exclusive create，正式冻结后只读；
- shard target 可从 128 samples 开始，实测内存后再冻结。

### 7.2 数据质量硬门

```text
schema validation                  = 100%
finite values on valid mask        = 100%
candidate/artifact/hash binding    = 100%
runtime/oracle namespace leakage   = 0
split group overlap                = 0
duplicate sample identity          = 0
comparable supervised samples      >= 256
decisive balanced subset           >= §6.2 minimum
actor future timestamp coverage    >= 95% valid target slots
```

另行报告但不一定阻塞：

- actor count/missingness 分布；
- family/candidate winner/TIE/BOTH_BAD 分布；
- candidate geometric/control separation；
- collision/TTC/off-road base rate；
- source Evidence failure rate；
- per-map、per-family、per-seed counts。

---

## 8. R3 基线

所有基线使用同一 Observable 和同一 split。

### 8.1 Persistence

actor 保持当前位置/朝向不变。它是最低预测基线，不用于 candidate 排序主张。

### 8.2 CV 与 CTRV

- CV：保持当前 `vx,vy`；
- CTRV：保持当前速度和 yaw rate；
- 输出同一 N×T actor future；
- 再用 candidate 与预测 actor 几何计算可观测风险特征。

### 8.3 Observable Rule/Reward

只允许使用：

```text
predicted clearance/TTC
candidate progress/speed/curvature
road-boundary intersection
comfort proxy
```

权重和 tie-break 在 test 前冻结。禁止调用 `evaluation.oracle` 或复用真实 outcome。

### 8.4 No-action

输入 scene，但遮蔽 candidate tensor。它是检验“数据是否真的包含动作条件信号”的
必要对照，不是部署模型。

### 8.5 Action swap/shuffle

- swap candidate 0/1 时，per-candidate 输出必须同样交换；
- 在 batch 内随机 shuffle candidate 时，scene 不变、分数绑定不能串位；
- 用另一 pair 的 candidate 替换时，预测/风险应发生可测变化；
- 若 action-conditioned 与 no-action 无差异，R3 不得声称 World-ready。

---

## 9. R3 验收

### R3-A：合同和 collector

- schema round-trip；
- actor identity 跨 cold rebuild；
- future timestamp 对齐；
- oracle namespace 隔离；
- candidate unavailable/TIE/incomparable masks；
- interrupted attempt 恢复且不覆盖。

### R3-B：24-pair probe

- raw trace 可重建；
- dataset tensor 可重建；
- persistence/CV/CTRV 全部可运行；
- 人工检查 3 类 family 的轨迹叠图；
- 不训练正式 World。

### R3-C：development collection

- 达到 §7.2 门；
- split 冻结；
- baseline report 冻结；
- action-signal report 说明 decisive、TIE、swap effect。

### R3-D：关闭

输出：

```text
r3_closure_report.json
DATASET_FREEZE.md
BASELINE_FREEZE.md
```

记录一种 R3 标签后停止，等待用户授权 R4。

---

## 10. R4 World-V0 架构

### 10.1 固定预算

```text
K=2, T=10, horizon=2.5 s
N<=8, M=1
object/vector latent
target parameters=4M–8M
one shared scene encoding + K candidate queries
```

RTX 4080 16GB 对这一尺寸通常不是显存瓶颈，但正式结论必须来自本机 measured
batch/VRAM/latency；文档不预写“已能跑”。

### 10.2 推荐最小网络

```text
ego history ───── MLP/temporal encoder ─┐
actor histories ─ shared actor encoder ─┼─ scene transformer/set encoder
road polylines ─ polyline encoder ──────┘                 │
                                                         │ shared scene latent
candidate k ─ candidate temporal encoder ─ candidate query┤
                                                         ↓
                                            candidate-conditioned decoder
                                              ├─ actor future head
                                              ├─ collision/TTC/offroad heads
                                              └─ scalar utility score
```

首个配置建议：

```text
d_model=128
n_heads=4
scene_layers=3
candidate_layers=2
decoder_layers=2
dropout=0.1
```

实现后必须登记真实参数量；若小于 4M 或大于 8M，先解释并更新配置，不能用隐藏第二
backbone 补参数。

### 10.3 输出

按 candidate k 输出：

```text
actor_future_mean[N,T,4] = x,y,vx,vy
actor_future_log_scale[N,T,2]   # 可选但推荐，用于校准
collision_logit
offroad_logit
ttc_value + ttc_censored_logit
utility_score
valid/error/calibration state
```

M=1 指一个 joint/conditional mode，不做 diffusion 或大量采样。未来真实多模态能力
属于后续版本，不在 R4 首版范围。

utility 只参与候选排序；它不能绕过 Guard，不能改变 candidate geometry。

### 10.4 K_eff=1 的处理

当前 defensive availability 不可靠，因此 runtime 和训练必须原生支持：

```text
K_eff=2 → World 同批评分并排序
K_eff=1 → NO_RANKING_NEEDED，直接 nominal
K_eff=0 → invalid，交给现有 runtime fail-closed/fallback
```

不允许：

- 给 unavailable candidate 设极低 collision probability；
- 补一个复制 nominal 的 candidate；
- 重采样/模板生成第二条路线；
- 把 singleton 计入 pairwise accuracy 分母。

---

## 11. R4 loss 与标签

建议初始总损失：

```text
L = λtraj L_actor_future
  + λcollision L_collision
  + λoffroad L_offroad
  + λttc L_ttc
  + λrank L_pairwise
  + λtie L_tie_margin
  + λconsistency L_risk_score_consistency
```

具体权重在 development val 上预注册一组初值后冻结；不得看 frozen test 调权。

- actor future：masked Smooth-L1 或 Gaussian NLL；
- collision/offroad：class-balanced BCE，报告未加权 base rate；
- TTC：对有限值回归，对 horizon 外用 censored loss；
- decisive rank：Bradley–Terry/logistic pairwise loss；
- TIE：约束 `|score0-score1|` 在预注册 margin 内；
- BOTH_BAD：风险头正常监督，relative rank 降权；
- incomparable/unavailable：rank mask=0。

slot 0/1 不能拥有不同参数头；共享 candidate encoder，避免学成“永远选 nominal”。

---

## 12. R4 训练顺序

### R4-A：32-sample integrity overfit

目标不是泛化，而是证明实现没断：

- 训练 loss 显著下降；
- decisive sample 可拟合；
- swap candidate 后输出按 slot 交换；
- mask 后梯度/输出不受 padded actor/candidate 影响；
- checkpoint save/load 后逐元素一致。

未过则停止，不扩大数据或网络。

### R4-B：baseline-sized development

- 使用 R3 frozen train/val；
- action-conditioned 与 no-action 同容量对照；
- 1 个主 seed 做调试；
- 仅 val 选择 epoch/初始 loss 权重；
- simple baseline 若更强，保留并诊断。

### R4-C：正式小模型训练

- 建议 3 个 seed；
- AdamW + mixed precision；
- gradient clipping；
- early stopping；
- 本机实测 batch size、P50/P95 step time、VRAM；
- 不与 VLA head 联合训练；
- 不读取图像或 SimLingo hidden feature。

checkpoint 必须保存：

```text
model/optimizer/scaler/RNG states
schema/config/code/data/split hashes
epoch/global_step/best metric
hardware/precision
parameter count
```

### R4-D：frozen test

一次性加载 frozen best checkpoint：

- 不再调阈值/权重；
- 与 persistence/CV/CTRV/rule/no-action 同协议；
- 分母分开报告 decisive/TIE/BOTH_BAD/unavailable/incomparable；
- 生成 R4 closure report 后停止。

---

## 13. R4 指标与门

### 13.1 辅助预测

```text
actor min/ADE/FDE on valid masks
collision Brier / AUROC / base rate
offroad Brier
TTC MAE + censored concordance
ECE/reliability bins（样本够时）
```

ADE/FDE 只是辅助；World 的核心是 candidate-conditioned 排序。

### 13.2 排序

```text
pairwise accuracy on decisive comparable only
balanced accuracy by winner slot and family
ranking regret against frozen oracle
top1→oracle gap recovery
TIE false-decision rate
both-bad selected-risk
```

必须同时报告原始分子/分母。8 个全 TIE 的旧 blind pair 不能产生“100% accuracy”。

### 13.3 因果/合同检查

硬门：

- candidate swap permutation equivariance；
- identity/hash binding 100%；
- unavailable mask 正确；
- no oracle feature leakage；
- NaN/OOM/invalid 显式错误；
- checkpoint resume 可复现。

信号门：

- action-conditioned 在 interaction/decisive subset 上优于或明显不同于 no-action；
- 换 candidate 时 actor future/risk/score 有方向一致的变化；
- 模型不是固定 slot prior。

若信号门失败，标签：

```text
WORLD_ACTION_SIGNAL_NOT_PROVEN
```

这仍可作为 R4 的诚实负结果，但不能宣称 World 有效。

### 13.4 R4 关闭标签

| 标签 | 含义 |
|---|---|
| `R4_WORLD_V0_READY` | 合同、训练、动作敏感、冻结测试均通过 |
| `R4_WORLD_V0_READY_WITH_LIMITS` | 模型可复现，但只在部分 family/弱信号 |
| `R4_WORLD_ACTION_SIGNAL_NOT_PROVEN` | no-action/简单基线不弱于 World |
| `R4_WORLD_DATA_BLOCKED` | R3 数据不足或泄漏 |
| `R4_WORLD_TRAINING_INVALID` | overfit/swap/checkpoint 合同失败 |

R4 结束不自动进入 R5 runtime。

---

## 14. 推荐文件布局

R3/R4 获得授权后，优先新增独立 `world/` 命名空间：

```text
configs/world/world_v0.toml
safedrive_foundry/driving_vla/world/
  contracts.py
  observable_builder.py
  dataset.py
  baselines.py
  model_v0.py
  losses.py
  metrics.py
  checkpoint.py
scripts/
  r3_collect_action_branches.py
  r3_build_action_branch_dataset.py
  r3_audit_action_branch_dataset.py
  r4_train_world_v0.py
  r4_eval_world_v0.py
tests/world/
  test_world_contracts.py
  test_world_observable_isolation.py
  test_world_dataset.py
  test_world_baselines.py
  test_world_model_v0.py
  test_world_checkpoint.py
```

现有 `evaluation` 只负责产生/读取 Evidence 和 oracle labels；runtime 后续只 import
`world.contracts/model_v0`，不得 import `evaluation.oracle`。

---

## 15. 分阶段精确停止规则

```text
R3-A schema/collector tests
  ↓ stop + review
R3-B 24-pair probe
  ↓ stop if trace/mask/identity invalid
R3-C development collection + baselines
  ↓ stop with R3 label
R4-A 32-sample overfit
  ↓ stop if action/swap/checkpoint invalid
R4-B/C training
  ↓ stop with measured checkpoint
R4-D frozen evaluation
  ↓ stop with R4 label
```

任何阶段出现以下情况立即停止：

- Oracle/runtime namespace 泄漏；
- split group overlap；
- candidate/anchor hash 不一致；
- 用 unavailable/failed branch 补标签；
- CARLA comparability 不可靠；
- 连续两次实质修复仍无 action signal；
- 需要改冻结 R2 Oracle 才能得到正结论。

---

## 16. 建议的下一轮口令

当前最小、正确的下一任务不是“训练 World”，而是：

```text
开始 R3-A：实现 ActionBranchDatasetV0 合同与 oracle-only actor future collector，
只做离线单测和 24-pair 采集前检查；不启动正式采集，不进入 R4。
```

---

## 17. 方法参考

以下仅用于设计依据，验收仍以本仓库 Evidence 为准：

1. Tolstaya et al., *Identifying Driver Interactions via Conditional Behavior
   Prediction*, ICRA 2021：
   <https://waymo.com/research/identifying-driver-interactions-via-conditional-behavior-prediction/>
2. Ngiam et al., *Scene Transformer: A Unified Architecture for Predicting
   Multiple Agent Trajectories*, ICLR 2022：
   <https://waymo.com/research/scene-transformer-a-unified-architecture-for-predicting-multiple-agent-trajectories/>
3. Varadarajan et al., *MultiPath++: Efficient Information Fusion and Trajectory
   Aggregation for Behavior Prediction*, ICRA 2022：
   <https://waymo.com/research/multipath-efficient-information-fusion-and-trajectory-aggregation-for-behavior-prediction/>
4. Huang et al., *GameFormer: Game-theoretic Modeling and Learning of
   Transformer-based Interactive Prediction and Planning for Autonomous Driving*,
   ICCV 2023：
   <https://openaccess.thecvf.com/content/ICCV2023/html/Huang_GameFormer_Game-theoretic_Modeling_and_Learning_of_Transformer-based_Interactive_Prediction_and_ICCV_2023_paper.html>
5. Hu et al., *Planning-Oriented Autonomous Driving*, CVPR 2023：
   <https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html>
