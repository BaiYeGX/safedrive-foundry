# R2-X：Spatial / Semantic K2 扩展

> **历史路线说明（2026-08-12）**：本文件记录 learned spatial/semantic K2 head 的设计、
> 实现和失败证据。144-anchor V4 pilot 在基线、balanced repair 和受限 q/v LoRA 后仍未
> 通过，状态冻结为 `R2_V4_HEAD_BLOCKED`。该 head 不再是当前 World 主线前置，也不得
> 在原 pilot 上继续调参。当前路线改为 `VLA K1 + Observable Classic Expert → Guard →
> World defer/rank`，见根目录 `ROADMAP.md` 和 `START_TASK.md`。本文件以下“最快路线”、
> formal calibration 与 World handoff 只作历史/可选研究记录，不授权执行。

**状态（2026-07-29）**：
`BASIC_1V1_INTERACTION_REPAIR_REQUIRED / NOMINAL_TOWN12_60S_PASS /
CURRENT_HEAD_NOT_FORMAL / WORLD_GATE_NOT_MET`。

## 0. 2026-07-29 接手摘要：最快完成 R2 的 Basic 1v1 路线

### 0.1 当前事实

当前系统不是“完全没有交通能力”，但只证明了无车长跑和单个脚本 NPC 的短时反应：

- nominal 在 Town12 完成 60 s / 243 m development endurance；
- 最新 Town04 cut-in development smoke 为 12/12 comparable、两分支各
  50/50 MPC solved、collision=0；
- formal paired branch 固定 50 ticks / 2.5 s，随后 cleanup fixture，所以屏幕中车辆
  消失是合同结束，不是 spectator 或 MPC 删除车辆；
- 当前 runtime scene adaptor 只编码第一辆非 ego actor 的 8 维状态，尚未覆盖多车；
- 当前 96 条训练集的 12 条 obstruction 全部被 teacher 标为
  `obstruction_requires_topology_authorization / NO_ALTERNATIVE`；
- 当前 authoring 又把 obstruction actor 放在相邻车道，不是“本车道受阻、相邻同向
  车道可超”的有效题目；
- 当前 checkpoint SHA256
  `8a54108dcfde79ddceb04c688b479bd8c889189a82225e70200eb6f8d6871f0f`
  状态为 `HEAD_TRAINED_NOT_FORMAL`，禁止 formal R2-K / World handoff。

因此，当前 head 会在前车/障碍物场景中频繁选择减速或停车，是标签与场景合同的直接
结果，不应通过调小 0.50 m 门、强制左右偏移或启动 World 来掩盖。

### 0.2 为什么先补 Basic 1v1，而不是先重训 World

World 只排序已有候选，不能制造 `overtake / yield / avoid` 候选。R3/R4 已经完成过
工程闭环，负结果是 action signal 不足；在 R2 不提供稳定选择空间时继续 World，
只会重复 `conditioned ≈ no-action`。

外部实现也采用“行为语义/意图 → 受约束路径”的分层方式：

- [Autoware Behavior Path Planner](https://autowarefoundation.github.io/autoware_universe/main/planning/behavior_path_planner/autoware_behavior_path_planner/)
  把 lane change、静态/动态避让、side shift 分为行为模块，并在生成和执行阶段持续做
  全对象碰撞检查；
- [Autoware Lane Change](https://autowarefoundation.github.io/autoware.universe_planning/main/planning/behavior_path_planner/docs/behavior_path_planner_lane_change_design/)
  明确区分准备、换道与完成阶段；若相邻车道不安全，车辆应提前停车等待；
- [Autoware Path Generation](https://autowarefoundation.github.io/autoware_universe/main/planning/behavior_path_planner/autoware_behavior_path_planner_common/docs/behavior_path_planner_path_generation_design/)
  使用受横向 jerk/加速度约束的平滑 shift，而不是固定横向偏置；
- [MultiPath](https://arxiv.org/abs/1910.05449) 与
  [MTR++](https://arxiv.org/abs/2306.17770) 支持“离散锚点/意图 query + 连续轨迹残差”
  的小 K 多模态设计；这与本项目一次 SimLingo forward + 小 head 的边界一致；
- CARLA 数据采集继续遵守
  [同步与固定步长](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)：
  `0.05 s` fixed step、唯一 tick owner；若后续使用 Traffic Manager，必须同步并冻结
  seed。Basic 1v1 阶段继续使用确定性 actor script，不引入随机背景车流。

### 0.3 冻结的最小行为范围

只完成一辆 ego + 一辆 conflict actor；不扩展密集 traffic、行人、自行车或三车博弈。

| 行为 | 场景 | candidate 0 | candidate 1 | 完整动作门 |
|---|---|---|---|---|
| `FOLLOW_STOP` | 同车道前车减速/停车，无合法绕行 | 跟车/停车 | 可 unavailable | 安全距离、停车不碰撞 |
| `CUT_IN_AVOID` | 左/右一辆车切入 | nominal progress | 减速 + 远离冲突侧 | 无碰撞、无越界、回归 route |
| `YIELD_WAIT` | crossing/merge/窄路迎面 | nominal progress | 冲突点前减速/停车 | 对方通过后重新起步 |
| `OVERTAKE_REJOIN` | 同车道静止/低速车，邻接同向车道空 | 跟车/停车 | 换道、通过、回归 | pass + rejoin + 持续行驶 |
| `CLEAR` | 无 conflict actor | nominal | `NO_ALTERNATIVE` | 禁止空路蛇形 |

### 0.4 最小合同变化：mixed semantic alternative

不得强迫待避/跟车候选产生 0.50 m 横移。V2.1 为 candidate 1 增加冻结字段：

```text
alternative_kind = SPATIAL_AVOID
                 | SPATIAL_OVERTAKE
                 | TEMPORAL_YIELD
                 | NONE
```

- `SPATIAL_*`：保留现有 inter-candidate lateral separation ≥0.50 m；
- `TEMPORAL_YIELD`：允许共享空间 path，但必须满足独立 speed/progress/stop-line
  分离、运动学重算和执行绑定；这不是降低空间门；
- `NONE`：candidate 1 unavailable，force 必须 fail-closed；
- learned confidence 不授予 executability；Guard 根据 artifact 内容与 kind 重算门；
- kind 使用固定语义分类 head（4 logits）+ residual，不引入 Diffusion 或大模型。

为防“看到前车却不知道哪边合法”，新增 versioned、runtime-observable lane topology
side channel：左右相邻 lane 是否存在、是否同向、lane width、junction/crosswalk flag。
该 side channel 只来自当前 CARLA/OpenDRIVE map 与当前 actor 状态，不含 oracle/future。
现有 `observable_scene_v1` 与旧 checkpoint 保留只读，V2.1 不做静默 ABI 复用。

### 0.5 最快实施顺序

1. **B1 合同/单测（离线）**
   - `alternative_kind` schema、Guard V2.1、lane topology observable；
   - single-actor teacher 覆盖上述五种行为；
   - obstruction actor 必须在 ego 当前 lane；只有相邻同向 lane 通过 topology/clearance
     检查时，才允许 `SPATIAL_OVERTAKE`；
   - 旧 Guard V1、V2 artifact、冻结 Evidence 继续可读。
2. **B2 long-horizon observer（离线实现后才开 CARLA）**
   - 独立于正式 2.5 s paired runner；
   - 单 fixture session 运行 15–20 s，末尾才 cleanup；
   - rolling replan、每 tick spectator follow、保存 maneuver trace；
   - PathManager 只保持已选 maneuver 的连续性，不替 VLA/World 做行为选择。
3. **B3 六场景 CARLA smoke**
   - follow-stop、left/right cut-in、yield-wait-resume、left/right overtake-rejoin；
   - 每项先固定 1 pair；clear/no-alternative 另作回归；
   - 任一 collision/offroad、未重启、未通过/未回归，先修根因，不扩大数据。
4. **B4 小规模真实 calibration**
   - 先 120–240 anchors，3 maps，family/direction/available 分层；
   - 先 32 条 overfit，再 heads-only；不启动 1008 大采集；
   - 只有 blind development gate 证明新增行为有效，才扩到已冻结 360-slot formal
     calibration。数据不足时按新 lineage extension，不按 winner 筛选。
5. **B5 formal R2 closure**
   - 冻结 checkpoint 后运行未见过结果的 12-pair core，再运行 84-pair audit；
   - 保留原 0.50 m spatial 门、Oracle deadband、失败、singleton 和 TIE；
   - 在既有 aggregate 门之外，必须逐行为通过下表的 long-horizon completion；
   - 双 candidate 有真实分母，candidate 0/1 都至少有独立胜例，无系统性 MPC
     timeout/fallback 和安全回归。
6. **B6 World handoff**
   - 只有 `R2 = BASIC_1V1_INTERACTION_USABLE / SPATIAL_K2_FORMAL_USABLE`
     后，才绑定唯一 R2 checkpoint 冻结 World campaign；
   - R3/R4 历史负结论保持只读；新 World 训练属于后续任务。

### 0.6 long-horizon 行为完成门

| 行为 | 必须证明 |
|---|---|
| FOLLOW_STOP | 无碰撞；最小间距过门；停止/跟车稳定；无不必要横移 |
| CUT_IN_AVOID | peak lateral 或制动策略与冲突侧一致；无碰撞/offroad；最终回归 route |
| YIELD_WAIT | 冲突点前停止；actor 清空后恢复到 ≥0.75 m/s；不能永远僵死 |
| OVERTAKE_REJOIN | lateral ≥0.50 m；ego 通过 actor；最终 cross-track ≤0.60 m；持续行驶 |
| CLEAR | candidate 1 unavailable 或与 nominal 合法合并；禁止蛇形 |

正式 2.5 s paired oracle 继续用于冷重建可比和局部 outcome；上述 15–20 s 指标只补充
完整行为，不修改或反解释历史 paired Evidence。

### 0.7 新对话恢复口令

```text
继续 R2-B1：按 docs/R2X_SPATIAL_K2.md §0 实现 Basic 1v1 mixed-semantic
合同、lane-topology observable 和 long-horizon observer 的离线单测；保持 CARLA
停止，不训练、不运行 360/1008、不进入 World。
```

下文保留 2026-07-25 起的设计与阶段记录。R3/R4 后来由用户单独授权并已负关闭；
这不改变 R2-X 的冻结结论，也不把 R2-X 重新解释为 World-ready。

R2-F/G、真实 SimLingo feature、Guard/selector、formal blind pilot 和 development
repair 均已完成；最终限制与 Evidence 见 `PROGRESS.md`。

### 曲率语义（强制）

| 阈值 | 含义 |
|---|---|
| PM/Guard `hard_max_abs_curvature=1.0` | 异常几何最终硬拒绝；**不是**车辆可跟踪上限 |
| PM soft `max_abs_curvature=0.20`（稳定 live 可 0.30） | 生产软门；Q90≤soft×1.25；local-max≤max(soft×2.5, soft+0.15) |
| MPC `κ_max=tan(δ_max)/L` | 默认 δ=0.60、L=2.70 → **≈0.253 m⁻¹** |
| `a_y=v²κ` | 须单独门控（默认 max_lat_accel≈1.0 m/s²） |

live 前联合门：steer-κ + a_y + PM Q90/local-max；**不得**仅用 κ≤1.0。
详见 `driving_vla/evaluation/executability_metrics.py` 与
`docs/runtime-evidence/r2x-training/r2x_closure_report.json`。

**边界**：本扩展**留在 R2 研究线内**，但是**新的 Evidence / 合同 / 配置树**。
**禁止**覆盖、改写或重新解释已完成的纵向 R2 pilot：

```text
docs/runtime-evidence/r2-g4a-paired-pilot/   # 只读冻结
结论：COMPLETED_WITH_LIMITS / NO_SELECTION_SPACE（纵向 K2）
```

新 pilot Evidence 必须写入独立目录，例如：

```text
docs/runtime-evidence/r2-spatial-k2-pilot-v2/
```

**不得**因本扩展自动启动 R3、World 数据或 World runtime。
纵向 `k2_v1`、Guard V1、旧 Oracle 阈值与旧 paired pilot 结论保持有效。

权威依赖：

| 文档 | 关系 |
|---|---|
| `docs/R2_PAIRED_ORACLE.md` | 纵向 R2 关闭结论与 orchestrator 合同（只读） |
| `docs/R1_REAL_K2.md` | K2/T10 与执行链基线 |
| `docs/WORLD_MODEL.md` | World 只排序，不生成候选 |
| 本文 | Spatial/Semantic K2 V2 设计与阶段 |

---

## 1. 问题：为什么纵向 K2 全 TIE

当前实现本质是：

```text
同一条 SimLingo spatial path
├─ candidate 0：nominal speed
└─ candidate 1：0.65 × speed
        ↓
同一 PathManager + MPC
```

三个收敛效应：

1. 两条候选的**执行空间路径完全相同**；
2. SpeedPlanner 迟滞、限加速度与 MPC 进一步压缩速度差；
3. 现有 2.5 s 场景通常不会让“早一点/慢一点”改变碰撞、TTC、进度或舒适性**等级**。

更关键：Guard V1 要求每条 `execution_spec.spatial_path_xy` 等于同一个
`bundle.native_path_xy`（见 `k2_builder.py`）。即便模型产出第二条空间路径，
也会被拒绝。数据结构看似支持 per-candidate path，验证逻辑仍冻结为**纵向 K2**。

因此：

> 全 TIE **不是** Oracle 偶然误判，而是**候选空间本身几乎只有纵向 timing 差**；
> 缩小 TIE deadband **不能**替代真正的空间/语义选择空间。

---

## 2. 推荐方案（本扩展唯一默认路线）

**不推荐**：随机采样、固定左右偏移、同一路径加噪、仅收紧 Oracle、一开始上 Diffusion、改全局导航目标、用 World 生成候选。

**推荐**：

> 在一次 SimLingo/VLM 特征前向中，加入两个带明确驾驶语义的 mode query，
> 输出两条**共享导航目标**、但局部空间与速度策略不同的可执行轨迹。

| 槽位 | 语义 | 作用 |
|---|---|---|
| candidate 0 | `nominal_progress` | 保持 SimLingo 主意图与通行效率 |
| candidate 1 | `defensive_alternative` | 画面条件的提前让行、减速、适度横向避让 |

candidate 1 **不是**永远左/右偏。应由场景决定：

- 前车制动：更早减速，路径可基本一致；
- cut-in：提前减速，小幅远离冲突侧；
- 局部障碍：可行驶走廊内平滑侧移；
- crossing/merge：yield 与 progress 的不同时序；
- 普通空路：允许显式 `NO_ALTERNATIVE`，禁止强制蛇形。

“不同路线”= 同一导航目标下的不同**局部 timed trajectory**，
**不是**改变全局 RoutePlanner 目的地（破坏 VLA 提案 → World 排序 → MPC 执行边界）。

---

## 3. 模型结构

保持 `K=2 / T=10 / 2.5 s`。不换 3B/7B 主干；**一次** feature forward。

```text
image + ego + navigation
        ↓
frozen SimLingo / InternVL2-1B backbone
        ↓ 一次 feature forward
shared native route head
        ↓
┌────────────────────┬────────────────────┐
│ nominal mode query │ defensive mode query│
├────────────────────┼────────────────────┤
│ Frenet δs, δd      │ Frenet δs, δd       │
│ speed profile      │ speed profile       │
│ confidence         │ confidence          │
└────────────────────┴────────────────────┘
        ↓
candidate-specific spatial path + timed T10
        ↓
Guard V2 → (rule/World selector) → PathManager + MPC
```

项目侧新增 adaptor（**不直接污染** `simlingo-main`）。
Frenet residual 优先于两套绝对 XY：

- 保留 native 导航方向；
- 易约束纵向单调与曲率；
- 第一段可强制共享，避免 MPC 瞬间换道；
- 残差来自模型特征，不是固定 lateral bias。

参考方向：MultiPath 锚点残差、MTR intention query、CoverNet 可行候选集；
SimLingo Action Dreaming 支持语义条件行为。Diffusion 仅作后续备选。

**PathManager 约束**：当前 ~5 m 切换横向上限约 1 m；首版空间候选限制为
**0.5–1.0 m** 平滑局部侧移，不是完整 3.5 m 换道。

---

## 4. 训练数据（关键）

仅复制同一专家轨迹给两个 head → 几乎必然再次 collapse。

| 标签 | 来源 |
|---|---|
| nominal | 现有专家 / SimLingo 正常轨迹 |
| defensive | Classic / Safety / 离线优化器生成的**可行**替代轨迹 |

要求：

- 替代标签经动力学、道路走廊、曲率、短闭环验证；
- **R2 原始 12-pair 与 Regression 不得进入训练**；
- teacher 可用 privileged 信息；runtime 输入仍仅 RGB / ego / history / navigation。

先只训：mode query、residual MLP、speed head、probability/margin head。
主干冻结；heads-only 不够再小 LoRA。

损失（示意）：

```text
L = L_nominal + L_alternative
  + λdyn L_kinematics + λroad L_drivable_corridor
  + λsmooth L_curvature_jerk + λanchor L_native_anchor
  + λdiv M_ambiguity · L_diversity + λprob L_mode_probability
```

`L_diversity` 必须由 `M_ambiguity` 门控：仅真实替代标签时鼓励分离；
空路 / 停车 / 红灯可 `NO_ALTERNATIVE`，不算故障。

---

## 5. Guard V2（不放宽 V1，新增并列合同）

纵向 Guard V1 **保留不动**。新增 `learned_spatial_k2_v2`：

- 每 candidate 自有 `spatial_path_xy` + content hash；
- **不再**要求 path == 唯一 native path；
- 验证：path↔T10 绑定、ego 连续起步、有符号 progress 单调、
  `v/a/yaw/kappa` 从轨迹重算、native corridor 一致、
  candidate-specific cross-track / 曲率 / 边界、
  两候选同源 observation / 一次 backbone forward；
- 禁止只改 ID/概率/诊断字段；
- candidate 0 可接近 native；
- candidate 1 必须有神经 head lineage，禁止固定偏移伪装。

配置建议：`safedrive_foundry/config/vla/k2_v2_spatial.toml`（新建）。

---

## 6. 阶段划分（R2 内扩展，不进入 R3）

### R2-F：旧 R2 封存与差异诊断

- [x] 关闭报告指标字段映射（`close-r2` + `test_g4a_metric_rollup`，2026-07-25）；
- [ ] 从 11 可比 pair 离线计算轨迹 / 控制 / 闭环差异诊断（不改旧 Oracle）；
- [ ] 文档声明：纵向 K2 → `NO_SELECTION_SPACE` 为**最终**结论，不可因 R2-X 改写。

### R2-G：Spatial K2 V2 合同

- `k2_v2_spatial.toml` + per-candidate geometry schema；
- Guard V2、hash、lineage、V1 兼容回归；
- K1 / 纵向 K2 / 旧 Evidence 全部可读。

### R2-H：双头数据与训练

- 冻结 train/val/regression split；
- nominal/defensive 双标签；
- heads-only 过拟合 + 20–100 step resource smoke；
- 固定输入验证 mode 不 collapse；再有限正式训练。

### R2-I：离线执行验收

每候选：Guard OK、PathManager accept、MPC 可解、走廊内、无曲率尖峰、
首段连续、hash / selected / executed ID 一致。

预登记门槛（示例）：

- ambiguity-eligible 中 ≥70% 有效空间差异；
- max spatial separation ≥0.5 m；
- 不能仅靠 progress/speed 冒充空间 K2；
- 两候选各自质量通过率 ≥90%；
- 普通/停止样本允许 `NO_ALTERNATIVE`。

### R2-J：真实 force smoke

先 3 类：partial obstruction / cut-in / merge-crossing yield。
force 0 与 force 1 分别执行，证明空间路径进入同一 MPC，而非 PathManager 合并。

### R2-K：新 paired pilot

- 新 registry + `docs/runtime-evidence/r2-spatial-k2-pilot-v2/`；
- 复用 immutable manifest、attempt 隔离、一次 forward、冷重建、分母保留、repeat；
- 退出门槛可沿用现有预注册 pilot 规则（comparable≥10/12、decisive、c1 wins 跨 family、repeat 一致等）；
- 诚实标签：`ENTER_WORLD` / `WEAK_SELECTION_SPACE` / `NO_SELECTION_SPACE` / `IMPROVE_VLA`。

---

## 7. Oracle

R2-X **第一轮 pilot 继续用现有 Oracle**，避免同时改候选、场景与评分。

仅当离线诊断证明“闭环数值明显不同但全落 deadband”时，才新建 `oracle.v2`：

1. 阈值由传感/仿真分辨率与控制意义决定；
2. 在新 registry 运行前冻结；
3. 使用新 Evidence 版本；
4. **不得**重新解释旧 R2 的全 TIE。

---

## 8. 与 R3 / World 的关系

```text
纵向 R2（已关）──NO_SELECTION_SPACE──┐
                                     ├─→ 若 R2-X 仍无选择空间 → 条件式 G6 或诚实 World 无增益
R2-X Spatial K2 ──有选择空间？──────┘
                                     └─→ 有 → 再授权 R3 World 数据（World 只排序，不造候选）
```

World **不得**生成候选或标签；候选必须来自 VLA K2。

---

## 9. 明确不做

- 固定左右偏移；随机噪声假多样性；
- 仅缩小 Oracle TIE 阈值“刷过”选择空间；
- 首版 Diffusion / 大采样 decoder；
- 修改全局导航终点伪装空间差；
- 覆盖 `r2-g4a-paired-pilot` 或改写其 `r2_closure_report` 结论。

---

## 10. 验收指针

当前任务入口以根目录 `START_TASK.md` 为准。
启动 R2-X 任一实施阶段需用户明确授权；默认不自动进入。

---

## 11. 研究假设、反证条件与成功含义

### 11.1 主假设

R2-X 检验：

> 在保持同一 Observable Observation、同一 SimLingo 主干、一次特征前向、同一导航
> 目标和同一 MPC 的前提下，带语义 mode query 的 learned Frenet residual head
> 能否稳定提出两条各自可执行、局部策略不同、在困难场景中产生可重复闭环 outcome
> 差异的 K2。

R2-X 不是检验：

- World 是否能学会排序；
- candidate 1 是否普遍优于 candidate 0；
- 空间偏移是否天然更安全；
- 2.5 s 局部轨迹是否等价于全局改道；
- CARLA 结果是否能外推到实车。

### 11.2 三个必须同时成立的“真选择空间”

```text
proposal sensitivity
  候选在模型输出端有场景条件的空间/速度差异
        ↓
executor sensitivity
  两条 candidate-specific path 确实分别进入 PathManager/MPC
        ↓
outcome sensitivity
  至少部分预注册场景产生超出噪声/阈值的闭环差异
```

只满足其中一层都不能声称 R2-X 成功：

| 现象 | 结论 |
|---|---|
| 模型输出不同，执行引用相同 | `EXECUTOR_COLLAPSE` |
| 输出和执行不同，闭环仍全 TIE | `SCENE_OR_HORIZON_INSENSITIVE` |
| 数值 outcome 不同，但 Oracle 全 TIE | `ORACLE_RESOLUTION_REVIEW` |
| candidate 1 靠固定偏移产生差异 | `INVALID_NON_NEURAL_DIVERSITY` |
| 一条经常 Guard reject / off-road | `LOW_QUALITY_MODE`，不是选择空间 |

### 11.3 允许的最终结果

R2-X 可以诚实关闭为：

| 结果 | 含义 | 后续 |
|---|---|---|
| `SPATIAL_SELECTION_SPACE_VERIFIED` | 候选和闭环差异稳定 | 可申请进入 R3 |
| `WEAK_SPATIAL_SELECTION_SPACE` | 有信号但跨 family / repeat 不足 | 只允许一次有实质差异的修复或扩大 |
| `NO_SPATIAL_SELECTION_SPACE` | 合同通过但闭环仍无稳定差异 | 不粉饰；重新评估 horizon/候选质量 |
| `IMPROVE_SPATIAL_K2` | collapse、低质量或 both-bad 主导 | 停在 VLA 候选生成 |
| `BLOCKED_DATA` | 无合法双标签或 split 泄漏 | 不训练 |
| `BLOCKED_EXTERNAL` | CARLA/环境不可用 | 保留已完成离线证据 |

“candidate 1 赢得越多越好”不是优化目标。目标是两条候选都具有使用价值，并在不同
场景形成可排序差异。

---

## 12. 术语与身份

### 12.1 固定术语

| 术语 | 定义 |
|---|---|
| native path | 原始 SimLingo route head 的 20 点局部空间路径 |
| anchor frame | native path 建立的局部 Frenet 坐标系 |
| mode query | 同一次主干 forward 内，对 nominal/defensive 槽位的可学习查询 |
| residual head | 输出相对 native path 的 `δs/δd` 与速度残差 |
| proposal path | residual 解码后的 candidate-specific 20 点空间路径 |
| timed trajectory | proposal path 上重参数化后的 T10 `x/y/yaw/v/a/kappa` |
| available | 该场景存在合法、可执行的第二候选 |
| eligible | 该 anchor 适合用于选择空间统计 |
| collapse | eligible 样本上候选差异未达冻结阈值 |

`available` 与 `eligible` 不同：

- `available=false`：模型明确表示没有合法防守替代；runtime 只执行 top-1；
- `available=true, eligible=false`：有两条合法候选，但当前状态不适合纳入选择空间统计；
- `available=true, eligible=true`：Guard 与 R2-X 统计均要求候选达到预登记差异。

### 12.2 必须绑定的身份

每次 K2 V2 输出至少绑定：

```text
run_id / frame_id / scenario_id / CARLA frame
observation content hash
model base revision / checkpoint hash
spatial-head checkpoint hash
config hash / schema version
backbone_forward_id
native path content hash
candidate 0/1 path hash
candidate 0/1 timed trajectory hash
candidate 0/1 mode id
```

两条候选必须具有相同的 `backbone_forward_id`。禁止分别 forward 后拼接成 K2。

---

## 13. K2 V2 数据合同

### 13.1 版本策略

不原地改变 `K2PredictionBundle` V1 语义。首选以下两种之一，并在 R2-G 只选一种：

1. 新增 `K2PredictionBundleV2` / `K2ExecutionSpecV2`；
2. 为现有类型增加显式 `schema_version`，由 V1/V2 verifier 分派。

不允许用 `if branch_type != longitudinal` 在 V1 verifier 中零散跳过检查。V2 必须有
独立、可单测的验证入口。

### 13.2 建议结构

```text
K2PredictionBundleV2
  schema_version = safedrive.k2.spatial.v2
  observation_identity
  observation_content_hash
  model_id
  base_checkpoint_hash
  spatial_head_checkpoint_hash
  config_hash
  backbone_forward_id
  native_path_xy
  native_path_hash
  candidates[2]
  execution_specs[candidate_id]
  top1_index
  probability_source
  guard_status / guard_reasons
  set_diagnostics
```

每个 candidate：

```text
candidate_id
mode_id                    # nominal_progress / defensive_alternative
available
availability_reason
probability
uncertainty
points_xy_yaw_v_a_kappa[T=10]
frenet_s[T]
frenet_d[T]
native_anchor_hash
proposal_path_hash
timed_trajectory_hash
head_lineage
```

每个 execution spec：

```text
candidate_id
spatial_path_xy            # candidate-specific 20-point/full path
speed_samples_mps
spatial_path_hash
timed_trajectory_hash
native_anchor_hash
branch_type = learned_spatial_semantic
```

### 13.3 availability 语义

`NO_ALTERNATIVE` 不应伪造成两条可执行候选：

- candidate 0 始终 available；
- candidate 1 可 `available=false`；
- selector/World 遇到 candidate 1 unavailable 时必须确定性返回 candidate 0；
- R2-X paired pilot 只在 registry 预先声明的 ambiguity-eligible anchor 上要求
  candidate 1 available；
- 不允许在看到 candidate outcome 后把 anchor 改成 unavailable。

对外适配到 `PolicyCandidateSet` 时必须保留 availability，不能由 adapter 静默改回
`True`。

---

## 14. Frenet residual 的构造合同

### 14.1 坐标定义

以 native path 的累计弧长为 `s`，单位法向为 `n(s)`。候选路径由：

```text
s_k[i] = s_anchor[i] + Δs_k[i]
p_k[i] = p_native(s_k[i]) + d_k[i] · n(s_k[i])
```

生成。模型不直接输出任意 map-frame 绝对坐标；它输出受界的增量：

```text
Δs_k[i] = cumulative_softplus(raw_Δs_k[i]) 或等价单调参数化
d_k[i]  = envelope[i] · d_max · tanh(raw_d_k[i])
```

`envelope` 在近场从 0 平滑增长，保证轨迹不在当前车体处瞬移。实现可用固定光滑
envelope，但 `raw_d` 的符号和幅值必须来自模型，不能把固定左/右偏当候选。

### 14.2 近场连续与远场分离

首版建议把阈值分成三类：

| 类别 | 建议初值 | 说明 |
|---|---:|---|
| first-step position residual | ≤0.05 m | 与 V1 运动学残差一致 |
| 前 0.5 s candidate 间横向差 | ≤0.20 m | 避免控制瞬跳 |
| ambiguity-eligible 最大空间差 | ≥0.50 m | 与旧 collapse 量尺一致 |
| 5 m probe 横向切换 | ≤1.00 m | 对齐当前 PathManager |
| 普通最大横向 residual | ≤1.00 m | 首版不做整车道换道 |

这些是**提案值**，不是已经冻结的事实。R2-F 必须先读取 PathManager/MPC 实际配置，
R2-G 再把最终值写入 TOML 与 manifest。看到新 pilot outcome 后不得改阈值。

### 14.3 时间参数化

每条 proposal path 各自进行时间参数化：

```text
a_i = clip((u_i - v_prev)/dt, -max_decel, max_accel)
v_i = max(0, v_prev + a_i·dt)
s_i = s_prev + 0.5·(v_prev + v_i)·dt
```

然后在**该 candidate 自己的 proposal path**上采样 `x/y/yaw/kappa`。禁止：

- 用 candidate 0 的空间 path 配 candidate 1 的速度；
- 只改 T10 点而 execution spec 仍指向 native path；
- 先生成不一致轨迹再依靠 Guard diagnostics 声称通过。

---

## 15. Guard V2 详细验收

### 15.1 Set-level

- `K=2 / T=10 / dt=0.25 / horizon=2.5s`；
- ID 唯一、top-1 合法、概率有限且声明来源；
- observation/model/config/forward identity 完整；
- 两候选同 `backbone_forward_id`；
- base/head checkpoint hash 可追溯；
- candidate 1 availability 与 reason 一致；
- V1 bundle 不得误走 V2 verifier，反之亦然。

### 15.2 Per-candidate

- 所有数值 finite；
- `v>=0`，加减速度在冻结 envelope；
- 从实际 T10 重算首步和逐步位置积分；
- 有符号 path progress 单调；
- 从实际点重算 yaw、curvature、acceleration；
- `spatial_path_hash` 与内容一致；
- T10 点投影到自己的 execution path，cross-track 低于阈值；
- path 长度足够，不在终点复制正速度点；
- PathManager 的 forward ratio、self-intersection、curvature 检查可通过；
- candidate 1 的 `head_lineage` 不能是 debug/fixed-bias/noise。

### 15.3 Candidate-relational

当 `eligible=true`：

- max spatial separation 达到冻结下限；
- 差异不能只存在于 probability/ID；
- defensive candidate 的方向不得由 scenario ID、seed 或 candidate ID 硬编码；
- 两条 path 共享近场连续前缀；
- 两条都满足质量门，不能用“一条坏轨迹”制造 oracle gap。

当 `available=false`：

- 不要求 diversity；
- candidate 1 不得被 selector 强制执行；
- Evidence 必须记录 `NO_ALTERNATIVE` 原因。

### 15.4 Fail-closed 原因码

至少预留：

```text
FORWARD_ID_MISMATCH
HEAD_LINEAGE_INVALID
PROPOSAL_PATH_HASH_MISMATCH
TIMED_PATH_BINDING_MISMATCH
FRENET_PROGRESS_NEGATIVE
FRENET_LATERAL_ENVELOPE
NEAR_FIELD_DISCONTINUITY
SPATIAL_COLLAPSE_ELIGIBLE
CURVATURE_ENVELOPE
PATH_MANAGER_PRECHECK
ALTERNATIVE_UNAVAILABLE
```

`ALTERNATIVE_UNAVAILABLE` 是可预期状态，不应和损坏 bundle 使用同一个错误码。

---

## 16. 双标签数据合同

### 16.1 样本结构

建议新增 `SpatialK2TrainingSampleV1`：

```text
sample_id / capture_run_id / frame_id
split_id / town / route_id / scenario_family / weather
front_rgb_uri
ego current + history
navigation targets / route observable
native SimLingo path + hash
nominal target trajectory
defensive target trajectory or null
alternative_available
ambiguity_type
teacher source + version + config hash
teacher privileged fields declaration
quality metrics
leakage audit fields
```

图像、标签和 identity 必须内容寻址。不得仅依赖容易漂移的绝对路径。

### 16.2 防守标签生成优先级

按以下顺序选择来源，R2-H 启动前必须完成资产审计：

1. 已有合法、许可清晰的 SimLingo Action Dreaming / instruction-action 配对；
2. 本项目 CARLA 采集的 Classic/Safety teacher 替代轨迹；
3. 受走廊和动力学约束的离线 Frenet lattice/optimizer。

第三项可以生成 teacher，但正式 head 输出必须依赖 scene feature。禁止把 lattice 的
固定左/右模板直接作为 runtime candidate。

### 16.3 Teacher 生成流程

```text
Observable anchor
  → 生成 nominal teacher
  → 生成多个 defensive teacher proposals
  → 动力学/走廊/曲率硬过滤
  → 短闭环或可验证 rollout
  → 依据预注册 teacher cost 选一个替代
  → 写入标签与完整 lineage
```

teacher cost 可用 collision/off-road/clearance/progress/comfort，但必须：

- 只在 train split 上使用 privileged future；
- 与 R2-X evaluation Oracle 分开版本；
- 不从旧 R2 Regression/Evidence 抄 outcome；
- 不把 scenario/seed ID 输入模型；
- 未找到合格替代时写 `alternative_available=false`，不得降级成固定偏移。

### 16.4 什么叫“有价值的替代标签”

防守标签必须同时：

1. 硬约束可执行；
2. 与 nominal 有超过训练噪声的动作差；
3. 不改变全局导航目标；
4. 在对应 ambiguity 中降低冲突风险或提供不同让行策略；
5. 不以显著 off-road、反向行驶或不合理停滞换取差异。

### 16.5 Split 与泄漏

至少按以下轴分组后切分：

```text
town
route_id
scenario_family
actor script family
failure cluster
teacher template family
```

旧 R2 registry、repeat pair、R1 Regression 与未来 R2-X pilot 全部进入 regression/holdout，
不得进入 train 或用于 early stopping。

训练前必须生成：

```text
split_manifest.json
sample_manifest.json
teacher_manifest.json
leakage_report.json
dataset_card.json
```

任一 hash 漂移或 overlap 非零，状态为 `BLOCKED_DATA`。

---

## 17. 模型与训练细化

### 17.1 最小模型

首版只增加：

- 2 个 mode query 组；
- shared 或轻量 mode-specific Frenet residual MLP；
- candidate-specific speed residual head；
- availability logit；
- 可选 probability/margin logit。

建议先使用 shared MLP + mode embedding，避免两个完全独立 head 各自过拟合。只有
消融证明 shared head collapse，才允许 mode-specific final layer。

### 17.2 冻结策略

训练顺序：

1. 冻结全部 SimLingo/VLM 参数；
2. 仅训练 query + residual/speed/availability head；
3. 小样本过拟合通过后做 20–100 step smoke；
4. heads-only 达不到 holdout action sensitivity 时，才允许一组小 LoRA；
5. 禁止同时解冻视觉主干、语言主干和新 head。

任何 LoRA 都必须记录 target modules、rank、alpha、dropout 和 base hash。

### 17.3 Loss

建议具体拆分：

```text
L_nom_xy       nominal trajectory regression
L_alt_xy       defensive trajectory regression（availability mask）
L_speed        两模式 speed profile
L_dyn          Δs-v-a 一致性
L_yaw_kappa    几何导数一致性
L_corridor     超出 teacher/observable corridor 惩罚
L_anchor       nominal 不无故偏离 native
L_div          ambiguity 样本上的最小模式距离
L_avail        alternative availability 分类
L_prob         可选；仅有合法 supervision 时启用
```

总损失：

```text
L = L_nom_xy
  + M_alt·L_alt_xy
  + λv·L_speed
  + λdyn·L_dyn
  + λgeo·L_yaw_kappa
  + λroad·L_corridor
  + λanchor·L_anchor
  + M_ambiguity·λdiv·L_div
  + λavail·L_avail
  + λprob·L_prob
```

禁止对所有样本无条件施加 repulsion。那会把 mode collapse 转换成 spurious mode。

### 17.4 概率语义

R2-X 首版可保持：

```text
top1_index = 0
probability_source = uncalibrated_head 或 fixed_prior
```

未做 calibration 前不得把概率解释为安全概率。paired Oracle 不读取候选概率。

### 17.5 训练门槛

按顺序关闭：

| Gate | 必须证明 |
|---|---|
| overfit-8/16 | 小样本能同时拟合 nominal 和 defensive，不 collapse |
| smoke-20/100 | 无 NaN/OOM；checkpoint 可恢复；资源记录 |
| val-quality | 两候选独立质量门通过 |
| val-sensitivity | ambiguity holdout 上空间差达到阈值 |
| regression | K1、纵向 K2、正常驾驶不回归 |

任何 gate 失败都不得直接进入 CARLA force smoke。

---

## 18. R2-F 离线差异诊断规范

### 18.1 输入

只读：

```text
docs/runtime-evidence/r2-g4a-paired-pilot/run_set_manifest.json
run_set_report.json
11 个 comparable pair 的 anchor/branch traces
```

禁止重跑、改写或补算缺失 live 数据。诊断输出写到新目录：

```text
docs/runtime-evidence/r2x-selection-space-diagnostic/
```

### 18.2 四层指标

**Proposal：**

- candidate T10 ADE/FDE、max/mean XY separation；
- Frenet `Δs/Δd`；
- speed/accel/progress profile gap；
- native/execution path hash 是否相同；
- collapse reason 与 eligibility。

**Executor input：**

- PathManager raw/committed path hash；
- 5 m lateral/heading probe；
- SpeedPlanner raw/calibrated/target gap；
- MPC reference path、reference speed gap。

**Control：**

- steer/throttle/brake MAE、RMSE、max gap；
- first-divergence tick；
- MPC solution/fallback/status 差异。

**Outcome：**

- collision/off-road；
- TTC/clearance；
- progress；
- jerk；
- continuous delta 相对 Oracle deadband 的比例。

### 18.3 诊断分类

每 pair 输出一个 primary bottleneck：

```text
PROPOSAL_SAME_PATH
PROPOSAL_TEMPORAL_ONLY
SPEED_PLANNER_COMPRESSION
PATH_MANAGER_MERGE
MPC_CONTROL_COMPRESSION
SCENE_INSENSITIVE
ORACLE_DEADBAND_ONLY
INCOMPLETE_EVIDENCE
```

总报告必须包含 counts、每 family 分布和原始 pair 指针，不能只写一句“纵向已死”。

### 18.4 R2-F 退出

- 11 个 comparable pair 全部有诊断行；
- 缺失字段显式标记，不以 0 填充；
- 量化主要压缩发生在哪一层；
- 给出 Guard V2/PathManager 阈值建议，但不冻结新 pilot outcome 阈值；
- 不开始模型训练。

---

## 19. R2-X 场景 Registry V2

### 19.1 设计原则

新 registry 不是把旧 6 场景换名重跑。每个场景必须预先说明：

- 为什么存在至少两条合理局部策略；
- nominal 与 defensive 的预期差异维度；
- actor script 为什么不依赖 candidate ID；
- 2.5 s 内为什么能观察到差异；
- 哪些条件使 anchor 不 eligible。

### 19.2 建议 family

| family | candidate 0 | candidate 1 | 主要差异 |
|---|---|---|---|
| partial obstruction | 正常通过 | 走廊内平滑侧移/让行 | spatial + clearance |
| directional cut-in | 保持进度 | 远离冲突侧 + 提前减速 | spatial + timing |
| merge/yield | 抢占合法 gap | 明确 yield | progress + TTC |
| crossing conflict | 正常通过 | 提前建立安全间距 | timing，可伴小侧移 |

首版仍建议 6 scenario × 2 seed = 12 pair。左右方向需要 counterbalance，防止 defensive
head 学成固定符号：

```text
left-origin conflicts ≈ right-origin conflicts
branch order 0→1 / 1→0 counterbalanced
```

### 19.3 冻结前 dry-run

允许检查：

- spawn 可行；
- actor script；
- 初始态可重建；
- route/navigation；
- 无 candidate outcome 的几何可用性。

禁止：

- 看 candidate 0/1 闭环结果后换 fixture；
- 按模型失败方向选择左右场景；
- 根据 Oracle winner 调整 seed。

registry 一旦产生 candidate outcome 必须冻结版本；任何 fixture 修改升版本并重新
冻结分母。

---

## 20. 阶段级输入、产物与停止门

### R2-F：Selection-space diagnostic

**允许修改：**诊断脚本、离线测试、本文、PROGRESS。
**产物：**

```text
r2x-selection-space-diagnostic/
  manifest.json
  pair_diagnostics.jsonl
  aggregate.json
  report.md
```

**停止：**报告完成后停，不自动进入 R2-G。

### R2-G：V2 contract and Guard

**输入：**R2-F 报告、PathManager/MPC 配置、V1 contract。
**产物：**

- V2 schema/types；
- Frenet codec；
- Guard V2；
- selector/execution binding；
- TOML；
- V1/V2 compatibility tests。

**离线验收：**

- 两条不同合法 path 通过；
- hash spoof、fixed-bias lineage、path/T10 mismatch、倒序、越界、曲率尖峰拒绝；
- candidate 1 unavailable 确定性回 top-1；
- V1 所有 R1/R2 regression 继续通过。

**停止：**不训练、不 CARLA。

### R2-H0：Data availability gate

先只做资产和许可审计：

- 本机是否有合法 SimLingo driving/Action Dreaming 数据；
- 本项目采集数据规模、字段和 split；
- teacher 可复用程度；
- 预计采集/生成成本。

输出 `R2H_DATA_DECISION`：

```text
REUSE_EXISTING
COLLECT_PROJECT_CARLA
GENERATE_TEACHER
BLOCKED_DATA
```

没有 data decision 不写训练循环。

### R2-H1：Dataset/teacher

先冻结 schema/split/teacher config，再生成标签。
退出条件：双标签质量抽检、泄漏报告、数据 card、hash 完整。

### R2-H2：Heads-only training

按 overfit → smoke → validation → regression 顺序。
每个 checkpoint 独立目录，不用失败 checkpoint 覆盖前一版。

### R2-I：Offline execution

使用 holdout Observation：

```text
one feature forward
→ K2 V2
→ Guard V2
→ force candidate 0/1
→ separate PathManager/SpeedPlanner/MPC offline state
```

禁止共享有状态 executor 导致候选顺序污染。记录执行引用 hash 与控制差异。

### R2-J：Live force smoke

每类先 1 pair，最多 3 pair：

- preflight READY；
- candidate 0/1 各一个独立冷重建；
- 同 anchor artifact；
- 每 branch 50 ticks；
- 空间引用和控制输出可区分；
- cleanup 后再下一 pair。

失败最多做两次有实质差异的修复。未通过不得直接跑 12-pair。

### R2-K：Frozen paired pilot

复用 R2 runner 合同，但 schema/identity 必须升级：

- frozen registry；
- immutable run-set manifest；
- planned attempt IDs；
- no-auto-retry；
- continue-all；
- checkpoint；
- failed/incomparable 进分母；
- repeat plan 在 outcome 前冻结；
- branch 禁止再次 VLA forward。

R2-K 完成后停止，由用户决定是否进入 R3。

---

## 21. R2-I/J/K 量化门槛

### 21.1 Proposal quality

在预登记 ambiguity-eligible holdout：

- ≥70% anchor 的 max spatial separation ≥0.50 m；
- 不得仅靠 speed/progress gap计为空间差；
- candidate 0/1 Guard pass 各 ≥90%；
- fixed-input inference 可复现；
- candidate 1 横向符号与冲突方向有关，而非恒定左/右。

这些门槛在 R2-G/H 数据 outcome 前冻结；若数据规模不足，报告 counts 与置信区间，
不得伪造高精度百分比。

### 21.2 Executor sensitivity

force smoke 至少证明：

- selected/executed/path source ID 完整绑定；
- committed path hash 在两 branch 间不同；
- MPC reference path 在两 branch 间不同；
- 至少一个控制量在合理 tick 内出现可解释差异；
- 无 path merge、fallback 或 candidate 顺序污染。

### 21.3 Pilot gate

建议沿用纵向 R2 的 counts-first 规则：

```text
denominator = 12
comparable >= 10
decisive >= 4
candidate1 wins >= 2
candidate1 wins across >= 2 families
repeat 2/2 label-consistent
```

同时增加质量限制：

- `both_bad_rate < 0.50`；
- Guard/availability 失败不能选择性移出分母；
- 至少一个 decisive pair 由 spatial/clearance 层决定，不能全部只是 progress；
- 失败和负收益完整保留。

`ENTER_WORLD` 仍不是“candidate 1 更强”，只表示存在稳定可排序动作空间。

---

## 22. Oracle 与评价版本控制

### 22.1 第一轮默认

沿用：

```text
collision
→ off-road
→ TTC bucket
→ clearance
→ progress
→ comfort
```

及现有 deadband。这样 R2-X 与纵向 R2 的选择空间量尺可比。

### 22.2 何时允许 oracle.v2

只有 R2-F 或 R2-I 的**独立诊断**显示：

- proposal 和 executor 均有稳定差异；
- continuous outcomes 超过仿真/控制噪声；
- 但现有分桶系统性吞掉有控制意义的差异；

才提出 oracle.v2。

oracle.v2 必须先写：

```text
metric semantics
threshold rationale
measurement resolution
tie policy
both-bad policy
version hash
frozen-before-outcome proof
```

不能在同一 12-pair 上调阈值后重新贴 `ENTER_WORLD`。

---

## 23. Evidence 布局

建议：

```text
docs/runtime-evidence/
  r2x-selection-space-diagnostic/
  r2x-data-audit/
  r2x-k2-v2-offline/
  r2x-force-smoke/
  r2-spatial-k2-pilot-v2/
    registry/
    run_set_manifest.json
    run_set_checkpoint.json
    run_set_report.json
    pairs/<pair_id>/attempt_N/
      anchor/
        observation_manifest.json
        k2_anchor_v2.json
        model_manifest.json
      branch-0/
      branch-1/
      pair_manifest.json
      pair_oracle.json
    repeat_audit_plan.json
    repeat_audit_report.json
    oracle_table.json
    r2x_closure_report.json
```

训练输出不混入 runtime Evidence：

```text
artifacts/r2x-training/<run_id>/
  config/
  split_manifest.json
  dataset_hash.json
  checkpoints/
  metrics.jsonl
  resource_report.json
```

若仓库已有统一 artifacts 路径，R2-H 启动时按现有规范选择，不在根目录散落 checkpoint。

---

## 24. 文件级实施计划

以下是建议路径，不代表已创建：

| 路径 | 任务 |
|---|---|
| `config/vla/k2_v2_spatial.toml` | V2 结构、Frenet、Guard、availability 参数 |
| `model/k2_spatial_types.py` | V2 bundle/spec/diagnostics |
| `model/frenet_codec.py` | native path ↔ Frenet residual |
| `model/k2_spatial_head.py` | mode query + residual/speed/availability head |
| `model/k2_spatial_builder.py` | head 输出 → candidate-specific T10/spec |
| `model/k2_spatial_guard.py` | 独立 Guard V2 |
| `model/simlingo_runtime.py` | 一次 forward 暴露 V2 head 所需特征/输出 |
| `runtime/k2_execution.py` | V1/V2 selector 与 per-path binding |
| `adapter/policy_adapter.py` | availability 与 V2 lineage 透传 |
| `data_pipeline/vla/spatial_k2_schema.py` | 双标签样本 |
| `data_pipeline/vla/spatial_k2_teacher.py` | teacher 生成/验证 |
| `training/` 或现有训练目录 | heads-only 训练入口 |
| `evaluation/r2x_diagnostic.py` | R2-F 四层诊断 |
| `evaluation/paired_contract.py` | K2AnchorArtifact V2 |
| `evaluation/paired_live.py` | V2 artifact 冷重建/force |
| `tests/g3/` | V2 builder/Guard/execution regression |
| `tests/g4/` | R2-X comparability/oracle/orchestrator |

实施前用 `rg` 查重；若已有合适模块则扩展，不平行复制第二套 runner。

---

## 25. 测试矩阵

### 25.1 Frenet/Builder

- 直线、弯道、短 path；
- 左/右 scene-conditioned residual；
- 近场 envelope；
- 单调 `s`；
- path 末端不足；
- available/unavailable；
- deterministic fixed input。

### 25.2 Guard

- 两条合法空间 path PASS；
- constant-offset 首步不连续 REJECT；
- reversed progress REJECT；
- path hash spoof REJECT；
- T10/spec path mismatch REJECT；
- fake diagnostics REJECT；
- fixed-bias/debug lineage REJECT；
- excessive lateral/curvature/self-intersection REJECT；
- unavailable candidate 不可 force；
- V1 bundle 仍走 V1 Guard。

### 25.3 Training/data

- split overlap；
- Regression 泄漏；
- alternative mask；
- 无替代样本不施加 diversity；
- teacher hash/config；
- checkpoint exact resume；
- NaN/OOM failure manifest。

### 25.4 Execution

- force 0/1 绑定各自 path hash；
- PathManager state 隔离；
- SpeedPlanner state 隔离；
- candidate order counterbalance；
- fallback 到 top-1；
- World 未接入时不改变执行语义。

### 25.5 Paired runner

- 一个 anchor forward；
- branch 不 forward；
- artifact V2 exact reuse；
- attempt 隔离/idempotency；
- timeout/cleanup failure；
- failed/incomparable 保留；
- repeat plan outcome 前冻结。

---

## 26. Failure taxonomy 与恢复

| 类别 | 示例 | 恢复 |
|---|---|---|
| `DATA` | 无双标签、split overlap、许可不明 | 停在 R2-H0/H1 |
| `MODEL_COLLAPSE` | defensive 恒等于 nominal | 检查标签/mask/query，不先加噪 |
| `LOW_QUALITY_MODE` | candidate 1 off-road/尖曲率 | teacher/constraint 修复 |
| `CONTRACT` | path/T10/hash 不绑定 | Guard/Builder 修复 |
| `EXECUTOR_COLLAPSE` | PathManager 合并路径 | 修执行绑定/状态隔离 |
| `SCENE` | 2.5 s 内无可观测差异 | registry 重设计并升版本 |
| `ORACLE` | 连续差异被 deadband 吞掉 | 只允许新 oracle 版本 |
| `CARLA` | tick/RPC/server timeout | 保留失败，ensure 后继续 |
| `RESOURCE` | OOM/超预算 | 缩 batch/head，不换大模型 |

同一问题最多一次初始实现和两次有实质差异的修复。连续两次无进展即停止。

---

## 27. 资源预算

固定 RTX 4080 16GB / CARLA 共卡边界：

- training 与 CARLA 不并发；
- heads-only：micro-batch 1 起步，gradient accumulation；
- 先 20–100 step smoke；
- 不以第二张 GPU/远程服务器为依赖；
- online 仍只允许一次 SimLingo 主干 forward；
- 新 head online 增量显存和 latency 必须单独测量；
- 不因 R2-X 把 K2 降为 K1；
- OOM 必须写失败 manifest，不静默换模型/精度继承旧结论。

资源报告至少：

```text
train peak VRAM / RAM
step latency / throughput
checkpoint size
online forward P50/P95/P99
whole-process peak VRAM
CARLA + VLA shared-GPU margin
```

---

## 28. 文献与方法依据

这些来源只支持设计方向，不替代本项目证据：

- [SimLingo](https://arxiv.org/abs/2503.09594)：当前主干与 Action Dreaming
  的语言-动作对齐依据；
- [MultiPath](https://arxiv.org/abs/1910.05449)：一次 forward 的 mode anchor +
  residual 多模态表示；
- [MultiPath++](https://arxiv.org/abs/2111.14973)：learned latent anchor；
- [MTR](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)：
  intention query + local refinement；
- [CoverNet](https://openaccess.thecvf.com/content_CVPR_2020/html/Phan-Minh_CoverNet_Multimodal_Behavior_Prediction_Using_Trajectory_Sets_CVPR_2020_paper.html)：
  动力学可行的候选集合；
- [DiffusionDrive](https://arxiv.org/abs/2411.15139)：多模态规划备选；因当前
  K=2、单卡、现有接口与实时成本，不作为 R2-X 首版。

预测论文的方法不能自动证明 ego planning 有效；最终仍以本项目 Guard、force
execution 与 paired closed-loop Evidence 为准。

---

## 29. 开工前清单

用户授权 R2-F 时：

1. 把 `START_TASK.md` 唯一任务改为 R2-F；
2. 确认旧 R2 Evidence 只读；
3. 检查分支、dirty worktree 和重叠改动；
4. 只实现离线诊断；
5. 写新 Evidence 目录；
6. 运行最小离线测试、compileall、diff check；
7. 更新 PROGRESS 后停止。

R2-F 关闭后，用户再逐阶段授权 R2-G。不得用“开始 R2-X”一次性跳过数据 gate、
离线验收或 force smoke。

建议下一条明确口令：

```text
开始 R2-F：只做纵向 R2 的 proposal/executor/control/outcome 四层离线差异诊断，
写入新 Evidence，不改旧 R2，不开始 Guard V2、训练或 CARLA。
```
