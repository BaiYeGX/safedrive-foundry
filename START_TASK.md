# SafeDrive Foundry 唯一任务启动入口

> 固定启动口令：`读取 START_TASK.md，启动 GX-XX。`

本文件是 G0～G8 全部 48 项任务的唯一启动路由。G3～G8 的最小范围、World 条件门禁和 Optional 状态以本文第5节为上位执行约束；旧任务文件中的更大 K/T/N/M、自动搜索、复杂后训练和 Agent 要求不得扩大第一版范围。

## 1. 一行启动协议

收到 `读取 START_TASK.md，启动 GX-XX。` 后，严格执行：

1. 从指令中提取且只提取一个严格格式任务 ID；未给 ID 时只读取 `PROGRESS.md` 并推荐，不自行启动。
2. 读取根目录 `AGENTS.md` 与 `PROGRESS.md`；检查当前分支和 `git status --short`。
3. 按第 4 节任务总表定位唯一任务文件并完整读取；0 个或多个匹配立即停止。
4. 按任务文件中的 `启动读取清单` 顺序读取指定文档章节、直接依赖的最终断点和明确产物。清单继续指向其他必读项时，只沿当前任务链继续读取，不横向通读。
5. 读取 `ROADMAP.md` 当前阶段表，核对任务编号、直接依赖和阶段关闭点；G3～G8依赖先应用本文第5节的World门禁和Optional覆盖，再检查任务文件其余依赖。
6. G2～G8 在实施前执行第 3 节的单机资源准入；依赖真实 CARLA 时再执行第 2 节预检。
7. 只实施这一项任务，按任务文件运行验证、记录断点并更新 `PROGRESS.md`；达到验收或停止条件后立即停止，不自动开始下一项。

恢复任务使用：`读取 START_TASK.md，恢复 GX-XX。` 此时优先核验任务末尾断点、文件版本、外部状态和未完成验证，禁止重做已经确认的步骤。

## 2. 真实 CARLA 预检

任务明确需要真实 CARLA 时，先运行：

```text
sdf sim preflight
```

处理规则：

1. `READY`：继续。
2. `RETRYABLE_FAILURE`：只执行一次 `sdf sim ensure`，成功后只重试一次 preflight。
3. `needs_user_action=true` 且 `error_code=NEEDS_USER_ACTION`：标记 `BLOCKED_EXTERNAL`，记录最小用户操作并停止。
4. 版本不匹配、tick owner 或依赖冲突：标记 `BLOCKED`/`DECISION_REQUIRED`，不得自行换版本。
5. 不需要真实 CARLA 的文档、schema、单元测试和离线训练任务不强制预检。

任何后续任务不得硬编码 CARLA IP、复制 G0 gateway、创建第二个 `carla.Client`/tick master 或直接调用 `world.tick()`；统一走已验证的 `sdf sim` 与 Runtime 接口。

## 3. G2～G8 单机资源准入

G2～G8 必须读取任务清单指向的 `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 章节，并在实现前声明当前 workload profile：

- `vla_train`：VLA 单卡训练；
- `world_train`：World Model 单卡训练；
- `online_eval`：CARLA + VLA + World + Safety 在线评测；
- `data_collect`：CARLA 采集与轻量在线推理；
- `regression`：串行/分片回归；
- `agent_offline`：Agent 只读离线研究。

训练 profile 与在线 CARLA profile 默认互斥；Windows CARLA 和 WSL 模型共享同一张 RTX 4080 16GB，不能按两张 GPU 计算。未通过显存、CPU、磁盘、温度和恢复 smoke，不得扩大数据、时域、候选数或模型规模。任务中的性能数字是准入目标，只有实测 `VERIFIED` 才能成为项目结论。

## 4. 48 项任务总路由

### 4.1 G0（6 项，已冻结）

| ID | 唯一任务文件 |
|---|---|
| G0-01 | `tasks/G0/G0-01_ENVIRONMENT_AND_VERSIONS.md` |
| G0-02 | `tasks/G0/G0-02_WSL_GPU_ROS2.md` |
| G0-03 | `tasks/G0/G0-03_CARLA_SERVER.md` |
| G0-04 | `tasks/G0/G0-04_SKELETON_AND_CONNECTIVITY.md` |
| G0-05 | `tasks/G0/G0-05_DETERMINISM_AND_DOCTOR.md` |
| G0-06 | `tasks/G0/G0-06_ACCEPTANCE.md` |

G0 已验收冻结。除用户明确要求修复某个 G0 缺陷外，只读任务文件、`PROGRESS.md`、任务明确引用的环境文档和历史 Evidence，不得修改 G0 任务、配置、代码或证据。

### 4.2 G1（9 项，经典运行时与专家）

| ID | 唯一任务文件 |
|---|---|
| G1-01 | `tasks/G1/G1-01_SYSTEM_CONTRACTS_AND_RUNTIME_PROFILES.md` |
| G1-02 | `tasks/G1/G1-02_SCENARIO_RUNTIME_AND_ACTOR_LIFECYCLE.md` |
| G1-03 | `tasks/G1/G1-03_MAP_LANE_GRAPH_ROUTE_BEHAVIOR.md` |
| G1-04 | `tasks/G1/G1-04_FRENET_LATTICE_AND_ST_SPEED.md` |
| G1-05 | `tasks/G1/G1-05_HYBRID_ASTAR_REEDS_SHEPP.md` |
| G1-06 | `tasks/G1/G1-06_MPC_PID_MULTIRATE_CONTROL.md` |
| G1-07 | `tasks/G1/G1-07_RISK_ADAPTIVE_PLANNING_OPTIMIZATION.md` |
| G1-08 | `tasks/G1/G1-08_ROBUST_ADAPTIVE_MPC_OPTIMIZATION.md` |
| G1-09 | `tasks/G1/G1-09_CLASSIC_EXPERT_ACCEPTANCE.md` |

G1 固定文档链：任务文件 → `docs/project/EXECUTION_ARCHITECTURE.md` 对应接口/Profile/Registry 章节 → G1-04～G1-09 需要的 `docs/project/decisions/CLASSIC_ALGORITHM_OPTIMIZATION.md` 章节 → 直接依赖最终断点。G1 已完成项默认只读，只有明确修复/重验才重新实施。

### 4.3 G2（5 项，Independent Safety Kernel）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G2-01 | `tasks/G2/G2-01_SAFETY_CONTRACTS_VALIDATOR.md` | 执行架构、工业架构 1～4/7、单机预算 |
| G2-02 | `tasks/G2/G2-02_LONGITUDINAL_QP_REPAIR.md` | Safety/RATO 契约、经典算法决策、G2-01 产物 |
| G2-03 | `tasks/G2/G2-03_RESTRICTED_RATO_SCP.md` | 可行走廊、SCP 预算、G2-01/02 产物 |
| G2-04 | `tasks/G2/G2-04_ARBITRATION_SHADOW_FALLBACK.md` | 仲裁/降级链、Runtime 接口、G2-01～03 产物 |
| G2-05 | `tasks/G2/G2-05_FAULT_AND_SAFETY_ACCEPTANCE.md` | CLAIMS C3、故障矩阵、G2 全阶段 Evidence |

### 4.4 G3（5 项，轻量驾驶 VLA）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G3-01 | `tasks/G3/G3-01_DATA_SCENARIO_SPLITS.md` | Observation/Data contract、split、VLA 路径 |
| G3-02 | `tasks/G3/G3-02_NONLANGUAGE_MULTICANDIDATE_BASELINES.md` | V0 K1 / V1 K2 公平基线 |
| G3-03 | `tasks/G3/G3-03_VLA_V0_UPSTREAM_CANONICALIZER.md` | VLA-V0 + F0 checkpoint 门禁、canonicalizer、5Hz 首门槛 |
| G3-04 | `tasks/G3/G3-04_VLA_V1_OPTIONAL_ROUTER.md` | VLA-V1；V2 FAST/REASON 递进 optional |
| G3-05 | `tasks/G3/G3-05_VLA_SAFETY_CLOSED_LOOP_ACCEPTANCE.md` | VLA→Safety→MPC 闭环与阶段 Evidence |

当前 pure VLA+MPC 发布检查点、启动命令、已知限制和 G4 路线统一见
`docs/architecture/G3_VLA_MPC_RELEASE_GUIDE.md`。其中 `MEASURED_WITH_LIMITS` 不替代
G3-05 的 Safety 正式验收。用户可明确授权 G4-01/G4-02 以 `PRE_G3_CLOSE` 只准备
Registry/replay 基础设施，但不得据此把 G3 或 G4 标记为关闭。

### 4.5 G4（5 项，场景搜索与反事实）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G4-01 | `tasks/G4/G4-01_SCENARIO_REGISTRY_SUITES.md` | Registry、场景层级、覆盖/可解性 |
| G4-02 | `tasks/G4/G4-02_FIXED_SCENARIO_REPLAY_EXECUTOR.md` | G4A 固定场景、seed/replay、可恢复执行 |
| G4-03 | `tasks/G4/G4-03_COVERAGE_GUIDED_MAP_ELITES.md` | G4B Optional：MAP-Elites/自动搜索 |
| G4-04 | `tasks/G4/G4-04_COMPARABLE_K2_ORACLE_BEST_OF_K.md` | G4A K2 可比分支/oracle；最小化聚类 optional |
| G4-05 | `tasks/G4/G4-05_SEARCH_COUNTERFACTUAL_ACCEPTANCE.md` | G4A 与 World 入口门禁 Evidence |

### 4.6 G5（5 项，动作条件世界模型）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G5-01 | `tasks/G5/G5-01_ACTION_BRANCH_DATA_BASELINES.md` | ActionBranchDataset、简单动力学/Reward 基线 |
| G5-02 | `tasks/G5/G5-02_OBJECT_CENTRIC_WORLD_MODEL.md` | World-V0 K2/T10/N8/M1、4M～8M |
| G5-03 | `tasks/G5/G5-03_RISK_CAUSAL_CALIBRATION.md` | collision/TTC/off-road、action/no-action |
| G5-04 | `tasks/G5/G5-04_VLA_WORLD_RUNTIME_ACTIVE_VERIFY.md` | VLA 候选排序、异步 Active CARLA、降级 |
| G5-05 | `tasks/G5/G5-05_VLA_WORLD_SAFETY_ACCEPTANCE.md` | CLAIMS C2、World 开/关 A/B 与阶段 Evidence |

### 4.7 G6（5 项，失败驱动安全后训练）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G6-01 | `tasks/G6/G6-01_EVENT_HARDCASE_DATA_GATES.md` | EventWindow、Hard Case、谱系/数据门禁 |
| G6-02 | `tasks/G6/G6-02_SHADOW_EXPERT_CORRECTION_COLLECTION.md` | Shadow 专家修正采集、查询预算 |
| G6-03 | `tasks/G6/G6-03_SUPERVISED_ADAPTER_LORA_POSTTRAIN.md` | 单轮监督 adapter/LoRA（禁多 preference/RL） |
| G6-04 | `tasks/G6/G6-04_REGRESSION_ANTI_FORGETTING.md` | 正常/OOD/历史失败回归与抗遗忘 |
| G6-05 | `tasks/G6/G6-05_FLYWHEEL_ACCEPTANCE.md` | CLAIMS C5、飞轮 A/B 与阶段 Evidence |

### 4.8 G7（3 项，Optional / After Release）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G7-01 | `tasks/G7/G7-01_AGENT_TOOLS_GOVERNANCE.md` | typed tools、白名单、权限/预算/审计 |
| G7-02 | `tasks/G7/G7-02_SCENARIO_FAILURE_RESEARCH_ASSISTANT.md` | 场景/失败研究助手与证据 grounding |
| G7-03 | `tasks/G7/G7-03_DETERMINISTIC_DATA_RELEASE_ACCEPTANCE.md` | 确定性裁决、LLM-off 对照、红队验收 |

G7 不属于正式主线完成条件，不是 G8 依赖。普通脚本、CLI、Registry、自动报告和 Evidence Bundle 由 G3～G6/G8 的确定性实现承担，不得为了这些基础能力强制启动 Agent。

### 4.9 G8（5 项，系统准出与成果证据）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G8-01 | `tasks/G8/G8-01_PROTOCOL_ASSET_FREEZE.md` | CLAIMS、四发布配置、协议/资产冻结 |
| G8-02 | `tasks/G8/G8-02_UNIFIED_SYSTEM_REGRESSION.md` | ID/长尾/OOD/故障/资源统一回归 |
| G8-03 | `tasks/G8/G8-03_ABLATION_STATISTICS_PARETO.md` | 消融、统计、异质性与 Pareto |
| G8-04 | `tasks/G8/G8-04_EVIDENCE_REPRODUCTION.md` | Evidence Bundle、演示、冷启动复现 |
| G8-05 | `tasks/G8/G8-05_FINAL_RELEASE_ACCEPTANCE.md` | 最终审计、限制登记与准出 |

## 5. VLA 与 World Model 的跨阶段实现路径

本节同时是最终范围覆盖层。作品级成功口径见 `docs/project/PROJECT_SUCCESS_PROFILE.md`（**稳定可跑 + VLA 与 World 都真实接入；效果可负；不上实车**）。本机路径与 **Python venv（`/home/sdf/.venvs/sdf`）** 见 `docs/project/LOCAL_ASSETS.md`。G3+ 涉及 torch 的命令必须在该 venv 内执行，不得使用系统 `python3`。

启动 G3 或 G6 时读取 `SDF_VLA_1B_DESIGN.md`；启动 G4-04/05 或任何 G5 时读取 `SDF_WORLD_MODEL_DESIGN.md`；G3-05、G5-04/05、G6-05 和 G8 还读取 `SDF_VLA_WORLD_SYSTEM_ARCHITECTURE.md`。当任务旧文本要求更大的第一版范围时，以本节与 `PROJECT_SUCCESS_PROFILE.md` 为准，且不得修改 G2。

### 5.1 VLA 路径

```text
G2-05 安全边界冻结（只读）
  → G3-01/02 数据契约与非语言基线
  → G3-03 F0 + VLA-V0: K1/T10/2.5s + 无 Classic 闭环
  → G3-04 VLA-V1: K2 + 低维 history；V2 Router optional
  → G3-05 VLA+Safety 与 Hybrid 验收
  → G4A 固定场景 + oracle best-of-K（科学标注）
  → G5 World-V0 必做接入（作品完整性）
  → G6 一轮困难样本监督适配（推荐；增益可负）
  → G8 冻结、统计、演示 Evidence
```

VLA 输出完整 `PolicyCandidateSet`，不直接输出未经约束的车辆控制。V0/V1 P95 首门槛为200ms，超时不能阻塞20ms控制环。VLA_SAFETY 模式禁止 Classic 当前帧候选。

基础模型选型冻结为 **SimLingo / InternVL2-1B** 路线，不再在 G3 前重新选型；不得直接切换 3B/7B。`F0` 在 **G3-03** 落地；仅当 SimLingo checkpoint 在真实 F0 中无法解决时，才启用干净 InternVL2-1B 并保留相同 Adapter/轨迹头/Safety 接口（`SDF-VLA-1B-IVL`，重跑 F0～F3）。详见 `docs/project/SDF_VLA_1B_DESIGN.md`。

VLA 固定首版为 V0 `K1/T10` 和 V1 `K2/T10`、2.5s/0.25s；不把上游输出外推成3秒预测。

### 5.2 World Model 路径（本项目必做接入）

```text
G4A 比较 VLA top-1 与 oracle best-of-K
  → 输出科学标签：ENTER_WORLD / WEAK_SELECTION_SPACE / NO_SELECTION_SPACE
  → 无论标签强弱，均进入 G5 实现（作品要求 VLA+World 都上）
       → G5-01 K2 branches + CV/CTRV/Reward 基线
       → G5-02/03 World-V0 K2/T10/N8/M1
       → G5-04/05 soft ranking + World 开关闭环 A/B
```

World Model 预测 `WorldRolloutBatch`，永远不是安全真值；不能成为 tick owner，不能当前帧在线 CARLA 分叉控车。首版 4M～8M World-V0。

| 科学标签 | 实现 | C2 表述 |
|---|---|---|
| `ENTER_WORLD` | 必做 | 可冲正收益 |
| `WEAK_SELECTION_SPACE` | 必做 | 谨慎/可能负 |
| `NO_SELECTION_SPACE` | **仍必做接入** | 负结论或“无稳定净收益” |

**禁止**用 `SKIPPED_BY_GATE` 跳过本项目 G5 实现。无净收益时仍保留 World 模块与 on/off 对照；可默认演示 World on，并保留一键 off。G4B/G7 仍为 Optional。

### 5.3 工业运行闭环

```text
ObservationBundle
  → Classic / VLA 产生候选
  → G2 Validator 预筛
  → World Model 软风险排序（可关闭/可超时）
  → G2 Validator + Safety Kernel 最终裁决/修复/回退
  → MPC/PID
  → Vehicle
```

学习模块只能改变候选和软风险代价，不能覆盖硬约束、slack 上限、回退权限或紧急动作。VLA、World 或 Agent 全失效时，Classic + Safety + MPC/PID 必须仍能闭环。

两种运行模式固定为 `VLA_SAFETY`（无 Classic 当前帧候选，World 可开）和 `HYBRID`（Classic+VLA，World 可开）。

## 6. 完成与停止

每次只完成一个任务。结束前必须检查真实 `git diff`/`git diff --stat`、运行任务规定的最小验证、记录实际命令和结果、更新任务断点与 `PROGRESS.md`。

- 必做验收通过：`COMPLETED` 或 `COMPLETED_WITH_LIMITS`  
- G4B/G7 未执行：`OPTIONAL_NOT_RUN`  
- **不得**用 `SKIPPED_BY_GATE` 逃避 G5 World 实现  
- 未运行测试不得写通过；预期指标不得写成实测；负结果必须保留  

达到验收、停止点、未知工作区改动、文档实质冲突、同一问题两次修复无进展、外部权限/GUI/登录/网络阻塞或需要扩大架构范围时，按 `AGENTS.md` 记录状态并立即停止。

## 7. 固定口令

```text
读取 START_TASK.md，启动 G3-01。
```

```text
读取 START_TASK.md，恢复 G3-01。
```

```text
读取 START_TASK.md，继续解决 G3-01；不要开始下一任务。
```

当前发布检查点后的推荐口令：

```text
读取 START_TASK.md，恢复 G3-05；按发布说明补 VLA+Safety 正式 live evidence。
```

```text
读取 START_TASK.md，启动 G4-01；允许 PRE_G3_CLOSE 基础设施，但不宣称 G3 VERIFIED。
```
