# SafeDrive Foundry 唯一任务启动入口

> 固定启动口令：`读取 START_TASK.md，启动 GX-XX。`

本文件是 G0～G8 全部 48 项任务的唯一启动路由。它负责定位任务、规定读取顺序、检查依赖和资源准入；任务目标、允许修改范围、验收命令与停止点仍以唯一匹配的 `tasks/GX/GX-XX_*.md` 为准。

## 1. 一行启动协议

收到 `读取 START_TASK.md，启动 GX-XX。` 后，严格执行：

1. 从指令中提取且只提取一个严格格式任务 ID；未给 ID 时只读取 `PROGRESS.md` 并推荐，不自行启动。
2. 读取根目录 `AGENTS.md` 与 `PROGRESS.md`；检查当前分支和 `git status --short`。
3. 按第 4 节任务总表定位唯一任务文件并完整读取；0 个或多个匹配立即停止。
4. 按任务文件中的 `启动读取清单` 顺序读取指定文档章节、直接依赖的最终断点和明确产物。清单继续指向其他必读项时，只沿当前任务链继续读取，不横向通读。
5. 读取 `ROADMAP.md` 当前阶段表，核对任务编号、直接依赖和阶段关闭点；以任务文件的 `依赖` 字段执行完成状态检查。
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
| G3-02 | `tasks/G3/G3-02_NONLANGUAGE_MULTICANDIDATE_BASELINES.md` | 非语言基线、公平预算、CLAIMS C2 |
| G3-03 | `tasks/G3/G3-03_LIGHTWEIGHT_FAST_SLOW_VLA.md` | Fast/Slow VLA、并行轨迹头、单卡准入 |
| G3-04 | `tasks/G3/G3-04_TRAINING_CALIBRATION_OOD.md` | LoRA/QLoRA、grounding、校准/OOD |
| G3-05 | `tasks/G3/G3-05_VLA_SAFETY_CLOSED_LOOP_ACCEPTANCE.md` | VLA→Safety→MPC 闭环与阶段 Evidence |

### 4.5 G4（5 项，场景搜索与反事实）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G4-01 | `tasks/G4/G4-01_SCENARIO_REGISTRY_SUITES.md` | Registry、场景层级、覆盖/可解性 |
| G4-02 | `tasks/G4/G4-02_RANDOM_LHS_SEARCH_BASELINES.md` | Random/LHS、公平预算、可恢复执行器 |
| G4-03 | `tasks/G4/G4-03_COVERAGE_GUIDED_MAP_ELITES.md` | MAP-Elites、有效性门禁、archive |
| G4-04 | `tasks/G4/G4-04_COUNTERFACTUAL_REPLAY_MINIMIZATION.md` | 可比重放、反事实、最小化与聚类 |
| G4-05 | `tasks/G4/G4-05_SEARCH_COUNTERFACTUAL_ACCEPTANCE.md` | CLAIMS C4、复现率与阶段 Evidence |

### 4.6 G5（5 项，动作条件世界模型）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G5-01 | `tasks/G5/G5-01_ACTION_BRANCH_DATA_BASELINES.md` | ActionBranchDataset、简单动力学/Reward 基线 |
| G5-02 | `tasks/G5/G5-02_OBJECT_CENTRIC_WORLD_MODEL.md` | object/vector latent World、单卡训练 |
| G5-03 | `tasks/G5/G5-03_RISK_CAUSAL_CALIBRATION.md` | 多模态风险、action intervention、校准 |
| G5-04 | `tasks/G5/G5-04_VLA_WORLD_RUNTIME_ACTIVE_VERIFY.md` | VLA 候选排序、异步 Active CARLA、降级 |
| G5-05 | `tasks/G5/G5-05_VLA_WORLD_SAFETY_ACCEPTANCE.md` | CLAIMS C1～C3、World 净收益与阶段 Evidence |

### 4.7 G6（5 项，失败驱动安全后训练）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G6-01 | `tasks/G6/G6-01_EVENT_HARDCASE_DATA_GATES.md` | EventWindow、Hard Case、谱系/数据门禁 |
| G6-02 | `tasks/G6/G6-02_SHADOW_DAGGER_TAKEOVER.md` | Shadow DAgger、Takeover、查询预算 |
| G6-03 | `tasks/G6/G6-03_PREFERENCE_RISK_POSTTRAIN.md` | CARLA 验证偏好、Risk Anticipation、LoRA |
| G6-04 | `tasks/G6/G6-04_REGRESSION_ANTI_FORGETTING.md` | 正常/OOD/历史失败回归与抗遗忘 |
| G6-05 | `tasks/G6/G6-05_FLYWHEEL_ACCEPTANCE.md` | CLAIMS C5、飞轮 A/B 与阶段 Evidence |

### 4.8 G7（3 项，受控 Agentic Research Loop）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G7-01 | `tasks/G7/G7-01_AGENT_TOOLS_GOVERNANCE.md` | typed tools、白名单、权限/预算/审计 |
| G7-02 | `tasks/G7/G7-02_SCENARIO_FAILURE_RESEARCH_ASSISTANT.md` | 场景/失败研究助手与证据 grounding |
| G7-03 | `tasks/G7/G7-03_DETERMINISTIC_DATA_RELEASE_ACCEPTANCE.md` | 确定性裁决、LLM-off 对照、红队验收 |

### 4.9 G8（5 项，系统准出与成果证据）

| ID | 唯一任务文件 | 任务内主阅读锚点 |
|---|---|---|
| G8-01 | `tasks/G8/G8-01_PROTOCOL_ASSET_FREEZE.md` | CLAIMS、四发布配置、协议/资产冻结 |
| G8-02 | `tasks/G8/G8-02_UNIFIED_SYSTEM_REGRESSION.md` | ID/长尾/OOD/故障/资源统一回归 |
| G8-03 | `tasks/G8/G8-03_ABLATION_STATISTICS_PARETO.md` | 消融、统计、异质性与 Pareto |
| G8-04 | `tasks/G8/G8-04_EVIDENCE_REPRODUCTION.md` | Evidence Bundle、演示、冷启动复现 |
| G8-05 | `tasks/G8/G8-05_FINAL_RELEASE_ACCEPTANCE.md` | 最终审计、限制登记与准出 |

## 5. VLA 与 World Model 的跨阶段实现路径

### 5.1 VLA 路径

```text
G2-05 安全边界冻结
  → G3-01 ObservationBundle / 数据 / split
  → G3-02 非语言与多候选基线
  → G3-03 轻量 Fast/Slow VLA + K 条轨迹
  → G3-04 LoRA/QLoRA + grounding + calibration/OOD
  → G3-05 VLA → Validator/Safety → MPC 闭环
  → G5-04/05 接 World 排序但不交出控制权
  → G6-02/03 Shadow 数据与安全后训练
  → G6-04/05 抗遗忘与飞轮验收
  → G8 四配置冻结、统计和复现
```

VLA 输出 `PolicyCandidateSet`（轨迹、概率、有效期、行为/关键 Actor/风险时域、不确定性），不直接输出未经约束的车辆控制。Slow 路径异步触发，超时不能阻塞 20ms 控制环。完整模块/文件落点和接口字段见工业架构第 12 节。

### 5.2 World Model 路径

```text
G3-01 复用身份、Observation 与 split
  → G4-04 可比反事实起点
  → G5-01 多动作分支数据 + CV/CTRV/IDM/Reward 基线
  → G5-02 轻量 object/vector latent 动作条件动力学
  → G5-03 多模态风险/奖励/不确定性 + action intervention
  → G5-04 只排序已通过预筛的 VLA/Classic 候选
  → G5-05 相对简单基线证明闭环净收益
  → G6-01 挖掘误排/不确定困难样本
  → G8 作为可关闭配置完成消融、资源与复现
```

World Model 预测 `WorldRolloutBatch`，永远不是安全真值；高风险/不确定候选只能进入异步 Active CARLA 研发验证队列，不能进行当前帧在线分叉，也不能成为 tick owner。完整模块/文件落点见工业架构第 13～14 节。

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

## 6. 完成与停止

每次只完成一个任务。结束前必须检查真实 `git diff`/`git diff --stat`、运行任务规定的最小验证、记录实际命令和结果、更新任务断点与 `PROGRESS.md`。只有全部验收通过才能写 `COMPLETED`；未运行测试不能写通过，预期指标不能写成实测结果。

达到验收、停止点、未知工作区改动、文档实质冲突、同一问题两次修复无进展、外部权限/GUI/登录/网络阻塞或需要扩大架构范围时，按 `AGENTS.md` 记录状态并立即停止。

## 7. 固定口令

```text
读取 START_TASK.md，启动 G2-01。
```

```text
读取 START_TASK.md，恢复 G2-01。
```

```text
读取 START_TASK.md，继续解决 G2-01；不要开始下一任务。
```
