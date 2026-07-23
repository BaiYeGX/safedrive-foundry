# SafeDrive Foundry 核心项目合同

本文是项目范围、系统边界、研究完成口径和证据规则的唯一权威说明。执行顺序见
根目录 `ROADMAP.md`，当前唯一任务见 `START_TASK.md`，动态事实见 `PROGRESS.md`。

## 1. 项目目标

SafeDrive Foundry 是运行在单台工作站上的 CARLA–ROS 2 纯软件在环（SIL）平台。
当前研究问题只有一个：

> 轻量驾驶 VLA 能否稳定产生有真实选择空间的 K2 定时轨迹，动作条件 World Model
> 能否比 VLA 原始 top-1 更好地选择闭环动作？

核心交付：

1. Observable 输入上的真实 VLA K2；
2. 固定场景、paired replay 和 oracle best-of-K；
3. 可开关的 World-V0 软排序；
4. `VLA-Top1 / VLA+World / Oracle-Best-of-K` 同协议证据；
5. World 故障时确定性回到 VLA 原始 top-1；
6. 可复现的配置、资源、失败和限制。

项目不宣称实车、公共道路或量产安全。Classic、完整 Safety、自动场景搜索、Agent
和后训练均不能替代核心 VLA+World 证据。

## 2. 系统边界

### 2.1 核心研究链

```text
CARLA camera + ego + navigation
  → VLA-V1 K2 timed trajectories
  → Contract Guard
  → World-V0 on/off
  → selected trajectory
  → MPC/PID
  → CARLA vehicle
```

- VLA 提出轨迹，不直接发未经约束的 steering/throttle/brake。
- Contract Guard 检查 schema、身份、freshness、有限值、时间单调、候选差异和
  运动学一致性；它不冒充完整 Safety。
- World 只重新排序同一 K2，不修改轨迹、不直接控车、不拥有 CARLA tick。
- World off、timeout、OOM、NaN 或 invalid 时执行 VLA 原始 top-1。
- MPC/PID 只跟踪已选择轨迹，不重做行为决策。
- 核心 A/B 使用相同 VLA、K2、场景、seed、initial-state 和 MPC。

### 2.2 可选工程链

```text
VLA K2 → hard pre-validation → World rank → final Safety → MPC/PID
```

G2 Safety 启用后，学习模块不能覆盖硬约束、slack、MRM 或 Emergency。Classic 可作
基线、修正标签、Shadow 或 Hybrid 候选，但不得偷偷进入核心 VLA/World A/B。

无完整 Safety 的核心链仅用于 CARLA SIL 研究，不能外推为道路安全系统。

## 3. 冻结接口

### 3.1 Observation

正式策略输入只允许 Observable：

- 当前前视图像；
- ego 当前状态和至多 4 个低维历史时刻；
- route/navigation、可行驶区域和交通灯可观测状态；
- actor 可观测状态、协方差、时间戳和丢失标志；
- run/frame/scenario/model/config/schema identity。

CARLA actor 真实未来、真实 TTC、隐藏意图、测试集专家轨迹只能用于训练标签、oracle
或评价，不能进入正式 VLA/World runtime 输入。

### 3.2 PolicyCandidateSet V1

```text
K=2
T=10
dt=0.25 s
horizon=2.5 s
fields=x,y,yaw,v,a,kappa,time,candidate_id,probability,source,identity
```

要求：

- candidate 0/1 来自同一 Observation；
- 两条轨迹可执行、可区分、运动学一致；
- 能分别强制执行并回溯；
- 静默复制或只改 speed 字段但不重新参数化位置不算 K2；
- 记录 collapse、margin、生成时间和模型/配置 hash。

### 3.3 WorldRolloutBatch V0

```text
K=2, T=10, horizon=2.5 s
N actors <= 8
M modes = 1
model size target = 4M–8M parameters
outputs = actor future + collision + TTC + off-road + pairwise score
```

每条输出必须绑定 candidate_id 和 frame identity。缺失、超时、未校准和非法输入必须
显式返回错误，禁止用缺失输出表示低风险。

## 4. 两级完成定义

### 4.1 `VLA_WORLD_RESEARCH_COMPLETE`

以下全部满足即可关闭核心研究主线：

1. 真实 K2 合同通过；
2. 6–8 个固定 pilot 场景、至少 2 个 seed 可 paired replay；
3. top-1 与 oracle best-of-K 可比较；
4. World-V0 真实接入并有 on/off；
5. VLA-Top1、VLA+World、oracle 使用同协议；
6. World 异常确定性回到 top-1；
7. 原始 run、hash、资源和限制可追溯。

World 正收益、无稳定净收益或负收益都允许。负结果不能用来删除模块或伪造结论。

### 4.2 `FULL_PROJECT_COMPLETE_WITH_SAFETY`

只有核心完成后，再满足下列条件才使用该标签：

- `VLA+Safety` 与 `VLA+World+Safety` 可重复运行；
- World 异常时退化为 `VLA+Safety`；
- Safety 批准/修复/MRM/Emergency 与 executed trajectory 可审计；
- Safety 工程回归和 Evidence Bundle 完成。

## 5. 核心主张

| ID | 问题 | 必须对照 |
|---|---|---|
| C1 | VLA 是否产生真实、非坍塌 K2 | K1、K2、强制 candidate 0/1、collapse |
| C2 | World 是否改善排序和闭环 | top-1、oracle、CV/CTRV、Reward、no-action、World off |
| C4 | paired replay 是否能量化选择空间 | 同 seed/initial-state、不可比分支、同步失败 |
| C5 | 条件式 G6 是否改善候选且不遗忘 | base/posttrain、K2、oracle gap、World on/off |

C3 Safety、C0 Classic 和 C6 Agent 是独立背景或扩展主张，不阻塞核心完成。

## 6. 核心发布配置

1. `VLA-Top1`；
2. `VLA+World`；
3. `Oracle-Best-of-K`，只作离线评价上限；
4. `PostTrained VLA+World`，仅 G6 被证据触发并完成时加入。

Classic、Hybrid、VLA+Safety 和 VLA+World+Safety 单列，不混入核心因果比较。

## 7. 统一证据规则

每个结论至少绑定：

```text
run_id / scenario_id / seed / initial_state_hash
code/config/model/data hash
candidate_id / selected_id / executed_id
hardware/profile/precision
latency P50/P95/P99 / deadline miss / VRAM/RAM
collision/TTC/off-road/progress/comfort
failure classification / limitations
```

证据状态：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

只有 `VERIFIED` 可作为无保留量化表述。未运行测试不得写通过；CARLA 结果不得写成
真实道路安全证明。

## 8. 可选模块的启用条件

- G6：只有 G5 证明两条候选经常双双失败或 VLA 质量限制 oracle/World 上限时启用。
- G3-05 Safety：核心效果完成后按工程需要补做。
- G4B Search：只有固定 paired set 不足以回答问题时启用。
- G7 Agent：只在确定性工作流已成立且确有研发效率问题时启用。

所有 optional 未执行时保持 `PENDING`，Evidence 标记 `OPTIONAL_NOT_RUN` 或
`DEFERRED`。
