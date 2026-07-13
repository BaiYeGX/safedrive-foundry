# SafeDrive Foundry v5.0：驾驶 VLA、世界模型与智能体安全闭环平台

> **正式名称**：基于 CARLA–ROS 2 的驾驶 VLA、反事实世界模型与智能体安全闭环平台
> **English**: **SafeDrive Foundry: A CARLA–ROS 2 Platform for Driving VLA, Counterfactual World Models and Agentic Safety Loops**
> **项目形态**：纯软件在环（SIL）自主驾驶研发平台
> **文档定位**：项目唯一执行主方案；功能、实验、简历表述和演示口径均以本文为准

## 0. 最终定义

SafeDrive Foundry 不以“在 CARLA 中跑通一辆车”为终点，也不把 VLA、世界模型、Agent 和端到端模型堆成相互独立的演示。

项目围绕一个统一问题展开：

> 如何构建一个可闭环运行的轻量驾驶 VLA，利用世界模型对候选动作进行交互式反事实推演，由确定性安全内核约束其输出，并让研发智能体持续发现高风险场景、组织失败证据和驱动安全后训练？

四个热点只承担四项不可替代的职责：

| 热点 | 项目中的唯一核心职责 | 明确不做 |
|---|---|---|
| 端到端 | 从传感器、车辆历史和导航直接生成未来轨迹 | 不直接输出未经约束的油门和方向盘 |
| VLA | 联合完成视觉理解、语言条件驾驶、行为决策、轨迹和风险预测 | 不训练行业级大参数通用模型 |
| 世界模型 | 预测“不同 Ego 动作下环境如何响应”，用于反事实排序和奖励估计 | 不训练高清长视频生成模型 |
| Agent | 编排场景搜索、失败诊断、数据治理和版本准出 | 不进入实时控制链，不替代确定性安全规则 |

经典规划控制不是被淘汰，而是承担专家教师、Shadow 对照、轨迹修复、安全回退和底层执行。

## 1. 故事闭环：三环一核

工业系统不是将 VLA、世界模型和 Agent 串成一条实时神经网络，而是同时运行三个闭环，并由一个独立安全内核约束。

### 1.1 驾驶闭环：Observe–Reason–Act–Guard

```text
CARLA 视觉/状态/导航
        ↓
轻量驾驶 VLA
场景语义 + 行为 + K 条候选轨迹 + 风险 + 证据
        ↓
交互世界模型
预测不同 Ego 动作下 Actor 响应、风险和收益
        ↓
策略仲裁
VLA / World-selected / Classic Candidate
        ↓
Safety Kernel
验证 → RATO 修复 → Classic 回退 → 最小风险停车
        ↓
MPC/PID 执行并记录真实结果
```

VLA 是主策略，但它不拥有最终控制权。语言推理必须与候选轨迹、关键 Actor 和风险时间域对应；世界模型提供“可能后果”，Safety Kernel 决定“是否允许执行”。

### 1.2 学习闭环：Failure–Counterfactual–Post-train

```text
闭环驾驶出现低 TTC / OOD / Shield 接管 / 失败
        ↓
保存 normal–pre-event–intervention–recovery 窗口
        ↓
从相同场景起点重建危险状态
        ↓
分支执行 Raw VLA / World-selected / RATO / Classic / Brake
        ↓
CARLA 给出真实碰撞、进度、规则和舒适性结果
        ↓
形成 SFT / DAgger / Preference / Reward / Risk 数据
        ↓
VLA 与世界模型后训练
        ↓
旧失败 + 正常能力 + 未见 ODD 回归
```

这个闭环的关键不是无限收集数据，而是只回流能改变模型决策边界的高价值样本，并严格隔离 Regression Case。

### 1.3 验证闭环：Hypothesis–Search–Evidence–Release

```text
安全要求 / 覆盖缺口 / 失败簇 / 世界模型不确定区
        ↓
Scenario/Failure Research Assistant 生成可证伪场景假设
        ↓
约束求解和 Schema 校验得到合法参数空间
        ↓
Random / LHS / MAP-Elites 同预算搜索
        ↓
失败复现、聚类和最小反例
        ↓
Research Assistant 组织证据草案，确定性指标给出结论
        ↓
Release Gate 运行历史失败与正常能力回归
        ↓
Evidence Registry 决定版本准出或拒绝
```

Agent 负责提出假设和调用工具，不能自定义事实、修改安全阈值或自行准出版本。

### 1.4 一核：Independent Safety Kernel

安全内核独立于 VLA、世界模型和 Agent，包含轨迹验证、风险门控、RATO-SCP、Classic Shadow Expert、最小风险停车和紧急制动。任何学习模型更新都不能绕过它。

### 1.5 闭环之间的数据契约

三个闭环通过四种版本化对象衔接：

- `PolicyOutput`：策略行为、轨迹、风险、不确定性和证据。
- `WorldRollout`：动作条件未来、奖励、终止和模型置信度。
- `SafetyEvent`：违反约束、接管原因、修复量和降级结果。
- `EvidenceBundle`：场景、seed、代码、模型、数据、指标、视频和结论。

最终必须回答五个可证伪问题：

1. VLA 是否在闭环中生成比纯模仿基线更好的可执行轨迹？
2. 世界模型是否比简单运动学/Reward 基线更准确地排序动作后果？
3. RATO 是否比规则减速和硬回退更好地平衡安全、进度与舒适性？
4. Agent-guided 测试是否在相同 rollout 预算下发现更多独立、有效、可复现失败？
5. 后训练是否降低重复失败，同时通过正常能力与未见 ODD 回归？

## 2. 项目边界与取舍

### 2.1 必须完成

- CARLA–ROS 2 确定性 SIL 闭环。
- 独立可运行的经典专家与安全回退栈。
- 轻量端到端驾驶 VLA：图像/状态/导航/语言到多候选轨迹。
- 结构化交互世界模型：状态、Actor 响应、风险、奖励和终止预测。
- Safety Kernel：Validator、RATO-SCP、降级和最小风险停车。
- 参数化场景、风险搜索、反事实分支与最小反例。
- DAgger、接管偏好和反事实安全后训练。
- 可关闭的 Agentic Research Assistant：工具调用、证据约束和自身评测；确定性工作流在关闭 Agent 后仍可运行。
- 历史失败、正常能力、OOD 和实时性的统一准出。

### 2.2 明确不做

- 不训练数十亿参数以上的模型全量权重。
- 不训练多相机高清生成式视频世界模型。
- 不将 CARLA Actor 未来、真实 TTC 或隐藏意图泄漏给正式 VLA。
- 不让 LLM/VLM 直接输出未经 Validator 的控制指令。
- 不同时实现多套功能重复的学习规划器。
- 不把 CARLA 反事实结果等同于真实道路安全证明。
- 不声称当前系统是硬件在环。
- 不将调用模型 API 包装成核心算法贡献。
- 不提前写入未经 Evidence Registry 验证的性能数字。

### 2.3 三项核心研究贡献

1. **可验证驾驶 VLA**：将语义理解、语言条件、轨迹、风险和不确定性统一到端到端策略输出，并通过外部安全内核执行。
2. **交互式反事实世界模型**：显式学习 `P(other future | scene, ego action)`，与 CARLA 构成快慢双世界。
3. **可审计安全数据飞轮**：覆盖缺口、模型不确定性和失败簇驱动搜索、回放、最小化与后训练；Agent 只提高研发效率，确定性门禁负责数据纳入和版本准出。

纵向 QP/受限 RATO-SCP 是两级安全修复，Random/LHS/MAP-Elites 是失败发现方法；它们服务 VLA、世界模型和安全闭环，不单独包装成脱离系统的噱头。

## 3. 六平面系统架构

### 3.0 面向驾驶任务的主链：感知—理解—预测—决策—控制

```text
前视摄像头 + Ego 历史 + 导航/局部路线
                    ↓
             共享视觉/时序编码
                    ↓
        ┌───────────┴────────────┐
        ↓                        ↓
Fast/Slow VLA 多候选轨迹      经典专家候选轨迹
行为/关键Actor/风险/不确定性   可解释基线与回退
        └───────────┬────────────┘
                    ↓
        Validator 硬约束快速预筛
                    ↓
动作条件结构化世界模型：批量推演合法候选的交互未来
                    ↓
        策略仲裁：风险/进度/舒适/不确定性
                    ↓
      最终 Validator + 两级最小修复 + 回退
                    ↓
              MPC/PID 车辆控制
```

在该链路中：

- **感知**不是单独重建一套完整检测栈，而是由共享视觉编码器形成 VLA 所需表征；CARLA 真值只用于训练监督、安全检查和评价。
- **理解**由 VLA 的 Behavior、Critical Actor、Risk Horizon 和 Intended Action 结构化头完成；不把自由文本推理当作安全证据。
- **预测**由动作条件世界模型完成；它不是预测一个固定未来，而是比较“执行候选 A/B/C 后环境分别怎样响应”。
- **决策**只在硬约束预筛通过的候选中结合 VLA、经典候选、世界模型评分和不确定性仲裁，选中后再次执行最终安全检查。
- **控制**始终由显式 MPC/PID 执行，学习模型不直接绕过车辆约束。

项目首个正式输入组合限定为单前视 RGB、Ego 状态历史和导航/局部路线，以适配 RTX 4080 16GB。毫米波雷达、激光雷达和多相机保留在接口中，但不是核心系统成立的前置依赖。

典型超车过程：

1. VLA 识别慢车、左侧车道语义、关键后车和当前超车意图，生成继续跟车、立即变道、延迟变道等候选轨迹。
2. 世界模型对每个候选分别 rollout，预测后车让行/加速响应、未来占用和碰撞风险。
3. 仲裁器综合 VLA 概率、世界模型风险、经典专家建议和 OOD 分数选择候选。
4. RATO-SCP 对选中轨迹进行最小必要修复；无法修复则执行经典回退或保持车道减速。
5. MPC/PID 跟踪最终轨迹，真实结果进入数据闭环。

### 3.1 Simulation Plane

- CARLA：道路、车辆动力学、天气、交通灯、传感器和 Actor。
- ScenarioRunner/OpenSCENARIO：参数化场景执行。
- Fault Injector：延迟、丢包、定位偏移、视觉退化、低附着和执行器异常。
- 固定步长、唯一 tick master、固定随机种子和 Recorder。

### 3.2 Classic Expert Plane

- OpenDRIVE/Waypoint Lane Graph 与多目标 A*。
- Behavior Tree：跟车、停车、让行、换道、绕障和紧急停车。
- 常规道路：Frenet Lattice + S-T 速度规划。
- 复杂机动：Hybrid A* + Reeds–Shepp。
- LTV/SQP-RTI MPC + OSQP。
- PID/Pure Pursuit/LQR 基线与降级。

### 3.3 Driving VLA Plane

- 冻结或轻量微调的视觉编码器。
- Ego History、Route 和 Language Encoder。
- 多模态时序融合骨干。
- Fast 路径使用并行轨迹头在 10 Hz 目标频率输出 4～8 条、3～5 秒候选；目标频率必须实测，不是预先结果。
- Slow 路径只在 OOD、候选分歧或复杂交互时以 1～2 Hz 目标频率触发，输出结构化行为依据，不阻塞控制。
- Behavior、Critical Actor、Risk Horizon、Trajectory 和 Uncertainty 多任务头；自由文本 VQA 仅为非核心诊断能力。

### 3.4 Dual World Plane

- **Executable World**：CARLA 权威短时 rollout。
- **Learned World**：5～10 Hz 目标频率的轻量 object/vector/BEV latent world model，批量预测不同 Ego 候选下的 Actor 响应、占用、风险、奖励、终止和不确定性；不生成像素视频。
- Active Verification：学习世界不确定时才调用 CARLA 高成本验证。
- VLA 与 World 可缓存共享视觉特征以节省显存，但接口、checkpoint、可用性状态和消融开关必须独立。

### 3.5 Safety Plane

- Output Monitor。
- Trajectory Validator。
- Risk and Uncertainty Gate。
- 一级纵向 QP 默认最小修复；二级受限 RATO-SCP 仅在存在合法横向走廊时触发。
- Classic Shadow Expert 与策略仲裁。
- Minimal Risk / Emergency Filter。

### 3.6 Agent and Assurance Plane

- Scenario/Failure Research Assistant：提出覆盖假设、组织证据和根因候选，不修改真值。
- Deterministic Data/Release Workflow：schema、hash、统计和冻结阈值作最终裁决。
- Run/Model/Data/Evidence Registry。
- Regression Suite 与静态准出报告。

## 4. 纯软件部署与同步

候选结构：

```text
Windows Host
├── CARLA Server
├── NVIDIA Driver / GPU
└── 图形显示与录制
        │ TCP
        ▼
WSL2 Ubuntu
├── ROS 2
├── CARLA Client
├── Classic Expert
├── Driving VLA
├── World Model
├── Safety Kernel
├── Scenario Search
├── Agent Tools
└── Data / Regression / Report
```

具体 CARLA、ROS 2、Ubuntu、Python 和 ScenarioRunner 版本必须经过兼容性烟雾测试后写入 `versions.lock`，不在方案阶段写死。

同步规则：

- 唯一节点调用 `world.tick()`。
- `episode_id + carla_frame` 为数据主键。
- ROS 2 使用 `/clock` 仿真时间。
- 控制命令记录生成帧、计划执行帧和实际执行帧。
- 墙钟只用于真实延迟测量，不用于传感器对齐。
- 过期、乱序、连续缺帧或节点超时必须触发降级。

## 5. 统一策略接口

### 5.1 Policy Input

```text
SensorPacket
├── front_rgb / short visual history
├── ego_state_history
├── navigation_command
├── local_route_points
├── optional language instruction
├── simulation_timestamp
└── frame_id
```

正式策略输入禁止包含 Actor 真实未来、真实 TTC、碰撞标签、隐藏控制意图和测试集专家轨迹。CARLA 真值只用于监督标签、Safety Kernel 和评价。

### 5.2 Policy Output

```text
PolicyOutput
├── behavior_logits
├── K candidate trajectories / action chunks
├── trajectory probabilities
├── risk_score and intervention probability
├── uncertainty_score
├── critical_actor_id
├── structured_evidence
└── inference_metadata
```

候选轨迹至少包含 `x, y, yaw, velocity, acceleration, relative_time`。

### 5.3 Model-Agnostic Adapter

```python
class DrivingPolicy:
    def reset(self, context): ...
    def observe(self, observation): ...
    def infer(self, request) -> PolicyOutput: ...
    def report_uncertainty(self): ...
    def export_trace(self): ...
```

平台支持 `ClassicPolicyAdapter`、`VLAAdapter`、`ActionTokenAdapter` 和未来新策略，不因模型热点变化而重构整个系统。

## 6. 经典专家与规划控制

### 6.1 全局路由

从 OpenDRIVE/Waypoint 构建 Lane Graph，记录 Road/Lane ID、方向、前后继、换道、路口、速度限制、停止线和交通灯。多目标 A* 同时考虑路径长度、转向、换道、路口复杂度与历史风险，输出 Route Corridor。

### 6.2 双局部规划器

- **Frenet Lattice**：车道跟随、换道、跟车和常规绕行。
- **Hybrid A* + Reeds–Shepp**：道路阻断、狭窄绕障、掉头、倒车和脱困。

二者共享车辆模型、碰撞检测、道路边界和轨迹接口，由行为层根据场景选择。

### 6.3 S-T 与 MPC

将动态障碍物占用投影到 S-T 图，DP 搜索速度走廊，QP 平滑 `s/v/a/jerk`。控制层采用运动学自行车模型和 LTV/SQP-RTI MPC，使用 OSQP warm start、固定稀疏结构、超时和不可行恢复。PID 完成加速度/转角到 throttle/brake/steer 映射。

### 6.4 专家质量门禁

只有无碰撞、无严重违规、动力学可执行、舒适性合格且无连续超时的专家轨迹可以进入正向训练集。不合格结果只能进入失败数据集。

### 6.5 经典算法的五种角色

1. 独立基线。
2. VLA 专家标签生成器。
3. Shadow Expert。
4. VLA 失效时的安全回退。
5. 反事实与偏好数据中的 chosen candidate。

## 7. 端到端驾驶 VLA

### 7.1 定义

端到端边界限定为：

```text
前视视觉 + Ego 历史 + 导航/路线 + 可选语言条件
        → 行为 + 多候选未来轨迹 + 风险 + 不确定性 + 证据
```

底层 MPC、PID、Validator 和 Safety Shield 位于模型外部。

### 7.2 模型结构

```text
Front RGB / Visual History → Frozen Vision Encoder
Ego History                → Temporal State Encoder
Navigation / Route         → Route Encoder
Language Condition         → Small Language Encoder
                              ↓
                  Multimodal Temporal Fusion
                              ↓
├── Behavior Head
├── K-mode Trajectory/Action Head
├── Risk and Intervention Head
├── Critical Actor Head
├── Uncertainty Head
└── Structured Evidence Head
```

语言只提供任务、场景语义和安全条件，不承担连续控制。模型规模以单机训练与推理为硬约束，优先冻结视觉/语言骨干，使用 LoRA/QLoRA、混合精度和梯度累积。

### 7.3 训练目标

```math
L = λ_traj L_traj
  + λ_nll L_nll
  + λ_behavior L_behavior
  + λ_risk L_risk
  + λ_actor L_actor
  + λ_unc L_uncertainty
  + λ_evidence L_evidence
  + λ_smooth L_smooth
```

### 7.4 训练范式

1. **VLA/VLT Supervised Pre-training**：专家示范、行为、语言、轨迹和风险联合监督。
2. **Closed-loop DAgger**：VLA 驾驶，Shadow Expert 在偏离和风险上升时提供纠正。
3. **Pre-intervention Learning**：学习危险发生前的风险 Actor、时间域和接管概率。
4. **Counterfactual Preference Training**：使用同状态下不同动作真实结果构造 chosen/rejected。
5. **Risk-aware Post-training**：资源允许时使用轻量 offline RL/GRPO-LoRA；DPO 或 pairwise ranking 为稳定基线。

## 8. 交互式世界模型

### 8.1 学习目标

世界模型不生成高清未来视频，学习规划需要的结构化未来：

```text
输入：scene latent + ego action chunk + actor histories + map
输出：next latent + actor responses + occupancy + risk + reward + termination
```

核心条件分布：

```math
P(X_{others}^{t+1:t+H}, R, done | X^t, A_{ego}^{t:t+H})
```

这使模型能够比较“加速、减速、保持、换道或停车”对其他参与者响应和最终风险的影响。

### 8.2 快慢双世界

```text
大量候选动作
    ↓
Learned World 快速筛选与排序
    ↓ 不确定 / 高风险 / 高价值
CARLA Executable World 权威短时验证
    ↓
新 rollout 回流校正 Learned World
```

### 8.3 世界模型评价

- Actor/occupancy 预测误差。
- Collision Ranking Accuracy。
- Reward Prediction Error。
- Calibration 与 uncertainty–error correlation。
- 动作排序一致率。
- 节省的 CARLA rollout 比例。
- 未见场景中的失效检测能力。

世界模型结果不能单独作为安全证明；高风险决策最终必须通过 CARLA 或确定性 Safety Kernel。

## 9. Safety Kernel 与 RATO-SCP

### 9.1 执行链

```text
VLA / Classic Candidate
        ↓
Output Monitor
        ↓
Trajectory Validator
        ↓
Risk and Uncertainty Gate
        ↓
World Model Counterfactual Check
        ↓
RATO-SCP Repair
        ↓
Classic Fallback / Minimal Risk / Emergency
        ↓
MPC + PID
```

### 9.2 Trajectory Validator

逐项输出 `pass/fail、severity、first_violation_time、related_actor、related_rule、measured_margin`，检查数值、时间、连续性、道路、动力学、静动态碰撞、交通灯和跟踪可行性。

### 9.3 RATO-SCP

在 Frenet 坐标中定义：

```text
z_k = [s_k, d_k, v_k, a_k]
u_k = [jerk_k, lateral_rate_k]
ξ_k = safety slack
```

```math
min Σ ||z_k-z_{policy,k}||_Q²
  + Σ ||u_k||_R²
  + Σ ||u_k-u_{k-1}||_S²
  + ρΣξ_k
  - w_p s_N
```

约束包括纵向动力学、横向连续性、道路边界、速度/加速度/jerk、动态障碍物安全半空间、停止线和交通规则。通过信赖域内的连续凸化迭代，尽量保留 VLA 意图；进度项防止以持续停车获得虚假安全。

分层降级：

```text
RATO-SCP
→ RATO-Longitudinal
→ Classic Expert
→ Minimal Risk Stop
→ Emergency Brake
```

对照：Raw Policy、Rule Slowdown、Hard Reject、RATO-Longitudinal、RATO-SCP。必须同时报告安全、路线完成、无故停车、舒适性、修改量、成功率和时延。

## 10. 不确定性与选择性驾驶

第一版只保留互补且可解释的方法：

- 多候选/多头轨迹分歧。
- 场景 embedding 距离或 energy score OOD。
- 风险头温度缩放与概率校准。
- 世界模型 ensemble/预测分歧。

决策：

```text
低风险 + 低不确定性 → 执行 VLA
中风险 / 中不确定性 → World Check + RATO
高风险 / 高不确定性 → Classic Expert
无可行轨迹 / 连续异常 → Minimal Risk Stop
迫近碰撞 → Emergency Brake
```

评价 ECE、Brier、Risk–Coverage、OOD Recall、Selective Success、误接管和漏接管。

## 11. Agentic Safety Loop

Agent 不直接驾驶，只通过白名单工具参与研发闭环。

### 11.1 Agent 角色

- **Scenario/Failure Research Assistant**：根据覆盖缺口、OOD、世界模型不确定性和失败簇提出逻辑场景，并基于视频、BEV、ROS trace、VLA、世界模型和 Safety 证据生成根因候选。
- **Deterministic Data/Release Workflow**：查询、去重、泄漏检查、版本登记、统计和准出均由确定性程序完成；Agent 只能提出 manifest 和报告草案。
- 关闭 LLM 后，场景执行、数据门禁、回归和版本准出仍必须完整可用。

### 11.2 白名单工具

```text
validate_scenario
run_simulation
query_run_registry
search_risky_region
replay_failure
branch_counterfactual
minimize_counterexample
cluster_failures
build_training_slice
compare_policy_versions
generate_release_candidate
```

Agent 不生成并执行任意 Python/Shell，不修改安全阈值，不绕过测试集隔离，不自行批准模型发布。

### 11.3 Agent 评测

- Schema/Tool-call 合法率。
- 场景可执行率和无效场景比例。
- Failure Type Accuracy 与 Top-k Root Cause Recall。
- Evidence Grounding Rate 和幻觉率。
- 固定预算内发现独立失败数量。
- 数据泄漏拦截率和重复样本率。
- 相比固定脚本或随机策略的净收益。

## 12. 场景搜索与反事实验证

### 12.1 风险引导场景搜索

场景参数包括 Ego/NPC 速度、初始间距、相对位置、事件时机、遮挡、摩擦、视觉退化、状态延迟和丢包率。

```text
Logical Scenario
→ LHS 初始采样
→ CARLA rollout
→ 多目标风险指标
→ MAP-Elites 覆盖 archive 更新
→ 重复验证
→ Failure Minimization
→ Clustering / Regression Registry
```

与 Random 和 LHS 使用相同仿真预算比较；Manual 仅作为人工设计场景套件。无效初始化、Actor 未进入冲突区、生成失败、路线不存在或仿真崩溃必须标为 INVALID，不得计作高风险发现。

### 12.2 反事实分支

CARLA Recorder 只作为回放证据，不假设可以无损恢复任意内部状态。采用固定场景、固定种子、脚本化关键 Actor，从起点重跑到分支帧并检查状态容差，再分别执行：

1. VLA 原始轨迹。
2. 世界模型优选轨迹。
3. RATO 修复轨迹。
4. 经典专家轨迹。
5. 保守制动或其他采样轨迹。

分支前比较 Ego、关键 Actor、交通灯、路线和事件进度；超出容差标为不可比较。结果用于失败解释、偏好训练、世界模型训练和版本验证。

## 13. 数据闭环与后训练

### 13.1 事件窗口

```text
normal_window
pre_intervention_window
intervention_window
recovery_window
```

保存图像、状态、导航、VLA、经典专家、世界模型、RATO、风险、不确定性和执行结果。

### 13.2 数据类型

- SFT/VLT：高质量专家和稳定策略数据。
- DAgger：策略偏离时的专家纠正。
- Preference：相同状态下 chosen/rejected 轨迹。
- Reward/World：动作分支与真实后果。
- Risk：接管前风险增长、关键 Actor 和风险时间域。
- Regression：只用于测试，不进入训练。

### 13.3 数据门禁

- 严格隔离训练与测试 Town、Route、场景族和参数范围。
- chosen 必须通过安全和可执行性验证。
- rejected 必须有更差的真实结果或可靠证据。
- 仿真异常不能当作策略失败。
- 反事实状态不一致不得进入偏好集。
- 高度重复样本去重或降权。
- 每个数据版本记录来源策略、commit、场景和质量指标。

### 13.4 后训练目标

```math
L_total = L_vla
        + λ_pref L_preference
        + λ_risk L_risk
        + λ_value L_world/reward
        + λ_cal L_calibration
```

必须同时评价安全改善和正常能力回归，避免模型通过过度停车获得虚假提升。

## 14. 场景与故障体系

核心场景族：连续弯道、红灯停车、前车急刹、车辆切入/切出、主动换道、换道冲突、静态绕障、无保护左转、路口让行、行人/自行车横穿、遮挡、环岛汇入、道路封闭和多 Actor 交互。

故障族：状态延迟、消息丢失/乱序、控制保持、执行器饱和与死区、定位偏移、相机遮挡/亮度退化、低附着、策略/规划/控制超时。

场景层级：

```text
Functional Scenario
→ Logical Scenario
→ Concrete Scenario
→ Regression Case
→ Minimal Counterexample
```

## 15. 指标与实验

### 15.1 驾驶与安全

- Route Completion、Driving Score。
- Collision、Minimum TTC/Distance。
- 红灯、停止线和车道违规。
- Intervention、Repeat Failure、Minimal Risk Success。

### 15.2 VLA

- Trajectory ADE/FDE/NLL。
- Behavior、Risk AUC/F1、Critical Actor Accuracy。
- Language/Instruction–Action Consistency。
- Evidence Grounding。
- ECE、Brier、Risk–Coverage、OOD Recall。

### 15.3 World Model

- 状态/Actor/occupancy 误差。
- Reward Error、Collision Ranking、Action Ranking。
- Calibration、uncertainty–error correlation。
- CARLA rollout 节省率与错误筛选率。

### 15.4 Safety/RATO

- 修复成功率、约束违反率和 slack 使用率。
- 修改范数、安全裕度、进度损失和无故停车率。
- P50/P95/P99 求解时间和降级率。

### 15.5 Agent/Search

- 首次失败 rollout、固定预算独立失败数。
- 无效场景、复现率、最小反例距离。
- Tool-call 合法率、诊断准确率、证据率和幻觉率。

### 15.6 数据闭环

- Hard Case 利用率和偏好一致性。
- Pre-intervention Recall 与 Hazard Anticipation Time。
- 重复失败下降和旧能力回归。
- 未见 ODD 的性能与置信度。

### 15.7 核心实验矩阵

发布配置：Classic、VLA+Safety、VLA+World+Safety、PostTrained-Full。
训练：SFT、+DAgger、+Risk Anticipation、+CARLA-verified Preference。
搜索：Engineered Suite、Random、LHS、MAP-Elites。
安全：Hard Reject、RATO-Longitudinal、RATO-SCP、Classic Fallback。
世界：无世界模型、Reward Only、Interactive World、Interactive World + Active CARLA Verification。

所有关键实验采用多个随机种子，报告均值、标准差和置信区间；碰撞等稀有事件使用 bootstrap。安全、效率和舒适性采用 Pareto 分析，不用单一加权分数掩盖退化。

## 16. 可复现性、版本与证据

### 16.1 Registry

- Run Registry：场景、参数、seed、代码、策略、数据、环境、状态和指标。
- Model Registry：架构、权重 hash、训练配置、数据版本和评测结果。
- Data Registry：来源、划分、质量、泄漏检查和许可。
- Scenario Registry：逻辑场景、参数空间、有效性规则和历史失败。
- Evidence Registry：所有可对外声明成果的证据。

Evidence 条目：

```text
claim_id
claim_text
status: PLANNED / IMPLEMENTED / MEASURED / VERIFIED
code_path
git_commit
run_ids
figure_path
video_timestamp
document_path
verified_date
```

只有 VERIFIED 内容可进入简历和答辩结果。

### 16.2 自动测试

- 单元测试：坐标、采样、约束、指标、schema。
- 属性测试：动力学、边界和安全不变量。
- 集成测试：ROS 2–CARLA 闭环和同步。
- 模型测试：输入输出、数值稳定、超时和 OOD。
- 回归测试：历史失败与正常能力。
- 压力测试：延迟、丢包、连续运行和资源占用。

## 17. 依赖门禁（不使用时间节点）

### G0：环境与确定性

- Windows/WSL2/CARLA/ROS 2 烟雾测试通过。
- 固定步长、tick master 和 frame 对齐稳定。
- `versions.lock` 和环境检查脚本完成。

### G1：经典专家

- Route、行为、Frenet/Hybrid、S-T、MPC 闭环运行。
- 专家质量门禁和自动指标完成。

### G2：Safety Kernel

- Validator、状态机、纵向 QP、受限 RATO 和回退可运行；学习模块全部不可用时仍能闭环。
- 故障注入可以触发预期降级。

### G3：轻量 VLA

- 数据划分和特权信息审计通过。
- VLA 完成非语言多候选、Fast/Slow、open-loop 与 VLA+Safety 闭环对照。
- 正式推理不读取 CARLA 隐藏真值。

### G4：场景搜索与反事实

- Random/LHS 与覆盖引导 MAP-Elites 公平对照。
- 失败可复现、可最小化。
- 分支状态一致性检查通过。

### G5：世界模型

- 世界模型有明确动作条件和交互响应。
- 与简单运动学/Reward 基线比较。
- 不确定时能够主动请求 CARLA 验证。

### G6：安全后训练

- Event/Hard Case、DAgger、CARLA 验证 Preference 和 Risk 数据门禁通过。
- 新策略同时通过历史失败和正常能力回归。

### G7：Agentic Loop

- Agent 仅调用白名单工具且关闭 Agent 后确定性流程仍可运行。
- Agent 相比固定脚本的净收益可以为负，但必须可量化并保留。
- 诊断和场景输出有 grounding 与幻觉评测。

### G8：版本准出

- Classic、VLA+Safety、VLA+World+Safety、PostTrained-Full 四发布配置及核心消融完成。
- 未见 ODD、故障、实时性和资源评测完成。
- 自动生成可追溯准出报告。

停止规则：前置门禁不通过时，不以新增热点模块绕开基础问题；RATO 未优于简单基线、MAP-Elites 未优于 Random/LHS、世界模型未优于简单预测、Agent 未优于固定脚本时，均保留实现和负结论，但不得默认启用或宣称有效。

## 18. 单机资源策略


### 18.1 固定硬件基线

```text
GPU: NVIDIA GeForce RTX 4080 Desktop, 16 GB VRAM
CPU: Intel Core i5-13600KF, 14 cores / 20 threads
Machine: single workstation, no cluster, no second GPU
```

所有模型、仿真并发和实验预算必须在该硬件上实际验证。项目不以未来可能获得的服务器资源作为必需条件。

### 18.2 显存预算

目标预算而非预先声称的实测值：

| Profile | GPU 主任务 | 显存控制原则 |
|---|---|---|
| Simulation | CARLA + 量化 VLA/World 推理 | CARLA Low/No Rendering；单前视低分辨率；共享特征缓存；VLA/World 流水或串行调度 |
| VLA Training | 冻结视觉/语言骨干 + LoRA/QLoRA | 4-bit/8-bit 权重、BF16/FP16、gradient checkpointing、梯度累积 |
| World Training | Vector/BEV latent dynamics | 不生成像素视频；控制 rollout horizon 与 batch |
| Agent | 本地量化小模型或受控 API | 不与 CARLA 高画质或训练任务并发 |
| Regression | CARLA + 单发布配置 | 四发布配置串行加载；控制与 Safety 不等待 GPU，GPU 超时走冻结回退 |

VLA 第一版优先选择可在 16GB 显存内微调的视觉编码器 + 小型时序/语言骨干，不把 10B/32B 工业模型本体作为训练目标。大型公开模型只作为架构参考、离线教师或可选 API，不作为项目运行依赖。

### 18.3 CPU 与进程调度

i5-13600KF 的 14 核 20 线程用于 CARLA client、ROS 2、数据编码和场景搜索。总控器必须：

- 为 CARLA client、ROS 2 executor、数据 writer 和监控器设置独立进程。
- 限制 PyTorch/DataLoader worker，避免占满 CPU 导致 tick 抖动。
- 数据压缩和视频编码异步执行，不阻塞控制回路。
- 记录每个进程 CPU、内存、GPU、显存和 I/O。
- 检测 GPU OOM、CARLA 断连、ROS topic 超时和磁盘空间不足并安全退出。

### 18.4 分时运行 Profile

- Simulation：CARLA Low/No Rendering + ROS 2 + 单个策略推理。
- Data Generation：经典专家批量生成，限制图像分辨率和 NPC 数量。
- VLA Training：关闭 CARLA，冻结骨干，LoRA/QLoRA、混合精度和梯度累积。
- World Training：优先 vector/latent 状态，不生成像素视频。
- Counterfactual：仅重放失败场景，串行运行短时分支。
- Agent：关闭训练任务，本地量化模型或受控 API 处理日志与工具调用。
- Regression：策略版本依次运行，统一写入 DuckDB/Parquet。

大文件（模型、rosbag、Recorder、Parquet、视频）不直接进入 Git，只登记路径、hash、来源和版本。

## 18A. Codex 全自动执行设计

“全自动”定义为：环境准备完成后，Codex 能通过版本化代码和一个总控入口执行构建、测试、仿真、训练、搜索、失败恢复、报告和证据登记；不依赖手工复制命令、手工改参数或手工整理结果。

### 18A.1 总控入口

```text
sdf doctor                 环境、驱动、CARLA、ROS 2、GPU 和磁盘检查
sdf build                  构建 C++/Python/ROS 2 工作区
sdf test                   单元、属性、集成和烟雾测试
sdf sim run <config>       运行一个场景/策略
sdf data collect <suite>   批量生成专家或失败数据
sdf train vla <config>     VLA 训练、断点续训和评测
sdf train world <config>   世界模型训练与校准
sdf search <config>        Random/LHS/MAP-Elites 搜索
sdf replay <run_id>        失败复现和反事实分支
sdf regress <release>      全量回归
sdf report <release>       生成静态报告与 Evidence Bundle
sdf status                 查看任务、资源、失败和恢复点
```

### 18A.2 任务状态机

每项任务使用持久化状态：

```text
PENDING → PRECHECK → RUNNING → VALIDATING → COMPLETED
                         ├→ RETRYABLE_FAILURE
                         ├→ BLOCKED_EXTERNAL
                         └→ FAILED_FINAL
```

任务保存 config hash、代码版本、输入数据、checkpoint、日志、资源峰值、失败原因和输出 artifact。进程中断后从最近合法 checkpoint 恢复，不重复完成的 rollout 或 epoch。

### 18A.3 自动故障处理

- GPU OOM：降低 batch/并发，启用梯度累积，记录配置变更后重试。
- CARLA 断连：清理本轮 Actor、重启 client、从 scenario seed 重跑。
- 场景 INVALID：记录原因，不计入风险样本，继续搜索。
- OSQP/SCP 无解：按降级链输出证据，不静默丢弃。
- ROS topic 超时：进入 Minimal Risk，保存 bag 和 trace。
- 磁盘不足：停止生成新数据，保留已完成索引，不删除用户文件。
- 训练 NaN：回滚 checkpoint、降低学习率/关闭异常样本并生成诊断报告。

### 18A.4 自动化边界

Codex 可以自动完成仓库内代码、配置、测试、脚本、训练、仿真编排、结果分析和文档更新。以下外部动作仍可能需要用户一次性授权或物理操作：

- 安装/升级 NVIDIA 驱动、WSL2、CARLA 和需要管理员权限的软件。
- 登录第三方模型或数据服务、接受许可证。
- Windows/WSL2 重启与防火墙弹窗。
- 启动无法被命令行可靠管理的 GUI 程序。
- 购买存储、API 额度或外部硬件。

这些步骤必须转化为明确 preflight blocker；用户完成一次后，后续工作继续自动运行。

## 19. 工程目录

```text
safedrive_foundry/
├── README.md
├── versions.lock
├── CHANGELOG.md
├── docs/
│   ├── product_requirements.md
│   ├── system_architecture.md
│   ├── interface_spec.md
│   ├── experiment_protocol.md
│   ├── acceptance_matrix.md
│   └── decisions/
├── configs/
├── ros2_ws/src/
│   ├── sdf_interfaces/
│   ├── carla_adapter/
│   ├── clock_master/
│   ├── classic_expert/
│   ├── policy_runtime/
│   ├── world_model_runtime/
│   ├── safety_kernel/
│   ├── rato_optimizer/
│   ├── scenario_manager/
│   ├── fault_injector/
│   ├── counterfactual_runner/
│   └── metrics_collector/
├── classic_stack/
├── driving_vla/
├── world_model/
├── agents/
├── validation/
├── data_pipeline/
├── registry/
├── tests/
├── scripts/
└── artifacts/
```

## 20. 最终交付

### 软件

- Classic Expert 双局部规划栈。
- Driving VLA 与模型无关适配器。
- Interactive World Model 与 CARLA active verification。
- Safety Kernel、RATO-SCP 和降级控制。
- Agentic Scenario/Search/Diagnosis/Data/Release 工具链。
- 一键环境检查、场景运行、回归和静态报告。

### 数据

- VLA/VLT 专家数据。
- 世界模型 action–response rollout 数据。
- DAgger、Risk、Preference 和 Reward 数据。
- Regression、Minimal Counterexample 和 Evidence Registry。

### 展示证据

- Classic、Raw VLA、World-selected、RATO 和 Expert 轨迹对比。
- 世界模型预测与 CARLA 真实后果对比。
- Research Assistant 与固定模板对比；MAP-Elites 与 Random/LHS 搜索对比。
- 接管前风险时间线和反事实分支。
- 后训练前后安全、效率、舒适性和回归对比。
- 架构图、演示视频、技术报告和版本准出报告。

## 21. 简历表述边界

项目标题：

> **SafeDrive Foundry｜驾驶 VLA、反事实世界模型与智能体安全闭环平台**

目标表述框架：

- 构建 CARLA–ROS 2 纯软件在环研发平台，打通经典专家、端到端驾驶 VLA、交互世界模型、Safety Kernel、主动场景搜索和模型准出闭环。
- 设计从视觉、车辆历史、导航与语言条件到多候选轨迹、风险、不确定性和结构化证据的轻量驾驶 VLA，并通过 DAgger、接管偏好和反事实后训练提升闭环表现。
- 构建动作条件交互世界模型预测周围参与者响应、碰撞风险和奖励，通过不确定性选择高价值 CARLA rollout，形成学习世界与可执行世界协同验证。
- 将危险轨迹修复建模为带道路、动力学、动态障碍物和交通规则约束的序列凸优化问题，并以 Classic、Hard Reject、纵向 QP 等基线验证安全与效率权衡。
- 开发受控工具调用 Agent，围绕覆盖缺口和失败簇主动生成测试、搜索并最小化反例、构建训练切片和准出证据，并与固定脚本和随机测试进行同预算对照。

所有量化位置必须由 VERIFIED Evidence 替换。纯软件阶段只称 SIL；CARLA 称可执行仿真世界，不称自研生成式世界模型；使用预训练骨干和开源求解器必须如实说明。

## 22. 前沿论文与工业实践到本项目的映射

本节不是背景罗列，而是说明每条外部信息具体改变了什么设计。

### 22.1 Waymo：多任务协同，但必须验证中间决策

Waymo 的 EMMA 将相机与文本映射到规划轨迹、3D 目标和道路图，并报告多任务共同训练相较独立训练有正迁移；同时其公开说明也明确指出长时序、推理成本、仿真评测和中间决策验证仍是挑战。[Waymo EMMA](https://waymo.com/blog/2024/10/introducing-emma/)

对应决策：

- VLA 不只训练轨迹头，同时训练 Behavior、Risk、Critical Actor 和 Evidence 辅助头。
- 辅助头不是为了重建完整感知栈，而是提供可验证的中间语义。
- 评测增加 reasoning/action consistency：语言所指风险 Actor、行为和实际轨迹必须一致。
- 正式结论以 CARLA 闭环和 Safety Event 为准，不以 open-loop 文本质量代替驾驶表现。

### 22.2 Wayve：语言输出必须与真实驾驶动作对齐

Wayve LINGO-2 将语言理解与闭环驾驶动作结合，而非停留在开环驾驶评论。[Wayve LINGO-2](https://wayve.ai/thinking/lingo-2-driving-with-language/)

对应决策：

- 项目不做独立“驾驶解说模型”。
- Evidence Head 必须绑定 `critical_actor_id、conflict_zone、risk_horizon、selected_trajectory_id`。
- 设计反事实一致性测试：若删除模型声称的关键 Actor，风险判断和动作是否按预期变化。
- 错误但语言流畅的解释应被判为 grounding failure。

### 22.3 NVIDIA Alpamayo：推理—动作一致性、闭环后训练和独立安全体系

NVIDIA Alpamayo-R1 采用 Chain of Causation 数据、推理模型与轨迹解码器、多阶段 SFT/RL，并强调 reasoning–action consistency；Alpamayo 平台同时包含闭环仿真、强化学习基础设施和独立安全体系，而不是单靠 VLA 本身承担安全。[Alpamayo-R1](https://research.nvidia.com/labs/avg/publication/wang.luo.etal.arxiv2025/)、[NVIDIA Alpamayo](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/)

对应决策：

- VLA 采用快/慢两种推理模式：普通场景只输出轨迹和结构化证据，复杂场景触发更完整语义推理。
- 后训练从稳定的 SFT/DAgger/Pairwise 或 DPO 开始，资源允许再增加轻量 RL。
- Safety Kernel 独立存在，VLA 更新不得更改安全约束。
- 4080 16GB 不复现 10B/32B 工业模型，只复现可测的训练范式和闭环关系。

### 22.4 AutoVLA 与 Reasoning-VLA：自适应思考和并行动作查询

AutoVLA 区分快速轨迹生成与慢速推理，并使用强化微调减少简单场景中的不必要推理；Reasoning-VLA 通过并行动作查询生成连续轨迹，强调推理效率和跨数据泛化。[AutoVLA](https://arxiv.org/abs/2506.13757)、[Reasoning-VLA](https://arxiv.org/abs/2511.19912)

对应决策：

- 不让语言自回归逐点生成连续轨迹；轨迹由并行 query/action head 输出。
- System 2 只在 OOD、候选分歧、复杂交互或 Safety 临界状态触发。
- 评测同时报告推理触发率、额外延迟、净驾驶收益和错误触发成本。

### 22.5 世界模型：动作条件因果性比画质更重要

近期工作指出，世界模型在无扰动指标上表现良好，并不代表它适合策略训练；评价必须覆盖 Ego 动作干预和因果相关 Actor。[Beyond Simulation](https://arxiv.org/abs/2508.01922) MAP-World 则展示了轻量世界模型按候选轨迹 rollout 未来 BEV 语义并辅助多模态规划的路径。[MAP-World](https://arxiv.org/abs/2511.20156)

对应决策：

- 本项目世界模型必须显式以 Ego action chunk 为条件。
- 训练数据必须包含同一场景的多动作分支，而非只有专家轨迹。
- 评价加入 action sensitivity、causal actor response 和 ranking consistency。
- 世界模型必须与 CV/CTRV、Reward MLP 和无动作条件模型比较。
- 学习世界不确定时调用 CARLA；不能将预测未来当作权威真值。

### 22.6 Waymo/Wayve/NVIDIA 的生成世界：工业规模方向，项目做可验证子集

Waymo 公开的生成式世界模型强调可控、多传感器和长尾场景；Wayve GAIA-2 强调可控多视角生成；NVIDIA Cosmos 将 Predict、Transfer、Embed 和 Reason 分别用于未来生成、条件变换、数据整理和推理。[Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)、[Wayve GAIA-2](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf)、[NVIDIA Cosmos](https://developer.nvidia.com/blog/simplify-end-to-end-autonomous-vehicle-development-with-new-nvidia-cosmos-world-foundation-models/)

对应取舍：

- 单卡项目不复现多视角像素生成。
- 将“世界模型”限定为动作条件的 vector/BEV latent dynamics 和 reward/risk model。
- 视觉天气变化可由 CARLA 参数化或现成生成工具作为可选数据增强，不作为核心贡献。
- 核心贡献放在 world model 是否改善候选排序、减少 CARLA rollout 和发现失败，而非视频是否逼真。

### 22.7 Waabi：仿真器本身也需要验证

Waabi 强调仿真真实性不是单纯视觉逼真，而是同一输入下虚拟环境与真实系统是否产生一致驾驶行为；其研究还覆盖语言场景编排与约束满足。[Waabi simulator realism](https://waabi.ai/insights/simulator-realism-the-new-safety-standard-for-the-av-industry)、[Waabi Research](https://waabi.ai/research)

对应决策：

- 本项目把 simulator/world-model validity 作为独立风险，不默认 CARLA 就是绝对真值。
- 对世界模型做 paired rollout；对 CARLA 配置做固定版本、确定性和物理参数审计。
- Agent 的自然语言场景必须经过 schema、约束满足和地图合法性检查后才能运行。
- 仿真结论只支持软件在环比较，不外推成真实道路安全声明。

### 22.8 Zoox：三类场景来源与持续版本准出

Zoox 公开将场景分为工程师设计、日志重建和系统生成三类，并使用仿真持续验证每个软件版本；其安全体系强调设计、验证和准出贯穿开发，而不是测试末尾附加。[Zoox structured testing](https://zoox.com/journal/structured-testing)、[Zoox edge-case loop](https://zoox.com/journal/edge-case-testing-zoox)、[Zoox mission assurance](https://zoox.com/journal/system-design-mission-assurance)

对应决策：

- Scenario Registry 同时接收 Engineered、Failure-derived、System-generated 三路场景。
- Agent 不替代工程场景和历史失败，而是补充尚未覆盖的区域。
- 每个策略版本必须运行历史失败与正常能力回归。
- Release Gate 输出证据包，不能只比较单一 Driving Score。

### 22.9 工业化三闭环总结

外部实践共同指向以下结构：

```text
Runtime Autonomy Loop
VLA / Planner / World / Safety

Development Data Loop
Logs / Simulation / Counterfactual / Post-training

Assurance Loop
Requirements / Scenarios / Regression / Evidence / Release
```

SafeDrive Foundry 的差异化不是缩小版复刻某一家企业，而是在单 RTX 4080 上做出这三个闭环的最小但完整实现，并对每个学习模块设置简单基线、失效检测和确定性回退。

## 23. 关键参考资料

### VLA 与后训练

- [Waymo EMMA](https://waymo.com/research/emma/)
- [Wayve LINGO-2](https://wayve.ai/thinking/lingo-2-driving-with-language/)
- [NVIDIA Alpamayo-R1](https://research.nvidia.com/labs/avg/publication/wang.luo.etal.arxiv2025/)
- [AutoVLA](https://arxiv.org/abs/2506.13757)
- [Reasoning-VLA](https://arxiv.org/abs/2511.19912)
- [Poutine: VLT Pre-training and RL Post-training](https://arxiv.org/abs/2506.11234)
- [TakeVLA: Learning from Takeover Data](https://arxiv.org/abs/2603.14972)

### 世界模型与仿真

- [Waymo World Model](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)
- [Wayve GAIA-2](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf)
- [MAP-World](https://arxiv.org/abs/2511.20156)
- [Beyond Simulation: World Models for Planning and Causality](https://arxiv.org/abs/2508.01922)
- [NVIDIA Cosmos World Foundation Models](https://developer.nvidia.com/blog/simplify-end-to-end-autonomous-vehicle-development-with-new-nvidia-cosmos-world-foundation-models/)
- [Waabi Research](https://waabi.ai/research)

### 闭环评测与工业安全

- [CARLA Leaderboard](https://leaderboard.carla.org/)
- [SafeBench](https://safebench.github.io/)
- [Bench2Drive](https://proceedings.neurips.cc/paper_files/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf)
- [ASAM OpenSCENARIO 2.0](https://www.asam.net/static_downloads/public/asam-openscenario/2.0.0/introduction.html)
- [Zoox Structured Testing](https://zoox.com/journal/structured-testing)
- [Zoox System Design and Mission Assurance](https://zoox.com/journal/system-design-mission-assurance)

## 24. 最终原则

1. VLA 是核心驾驶策略，不是聊天层包装。
2. 端到端止于轨迹/动作块，底层控制和安全约束保持显式。
3. 世界模型必须预测动作条件下的交互未来，而不只是拟合静态 reward。
4. Agent 必须调用可审计工具并接受独立评测，不进入实时控制环。
5. 经典算法负责教师、Shadow、修复、回退和安全下限。
6. 每个热点只有一个核心职责，避免功能重复。
7. 所有提升都必须同时报告安全、效率、舒适性、泛化和旧能力回归。
8. 所有成果必须可追溯至代码、配置、数据、run、图表和视频。
9. 项目使用依赖门禁推进，不使用日历工期包装完成度。
10. 热点模型可以替换，统一接口、反事实闭环、安全门禁和证据体系保持稳定。
