# G2～G8 工业化 VLA + World Model 系统规范

> 本文定义 G2～G8 跨阶段必须保持一致的运行链、接口、资源和降级原则。具体实现范围和验收以当前任务文件为准；频率、时延和资源数值均为待实测目标，不是既有结果。

## 1. 系统目标与不可变边界

SafeDrive Foundry 的必做学习主线是轻量驾驶 VLA；动作条件结构化 World Model 只有在 oracle best-of-K 证明存在选择空间后才进入。独立 Safety Kernel 决定轨迹是否允许执行，Classic Expert 提供基线、标签、Shadow 和 Hybrid 候选。任何模型升级都不得改变以下边界：

1. VLA 和 World Model 不直接输出未经约束的车辆控制；
2. World Model 是候选结果预测与软排序器，不是安全真值；
3. Safety 硬约束、slack 上限、Emergency 和回退权限不能被学习模块修改；
4. MPC/PID 50Hz 控制不等待 GPU、Active CARLA 或 Agent；
5. Agent 只参与离线研发，不进入实时执行链；
6. Oracle 与 Observable 严格隔离，Regression 不进入训练或调参。

## 2. 在线执行链

```text
Current camera + low-dimensional Ego history + Route/navigation
                          │
                 temporal feature cache
                          │
             ┌────────────┴────────────┐
             │                         │
       VLA-V0/V1                 Classic Expert
    K1/K2 trajectory             Hybrid candidate/shadow
             └────────────┬────────────┘
                          │
       Candidate freshness + hard pre-validation
                          │
        optional gated action-conditioned World
        batched actor/occupancy/risk rollouts
                          │
             deterministic soft arbitration
                          │
                   final Validator
                          │
        longitudinal QP → restricted RATO-SCP
                          │
       Classic fallback / Minimal Risk / Emergency
                          │
                    50Hz MPC/PID
```

硬预筛的目的，是避免 World Model 为数值非法、过期或明显违反硬约束的轨迹浪费算力。最终 Validator 的目的，是确保候选排序、缓存和异步调度没有改变安全事实。

## 3. 六个跨阶段接口

### 3.0 ObservationBundle 与 TrackedObjectSet

VLA、World 和 Safety 必须共享同一帧身份和权限标记，但允许读取不同字段子集。`ObservationBundle` 至少包含 camera history、Ego history、route/navigation、drivable area、traffic lights、freshness 和 Observable/Oracle 权限；`TrackedObjectSet` 至少包含 actor_id、类别、位置/速度/朝向、尺寸、协方差、观测时间、丢失状态和来源。

项目不在 G2～G3 重建完整感知栈；CARLA/ROS Observable adapter 提供版本化对象与道路观测。CARLA 隐藏未来、真实 TTC 和隐藏意图只能进入 privileged/evaluation 字段。三个消费者不得各自建立无法对齐的 Actor 身份。

### 3.1 PolicyCandidateSet

最少包含：

- `run_id`、`frame_id`、`scenario_id`、`model_id`；
- `candidate_id`、来源、生成时间、有效期和候选概率；
- 首版 V0/V1 为 2.5 秒、T10、K1/K2，含时间戳、坐标系和动力学元数据；
- behavior/intended_action；复杂 grounding 为 optional；
- Runtime Guard、FAST/REASON、margin 和 availability。

语言解释不是必需字段，也不参与 Safety 硬判断。

### 3.2 WorldRolloutBatch

每个输入候选对应一个不可混淆的 rollout：

- `candidate_id`、world model/data/config hash；
- World-V0 为2.5秒、N≤8、M=1 Actor future；
- collision/TTC/off-road 与两候选排序；复杂多模态/不确定度为升级项；
- unavailable、timeout、uncalibrated、invalid-input 等显式错误。

不得用缺失输出或异常隐式表示“低风险”。

### 3.3 SafetyDecision

最少记录：

- 候选预筛、最终检查和修复前后轨迹 ID；
- 每条约束的裕度、首次违反时间、关联 Actor/规则；
- hard reject、QP、RATO、Classic、Minimal Risk、Emergency 决策；
- 修改范数、slack、进度损失、solver 状态和时延；
- 恢复条件、状态机转移和最终交给控制器的轨迹。

### 3.4 SafetyEvent / TrainingEvent

SafetyEvent 是在线事实；TrainingEvent 是离线窗口。转换时必须保留：

- normal/pre-event/intervention/recovery 边界；
- VLA、World、Classic、Safety 和 executed action；
- CARLA evaluation-only 真值与其权限标签；
- 数据纳入/拒绝原因、failure cluster 和 split；
- 全链路 run/model/data/scenario/evidence ID。

## 4. 调度与资源目标

| 模块 | 目标频率 | 超时或不可用时 |
|---|---:|---|
| MPC/PID | 50Hz，20ms 控制周期 | 使用冻结控制降级，不等待学习模块 |
| Safety 状态监控 | 50Hz | 进入 DEGRADED/MINIMAL_RISK/EMERGENCY |
| VLA-V0/V1 | 首门槛P95≤200ms（约5Hz），10Hz后续优化 | VLA_SAFETY进入MRM；Hybrid可Classic |
| World-V0 | 门禁通过后目标约5Hz | 跳过World排序，退化为VLA+Safety |
| VLA-V2 REASON | 事件触发、optional | 不阻塞FAST或控制 |
| Active CARLA / Agent | Optional/离线 | 不影响在线车辆或发布主线 |

实现时必须测量端到端 freshness，而不仅是单模型 kernel latency。GPU 任务采用共享只读 feature cache 和串行/流水调度；VLA/World checkpoint、availability 和消融开关保持独立。训练、CARLA 高画质、Agent 本地大模型不得并发争抢 16GB 显存。

## 5. 轻量 VLA 设计

### 5.1 输入

VLA-V0使用上游所需当前Observation；VLA-V1固定当前前视图像、当前ego、4时刻低维ego history和Route/navigation。完整图像历史、多相机、LiDAR和radar不是第一版。

### 5.2 模型

- V0冻结上游并确定性canonicalize为K1/T10/2.5s；
- V1只训练低维history adapter、K2 nominal/conservative和简单概率；
- V2学习型Router只做FAST/REASON，DEFER由deterministic Runtime Guard负责；
- heads-only优先，必要时少量LoRA；
- 大模型仅作离线教师或标签审查，不成为部署依赖。

### 5.3 输出与训练

V0输出K1，V1输出K2，两者均为T10/2.5s。Grounding最多keep/slow/stop/turn且非闭环前置。只有oracle best-of-K显示选择空间、候选不坍塌时才扩K4/3秒或复杂grounding/OOD。

## 6. 动作条件 World Model 设计

### 6.1 门禁后必做能力

先比较VLA top-1与oracle best-of-K并登记科学标签；本项目World-V0仍须实现接入（弱选择空间时C2可负）。World-V0学习`P(other future | scene, ego trajectory)`：

- ego、最多8个Actor、简化道路与K2候选；
- K2/T10/N8/M1、约4M～8M；
- actor future、collision、TTC、off-road与两候选排序；
- invalid/unavailable/timeout显式错误。

### 6.2 禁止捷径

- 不生成高清像素视频作为核心目标；
- 不只训练 expert action，必须包含非专家、危险和扰动候选；
- 不只报告 ADE/FDE，必须报告 action ranking/regret 和闭环结果；
- 不让 World 生成并验证自己的安全标签；
- 不把共享 latent 变成无法关闭和无法消融的耦合。

### 6.3 有效性证明

至少执行action shuffle/swap、无动作条件和CV/CTRV/Reward MLP基线。只有闭环A/B稳定改善且护栏不退化，才允许扩N/M/T/K或默认在线；否则Shadow/离线。

## 7. Safety 与回退

一级纵向 QP 是默认修复器，负责减速和停车。受限 RATO-SCP 只在存在合法横向走廊、纵向修复不足且预算允许时触发。任何 solver 超时、振荡、不可行或 stale 都进入冻结回退链。

学习模块全失效时，系统仍必须提供：

```text
Classic candidate → Validator → optional QP/RATO → MPC/PID
                                  ↓
                    Minimal Risk / Emergency
```

RATO 未优于纵向 QP时默认关闭二级，但不影响 Validator、状态机和 Classic 回退作为 G2 必需交付。

## 8. 场景、反事实与数据飞轮

G4A固定场景Registry、seed/replay、initial-state hash、20～40困难场景、可比K2分支和oracle best-of-K。MAP-Elites、自动搜索、最小化、聚类和大规模覆盖均属于G4B optional。

反事实分支必须检查 Ego、关键 Actor、信号、路线和事件进度容差。超容差结果标不可比较；CARLA Recorder 不视为无损状态恢复。

G6第一轮只使用：

- 版本化专家纠正；
- VLA失败前2～4秒窗口；
- Classic/Safety/人工规则生成的修正轨迹；
- 通过泄漏、冲突、许可、去重和分布门禁的数据。

只训练一个监督adapter或少量LoRA checkpoint，并用同一冻结困难集/正常集做前后比较。第一轮禁止PPO、GRPO、RL、多种preference、多轮自动飞轮和同时训练VLA/World。

## 9. Agent 与确定性工作流

Research Assistant 可提出场景和根因候选，但不能执行任意命令、修改真值、阈值、split 或发布结论。Data/Release 工作流必须完全确定性：

```text
Agent draft (optional)
        ↓
schema / hash / leakage / statistics / frozen thresholds
        ↓
human approval where required
        ↓
deterministic manifest and release decision
```

C6未执行或为负时关闭Agent，不影响VLA+Safety主线及条件式World准出。

## 10. 四个发布配置与负结果策略

最终只维护：

1. `Classic`；
2. `VLA+Safety`；
3. `VLA+World+Safety`；
4. `Post-trained VLA+World+Safety`。

World未通过入口或净收益门禁时，第3/4项标Shadow/SKIPPED/negative，并正式保留`Post-trained VLA+Safety`结果；Hybrid单独报告，不混入主要因果阶梯。

核心主张可以得到负结果，范围规则为：

- VLA 必须完成训练、无特权审计、真实闭环和模型卡；
- World 先做 oracle best-of-K 科学标注；无选择空间时仍接入模块，C2 记负/无增益，不得假冒正收益；
- 门禁通过后必须完成 World-V0、干预消融、候选排序和模型卡；
- World 无收益时转 Shadow/离线，而不是伪造在线价值；
- 后训练无收益时保留数据/模型谱系和抗遗忘回归；
- Agent 为 Optional/After Release，不阻塞准出；
- Safety 的 Validator、状态机和回退失败则系统不得准出。

## 11. 阶段交付顺序

```text
G2  独立安全执行边界
G3  数据隔离 + 非语言基线 + 轻量 VLA 闭环
G4A 固定场景 + 可比分支 + oracle best-of-K；G4B optional
G5  门禁通过时完成 World-V0；否则保存负结论
G6  一轮困难样本监督适配与抗遗忘
G7  Optional / After Release
G8  四配置回归、统计、证据和最终发布
```

阶段关闭只代表实现和预登记验收完成，不代表所有研究方法获得正收益。

## 12. VLA 研发与运行路径

### 12.1 跨阶段研发路径

```text
G3-01  Observation/Data/Scenario identity 与冻结 split
  ↓
G3-02  V0 K1 / V1 K2 公平基线
  ↓
G3-03  F0 + VLA-V0 checkpoint + canonicalizer + K1/T10闭环
  ↓
G3-04  VLA-V1低维history+K2；V2 FAST/REASON递进optional
  ↓
G3-05  VLA+Safety无Classic当前帧候选闭环 + Hybrid
  ↓
G4A oracle best-of-K门禁；通过时可进入G5 World-V0
  ↓
G6 一轮Shadow修正采集 + 单个监督adapter/LoRA + 抗遗忘
  ↓
G8 四发布配置回归、证据和准出
```

### 12.2 代码路径

```text
safedrive_foundry/driving_vla/
├── schema/          VLA 输入、输出和训练标签 schema
├── data/            数据 adapter 与 dataset，不保存 Registry 真值副本
├── baselines/       Route/Ego、单轨迹、非语言多候选
├── model/           V0 anchor、V1 K2、V2 FAST/REASON
├── training/        heads-only/少量LoRA、checkpoint、恢复
├── runtime/         deterministic Guard / DEFER / deadline
├── evaluation/      open-loop、grounding 和 slice 评价
├── adapter/         Runtime/ROS/Safety 的稳定边界
└── posttrain/       G6 单轮监督 adapter/LoRA
```

数据、模型、adapter 和 post-training 不得互相复制 schema；正式 schema 只能由版本化接口包定义。G3-03 之前不得实现 G6 posttrain，G6 不得改变 G3 冻结的 Regression split。

## 13. World Model 研发与运行路径

### 13.1 跨阶段研发路径

```text
G3-01  冻结 scene/frame/candidate/actor identity
  ↓
G4A  top-1 vs oracle best-of-K 入口门禁
  ↓
ENTER_WORLD 后：G5-01 K2 action branch + CV/CTRV/Reward 基线
  ↓
G5-02  K2/T10/N8/M1、4M～8M action-conditioned dynamics
  ↓
G5-03  actor future、collision/TTC/off-road、action/no-action
  ↓
G5-04  两候选软排序 + World异常退化VLA+Safety
  ↓
G5-05  VLA+World+Safety 闭环净收益/负结果验收
  ↓
G8 统一回归、消融、模型卡和发布开关
```

### 13.2 代码路径

```text
safedrive_foundry/world_model/
├── schema/              ActionBranch、WorldRolloutBatch 与错误状态
├── data/                可比分支、shard 和 dataset adapter
├── baselines/           persistence、CV/CTRV、IDM、Reward MLP
├── model/               object/vector/BEV latent dynamics
├── training/            rollout training、checkpoint、资源恢复
├── heads/               actor、occupancy、risk、reward、termination
├── calibration/         uncertainty、ECE/Brier、OOD
├── evaluation/          action/actor intervention、ranking/regret
├── runtime/             batch scoring、feature cache、availability
├── verification_queue/  异步 CARLA 请求与预算
└── cache/               带版本和 provenance 的短期缓存
```

World Model 不拥有 Safety、控制或场景 tick 权限。VLA/World 共享的只能是只读 feature cache 和稳定身份；模型权重、availability、配置、数据版本和消融开关独立。

## 14. Active CARLA 的准确语义

Active CARLA 不是当前帧在线规划器。当前帧只使用 Learned World 的已完成评分；高风险、高不确定、OOD 或候选分歧被异步写入 verification queue，随后从可比起点执行 CARLA 分支，用于：

1. 测量 World ranking error 和校准误差；
2. 生成经过门禁的 World/VLA 后训练数据；
3. 增加 Regression/Minimal Counterexample；
4. 决定未来模型版本是否准出。

异步 CARLA 结果不得回写已经执行的动作，也不得与在线场景争夺唯一 tick master。在线 World 超时直接退化到 VLA+Safety，不等待 CARLA。
