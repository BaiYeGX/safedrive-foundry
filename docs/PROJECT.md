# SafeDrive Foundry 项目定义

## 1. 项目定位

SafeDrive Foundry 是 CARLA–ROS 2 纯软件在环研究项目，研究如何在可审计安全边界内组合
预训练 VLA 与 Classic Expert。项目不试图用单个基础模型替代所有规划、安全和控制模块，
而是把两种不同 inductive bias 的 planner 作为独立候选来源：

- SimLingo VLA 提供视觉语义、语言—动作对齐和长尾先验；
- Classic Frenet/ST Expert 提供几何、规则、动力学和确定性先验；
- learned World 只预测候选后果、排序或 defer；
- Guard、Safety Kernel 和 MPC/PID 保留硬约束与执行权。

当前最终主线为：

> **CORA-Drive：Counterfactual Outcome Routing with Abstention for Hybrid
> VLA–Expert Driving。**

## 2. 术语与状态

| 术语 | 固定含义 |
|---|---|
| proposal/candidate | generator 在当前 observable anchor 独立提出的原始轨迹 |
| Guard eligible | `PASS` 或 `REVIEW`；`REJECT` 对 World 不可见 |
| selected | router/fallback 请求 Safety 检查的当前候选 |
| executable | Safety 接受或有 provenance 地 repair 后批准的轨迹 |
| applied | controller 实际跟踪的 executable identity |
| hold | 保持 source 选择，不复用上一 tick 的过期轨迹 |
| defer | 把 learned 排序权交给冻结非学习 fallback，仍输出受 Safety 审查的动作 |
| source-blind | World feature schema 无 source/slot/order 元数据，不等于轨迹风格不可被推断 |

项目进度与算法 Evidence 是两个维度：C0 已完成、C1 当前进行，表示结题 program 已启动；
新的 CORA 数据、模型和闭环 Evidence 仍是 `PLANNED`。

## 3. 研究历史与当前问题

H3–H5 的问题是：

> 在候选、Safety、控制器和场景相同的条件下，旧 World selector 是否比冻结非学习 selector
> 带来可复现的闭环净收益？

H5 正式 Evidence 回答：未达到冻结门。H6 v1/v2 又尝试让 World/VLA 成为主驾，但 seed
101 pilot 的逐 tick preference 和实际 VLA 使用仍失败。历史结果保持不变。

代码/数据审计确认旧路线的主要因果缺口：一个 live tick 只执行一个候选，因此未执行候选
的 outcome 缺失；旧 H6 tick rows 没有双 outcome，episode 第一 tick 又被用于监督整段结果。
模型可能学习 source/episode shortcut，而不是：

\[
p_\theta(Y\mid O_t,\tau_i)
\]

当前结题问题改为：

> 在同一 observable anchor 上分别执行 VLA/Expert 候选，获得两个真实 potential outcomes；
> 使用 metadata-source-blind、candidate-swap-equivariant World 预测结构化后果；通过独立校准在证据
> 不足时 defer。该方法能否降低 selective regret，并在 Safety 不变时取得安全不劣的闭环
> 效用？

更准确的估计对象是冻结下游栈下的 proposal-level intervention：

\[
Y_i = Y\!\left(do(\text{proposal}=\tau_i);\pi_{Guard},\pi_{Safety},\pi_{control}\right)
\]

因此 label 包含 Safety 接受、bounded repair 或 MRM、controller 执行及其后果，但 branch
禁止跨候选 fallback，另存 `would_require_fallback`。模型输入仍是 Guard eligible 的原
proposal；它预测“把这条 proposal 交给冻结下游系统”的后果，而不是
声称识别实车世界中不可观测的自然因果效应。这个结论只适用于冻结 CARLA 场景、reset 和
下游 policy 版本。

## 4. 系统主干

```text
CARLA/ROS observable snapshot
        ├── pretrained SimLingo VLA ── candidate_vla ───── Guard ──┐
        └── Classic Frenet/ST Expert ─ candidate_expert ── Guard ──┤
                                                                    ▼
                                      Counterfactual Outcome World
                                                                    ▼
                                  calibrated choose / hold / defer
                                                                    ▼
                                independent Safety → MPC/PID → CARLA
```

World 的结果重新通过 offline-only exact-reset branch collection 形成下一版开发标签，但
Oracle、actor future、outcome 和 formal answer 永远不进入在线输入。

两候选均 eligible 但 World 证据不足时，router 输出 `DEFER_AMBIGUOUS`：若当前 held source
仍 eligible，则使用它在当前 tick 的 fresh candidate；否则按 Expert→VLA 的冻结顺序请求
Safety，均失败才 MRM。该顺序必须预注册和记录，不能通过 defer 暗中实现固定 source quota。

## 5. 核心贡献

### 5.1 异构、独立的双候选

VLA 与 Expert 同锚点、同坐标、同 horizon，各自独立生成一条轨迹。学习模块不能复制、
扰动或重命名其中一条来伪造在线第二候选。两种来源的价值来自互补先验，而不是人为保证
VLA 胜率。

### 5.2 反事实 potential-outcome 数据

对同一个 anchor：

\[
(O_t,\tau_V,Y_V,\tau_E,Y_E)
\]

CARLA 场景被精确重建，两条候选各执行一次 short horizon；reset、actor/light script、
candidate hash、Safety binding 和控制必须可比。只观察到一个 outcome 的 tick 不进入
pairwise loss。

### 5.3 source-blind outcome model

共享候选编码器预测：

```text
progress distribution
completion probability
collision / red-light / offroad probability
comfort distribution
feasibility / repairability
epistemic disagreement
pairwise utility difference
```

source、slot、branch order、Guard verdict、Oracle、rollout future 和真实 outcome 不进入
World feature view。candidate swap 后 absolute outcomes 必须跟随轨迹交换，pair difference
必须反号。

这里的 source-blindness 是 schema 和干预不变性主张。VLA/Expert 的轨迹曲率、速度或停止
风格可能让来源从合法 trajectory feature 中被统计推断；这不是靠删除字段就能消除的。项目
必须另外报告 trajectory-to-source probe、按 source winner 分层的 regret，以及在 source
metadata swap、候选交换和物理 action intervention 下的稳定性。

### 5.4 可校准拒绝

World 不直接把 softmax 当作安全置信度。独立 calibration split 产生 outcome/utility bounds；
只有候选效用区间明确分离时才 choose/switch，否则 hold 或 defer。统计 coverage 不等于全域
安全证明，最终硬权限仍由 Safety Kernel 保留。

### 5.5 可证伪闭环 Evidence

最终不是以离线 accuracy 或 VLA 使用率结题，而是用冻结多臂 exact-reset closed loop 对比：

```text
Classic selection baseline（dual-generator shadow load）
factual H6 World
CORA without abstention
full CORA with calibrated abstention
```

报告安全、进度 CI、regret、risk-coverage、defer、切换、deadline、显存和 provenance。
formal 无论正负都冻结并停止。

A–D 主对照固定运行两个 generator、同一 Guard/Safety/controller 和相同 workload；A 只是
把选择锁为 Expert，以隔离 selector 效应。真正关闭 VLA 的 Classic profile 只用于部署成本
比较，不能与 A–D 的效果差异混算。

## 6. 固定在线边界

- 两个 generator 只读同一决策时刻的可观测输入。
- candidate 统一为 map frame、`T=10`、`dt=0.25s`、`horizon=2.5s`。
- Guard 在 World 前逐候选运行；`REJECT` 不可被 World 复活。
- World 可预测、排序、hold 或 defer；不能生成轨迹、改写 Guard/Safety 或直接控车。
- Safety 对最终轨迹重新验证，可 repair、fallback 或 MRM。
- controller 只跟踪由 Safety 批准且 executable ID 完整绑定的轨迹。
- 每次 run 只有一个登记 tick owner 推进 CARLA；正式 collector 用 `ScenarioRuntime`，ROS
  `carla_sync_driver` 只作互斥 bring-up 模式；业务/清理脚本不得直接创建第二 tick master。
- 仅限 CARLA SIL，不声明实车、公共道路、生产或 ISO 26262 认证安全性。

## 7. 数据隔离

| 数据 | Generator | World 在线输入 | World 离线标签 | Oracle/审计 |
|---|---:|---:|---:|---:|
| 当前图像、ego、route、history | 是 | 是 | — | 是 |
| 当前可观测 actor/light/lane | 是 | 是 | — | 是 |
| candidate trajectory | 不回灌 | 是 | — | 是 |
| source/slot/order/provenance | 仅自身 provenance | 否 | 仅绑定 | 是 |
| Guard/Safety verdict | 否 | 否 | 是（feasibility/repair/MRM label） | 是 |
| branch actor future/outcome | 否 | 否 | 是 | 是 |
| formal label/seed answer | 否 | 否 | 否 | 是 |
| Regression/故障注入答案 | 否 | 否 | 否 | 是 |

train、validation、calibration、pilot 和 formal 必须按 root lineage/map/family/seed 隔离。
同一 anchor、近重复轨迹或其 intervention 不能跨 split。

## 8. 成功口径

### 数据

- pairwise 样本两条候选 outcome 都真实有效；
- reset/signature/provenance 全通过；
- slot/source/branch permutation 不改变标签；
- formal 数据不进入训练、选择 checkpoint 或 calibration。

### 模型

- 优于冻结 simple/candidate-only/factual baselines；
- metadata-only source probe 确认 schema 不含 source；trajectory-to-source probe 作为 shortcut
  风险诊断报告，不把物理可预测性伪装成必须随机；
- candidate swap、action/context intervention 和 outcome consistency 通过；
- NLL/Brier/ECE、unsafe recall、pairwise regret 和 worst-group 完整报告；
- 三 seed 方向一致，不只挑最佳 checkpoint。

### Router

- offline/live trace parity；
- risk-coverage、selective regret 和 defer reason 可审计；
- source-stable EMA/hold，不发生 frame-ID 重置；
- P99、deadline miss、显存和 switching 达到预注册门。

### Closed loop

- full CORA 满足 C5 预注册的 paired unsafe non-inferiority；同时报告原始事件数、CORA-only
  unsafe 和置信区间；
- paired progress bootstrap lower-95 `>=0`；
- selective regret 优于 factual/no-abstention；
- reset、candidate、selected、final、executed、applied 身份完整；
- formal 结果冻结，不根据结果改 map/family/seed/gate。

具体数值阈值必须在相应阶段 `START_TASK.md` 中先冻结；本文不给尚未测量的结果升级状态。

## 9. 非目标

- 不从头训练 GAIA/Drive-WM 类像素视频生成模型；
- 不让 World 伪造在线第三候选或学习扰动第二候选；
- 不把固定 VLA 使用率当作模型质量目标；
- 不用语言 CoT 代替 action/outcome 干预验证；
- 不在 CORA 完成前启动 VLA LoRA、RL、全 ROS 2 重构或新 planner；
- 不用 archive、开发强制采样、随机模型 benchmark 或单元测试数字冒充正式 Evidence。

## 10. 证据状态

只允许：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

失败和负收益与正结果同等保留。代码存在不等于测量，单元测试通过不等于 CARLA 正式通过，
统计 coverage 不等于功能安全保证。
