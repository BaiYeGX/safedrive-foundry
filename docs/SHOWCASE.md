# CORA-Drive 结题展示与面试合同

当前可展示的是 H1–H6 已冻结实现、Evidence 和从负结果得到的 CORA 设计；CORA paired data、
checkpoint、calibrated router 与四臂 formal 仍为 `PLANNED`。本文同时包含“现在可讲版本”和
“C6 formal 后版本”，演示时必须按实际状态选择，不能提前使用未来时态成果。

## 1. 面试官应记住的一句话

> VLA 提出具有语义泛化能力的 nominal 轨迹，Classic Expert 提出具有几何/规则先验的
> 轨迹；反事实 World 预测每条候选的后果并在证据不足时拒绝决策；独立 Safety Kernel
> 始终保留最终执行权。

英文：

> VLA proposes; the World model predicts consequences; calibrated routing decides whether it
> knows enough; the safety kernel retains final authority.

## 2. 主干—枝干结构

```text
主干：Observable → VLA/Expert → Guard → Counterfactual World
                 → calibrated choose/defer → Safety → MPC/PID → CARLA

枝干：
VLA          → vision-language-action、预训练模型集成、path/speed、alignment
Classic      → Frenet/ST、几何/规则先验、强 baseline
Data         → potential outcomes、exact reset、counterfactual intervention
World        → cross-attention、multi-head outcome、pair difference
Uncertainty  → ensemble、NLL/Brier/ECE、conformal、risk-coverage
Safety       → Guard、repair、MRM、fail closed、authority separation
Control      → executable identity、MPC/PID、freshness、20Hz deadline
Systems      → CARLA synchronous runtime、ROS 2 bridge、single tick owner
Evidence     → split、run-lock、hash、bootstrap CI、负结果与停止条件
```

所有枝干必须回答“为什么、怎么做、效果、边界”，不能只罗列名词。

## 3. 五分钟主讲

### 0:00–0:40 问题

VLA 语义和长尾先验强，但轨迹几何、安全与 confidence 不稳定；Classic 稳定可解释，但长尾
语义弱。项目研究如何在不把安全权交给基础模型的前提下利用二者互补。

### 0:40–1:30 架构

说明两个候选独立、Guard 在 World 前、World 对 source 元数据不可见且可 defer、Safety 最终重验、
controller 只跟踪绑定 executable。强调 World 不能生成第三轨迹或绕过安全。

### 1:30–2:20 旧结果与失败

```text
H4 locked: World 64/64 vs simple 57/64
H5 closed loop: 222 runs / 74 paired roots, unsafe 4 vs 6,
                progress lower-95 = -0.0709 m, gate failed
H6 pilot: strict VLA preference 21.83%, applied VLA 47.50%,
          pair score 590/600, provenance gate failed
```

结论不是“模型完全没用”，而是离线准确率没有转化为可复现闭环净收益。

### 2:20–3:30 根因与 CORA

- logged tick 只有执行候选 outcome；
- 旧 H6 逐 tick 双 outcome 为 0；
- episode 第一 tick 监督 whole outcome；
- source/episode shortcut 和 optimistic bias；
- 未校准二选一、offline/live temporal mismatch。

CORA 用同锚点双分支 potential outcomes、swap-equivariant outcome model 和 calibrated
abstention 修复。

### 3:30–4:30 当前计划与证伪方式

现在应说“已经预注册将比较”Classic、factual World、CORA no-abstention、full CORA 四臂，
而不是说已经获得 CORA 结果。说明将报告 pairwise/selective regret、risk-coverage、安全/
进度 CI、切换、延迟和显存，并且 formal 无论正负都冻结。C6 完成后才把这一段换成实际图表。

### 4:30–5:00 技术判断

强调生成式视频 World、BEV/latent dynamics 和 outcome World 的区别；解释为什么在单张
4080、实时 selector 和安全组合目标下选择结构化 outcome，而不是缩小版 GAIA。

## 4. 深挖问题准备

| 面试追问 | 必须能讲清 |
|---|---|
| VLA 是否是你训练的？ | 预训练 SimLingo 的真实集成、适配、一次 forward、raw path/speed、运动学平滑；不是从头训练 |
| 为什么保留 Classic？ | 不同 inductive bias、强 baseline、fallback、可解释性；不是故意弱化对照 |
| World 为什么叫 World？ | action/candidate-conditioned future outcome；不是像素生成，明确能力边界 |
| 你预测的 counterfactual 到底是什么？ | Guard eligible proposal 经过冻结 Safety/repair/controller 后的 CARLA interventional outcome，不是实车自然因果真值 |
| 为什么离线 64/64 仍闭环 gate failed？ | 小 locked set、label mismatch、covariate shift、calibration/temporal drift |
| 反事实如何获得？ | 同 anchor exact reset、两个 forced branch、相同初态/外生脚本/Safety/controller；交互导致的 future 可以不同 |
| 如何避免 source shortcut？ | metadata 从 feature schema 物理排除；再用 shared weights、source/candidate swap、trajectory-to-source 与干预 probe 审计风格捷径 |
| uncertainty 怎么做？ | aleatoric distribution head、epistemic ensemble、独立 calibration、risk-coverage |
| conformal 是否保证安全？ | 只在统计假设下保证 calibration coverage；独立 Safety 才是在线硬边界 |
| defer 后谁开车？ | held source 的 fresh candidate；不可用时按 Expert→VLA 的非学习顺序，再经 Safety/MRM；不是 Oracle 或无控制 |
| World 选错怎么办？ | Guard 前置、Safety 重验、repair/fallback/MRM、executable identity fail closed |
| 实时性如何保证？ | 轻量 vector model、P99/deadline、同卡 workload profile、no hidden demo override |
| ROS 2 做到什么程度？ | status/tick synchronization bridge；主算法仍主要 Python runtime，不夸大全栈 |
| 如何保证实验可信？ | frozen split/seed/gate、run-lock、hash、失败保留、pilot→formal、bootstrap CI |

## 5. C6 计划中的 Live demo 合同

以下场景只有在对应 CORA checkpoint、calibration 和 frozen config 真实存在后才能标为
`live CORA`。在此之前，可以用现有 H1/H5/H6 Evidence replay 解释候选、Guard、Safety 和负
结果，但必须在画面标注 `historical replay / CORA not yet implemented`。

### 固定场景

1. `semantic-win`：在冻结 Evidence 中 VLA proposal 后果优于 Expert，不预先指定必须出现；
2. `rule-risk`：红灯/障碍场景中 Expert 更优，或 VLA 被 Guard/World/Safety 拒绝；
3. `ambiguous-cut-in`：World bounds 重叠并 defer，非学习 fallback 与 Safety 保持稳定执行。

场景、map、seed、weather、route 和 checkpoint 在录制前冻结。live 失败时可以播放同一
配置的预录后备，但必须标注 replay/live 状态。

### 必须显示

```text
frame / sim time / map / seed
VLA and Expert polylines
candidate ids and short hashes
Guard PASS/REVIEW/REJECT
per-candidate progress/risk/comfort/uncertainty
utility lower/upper bounds
raw selection / stabilized selection / defer reason
Safety accept/repair/fallback/MRM
final executable and applied source
controller mode
per-stage latency / deadline
collision / red-light / route progress
checkpoint / config / code hash
```

### 禁止

- Safety/control 后强制 throttle 或 steer；
- `--map` 仅记录不真正加载；
- 随机/未训练 checkpoint 冒充模型质量；
- 静默吞掉相机、tick、cleanup、identity 错误；
- 普通 demo JSON 命名为正式 Evidence；
- 只展示车在动而不展示候选、World、defer 和 Safety。

## 6. 最终图表

结题至少提供：

1. 系统主干图；
2. exact-reset potential-outcome 数据图；
3. factual vs counterfactual label coverage；
4. per-head calibration/ECE/Brier；
5. pairwise/selective regret；
6. risk-coverage/defer 曲线；
7. 四臂 paired progress bootstrap CI；
8. unsafe、switch、ping-pong、latency/VRAM 表；
9. action/context intervention 单调性；
10. 失败场景时间线。

图中明确区分 development、calibration、pilot、formal，不能把不同 split 混在一个最优值中。

## 7. 简历边界

### 当前已可使用

- 基于 CARLA 0.9.16 同步 runtime 集成预训练 SimLingo VLA 和独立 Frenet/ST Expert，并提供
  ROS 2 clock/status bridge；实现逐候选 Guard、fail-closed Safety 与 executable-bound MPC/PID；
- 构建 metadata-source-blind candidate-conditioned outcome scorer，在小型 locked decisive set 上
  达到 64/64、simple baseline 57/64；
- 完成 222 次 run / 74 个 paired roots 的 exact-reset World ON/OFF closed-loop 评测并报告
  bootstrap CI、tail latency、
  switching 和完整负结果；
- 2026-08-27 离线基线为 436 tests：435 passed、1 个真实 GPU/checkpoint live-forward 项 skipped；
  该数字不是当前覆盖率，也不等于实时 CARLA 全覆盖。

### 只有 CORA formal 通过后可使用

- 提出并验证 exact-reset counterfactual potential-outcome learning；
- 显著降低 pairwise/selective regret；
- 在安全不劣条件下取得闭环 progress 净收益；
- calibrated abstention 改善 risk-coverage/switching。

如果 CORA formal 为负，仍可写成“设计并完成可证伪的反事实/拒绝消融，定位未转化为闭环
收益的瓶颈”，但不能继续使用本节四条正向效果描述。

### 永远不能夸大

- 从头训练了 VLA foundation model；
- 构建了 GAIA 类生成式视频 World；
- 已经证明实车或量产安全；
- 完成全 ROS 2 自动驾驶栈；
- 单元测试、开发强制采样或随机 latency 等于 formal 结果。

## 8. 最终结题包

```text
README and architecture
frozen task/config/matrix
data-quality and model cards
locked Evidence summaries and hashes
ablation report
three-scenario live/replay demo
three-minute video
five-slide interview deck
resume bullets
limitations and negative-result page
environment/license/attribution
```

C6 完成后把最终状态写入 `PROGRESS.md` 并停止，不再自动开展生成式 World、VLA fine-tune、
RL 或第三候选研究。
