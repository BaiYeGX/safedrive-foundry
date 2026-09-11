# 本周 World–VLA 模型合同

当前执行入口：[C4 执行单](C4_EXECUTION_PLAN.md)。该文件负责日期、实现顺序、固定配方和
交付节点；本文保留模型与梯度公式。当前任务为一次真实 M2 联合训练及 C5 接入准备。

## 唯一模型与研究问题

状态：C3 按学校项目的离线基线范围收尾，保留当前 M1。C4 M2 后果辅助联合微调为
`PLANNED / NOT_RUN`。基线 run 为 `generated/h6/cora/c3-repair-20260910T165902Z/`，
阶段记录见 [EVIDENCE](EVIDENCE.md)。M1 与 M2 都必须从原始
M0 同一起点训练，M2 禁止从 M1 续训，使用同 SFT 数据、seed、步数、LoRA 范围和预热/学习率日程。
本周只有这两份新训练模型，不做执行中介、ensemble、视频 World、RL 或多配置搜索。

项目截止为周日 2026-09-13，且必须现场 CARLA 闭环。C4 集中在周六 09-12 完成一次
真实 M2 训练/重载及接入；周五先验证 CARLA/M1 链路，同时完成不占 GPU 的 C4 准备。
仅保留直接影响正确性的联合 batch、候选绑定、梯度和评估检查；不追加多 seed 或大规模
消融。若检查未通过或预算不足，保留具体缺项，不能以截止时间替代有效训练。

2026-09-11 起按 [PROJECT](PROJECT.md) 的 school_delivery_v2 优先学校项目交付。C4 正式
M2 前锁定 P-WP 为策略主指标，P-ADE/有效率/失败率同表，四头后果分别与 b 比较；旧 v1
主次留档。本次改变未来实验主次，不修改 C3 已冻结结果。不要求每头或每个 root 都胜出。
当前 C3 的成本 smoke 仅证明部分反向通路，不能替代本节的真实 b+R、配对损失、候选绑定
和联合更新检查；将这些检查与资源核对放在 C4 正式运行前，复用已有独立评估，其他非关键扩展后置。

当前改进是：**保留简单候选基线，学习它的后果残差；限制辅助梯度干扰驾驶主任务。**
这是降低退化风险的待验证组合，既有残差和梯度门控并非新发明，不保证正收益。

## 表示与残差后果

```text
h = E_phi(current image, navigation, ego/history)
route_hat, speed_hat = P(h)
b = frozen_candidate_ridge(canonical proposal)
residual = R(h, canonical proposal)
y_hat = b + train_target_scale * residual
y = [progress, acceleration_rms, jerk_rms, lateral_acceleration_rms]
```

b 只使用允许的 canonical candidate 编码，不能读 source/slot/order/provenance 或未来。
优先复用 C2 candidate-only ridge 算法与固定正则设置，在本次允许的同一 train 子集重新
拟合并保存新系数；不改 C2 release，不扫描旧 Hash。b 不随 M2 更新，拟合耗时入总账。
不得拿 validation 选择 b 的正则或逐头挑赢家。继续报告 train-mean/context-only 等参照，
candidate ridge 并非每个舒适性头都已证明最强。

R 使用 concat(h, flatten(candidate)) → 128/64 ReLU MLP → 四维线性输出。
最后输出层权重/bias 零初始化；因此初始 y_hat 等于 b。隐藏层正常初始化，
不能全层置零导致不能学习。零初始化只保证初始预测一致，不保证训练后保持基线水平。
预热后 R 最后一层已更新，再检查 World-only LoRA 梯度；第一个零初始化 step 的共享
梯度为零是预期现象，不能被误判为永久 detach。

h 在候选特有输入前提取，显式 observation token mask 排除 padding、答案和 future。
两候选共享同一次 VLA 的 h 与同一个 R。禁止用第二个冻结 VLA 代替共享梯度。
不缓存随 LoRA 改变的 h；只允许缓存不变图像处理或冻结视觉特征，并绑定版本。

## 数据、loss 与更新

四头均为原有真实短时分支标签，不新增序列或风险概率。其语义是冻结 Safety/controller
接受 proposal 后的后果；SFT teacher 与 branch outcome 分离，不复制另一个候选的后果。
canonical T=10、dt=0.25 s 不变。有效 mask、单位、early terminal 规则沿用数据合同。

M1/M2 用同一 SFT root 顺序与采样次数；World 缺标签只关对应辅助 mask。
本周不额外采样无 SFT 的 World-only roots；b 也使用相同允许 train 子集，记录实际有效数。
训练集拟合 target mean/std，std 下限 1e-3；常量头显式报告，不把低误差当能力。

```text
L_sft = 与 C3 完全相同的 route/speed SmoothL1
L_aux = 0.1 * L_outcome + 0.05 * L_pair
L_outcome = standardized(y_hat, y) 的有效候选/四头 Huber 均值
L_pair = standardized((y_hat_A-y_hat_B), (y_A-y_B)) 的有效双分支四头 Huber 均值
```

预测差由同一个共享模型相减保证反对称。b 贡献也必须相减，不能只监督残差差分却忽略
真实标签中的基线差。物理单位用于最终报告，归一化只用于训练。

前 max(1,ceil(0.05*T)) optimizer steps，M1/M2 都冻结 LoRA、更新驾驶头；
M2 同时拟合新 R。随后共同解冻 LoRA，M2 在一个梯度累积窗口上分别计算：

```text
g_s = grad(L_sft, shared_LoRA)
g_a = grad(L_aux, shared_LoRA)
w = max(0, cosine(g_s, g_a))
q = min(1, 0.25 * norm(g_s) / (norm(g_a) + eps))
shared_LoRA.grad = g_s + w * q * g_a
```

零范数时辅助项置零。梯度在 unscale 后以 FP32 统计，在整个 accumulation window 聚合后
门控；不能每个 microbatch independently gate 后声称公式等价。
驾驶头只接 SFT 梯度，World head 正常接 L_aux 梯度。合成后，LoRA+驾驶头作为驾驶参数组
共同 clip=1.0（M1/M2 相同），World head 另行 clip=1.0，再做 optimizer.step；
不能将 World head 与驾驶参数一起计算全局裁剪范数。两组参数不得重复或遗漏。
记录余弦、w/q、辅助激活比例、两类梯度范数与参数变化。门控不能把共享辅助永久关闭后仍
宣称完成联合表示学习；若实际全部关闭，报告联合机制未发生，不能硬开门制造证据。

这是训练梯度层面的干扰控制。AdamW 动量/预条件、非线性和数据外推下没有保证，
不能称为泛化提升证明或保证每步主 loss 不增。门控与残差是一个组合配方；
本周没有单独训练完整消融，不能据结果拆分两者各自贡献。
M1 已完成 200 次更新；M2 的实际 backward/门控成本在本轮真实联合 smoke 测量后用于
检查 T=200 的预算可行性。共同 SFT 日程沿用 C3，不减少已完成的 M1 更新或混称不同预算。

## C4 可直接设置的 goal

> 完成 C4：按 docs/C4_EXECUTION_PLAN.md 和 docs/WORLD_MODEL.md，读取 C3 的原始起点、数据和共同预算，完成
> 固定基线残差 World 与驾驶梯度门控的 M2 联合微调；验证真实辅助梯度、候选交换、
> 无泄漏与重载，交付预登记指标、完整逐 root 对照和资源记录。达到验收后更新
> PROGRESS、将入口指向 C5；现场链路前置检查复用 C5 预算，正式三臂对照在 C5 执行。

## C4 执行与验收

前提为 C3 有效 M1、原始 checkpoint、manifest、共同日程与真实命令；缺项报告，
不隐式重跑 C3。允许改 h 暴露、b/R、联合 loader/梯度更新、配置与直接测试。
先在 train smoke 验证 b 的 Torch/离线实现等价、零残差等于 b、非零残差可学习、
预热后辅助能到 LoRA、无效 mask、候选交换与 checkpoint round-trip。
实现错误在正式训练前定位并修复，只重跑直接受影响的检查；普通报错不触发整轮重训。

正式仅一份配方、一个 seed、共同 T；采用最后有效 checkpoint，保存恢复状态。
M0/M1/M2 用同一完整冻结 validation 清单与 evaluator。保留全部分母、缺失与失败；
World 按各头共同有效的支持比较，新 b/ridge 指标重新计算，不混用旧聚合数字。
主要研究指标以 PROJECT 为准；额外报告四头 MAE、b 与 b+R、残差大小、
no-action/context masking、swap 和梯度门控分布。这些诊断不增加新训练。

C2 validation 中 ridge 进度 regret=0，ranking 52/52 的口径继续保留，
不是“必须进一步提升”的可达目标。绝对误差仍有空间，但也不能改指标掩盖旧负结果。
b+R 如未胜 b，则 learned residual 收益未证实；胜旧 MLP 不等于胜强基线。

必须达到共同 T、真实共享辅助更新、重载和有效开发评估；预算不足标 PARTIAL，
不同更新数标不等预算，不能宣称公平增益。完整 M2 优化最多 4 小时；
所有训练含预热/smoke/失败合计仍最多 10 小时，不另启第三模型或参考网络。

新 run-id 保存 M2 adapter/heads、b 系数、normalization、feature schema、config、
manifest 引用、预测、梯度/消融诊断与 stage-summary.json。输出声明 PLANNED/
IMPLEMENTED/MEASURED/VERIFIED 按实际证据升级，不按结果正负升级。

## C5 在线使用与归因

加载同版本的 M2、b、R、normalization 与 h 提取函数；复用一次 VLA forward。
World 只看 Guard eligible 候选，输出连续后果排序或 defer，无校准风险概率。
Guard/Safety/控制权限不变。异常、无支持或优势不足时沿用 held-source/Expert→VLA defer，
HOLD 仍用 fresh candidate，0/1/2 eligible 与时间状态机遵守候选合同。

A/B 保留同结构 b+R shadow、同次数调用，不用其分数控制；M1 shadow 不是有效质量证据。
C 使用真实 b+R。额外从同一 tick 保存 b-only 的排序与 learned residual 是否改变排序，
只作决策诊断，不能把未执行选择的后果伪造成真实闭环反事实。

C/B 有增益可能来自 b 或非学习回退，而非 R；没有独立闭环 b-only 臂就不宣称残差的
独立闭环贡献。已有配对开发数据可比较 b 与 b+R 的离线作用，局限必须同时报告。
本周不增加第四臂、风险校准、线上模型选择或训练。

## 历史基线（冻结）

C2 release h6-cora-c2-devbaseline-20260907-v1：train 158 / validation 53，
499-D context + 10x8 candidate，共享 128/64 MLP，seed 17/29/43。
candidate ridge progress MAE 约 0.302 m，MLP 约 0.555/0.617/0.555 m，
开发结论 NO_DEMONSTRATED_GAIN。这些不是 M2 的结果，完整记录见 EVIDENCE。

## C4 量化交付

执行 [PROJECT 当前实验口径](PROJECT.md) 与 [C4 执行单](C4_EXECUTION_PLAN.md)。M2/M1 的
P-WP 是当前主要方法指标，连同 P-ADE/P-FDE/有效率/失败率和训练成本完整比较。
b+R/b 必交四头 W-MAE、W-RMSE、W-NMAE、W-PAIR、进度 W-RANK/W-REGRET 及全部有效分母。
新 b 与 normalizer 按每头有效 train label 拟合；不得用缺失值填零训练。
原始数据、Guard eligible 子集及旧 C2 聚合口径分别标明，不跨版本相减。
记录辅助梯度参与比例、实际 LoRA 更新、b/R 输出尺度；门控参与不是预测或驾驶收益。
所有新指标用同一逐 root 明细生成绝对/相对改善和 CI；不因 World 辅助指标获胜改写主结果。

## 联合学习的额外干扰检查

独立裁剪用于消除新增 World head 对驾驶梯度缩放的隐式耦合：即使辅助项被门控为零，
World head 的大梯度也不应单凭全局 norm 把驾驶更新缩小。该修订不保证 AdamW 更新或泛化占优。
烟雾验证中人为放大 World-head 梯度，应只影响该组裁剪系数，不能改变合成前的驾驶梯度；
正式记录两组 clip 前后 norm、系数、辅助贡献及实际更新。此检查在现有 smoke 内完成。

g_s 是 route+speed 的总 SFT 梯度，与 g_a 相容不保证单独 route ADE 改善；
在已有日志同时记录 route/speed loss，结果同时报告 P-ADE/P-WP，不能将总 loss 下降替代主指标。
不因此新增第三套每头梯度门控或额外反向搜索，仍为原两类共享梯度、相同共同 T。
每个 root 的有效候选/头先平均，再按实际 accumulation window 等权；pair 项只对双方有效的头
求均值。无 pair 的 root 辅助 pair 项为零并计缺失，不能跳过它改变 SFT 暴露或优化步数。

h 明确由当前观察的有效 token 做 masked mean，排除 padding、teacher/future 和候选特有 token；
沿用可微张量，不使用 runtime 已转 numpy 的输出。保存 token mask 与展开顺序的版本，
两候选复用同一个 h、同一次前向。R 不增加 dropout，避免额外随机流干扰 M1/M2 对照。
本周不新增 full VLM teacher、LoRA 重初始化、视频生成或 RL；优先修正原生监督与优化耦合。
