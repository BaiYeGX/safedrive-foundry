# C3 — 数据接好，完成 VLA 微调

状态 `VERIFIED / ENGINEERING_COMPLETED / ALGORITHM_MEASURED`（2026-09-09）。本文件是完整
C3 执行合同；权威实测记录位于 `generated/h6/cora/c3-vla-sft-20260909-final-v3/`，普通检查
均在同一阶段内完成。目标已交付 M0 离线基线、M1 常规微调，范围与预算服从 ROADMAP/RESOURCES。
本文件只保留项目实施决定，不要求执行时重新检索论文。

## 可直接设置的 goal

> 完成 C3：按 docs/VLA_FINETUNE_WEEK_PLAN.md，在当前项目内完成真实数据适配、
> 原生权重加载与梯度检查、M0 离线基线和 M1 常规 LoRA 微调，保存可重载权重、
> 逐 root 评估和实际运行记录。遵守既定资源与数据边界，验收后更新 PROGRESS、
> 将下一入口指向 C4 并停止，不自动执行 C4。

## 开始所需与改动范围

读取 START_TASK、PROGRESS、本文及数据/模型/候选/资源/环境/Evidence 合同；
检查当前分支和已有改动。C2 dev release 已交付即可开始，不要求旧稀有事件门转为通过。
允许改必要的 loader、VLA 训练适配、配置、专项测试和文档；不改冻结数据、阈值或模型。

现有集成入口是 safedrive_foundry/driving_vla/model/nominal_policy.py，
canonicalizer 位于同目录；上游 simlingo-main/simlingo_training 仅作为适配参考。
scripts/h6_cora_devbaseline.py 是旧 C2 工具，不能重跑其 prepare/finalize 来覆盖冻结 release。
新 SFT 命令已实现为 [`scripts/h6_cora_sft.py`](../scripts/h6_cora_sft.py)，并已按下方合同
完成 audit、smoke、baseline、train、evaluate、verify；不能拿旧 World 训练命令冒充。

## 执行方法

**先固定数据。** 使用既有 train 158 / validation 53 的允许范围，逐 root 盘点：
图像可读、当前导航/ego/history、可信 expert 原生 route/speed、时间/坐标/单位、
teacher 来源、候选身份和 mask。统计可用与排除原因，不能只报总 frame 数。
数据增强首轮关闭，按物理 root 均匀采样；同 root 多帧/分支不能增加独立样本数。
mask 不允许把缺失当零。旧 calibration/locked/pilot 不用于模型选择或训练。

SFT target 必须在原生空间/时间采样域内真实存在；canonical 10 点轨迹不保证能恢复原生全部目标。
先合法读取原始 expert/timeline；不足按数据与资源合同做唯一一次有界正常专家补采，
新数据先分 train/dev 并排除旧 lineage 重叠，不改变原 split。
第一天结束仍无有效监督/真实 batch 就报告具体缺项和受影响步骤。

**再固定加载与输入。** 验证原生 checkpoint 的 base/LoRA/驾驶头 keys，记录允许的新 head
及其初始化，未知 missing/unexpected key 视为失败。原生 rank/alpha 优先保持兼容。
确认训练和推理的图像处理、导航编码、速度单位一致，保留 raw/canonical 映射。
不训练语言生成任务，不将 future label 或答案 token 拼进当前观察。

**用小批次证明更新。** 在真实样本上完成前向、SFT-only backward、小步优化、
保存/重载。记录可训练参数、LoRA/驾驶头变化、冻结基座未变化和输出误差。
先核对 dtype，再登记数值比较容差；不能失败后无依据调宽。
过拟合 smoke 只证明链路，样本仍属于 train，不宣传为微调评估。

**固定一次训练配方。** 以下是项目起始默认值，不是论文最优配置或已实测超参：

| 项目 | 起始设置 |
|---|---|
| seed | 17，M1/M2 共同使用 |
| optimizer | AdamW，LoRA 峰值 lr=2e-5，驾驶头峰值 lr=1e-4；C4 新 World head lr=5e-4；weight_decay=0.01 |
| batch | microbatch=1，梯度累积=4；实际步数按 optimizer.step 计 |
| 稳定性 | 驾驶参数组 grad clip=1.0；C4 World head 独立 clip=1.0，不合并全局范数；CUDA 支持时 BF16，否则经真实 batch 确认可用精度 |
| loss | route/speed 各自有效元素 SmoothL1 均值，再按各 1.0 权重求和 |
| 训练长度 | 起始最多 200 optimizer updates，4 小时完整训练上限先到即停 |
| checkpoint | 最后一个有限有效更新作为主比较；中间保存只为恢复，不挑最好 checkpoint |
| 预热与衰减 | 前 max(1,ceil(0.05*T)) 步只更新已有驾驶头；随后解冻 LoRA 并在同等长度内线性升到峰值，再余弦衰减到峰值的 10% |

C3 smoke 同时估计共享表示/小 World head 的增量成本，不做额外完整训练。
在 M1 正式训练前，按 M1/M2 预计耗时将共同步数一次性下调到预算内并记录；
真实 M2 超时无法达到共同步数时标预算不匹配，不能隐瞒后宣称公平对照。
配置只能在完整训练前因真实兼容/稳定性问题调整，不按 validation 收益反复搜索。
恢复保留 optimizer/scheduler/RNG 状态与累计预算，不把恢复当一轮新训练。

**降低小样本退化。** 保留已有驾驶头权重，不无故重新初始化；只对真实缺失且明确新增的
输出层初始化。首轮不加 teacher 网络、SAM、额外视觉正则或模型平均，避免同时引入新依赖。
M1/M2 相同预热长度、LoRA 更新数、数据顺序与学习率日程；辅助 head 初始化使用独立 RNG，
不改变 SFT 采样/初始化随机流。预热是一次训练中的步骤，不增加模型或 C 子阶段。
只检查当前可观测语义和 teacher 合法性，不按 M0 在某个样本上的误差挑训练集。

在 M0 评估前将本轮主要比较/全部辅助指标登记到 run config，口径见 PROJECT。
M0 重复前向与 round-trip 用于建立数值误差量级，不能把同一模型的重复运行当独立 seed。
“微弱正向”必须大于已测数值误差，并保留所有 root 与退化指标。
现有 validation 已用于历史研究，任何结果均称开发集结果，不能标成新独立测试。

**完成离线对照。** M0/M1 使用同一有效 validation root 清单和无增强预处理。
报告原生 route 平均距离误差/终点误差、speed-waypoint 位置误差；若转换得到速度，
另报告 m/s 误差并说明转换，不能把 waypoint 位置误差误写为速度误差。
同时报告 canonical 可用率、非有限输出、root 数和资源。按 root 先聚合再总计，
保存全部逐 root 预测，完整比较不只展示变好的例子。

## 验收与产物

沿用 generated/h6/cora/<run-id>/，具体 run-id 在启动时登记且不可覆盖。
计划产物包括数据 manifest、配置、M0/M1 预测、M1 adapter/heads、恢复状态、
训练日志、评估报告、资源账本和 stage-summary.json；这些名称在权威 C3 run 中均已存在，
大型权重仍按规则保留在本机 ignored 目录。
摘要必须给出真实复现命令、输入/输出路径、代码/dirty-worktree/模型身份和测试结果。

完成要求：达到训练前锁定的 optimizer updates、LoRA 确实更新、重载通过、有效开发预测齐全、
无泄漏与旧保留集混用。loss 降低或胜 M0 不是工程完成的替代条件。
超时未达锁定步数只保存可恢复 checkpoint 并标 PARTIAL，不能把几步 smoke 当完整训练。
M1 负收益仍可交付，但正文保留它。新行为有最小相关测试，修改广泛才跑全量回归；
真实 GPU 测量与单元测试各自报告。

C3 最多两天；完整训练最多 4 小时，全部 smoke/失败计入总 10 小时 GPU 账。
出现环境/数据/未知改动阻塞或达到修复上限时保存具体未完成项。
验收后更新 PROGRESS、START_TASK 和实际命令入口，停止于 C3，不执行 C4。

## C3 量化交付

执行 [PROJECT 论文量化合同](PROJECT.md#论文量化合同c3_c6_metrics_v1planned)。在正式训练前
冻结 metrics-spec、数据 mask、root 聚合与 eps_ADE；目标幅度为规划值，不是训练保证。
M0/M1 必交 P-ADE、P-FDE、P-WP、P-VALID、P-FAIL；P-SPEED 只在时间与真值成立时计算。
保存双方同输入的逐 root 数值、有效点数、失败原因，报告绝对/相对差和配对 CI。
显存、时间、参数数、实际 updates 与重载误差同时交付。不能按 M1 的预测成功情况删验证样本。
工程完成不依赖误差获胜；量化记录与最小聚合测试已实现并通过，M0/M1 实测结果、hash 和
复现命令见 PROGRESS、EVIDENCE 及权威 run。C4 仍需独立实现和验收。

## 面向 2%–3% 目标的实施优先项

以下是代码核对后的适配要求与待验证改进，不代表已修复代码或已测得收益。

**先保证学的是原生目标。** 本地上游 DrivingAdaptor 的 route/speed head 输出增量，
get_predictions 与 compute_loss 都做 cumsum(1)。新训练必须且只累加一次，监督累计位置。
route 为 20 个空间点：本地 equal_spacing_route 在弧长 0…19 m、间距 1 m 处插值；
speed-waypoint 为 10 个时间点，dt 必须从实际记录解析。不能把空间点按 0.25 s 当未来轨迹。
原生首点与末点、ego/map 坐标、左右轴和 yaw 单位逐项验证；raw ADE 在任何控制平滑前计算。
明确 loss 为 SmoothL1(beta=1.0) 的有效坐标元素均值，route/speed 各权重 1；
与上游 sum(-1) 的标量缩放区别登记，不无意改变 M1/M2 的 loss 比例。
上游缺 future 文件时重复上一帧、短 route 重复末点的实现不可直接用来填有效新标签；
新 loader 保存可支持区间及 mask，缺失不能伪装成停车或完整 20 点监督。

**训练与部署共享输入构造。** 复用当前 build_driving_input 的图像布局、裁剪/JPEG 选项、
导航 prompt、两个 target points 和实际 resolved speed；保存这些参数的身份。
标签来自 expert 不意味着可以把 expert 的完整未来路线或特权 actor 输入拼进 prompt。
同一真实样本在适配入口与部署入口的 tensor/prompt/预测应在预登记容差内一致。
停车、低速、直行/转弯、导航分支与 teacher 可观测性只做预登记分布诊断，
不因难样本或负收益删除 validation；无法证明可观测性时记录 UNKNOWN，不假定为通过。

**独立训练入口，干净起点。** 本地 simlingo_runtime.keep_model_on_gpu 会冻结全部参数，
推理入口使用 inference_mode；训练应直接调用原生可微 forward/head，复用输入构造与加载逻辑，
不能经 numpy/.detach()/inference_mode 训练。冻结基座不等于对整个共享前向禁用 autograd。
LoRA 与 route/speed head 使用显式参数白名单；原生 query embeddings 本周保持 M0 值冻结，
不因调用 model.train() 意外解冻视觉基座/投影/其他参数。R 的输入不截断共享 LoRA 梯度。
真实 train smoke 完成后丢弃试更新，正式 M0/M1/M2 都从原始 M0 身份重新加载；
optimizer/scheduler/RNG 重置到登记起点，M2 不继承 M1 或 smoke 的权重/动量。
保留已有 LoRA 因子，不用新初始化策略覆盖已经训练过的 adapter。

**让有限更新覆盖数据。** 对有效 SFT roots 按 seed=17 确定性打乱、逐轮无放回遍历；
一个 root 有多个有效 anchor 时按独立确定性顺序轮换，每次仍只贡献一个 SFT 样本。
M1/M2 共享同一序列，World 缺标不改变抽样顺序；不按验证误差做难例挖掘。
记录每 root 暴露次数及 min/max，区分独立 root 数与累计训练样本数。
梯度累积用实际窗口长度归一化，最后不足 4 个样本不固定除以 4；
route/speed 各 mask 先在样本内平均，缺某头时该项为零并记录，不临时加倍另一头。
无放回采样与独立梯度裁剪为项目实现选择，收益待实测，不宣称来自某论文的保证。
