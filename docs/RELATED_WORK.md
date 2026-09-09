# 相关工作与本周方法定位

本次只整理已有文献检索，不新增论文结论或声称复现外部成绩。
原始来源保留如下；未确认会议状态的不推断录用。

## 本周采用的研究方向

从 DriveVLA-W0、FLARE、SimWAM 借鉴“真实未来辅助监督可以塑造策略表示”的方向，
在本项目中用现有四维真实驾驶后果与配对差分训练共享 LoRA；当前具体配方采用
固定候选基线残差与驾驶梯度相容门控，细节见 WORLD_MODEL。
不移植视频生成、想象 RL、执行中介或多教师结构；这些不在本周工作量内。

M1 常规 SFT 对 M2 联合微调回答策略作用；同一 M2 的 World off/on 回答在线选择作用。
已有 C2 ridge 很强，必须保留；新实验不能用改指标掩盖旧 NO_DEMONSTRATED_GAIN。
本项目旧损失已有配对差分，联合未来监督也有强近邻，不能将二者重新命名为新发明。
具体贡献取决于新的可复核作用、适用范围和失败分析，不能保证论文新颖性已成立。

## 论文证据表

下列是本轮重点核对的原始来源。会议状态只在来源支持时注明；未注明的不推断录用。
检索摘要不等于完整复现，作者报告收益不等于本项目可达到的收益。

| 工作 | 方法及与本项目关系 | 原始来源 |
|---|---|---|
| DriveVLA-W0，ICLR 2026 | 未来图像监督补充低维动作监督；借鉴联合学习动机，不复现数据扩展规模 | https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d70423f59c5fdd24f0dd3fa52e34623-Abstract-Conference.html |
| FLARE: Robot Learning with Implicit World Modeling，2025 | 在策略中增加未来 token，对齐冻结未来视觉表示；辅助未来监督已有强近邻 | https://arxiv.org/html/2505.15659v1 |
| FRAPPE，2026-02 | 分阶段、多视觉未来表征对齐；多教师/并行专家不是本周必要项 | https://arxiv.org/html/2602.17259v1 |
| DriveLaW，CVPR 2026（作者仓库） | 视频生成 latent 注入动作 diffusion planner；借鉴显式信息接口 | https://github.com/xiaomi-research/drivelaw |
| DriveWorld-VLA，2026-02 | 共享 latent、动作条件未来生成、未来引导评价与修正；总框架已被覆盖 | https://arxiv.org/html/2602.06521v1 |
| VLA-World，2026-04 | 候选引导下一帧生成，再反思修正；正文报告 8 A100 训练、4 A100 推理 | https://arxiv.org/html/2604.09059v1 |
| LCDrive，初稿 2025-12 | 动作 token 与世界 token 交替；附录用未来车辆 boxes 与候选 ego pose 构建紧凑目标 | https://arxiv.org/html/2512.10226v1 |
| DIAL，2026-03 | 以预测 latent intent 为结构瓶颈，再由逆动力学产生动作；提示辅助预测未必被策略利用 | https://xpeng-robotics.github.io/dial/ |
| HyWorldVLA，2026-07 | 前期像素与 latent 双监督，后期 latent 与动作专家联合训练；过大辅助损失权重可退化 | https://arxiv.org/html/2607.20988v1 |
| SimWAM，2026-08 | 联合 video/action flow matching；未来与动作分支互不可见，部署去除视频分支 | https://arxiv.org/html/2608.07468v1 |
| VLA-MBPO，2026-03 | chunk 级世界模型与从真实数据起点出发的短分支 rollout，降低误差累积 | https://arxiv.org/html/2603.20607v1 |
| WIMLE，ICLR 2026 | ensemble/latent sampling 不确定性给合成 transition 加权；可靠性加权不新 | https://arxiv.org/html/2602.14351v2 |
| REVAMP，作者页标明投稿 CoRL 2026 | 动力学/Q 共享模型、双可靠性信号、定向真实交互与策略更新；没有核实录用 | https://revampcorl.github.io/REVAMP/ |
| RENEW，2026-07 | 人类偏好修复世界模型动力学，不确定性定向查询；不采用人工偏好为本周依赖 | https://arxiv.org/html/2607.14180v1 |
| DreamZero，2026-02 | 视频基础模型联合生成动作与视频；其跨机器人适配结果不能外推成本地驾驶收益 | https://arxiv.org/abs/2602.15922 |
| Pre-VLA，2026-05 | 在执行或 WM rollout 前验证动作并有界重采样；Guard-before-World 不能称为新贡献 | https://arxiv.org/abs/2605.22446 |
| WAM robustness study，v5 2026-07 | 比较视觉/语言扰动下的 WAM/VLA；提示结构、预训练、数据混杂需要控制 | https://arxiv.org/abs/2603.22078v5 |

已核对近邻仍需纳入：SafeAlign-VLA（https://arxiv.org/html/2605.19524v1）、
FACT（https://arxiv.org/html/2608.10232v1）、Delta-JEPA
（https://arxiv.org/html/2606.31232v1）、DynaDreamer
（https://arxiv.org/html/2607.13410v1）。
注意：机器人 FLARE 与驾驶 FLARE 是不同论文，不能混用模型配置与结果。

## 驾驶基础与评估边界

[SimLingo](https://arxiv.org/abs/2503.09594)与
[官方代码](https://github.com/RenzKa/simlingo)是本地 VLA 微调的基础。
[PDM](https://arxiv.org/abs/2306.07962)、
[BEV-Planner](https://arxiv.org/abs/2312.03031)提示强先验与 shortcut 对照；
[DAgger](https://proceedings.mlr.press/v15/ross11a.html)提示换策略后需真实闭环检查。
本周只做小样本开发评估，无概率校准或安全非劣证明。
完整步骤只有 [ROADMAP](../ROADMAP.md) 中的 C3、C4、C5、C6；
论文多不意味着必须多实现模型。硬件只写实验设置。

本次进一步研究只更新执行决定，没有新增论文综述。残差预测与辅助梯度门控都有先例，
梯度不冲突也不自动意味着泛化提升；不可将该组合重命名为已证实创新。
