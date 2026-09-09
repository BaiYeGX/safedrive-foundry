# SafeDrive Foundry

CARLA–ROS 2 软件在环驾驶研究：**真实 VLA 微调 → 短时后果 World 联合学习 →
独立双候选选择 → Safety 执行 → 小型闭环验证**。本机资源属于实验设置，不是标题或贡献。

## 现在只做四步

| 阶段 | 工作 | 时间 |
|---|---|---|
| C3 | 接好数据并完成常规 VLA LoRA 微调 | 第 1–2 天 |
| C4 | 加一个四输出 World 辅助任务，完成联合微调 | 第 3–4 天 |
| C5 | 接在线，做三臂 18 次开发运行 | 第 5 天 |
| C6 | 整理权重、结果、重放和论文材料 | 第 6 天；第 7 天缓冲 |

只训练 M1 常规微调、M2 联合微调；M0 原模型作离线基线。
不再要求执行机制分解、ensemble、正式校准或 108-root formal。
一周目标是完成可信训练与闭环记录，不保证训练必然提升。

## 已有进展

C0/C1 已完成；C2 有 340 usable roots，其中 train 158 / validation 53。
C2 World 基线未超过 candidate-only ridge，原稀有事件门仍失败；历史结果完整保留。
C3 已于 2026-09-09 完成并验证：真实数据适配、CUDA batch smoke、M0 离线基线和 200 更新
M1 LoRA SFT 均有可重载证据。权威产物位于
`generated/h6/cora/c3-vla-sft-20260909-final-v3/`；当前入口已切到 C4，详见
[START_TASK](START_TASK.md)、[PROGRESS](PROGRESS.md) 和 [ROADMAP](ROADMAP.md)。

## 文档入口

| 内容 | 文件 |
|---|---|
| 当前任务、长期规则、进展 | [START_TASK](START_TASK.md)、[AGENTS](AGENTS.md)、[PROGRESS](PROGRESS.md) |
| 项目与方法 | [PROJECT](docs/PROJECT.md)、[WORLD_MODEL](docs/WORLD_MODEL.md) |
| 数据与双候选 | [COUNTERFACTUAL_DATA](docs/COUNTERFACTUAL_DATA.md)、[HYBRID_CANDIDATES](docs/HYBRID_CANDIDATES.md) |
| 资源与环境 | [RESOURCES](docs/RESOURCES.md)、[ENVIRONMENT](docs/ENVIRONMENT.md) |
| 文献、证据、展示 | [RELATED_WORK](docs/RELATED_WORK.md)、[EVIDENCE](docs/EVIDENCE.md)、[SHOWCASE](docs/SHOWCASE.md) |
| 代码入口 | [C3 SFT CLI](scripts/h6_cora_sft.py)、[runtime](safedrive_foundry/README.md) |

三份研究备忘录仅解释背景，不另设任务。archive、冻结 Evidence 与第三方说明保持只读。

## 固定边界

Windows CARLA、WSL2 模型/训练/ROS 2；RTX 4080 16GB、i5-13600KF。
Classic Expert 和 nominal VLA 各自独立生成一条轨迹，逐候选 Guard 后 World 才能 rank/defer。
Safety 与 MPC/PID 保持最终执行权限；World 无轨迹生成、底盘或 tick 权限。
只有 ScenarioRuntime 持有正式 tick；真实 CARLA 运行先执行本次 preflight READY 检查。

## 单阶段 goal 入口

直接复制 [ROADMAP 的阶段索引](ROADMAP.md) 所链接文档中的 goal 文本即可。
C3 数据/常规微调、C4 联合微调、C5 开发闭环、C6 交付各自有完整验收与停止条件。
C3 已完成；C4 尚未启动。本地复现实验入口：

```bash
python scripts/h6_cora_sft.py audit --run-id c3-vla-sft-20260909-final-v3 --device cuda
python scripts/h6_cora_sft.py evaluate --run-id c3-vla-sft-20260909-final-v3 --device cuda
python scripts/h6_cora_sft.py verify --run-id c3-vla-sft-20260909-final-v3 --device cpu
```


## 本轮研究优化

四阶段不变；C3 已按保守微调配方完成，C4 采用固定候选基线的后果残差与
驾驶梯度相容门控，C5/C6 按预登记比较呈现微弱收益与代价。
主要方法目标是 M2 对 M1 的 route ADE 改善，不要求超过已饱和的旧 ridge 排序。
当前已有 C3 新训练结果，尚无 C4 联合训练或 C5 闭环结果；完整口径见 [PROJECT](docs/PROJECT.md)。

## 论文指标与优势证据

统一定义见 [论文量化合同](docs/PROJECT.md#论文量化合同c3_c6_metrics_v1planned)。
主要方法比较是 M2/M1 的原生 route ADE；适配、后果预测、闭环进度和运行代价分别报告。
规划目标与实测收益分开；C3 的新训练指标已测，C4/C5/C6 仍按路线执行。
