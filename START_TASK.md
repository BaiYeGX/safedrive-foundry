# 当前任务：C4 — World 辅助任务（C3 已完成）

状态：C3 `VERIFIED / ENGINEERING_COMPLETED / ALGORITHM_MEASURED`，C4 当前为 `NOT_RUN`。
权威 C3 run 为 `generated/h6/cora/c3-repair-20260910T165902Z/`，最终记录见
[C3 修复最终验收](docs/runtime-evidence/h6/c3-final-repair-20260910T165902Z.md)，旧运行和
`90.77%` 结论继续作为已撤回历史保留。C3 已完成，本轮停止在 C3，不自动执行 C4；本文件
现在只登记下一阶段入口。

当前分支为 `codex/c3-repair`，最终提交前保留 `.codex/` 和 `test_registry.sqlite3` 为本机文件。
修复版产物根目录为 `generated/h6/cora/c3-repair-20260910T165902Z/`，诊断、smoke、正式训练
分目录，未覆盖旧运行。C4 只能从原始 M0 重新开始，不能使用 M1 或 smoke checkpoint 续训。

## C3 修复已交付的验收摘要

- 旧 C2 release 的 211/211 个原生规划身份恢复，train/validation 保持 158/53；route 和
  speed 逐头 mask，未用导航拼接、未来速度回填或外推路线。
- 真实 CUDA/BF16 batch、可微 forward/backward、LoRA/head 更新、冻结参数指纹、独立重载、
  200/200 正式 M1、尾窗口和资源账本均通过验收。
- 48-root/96-attempt/4-hour 补采清单完成预登记，但 CARLA 在 READY 前外部崩溃，接受 0 个；
  未生成伪造数据。C3 当前评估范围为 `partial_native_support`。
- `verify`、独立指标重算和 6 类篡改攻击均通过；route 结果有 29/53 个 root 退化，不能写成
  普遍驾驶能力提升。C4 的 M2 仍必须从原始 M0 起点开始。

## C4 当前入口（C3 已验收，本轮未执行）

从原始 M0 checkpoint 分别启动 M2 联合微调，不能从 M1 续训；复用同一 train/validation
split、seed、输入处理和更新预算，增加 C2 已有四个连续后果目标（进度、加速度 RMS、jerk
RMS、横向加速度 RMS）。World 输入只能是当前共享表示与候选，真实未来、source/slot/order
元数据和安全真值不进入在线特征；候选交换、驾驶梯度相容门控、分组裁剪与 zero-residual
对照均需有直接验证。输出 M2 与 M1 的配对离线指标、四头 mask/误差、梯度和资源证据；未超过
基线时照实记录。C4 验收后再把入口切到 C5。

## C3 交付合同（已完成）

在一个 C3 内完成数据适配、真实 batch 检查和常规 LoRA SFT，交付 M0 原始离线基线、
M1 新 adapter/驾驶头、逐 root 开发预测及可重载训练记录。目标两天，不再拆子阶段。

允许修改必要的非冻结 loader、训练适配、配置、测试和活动文档；复用本地 SimLingo。
不改旧模型、数据、阈值、保留集或 Evidence，不重扫旧 Hash；在已有修复分支工作。
依据：[数据](docs/COUNTERFACTUAL_DATA.md)、[模型](docs/WORLD_MODEL.md)、
[候选](docs/HYBRID_CANDIDATES.md)、[资源](docs/RESOURCES.md)、
[环境](docs/ENVIRONMENT.md)、[证据](docs/EVIDENCE.md)。

## 执行顺序

- 检查分支、已有改动、真实 GPU 与模型环境。
- 清点现有 train 158 / validation 53 中当前图像、导航、ego 与可信 expert route/speed；
  核对时间、单位、坐标与 root 隔离。旧保留集只审计。
- 核对原生 LoRA/驾驶头 keys 和预处理；验证原生输出到 canonical T=10、dt=0.25 s 的转换。
- 用真实 batch 做 forward/backward 和保存重载，确认 LoRA 更新、冻结参数不变。
- 核对 route 空间采样、speed 时间采样和 head cumsum；训练前丢弃 smoke 试更新并重载 M0。
- 按 C3/C4 合同冻结 root 无放回顺序、原生 query 冻结及驾驶/World 分组裁剪，消除额外干扰。
- 按 C3 合同采用相同短预热/小学习率，在正式 M1 前登记 PROJECT 的主要指标与数值阈值。
- smoke 包括 C4 残差 head/双梯度门控成本；据此冻结共同 T，不另增加模型或训练预算。
- 冻结一个 seed、一份超参和训练步数；运行 M0 离线基线及 M1 SFT，完整保存正负结果。
- 更新 PROGRESS、登记真实可运行命令，并把下一入口切到 C4 后停止。

第一天结束仍无有效示范或真实 GPU batch，则报告具体阻塞与剩余日程影响。
缺标签先从现有原始记录合法重建；有必要且可运行时只允许一次正常专家补采，须先写
新 train/dev manifest、root/attempt 上限和预算，并通过本次 CARLA READY 检查。
不为补采另建研究阶段、不动旧 C2 稀有事件门；控制在 RESOURCES 的同一总账内。

## 验收与验证

有效真实训练、无未来泄漏、专家标签与分支后果分离、checkpoint 重载、逐 root 预测、
资源/失败日志齐全。损失下降不代替这些检查；没有正收益仍可完成工程验收。

新行为有直接测试；适配代码完成后运行最小相关测试、实际 checkpoint round-trip，
再根据影响范围决定是否需要全量回归。通用命令：

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry
git diff --check
```

已实现并实际运行的入口为 [`scripts/h6_cora_sft.py`](scripts/h6_cora_sft.py)，提供
`audit`、`smoke`、`baseline`、`train --resume`、`evaluate` 和 `verify` 子命令；权威命令与
结果保存在旧 C3 run 的 `run_config.json` 和 `stage-summary.json`。旧修复版执行过最小适配测试、
真实 CUDA round-trip、独立 M0/M1 评估、验收篡改拦截、全量 unittest、compileall 与
`git diff --check`；历史全量 unittest 为 509 tests、1 skipped、`OK`（75.094 s）。这些不足以
覆盖再复核反例，不能替代本次重新验证。数据泄漏、
未知重叠修改、环境或安全接口异常仍按 AGENTS 停止。

## 本阶段的论文量化要求

按 [PROJECT 论文量化合同](docs/PROJECT.md#论文量化合同c3_c6_metrics_v1planned)
在 M1 前冻结指标定义、eps_ADE、模型/数据身份与目标；交付逐 root 明细和可复算摘要。
C3 已按该合同冻结指标、`eps_ADE=1e-5 m`、模型/数据身份并保存逐 root 产物；M1/M0
实际结果见 [PROGRESS](PROGRESS.md) 与本地权威 run。M2/C4 仍为 `NOT_RUN`，不能用 C3
结果替代 C4 的联合学习验收。
