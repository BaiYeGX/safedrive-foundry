# 当前任务：C4 — 后果辅助联合微调与现场接入准备

状态：C3 按学校项目的离线基线范围收尾，M1、数据及评估可复用；C4 为 `PLANNED / NOT_RUN`。
历史资源记录的核对列入新增 GPU 优化的前置工作。当前阶段不以重新训练 C3 为默认起点。

截止：周日 **2026-09-13**；老师要求实时 CARLA 闭环。当前分支 `codex/c3-school-closeout`。
本文件只登记下一执行入口，本次为文档整理，未启动 C4 训练或 CARLA。

## 执行依据

1. [C4 执行单](docs/C4_EXECUTION_PLAN.md)：顺序、固定配方、交付节点与待实现入口。
2. [World 模型合同](docs/WORLD_MODEL.md)：表示、残差、配对损失、共享梯度门控。
3. [环境](docs/ENVIRONMENT.md)、[资源](docs/RESOURCES.md)：统一 sim、tick owner、运行预算。
4. [项目定义](docs/PROJECT.md)、[已确认进度](PROGRESS.md)：范围与实际状态。

## 直接推进顺序

- 读取 C3 manifest、原始 M0 身份、M1 权重和 200-step 训练配置，建立唯一 C4 run directory。
- 优先定位 CARLA 启动故障，通过本次 READY 后验证现有 M1 的最小现场闭环；
  同时核对 World 候选/后果绑定和历史预算。CARLA 未恢复时继续 CPU 数据与代码准备。
- 实现共享 h、冻结 b、四输出 R 和完整累积窗口的双梯度门控，完成真实联合 smoke 与恢复检查。
- 丢弃 smoke 更新，从原始 M0 开始 M2，按与 M1 对齐的 seed、SFT 顺序和 200 updates 完成一次训练。
- 独立加载并评估 M0/M1/M2，生成逐 root 结果、b+R/b 对照、资源证据和 C5 加载配置。
- C4 验收后更新当前入口至 C5；C5 执行小范围三臂现场对照，C6 完成排练与交付。

## 固定输入与范围

C3 基线目录：`generated/h6/cora/c3-repair-20260910T165902Z/`。
使用其 `manifest.json`、`run_config.json`、`metrics-spec.json`、`m1_adapter.pt`、
`m1_training_summary.json` 和保存预测。train/validation 为 158/53，运行逻辑从清单推导数量。
M2 从原始 M0 初始化；沿用 seed 17、AdamW、microbatch 1、累积 4 和原 LoRA/驾驶头日程。
World 增加 progress、acceleration RMS、jerk RMS、lateral acceleration RMS 四个真实后果目标。

C4 主比较为 M2/M1 的 P-WP，P-ADE、P-VALID/P-FAIL 同表；b+R 与本次冻结 b 分头比较。
相同 validation 输入、真值 mask 和 root 分母，正式运行前冻结评价定义，结果由保存预测复算。
不追加多 seed、大规模消融、新公开数据或新的模型路线。

## 资源与现场约束

单次 M2 优化最多 4 小时；C3/C4 含 smoke/失败累计最多 10 小时，整卡采样峰值最多 14.5 GiB。
历史未知耗时须有可证明上界，新增优化前确认剩余量；现场试跑纳入既有 CARLA 总账。
训练与渲染不并发。CARLA 必须取得本次 RPC READY，ScenarioRuntime 唯一持有 tick。
现场显式绑定当前 M1/M2，记录模型候选实际参与控制、回退和延迟。

## 完成节点

C4 完成需有真实联合更新、200-step M2、可重载权重、完整离线对照、资源账本和 C5 加载配置。
算法是否有收益与工程是否完成分别记录；现场三臂运行属于 C5，状态按实际执行更新。
周日中午为建议版本冻结点，下午留给现场排练、C6 材料和最终 GitHub 同步。

实现新增入口后登记真实 `--help`、执行命令、直接测试和产物路径；目前完整 M2 CLI 尚待实现。
保留 `.codex/`、`test_registry.sqlite3` 及机器状态文件为本机内容，提交时显式排除。
