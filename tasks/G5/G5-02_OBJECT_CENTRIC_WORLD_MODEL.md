# G5-02：轻量动作条件 Object-Centric 世界模型

**状态**：PENDING
**依赖**：G5-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3、5、6、9、13 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1～4、7、8 节；
3. docs/project/CLAIMS.md C2 与统一指标；
4. G5-01 ActionBranchDataset、简单基线、split 和 resource smoke；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

训练 object/vector/BEV latent 交互动力学，以 scene/Actor/map/route token 和 Ego 候选为条件批量预测 1～5 秒 Actor 多模态未来与占用；不生成像素视频。

## 实现范围与边界

- Actor token、交互/地图编码、action conditioning、rollout 和 mask；
- VLA/World 可缓存共享特征，但 checkpoint/schema/可用性/开关独立；
- mixed precision、批量 K 候选、checkpoint 和 OOM 降级；
- 运行目标 5～10Hz 是待实测指标，不是预写结果；

## 完成标准与验证

- 不同 Ego 候选产生可区分 future，action permutation 和小样本过拟合通过。
- 报告参数、显存、吞吐、时延分位、actor error 和 interaction slice。
- 多步误差、missing actor、变长输入和 unavailable 行为明确。
- 单卡训练/推理与 checkpoint 精确恢复通过。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `world_model/model、training、config、tests/g5、registry、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
