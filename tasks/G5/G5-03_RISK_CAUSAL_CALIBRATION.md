# G5-03：多模态风险、不确定性、动作敏感性与校准

**状态**：PENDING
**依赖**：G5-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3、5、9、10、13 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2～4、7、8 节；
3. docs/project/CLAIMS.md C2 与统一指标；
4. G5-01 数据/基线与 G5-02 schema、checkpoint、可用性和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

形成 WorldRolloutBatch，预测碰撞/TTC、规则、舒适、进度、termination/reward 和 epistemic/aleatoric 不确定性，并用 intervention 证明模型真正使用 Ego action 与关键 Actor。

## 实现范围与边界

- action swap/removal、critical actor removal/perturbation、无动作条件对照；
- actor/occupancy、collision/action ranking、regret、ECE/Brier；
- 不确定性—误差相关、模式坍塌、过度自信和 OOD；
- paired CARLA 是评价真值，World 永远不是 Safety 真值；

## 完成标准与验证

- 安全/擦碰/碰撞/违规/让行样例输出分项风险和置信。
- 至少一个预登记交互 slice 优于简单基线或保留负结论。
- 无动作条件更好时不得宣称动作条件有效。
- 失败返回 unavailable/uncalibrated/OOD 而非零风险。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `world_model/heads、losses、calibration、evaluation、interventions、config、tests/g5` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
