# G6-03：CARLA 验证偏好、Risk Anticipation 与轻量后训练

**状态**：PENDING
**依赖**：G6-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 5、6、10、12、13 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1～4、7、8 节；
3. docs/project/CLAIMS.md C5 与统一指标；
4. G6-01 数据门禁、G6-02 采集 manifest、G3-04 base checkpoint 和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

用可比 CARLA 分支、版本化专家反馈和接管前窗口构造 pairwise/ranking 数据，以 LoRA/QLoRA SFT、trajectory ranking/DPO 类目标和 Risk Anticipation 单卡后训练；首版不做 PPO。

## 实现范围与边界

- chosen/rejected 共享可比起点并绑定结果和验证来源；
- World 可提议但未验证预测不能成为真值；
- 主损失保持轨迹可执行，偏好/风险提前量为辅助；
- base/adapter/data/config/seed/checkpoint 全谱系；

## 完成标准与验证

- 训练可恢复，NaN/OOM 审计且适配 16GB。
- 独立 split 的偏好、风险提前量、校准和可执行率公平比较。
- 泄漏、冲突偏好、不可比和 World-only 标签被拒绝。
- 训练指标不冒充闭环收益，必须进入 G6-04。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `driving_vla/posttrain、losses、config/posttrain、registry、tests/g6、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
