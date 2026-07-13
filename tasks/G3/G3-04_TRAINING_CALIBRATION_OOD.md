# G3-04：VLA 多任务训练、动作 Grounding、校准与 OOD

**状态**：PENDING
**依赖**：G3-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 5、6、10、12 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1～4、7、8 节；
3. docs/project/CLAIMS.md C1 与统一指标；
4. G3-01 split/数据 manifest、G3-03 模型配置、resource smoke 和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

训练多候选轨迹、行为、关键 Actor、风险时域、意图和不确定性；建立语言—Actor—风险—轨迹一致性、选择性驾驶及单卡稳定训练。

## 实现范围与边界

- LoRA/QLoRA、混合精度、梯度累积、checkpoint 精确恢复和 OOM/NaN 审计；
- 主目标为轨迹 NLL/匹配/可执行/平滑，语义头是动作 grounding 辅助目标；
- 温度校准、候选分歧、embedding OOD、Risk-Coverage 和 Fast/Slow trigger；
- 关键 Actor 删除/移动、路线冲突与风险时域反事实；

## 完成标准与验证

- 冻结 split 报告轨迹、行为/Actor、ECE/Brier、OOD、误漏接管、资源和区间。
- checkpoint 规则预登记，训练/调参不可访问 Regression 标签。
- 视觉退化、未见 Town/天气、路线冲突和分歧用例通过。
- 语言无净收益时保留负结论，不换 split/阈值。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `driving_vla/training、losses、uncertainty、evaluation/grounding、config、registry、tests/g3` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
