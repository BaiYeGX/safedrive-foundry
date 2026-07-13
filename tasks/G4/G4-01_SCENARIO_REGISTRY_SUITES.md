# G4-01：场景注册表、核心长尾套件、覆盖与可解性

**状态**：PENDING
**依赖**：G3-05

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md Registry、身份与 Oracle/Observable 章节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 2、4、6～8 节；
4. G3-01 场景身份/split 与 G3-05 发布模型、失败清单和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

扩展 G3 场景身份为 Functional→Logical→Concrete→Regression→Minimal Counterexample 层级，统一 Engineered、Failure-derived、System-generated 来源。

## 实现范围与边界

- 切入/急刹、换道冲突、无保护左转、VRU、遮挡、阻断/U-turn、低附着和故障；
- 地图/Actor/天气/语言/故障参数域、有效性、覆盖和版本迁移；
- 每族 base/边界/可解条件/预期事件/termination；
- 正常、长尾、故障和 Regression 套件隔离；

## 完成标准与验证

- 错误 spawn、不可达、物理重叠、先天不可解和非法参数标 INVALID。
- Classic-Oracle 只用于可解性，不进入 Observable 策略。
- 固定 seed 可复现，属性和 schema migration 测试通过。
- registry/provenance 可查询并与 G3 split 兼容。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `scenario、solvability、registry/scenario、config/scenarios、tests/g4、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
