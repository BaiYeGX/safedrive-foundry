# G4-04：可比重建、反事实分支、最小化与聚类

**状态**：PENDING
**依赖**：G4-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、13、14 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、6～8 节；
3. G4-01 场景 schema、G4-02/03 执行器与 archive；
4. G3-05 发布模型、G2 SafetyEvent 接口；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

从危险场景的可比起点执行 Raw VLA、VLA+Safety、Classic、RATO 和 Brake 分支，最小化失败并形成稳定 failure cluster。

## 实现范围与边界

- 检查 Ego、关键 Actor、信号、路线和事件进度容差；
- 二分、坐标下降或 delta debugging 最小化；
- 保存起点差异、动作、轨迹、风险、进度、舒适和模块事件；
- 超容差标不可比较，Recorder 不视为无损恢复；

## 完成标准与验证

- 最小反例可重复、可去重并绑定 provenance。
- failure cluster ID 稳定且连续帧不虚增。
- 不可比较结果不进入偏好、World 或因果主张。
- Counterexample/Regression Registry 可查询和重放。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

