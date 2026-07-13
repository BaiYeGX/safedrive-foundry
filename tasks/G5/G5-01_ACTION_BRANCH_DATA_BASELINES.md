# G5-01：动作分支数据与简单世界基线

**状态**：PENDING
**依赖**：G4-05

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 身份、时间、Oracle/Observable 与 Registry 章节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3、5、8、13、14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1～4、6～8 节；
4. G4-05 发布场景/反事实产物与 G3/G2 候选、安全接口；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

从可比起点执行专家、VLA、非专家、危险和扰动 trajectory chunk，构建 P(other future | scene, ego action)，实现 persistence、CV/CTRV、IDM/规则和 Reward MLP。

## 实现范围与边界

- scene/candidate/action、Actor 现态与 1～5 秒未来严格对齐；
- occupancy、碰撞/TTC、规则、进度、舒适、termination 和可比性；
- 非专家动作覆盖，Regression/VLA 测试集隔离；
- 不可比、仿真异常、Safety 提前终止分别登记；

## 完成标准与验证

- 基线报告 actor/occupancy、collision/ranking、校准、时延和资源。
- action swap 改变基线输出，固定 seed 数据 hash 一致可续采。
- 数据卡含动作分布、场景覆盖、危险动作和 Oracle 字段。
- CARLA 分支只走统一 runtime/反事实入口。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `world_model/data、baselines、schema、registry、config、tests/g5、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
