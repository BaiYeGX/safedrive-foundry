# G3-05：VLA+Safety 闭环、选择性驾驶与阶段验收

**状态**：PENDING
**依赖**：G3-01～G3-04

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C1、C3 与统一指标；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～7、10、12 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、5、7、8 节；
4. G2-05 发布接口与 G3-01～G3-04 最终产物、失败 Evidence 和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

接入 Runtime、G2 Safety 和 MPC，比较 Classic-Observable、非语言多候选、Raw VLA、VLA+Validator、VLA+完整 Safety，验证无特权闭环与失效边界。

## 实现范围与边界

- ID/OOD、退化视觉和交互长尾多 seed；
- 记录拒绝/修复、OOD/abstain、超时、接管和资源；
- 比较 open-loop/closed-loop 排名并登记失败类型；
- 控制 20ms loop 不等待 GPU；

## 完成标准与验证

- 安全、完成、舒适、grounding、时延和资源证据完整。
- unavailable/timeout/stale/OOD 按 G2 链降级。
- C1 可为负，但真实闭环、模型卡和 Evidence 必须完成。
- Evidence 自检通过且不自动开始 G4。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `G3 小型缺陷、runtime adapter、validation/g3、registry、artifacts/g3、tests/g3、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
