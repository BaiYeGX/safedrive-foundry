# G5-04：VLA 候选排序与 Active CARLA Verification

**状态**：PENDING
**依赖**：G5-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 时间、tick owner、Profile 与组件权限章节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～7、9、12～14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、5、7、8 节；
4. G2-05、G3-05 发布接口和 G5-01～G5-03 最终产物、断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

只对 G2 预筛合法候选做 World 风险—进度—舒适排序；最终轨迹再次进入 Safety。高风险/高不确定/分歧/OOD 候选进入异步 CARLA verification queue。

## 实现范围与边界

- VLA 目标 10Hz、World 目标 5～10Hz，采用特征缓存和流水/串行调度；
- World 超时继续新鲜安全轨迹，过期后冻结降级；
- Active CARLA 有预算、优先级、幂等、超时、恢复和 cache provenance；
- unavailable/overconfident/timeout/OOD 注入回到 VLA+Safety+Classic；

## 完成标准与验证

- 报告 top-1/regret、排序收益、错误筛选、CARLA 节省、时延/显存/降级。
- Active Verify 不争夺实时 tick/control，车辆不等待验证。
- World 不能解除硬拒绝、改 Safety 阈值或直接控制。
- 关闭 World 后 VLA+Safety 可复现。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `world_model/runtime、verification_queue、cache、runtime arbitration adapter、tests/g5、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
