# G8-02：四发布配置统一回归、故障与资源长稳

**状态**：PENDING
**依赖**：G8-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md 全文；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～7、9～14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 全文；
4. G8-01 冻结协议、四配置 manifest、运行矩阵和缺失规则；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

在常规、交互长尾、历史失败、最小反例、OOD 和故障上运行四配置，并测量 20ms 控制 deadline、模型周期、GPU/CPU/内存/磁盘/温度和连续恢复。

## 实现范围与边界

- Town/路线/天气/布局/行为 shift 与视觉退化；
- 传感器/通信/定位/模型/solver/执行器/CARLA 故障；
- 安全、完成、无故停车、舒适、接管、修复、OOD/abstain；
- 控制和 VLA/World 时延、显存、资源峰值、OOM/断连；

## 完成标准与验证

- 矩阵完整或按预登记规则标缺失，失败可定位。
- 关键 case 跨 seed 并抽查视频/轨迹/事件。
- 四配置串行加载，训练前后差异可追踪。
- 长稳和频率只报告真实结果，不预写达标。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `validation/g8/regression、stress、collector、artifacts/g8、最小发布阻塞修复` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
