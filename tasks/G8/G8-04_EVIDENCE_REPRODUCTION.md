# G8-04：Evidence Bundle、演示与冷启动复现

**状态**：PENDING
**依赖**：G8-02、G8-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md 全文；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 10～14 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 全文；
4. G8-01 冻结清单、G8-02 runs、G8-03 统计产物；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

把原始 run、指标、统计、图表、视频、失败回放和复现入口绑定到同一证据谱系，完成代表性冷启动复现。

## 实现范围与边界

- Evidence manifest、结果卡、模型/World/系统/数据卡；
- 四发布配置演示、关键失败和负结果回放；
- 固定 seed 的构建/运行/聚合/报告入口；
- 代码/环境/模型/数据/场景/config hash 链接检查；

## 完成标准与验证

- 每项主张可追溯到协议、原始日志和聚合脚本。
- 全新工作目录复现代表性正/负结果。
- 产物损坏、重复、缺失和选择性删除检测通过。
- 简历数字来源索引不包含未 VERIFIED 结果。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

