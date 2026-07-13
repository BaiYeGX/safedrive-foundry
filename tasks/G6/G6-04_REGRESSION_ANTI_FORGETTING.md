# G6-04：历史失败、正常能力、OOD 与抗遗忘回归

**状态**：PENDING
**依赖**：G6-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C5 与统一指标；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 8、10、12～13 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md G6 与资源字段；
4. G6-01～G6-03 数据、采集、checkpoint 和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

按配对 seed 比较后训练前后策略在目标失败、相邻失败簇、正常能力、未见 OOD、故障和实时性上的变化。

## 实现范围与边界

- 预先冻结提升、不可接受回归和无结论区间；
- 目标失败迁移、新失败和无故停车单独登记；
- 与 base model 和随机/周期采样后训练公平对照；
- 四类指标：安全、效率/舒适、泛化、资源/实时性；

## 完成标准与验证

- 报告效应大小、区间、重复失败、正常指标、OOD 和资源。
- 不以平均分掩盖关键场景退化。
- 模型/data/config/seed/run 全谱系。
- 结果只进入 G6-05，不在本任务宣称飞轮完成。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

尚未开始。恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

