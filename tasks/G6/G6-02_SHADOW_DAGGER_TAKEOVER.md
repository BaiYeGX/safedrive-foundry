# G6-02：Shadow DAgger、Takeover 与 Pre-intervention 采集

**状态**：PENDING
**依赖**：G6-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～5、8、12、13 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4～8 节；
3. docs/project/CLAIMS.md C5 与统一指标；
4. G6-01 EventWindow/HardCase schema、门禁、manifest 和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

并行运行 VLA、Classic、World 和 Safety，以分歧、风险、World 不确定/误排和 OOD 触发专家查询，采集 state/policy/world/expert/safety/executed action。

## 实现范围与边界

- 执行、影子和反事实动作严格隔离；
- 专家监督绑定版本、Observable 输入和 candidate_id；
- query budget、原因、重复率、场景平衡和来源可审计；
- World 不可自己验证偏好，训练不可写 Regression；

## 完成标准与验证

- 触发样本价值相对随机采样可量化。
- Shadow 不获取控制权、tick master 或延迟 50Hz 控制。
- 中断恢复不重复，样本可回放到 run/frame/candidate。
- unavailable/timeout/OOD 注入产生正确 trigger/abstain。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `data_pipeline/dagger、runtime shadow adapter、config/collection、tests/g6、reports` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
