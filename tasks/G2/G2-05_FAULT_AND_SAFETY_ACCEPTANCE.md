# G2-05：故障注入、Observable 退化与 Safety 阶段验收

**状态**：COMPLETED_WITH_LIMITS
**依赖**：G2-01～G2-04

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/CLAIMS.md C3 与统一指标；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～4、7、10 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md G2 与资源字段；
4. G2-01～G2-04 最终产物、断点和失败 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

证明独立 Safety 在正常、退化和故障条件下改善安全，且不靠持续停车、Oracle 特权或进度损失掩盖问题。

## 实现范围与边界

- stale/丢包/乱序、定位偏差、漏检/偏移、视觉退化；
- 低附着、执行器饱和、solver/模型超时和数值异常；
- 比较 Raw、Validator、Hard/Rule、QP、RATO 和 Classic；
- Oracle/Observable、正常/故障分开报告；

## 完成标准与验证

- 每个故障有开始、持续、严重度、恢复、seed 和预期动作。
- Classic+Safety 在学习模块缺失时通过真实短闭环。
- 安全/效率/舒适/检测/恢复/时延/降级率完整。
- C3、Evidence 和负结果登记通过；完成后停止。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

**COMPLETED_WITH_LIMITS**（2026-07-13，offline CPU）

### 交付摘要

- 故障矩阵：stale/丢包/乱序/定位偏差/漏检/偏移/stale candidate/NaN/vision_soft（含 start/duration/severity/seed/expected_action/recovery）
- **State lock**：stale obs 等状态硬故障禁止 ACCEPT/QP/RATO
- soft_stale 仅学习源；Classic 用硬 freshness
- Final 对全 ranked hard-legal 穷尽后再 repair
- 学习模块关闭时 Classic+Safety 对全部故障仍给出确定性决策
- Raw/Rule/HardReject/Longitudinal/RATO 统一对照
- Evidence：`docs/architecture/evidence/g2-05/`（含负结果登记；post state-lock remeasure）

### 验证

```text
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g2 -t . -v
# 98 tests OK（G2-01～G2-05 offline，含语义修复回归）
```

### 限制（诚实）

1. **无** live CARLA 短闭环（任务要求的“真实短闭环”未跑 → 非 live VERIFIED）
2. CLAIMS C3 为 offline MEASURED，**不是** live VERIFIED
3. Missed-actor 故障下 Observable-only 可能漏风险 → 负结果已登记
4. 碰撞 CV+圆包络；红灯近距速度门

恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态；若需 live 短闭环，先 `sdf sim preflight`。

