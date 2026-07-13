# G2-04：候选预筛、仲裁、Shadow 与回退链

**状态**：COMPLETED
**依赖**：G2-03

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 第 2～8 节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～4、7 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、5、7、8 节；
4. G2-01～G2-03 最终接口、配置和断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

冻结 hard precheck→soft score→deterministic arbitration→final Validator→QP/RATO→Classic/Minimal Risk/Emergency→MPC 的执行顺序。

## 实现范围与边界

- Raw、Classic、Repaired 与未来 VLA/World 候选统一 adapter；
- 候选来源、版本、风险、拒绝、修复和最终动作审计；
- Classic Shadow 不争夺控制权或 tick ownership；
- 冻结模型 unavailable/timeout/OOD/overconfident 的降级接口；

## 完成标准与验证

- 正常、全拒绝、修复成败、超时和紧急时间线确定性一致。
- 学习模块关闭时 Classic+Safety 不依赖 GPU。
- Emergency 不被分数、语言或热配置覆盖。
- 报告仲裁时延、切换、停车、进度损失和回退成功率。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

**COMPLETED**（2026-07-13，offline CPU regression）

### 交付摘要

- 冻结流水线：hard precheck → soft score → rank → final Validator → QP/RATO → fallback → Shadow
- Classic Shadow 仅对比，不争夺控制权 / tick ownership
- 降级门控：unavailable / timeout / OOD / overconfident / soft-stale
- 审计：`ArbitrationRecord`（stages、ranked_ids、audits、fallback、shadow）
- Evidence：`docs/architecture/evidence/g2-04/`

### 验证

```text
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g2 -t . -v
# 含 G2-04 用例；全量 g2 见 PROGRESS
```

### 限制

- offline CPU，非 live CARLA
- 故障矩阵与阶段 C3 属 G2-05

恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。
