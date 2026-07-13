# G2-03：受限二维 RATO-SCP 与可行走廊

**状态**：COMPLETED
**依赖**：G2-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 2、3、7 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、7、8 节；
3. docs/project/decisions/CLASSIC_ALGORITHM_OPTIMIZATION.md 第 3～4 节；
4. G2-01、G2-02 最终接口、solver 配置和 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

仅在纵向 QP 无法兼顾安全与进度且存在合法横向走廊时，执行有限迭代的二维最小修复；它不是第二套局部规划器。

## 实现范围与边界

- Frenet/局部坐标凸走廊、信赖域、有限 SCP 迭代和 warm start；
- 冻结硬约束、slack 上限、触发条件和时间预算；
- 静态绕障、狭窄通道、换道冲突与不可行检测；
- 超时、振荡、不可行直接退回 QP/Classic/Minimal Risk；

## 完成标准与验证

- 触发条件外不得运行，关闭二级后 G2-02 独立可复现。
- 报告迭代、约束、slack、修改量、进度、舒适和 P50/P95/P99。
- 与纵向 QP 固定场景/seed/deadline 公平比较。
- 无净收益时默认关闭并保留负结果。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

**COMPLETED**（2026-07-13，offline CPU regression）

### 交付摘要

- Frenet 凸走廊 + 有限 SCP 迭代 + 信赖域 + warm start（`repair/corridor.py`、`repair/rato_scp.py`）
- 触发门控：合法横向走廊 +（QP 失败或 progress < `min_qp_progress_to_skip`）；纯红灯不进 RATO
- Kernel 级联：QP → 条件 RATO；`rato.enabled=false` 时 G2-02 路径独立可复现
- 场景：静态绕障、狭窄/偏移、换道冲突、阻塞不可行、stale/disabled
- Evidence：`docs/architecture/evidence/g2-03/`（P50/P95/P99、fair QP 对照、资源字段）

### 验证

```text
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g2 -t . -v
# 71/71 OK
```

### 限制

- offline CPU，非 CARLA live 50Hz VERIFIED
- 仲裁/Shadow/故障注入属 G2-04/G2-05

恢复时先复核启动读取清单、直接依赖接口、版本、冻结协议和外部状态。

