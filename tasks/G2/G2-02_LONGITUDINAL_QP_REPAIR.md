# G2-02：RATO 纵向 QP 最小修复基线

**状态**：COMPLETED
**依赖**：G2-01

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 2、3、7 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、4、7、8 节；
3. docs/project/decisions/CLASSIC_ALGORITHM_OPTIMIZATION.md 第 3～4 节；
4. G2-01 最终接口、断点和 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

实现默认启用、可解释且低成本的纵向轨迹修复，只调整速度/加速度时间序列，处理跟车、红灯、切入和停车。

## 实现范围与边界

- OSQP 速度、加速度、jerk、停止线、前车/占用和 warm start；
- 统一 Raw/Rule/HardReject/Longitudinal 接口；
- 冻结 slack 上限、进度与舒适代价，防止持续零速伪安全；
- 超时、不可行、数值错误和 stale 输入进入 G2 回退契约；

## 完成标准与验证

- 红灯、急刹、切入和跟车覆盖正常/边界/无解。
- 报告安全裕度、修改量、无故停车、进度、舒适、solver 状态和时延。
- 同输入下优于或诚实对照 Rule/HardReject。
- 配置、solver trace、测试和负结果可复现。

## 允许修改与交付物

实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根。允许修改当前能力直接需要的实现、配置、测试、验证、Registry、Evidence、本任务和 PROGRESS.md；不得提前实现下一任务。涉及真实 CARLA 时先执行 sdf sim preflight。

## 断点记录

| 字段 | 值 |
|---|---|
| 时间 | 2026-07-13 |
| 状态 | **COMPLETED**（offline CPU regression；任务范围内已做满） |
| 工作分支 | `grok/g2-01-safety-contracts`（G2-02 同分支实现） |
| Workload profile | `regression`（offline CPU；无 CARLA / 无 GPU） |
| 实现 | `safedrive_foundry/safety_kernel/repair/` + Kernel 接入 |
| 关键能力 | 纵向 QP（v/a/jerk、停止线、前车、warm start、slack 上限、进度/舒适代价）；统一 Raw/Rule/HardReject/Longitudinal；超时/不可行/stale 回退契约 |
| 配置 | `safedrive_foundry/config/safety_kernel/baseline.toml` `[qp]` |
| 测试 | `tests/g2/` **55/55 OK**（含 G2-01 + G2-02） |
| Evidence | `docs/architecture/evidence/g2-02/{summary,manifest,README}` |
| Solver | OSQP-form reduced-space；优先 `osqp`（`tools/wsl_site_packages` 可选安装），回退 `scipy_slsqp`/`numpy_admm` |
| 明确不做（后续） | RATO-SCP (G2-03)、仲裁 Shadow (G2-04)、故障阶段验收 (G2-05)、live CARLA |
| 恢复 | 已完成；勿重做实现。下一任务仅当用户启动 G2-03 |

### 验证命令（已执行）

```text
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g2 -t . -v
python3 -m compileall -q safedrive_foundry/safety_kernel
```

结果：55/55 通过；evidence 由 `test_g2_02_latency_evidence` 写入。  
可选 OSQP：`bash scripts/install_optional_osqp.sh`（写入 gitignored `tools/wsl_site_packages`）。

