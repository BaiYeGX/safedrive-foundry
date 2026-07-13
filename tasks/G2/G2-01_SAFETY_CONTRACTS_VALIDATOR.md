# G2-01：独立 Safety 契约、Validator 与状态机

**状态**：COMPLETED
**依赖**：G1-09

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/EXECUTION_ARCHITECTURE.md 第 2～8 节；
2. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 1～4、7 节；
3. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1、2、5、7、8 节；
4. G1-09 最终运行时接口、Observable 数据契约、断点和 Evidence；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

冻结 PolicyCandidateSet、SafetyDecision、SafetyEvent、FallbackRequest 与组件可用性契约；实现不可绕过、可脱离所有学习模块运行的 Validator 和 NORMAL/DEGRADED/MINIMAL_RISK/EMERGENCY 状态机。

## 实现范围与边界

- 数值/时间/freshness/schema、道路、动力学、碰撞、规则和可跟踪性检查；
- 50Hz 状态检查与候选更新时完整检查；频率及时延均须实测；
- Oracle 只用于离线标签/评价，运行时仅使用 Observable 输入；
- 状态 debounce、最短驻留、恢复滞回和不可静默吞错日志；

## 完成标准与验证

- 非法、NaN、过期、乱序、缺字段和极端轨迹确定性拒绝。
- 属性测试、G1 真实轨迹回放和故意违规注入覆盖边界。
- 学习模块全失效时 Validator、Classic 回退和 Emergency 仍工作。
- 保存 schema/hash、覆盖率、P50/P95/P99、deadline miss 和失败样例。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `safety_kernel/contracts、validator、state_machine、相关 ROS adapter、config、tests/g2、docs/Evidence` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

| 字段 | 值 |
|---|---|
| 时间 | 2026-07-13 |
| 状态 | **COMPLETED**（offline CPU regression；任务范围内已尽量做满） |
| 工作分支 | `grok/g2-01-safety-contracts` |
| Workload profile | `regression`（offline CPU；无 CARLA / 无 GPU） |
| 实现 | `safedrive_foundry/safety_kernel/` |
| 关键能力 | contracts + PREFILTER/FINAL validator + state machine + `SafetyKernel` facade + oracle offline + ROS msg-shaped adapters + metrics |
| 配置 | `safedrive_foundry/config/safety_kernel/baseline.toml` |
| 测试 | `tests/g2/` **39/39 OK** |
| Evidence | `docs/architecture/evidence/g2-01/{summary,manifest,README}.json|md` |
| 契约 | `safedrive.safety.contracts.v1` |
| G1 回放 | follow (`g1-04`) + hybrid detour (`g1-05`) |
| 明确不做（后续任务） | live CARLA、QP 修复 (G2-02)、RATO-SCP (G2-03)、仲裁 Shadow (G2-04)、故障阶段验收 (G2-05) |
| 恢复 | 已完成；勿重做实现。下一任务仅当用户启动 G2-02 |

### 验证命令（已执行）

```text
PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g2 -t . -v
python3 -m compileall -q safedrive_foundry/safety_kernel
```

结果：39/39 通过；evidence 由 latency 测试写入。
