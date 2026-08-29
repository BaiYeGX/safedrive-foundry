# 2026-08-27—29 CORA 文档收敛归档

本目录保存 H6-CORA 结题合同冻结前的活动文档快照，以及从活动 `docs/` 移出的阶段性交付/
交接材料。归档只用于历史恢复，不是后续实现、阈值或任务的来源。

## 收敛原因

2026-08-27 用户要求把活动文档收敛为唯一正确的项目定义、当前事实和下一步计划；整理在
2026-08-29 完成。旧活动
文档同时包含 H3/H4/H5 已完成阶段说明、H6 VLA75 交接路线、过期分支/环境叙事，以及
已经被反事实后果建模与可校准拒绝设计取代的前向计划。为避免后续任务混用旧目标，活动
目录改为 H0-H5 冻结事实加 H6-CORA 结题路线。

## 原始快照

`original-active/` 保存收敛前下列文件的逐字副本：

```text
AGENTS.md
README.md
START_TASK.md
ROADMAP.md
PROGRESS.md
docs/PROJECT.md
docs/HYBRID_CANDIDATES.md
docs/WORLD_MODEL.md
docs/RESOURCES.md
docs/ENVIRONMENT.md
docs/EVIDENCE.md
safedrive_foundry/README.md
```

恢复旧版本时，从 `original-active/<原路径>` 复制回原路径。恢复属于显式历史回滚，必须由
用户授权；不得由代理自动执行。

## 移出的阶段文档

| 原路径 | 归档路径 | 原因 |
|---|---|---|
| `docs/H3_DELIVERY_REPORT.md` | `historical-stage-docs/H3_DELIVERY_REPORT.md` | H3 已 VERIFIED/STOPPED，详细交付转为历史参考 |
| `docs/H4_DELIVERY_REPORT.md` | `historical-stage-docs/H4_DELIVERY_REPORT.md` | H4 已 VERIFIED/STOPPED，详细交付转为历史参考 |
| `docs/H5_ENTRY.md` | `historical-stage-docs/H5_ENTRY.md` | H5 进入检查已完成，结果已由正式 Evidence 取代 |
| `docs/H5_EXPERIMENT_MATRIX.md` | `historical-stage-docs/H5_EXPERIMENT_MATRIX.md` | H5 矩阵已执行并冻结，不能作为新计划复用 |
| `docs/H6_VLA75_HANDOFF.md` | `historical-stage-docs/H6_VLA75_HANDOFF.md` | 90% source preference / 75% usage 路线被 CORA 结题合同取代 |
| `docs/PAIRED_OUTCOMES.md` | `historical-stage-docs/PAIRED_OUTCOMES.md` | H2 阶段合同冻结；当前数据合同改由 `docs/COUNTERFACTUAL_DATA.md` 负责 |

若需要恢复某个移出文件，将 `historical-stage-docs/<文件名>` 移回表中原路径；恢复前必须
检查当前 `START_TASK.md`，不能把已消费 seed、旧 gate 或历史失败重新解释为活动任务。

## Evidence 边界

`docs/runtime-evidence/`、`generated/` 中的冻结运行和数值没有被本次整理改写或删除。
历史 H2-H6 Evidence 仍按原 config、schema、seed 和 gate 解释。
