# Evidence 与归档索引

活动目录只保留当前运行入口和精简结论。历史安装记录、任务文档、运行日志、视频、
轨迹和原始 Evidence 均保存在：

```text
archive/2026-07-23-repository-consolidation/
```

该目录为本机可恢复归档，默认被 Git 忽略；Git 中已追踪过的历史文件仍可从版本历史
恢复。归档内容不得改写成新的测量结果。

## 1. 归档分区

| 路径 | 内容 |
|---|---|
| `docs/` | 被新核心文档替代的原架构、环境、设计、审计与日志 |
| `tasks/` | 原 G0–G8 48 份微任务文件 |
| `evidence/` | G0–G3 原始 Evidence 与运行产物 |
| `temp/` | 根目录一次性分析/运行/检查脚本 |
| `runtime_outputs/` | 可重建的 build/install/log 和旧本地 venv |
| `installers/` | CARLA 与 AdditionalMaps 安装压缩包 |
| `tooling/` | 非核心协作/任务目录检查工具 |

归档内 `README.md` 记录原路径到新位置的映射。

## 2. 当前证据状态

| 能力 | 状态 | 说明 |
|---|---|---|
| G0 environment | `COMPLETED / FROZEN` | 原始报告已归档 |
| G1 classic/runtime | `COMPLETED_WITH_LIMITS` | 背景能力 |
| G2 Safety offline | `COMPLETED_WITH_LIMITS` | optional engineering foundation |
| G3 K1 pure VLA | `MEASURED_WITH_LIMITS` | 原始 run 已归档 |
| G3 real K2 | `REPAIR_REQUIRED` | 当前任务 |
| G4/G5 | `PENDING` | 无新 Evidence |

## 3. 新证据规则

临时运行输出先写：

```text
docs/runtime-evidence/<task-or-run>/
```

任务关闭后：

1. 生成 manifest/hash；
2. 在 `PROGRESS.md` 登记结论和限制；
3. 把冻结包移动到 `archive/evidence/` 或新的日期归档；
4. 活动目录只保留索引，不长期堆放大文件。

禁止选择性删除失败 run、改写原始 JSON 或把未验证结果写成 VERIFIED。
