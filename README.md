# SafeDrive Foundry

SafeDrive Foundry 是单机 CARLA–ROS 2 纯软件在环自动驾驶研发平台。当前核心是：

```text
VLA 真实 K2 → World Model 动作条件软排序 → MPC/PID → CARLA
```

项目重点是 VLA 驾驶质量与 World on/off 的真实效果，不把 Classic 或独立 Safety
当作核心效果前置；二者保留为基线、标签和可选工程扩展。

## 当前状态

| 能力 | 状态 |
|---|---|
| G0 environment | `COMPLETED / FROZEN` |
| G1 classic/runtime | `COMPLETED_WITH_LIMITS` |
| G2 Safety offline | `COMPLETED_WITH_LIMITS` |
| G3 K1 pure VLA+MPC | `MEASURED_WITH_LIMITS` |
| G3 real K2 / R1 | `COMPLETED_WITH_LIMITS` |
| R2 / G4A paired oracle | **`COMPLETED_WITH_LIMITS`**（纵向；`NO_SELECTION_SPACE`） |
| R2-X Spatial/Semantic K2 | **`COMPLETED_WITH_LIMITS`**（nominal 可用；defensive availability 不可靠） |
| World / G5 | `PENDING`（不得自动启动） |

纵向 R2 已关闭；R2-X 以“可用但非 World-ready”收尾，不覆盖旧 Evidence。
R3/R4 已有实施规格，但仍未授权。详见 `START_TASK.md` / `PROGRESS.md`。

## 从这里开始

1. [当前任务](START_TASK.md)
2. [后续最短路线](ROADMAP.md)
3. [项目合同](docs/PROJECT.md)
4. [当前进度](PROGRESS.md)

实现参考：

- [R1 真实 K2 实施任务](docs/R1_REAL_K2.md)
- [R2 Paired Outcome + Oracle](docs/R2_PAIRED_ORACLE.md)
- [R2-X Spatial/Semantic K2](docs/R2X_SPATIAL_K2.md)
- [VLA/K2](docs/VLA.md)
- [World Model/G4A](docs/WORLD_MODEL.md)
- [R3/R4 World 数据与模型实施规格](docs/R3_R4_WORLD_DATA_MODEL.md)
- [本机资产与预算](docs/RESOURCES.md)
- [环境与运行](docs/ENVIRONMENT.md)
- [K1 基线](docs/G3_BASELINE.md)
- [Evidence/Archive](docs/EVIDENCE.md)

## 目录

```text
safedrive_foundry/  核心 Python/ROS 2/runtime/VLA/Safety 代码
scripts/            sdf 入口、环境维护和 smoke
tests/              单元、集成和 live runner
models/             本机模型权重（Git 忽略）
simlingo-main/      上游 VLA 代码（Git 忽略）
tools/              本机工具链（Git 忽略）
docs/               仅保留活动权威文档
archive/            本机历史归档，不是活动路线
```

## 最短环境入口

```bash
cd "/mnt/e/autonomous driving"
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py doctor
python scripts/sdf.py sim preflight --json
```

CARLA 未运行且 preflight 为 `RETRYABLE_FAILURE` 时只执行一次：

```bash
python scripts/sdf.py sim ensure --map Town03 --rhi dx12 --startup-timeout 180 --json
```

## 关键边界

- CARLA SIL ≠ 实车或道路安全证明；
- VLA/World 不直接发无约束底盘控制；
- World 只排序同一 K2，异常回到 VLA 原始 top-1；
- 核心 A/B 固定 VLA、候选、场景、seed、initial-state 和 MPC；
- Safety 启用后，学习模块不能覆盖硬约束、MRM 或 Emergency；
- 大型权重、安装包、运行 Evidence 和历史文档不放在活动源码目录。

当前执行口令：

```text
读取 START_TASK.md，开始当前任务。
```
