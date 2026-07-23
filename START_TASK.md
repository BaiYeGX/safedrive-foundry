# 当前唯一任务：G3-04R 真实 VLA K2

**状态**：`CURRENT / REPAIR_REQUIRED`

**停止点**：完成并验证真实 K2 后停止；不得自动启动 G4A。

## 1. 目标

把现有“接口存在但两候选坍塌、只改速度且运动学不一致”的 V1 K2 修成：

- 同一 Observation 的 candidate 0/1；
- `K=2/T=10/dt=0.25s/horizon=2.5s`；
- 可执行、可区分、运动学一致；
- 可分别强制进入同一 PathManager/MPC；
- 概率、margin、collapse 和谱系可查询；
- 不破坏现有 K1 pure VLA+MPC 基线。

## 2. 启动读取

只读取：

1. `AGENTS.md`；
2. `PROGRESS.md`；
3. `docs/PROJECT.md` 第 2–4 节；
4. `docs/VLA.md`；
5. `docs/RESOURCES.md`；
6. 与 V1 residual/K2/candidate adapter 直接相关的代码和测试。

不读取 archive 作为需求，不启动 CARLA，除非离线 K2 已通过且任务确实需要 live smoke。

## 3. 已知根因

旧实现：

1. `_apply_residual` 的 lateral bias 为 0；
2. candidate 空间位置相同；
3. 只改 speed，没有重新积分/时间参数化位置；
4. `x/y/yaw/v/a/kappa` 不完全一致；
5. 位置型 oracle 无法区分候选。

## 4. 最短允许实现

首版允许：

```text
k0 = upstream nominal
k1 = conservative speed/time branch
```

但必须沿路径重新时间参数化，使相同时刻的 `x/y/v/a` 真正不同，并重新计算一致的
yaw/kappa。若只有纵向差异，必须在 manifest 明确写出，不能伪装空间双路径。

本任务不要求：

- 空间双头或复杂 residual；
- K4/3s；
- VLA-V2 FAST/REASON；
- LoRA；
- World；
- Safety live；
- G4 场景基础设施。

若真实 conservative 分支不能在现有接口内成立，再停止并报告是否需要空间 residual，
不得静默扩架构。

## 5. 允许修改

```text
safedrive_foundry/driving_vla/
safedrive_foundry/config/vla/
tests/g3/
docs/VLA.md
PROGRESS.md
START_TASK.md（只更新断点/状态）
```

需要修改公共 schema、PathManager/MPC 或范围外目录时，先证明必要性；不得计划外重构。

## 6. 必做测试

至少新增/更新：

1. K2 shape/time/identity；
2. fixed input determinism；
3. candidate 0/1 差异度；
4. 位置、速度、加速度、yaw、kappa 一致性；
5. collapse detector；
6. 强制 candidate 0/1 分别进入同一执行接口；
7. K1 baseline 回归；
8. invalid/NaN/stale rejection；
9. 最小训练或 forward smoke 无 OOM。

建议命令：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -m unittest discover -s tests/g3 -t . -v
python -m compileall -q safedrive_foundry/driving_vla
git diff --check
```

只有需要真实 CARLA 时才先运行：

```bash
python scripts/sdf.py sim preflight --json
```

## 7. 完成标准

以下全部满足才能关闭：

- 两候选不是静默复制；
- T10 每个点运动学一致；
- candidate 0/1 可分别强制执行；
- 同输入可复现；
- collapse/margin/provenance 可查询；
- K1 测试不回归；
- 无 OOM/NaN；
- 实际测试结果写入 `PROGRESS.md`；
- `git diff --check` 通过。

允许 `COMPLETED_WITH_LIMITS` 的唯一情况：K2 真实且可执行，但 G4A 尚未证明纵向
选择空间。不能用该状态掩盖坍塌或运动学错误。

## 8. 中断记录

中断时在 `PROGRESS.md` 写：

- 最后状态；
- 已完成代码；
- 失败测试原文；
- 修改文件；
- 精确恢复命令；
- 是否需要空间 residual 决策。

## 9. 固定口令

```text
读取 START_TASK.md，开始当前任务。
```
