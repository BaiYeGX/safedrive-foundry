# 项目当前进度

本文只记录已确认动态事实；需求与验收来自 `START_TASK.md` 和 `docs/PROJECT.md`。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前任务 | G3-04R 真实 VLA K2 |
| 状态 | `CURRENT / REPAIR_REQUIRED` |
| 已有基线 | K1 pure SimLingo VLA + constrained MPC |
| 基线证据 | `MEASURED_WITH_LIMITS` |
| 下一动作 | 完成真实 K2，验证后停止 |
| 禁止自动开始 | G4A |
| 更新日期 | 2026-07-23 |

## 阶段状态

| 阶段 | 状态 | 事实 |
|---|---|---|
| G0 | `COMPLETED / FROZEN` | Windows/WSL/CARLA/ROS 2 基线 |
| G1 | `COMPLETED_WITH_LIMITS` | Runtime/Classic 背景能力 |
| G2 | `COMPLETED_WITH_LIMITS` | offline Safety foundation |
| G3 K1 | `MEASURED_WITH_LIMITS` | 真模型 path/speed → MPC → CARLA |
| G3 K2 | `REPAIR_REQUIRED` | 接口存在，候选坍塌/运动学不一致 |
| G4/G5 | `PENDING` | 尚无 World 效果证据 |

## 已确认 K1 能力

- SimLingo/InternVL2-1B 真实 CUDA forward；
- 官方 1024×512 camera/JPEG/crop/coarse-target 契约；
- VLA 原生 path 与 speed head；
- PathManager 连续路径接入；
- constrained MPC 实际执行；
- DX12 解决当前已测组合的 D3D11 forward/hang 问题；
- 多轮短测/长测、碰撞 episode、轨迹和控制证据已保存；
- 高速、复杂路口、停死和 raw path 波动仍有限制；
- 没有实车或道路安全结论。

详情见 `docs/G3_BASELINE.md`，原始历史见 `docs/EVIDENCE.md`。

## 当前 K2 缺口

1. 两候选空间位置相同；
2. 只改速度字段，没有重新时间参数化位置；
3. 运动学字段不完全一致；
4. oracle best-of-K 无法有效区分；
5. candidate 0/1 强制执行尚未验收。

因此不得开始 G4/G5，不得声称 World-ready。

## 当前路线决策

```text
real K2
→ 6–8 场景 paired G4A + oracle
→ World data/baselines
→ 4M–8M World-V0
→ VLA-Top1 vs VLA+World vs Oracle
→ 条件式 G6
→ 核心 Evidence
```

- World 必做，增益可负；
- G6 只在 VLA 候选质量限制 World 时做；
- Safety、G4B 和 Agent 不阻塞核心；
- 核心标签为 `VLA_WORLD_RESEARCH_COMPLETE`。

## 2026-07-23 仓库收口

- 活动文档合并为根目录 5 份入口文档和 `docs/` 少数权威文档；
- 原 48 份微任务、重复架构/愿景/审计/日志和历史 Evidence 移入本机 archive；
- 根目录临时实验脚本、安装压缩包和可重建运行输出移入 archive；
- archive 内容保留可恢复，不作为新需求来源；
- 当前开发入口保持 G3-04R，不因整理改变模型状态。

## 最近验证

文档路线统一前已实际运行：

旧体系归档前，48 份任务目录一致性检查为 `PASS 48/48`，维护测试为
`PASS 19/19`。

上述任务目录检查器随后随旧 48-task 体系归档，不再是活动验证入口。

仓库收口后实际验证：

```text
13 份活动项目 Markdown 相对链接检查
PASS: missing=0

python3 -m unittest discover -s tests/g1 -t . -v
PASS: 94/94

python3 -m unittest discover -s tests/g2 -t . -v
PASS: 111/111

/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v
OK: ran=154, passed=152, skipped=2（显式 live/GPU 开关）

bash -n scripts/maintenance/local_env_check.sh
python3 -m compileall -q safedrive_foundry
git diff --check
PASS
```

本次未运行 CARLA live 或真实模型 forward；不得据此新增 K2/World 效果结论。

## 恢复命令

```text
读取 START_TASK.md，开始当前任务。
```
