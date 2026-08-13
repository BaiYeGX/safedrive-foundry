# 当前唯一任务：H2 Paired Outcomes

## 状态

```text
H0 route consolidation = VERIFIED / STOPPED
H1 hybrid candidate contract = VERIFIED / STOPPED
H2 paired outcomes = COMPLETED / VERIFIED / GATE_FAILED / STOPPED
H3 World training/runtime = NOT_AUTHORIZED
Online Oracle = PROHIBITED
```

## 目标

在 Town01、Town03、Town05 上物化并冻结 120 个 anchor。每个 anchor 只从同一个
observable capture 生成一次 Classic Expert 和一次 nominal SimLingo 候选；只有两条候选
均通过 H1 Guard 且为 `DISTINCT` 时，才在可比较 reset 后分别执行 2.5 秒，保存真实 paired
outcome，并由隔离的离线 Oracle 生成 source-neutral 标签。

## 允许范围

- H2 pair/branch/outcome 合同、Parquet store、内容寻址图像、压缩 timeline 与 manifest；
- H3 observable-only feature view 和数据泄漏审计，但不实现或训练 H3；
- ScenarioRuntime 的稀疏事件读取、同 tick 多 actor 控制与可验证初态；
- `sdf sim restart` 的单进程、异步、tick-owner-free 安全冷重启；
- paired capture/branch collector、离线 Oracle/audit、H2 测试、文档和本机 Evidence。

禁止在线 Oracle、World、训练、候选重规划/第二次 VLA forward、`client.load_world()`、第二
tick master、补挑场景或采集后修改矩阵和标签阈值。

## 冻结矩阵与执行合同

```text
maps     = Town01, Town03, Town05
families = free_flow, slow_lead, stopped_lead, cut_in, red_light_hold
seeds    = 0, 1, 2, 3
weather  = ClearNoon, CloudyNoon
total    = 120 anchors
pilot    = each map × each family × seed=0 × ClearNoon = 15 anchors

history  = 1.0 s at 20 Hz
branch   = 2.5 s / 50 ticks at 20 Hz
buffer stamp refresh = every 0.2 s; trajectory points/hash unchanged
```

Reset 比较硬门：actor roles、route、weather、light 和 script hash 完全相同；位置差
`<=0.05 m`、yaw 差 `<=0.5 deg`、速度差 `<=0.10 m/s`。

## 冻结离线 Oracle

1. reset、Safety execution、50 ticks 或 cleanup 不完整：`INVALID_PAIR`；
2. collision、red-light violation 或 off-corridor 超过 0.25 秒为 hard unsafe；
3. 仅一条 unsafe：安全分支胜；两条都 unsafe：`UNRESOLVED`；
4. 两条都安全且 completion 不同：完成者胜；
5. 否则进度差至少 1.0 米：进度高者胜；
6. 否则 jerk RMS 差至少 1.0 m/s³，且舒适侧进度不落后超过 0.25 米：更舒适者胜；
7. 其余为 `TIE`。

Oracle 只输出 candidate id/hash；source、slot 和 branch order 置换不得改变理由或物理赢家。

## 验收与停止点

Pilot 扩大门：15/15 terminal、至少 8 个双 PASS `DISTINCT`、至少 7 个 reset comparable
完整 pair，且无 orphan、first-available、第二 forward 或第二 tick master。

完整门：120 个 anchor 全可解释；双 PASS `DISTINCT >=80`；valid pairs `>=72`；每地图
`>=18`、每 family `>=8`、每 weather `>=30`；decisive `>=24` 且不少于 valid 的 25%；
Expert/VLA 各占 decisive 至少 20% 且各至少 5；slot 分布 40%–60%；swap invariance 100%；
source-only majority baseline `<=80%`；执行绑定前后 trajectory hash 100% 不变；artifact
hash 100% 通过；数据 `<=15 GB`；whole-GPU peak 不超过 14.5 GB admission target。

- 全门通过：`H2 VERIFIED / GATE_PASSED / STOPPED`；
- 固定矩阵完成但数据门失败：`H2 VERIFIED / GATE_FAILED / STOPPED`，关闭 H3；
- CARLA/CUDA/模型/冷重启阻塞使矩阵未完成：`H2 IMPLEMENTED / NOT_VERIFIED / BLOCKED`。

无论结果如何，本轮不得自动进入 H3。固定矩阵已完成但数据门失败时保留完整负结果，
关闭 H3，不补挑场景、不修改阈值。

## 本轮最终停止记录

状态冻结为 `H2 COMPLETED / VERIFIED / GATE_FAILED / STOPPED`。真实 CARLA/CUDA 上已完成
三地图冷重启、restart smoke、15-anchor pilot 和完整 120-anchor 固定矩阵；所有 120 个
anchor 均有 terminal pair 记录并完成离线 Oracle、manifest/hash 和 permutation audit。
Pilot 门通过后才扩大矩阵；完整数据门按冻结阈值失败，保留负结果并关闭 H3。

最终 dataset：`h2-final-20260813-scenariov2-cleanup`（本机 Git-ignored）。物理 manifest
SHA256 为 `f89a2f8a039144b089ae27c2584aa112a3d35d061d6a22cc6d12359fabd11a9f`，store manifest
SHA256 为 `b63f4f63de12aa0a84762c398fecc7bb78ffbcc4d59f22b3552da1ba2c82727e`，配置 SHA256 为
`9ff604d7d3d64122af76b41df58533d722de3f27c8d0c9dbe7e89e6a7552ceaa`。最终 audit 位于
`docs/runtime-evidence/h2/h2-final-20260813-scenariov2-cleanup/offline-label-audit-full.json`；最终
terminal summary 为 `final-delivery.json`，Evidence SHA256 为
`6ceb3c74aede8c3895cbefb302a10f9ec98f0b5425af616fe1e9da77ec163f23`。

完整门实际指标：120 terminal、58 valid/distinct（要求 ≥72）、Town01/Town03/Town05
分别 21/23/14（每张 ≥18）、family `cut_in/free_flow/red_light_hold/slow_lead/stopped_lead`
分别 14/12/4/13/15（每个 ≥8）、weather ClearNoon/CloudyNoon 为 33/25（每种 ≥30）、
58 decisive、Expert/VLA wins 为 58/0（各至少 5 且占 decisive ≥20%）、slot Expert=0 比例
`0.4827586`、swap/source/branch permutation 通过、轨迹 hash 保持 100%、source-only baseline
100%、整卡 GPU 峰值 8.3945 GiB、dataset 1.444 GB。manifest/hash、执行绑定和 cleanup
均通过；失败项是 `eligible_distinct`、`valid_pairs`、`per_map_valid_pairs`、
`per_family_valid_pairs`、`per_weather_valid_pairs`、`vla_wins`、`source_only_majority`。

QP 时序修复已固化：OSQP algebra 只探测一次并冻结为显式 backend；全量串行测试现为
`327 tests: 326 passed, 1 skipped, 0 failed`。新增 label 批量写入的延迟 manifest 扫描
选项，未改变不可变 artifact 或 manifest 最终校验。CARLA、模型、CUDA 和冷重启均已真实
运行；没有在线 Oracle、World、训练、补采或第二 tick master。H3 继续保持
`NOT_AUTHORIZED / CLOSED_AFTER_H2_GATE_FAILURE`。
