# 当前唯一任务：H2 Paired Outcomes

## 状态

```text
H0 route consolidation = VERIFIED / STOPPED
H1 hybrid candidate contract = VERIFIED / STOPPED
H2 paired outcomes = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
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

状态冻结为 `H2 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。在真实 CARLA 0.9.16、
RTX 4080/CUDA 和 Town01 → Town03 → Town05 冷重启序列上完成 restart smoke、15-anchor
pilot、冻结 120-anchor manifest、完整 paired rollout、离线 Oracle、permutation、字段隔离、
执行绑定和 artifact audit。H3 仍为 `NOT_AUTHORIZED`。

最终 dataset：`h2-gatepass-20260813-routefix`（本机 Git-ignored）。物理 manifest payload
SHA256 为 `6e74a789647182d9333cd99a69305bc2700a95216ebc7f34d2af21024a6d48ed`，store manifest
payload SHA256 为 `22d11961c74509843a1df6ea453794fad2519fcc42077540c33ce46e9f3c3524`，配置
SHA256 为 `70996b2b2a0d88cd02c210e75206cc1be1f189fae249979d14c417c866092043`。最终 Evidence 位于
`docs/runtime-evidence/h2/h2-gatepass-20260813-routefix/final-delivery.json`，Evidence SHA256 为
`5e1ca730978b9c4ff54ee001f76dbe871348927979d853f19c0b6a925453d8b8`。

完整门实际指标：120 terminal、108 eligible/distinct、108 valid；Town01/Town03/Town05
分别 36/34/38；family `cut_in/free_flow/red_light_hold/slow_lead/stopped_lead` 分别
21/21/21/24/21；weather ClearNoon/CloudyNoon 为 56/52；83 decisive；Expert/VLA wins
为 51/32；slot Expert=0 比例 `0.5`；swap/permutation 通过；轨迹 hash 保持 100%；
source-only baseline `0.6144578313`；整卡 GPU 峰值 8.3720703125 GiB；dataset
1,480,172,014 bytes。全部冻结数据门通过。

本轮定点修复包括 route-relative traffic-light stop line、有限时域安全停止前缀、moving
family 闭环预滚、cut-in spawn probe cleanup 和 freshness-only trajectory stamp refresh；
没有在线 Oracle、World、训练、补挑场景、第二 forward、第二 tick master 或 `load_world()`。
全量串行测试为 `329 tests: 328 passed, 1 skipped, 0 failed`；compileall、文档链接和
`git diff --check` 通过。
