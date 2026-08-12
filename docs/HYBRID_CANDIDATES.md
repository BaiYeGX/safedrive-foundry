# H1 Hybrid candidates 与逐候选 Guard

## 1. 目标

H1 只做一件事：在同一 observable anchor 上得到两条独立、可执行、可追踪的候选。

| source | 实现 | 输出 |
|---|---|---|
| `expert` | Classic route/behavior planner | 一条确定性 expert trajectory |
| `vla` | `NominalVLAPolicy` 的一次真实 SimLingo forward | 一条 nominal trajectory |

不训练多模态 head，不从某条轨迹做 learned perturbation，不把 teacher/Oracle 当在线候选。

## 2. 同步输入

两来源绑定同一个 `observation_id`：CARLA frame、simulation time、ego state、route revision、
actor/light snapshot 与传感器时间戳必须一致。超过 freshness/deadline 的结果丢弃，不能
与下一帧候选拼接。

## 3. 统一轨迹合同

```text
coordinate_frame = map
T = 10
dt = 0.25 s
horizon = 2.5 s
point = [x, y, yaw, v, a, kappa]
```

canonicalization 不能改变候选语义，只允许坐标变换、按弧长/时间重采样及有限差分补齐。
每次变换记录 input hash、canonical hash、版本与误差。

## 4. provenance

候选最少记录：source、candidate id、observation id、generator model/config hash、route
revision、raw/canonical hash、generation latency、freshness、Guard verdict 与 reject reasons。
World 选中后还要记录 selected id、executed id 和任何 Safety fallback id。

## 5. Guard 顺序

每条候选独立检查：

1. schema、finite、时间与坐标；
2. freshness 与 observation binding；
3. route/lane corridor 与导航合法性；
4. curvature、速度、加速度、jerk 与可跟踪性；
5. 静态/当前可观测碰撞约束；
6. controller feasibility 与最小执行时域。

Guard 只给 `PASS` 或 `REJECT(reasons)`；World 不接收 REJECT candidate。Guard 不使用
rollout future 或 Oracle。

## 6. set-level 规则

- 0 PASS：Safety 回退；
- 1 PASS：直接进入 Safety，World defer；
- 2 PASS：计算差异与选择空间，再交给 World；
- 近重复：标记 `NO_SELECTION_SPACE` 并使用冻结 selector；
- slot 顺序每次可置换，语义只由 candidate id/provenance 确定。

## 7. H1 验收

- 离线 contract/unit tests 覆盖两个 source、坐标/时间、拒绝理由和 set-level 路由；
- fake runtime 证明 nominal VLA 仍保留 20 点原生路径并 canonicalize 为固定合同；
- CARLA smoke 证明同帧两来源、逐候选 Guard、selected/executed id 连贯；
- 不产生训练数据、不启用 Oracle、不启动 World。

H1 完成后停止，H2 必须单独授权。
