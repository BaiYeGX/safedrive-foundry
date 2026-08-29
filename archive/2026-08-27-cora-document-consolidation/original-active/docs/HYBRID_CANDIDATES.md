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

实现使用严格 `safedrive.trajectory_canonicalizer.v2`：原生路径退化、时域/弧长覆盖不足、
非有限数、坐标未知或需要外推时直接失败，不再生成直线替代。Classic 输入原生 timed
trajectory；VLA 保留真实 forward 的 20 点空间路径，再只 canonicalize 一次。

## 4. provenance

候选最少记录：source、candidate id、observation id、generator model/config hash、route
revision、raw/canonical hash、generation latency、freshness、Guard verdict 与 reject reasons。
World 选中后还要记录 selected id、executed id 和任何 Safety fallback id。

H1 的公开 source 为 `expert | vla`，在兼容的 Safety v1 合同中分别映射为
`CandidateSource.CLASSIC | CandidateSource.VLA_FAST`；完整 H1 provenance 同时保存在包装
合同与 `dynamics_meta`。Classic generator hash 是冻结 Frenet 配置 hash；VLA generator
hash 组合 model id、checkpoint SHA256 与 Hydra 配置 SHA256；route revision 是规范化
route polyline SHA256。

## 5. Guard 顺序与三态结果

每条候选独立检查：

1. schema、finite、时间与坐标；
2. freshness 与 observation binding；
3. route/lane corridor 与导航合法性；
4. curvature、速度、加速度、jerk 与可跟踪性；
5. 静态/当前可观测碰撞约束；
6. controller feasibility 与最小执行时域。

H6 起 Guard 给三种结果：

- `PASS`：合同干净，正常进入 World；
- `REVIEW`：候选基本可执行，但道路/规则/碰撞预测/动力学/控制检查处于边界，仍进入
  World，由 World 结合场景、效果和可信度综合比较；
- `REJECT`：坏数据或绑定、严重路线/道路偏离、非有限数/跳点、明显迫近碰撞或不可执行
  控制合同，World 不得接收或复活。有限的速度/加速度/jerk/曲率越界改为 `REVIEW`，
  交给 World 排序并在最终 Safety 中修复、重验。

Guard 不使用 rollout future 或 Oracle。`REVIEW` 不是 Safety 通过证书：World 排序后最终
候选仍必须经过 Safety 的完整硬检查。

候选生成 wall latency 的硬门为 `2.5 s`；它与 Safety 既有 `0.25 s` simulation freshness
门独立。某阶段硬失败后不再运行后续可能不安全的检查，最后一阶段使用隔离的
`ControlLoop` dry-run 验证最小执行时域与控制可行性。碰撞检查使用车辆有向矩形并把
actor 状态预测到每个候选点的真实相对时间；红灯检查判断是否能在停止线前停车或是否
真正越线，不再因为一条正在合理制动的轨迹前段速度较高就直接拒绝。

## 6. set-level 规则

- 0 eligible（`PASS/REVIEW`）：Safety 回退；
- 1 eligible：直接进入 Safety，World defer；
- 2 eligible：计算逐点位置与速度差异；H1 中 World 始终 defer；
- `max position delta <= 0.5 m` 且 `RMS speed delta <= 0.5 m/s` 时标记
  `NO_SELECTION_SPACE`；
- 两条 eligible 无论是否近重复，H1 都复用冻结 Safety soft score，并以 Classic、candidate id
  稳定破平；
- slot 顺序每次可置换，语义只由 candidate id/provenance 确定。

H6 的 World v3 即使两条轨迹近重复也实际打分，避免旧 `NO_SELECTION_SPACE` 直接绕过
World；旧 World/H1 仍保留冻结行为。H6 router 把所有 eligible 候选按 World 顺序交给
Safety：首选 VLA 若硬检查失败，先做一次有边界且最终重验的修复，仍失败则检查同一 tick
Expert；两者都失败才进入 MRM。控制绑定必须能解析
`selected id → final/repair id → executed id → applied id`，不存在无记录的 first-available
回退。

H6 还在 generator 侧做两项不改变来源独立性的修复：VLA 使用意图保持运动学滤波，把
`x/y/yaw/kappa/v/a` 一致重算并保持在 Safety 数值内；Classic 主规划失败或只返回短前缀时，
生成带明确 `classic-bounded-stop` provenance 的完整时域受限停车候选。两者都不是从另一
来源复制轨迹。

## 7. H1 验收

- 离线 contract/unit tests 覆盖两个 source、坐标/时间、拒绝理由和 set-level 路由；
- fake runtime 证明 nominal VLA 仍保留 20 点原生路径并 canonicalize 为固定合同；
- CARLA smoke 证明同帧两来源、逐候选 Guard、selected/executed id 连贯；
- 不产生训练数据、不启用 Oracle、不启动 World。

实现入口位于 `driving_vla.hybrid`，live smoke 位于 `scripts/h1_hybrid_smoke.py`。H1 已在
Town03 完成真实 smoke 并停止；该 Evidence 只验证候选与执行合同，不是 H2 训练数据。
H2 已由后续任务单独授权，其 paired 执行合同见 [PAIRED_OUTCOMES](PAIRED_OUTCOMES.md)。
