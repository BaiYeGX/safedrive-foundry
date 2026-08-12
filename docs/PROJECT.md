# H 项目定义

## 1. 研究问题

在单机 CARLA 软件在环中，比较两个独立、可执行的候选来源：Classic Expert 与
nominal VLA。先逐候选执行硬合同检查，再让 candidate-conditioned World 预测相对结果
并排序。研究目标不是证明 World “会开车”，而是回答：

> 在候选、Safety、控制器和场景相同的条件下，World 排序是否比冻结的非学习 selector
> 带来可复现的闭环净收益？

## 2. 固定系统边界

```text
CARLA/ROS observable snapshot
        ├── Classic Expert ── candidate_expert ── Guard ──┐
        └── nominal VLA ───── candidate_vla ───── Guard ──┤
                                                          ▼
                                            World rank or defer
                                                          ▼
                                            Safety → MPC/PID
```

- 两个 generator 只读同一决策时刻的可观测输入。
- candidate 必须统一坐标、采样间隔、时域与 provenance。
- Guard 对候选逐条运行，发生在 World 之前。
- World 共享同一 scorer 参数逐候选计算，不依赖槽位顺序。
- World 可排序或 defer，不生成候选、不修改硬安全结果。
- Safety 与 MPC/PID 是执行边界，学习模块不直接输出 throttle/brake/steer。
- 仅限 CARLA SIL；不声明实车安全性。

## 3. 数据隔离

| 数据 | 可用于 generator | 可用于 World | 可用于 Oracle | 可用于回归 |
|---|---:|---:|---:|---:|
| 当前图像/ego/route/history | 是 | 是 | 是 | 是 |
| 当前可观测 actor/light 状态 | 是 | 是 | 是 | 是 |
| 候选轨迹与 Guard 结果 | 不回灌 | 是 | 是 | 是 |
| rollout actor future/碰撞/最终进度 | 否 | 仅训练标签 | 是 | 是 |
| 场景答案、Regression 注入、测试标签 | 否 | 否 | 否 | 是 |

Oracle 与 online package 必须物理隔离导入；测试 split 在 H4 前锁定。

## 4. 候选合同

每个候选至少包含：

```text
candidate_id
source = expert | vla
observation_id / frame_id / simulation_time
model_or_config_hash
coordinate_frame
T=10, dt=0.25 s, horizon=2.5 s
trajectory[x,y,yaw,v,a,kappa]
generation_latency
guard_status / reject_reasons
```

source id 只用于 provenance 和分层报告；World 训练默认不把 source id 作为捷径特征。

## 5. 选择逻辑

1. 两候选都未通过 Guard：调用现有 Safety 回退。
2. 仅一个通过：直接把该候选交给 Safety，World 不制造选择。
3. 两个通过且 World defer：使用冻结的非学习 selector。
4. 两个通过且 World 有效：按 World score 排序，再交给 Safety。
5. Safety 可拒绝或降级最终选择；World 无权覆盖。

## 6. 成功口径

H1/H2 先证明真实选择空间：

- 候选几何/速度差异可测且不是数值噪声；
- 两来源均有通过 Guard 并胜出的样本；
- paired reset、actor/light phase 与执行控制可比；
- label 不受 source/slot 泄漏支配。

H3/H4 再证明 World 学到了 action-conditioned 差异：

- 优于 no-action、CV/CTRV、手写 reward 与简单 MLP；
- candidate swap 后预测随 trajectory 而不是槽位移动；
- calibration、regret、defer 曲线和跨 split 泛化达标；
- 单机 16GB 资源与在线 deadline 可接受。

H5 最终以闭环 on/off 判定：安全指标不退化，路线完成/进度或效率有可复现净收益，
且尾延迟、显存、回退率没有抵消收益。未满足即记录负结果，不能只凭离线 accuracy
宣称 World 有效。

## 7. 非目标

- 不重训候选生成 head；
- 不让 World 直接规划或控制底盘；
- 不用像素视频生成作为首个 World 目标；
- 不用 Oracle/Regression 进入训练输入；
- 不为得到正结果修改冻结测试门；
- 不把 archive 中的旧数字改名为 H 指标。

## 8. Evidence 合同

每次正式运行保存 config、commit/worktree hash、split、seed、输入/候选/选择/执行链、
Guard/Safety 决策、outcome、P50/P95/P99、deadline miss 与资源。证据状态只允许
`PLANNED → IMPLEMENTED → MEASURED → VERIFIED`。失败和负收益必须与正结果同等保留。
