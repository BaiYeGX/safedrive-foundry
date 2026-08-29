# 当前唯一任务：H6 VLA 主驾闭环验收

## 状态

```text
H0 route consolidation = VERIFIED / STOPPED
H1 hybrid candidate contract = VERIFIED / STOPPED
H2 paired outcomes = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H3 World scorer development = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H4 locked evaluation = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H5 World on/off closed loop = COMPLETED / VERIFIED / GATE_FAILED / STOPPED
H6 VLA-primary redesign = IMPLEMENTED / MEASURED / NOT_VERIFIED
Online Oracle = PROHIBITED
```

H5 的正式结论仍是负结果，不能改写。H6 是用户在看到 H5 后明确授权的新目标：重新训练
World，让它在至少 90% 的正式决策时刻确实给 VLA 更高的综合评价。用户在 2026-08-27
把最终实际执行硬门修订为：VLA 至少 `75%`，Classic Expert + MRM 合计最多 `25%`。
World 原始高分门仍保持 `90%`，这样不能靠路由锁定或单候选幸存伪造“World 看好 VLA”。

## H6 已实现

- Guard 改成 `PASS / REVIEW / REJECT`：`PASS` 和 `REVIEW` 都交给 World，只有坏数据、
  严重越界、明显迫近碰撞或不可执行轨迹才硬拒绝。
- World v3 分开预测任务效果和可信/风险，并分别输出进度、完成、碰撞、红灯、越界、
  舒适性与修复成功率；正式路由只有在 VLA 综合分不低于 Expert、可信度达标且风险低于
  上限时，才把 VLA 排第一。
- Safety 按 World 顺序检查；首选 VLA 失败时先做一次有边界的修复，仍失败则使用同一拍
  Expert，两者都失败才进入 MRM。Safety 的硬碰撞、规则和状态机权限没有交给 World。
- H6 训练 seed `89/97` 与正式验收 seed `101/103` 隔离；训练只读取真实 Classic-off 与
  VLA-primary 闭环结果，不用正式验收数据。
- 正式验收同时要求：World 真正给 VLA 高分比例 `>=90%`、VLA 实际执行比例 `>=75%`、
  Classic Expert + MRM 合计 `<=25%`、
  相对纯 Classic 的不安全率增加 `<=1pp`、配对进度 bootstrap 95% 下界 `>=0`，并检查
  延迟、切换、来回抖动和完整 provenance。

## 当前实测边界

H6 已完成 24-pair 平衡训练 pilot、新 World v3 训练和一次隔离 seed `101` 的正式
12-pair pilot。正式 gate 失败，状态仍只能是 `MEASURED / NOT_VERIFIED`：

- 训练 pilot 同时让 seed `89/97` 覆盖四类场景，避免旧 12-pair pilot 把简单题全放到
  seed 89、难题全放到 seed 97。seed 89 的 VLA 实际执行为 `566/600 = 94.33%`，
  seed 97 为 `544/600 = 90.67%`；on/off 不安全运行数在两个 split 都相同；
- World v3 加入每场真实 Classic-off/VLA-primary-on 整段配对监督后，旧的“只看场景
  第一拍”校准报告 VLA 高分 `11/12 = 91.67%` 并通过 readiness；
- 隔离正式 pilot `h6-vla90-formal-pilot-20260820-v1` 证明该校准口径错误：逐 tick 的
  World 严格 VLA 高分只有 `131/600 = 21.83%`，VLA 实际执行 `285/600 = 47.50%`。
  相对 Classic 的不安全率增量为 `0`，配对进度 bootstrap 下界 `0.629m`，scorer P99
  `9.12ms` 且 0 deadline miss，但 31 次切换、5 个 ping-pong 场景也未过门；
- 正式 600 个 VLA Guard 结果为 `PASS 453 / REVIEW 147 / REJECT 0`。Safety 只产生
  18 次 Classic fallback（全部 tick 的 3%），并成功执行 42 次 RATO、26 次 QP。
  因此正式低占比的主因已经确定为 World 逐 tick 泛化/校准失败，不是 Guard 或 Safety
  大量杀掉 VLA；
- 正式 on/off 各有 `3/12` 不安全运行：Town03 free-flow 的静态物体碰撞，以及
  Town03/Town05 红灯违规。Town03 训练红灯还出现 `other_actor_id=0` 的连续地面/静态
  接触，说明坡路预滚高度或路线仍需专门修复；
- 所有真实 CARLA 测试均启用独立 20Hz 跟车视角。24-pair 训练的 48 条运行每条更新
  `285–847` 次，正式 24 条运行每条更新 `390–1113` 次，错误数均为 0。

正式失败后已把校准改成：验证集每一个 on-arm tick 负责 90% World 高分门，整段 paired
policy outcome 只负责安全/进度门，第一拍不能再冒充整段 90%。专项测试已通过；但重新
GPU 训练被工作区 `credits exhausted` 拒绝，尚无新的 tick-wise readiness 结果。

## 下一验收顺序

1. 恢复工作区 GPU/CUDA 训练能力后，只用开发 seed `89/97` 重跑 tick-wise 校准；旧模型若不能在
   600 个 on-arm 验证 tick 达到 90%，readiness 必须停止，不能再进 CARLA formal。
2. 在开发数据上修 World 的逐 tick 排名泛化和路由稳定性；同时修 Town03 坡路/静态碰撞与
   红灯停车。Guard 硬安全和 Safety 最终重验不再继续削弱。
3. seed `101` 已被本次正式失败消耗，不得用它调参后重跑冒充 held-out。若继续下一轮正式
   研究，必须由用户明确授权新的预注册 seed lineage；seed `103` 也不能与已看过的 101
   混成原 H6 full 后声称隔离。
4. 先把 acceptance/readiness 拆成两个独立阈值：World 原始高分 `>=90%`、VLA 实际执行
   `>=75%`，并升级配置/schema/hash；历史 `vla90` Evidence 不改写。
5. 只有新的 tick-wise readiness 先通过，才能跑新的 12-pair formal pilot；pilot 通过后
   才允许 108-pair full。不降低 World 90%、实际 75%、+1pp、进度下界、切换或 provenance 门。

入口：

```text
python scripts/h6_retrain.py --dataset-id <train-id> --base-world-summary <old-summary> \
  --output-dir generated/h6/<model-id> --scope pilot
python scripts/h6_run.py --dataset-id <formal-id> \
  --world-v3-summary generated/h6/<model-id>/training-summary.json --pilot
```

当前停止点是正式 pilot 已诚实失败且工作区 GPU/CUDA 训练能力不可用，阻塞新的开发校准。不得把开发态
91.67% 第一拍校准或 90.67% 强制采样执行率当作正式完成。

下一轮分析、训练、验收顺序和新对话交接见
[`docs/H6_VLA75_HANDOFF.md`](docs/H6_VLA75_HANDOFF.md)。
