# Hybrid VLA–Expert 候选与 Guard 合同

## 1. 目标

同一 observable anchor 上只允许两个在线规划来源：

| source | 实现 | 系统作用 |
|---|---|---|
| `vla` | 预训练 `NominalVLAPolicy` / SimLingo 一次真实 forward | 视觉语义与 language-action 先验的 nominal proposal |
| `expert` | Classic route/behavior/Frenet-ST planner | 几何、规则、动力学和确定性 proposal |

二者互补但平权：系统不预设 VLA 必须胜出，也不允许 Classic outcome 被当作 VLA 标签。
在线学习模块不得从一条轨迹做 perturbation、复制或重命名来伪造第二候选。offline-only
intervention 只用于反事实数据和 benchmark，必须有独立 provenance，不能进入 live set。

本文固定三种不同身份：`candidate` 是 generator 原始 proposal，`selected` 是被请求交给 Safety
的候选，`executable` 是 Safety 接受或 repair 后批准的轨迹。三者不能用同一个 source 字段
相互覆盖。

## 2. 同锚点输入

两来源绑定同一个：

```text
observation_id
CARLA frame / simulation time
ego state and observable history
route revision / navigation command
actor and traffic-light snapshot
sensor timestamps
```

候选不得跨帧拼接。超出 generation deadline、simulation freshness 或 observation binding
失败的候选无资格进入 World。

## 3. 统一轨迹合同

```text
coordinate_frame = map
T = 10
dt = 0.25 s
horizon = 2.5 s
point = [x, y, yaw, v, a, kappa]
```

canonicalization 只允许坐标变换、重采样和由同一轨迹重算动力学量，不能改变候选语义或
补造一条直线替代。每次变换记录：

```text
raw trajectory hash
canonical trajectory hash
canonicalizer version
model/config/checkpoint hash
route revision hash
generation latency
```

VLA 保留真实 forward 的 raw route/path points 与 speed head，再经意图保持的运动学平滑；
Classic 主规划失败时只允许生成自身规则定义的有 provenance 受限停车候选，不能复制 VLA。

## 4. Guard 三态

每条候选独立检查：

1. schema、finite、时间和坐标；
2. observation/freshness/provenance binding；
3. route/lane corridor 与导航合法性；
4. curvature、speed、acceleration、jerk、lateral acceleration；
5. 当前可观测 collision envelope；
6. traffic-light/stop-line；
7. controller feasibility 与最小执行时域。

结果：

- `PASS`：合同和当前检查干净，可进入 World；
- `REVIEW`：候选基本有效但处于冻结边界，可进入 World，最终仍必须由 Safety 重验；
- `REJECT`：坏数据/绑定、严重道路偏离、明显迫近碰撞、非有限或不可执行，World 不可见且
  不得复活。

Guard 只使用当前可观测状态和候选本身，不读取 rollout future、Oracle、formal label 或
Regression answer。`REVIEW` 不是 Safety 证书。

后文统一把 `PASS/REVIEW` 称为 `eligible`；“通过 Guard”不得被误读为只有 `PASS`。

## 5. 候选集合规则

| eligible 数 | World 行为 | 后续 |
|---:|---|---|
| 0 | 不运行 | Safety fallback / MRM |
| 1 | `DEFER_SINGLE_CANDIDATE` | 唯一候选仍需 Safety 完整重验 |
| 2 | 对两条候选共享参数预测 outcome | calibrated choose/hold/defer 后交 Safety |

World 的 formal pairwise coverage 只计算两条候选都具有完整原始预测的 tick。仅 VLA 或仅
Expert 幸存不能计作 World 成功选择，也不能由 router 补成 pair。

候选 slot 可以置换，语义只由 candidate ID 和 canonical trajectory hash 决定。World 在线
feature view 物理排除 source、slot、branch order、Guard verdict 和 provenance；这些字段只
用于身份绑定与审计。

source 元数据被排除不代表 planner 风格从轨迹几何中不可推断。source metadata swap 必须
保持预测不变；trajectory-to-source probe 则作为数据 shortcut 诊断单独报告，不能把真实
曲率/速度差异也从候选中抹掉。

### 两候选时的 router 结果

| 结果 | 固定语义 |
|---|---|
| `CHOOSE` | 当前 tick 请求分数/区间占优候选进入 Safety |
| `HOLD` | 保持上个 source，但使用该 source 当前 tick 的 fresh candidate，不重放旧轨迹 |
| `SWITCH` | 满足 hysteresis/min-hold 后切换到另一 source 的 fresh candidate |
| `DEFER_AMBIGUOUS` | learned World 放弃排序；当前 held source 仍 eligible 时先用它，否则按 Expert→VLA 冻结顺序 |
| `DEFER_SINGLE_CANDIDATE` | World 不做 pair claim，唯一 eligible candidate 直接交 Safety |

defer 不是在线 Oracle、人工接管、无控制或直接选择 MRM；每条路径仍经过完整 Safety。冻结
fallback 也不能把 `REJECT` 复活。

## 6. 选择与执行身份链

每个 tick 必须记录：

```text
generated candidate ids/hashes
guard-eligible ids
world raw per-candidate outcomes
router raw / stabilized / deferred selection
safety selected / repaired / fallback id
final executable id/hash
controller applied id
```

有效链：

```text
generated → Guard eligible → World ranked/deferred
          → Safety selected/repaired/fallback
          → final executable → applied control
```

任何 orphan、跨 source 无记录替换、executed/applied hash 不一致或无法解析 final ID 都必须
fail closed，并使该 Evidence 行无效。

## 7. Safety 与 repair

- World 只提供顺序或 defer，不能批准轨迹；
- Safety 对原始或 repaired trajectory 重新执行全部硬检查；
- repair 必须有输入/输出 hash、mode、理由和 final validation；
- 首选候选失败后可按冻结 fallback 顺序检查另一候选；
- 两者都失败才进入 MRM/emergency；
- controller 不能在 Safety 输出之后增加 throttle、替换轨迹或修改 source。

## 8. CORA 必做测试

- VLA forward count 等于 1；
- 两来源同 frame/time/route；
- raw/canonical hash 和 provenance 完整；
- source/slot/candidate order permutation；
- Guard PASS/REVIEW/REJECT 边界；
- 单候选不能计入 pairwise World coverage；
- candidate swap 后 World outcome 跟随 trajectory；
- repair final validation；
- selected/final/executed/applied identity；
- expired/orphan/deadline fail closed；
- `HOLD` 不复用旧 candidate ID；
- `DEFER_AMBIGUOUS` 回退顺序、reason code 与 offline/live trace parity。

反事实执行与标签合同见 [COUNTERFACTUAL_DATA](COUNTERFACTUAL_DATA.md)，World 与 router
合同见 [WORLD_MODEL](WORLD_MODEL.md)。
