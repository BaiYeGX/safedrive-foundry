# Hybrid VLA–Expert 候选与 Guard 合同

## 1. 目标

同一 observable anchor 上只允许两个在线规划来源：

| source | 实现 | 系统作用 |
|---|---|---|
| `vla` | 阶段锁定 M0/M1/M2 的 `NominalVLAPolicy` / SimLingo 一次真实 forward | 视觉语义与 language-action 先验的 nominal proposal |
| `expert` | Classic route/behavior/Frenet-ST planner | 几何、规则、动力学和确定性 proposal |

二者互补但平权：系统不预设 VLA 必须胜出，也不允许 Classic branch outcome 被复制为 VLA branch 标签。
可信 Classic 专家轨迹可以作离线 SFT label，必须与 World 分支后果标签分离。
在线学习模块不得从一条轨迹做 perturbation、复制或重命名来伪造第二候选。offline-only
intervention 只用于反事实数据和 benchmark，必须有独立 provenance，不能进入 live set。

本文固定三种不同身份：`candidate` 是 generator 原始 proposal，`selected` 是被请求交给 Safety
的候选，`executable` 是 Safety 接受或 repair 后批准的轨迹。三者不能用同一个 source 字段
相互覆盖。

## 本周模型绑定

C3 训练 M1，C4 训练 M2；C5 只比较 M1/M2，M0 留在离线基线。
nominal 指候选角色，不要求固定原始权重。一条 VLA 候选仍只来自一次真实 forward。
M2 World 复用候选特有输入前的当前 h；固定 b 与学习残差 R 的完整预测用于排序。
两个候选共享同一模型，不输入 teacher 或未来。
绑定 LoRA/heads/normalization/canonicalizer；本周无新校准，router 标 UNCALIBRATED。
C5 三臂与 shadow 规则以 ROADMAP 为准，不额外训练 M0 配套 World。

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
| 2 | 对两条候选共享参数预测 outcome | choose/hold/defer 后交 Safety（本周未校准） |

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


## C5 可直接设置的 goal

> 完成 C5：按 docs/HYBRID_CANDIDATES.md，接入 C4 的 M2 与配套 World，
> 在不改变 Guard/Safety/controller 权限的前提下完成固定三臂、6 roots、
> 最多 18 次开发运行；保存全部成功/失败、逐 root 对照、身份链、尾延迟和重放记录。
> 验收后更新 PROGRESS、将下一入口指向 C6 并停止，不追加正式矩阵或调参重测。

## C5 开始条件与实现范围

读取 C3/C4 stage-summary，核对 M1/M2、原始 base、原生适配、M2 World/b 系数/normalization、
shared h schema、训练数据/预处理身份和离线检查。没有有效 M2 或配套 head 则停止依赖运行。
允许修改 nominal adapter、World router 接口、run-lock/collector/报告及直接测试；
不改 Guard/Safety 阈值，不从旧 scorer 的 readiness 继承新模型 pass。

**先做离线接入。** 一次真实 VLA forward 同时提供轨迹和允许的 h；cache 绑定 observation、
route revision、checkpoint 与 preprocessing。检查缺失、过期、不同模型缓存均拒绝。
输出 canonical 轨迹后逐候选 Guard；World 只看 eligible，排序结果再交 Safety。
M1/M2 各臂只加载一个 VLA；A/B shadow scorer 保持同形状和调用次数，不使用无效分数作评价。
记录 dtype、调用次数与实际耗时，shadow 不等于硬件成本完全相等。

**把规则一次固定。** A=M1 无 World；B=M2 无 World；C=M2 有 World。
A/B 两候选 eligible 时请求 VLA，否则 Expert；全部仍经同一 Safety。
C 在有效支持范围内按四维连续后果评分，使用同一 hold/defer 状态机；
无合法预测、无支持、差值接近则 defer。只使用 train/validation 定尺度、权重与差值阈值，
不使用这次 6 roots 的 outcome。默认从 progress 主项与舒适项小权重开始，
具体数值在首个运行前写入 run-lock；不通过挑有利阈值提高 VLA 使用率。
未实现可靠支持检查时采用保守 defer，并记录覆盖不足，不宣称已解决 OOD 或风险校准。

还需固定：控制 20Hz 的实际配置、VLA 决策 cadence、chunk、传感器 freshness、
deadline、Safety/fallback、初始化/终止规则。候选未来标签基于旧固定 proposal 短分支；
新闭环重规划结果不能回写成同标签或被当成训练外推已验证。

**预登记 6 roots。** 正常/候选分歧各 3，按路线几何/障碍设置先定义场景类别，
不能运行微调策略后筛选“确实分歧且有收益”的样本。实际无分歧照实报告。
注册新 seed/recipe/route/weather/actor/初始化，排除旧 seed 101、reserved 173/179，
并与 train/validation lineage 做重叠检查。每 run 最多 60 s 仿真。

按 root 配对，执行臂顺序轮换 ABC、BCA、CAB，各两次，降低固定运行次序影响；
先登记全部清单，再用第一个 root 的三臂确认工程链。工程/安全接口失败立即停止，
已执行结果保留，不另外增加 pilot 或替换场景。

**真实运行。** 本次 preflight READY 后由 ScenarioRuntime 唯一持有 tick。
允许的 ensure/recheck 按 ENVIRONMENT；不得新建 tick master 或 force-kill 不明进程。
每臂 reset 可比，记录 route/actors/lights 与传感器绑定；不可比结果不得作配对收益证据。
总计最多 18 次 episode attempts，失败也占一个 attempt；不自动替换/重跑坏结果。
总 CARLA 含启动/恢复最多 4 小时，先到上限就冻结 PARTIAL。

## C5 输出与验收

每 run 保存 root/arm/版本、全部可用 raw 预测、候选/Guard/selected/repaired/executable/applied、
事件、路线进度、干预/defer/source usage、latency/deadline、显存、终止与 cleanup。
失败 root 单列；pair comparison 只用双方有效的 root，同时列出原始分母和失配原因。
A/B 报联合训练的闭环差异；B/C 报同策略下 World 选择作用；不同臂不混入不同 Safety。

主报告为原始事件计数、逐 root 差值和均值/中位数、实际 VLA 暴露、尾延迟。
6 roots 不做普遍安全结论或显著性包装；World 自评不作为最终收益标签。
如果 C 始终 defer 或 A/B 几乎未执行 VLA，说明缺乏相应作用的证据，不写成改进已成立。
仅从已有 trace 估算去除 shadow 的计算耗时不能称真实部署测量；未实测则标 NOT_MEASURED，
不额外追加一组闭环以测成本。

验收是接口/身份/权限正确且清单被真实处理；完整完成需 18/18 planned runs 有合法
实验结果，基础设施失败导致缺项则 PARTIAL。负收益或合法碰撞事件作为结果保留，
不可通过改变 recipe/阈值重跑获得“通过”。实际安全接口越权与普通任务失败必须区分。

新 stage-summary.json 关联清单、资源、逐 run 记录、逐 root 表与 replay 输入；
更新 Evidence/PROGRESS/START_TASK 后停止于 C5。


## C5 微弱收益与归因检查

本轮 C/B 正向趋势口径在 PROJECT 预登记；场景/指标/失败处理不根据运行结果修改。
保留相同初始化、既定臂顺序与全部失败；实际无候选分歧也报告，不替换成更易赢场景。
C 的完整预测为 b+R，另从同一已有 forward 记录 b-only 排序和 residual override 次数。
这不增加模型或闭环臂，也不提供未执行动作的真实后果。

若 C/B 有提升却与 b-only 决策相同，只能报告路由组合收益，不能说学习残差造成提升。
若辅助门控始终关闭、R 恒为零或 C 全程 defer，也要指出学习/选择作用未被证实。
不以降低 Guard/Safety、增加 VLA quota、临时放宽 deadline 或删掉失败制造正向。
使用已有 trace 作诊断，未做真实部署成本对照的部分仍为 NOT_MEASURED。

## C5 量化交付

执行 [PROJECT 论文量化合同](PROJECT.md#论文量化合同c3_c6_metrics_v1planned)。逐 root 表
列 A/B/C 进度、终止原因、三类原始违规、deadline n/N；主差值 C−B，补充 B−A。
完整 6 roots 按预定正常/分歧组展示，不从实际分歧或收益重新选组。
再由已有 trace 计算 C-RC、C-COMFORT、C-COVER、C-DEFER、C-OVERRIDE、C-INTERVENE、
矩阵完成率和时延/新鲜度；不支持的字段为 N/A 并说明原因，不能默认为零。
C/B 的 deadline 不增加具体指配对全矩阵 miss 率不增加，同时列各臂原始 miss/request 数及
逐 root 差；调用周期、timeout、Safety 和 shadow 配置在运行前固定。
原 >0.01 m 趋势阈值不变，≥1 m 为新增规划目标而非必须重跑直到通过的门。
全量事件与合法负结果保留，基础设施缺项标 PARTIAL；不新增闭环或计时专用 CARLA runs。
