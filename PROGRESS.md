# SafeDrive Foundry 进度

## 2026-08-27 — H6 VLA75 v2 工程闭环已落地，正式证据仍未验证

状态仍为 `H6 IMPLEMENTED / MEASURED / NOT_VERIFIED`；没有把离线/单元测试结果写成
CARLA 正式通过。已按用户计划切换到分支 `codex/h6-vla75-completion`，未 commit、未
push，并在 Git-ignored 的
`generated/h6/h6-vla75-worktree-baseline/` 保存了 tracked binary patch、任务相关
untracked 快照、工作树状态和 SHA256 manifest，保留原有 H6 开发成果可恢复。

本轮完成的 v2 工程范围：

- 新增 `safedrive.h6.vla75.v2` 配置/哈希、A/B/C 正式 lineage（`103/107`、`109/113`、
  `127/131`）、12-pair pilot 与 108-pair full 矩阵；seed 101 仍只保留为已消耗的旧
  诊断数据，开发 seed 固定为 `89/97`；旧 `h6-vla90-*` 配置、hash、acceptance 和
  Evidence 接口未改写。
- 新增 `WorldVLA75Prediction`/`WorldVLA75Scorer` 的 14-output source-blind 模型，保留
  World v3 原 12 个输出，并加入 preference utility、executability、paired coverage、
  group-DRO 和 event-aware temporal loss；训练器实际按 `pair_id × arm × tick` 接入
  event-aware 相邻 tick loss，并按固定 v2 字典序选择 checkpoint；旧 World v3
  loader/模型接口继续可用。
- 新增逐 tick raw gate（双候选完整评分、score/preference/trust/risk 四项）、实际
  `AppliedControl.executed_id` 绑定的 75% gate、Classic+MRM 25% gate、target-only
  unsafe、paired progress bootstrap、资源/切换/ping-pong、Guard/World/Safety/repair
  转移矩阵和完整 provenance 校验。EMA/hold/force-VLA/单候选幸存不能提高 raw 90%。
- 新增温度校准、VLA75 temporal stabilizer、repair 后 final validation、正式 run-lock
  （配置/矩阵/checkpoint/ensemble/calibration/代码 scoped hash/完整 dirty worktree
  identity），并将 run-lock 与 dataset、lineage、matrix scope 绑定；正式 collector
  使用并记录 run-lock 中冻结的 router calibration，拒绝 summary/lock 参数漂移。
- collector 记录 raw/stabilized/selected/safety-executed/applied 链、repair 输入输出、
  模型/feature/worktree hash、phase 边界和独立 spectator follow；Classic-only baseline
  仍不把 VLA 候选交给控制。
- CLI 已支持 `--contract vla75-v2`、`--formal-lineage a|b|c`、`--run-lock`，新正式
  dataset 必须为 `h6-vla75-*`；readiness 对 summary schema、三模型 seed `11/23/37`、
  开发数据隔离、90% raw calibration、75% applied proxy、90% outcome attribution purity
  和 CUDA 做硬检查。

离线验证（本轮实际运行）：

```text
/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests -t . -v
436 tests: 435 passed, 1 skipped, 0 failed
/home/sdf/.venvs/sdf/bin/python -m compileall -q safedrive_foundry scripts tests
passed
git diff --check
passed
```

新增的 VLA75 hardening 回归覆盖缺双评分仍留在分母、unknown/orphan/non-TRACK applied、
executed↔applied ID 不一致、repair final validation、temporal break、group-DRO、14-output
source-blind 模型、run-lock 矩阵内容篡改，以及 summary checkpoint 顺序/ensemble hash
绑定；本轮新增 hardening 测试 16 项均通过。

正式 lineage 状态账本已接入编排器：pilot 失败或 full 失败写入不可重复使用的终态，
full 必须先有同一 run-lock 的通过 pilot；pilot 通过后升级 full 会保留 pilot hash/history，
A/B/C 三条均失败时返回整体关闭标记，避免重复盲测或改写负 Evidence。

环境硬门按规范执行：`doctor` 报告 GPU 不可见、CARLA RPC 未启动；`sim preflight --json`
为 `RETRYABLE_FAILURE / SERVER_NOT_RUNNING`。按允许的唯一恢复动作执行一次 `sim ensure`，
但因 CARLA 配置备份路径为只读而返回 `FAILED_FINAL / Read-only file system`；随后唯一一次
复查仍为 `RETRYABLE_FAILURE`。因此未运行 CUDA 训练、CARLA pilot/full、正式 lineage 或
acceptance，不能声明 H6 `VERIFIED`、不能声明 90%/75% 门通过。环境诊断输出已保留在
`docs/environment/evidence/g0-05/doctor.{json,md}`。

正式接管顺序保持冻结：恢复 GPU/CUDA 与 CARLA 权限后，只用 seed 89/97 重训并通过新的
tick-wise readiness；再生成同一 lineage 的不可变 run-lock，跑 12-pair pilot，pilot 全门
通过后才可跑 108-pair full。任一 lineage 失败即冻结该 lineage 的负 Evidence；三条均失败
后正式关闭 H6，不降低门槛、不复用 seed 101、不删除失败记录。
## 2026-08-27 — H6 前向实际主驾门修订为 75%，完成逐 tick 诊断与迭代交接

状态仍为 `H6 IMPLEMENTED / MEASURED / NOT_VERIFIED`。用户把下一条正式 lineage 的
VLA 最终实际执行硬门从 90% 修订为 `>=75%`，Classic + MRM 合计 `<=25%`；此前要求的
World 原始双候选高分门没有撤回，仍为 `>=90%`。历史 `h6-vla90-*` Evidence 保持原合同，
不能按新门回写成成功。

重新读取 seed 101 正式 600 tick 后确认：World 选择 VLA 303 次，Safety 仅 18 次回到
Expert，最终 VLA 285 次。达到 75% 至少需要 450 次 VLA，还差 165 tick；若 fallback
固定为 18 次，World 至少要选择 VLA 468 次，若维持当前 5.94% fallback 比例则约需 479
次。Guard 对 VLA 仍为 `PASS 453 / REVIEW 147 / REJECT 0`，所以主瓶颈不是 Guard。

World 590 个完整双评分 tick 中只有 131 个同时通过分数、可信和风险门；74 个只差可信，
87 个分数低但可信/风险通过，64 个分数与可信同时失败，234 个分数/可信/风险全失败，另
有 10 个 missing pair。即使只降低可信门也远不足以达到目标，必须重新训练排名、风险和
时间一致性。

新增 `docs/H6_VLA75_HANDOFF.md`，记录新验收合同、逐地图/场景诊断、SelectiveNet、
learning-to-defer、DAgger/SafeDAgger、group DRO、temporal consistency、calibration/
conformal risk control 等一手研究依据，以及新对话的实施顺序。同步更新 START_TASK、
ROADMAP、PROJECT、WORLD_MODEL、EVIDENCE 和 README，避免新对话继续误用实际 90% 门。

本轮只修改文档，没有训练、CARLA 运行或正式 seed 操作。当前完整代码回归的最近实测仍为
`419 tests: 418 passed, 1 skipped, 0 failed`；本轮检查 8 个活动/交接文档的本地链接，
0 个缺失，`git diff --check` 通过。GPU/CUDA readiness 仍未恢复，新模型、24-pair 新
开发 pilot、完整 108-pair 训练和新 held-out formal 均未执行。

## 2026-08-20 — H6 平衡训练通过旧 readiness，但正式逐 tick gate 失败

状态：`H6 IMPLEMENTED / MEASURED / NOT_VERIFIED`。没有进入 108-pair full；H5 的正式
负结论不变。

本轮先修复了紧急场景把 ego 错误投影到路线终点的问题，并完成新的平衡训练 pilot：

- `h6-vla90-train-pilot-20260820-v2` 共 24 pair / 48 runs，训练和校准 seed 都覆盖
  free-flow、紧急刹车、加塞、红灯四类场景；
- seed 89 的 VLA 实际执行 `566/600 = 94.33%`，Expert 33、MRM 1；seed 97 为
  `544/600 = 90.67%`，Expert 54、MRM 2；两个 split 的 on/off 不安全运行数分别同为
  `2/12`；
- 新训练 loader 按 tick 保存真实执行 outcome mask，并加入 12 条整段 exact-reset
  Classic-off/VLA-primary-on 配对排名监督；World 三模型训练使用 1212 train / 1212 val
  rows；
- 旧校准只在每场第一拍检查 World 分数，得到 `11/12 = 91.67%`、可信门 `0.943988`、
  风险上限 `0.35`、进度差 `+5.259m`，readiness 当时无失败项。该模型为
  `generated/h6/world-v3-vla90-pilot-20260820-v3/training-summary.json`。

随后运行一次隔离 seed 101 的正式 12-pair pilot：

```text
dataset  h6-vla90-formal-pilot-20260820-v1
evidence docs/runtime-evidence/h6/h6-vla90-formal-pilot-20260820-v1/final-delivery.json
sha256   8dae5c2e661abafc1dceab633d3338201a7fe1e6b50ecd5e334641aa68223194
```

正式结果为 `GATE_FAILED`：

- World 严格 VLA 高分 `131/600 = 21.83%`，VLA 实际执行 `285/600 = 47.50%`，远低于
  两个 90% 门；Classic 实际 315，MRM 0；
- Guard 对 600 个 VLA 候选为 `PASS 453 / REVIEW 147 / REJECT 0`；Safety 仅 18 次
  Classic fallback，并成功执行 RATO 42、QP 26。因此正式低占比主因是 World，不是
  Guard/Safety 大量杀 VLA；
- on/off 不安全运行数同为 `3/12`，unsafe delta 0；进度 mean `+2.455m`，bootstrap
  lower-95 `+0.629m`；scorer P99 `9.12ms`，0 deadline miss；
- 31 次切换（1.033/s）和 5 个 ping-pong 场景失败；Town01 aggressive 最后 10 tick 的
  Expert 因 navigation-start 距离 5.73–8.79m 被 Guard 硬拒，World 合法地没有收到双候选，
  acceptance 同时记录 10 个 missing-pair provenance failure；
- Town03 free-flow 的 on/off 各撞静态物一次；Town03/Town05 red-light on/off 均闯灯。
  训练 Town03 red-light 还记录了 `other_actor_id=0`、水平冲量 0 的连续竖直接触，坡路
  预滚高度/路线需要独立修复。

由此确认旧 readiness 存在口径漏洞：它用每场第一拍的 12 个分数代表正式 600 tick。
代码已改为用校准集每一个 on-arm tick 检查 90% World 高分，整段 paired outcome 只检查
安全和进度；新增回归保证“第一拍好看、全程只有 80%”不能通过。22 项 World 专项和
compileall 已通过。计划用开发数据重训验证新门时，GPU 执行被工作区系统以
`workspace credits exhausted` 拒绝；没有绕过，也没有生成新 summary。

验证：全量 `419 tests: 418 passed, 1 skipped, 0 failed`；`compileall` 通过。GPU 重训练
命令未获执行权限，因此不能声称新 tick-wise calibration 已实测通过或失败。

跟车视角：平衡训练 48 runs 每条 `285–847` 次更新，正式 24 runs 每条 `390–1113` 次；
全部 `spectator_follow_error=null`。后续所有真实 CARLA 测试继续强制跟车。

停止边界：seed 101 已被正式失败消耗，不能调参后重跑冒充 held-out；未获用户新 seed
lineage 授权前不继续 formal，GPU credits 恢复前也无法完成新 tick-wise readiness。

## 2026-08-19 — H6 训练 pilot 两轮修复后仍差 1.33pp，跟车视角已修复

状态：`H6 IMPLEMENTED / MEASURED / NOT_VERIFIED`。已完成真实 CARLA 12-pair training pilot
的初始运行和两次有实质差异的修复验证；因仍未达 90%，训练 loader 正确拒绝数据，
没有生成新 World checkpoint，也没有进入 formal seed。

三轮实测：

- `h6-vla90-train-pilot-20260819-v1`：VLA `422/600 = 70.33%`，Classic-off 因 80 tick
  直线预滚只完整执行 468/600，数据来源纯度门拒绝；
- `h6-vla90-train-pilot2-20260819-v1`：沿路线且保留场景原始预滚后，Classic-off
  `600/600`，VLA `486/600 = 81.00%`，仍被 VLA 来源纯度门拒绝；
- `h6-vla90-train-pilot4-20260819-v1`：修复动态红灯时序、前车半车长缺失和有限动力学
  越界误硬拒后，VLA `532/600 = 88.67%`，Expert 64 tick，MRM 4 tick。训练在
  `Town01__free_flow` 仅 `42/50` VLA 的第一个不纯 episode 处停止；这是诚实失败，
  没有降低 90% 门槛。

最后一轮的实际分层证据：

- 600 个 VLA 候选的 Guard 结果为 `PASS 292 / REVIEW 302 / REJECT 6`；只有 1% 被 Guard
  硬杀，说明 Guard 已不是主要瓶颈；
- 开发路由在 `588/600 = 98%` tick 把 VLA 排在第一；因此这批数据不能用来证明新 World
  已学会 90%，但它能明确证明“World 不选 VLA”不是当前实际执行不足的主因；
- 最终 Safety 中，VLA 原样通过 291 tick，纵向 QP 修复成功 224 tick，RATO 修复成功
  17 tick；57 tick 改用 Expert，4 tick 进 MRM。未执行 VLA 的主要原因是 33 次
  `red_light_unstoppable`、18 次碰撞包络、6 次横向加速度、2 次超速，以及 6 个真正
  offroad 的 Guard REJECT；一个 tick 可同时有多个原因；
- 24 次 RATO 修复在最终重验时仍因 yaw-rate/碰撞失败。Town05 emergency 的 VLA 从上轮
  `17/50` 升到 `49/50`，证明补入前车半车长的修复有效；Town03 aggressive 仍受急弯
  yaw-rate/碰撞重验限制；
- on/off 的不安全运行数同为 `2/12`，均是 Town03/Town05 红灯场景的红灯违规；
  VLA-on 总进度 `106.79m` vs Classic-off `50.14m`。这些只是开发训练 pilot 诊断，
  不是 formal acceptance。

用户指出视角没有跟车后，中断了 `pilot3`并保留 `KeyboardInterrupt` Evidence。
`scripts/h5_collect.py` 改为独立 20 Hz 后台跟车视角，不再等候慢速的模型/规划循环。
`h6-camera-follow-smoke-20260819-v1` 单场景完成；off/on 分别更新 572/473 次且无错误。
最后 24 个 pilot4 运行每场更新 `266–1138` 次，全部 `spectator_follow_error=null`。

停止原因：同一 pilot 已用完初始实现加两次实质修复上限，不进行第三次边跑边改。
下一个独立任务应重新设计按 tick 的 VLA/修复 VLA 训练标签，并在新数据集前专门改善
红灯停车、碰撞纵向规划和急弯修复。

验证：全量 `409 tests: 408 passed, 1 skipped, 0 failed`；`compileall` 通过。全量回归包含
跟车线程在决策计算期间仍持续更新、动态红灯时序、前车半车长、有限动力学
`REVIEW` 和横向加速度纵向修复的新回归。

## 2026-08-19 — H6 VLA 主驾改造已实现，正式 90% 门尚未验证

状态：`H6 IMPLEMENTED / MEASURED / NOT_VERIFIED`。H0–H4 历史结论不变；H5 仍是
`VERIFIED / GATE_FAILED / STOPPED`，没有用 H6 开发结果覆盖 H5 的负结论。

本轮按用户明确目标把“World 在 90% 情况下真给 VLA 高分、VLA 实际执行也至少 90%，
Classic 约 10% 兜底”落成了独立 H6 合同：

- Guard 由二态改为 `PASS / REVIEW / REJECT`。轻微道路、规则、碰撞预测、动力学或控制
  擦边进入 `REVIEW` 并交给 World；只有坏数据/绑定、严重偏航或越界、明显迫近碰撞和
  不可执行合同硬拒绝。World 仍看不到 `REJECT` 候选。
- 修正了 Guard/Safety 大量误杀的两个来源：碰撞从过大的外接圆改为车辆有向矩形，actor
  预测对齐候选时间；红灯从“整条轨迹都必须接近停车”改为检查能否在停止线前停车或是否
  真正越线。最终 Safety 的硬碰撞、红灯、越界和状态机没有删除。
- World v3 使用共享、source-blind 的候选编码，分别预测综合任务效果、进度及方差、完成、
  碰撞、红灯、越界、舒适性、修复成功率和可信度。正式 VLA-primary 必须同时满足 VLA
  综合分不低于 Expert、可信门通过、风险门通过；高可信不能伪造高综合分。
- Safety 接收完整 World 顺序：VLA 第一时先做一次有边界且最终重验的修复，失败后使用同
  一 tick Expert；两者都不安全才 MRM。RATO 新增最终硬规则复查，修复器不能把车辆矩形
  仍有角部重叠的轨迹谎报为成功。
- VLA 增加意图保持的运动学滤波，几何、航向、曲率、速度和加速度一起重算，并把加速度/
  横向加速度上限收进最终 Safety 数值内。Classic 规划失败或短时域时生成有 provenance
  的完整时域受限停车候选；off arm 使用纯 Classic router，不再被 VLA 污染。
- H6 训练闭环和正式验收隔离：训练 seed `89/97`，正式 seed `101/103`，各 108 pair / 12
  pilot。训练只载入真实 Classic-only off 与 development VLA-primary on 的配对结果；正式
  seed 被 loader/readiness 明确拒绝。
- 正式门直接审计 World 的两条原始评分，只有 Expert/VLA 都被评分且 VLA 综合分不低于
  Expert、可信和风险门都过，才计入 World 的 90%；只有一条 VLA 幸存不能冒充 World 高分。
  同时要求实际 VLA `>=90%`、相对纯 Classic 不安全率增量 `<=1pp`、配对进度 bootstrap
  lower-95 `>=0`、P99/0 deadline miss、低切换且无 ping-pong。

真实 CARLA 开发测量（全部非正式，失败原样保留）：

- `h6-vla90-explore-20260819`：两候选均 50/50 Guard eligible，但零速预滚与旧近重复 defer
  使 VLA 实际 0%；
- `h6-vla90-explore2-20260819`：开发强制采样中 World 选 VLA 50/50、实际 VLA 48/50
  （96%），本短跑碰撞/红灯/越界为 0，进度约 3.03m；但 Classic 生成不完整、baseline
  被污染且配置非正式，`final-delivery.json` 正确判为失败；
- `h6-vla90-explore3-20260819`：纯 Classic baseline 暴露短时域生成失败；on arm 在 17 tick
  后暴露 VLA 3.5m/s² 超过 Safety 3.0m/s²。之后完成了 Classic 受限停车和 VLA 2.8m/s²
  滤波修复，但按同问题最多两次差异化修复规则未做第四次 CARLA 重跑。

旧 H2/H3 数据训练出的 `generated/h6/world-v3-dev-v1/training-summary.json` 无法通过严格
校准：旧数据里 VLA 进度总体为负，不能通过调门伪造 90%。下一步必须先采新 H6 training
matrix、重训并通过 readiness，再跑 held-out formal pilot/full。

验证：专项 60 项通过；修复后全量 `399 tests: 398 passed, 1 skipped, 0 failed`；随后新增
“必须有 Expert/VLA 双评分、纯 Classic 基线和 90% 训练标签纯度”的 29 项 World/路由/Safety 专项也通过。正式 108-pair 训练、
新 World checkpoint、12-pair formal pilot 与 108-pair full acceptance 均未执行，因此不得
写成 `VERIFIED` 或声称已达到正式 90%。

## 2026-08-16 — H5 World on/off 完整闭环执行完成（负结果）

状态：`H5 COMPLETED / VERIFIED / GATE_FAILED / STOPPED`。H6 closure 仍未开始。

最终运行：`h5-pilot-all2`（同一 dataset 先 pilot 后 full）。

Evidence：

```text
docs/runtime-evidence/h5/h5-pilot-all2/final-delivery.json
evidence_sha256 846ef8a6f5ff6b3ca330a55ba53f69849f043346b74910f6a405eb95f5543517
gate_status     GATE_FAILED
gate.failures   ["reset_not_comparable:Town01__cut_in__s2__CloudyNoon:off", "progress_net_benefit", "chattering_noninferior", "resource"]
```

实测：

- 222/222 runs 完成，协议完整性中 1 个 reset 不可比；
- safety non-inferior：通过（on unsafe 4 vs off unsafe 6，无 on unsafe/off safe）；
- progress net benefit：未通过，mean delta `0.2549`，lower 95 `-0.0709`；
- chattering：未通过，on 总 switch 14 vs off 0；
- resource：未通过，scorer deadline miss 1，P99 `47.22 ms`（<=50ms 但存在 1 miss）；
- 结论：World 闭环收益不可复现，按冻结协议记录负结果，不重跑、不调门。

H5 工程实现已落地：H5WorldRouter、74 个 locked-test 场景矩阵、off/on/defer 三臂
collector、acceptance、orchestrator、风险/温度校准、readiness。


## 2026-08-16 — H4 Locked Evaluation 完成

状态：`H4 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。H5 Closed-Loop 与在线 Oracle 仍未授权。

最终运行：`h4-locked-20260816-final`。

Evidence：

```text
docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json
evidence_sha256 35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4
gate_status     GATE_PASSED
gate.failures   []
```

实测：

- test valid 74，decisive 64；World accuracy `1.0000`（64/64），AUROC `1.0000`；
- best simple baseline `candidate_only=0.890625`；bootstrap delta `0.109375`，lower 95 `0.03125`；
- defer coverage `0.984375`（63/64 ranked），ranked accuracy `1.0000`；
- end-to-end accuracy `0.984375` vs h1_soft_selector `0.890625`；
- Expert/VLA wins `39/25`；ECE `0.00884`；swap max error `0.0`；isolation `passed`；
- P99 `15.23 ms`，增量显存 `0.03125 GiB`，0 deadline misses；
- `h5_authorized = true`，但 H5 不自动进入。
- 全量测试通过；compileall 与 `git diff --check` 通过。
- 审计后加固：batch_size 生效、final checkpoint 独立验证、runtime risk gate 接入、H5 hysteresis router 与 readiness 脚本。

H4 使用 H3 5-seed final checkpoints 加 dev-only per-model utility normalization 修复了
H3 raw ensemble uncertainty 导致的全 defer 问题；该 normalization 已冻结并写入 H4 evidence。


## 2026-08-15 — H3v2 真实 CARLA Challenge + 嵌套 OOF 最终验收

状态：`H3 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。H4 locked evaluation 已在后续完成；H5 Closed-Loop 与在线 Oracle 仍未授权。

最终运行：`h3-v2-20260815d-final`。数据联合：
`h2-gatepass-20260813-routefix`（120 H2）+ `h3-challenge-v2-20260815d-dev`（96 H3v2 Challenge）。

最终 Evidence：

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/final-delivery.json
evidence_sha256 f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e
```

最终门：

```text
gate.passed true
gate.failures []
```

实测：

- OOF decisive accuracy `1.0000`（91/91）；best simple baseline `candidate_only=0.9231`（非学习简单规则口径）；额外 candidate-only MLP `1.0000`、full-feature MLP `0.9560`；
- paired bootstrap delta `0.0769`，95% 下界 `0.0330`；
- action sensitivity `28.57pp`；history sensitivity `28.57pp`；
- ECE `0.00113`（嵌套 T*=0.0500）；progress/jerk regret 不劣于基线；
- 5/5 seeds `1.0000`；swap max error `0.0`；leakage `passed`；
- P99 `13.89 ms`，增量显存 `0.0566 GiB`，0 deadline misses；
- 全量测试 `340 tests: 339 passed, 1 skipped, 0 failed`；`compileall` 与 `git diff --check` 通过。

Challenge 数据质量门：

- 96/96 terminal；78 valid；87 distinct；72 decisive；
- hard-unsafe branches 13；Expert/VLA wins 53/19；source-only 0.7361；
- store manifest ok；offline-label permutation 全通过。

H3v2 已冻结（H4 后续已完成）。


## 2026-08-14 — H3 World Scorer 真实 CARLA 物理采集与端到端交付终态

状态：`H3 COMPLETED / VERIFIED / GATE_FAILED / STOPPED`（旧 v1 物理 Challenge；已被 H3v2 最终记录取代）。

最终运行：`h3-carla-joint-20260814-v1`（联合使用 H2 冻结的 120 场真实物理数据与在 CARLA 0.9.16 仿真中 100% 物理执行的 96 场真实对抗博弈数据 `h3-carla-challenge-20260814-v1`）。Evidence 位于 `docs/runtime-evidence/h3/h3-carla-joint-20260814-v1/`，模型 checkpoint 位于 `generated/h3/h3-carla-joint-20260814-v1/checkpoints/`。

冻结 Evidence：`final-delivery.json`（Evidence SHA256: `e9470c6880da0e00ab6f6fae7d201cb2c363d7479f705643089263c2e3414e8c`）；泄漏审计 Evidence `leakage-audit.json`；基线对比 Evidence `baselines.json`。

在 RTX 4080 16GB / CUDA、CARLA 0.9.16 物理仿真（Town01/Town03/Town05）与 3-fold OOF $\times$ 5-seed 深度集成训练上完成：

- **100% CARLA 真实物理对抗博弈采集**：在 CARLA 0.9.16 中完成全部 96 场真实物理对抗博弈（涵盖 `emergency_lead_brake`、`aggressive_cut_in`、`red_light_dilemma`、`cross_traffic_conflict` 4 类对抗博弈动作），拒绝一切合成/运动学造假数据，通过分层字典序硬安全 + 侵入裕度离线 Oracle 进行真值判定并生成 Parquet Shards（`pairs/`、`labels/`、`manifest.json`）；
- **数据与泄漏隔离**：联合载入 216 场真实 CARLA records（120 H2 + 96 Challenge），严格只读保护 H2 原始数据，零 Feature/Target 混淆，216 records 泄漏审计全部通过；
- **模型因果与温度自适应**：引入 Scipy 动态拟合验证折最优温度参数 $T^*$（杜绝静态常数硬编码）；在 3 折 OOF 交叉验证上模型达到 100.0% 胜负判定准确率；Action Masking 消融测试准确率下降 **20.83 pp**（从 100% 降至 79.17%），严密证明模型对候选规划轨迹未来动作的因果敏感性；
- **置换等变性与数学对称**：Swap Invariance 误差为 `0.000000`（$\le 10^{-6}$ 完全对称）；ECE 校准误差达到 $10^{-8}$ 级别；
- **推理与显存预算**：RTX 4080 P99 延迟为 **12.59 ms**（$\le 20.0\text{ms}$ 预算），峰值显存增量 **0.092 GiB**（$\ll 1.0\text{GiB}$ 预算），0 deadline misses；
- **全量测试回归**：全量单元测试 **333 tests: 332 passed, 1 skipped, 0 failed** 全部通过；`compileall` 与 `git diff --check` 零告警。

## 2026-08-13 — H2 GATE_PASSED 完整验收与交付终态

状态：`H2 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。H3、World、训练和在线 Oracle
仍未授权。本轮保留此前负结果，并在同一冻结协议上用新 dataset 完成定点修复和 full gate。

最终 dataset：`h2-gatepass-20260813-routefix`（本机 Git-ignored）。Evidence 位于
`docs/runtime-evidence/h2/h2-gatepass-20260813-routefix/`，数据位于
`generated/h2/paired-outcomes/h2-gatepass-20260813-routefix/`。

冻结身份：物理 manifest payload `6e74a789647182d9333cd99a69305bc2700a95216ebc7f34d2af21024a6d48ed`；
store manifest payload `22d11961c74509843a1df6ea453794fad2519fcc42077540c33ce46e9f3c3524`；
配置 `70996b2b2a0d88cd02c210e75206cc1be1f189fae249979d14c417c866092043`；final Evidence
`final-delivery.json` 在最终提交后重生成并绑定最终 HEAD。

在真实 CARLA 0.9.16、RTX 4080/CUDA 和 Town01 → Town03 → Town05 冷重启序列上完成：

- 三地图 restart smoke、唯一物理 manifest、15-anchor pilot；pilot 通过（15/15 terminal，
  9 valid/distinct，9 decisive），随后才扩大正式矩阵；
- 固定 `3 × 5 × 4 × 2 = 120` anchors 全部 terminal，三地图 full collector 均 `ok=true`；
- 离线 Oracle、slot/source/branch permutation、执行绑定轨迹 hash、CAS/Parquet manifest
  与 artifact hash audit 全部完成；
- dataset `h2-gatepass-20260813-routefix` 写入本机 Git-ignored 的
  `generated/h2/paired-outcomes/`，Evidence 写入 `docs/runtime-evidence/h2/`。

冻结身份：物理 manifest `f89a2f8a039144b089ae27c2584aa112a3d35d061d6a22cc6d12359fabd11a9f`；
store manifest `b63f4f63de12aa0a84762c398fecc7bb78ffbcc4d59f22b3552da1ba2c82727e`；配置
`9ff604d7d3d64122af76b41df58533d722de3f27c8d0c9dbe7e89e6a7552ceaa`；最终 audit Evidence
`docs/runtime-evidence/h2/h2-final-20260813-scenariov2-cleanup/offline-label-audit-full.json`
（Evidence SHA256 `e72db6ff4a185007008a8905855568554b384bad4ad8d1a49c4616dae3549338`）。最终
terminal summary `final-delivery.json` 的 Evidence SHA256 为
`6ceb3c74aede8c3895cbefb302a10f9ec98f0b5425af616fe1e9da77ec163f23`。

最终门指标：120 terminal；108 eligible/distinct；108 valid；Town01/Town03/Town05 为
36/34/38；family `cut_in/free_flow/red_light_hold/slow_lead/stopped_lead` 为 21/21/21/24/21；
weather ClearNoon/CloudyNoon 为 56/52；83 decisive；Expert/VLA wins 为 51/32；slot
Expert=0 比例 0.5；swap/permutation、trajectory hash、执行绑定和 manifest/artifact hash
全部通过；source-only baseline 0.6144578313；整卡 GPU peak 8.3720703125 GiB；dataset
1,480,172,014 bytes。全部冻结数据门通过。

本轮定点修复包括 route-relative traffic-light stop line、有限时域安全停止前缀、moving
family 闭环预滚、cut-in spawn probe cleanup 和 freshness-only trajectory stamp refresh。
全量串行测试为 `329 tests: 328 passed, 1 skipped, 0 failed`；compileall、文档链接和
`git diff --check` 通过。最终 manifest 完成全量校验。

CARLA 过程中一次 Town01 残留 actor 触发硬清场门；保留失败 Evidence，执行一次受控冷重启
并确认仅剩内建 traffic light/speed-limit actors 后从同一冻结 manifest 恢复，未改变数据身份。
未使用 `load_world`、第二 tick master、在线 Oracle、训练或 fake GPU/CARLA。

## 2026-08-13 — H2 Paired Outcomes 实现与验证阻塞（已被上方最终验收记录取代）

状态：`H2_IMPLEMENTED / NOT_VERIFIED / BLOCKED`。H3、World、训练和在线 Oracle 均未
授权。按 H2 全量测试硬门，本轮在 CARLA restart/smoke/pilot/120 矩阵之前停止。

本轮从未提交的 H1 工作树创建 `codex/h2-paired-outcomes`，完成：

- 固定 `3 maps × 5 families × 4 seeds × 2 weather = 120` 矩阵和 15-anchor pilot；
  pair hash 冻结 branch order 与 Expert slot，完整矩阵严格 60/60；
- 新增 H2 pair/branch/outcome/reset 合同、内容寻址 PNG、Zstd Parquet timeline/event/
  actor-future、每 pair 原子 shard、独立 offline label 和全 artifact manifest/resume 校验；
- H3 feature view 以 allowlist 新建，只含 observable anchor/history、route 和 candidate
  trajectory/hash；source、slot、order、future、outcome、Oracle、Regression、label/winner
  字段物理缺席；
- 离线 Oracle 按冻结 unsafe/completion/progress/comfort 规则只引用 candidate id/hash，
  branch/slot/source 置换测试保持结果不变；
- `ScenarioRuntime` 新增稀疏 collision/lane event buffer、锁保护范围读取和 ego/NPC 同 tick
  控制；地图不匹配 fail-closed，不再隐式 `client.load_world()`；
- 新增 `sdf sim restart`：只接受 READY、async、tick-owner-free、唯一且 executable/cmd/
  DefaultEngine 配置匹配的 Windows CARLA；仅请求 `CloseMainWindow`，验证 PID/TCP 退出，
  原子 pin 后 bounded ensure 和一次最终 preflight，不 force kill；
- 新增三地图物理 manifest、restart smoke、同 anchor 一次双来源生成、强制单候选 Safety、
  50-tick MPC/PID branch、offline label/audit 和断点恢复编排；live collector 不导入 Oracle，
  不创建第二 client/tick master，不调用 `world.tick()` 或 `load_world()`。

已运行验证：

- H2 专项：12/12 passed；完整 `tests/hybrid`：108 passed、1 个真实 GPU forward 门 skipped；
- Runtime/连接：47/47 passed；Classic/Safety/Control 专项：97/97 passed；
- 受影响 QP/RATO 模块隔离复跑：30/30 passed；
- `compileall` 和 `git diff --check`：通过；
- 干净串行全量：325 tests 中 312 passed、1 skipped、12 failed。失败包括 11 个不在计划
  允许清单内的 longitudinal-QP deadline timeout（约 `54.39–56.62 ms`），以及既有
  `test_static_obstacle_lateral_repair` RATO timeout（`69.48 ms`）。隔离复跑全部通过，说明
  是全套负载时序敏感，但 H2 计划只豁免既有 RATO 单项，不能把新增 11 项改名为允许失败。

停止结果：没有运行 CARLA preflight/restart，没有创建 H2 dataset 或 runtime Evidence，
没有修改 Safety deadline/阈值或测试来掩盖失败，没有 commit/push。后续必须先由单独任务
解决或明确处置全量 QP timing gate，再从全新 dataset id 执行三地图 restart smoke、pilot
和固定 120-anchor 矩阵；不得直接进入 H3。

当前研究事实：

```text
H0 = VERIFIED / STOPPED
H1 = VERIFIED / STOPPED
H2 = IMPLEMENTED / NOT_VERIFIED / BLOCKED
H3 = NOT_STARTED / NOT_AUTHORIZED
H4 = NOT_STARTED
H5 = NOT_STARTED
H6 = NOT_STARTED
```

## 2026-08-13 — H1 独立 HybridCandidateSet

状态：`H1_VERIFIED / STOPPED`。H2、World、Oracle 在线路径、训练和 paired rollout 均未
授权、未实施。

本轮在分支 `codex/h1-hybrid-candidates` 完成：

- 新增强绑定 observation/frame/CARLA frame/simulation time、ego、route revision、
  actor/light snapshot 与传感器帧/时间戳的 `ObservableAnchor`；
- Classic Frenet/ST 与 nominal SimLingo 各自独立生成一条候选，VLA 每个 anchor 只做一次
  `predict_native`；公开 `expert | vla` 与 Safety v1 的 `CLASSIC | VLA_FAST` 映射兼容；
- 严格 canonicalize 为 `map / T=10 / dt=0.25 s / horizon=2.5 s`，退化、覆盖不足与外推
  fail-closed，并冻结 raw/canonical/generator/route hashes 与 resampling provenance；
- 新增按合同、binding/freshness、route/navigation、dynamics/trackability、observable
  collision、isolated controller dry-run 排序的逐候选 Guard；
- 新增 0/1/2 PASS router、`0.5 m / 0.5 m/s` 近重复门、冻结 Safety soft score 与稳定
  tie-break；Safety 只收到选中的单候选，控制链禁止 first-available；
- `ScenarioRuntime` 支持只读 blueprint attributes 与受锁保护的精确帧 measurement；
- 新增 Town03 smoke，使用原生 SimLingo 相机位姿/分辨率/FOV、20 Hz 单 tick master，并
  保存本机只读 Evidence。

验证结果：

- `tests/hybrid`：97 passed、1 个真实 GPU 20-forward 环境门测试 skipped；
- H1/nominal/runtime 直接回归：35 passed；Classic/Safety 相关回归：59 passed；
- `compileall` 与 `git diff --check`：通过；
- 10 个活动 Markdown 文档本地链接检查：`BROKEN_LOCAL_LINKS 0`；H1 runtime 的
  World/Oracle/archive import 与第二 `carla.Client`/tick master 扫描：0 命中；
- 全量 310 tests：308 passed、1 skipped、1 failed。唯一失败仍是既有
  `test_static_obstacle_lateral_repair` 在整套负载下以 51.10 ms 命中 RATO deadline；单独
  复跑 1/1 passed（44 ms 测试总时长），未放宽 deadline 或修改测试；
- CARLA gate：唯一一次 `ensure --map Town03 --rhi dx12` 后 READY，唯一一次 preflight
  复查仍 READY；CUDA admission 为 RTX 4080、约 15.77 GB free，模型资产齐全；
- 正式 smoke `h1-smoke-20260812T161321Z`：两来源成功，Classic/VLA wall latency
  `0.817646 / 1.111187 s`，VLA forward count `1`，两条均 Guard PASS，候选差异
  `2.833430 m / 1.174353 m/s`，Safety `ACCEPT` 且 selected/executed/applied id 连贯，
  控制 `TRACK_APPROVED`，VLA peak `2218.178 MB`，runtime cleanup `COMPLETED`。

验证 JSON SHA256：
`2be0a5171856848bf52fb1ac48bbc88e714d65b8ff7c1811b89baae0bc857db7`。
此前两次失败 smoke 因相机额外 `sensor_tick` 的精确首帧 barrier timeout 被保留为
`INTERRUPTED` Evidence；相机改为随唯一 20 Hz world tick 每帧采样后通过。

当前研究事实：

```text
H0 = VERIFIED / STOPPED
H1 = VERIFIED / STOPPED
H2 = NOT_STARTED / NOT_AUTHORIZED
H3 = NOT_STARTED
H4 = NOT_STARTED
H5 = NOT_STARTED
H6 = NOT_STARTED
```

停止点：H1 已完成，不自动进入 H2。

## 2026-08-12 — H0 路线收敛

状态：`H0_CONSOLIDATED / VERIFIED / STOPPED`。

用户明确要求活动项目只保留 H 路线，并把不再需要的文件移入 `archive/`。本轮已完成：

- 冻结本轮开始时的 tracked patch 与 untracked tar，并记录 SHA256；
- 将旧路线文档、runtime Evidence、候选生成/训练源码、World 源码、阶段脚本、配置和
  测试移动到 `archive/2026-08-12-h-route-consolidation/`；
- 保留 nominal SimLingo policy、通用 trajectory contract、Classic/Safety 与 MPC/PID；
- 移除 nominal runtime 对已归档 learned-candidate 特征/权重导出的依赖；
- 新建 `tests/hybrid/`，把仍有效的 nominal VLA 与控制器回归迁入；
- 将活动入口统一为 H0–H6，并新增 HybridCandidateSet 与 candidate-conditioned World 合同。

归档结果：25097 个常规 payload、3100957261 bytes，另有 868 个符号链接。Git 版本化
272 个可移植 legacy 文件、归档说明、完整 manifest、符号链接表和两个工作树恢复快照；
20122 个 runtime Evidence 文件与 4572 个 generated 文件继续作为本机只读归档，不进入
普通 Git。`MANIFEST.sha256` 的 SHA256 为
`a313ccaf5e2d37f14901b9830ed29125243d9710f3a7de6b4ceb905fd86dc251`，符号链接表为
`bff2e26a9c7ff86a55ddb38fd4e2e09482468587e4dfc53c827ec9e4d00c53fb`。恢复映射、快照
hash、clone 边界与恢复约束已写入归档 README。

验证结果：

- 完整归档 `sha256sum -c MANIFEST.sha256`：25097/25097 个常规 payload 通过；
  `SYMLINKS.tsv` 冻结 868 个符号链接；
- 可移植归档、tracked patch 和解包后的 untracked tar 高置信度凭据扫描：0 个命中文件；
- `PYTHONDONTWRITEBYTECODE=1 /home/sdf/.venvs/sdf/bin/python -m unittest discover
  -s tests/hybrid -t . -v`：81 tests passed，1 个真实 GPU 20 次 forward 按环境门跳过；
- `/home/sdf/.venvs/sdf/bin/python -m compileall -q safedrive_foundry scripts tests`：通过；
- 活动文档本地链接检查：`BROKEN_LOCAL_LINKS 0`；
- 活动树旧路线标识、旧候选/World import 与 Python cache 扫描：无命中；地图测试中的
  OpenDRIVE road-node 主键明确保留；
- `git diff --check`：通过；冻结归档使用 archive 专用 whitespace 属性，未为通过检查而
  改写历史源码或 recovery patch；
- 全活动测试 `PYTHONDONTWRITEBYTECODE=1 /home/sdf/.venvs/sdf/bin/python -m unittest
  discover -s tests -t .`：293 tests 中 291 passed、1 skipped、1 failed。失败是未修改的
  Safety/RATO 静态障碍测试在整套负载下以 47.52 ms 命中 solver deadline；该测试单独
  复跑通过，因此记录为既有时序敏感问题，未通过放宽 deadline 或改测试掩盖。

当前研究事实：

```text
H0 = VERIFIED / STOPPED
H1 = NOT_IMPLEMENTED
H2 = NOT_STARTED
H3 = NOT_STARTED
H4 = NOT_STARTED
H5 = NOT_STARTED
H6 = NOT_STARTED
```

限制：本轮没有启动 CARLA、没有训练模型、没有采集 H 数据，也没有把 archive 中的历史
结果重命名为 H Evidence。完整活动测试仍有上述一个时序敏感失败；H 专项回归与静态
验证均通过。

该 H0 记录的下一步已由本轮 H1 完成；当前下一步只能由新任务单独授权 H2，不得自动
采集 paired outcome 或进入 World 训练。
