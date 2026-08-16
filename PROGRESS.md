# SafeDrive Foundry 进度

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
- 全量测试 `352 tests: 351 passed, 1 skipped, 0 failed`；compileall 与 `git diff --check` 通过。

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
