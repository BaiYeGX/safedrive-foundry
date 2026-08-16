# H Evidence 与归档索引

## 1. 活动 Evidence 规则

H 路线只承认新建且 provenance 完整的 Evidence：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

每个正式 artifact 必须绑定 worktree/commit、config、split、seed、CARLA/模型版本、输入
observation、candidate、Guard、selector、Safety、executed trajectory、outcome、延迟与资源。
没有实际运行的数字不能进入 `MEASURED`，没有冻结复核不能进入 `VERIFIED`。

H0 仅是仓库收敛，不产生驾驶性能数字。H1 Evidence 只验证独立候选、Guard、选择和
一个控制 tick 的合同，不是驾驶性能评估或 H2 训练数据。H2 runtime Evidence 与数据按
dataset id 分别位于 `docs/runtime-evidence/h2/` 和 `generated/h2/paired-outcomes/`；二者
均为本机 Git-ignored artifact。H3 World Scorer 离线训练与评测 Evidence 位于
`docs/runtime-evidence/h3/`，checkpoint 位于 `generated/h3/`。

### H3 最终验收（H3v2）

状态：`H3 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。H4 locked evaluation 已在后续完成；H5 Closed-Loop 仍未授权。

最终运行：`h3-v2-20260815d-final`，联合 `h2-gatepass-20260813-routefix` 与
`h3-challenge-v2-20260815d-dev`（96 场真实 CARLA Challenge，见 Challenge audit）。

本机 artifact（均 Git-ignored）：

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/
docs/runtime-evidence/h3/h3-challenge-v2-20260815d-dev/
generated/h3/h3-v2-20260815d-final/checkpoints/
generated/h3/carla-challenge-v2/h3-challenge-v2-20260815d-dev/
```

冻结 hash：

```text
evidence_sha256       f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e
challenge manifest     a87871c9c858d440f7b4d6553663ca63e43469f510e950e19d84ee06d3aa35ef
challenge physical     7b27cc140dda3bc1bcf1d1587f95616dcd3c120c9596b0642caf63bd98a029d9
```

最终门指标：

| 检查项 | 要求 | 实测 | 状态 |
|---|---:|---:|---|
| leakage | 0 failures | passed | PASSED |
| swap | max error <= 1e-6 | 0.0 | PASSED |
| OOF accuracy | >= best baseline + 2pp | 1.0000 vs 0.9231 | PASSED |
| bootstrap lower 95 | >= 0 | 0.0330 | PASSED |
| action sensitivity | >= 5pp | 28.57pp | PASSED |
| history sensitivity | >= 2pp | 28.57pp | PASSED |
| ECE | <= 0.10 | 0.00113 | PASSED |
| seed stability | >= 4/5 | 5/5 | PASSED |
| P99 / VRAM / deadline | <= 50ms / <= 1.5GiB / 0 | 13.89ms / 0.0566GiB / 0 | PASSED |

Challenge 数据质量门：96 terminal；78 valid；87 distinct；72 decisive；
hard-unsafe branches 13；Expert/VLA wins 53/19；source-only baseline 0.7361；
store manifest 与 label permutation 全通过。

World scorer 模型包含显式 scene gate：history masking 使 observable context 为 0 时，
candidate 分支被合同性关闭并退化为 no-action，因此 history sensitivity 不是靠调参
偶然得到的。

额外 learned 基线：candidate-only MLP OOF `1.0000`、full-feature MLP OOF `0.9560`；
排序超越门按冻结口径只与最佳非学习简单规则 `candidate_only=0.9231` 比较，两个 MLP
作为诊断对照报告，不替代简单规则门槛。

### H4 Locked Evaluation（H4）

状态：`H4 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。H5 Closed-Loop 仍未授权。

最终运行：`h4-locked-20260816-final`。

本机 artifact（均 Git-ignored）：

```text
docs/runtime-evidence/h4/h4-locked-20260816-final/
generated/h4/h4-locked-20260816-final/
```

冻结 hash：

```text
evidence_sha256       35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4
h3_evidence_sha256    f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e
split_manifest_sha256 17dedd305aaf2933266a15345926f035aa7ebcd3210b6c636cc92d99e676b08c
```

最终门指标：

| 检查项 | 要求 | 实测 | 状态 |
|---|---:|---:|---|
| isolation | 0 failures | passed | PASSED |
| sufficient test power | >= 20 decisive | 64 | PASSED |
| ranking vs best simple baseline | World > best simple | 1.0000 > 0.890625 | PASSED |
| resource | <=50ms / <=1.5GiB / 0 miss | 15.23ms / 0.03125GiB / 0 | PASSED |
| defer coverage | report | 0.984375 | report |
| end-to-end vs fallback | report | 0.984375 > 0.890625 | report |
| h5_authorized | — | true | report |

H4 同时修复了 H3 raw 5-seed ensemble uncertainty 全 defer 的问题：使用 dev-only
per-model utility normalization（冻结）后再集成，dev 和 locked test 的 defer coverage
均恢复至接近/等于 1.0。

### H2 Evidence 边界

H2 每个 dataset 必须包含三地图 restart smoke、物理 scenario manifest、逐地图 pilot/full
collector Evidence、每 pair live shard、独立 offline label 和最终 audit。`manifest.json`
覆盖 dataset 内全部 artifact；Evidence 同时绑定 HEAD、tracked diff SHA256、untracked
manifest SHA256、CARLA/config/model/route/candidate hashes、执行链、延迟和显存。

H2 状态只有在 120 个固定 anchor 全部 terminal 并完成 offline label/audit 后才可写为
`VERIFIED`。Pilot 或环境阻塞不能借用 H1 smoke、archive 或合成数据升级为 H2 证据；门失败
也必须冻结为负结果并关闭 H3。

### H2 最终验收（GATE_PASSED）

最终 dataset `h2-gatepass-20260813-routefix` 已完成 Town01、Town03、Town05 冷重启、
restart smoke、15-anchor pilot 和完整 120-anchor 固定矩阵；pilot 与 full gate 均通过。
H3 保持 `NOT_AUTHORIZED`，不进入 World、在线 Oracle 或训练。

本机 artifact（均 Git-ignored）：

```text
generated/h2/paired-outcomes/h2-gatepass-20260813-routefix/
docs/runtime-evidence/h2/h2-gatepass-20260813-routefix/
```

冻结 hash：

```text
physical_manifest_sha256 6e74a789647182d9333cd99a69305bc2700a95216ebc7f34d2af21024a6d48ed
store_manifest_sha256    22d11961c74509843a1df6ea453794fad2519fcc42077540c33ce46e9f3c3524
config_sha256            70996b2b2a0d88cd02c210e75206cc1be1f189fae249979d14c417c866092043
offline_audit_file_sha256 3dc0573b5fe7a80fc3358f1e11d1c981d1fbe900f07357acc30a7b40d389b585
final_delivery             docs/runtime-evidence/h2/h2-gatepass-20260813-routefix/final-delivery.json
```

完整门指标：120/120 terminal；108 eligible/distinct；108 valid；Town01/Town03/Town05
为 36/34/38；family 为 21/21/21/24/21；weather 为 56/52；83 decisive；Expert/VLA
wins 为 51/32；slot Expert=0 比例 0.5；swap/permutation、trajectory hash、执行绑定和
manifest/artifact hash 全部通过；source-only baseline 0.6144578313；整卡 GPU peak
8.3720703125 GiB；dataset 1,480,172,014 bytes。

全量测试为 `329 tests: 328 passed, 1 skipped, 0 failed`；compileall、活动文档链接检查、
旧路线/import/cache 扫描和 `git diff --check` 通过。

### H2 之前的失败数据集（保留负结果）

最终 dataset `h2-final-20260813-scenariov2-cleanup` 已完成 Town01、Town03、Town05 冷重启、
restart smoke、15-anchor pilot 和完整 120-anchor 固定矩阵。pilot 通过后才扩大；完整数据门
按冻结规则失败，因此状态为 `H2 COMPLETED / VERIFIED / GATE_FAILED / STOPPED`，H3 关闭。

本机 artifact（均 Git-ignored）：

```text
generated/h2/paired-outcomes/h2-final-20260813-scenariov2-cleanup/
docs/runtime-evidence/h2/h2-final-20260813-scenariov2-cleanup/
```

冻结 hash：

```text
physical_manifest_sha256 f89a2f8a039144b089ae27c2584aa112a3d35d061d6a22cc6d12359fabd11a9f
store_manifest_sha256    b63f4f63de12aa0a84762c398fecc7bb78ffbcc4d59f22b3552da1ba2c82727e
config_sha256            9ff604d7d3d64122af76b41df58533d722de3f27c8d0c9dbe7e89e6a7552ceaa
offline_audit_sha256     e72db6ff4a185007008a8905855568554b384bad4ad8d1a49c4616dae3549338
final_delivery_sha256    6ceb3c74aede8c3895cbefb302a10f9ec98f0b5425af616fe1e9da77ec163f23
```

门指标：120/120 terminal；58 valid/distinct；Town01/Town03/Town05 为 21/23/14；family
为 14/12/4/13/15；weather 为 33/25；58 decisive；Expert/VLA wins 为 58/0；slot Expert=0
比例 0.4827586；swap/source/branch permutation 通过；trajectory hash 100%；source-only
baseline 100%；整卡 GPU peak 8.3945 GiB；dataset 1,443,906,253 bytes。manifest、artifact
hash、执行绑定和 cleanup 全通过。失败门为 valid/distinct 数量、地图/family/weather 配额、
VLA wins 和 source-only baseline，均按原阈值记录，没有补采或改门槛。

正式 Evidence 包含每地图 pilot/full collector、冷重启/smoke、pair shard、label shard、
store manifest、full offline audit 和历次失败/中断记录。Town01 一次残留 actor 清场失败被
保留；随后一次安全冷重启、只读清场核验后在同一物理 manifest 上恢复，未伪造或重写成功记录。

### H1 Town03 live smoke

验证 Evidence（UTC run id `20260812T161321Z`）：

```text
docs/runtime-evidence/h1/h1-smoke-20260812T161321Z/h1_smoke.json
sha256 2be0a5171856848bf52fb1ac48bbc88e714d65b8ff7c1811b89baae0bc857db7
```

- CARLA 0.9.16 / Town03 / RTX 4080，anchor、front camera 与两候选共同绑定 frame
  `112794`；
- Classic 与真实 SimLingo 各生成一次，wall latency 分别为 `0.817646 s`、`1.111187 s`，
  VLA forward count 为 `1`；
- 两条候选均为 Guard `PASS`，差异为最大位置 `2.833430 m`、RMS 速度 `1.174353 m/s`，
  因而是 `DISTINCT`；
- H1 冻结 selector 选择 Expert，Safety 为 `ACCEPT`，selected/final/post-repair/executed/
  applied id 连贯，实际控制为 `TRACK_APPROVED`；
- VLA forward peak 为 `2218.178 MB`，运行结束 registry 为 `COMPLETED`；复查 CARLA 已恢复
  asynchronous、tick owner free。

失败 Evidence 同样保留：

```text
h1-smoke-20260812T160549Z/h1_smoke.json
sha256 6384affd3afbe47d1f81d076ea7b5df572c1d5a23085f4b2d32567ca839cd898

h1-smoke-20260812T160859Z/h1_smoke.json
sha256 9c281cd69a88880acc4a1f7ad3508477144f27e729ac2d542add7a56b5c2d803
```

两次都因相机首帧 barrier timeout 中止，registry 为 `INTERRUPTED`。根因是相机额外
`sensor_tick=0.05` 在同步 world 首帧未发回精确 frame；最终让相机随唯一 20 Hz world
tick 每帧采样后通过。失败没有改名为成功或删除。

## 2. 2026-08-12 路线收敛归档

归档根：

```text
archive/2026-08-12-h-route-consolidation/
```

| 内容 | 状态 | 恢复方式 |
|---|---|---|
| 旧设计文档 | Git 版本化的可恢复历史 | 按 archive README 的原相对路径复制 |
| runtime Evidence/checkpoint | 本机只读，不进入普通 Git | 从本机 `legacy-active/docs/runtime-evidence/` 恢复 |
| 旧候选生成/训练/World 源码 | Git 版本化，不参与活动 import | 从 `legacy-active/source/` 恢复 |
| 旧脚本、配置与测试 | Git 版本化，不是活动入口 | 从对应 `legacy-active/` 子目录恢复 |
| 本轮开始时 tracked dirty worktree | patch 快照 | `recovery/tracked-worktree.patch` |
| 本轮开始时 untracked 文件 | tar 快照 | `recovery/untracked-worktree.tar.gz` |
| 本地生成环境/runtime 输出 | 本机只读，可恢复或重建 | `generated/` |

恢复前必须先阅读
[`archive/2026-08-12-h-route-consolidation/README.md`](../archive/2026-08-12-h-route-consolidation/README.md)，
在临时目录验证，不得直接覆盖活动 H 文件。

工作树快照校验：

```text
tracked-worktree.patch
sha256 1cd44ae9f0f5bcea4589dcbf1f90087259e9d969817110b0521d9155436272aa

untracked-worktree.tar.gz
sha256 0d1fde09623dceb11527bae5cb33ed817630b1a417e5ba78a79011259c828fe7
```

完整归档冻结边界：25097 个常规 payload、3100957261 bytes，另有 868 个符号链接。
`MANIFEST.sha256` 覆盖全部常规 payload；`SYMLINKS.tsv` 冻结符号链接路径与目标：

```text
MANIFEST.sha256
sha256 a313ccaf5e2d37f14901b9830ed29125243d9710f3a7de6b4ceb905fd86dc251

SYMLINKS.tsv
sha256 bff2e26a9c7ff86a55ddb38fd4e2e09482468587e4dfc53c827ec9e4d00c53fb
```

可移植归档为 272 个 legacy 文件、8860549 bytes；
20122 个 runtime Evidence 文件（2960996577 bytes）和 4572 个 generated 文件
（126980647 bytes）只保存在本机。

干净 clone 可以恢复可移植旧源码/文档/配置/测试和两个工作树快照，但不会包含 3GB
历史 runtime Evidence。`evaluation/` 与 `evaluation-remaining/` 的合并恢复规则，以及
迁移到 `tests/hybrid/` 的回归边界，以 archive README 为准。

## 3. 历史材料的解释边界

- archive 保存失败、负收益、冻结阈值和旧实现，内容不重写；
- archive 不是活动任务、接口、数字或门槛来源；
- 历史候选生成失败只支持“该旧方法停止”，不证明 H World 有效；
- 历史 World 无收益只支持“旧数据/旧条件化不足”，不等同于 H3 的结果；
- 若未来需要引用历史事实，必须同时引用原 artifact、状态和限制，不能改名为 H 指标。

## 4. H 当前状态

| 阶段 | Evidence 状态 |
|---|---|
| H0 route consolidation | `VERIFIED / STOPPED` |
| H1 independent candidates | `VERIFIED / STOPPED` |
| H2 paired outcomes | `COMPLETED / VERIFIED / GATE_PASSED / STOPPED` |
| H3 World development | `COMPLETED / VERIFIED / GATE_PASSED / STOPPED` |
| H4 locked evaluation | `COMPLETED / VERIFIED / GATE_PASSED / STOPPED` |
| H5 World on/off | `NOT_STARTED` |
| H6 closure | `NOT_STARTED` |
