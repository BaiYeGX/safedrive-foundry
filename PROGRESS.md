# SafeDrive Foundry 已确认进度

本文只保留当前可操作事实、冻结结果和下一接管点。2026-08-27 以前的逐轮日志已保存在
[`archive/2026-08-27-cora-document-consolidation/`](archive/2026-08-27-cora-document-consolidation/README.md)
的原始快照和阶段文档中，不能作为活动任务或阈值来源。

## 2026-09-10 — C3 修复完成：监督、训练与验收重新验证（VERIFIED）

状态：

```text
H6-CORA C3 repair = VERIFIED / ENGINEERING_COMPLETED / ALGORITHM_MEASURED / STOPPED
authoritative run = c3-repair-20260910T100651Z
next entry = C4（本轮不自动执行）
```

本次修复撤回旧 `c3-vla-sft-20260909-final-v3` 及此前 repair attempts 的验收和 90.77%
路线改善结论。旧适配曾把未按当前位置投影裁切的导航/路线前缀带入监督；旧产物和 hash 仍
保留作不可覆盖的勘误历史，但不再用于模型选择、比较或 C3 完成判定。修复代码将当前
输入、专家监督、离线后果和审计元数据分开，按有序路线投影去除已通过前缀（保留合法转弯），
不以负 x 删除点，不以导航替代专家标签；原生 20 点 route 使用 0…19 m，canonical speed
使用 10 点、0.25 s 间隔，缺失和提前终止按逐头 mask 保留。

修复版复用冻结 C2 release `h6-cora-c2-devbaseline-20260907-v1`，保持 train 158 /
validation 53 和 211 个 root 身份。manifest SHA-256 为
`54281a1de6207b4e0c553ee8456de40f43131a6ef30b434e7bd1064da776c2fd`；route 有效点为
3160/1060，speed 有效点为 1550/530，3 个碰撞终止 train root 的 speed 全部 mask，route
标签仍绑定独立的 audited native expert reference path。全部 timeline 的原生 dt 为 0.05 s，
但相对 anchor 均为 branch preroll，故 P-SPEED 为 N/A；anchor observable speed 的 211 个
零值及其与 history 的 211 个差异均披露，未用未来执行速度替换输入。无须补采或启动 CARLA。

真实 RTX 4080 CUDA/BF16 smoke 完成 forward/backward、小步更新、adapter 保存和独立重载；
336 个 LoRA key、10 个驾驶 head key 无 missing/unexpected，18,417,664 个可训练参数有限
变化，视觉投影、基座和原生 query embeddings 指纹不变。C4 成本 smoke 的四输出 residual
起点误差为 0，shared LoRA gradient 非零，分组 clip 后范数均不超过 1.0。正式 M1 从原始
M0 重新开始，以 seed 17、AdamW、LoRA `2e-5`、驾驶头 `1e-4`、weight decay `0.01`、
microbatch 1、累积 4 完成 200/200 updates；前 10 步 head-only，第 11 步解冻 LoRA。训练
保留 158 roots 每轮的 39 个四样本窗口和一个两样本尾窗口，5 轮共 790 个样本暴露，root
exposure 全部为 5；checkpoint 保存 optimizer、RNG、游标、日志边界和完整冻结指纹。

修复版在固定 53 个 validation root 上运行同输入的 M0、重复/reload M0 和 M1；root 内聚合
后 root 等权，配对 bootstrap 1000 次、seed 71，`eps_ADE=1e-5 m`：

| 指标 | M0 | M1 | Δ（M0−M1） | 相对改善 | 95% 配对 bootstrap CI |
|---|---:|---:|---:|---:|---:|
| P-ADE route (m) | 0.210514 | 0.232211 | -0.021698 m | -10.3069% | [-0.159394, 0.103600] m |
| P-FDE route (m) | 0.735295 | 0.806690 | -0.071395 m | -9.7097% | [-0.625551, 0.471091] m |
| P-WP speed (m) | 2.178557 | 0.336781 | 1.841776 m | 84.5411% | [1.570158, 2.128422] m |
| P-VALID | 100.0% (53/53) | 100.0% (53/53) | — | — | — |
| P-FAIL | 0.0% (0/53) | 0.0% (0/53) | — | — | — |

P-SPEED 为 `N/A_without_trusted_time_speed_ground_truth`。route ADE/FDE 在本次单 seed 开发
集上变差，speed-waypoint 位置误差下降；这不是“全面改善”，也不把旧 90.77% 作为可比结果。
工程验收与算法收益分开，负 route 结果、所有逐 root 数值、mask、失败原因和独立重算均保留。

权威修复产物在本机唯一目录
`generated/h6/cora/c3-repair-20260910T100651Z/`，verify content SHA-256 为
`29f9c56e4bfff8780c1a6a3f902872f0f81b4b56e271bdaa4ff5d521341f8af3`，stage summary 为
`VERIFIED`。resource ledger 的优化总账保守上界为 `8.622364703124443 h`，观测峰值 allocated /
reserved 为 `9.889132/10.845703 GiB`；历史失败和已撤回运行保留在账本，未知失败按四小时上界
计入。C3 不执行 C4；C4 的 M2 必须从原始 M0 起点开始，不能从 M1 续训。

验收器篡改回归已实际执行：预测、重复 root、manifest、metrics、checkpoint identity、训练
日志删尾和资源监测字段篡改均被拒绝；删除 prediction-failure sidecar 也被 CLI 拒绝。恢复原件
后再次 `verify` 返回 `VERIFIED` 且 `errors=[]`。验证命令
`/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests -t . -v` 实际运行 509 tests、
1 skipped、`OK`（75.094 s）；`compileall -q safedrive_foundry scripts/h6_cora_sft.py` 与
`git diff --check` 均通过。

## 2026-09-09 — C3 初版结果（SUPERSEDED：监督错误，验收与 90.77% 结论撤回）

> 本节仅作历史勘误。其 manifest、指标和 `VERIFIED` 状态均不再是活动证据；请以 2026-09-10
> 的 `c3-repair-20260910T100651Z` 及 [C3 修复勘误](docs/runtime-evidence/h6/c3-sft-repair-erratum.md)
> 为准。

状态：

```text
H6-CORA C3 = VERIFIED / ENGINEERING_COMPLETED / ALGORITHM_MEASURED / STOPPED
authoritative run = c3-vla-sft-20260909-final-v3
next entry = C4（本轮不自动执行）
```

本轮使用冻结 C2 dev release `h6-cora-c2-devbaseline-20260907-v1`，没有重跑 C2、重扫旧
Hash 或使用 calibration/locked/pilot。适配层从独立的 `train` 与 `validation` split 读取
158/53 个 root（共 211 个 sample），检查 base anchor、navigation、当前图像、ego/history、
canonical expert proposal 与执行时间线的身份绑定。manifest SHA-256 为
`e5116aedd2b79e1512f4f0567f1b740880623fce4f977e9470bf6786f21c1bb3`，release index SHA-256
为 `0c867199d9b6682648471b21a2ab850c86bf8f1eb4d8eb99d73abe3e7c8789b8`。route 使用原生 20
个空间点（含当前 ego 原点，弧长 0…19 m），speed 使用执行时间线上的原生 10 点（0.25 s
间隔）；有效 route 点为 train/validation `3160/1060`，有效 speed 点为 `1550/530`。
3 个提前终止的 train sample 保留 10 个 speed 缺失 mask；缺失没有被填充为有效标签。输入中
没有 future 或 World label，audit 为 `AUDIT_PASSED`，所有 53 个 validation root 与 train
root 隔离，执行时间线行数为 1…50，实测 dt 为 0.05 s（浮点表示差异已记录）。C2 已有完整
可信 expert timeline，因此没有执行补采，也没有启动 CARLA；配置仍记录确认的
`E:\\CARLA_0.9.16\\CarlaUE4.exe` 路径和“不需要补采”的原因。

真实 GPU smoke 在 RTX 4080 上通过：CUDA 可用，模型使用 BF16，checkpoint 的 336 个 LoRA
匹配项与 10 个驾驶 head 匹配项均无 missing/unexpected key；LoRA 参数 17,596,416 个，
驾驶 head 参数 821,248 个，共 18,417,664 个可训练参数，冻结视觉投影、基座和 query
embeddings。真实 batch 完成 forward/backward、小步更新、adapter 保存和独立 fresh reload，
round-trip 最大绝对差为 0（容差 `1e-5`），峰值 allocated/reserved 为
`4.319/4.914 GiB`。C4 共享 hidden 的诊断梯度通路也测到非零 auxiliary gradient（hidden
896，SFT norm 21.1636，aux norm 2.9575，cosine -0.3814，门控权重 w=0，q=1）；它只用于
共同预算登记，没有混入 C3 SFT loss。

正式 M0/M1 使用 seed 17、AdamW（LoRA `2e-5`、驾驶 head `1e-4`、weight decay `0.01`）、
microbatch 1、累积 4、200 个有限有效更新和冻结日程（前 10 步 head-only，第 11 步起 LoRA
线性预热并余弦衰减）。M1 达到 `200/200` 更新，独立 checkpoint 可重载，LoRA 与驾驶 head
均实际变化，冻结参数指纹未变。训练耗时 267.697 s（约 4.46 min），峰值 allocated/reserved
为 `4.259/4.398 GiB`，低于 14.5 GiB 峰值限制；adapter 为 72,156,073 bytes，完整恢复
checkpoint 为 216,521,989 bytes。资源账本记录 C3/C4 合计 10 h GPU 优化预算和本次实际消耗。

M0/M1 在固定 53 个 validation root 上先 root 内聚合、再 root 等权；配对 bootstrap 固定
1000 次、seed 71，重复 M0/reload 得到 `eps_ADE=1e-5 m`。结果（越低越好）：

| 指标 | M0 | M1 | 绝对改善 M0−M1 | 相对改善 | 95% 配对 bootstrap CI |
|---|---:|---:|---:|---:|---:|
| P-ADE route (m) | 5.032139 | 0.464337 | 4.567802 m | 90.7726% | [4.204761, 4.863890] m |
| P-FDE route (m) | 5.576035 | 0.585869 | 4.990166 m | 89.4931% | [4.614955, 5.325819] m |
| P-WP speed (m) | 1.984764 | 0.531818 | 1.452946 m | 73.2050% | [1.240386, 1.675059] m |

两组均为 53/53 root，预测失败率均为 0%；P-SPEED 按合同为 `N/A`，因为没有可信的真实速度
换算 ground truth。上述是一次 seed 的开发集结果，不代表独立测试或训练稳定性；正向幅度远
超出预登记 0.01 m / 2% 规划目标，但工程验收不依赖正收益，所有逐 root 明细、有效点数、
mask 和失败原因均保留。

权威产物均在本机唯一目录
`generated/h6/cora/c3-vla-sft-20260909-final-v3/`：`manifest.json`、`run_config.json`、
`audit.json`、`smoke.json`、`m0_predictions.json`、`m1_adapter.pt`、`checkpoint_latest.pt`、
`m1_training_summary.json`、`training.jsonl`、`m1_predictions.json`、`metrics-spec.json`、
`metrics.json`、`resource-ledger.json`、`evaluation-report.json`、`verify.json` 和
`stage-summary.json`。核心内容 hash 见 `verify.json`，其状态为 `VERIFIED`；stage summary
绑定代码 SHA-256 `a41d4740993b3d3d67950940abb5b7c2ff4a1d9b0a42e66e57f912a4c862bf81`、运行时
Git HEAD `598308fd4cb57df94f02f784404635e942c3ff9c`、模型 SHA-256
`ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28` 与 release 身份。
大型权重和本地产物按规则保持 ignored，不进入 Git；报告和复现入口随代码提交。

实际验证命令：

```text
/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests -t . -v
# Ran 506 tests in 73.859s; OK (skipped=1)
/home/sdf/.venvs/sdf/bin/python -m compileall -q safedrive_foundry scripts
git diff --check
```

下一入口已切到 C4；根据单任务循环，本轮在 C3 验收后停止，不自动开始联合训练。

## 2026-09-08 — 论文与源码联合研究后的实施优化（DOCUMENTATION COMPLETE）

本轮检索并核对驾驶 VLA/WAM、专家学生信息不一致、表示保持、LoRA 初始化与辅助梯度
文献；重点查看部分原文方法段，并读取本地 SimLingo adaptor/dataset、在线运行器、
C2 ridge 拟合和 release 元数据。研究范围包括近期预印本，不把摘要结果当本项目实测。
文档只落实实施决定，未增加论文列表或新任务。

源码确认 route 为 20 个空间点（弧长 0…19 m），speed 为 10 个时间点，驾驶 head 对增量
做 cumsum；在线运行器有全参数冻结/inference_mode，不能直接充当训练入口。
上游 future 缺文件可复制上一记录、旧 ridge 拟合未逐头 mask，新适配不得将缺失填作有效标签。
这些是适配风险与源码事实，不表示当前未实现的 C3 已发生同样训练错误。

调整 C3/C4 合同：驾驶/World 分组裁剪，smoke 后恢复 M0、显式可训练白名单和 query 冻结、
root 无放回轮转、实际累积窗口归一化、训练部署输入一致性与原生采样/累加检查。
2% 目标/3% 争取目标、两次训练、6 roots/18 attempts 和所有预算不变。
不移植新视频 backbone、RL、多模型 teacher 或覆盖已有 LoRA 的初始化方法。
本轮只修改 10 份既有活动文档，未修改运行代码或冻结 Evidence；方案 PLANNED，收益 NOT_MEASURED。

验证：执行 CPU 裁剪代数断言（不是模型测试），检查文档链接/围栏/阶段编号与 diff/stat，
git diff --check；原 EVIDENCE 全文保留为前缀。未训练、未跑 CARLA、未设置 goal、未 commit/push。
当前仍 C3 NOT_STARTED；真实监督可用量、梯度链路、显存/耗时与方法收益待对应阶段实测，
本轮无用户接管操作。

## 2026-09-08 — 离线改善规划提高至 2%–3%（DOCUMENTATION COMPLETE）

按用户要求，将 PROJECT 中 M1/M0、M2/M1 的 P-ADE 及 b+R/b 的进度 W-MAE
相对改善规划由 1% 提高为目标 ≥2%、争取 ≥3%；仍须同时满足绝对下降 ≥0.01 m。
补充相对/绝对幅度算例。C5 的进度目标、微弱趋势口径、两次训练和资源预算保持不变。
当前尚未正式训练，此调整为 PLANNED 目标变更，收益仍 NOT_MEASURED，C3 NOT_STARTED。
本轮只修改 PROJECT 与本进度记录；检查目标文本、diff/stat 与 git diff --check，
未启动训练/CARLA、未改历史 Evidence，无需用户接管。

## 2026-09-08 — 论文量化合同完善（DOCUMENTATION COMPLETE）

用户要求将项目优势落实为量化指标。本轮只修改 12 份既有活动文档，在 PROJECT 建立
c3_c6_metrics_v1 计划合同，并同步 C3/C4/C5/C6、数据、资源、证据和入口。
定义原生策略误差、四头后果误差、配对排序/遗憾、闭环进度/违规/舒适性、覆盖/干预、
时延/显存/训练参数与数据接口不变量的单位、公式、分母、缺失处理及解释边界。
主要方法指标仍为 M2/M1 route ADE；新增规划幅度不改变原微弱趋势阈值或工程验收。
统一 root 等权与配对区间，保留旧 C2 聚合/平局口径，禁止与新统计直接混算。
C6 计划从同一原始明细生成四组论文表和配对差图；不新增训练或 CARLA 次数。

复核 SimLingo 官方补充材料和 CARLA 官方评估说明，区分本地开发指标与官方榜单分数；
文档没有添加论文列表。量化实现/目标均 PLANNED，新增收益 NOT_MEASURED，仍 C3 NOT_STARTED。
检查 12 份文档的本地链接、代码围栏和阶段编号，历史 EVIDENCE 全文前缀保留；
diff 首次发现一处行尾空白并修复，最终重新运行 git diff --check。未运行模型测试、训练或闭环，
未改变冻结模型/数据/运行 Evidence，无 commit/push 或 goal 启动。
待后续 C3 实现 metrics-spec/逐 root 记录与实际运行验证；本轮无用户接管操作。

## 2026-09-08 — 提高单次实验正向机会的研究修订（DOCUMENTATION COMPLETE）

用户要求进一步搜集论文、提高一次固定实验获得微弱真实收益的机会。
本轮筛查近期 VLA 适配/能力保持、辅助负迁移、残差 World 与稳定训练方法，
核对部分原文/作者来源与本地 C2 ridge 报告/代码；没有运行新训练或闭环。
已有 validation ridge 的进度 ranking 52/52、53 pairs regret=0 为旧开发报告数值，
不是本轮新实验，也不能作为仍有上升空间的主目标。

C3 改为共同短 head 预热与较小 LoRA 学习率；C4 在固定候选 ridge 上学残差，
以驾驶梯度相容门控/限幅约束共享辅助更新；新增梯度成本在 C3 smoke 后统一冻结 T。
仍仅 M1/M2 两次训练、C5 三臂 18 attempts，资源总上限不变。
主要方法指标明确为 M2/M1 route ADE；M1/M0、C/B 与 World 误差分别报告，
不得挑次指标包装主任务成功。点估计、数值误差、CI 和代价分别解释。

更新 C3–C6 合同及相关活动文档，未堆论文列表，未设置/启动 goal。
新配方 PLANNED / NOT_RUN，收益 NOT_MEASURED；当前仍 C3，前置数据/GPU 尚待实测。
没有修改冻结数据、阈值、模型或运行 Evidence，没有安装/下载/commit/push。
验证：18 份活动文档的 59 个本地链接、围栏、无小数阶段与四份 goal 合同检查通过，
git diff --check 通过，已检查 diff/stat。EVIDENCE 原有全文保持为前缀，只追加本轮合同；
未改冻结运行 Evidence。本轮没有模型回归或训练，不将旧测试数当新回归。

## 2026-09-08 — 四阶段 goal 执行合同完善（DOCUMENTATION COMPLETE）

用户要求现在研究论文、任务文档只写可执行细节。本轮检索与核对驾驶 VLA、辅助未来监督、
开发闭环评估的一手方法/代码，并检查本地原生 hidden-state 与训练配置入口。
不新增论文综述，不扩大四阶段/两次训练/一次开发闭环范围。

C3/C4/C5/C6 各有可复制 goal、前置输入、允许改动、执行方法、产物与验收/停止条件；
统一 ROADMAP 索引、START_TASK 阶段选择与 stage-summary 交接。新增训练默认配置、
同预算/数据暴露控制、h/teacher 信息隔离、配对运行次序、失败占预算与 C6 离线复现。
配置是待实测的工程默认值，未宣称来自论文最优结果。当前仍 C3 NOT_STARTED，
M1/M2/新闭环 NOT_RUN；没有设置 goal、运行训练/CARLA、修改冻结 Evidence 或安装下载。

验证：18 份活动文档的 61 个本地链接、代码围栏、无小数阶段与四份 goal 合同检查通过，
git diff --check 通过，已检查 diff/stat。EVIDENCE 原有全文保持为前缀，仅追加交接摘要合同，
既有哈希引用保留。最初一次代码搜索使用不存在的 nominal_policy 路径，随后定位到
driving_vla/model/nominal_policy.py；未运行模型测试，未将搜索当运行验证。
没有用户接管事项；实际标签、权重兼容和训练/在线资源仍由后续对应 goal 实测。

## 2026-09-08 — 一周四阶段重构（DOCUMENTATION COMPLETE）

用户要求进一步简化全部项目文档，不使用小数子阶段。本轮在原分支直接重构 18 份活动
文档；当前仅 C3 数据与常规微调、C4 四维后果联合微调、C5 三臂 18-run 开发闭环、
C6 复现与论文交付。目标第 1–6 天完成，第 7 天缓冲。

必做训练由三种变为两种新模型 M1/M2；M0 只作离线基线。
撤下执行机制分解、ensemble、正式校准、四臂/大规模 formal 与额外公开数据下载。
World 用 C2 已有四个连续标签，不要求新增控制/未来序列。当前任务为完整 C3，
数据与 GPU smoke 是阶段内检查，不再拆任务。旧正式矩阵与保留集仍冻结。

状态：C3/C4/C5/C6 PLANNED / NOT_RUN；本轮仅文档，没有训练、CARLA、安装或下载。
C0–C2、历史负结果与运行 Evidence 未改。新的简化计划替代下文当时的复杂路线；
下文所有排期/模型数量/旧子步骤仅作历史，不能覆盖当前 START_TASK 与 ROADMAP。

验证：18 份活动文档的 59 个本地链接有效，代码围栏配对通过，未残留小数 C 阶段编号；
C3/C4/C5/C6 各有唯一阶段定义。已检查 diff/stat，git diff --check 通过。
Evidence 中 12 个历史证据/环境/归档章节原文不变，既有哈希引用全部保留；
没有修改代码或冻结运行 Evidence，不将历史模型测试充当本轮回归。
无需用户接管本次文档修改。下一实施按 START_TASK 完成 C3；监督与 GPU 可用性尚待实测。

## 2026-09-08 — C3–C6 完整路线同步（DOCUMENTATION COMPLETE）

用户明确要求直接修改 C3 起所有步骤与现有文档，不新开。本次在原分支完成文档统一：
原数据准备子步骤 数据/权重/真实梯度准备 → 原常规微调子步骤 常规 VLA SFT → 原联合训练子步骤 直接未来辅助与执行分解联合
对照 → C4 版本绑定/在线接入/有条件校准 → C5 四臂闭环 → C6 复现与论文材料。

已替换“结题后才微调 VLA”、旧 C1/C2 当前入口与 C3 未授权的活动状态；历史授权/任务
记录保留并标注日期范围。同步 README、START_TASK、ROADMAP、AGENTS、项目/数据/模型/
候选/文献/展示/资源/环境/Evidence、runtime 入口及三份已有研究备忘录。
没有新增文件、分支或任务，没有修改代码、冻结数据/阈值、模型或运行 Evidence。

当前状态：C3–C6 路线已确认 / PLANNED；原数据准备子步骤 NOT_STARTED；SFT/joint training NOT_RUN；
新 calibration/closed-loop/formal NOT_RUN。C2 工程门通过、原稀有门失败、MLP 未超过 ridge
均保持原状。M1/M2 为必做真实训练；M3 为必须尝试并如实结论的机制对照；不保证一次正收益。
本周预算和日期为规划而非实测。下一实施任务严格读取 START_TASK 的 原数据准备子步骤。

验证：18 份活动文档的 74 个本地 Markdown 链接均可解析；阶段/状态交叉检查与
`git diff --check` 通过；Evidence 中 12 个历史证据/环境/归档章节原文保持一致，
10 个既有哈希引用全部保留。修改仅涉及 15 个 tracked Markdown 和 3 个此前已存在的
untracked 研究备忘录；代码围栏配对通过，已检查 diff/stat。未触碰 archive 或运行 Evidence。
本轮未运行模型测试、GPU batch、CARLA、训练、下载或安装；历史 495 tests 不是本轮回归。
用户无需为本次文档同步接管环境；原数据准备子步骤 尚需实际核验标签序列、checkpoint 兼容和显存。

## 2026-09-08 — World–VLA 完整主线文献深化（RESEARCH / PROPOSED）

用户进一步明确：完整主线不止 VLA 微调。本轮检索并核对 World/VLA 联合学习、未来表征、
模型内策略优化和可靠性近邻，形成 [研究记录](docs/WORLD_VLA_FRONTIER_20260908.md)。
建议优先真实未来辅助监督与 VLA LoRA 联合学习，保留 World 在线 rank/defer 和既有执行链；
执行机制分解仅为待验证创新候选，不声称已实现或新颖性已证实。前一份 VLA 计划作为训练
工作包，不能代表整个项目。未运行训练/CARLA、未改模型/数据或冻结合同；文档检查通过。

## 2026-09-08 — 后续主线改为真实 VLA 微调（准备中）

用户明确希望做 VLA 微调，上一轮以审计为主的建议不再采用。新增
[VLA 微调方案](docs/VLA_FINETUNE_WEEK_PLAN.md)，既有 H 阶段证据全部保留。
本轮确认本地驾驶权重、InternVL2-1B 权重、原生 LoRA/动作头代码与主要包可定位。
受限进程 NVML 被阻断后，通过获准只读 probe 读到 RTX 4080 / 16376 MiB / 已用 1073 MiB。
此结果不代表训练资源实测。训练数据对齐、checkpoint adapter 映射及真实 batch 梯度检查
尚未完成；VLA fine-tune 仍为 NOT_RUN。未启动 CARLA、未安装包、未下载数据、未修改模型。
本轮为代码/资产检查与方案文档更新，`git diff --check` 通过；未运行训练或回归测试。

## 2026-09-07 — 一周收尾研究方案（PROPOSED，未实施）

用户补充一周内结束、保留既有成果、不依赖训练正收益的要求。本轮只完成进一步文献检索、
既有 baseline-report 与指标源码核对，并新增 [独立方案](docs/ONE_WEEK_RESEARCH_PLAN.md)。
报告中 candidate-only ridge 的 validation progress ranking 为 52/52，53 个有效配对的
selection regret 为 0；ranking 仅计真实差值绝对值大于 0.5 m 的配对。上述为既有 MEASURED
报告的核对，未独立重跑实验。方案建议执行感知的决策能力审计，不表示已获得新创新或收益。
未训练、未启动 CARLA、未读取 calibration/locked 的结果，未改变现有 H 阶段停止状态。
本轮仅文档修改；`git diff --check` 通过，没有运行模型测试或全量回归。

## 2026-09-07 — C2 调整完成：开发数据与 World 基线已交付

状态：

```text
H6-CORA C2 dev baseline = COMPLETED / DEV_BASELINE_GATE_PASSED / STOPPED
data release = DEV_DATA_READY (351 raw roots -> 340 usable roots, 11 isolated duplicates)
split roots = coverage_pilot 25 / train 158 / validation 53 / calibration 52 / locked_development 52
World baseline = MEASURED / NO_DEMONSTRATED_GAIN
original rare-event coverage gate = GATE_FAILED (保留，不由新合同改写)
closed_loop = NOT_MEASURED; calibration = NOT_RUN; VLA fine-tune = NOT_RUN; CARLA = 0 s
```

本轮把 C2 从“补齐稀有事件覆盖”改为“现有配对开发数据＋小型 World 基线”。新 release
只引用原始 base 文件和已有 v3 更正 sidecar，不复制运行数据，不扫描或重算旧 Hash。审计读取
全部 351 个原 root；11 个物理重复按规范化初态和 capture 条件隔离，不能靠 ID、seed、时间戳
增加样本。训练入口只接受 `quality_profile=c2_dev_baseline_v1`，training 只允许 train，evaluation
只允许 validation；coverage_pilot、calibration、locked_development 仅用于盘点。

World 使用 499-D context 与 `10x8` candidate 编码的共享 128/64 MLP，目标为四个连续短时
progress/comfort head，三个固定 seed（17/29/43），并与均值、context-only ridge、candidate-only
ridge、Expert/VLA 固定策略比较。三 seed 都跑完，候选交换等变和 source metadata 隔离检查通过；
candidate-only ridge 的 validation progress MAE 约 `0.302 m`，MLP 三 seed 约 `0.555/0.617/0.555 m`，
所以报告写入 `NO_DEMONSTRATED_GAIN`。这只是当前采样分布上的开发评估，不是安全概率、闭环收益
或通用策略价值声明。

证据路径：

```text
generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1/release-index.json
generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1/audit-report.json
generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1/world-baseline/baseline-report.json
generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1/final-delivery.json
docs/runtime-evidence/h6/h6-cora-c2-devbaseline-20260907-v1/test-report.json
```

最终报告为 `DEV_BASELINE_GATE_PASSED`，同时明确记录原质量门 `GATE_FAILED`、closed-loop
`NOT_MEASURED` 和 VLA 微调 `NOT_RUN`。全量回归为 495 tests、1 skipped、OK。下一接管点是
先决定是否基于该负收益结果改进 World/补做独立评估，再单独核验 VLA 微调输入合同；不因本轮
MLP 不占优追加模型搜索或 CARLA 采集。

## 2026-09-06 — C2 repair v3 收尾，诊断门仍失败

状态：

```text
H6-CORA C2 repair v3 = COMPLETED / DATA MEASURED / GATE_FAILED / STOPPED
added CARLA roots = 12 diagnostic roots / 34 executed branches; formal roots = 0
diagnostic gate = FAILED (repair-failure roots 1/2, offroad roots 5/1)
H6-CORA C3 = NOT_AUTHORIZED / NOT_STARTED
calibration execution = NOT_RUN; reserved_formal = NOT_COLLECTED
```

- 修复版 `h6-cora-c2-repair-20260906-v3` 新增 train-screening recipe 到 live collector 绑定、
  split-local target 分配、诊断/批次独立 manifest 和 run-lock、物理初态去重、Safety trace 到
  `repair_attempted/repair_success` head 的映射，以及可恢复累计预算账本。
- 真实 Town03 运行确认 CARLA 0.9.16、单实例、单 `ScenarioRuntime` tick owner；12 个诊断 root
  生成 34 条实际 branch，所有 cleanup 完成，CARLA 已正常关闭。
- 诊断结果为 1 个独立 root 实际尝试 QP/RATO 后仍失败、5 个独立 offroad root。正式 collector
  入口在读取更正标签后拒绝 batch-1，未消费正式 seed，也没有执行 calibration 或 reserved formal。
- 全量回归为 `492 tests run, 1 skipped, OK`；v3 final delivery 为 `GATE_FAILED`，原始 351 root
  与 1295 branch 保留，合并报告为 363 root、1329 branch（新增 12 root、34 branch）。
- v3 持久账本记录 CARLA 工作下限 118.4057 秒；旧 v2 的 445.23 秒不混入本版本总账本。
  旧数据、旧模型未做 Hash 重扫。

最终证据：

```text
generated/h6/cora/h6-cora-c2-repair-20260906-v3/final-delivery.json
docs/runtime-evidence/h6/h6-cora-c2-repair-20260906-v3/data-quality.json
generated/h6/cora/h6-cora-c2-repair-20260906-v3/budget-ledger.json
```

接管事项：本版本已用尽 12 个诊断 root，距离诊断门还差 1 个 repair-failure root。不能在本版本
内追加 root、改 recipe/seed/阈值、把同 root branch 膨胀成独立样本或把工程修复表述为门通过。

## 2026-09-05 — C2 repair v2 已实现，Town03 诊断完成，正式批次按诊断门停止

状态：

```text
H6-CORA C2 repair v2 = COMPLETED / DATA MEASURED / GATE_FAILED / STOPPED
added CARLA roots = 12 diagnostic roots / 36 branches; formal roots = 0
diagnostic gate = FAILED (repair-failure roots 0/2, offroad roots 3/1)
H6-CORA C3 = NOT_AUTHORIZED / NOT_STARTED
calibration execution = NOT_RUN; reserved_formal = NOT_COLLECTED
```

- 新修复版本 `h6-cora-c2-repair-20260905-v2` 保留 v1 数据引用，已生成 351 个原始 root、1295
  个原始 branch 的 `safedrive.cora.outcome_labels.v3` 更正标签；v1 sidecar、时间线和历史
  Evidence 未覆盖。旧文件、旧数据、旧模型未做 Hash 重扫，只有新 delta/report 使用内容身份。
- v3 统计以 `root_cluster_id` 去重，确认覆盖缺口为 10 项：train offroad 正类 7/8、四个 split
  的 repair-success 负类 0/(12,3,3,3)，executable 负类 9/1/0/2（门为 12/3/3/3），以及
  locked-development offroad 正类 1/2。报告明确撤回此前“仅缺样本”的结论；修复尝试可观测性、
  root 统计和 route/light derivation 也需要修正。
- train/Town03 真实 anchor/nominal 的离线筛选完成 648 个 1x/2x/3x Guard+Safety 组合，结果
  固定写入 `screening.json`，不作为 CARLA outcome；目标 recipe 不读取 validation、calibration
  或 locked-development 结果。
- 通过可恢复的 Windows-side `DefaultEngine.ini` 临时覆盖解决了 Town03 admission；原文件已先保存，
  不改旧数据/模型 Hash，也不把安装配置提交到仓库。单 CARLA、单 tick owner 下完成冻结的 12 个
  Town03 诊断 root、36 个 branch，覆盖 ClearNoon/CloudyNoon；RTX 4080 峰值约 8.35 GiB，
  诊断 elapsed 256.77 s，所有 branch cleanup 完成。
- 诊断结果为 offroad 有效 root 3 个，但实际尝试且 `repair_success=false` 的不同 root 为 0 个，
  未达到正式采集前要求的 2 个 repair-failure root。因此不执行两批正式 48-root 补采；诊断 root
  排除核心 coverage 门，最终报告保留 10 项原覆盖缺口并新增 diagnostic gate 缺口。
- 代码稳定后全量回归 `491 passed, 1 skipped`；repair final delivery 为 `GATE_FAILED`，added roots
  12（均为 diagnostic）、independent roots 363、branches 1331，CARLA aggregate wall 445.23 s。
  最终 Evidence 为
  `docs/runtime-evidence/h6/h6-cora-c2-repair-20260905-v2/final-delivery.json`，admission 证据
  为同目录 `admission.json`，质量报告为 `data-quality.json`。

接管事项：当前诊断 root 上没有满足条件的 repair-failure root，按冻结合同不能凭结果进入正式批次。
若要重新争取 coverage gate，必须创建新的修复版本/新诊断预算，不得在本版本中改 recipe、seed、
阈值或删除失败样本。CARLA 已关闭、临时配置已恢复；当前仍不得把工程修复完成表述为数据门通过。

## 2026-09-05 — H6-CORA C2 完成，development gate 失败并停止

状态：

```text
H6-CORA C2 = COMPLETED / DATA MEASURED / GATE_FAILED / STOPPED
H6-CORA C3 = NOT_AUTHORIZED / NOT_STARTED
reserved_formal = NOT_COLLECTED
```

交付与数据：

- 固定数据集 `h6-cora-c2-dev-20260830-v1` 已完成 351/351 terminal root attempts：pilot 27、
  development 324；351/351 nominal pairs 有效，nominal VLA forward 精确 351 次；
- 保存 1295 个真实 CARLA short-horizon branch outcomes，其中 development 1196；development
  split 为 train 162、validation 54、calibration 54、locked development 54，三个 map 各 108、
  九个 family 各 36、两种 weather 各 162；
- 新增 exact 29-head `safedrive.cora.outcome_labels.v2` public label、逐 head value/unit/valid mask/
  derivation version、公开训练 loader、observable feature 重建审计、完整 missingness 与资源审计；
- Guard、Safety、repair/MRM、executed/applied identity 和 terminal/cleanup 全链保留；1295/1295
  branch outcomes 有效，cross-candidate fallback、reset、identity、cleanup failure 均为 0；
- 所有七种 intervention 均达到 terminal `>=12` 和 Guard-eligible core `>=6` 的 development 门；
  reserved formal 108 roots 未采集，未训练 checkpoint、未 calibration、未启动 C3。

质量门与负结果：

- coverage pilot 全部通过；development 的 pair/split/map/family/weather/operator、manifest、
  run-lock、public label、feature reproducibility、inventory 和 resource audit 均通过；
- development gate 的真实失败为 locked development offroad positive `1 < 2`；
  `repair_success=false` 在四个 split 均为 0（未尝试 repair 的样本按正确 mask 记为无效，不能伪造
  repair 失败）；`executable=false` 为 train 9、validation 1、calibration 0、locked development 2，
  分别低于冻结的 12/3/3/3；
- 没有修改 matrix、seed、threshold、Guard 或 outcome，也没有复制 branch/intervention/tick 膨胀
  root 样本量，因此终态诚实冻结为 `DATA MEASURED / GATE_FAILED`。

资源、故障与恢复：

- aggregate collector wall 为 41184.8297303014 s（11.44 h），whole-GPU peak 9.9462890625 GiB，
  最低观测 free disk 124.15421295166016 GiB，dataset 292689739 bytes，均在冻结上限内；
- 三次外部采集故障均保存为 Evidence：Town01 branch 内 CARLA server exit、Town05 collector 与
  server 在 roots 间退出、Town05 collector 在 roots 间退出；immutable resume 后完成全部矩阵，
  失败耗时也计入总 wall；
- 完成后 preflight 显示 CARLA `NOT_RUNNING`、tick owner free、无 user action；既有未跟踪
  `test_registry.sqlite3` 保持 12288 bytes、mtime 不变，未删除、未暂存。

验收：

- C2 专项 26/26、H2 15/15、H3 challenge 9/9、World v3 24/24、VLA75 hardening 32/32；
- 全量 `unittest` 482 passed、1 skipped；`compileall` 与 `git diff --check` 通过；
- 最终 Evidence：`docs/runtime-evidence/h6/h6-cora-c2-dev-20260830-v1/final-delivery.json`，同时保存
  `collection-summary.json`、pilot/development `data-quality*.json`、`execution-history.json` 和
  `runtime-release.json`。本任务在 C2 停止点结束。

## 2026-08-30 — H6-CORA C2 已授权并开始实施

状态：

```text
H6-CORA C0 = COMPLETED
H6-CORA C1 = IMPLEMENTED / COMPLETED / STOPPED
H6-CORA C2 = CURRENT / AUTHORIZED / IN_PROGRESS
H6-CORA C3 = NOT_AUTHORIZED / NOT_STARTED
```

- 用户明确要求完整 C2 成品，包括实际 CARLA smoke、coverage pilot、development 数据和终态
  Evidence，不接受只完成脚手架或单元测试；
- 从 C1 完成提交 `7031786` 创建 `codex/h6-cora-c2`；开始时只有任务前既有未跟踪
  `test_registry.sqlite3`，无来源不明 tracked 修改；
- `START_TASK.md` 已冻结 dataset ID、351-root matrix、split/seed、intervention、reset、quality gate、
  资源上限、验收和停止点；
- `test_registry.sqlite3` 继续保持未跟踪、不得修改/删除/暂存，并从 C2 source identity 排除；
- 本条只记录任务进入与冻结合同，尚未产生 CORA 数据或新测量，algorithm Evidence 仍为
  `PLANNED / NOT_MEASURED / NOT_VERIFIED`。

## 2026-08-30 — H6-CORA C1 正确性加固完成并停止

状态：

```text
H6-CORA C0 document consolidation and QA = COMPLETED
H6-CORA C1 correctness hardening = IMPLEMENTED / COMPLETED / STOPPED
H6-CORA algorithm Evidence = PLANNED / NOT_MEASURED / NOT_VERIFIED
H6-CORA C2 counterfactual data = AWAITING SEPARATE AUTHORIZATION / NOT_STARTED
```

实施边界与 Git：

- 开始时只发现已知 C0 文档改动和既有未跟踪 `test_registry.sqlite3`，没有新的来源不明重叠修改；
- C0 文档经 staged diff/check 后提交为 `b8f3707 docs: consolidate H6-CORA active contracts`；
- C1 在 `codex/h6-cora-c1` 分支实施，默认保持未提交、未 push；
- `test_registry.sqlite3` 始终未跟踪、未删除、未暂存，并已从 C1 worktree/run-lock identity
  中显式排除；
- 没有采集 CORA 数据、训练项目/CUDA checkpoint、启动 CARLA、消费 formal seed、修改冻结
  dataset/split/seed/threshold 或改写 `docs/runtime-evidence/`；
- `scripts/h6_run_lock.py` 不在 C1 原允许路径表内，但用户本轮计划明确要求 C1 后 run-lock
  绑定 evaluator、validation lineage 和训练输入哈希，因此只做该必要调用链修改，没有扩展重构。

### C1 历史工作项 validation、evaluator、readiness

- `safedrive_foundry/data_pipeline/h6/dataset.py` 对 validation row 的 feature、trajectory、target、
  mask、split、seed、group 和样本身份生成稳定 lineage hash；
- `safedrive_foundry/data_pipeline/h6/model.py` 的 checkpoint selection 只接收 evaluator 实际产生
  的 loss、per-head count/loss、pair accuracy/regret、worst-group 和 candidate swap；缺字段、
  非有限值、零有效样本或无 lineage 均 fail closed，source/VLA usage 只保留诊断意义；
- 新增 `safedrive_foundry/data_pipeline/h6/evaluator.py`，定义
  `safedrive.world.vla75.evaluator.v1`，绑定 checkpoint/seed、validation/config/code/worktree/input
  hash、per-head/group/pair/probe、实测 latency、资源状态、Evidence 状态和自哈希；
- `scripts/train_world_v3.py` 产生 `safedrive.world.vla75.training_summary.v2`，绑定 train/validation
  lineage、三个 checkpoint 和三个 evaluator，并明确 artifact `VERIFIED` 不等于 CORA algorithm
  `VERIFIED`；
- `scripts/h6_readiness.py` 只接受完整 C1 v2 summary，识别但拒绝 v1，验证 summary/evaluator/
  checkpoint/input/self hash、三 seed 顺序、有效计数、probe、latency 和 GPU 实测状态，并输出
  `readiness_sha256`；
- `run_lock.py` 与 `scripts/h6_run_lock.py` 将新建 C1 lock 升级为
  `safedrive.h6.vla75.run_lock.v2`：calibration payload 必须带 evaluator、validation lineage
  和训练输入绑定，并在落盘前验证；显式 v1 只保留历史只读兼容。

失败行为：缺 evaluator/metric/lineage、`NOT_MEASURED`、零样本、零占位资源、未观察到
action/context sensitivity、swap 不变量失败或任一 hash/顺序不一致时 readiness 失败；CPU evaluator
的 incremental GPU peak 正确记录为 `NOT_MEASURED/value=null`，不能获得正式 readiness。

### C1 历史工作项 per-sample multi-task 与 Group-DRO

- `model.py` 新增逐样本 head report：objective、progress、completion、collision、red-light、
  offroad、comfort、repair、trust、pair preference 和 executable 各有独立 unreduced loss/mask；
- candidate head 先在样本内有效候选聚合，pair head 只在双 outcome 有效时启用，再按冻结权重
  计算“有效 head 加权和/有效权重和”；无有效 head 的 row 排除，整 batch 无有效监督时直接失败；
- `world_v3_loss`/`world_vla75_loss` 保持既有返回形状兼容，内部统一使用逐样本 reducer 并报告
  每个 head 有效计数；
- 持久 `GroupDROState` 预登记 map/family/weather/group，按 detached 真实多任务 group mean
  指数更新并应用 floor；当前 batch 未出现的 group 保留历史，空 group 记录
  `NOT_MEASURED/count=0/loss=null`；coverage/temporal penalty 不进入 Group-DRO 风险。

失败行为：mask 外 target 变化不影响 loss，repair/executable 独立 mask；空监督 batch 不可优化，
空 group 不会伪装成零风险或 gate pass。

### C1 历史工作项 唯一 temporal selector

- 新增 `safedrive_foundry/data_pipeline/h6/temporal.py` 的纯状态机，状态仅保存作用域内
  `expert`/`vla` source 和 source EMA，不保存 candidate ID；HOLD 总是返回本 tick fresh ID；
- scope 固定由 run/episode identity 与 route revision 组成，变化时重置 EMA/hold/history；
- 顺序固定为 scope/eligibility、EMA、held unavailable/emergency risk、emergency margin、minimum
  hold、hysteresis、普通 switch/choose；disposition/reason code 与 C1 合同一致；
- VLA75 live `H5WorldRouter` 和 offline `select_vla75_router_config` 调用同一核心并输出同一 trace；
  旧 H5 historical 路径保持原行为；single candidate、feature/deadline/low-confidence/forced defer
  也进入统一 defer core，按 held→Expert→VLA→MRM，非 MRM 结果仍交给 Safety。

失败行为：scope 不匹配会重置，held source 不可用或 unsafe 不会复活，Guard REJECT 不会重新
进入候选；无 eligible source 返回 `DEFER_SINGLE_CANDIDATE`/MRM 稳定原因。

### C1 历史工作项 single tick owner 与 cleanup

- `scripts/h5_collect.py` 删除 runtime 外直接 `world.tick()` 和强制推进/强杀恢复；正常运行仍只
  通过 `ScenarioRuntime.tick_controls()`；
- infrastructure retry 前只读检查 scene/settings/tick owner：clean scene 允许既有的一次 bounded
  retry；存在 vehicle/walker residue 或 ownership/settings 不可确认时，写
  `NEEDS_USER_ACTION/CLEANUP_RESIDUE` artifact 后停止；
- failure artifact 记录 residue IDs、tick owner、`tick_advanced=false`、cleanup/retry 状态和自哈希。

失败行为：residue 永不调用 fake/world tick；clean scene 也不推进 world，只允许一次既有 retry。

### C1 历史工作项 benchmark 与 Evidence 真实性

- `scripts/h5_ultimate_benchmark.py` 现在只产生
  `benchmark_scope=latency_only_smoke`、`model_state=random_untrained`、
  `quality_gate_eligible=false` 的 self-hashed artifact；
- 默认写入非冻结 `generated/h6/c1-smoke/ultimate-latency-smoke.json`；总状态只能是
  `SMOKE_COMPLETED` 或 `SMOKE_FAILED`，CPU/GPU 使用各自 latency threshold，局部 PASS/FAIL
  只描述 latency，不能解释为模型质量；
- C1 工程 Evidence 更新为 `IMPLEMENTED`；没有新增 GPU/CARLA 数字，CORA algorithm 仍为
  `PLANNED / NOT_MEASURED / NOT_VERIFIED`。

### 实际验证

第一次在未激活虚拟环境的 shell 直接运行 `python` 返回 `command not found`（exit 127）；随后按
任务文件激活 `/home/sdf/.venvs/sdf` 并重跑。最终实际结果：

```text
python -m unittest tests.hybrid.test_world_v3 -v
  24 tests / OK

python -m unittest tests.hybrid.test_vla75_hardening -v
  32 tests / OK

python -m unittest discover -s tests -t . -v
  456 tests / OK / skipped=1
  skipped 项是既有 SDF_CONTRACT_LIVE_FORWARD GPU+checkpoint 条件测试

python -m compileall -q safedrive_foundry scripts tests
  passed

git diff --check
  passed

git diff --stat / 完整 git diff 人工核对
  passed；17 个 tracked 文件有 C1 改动，另有 evaluator.py、temporal.py 两个新实现文件
  test_registry.sqlite3 仍是唯一与 C1 无关的未跟踪文件
```

补充审计期间曾两次把单个测试指定到不存在的类名，unittest loader 各返回 2 个加载错误；改用
文件内真实类名后对应 Group-DRO/selection/readiness 定向测试均通过。这是测试命令定位错误，
未触发产品代码修复或删除测试。最终专项和全量结果以上述完整模块/发现命令为准。

专项 evaluator 测试只在临时目录以微型 CPU 模型生成测试 checkpoint/evaluator，用于验证真实
计算与 hash/fail-closed 消费；它不是项目/CORA 训练结果，不升级 algorithm Evidence。GPU peak
在该测试中为 `NOT_MEASURED`，没有 CUDA/CARLA 实测数字。

剩余接管条件：C2 必须由用户单独授权、更新 `START_TASK.md`，并按
`docs/COUNTERFACTUAL_DATA.md` 冻结 development collection contract；C1 完成后不得自动采集、
训练、运行 pilot/formal 或进入 C2。

## 2026-08-29 — 活动文档收敛与第二轮全量 QA 完成

状态：

```text
H0-H4 = VERIFIED / STOPPED
H5 = VERIFIED / GATE_FAILED / STOPPED
H6 v1/v2 = IMPLEMENTED / MEASURED / NOT_VERIFIED
H6-CORA C0 document consolidation and QA = COMPLETED
H6-CORA C1 correctness hardening = CURRENT TASK / NOT_STARTED_THIS_TURN
```

本轮只做文档和可恢复归档：

- 根 README、START_TASK、ROADMAP、PROGRESS 和活动设计文档已统一为 CORA 结题主线；
- H3/H4 交付报告、H5 进入/矩阵、旧 H6 VLA75 handoff、旧 H2 paired contract 移入
  `archive/2026-08-27-cora-document-consolidation/historical-stage-docs/`；
- 收敛前的根/设计文档逐字快照保存在同一归档的 `original-active/`；
- 冻结的 `docs/runtime-evidence/` 和 `generated/` 数据没有改写或删除；
- H6-CORA 保持在 H6 内，不创建 H7；
- 90% VLA preference / 75% VLA usage 不再是 CORA 主要优化目标，历史 H6 仍按原门解释；
- 当前唯一下一任务是 C1 正确性加固，禁止自动采数据、训练或运行 formal。

用户要求重新检查后，第二轮逐份重读全部活动权威/入口文档、冻结 doctor 记录、关键代码
常量与调用点、Evidence 路径和相关工作的一手来源。修正的实质问题：

1. 分开 `program status` 与 `algorithm Evidence`：C0 完成、C1 当前不等于 CORA 算法已实现；
2. 把 outcome estimand 定义为 Guard eligible proposal 在冻结 single-candidate Safety/controller
   下的 CARLA intervention，采集 branch 禁止跨候选 fallback；
3. 定义 `CHOOSE/HOLD/SWITCH/DEFER`，其中 hold 使用 fresh candidate，defer 交回冻结非学习
   fallback，不是 Oracle、人工接管或无控制；
4. 修正 pair head：普通 MLP 接收差分不能保证反对称，必须用效用差或显式反对称化；
5. 区分 metadata-source-blind 与 trajectory-to-source 可预测性，避免伪称物理 planner 风格
   可以从 feature 中消失；
6. 修正 C2 选择偏差：Guard eligible 不等于预先 Safety 可执行；两个 branch 可按同一规则在
   不同 tick/reason terminal；missingness 和 root-anchor cluster 必须报告；
7. 明确 `ScenarioRuntime` collector 与 ROS `carla_sync_driver` 是互斥 tick-owner 模式；
8. 重写 C5 A–D 对照，使 selector 效果不被 generator workload 混淆，并要求 paired unsafe
   non-inferiority 口径而非只比原始事件数；
9. SHOWCASE 分开“现在可讲”和“C6 formal 后可讲”，移除当前枝干中的 LoRA 暗示；
10. RELATED_WORK 复核论文/官方来源，加入 CVPR 2026 Latent-CoT-Drive，并把 novelty 收敛为
    可证伪的组合贡献而非单概念首创。

C0 第一轮开始时：

```text
branch = main
tracked worktree changes = none
pre-existing untracked = test_registry.sqlite3
```

`test_registry.sqlite3` 没有被删除或纳入正式身份。

第二轮 QA 在上述未提交的 C0 文档工作树上继续；没有发现新的来源不明重叠修改，
`test_registry.sqlite3` 仍未触碰。

本轮文档验证：

```text
active Markdown local-link check = 16 files / 0 missing
active docs top-level set = 9 expected files / passed
cross-document semantic assertions = passed
frozen H1-H6 Evidence value assertions = passed
configured asset/path existence check = passed
archive original-to-HEAD byte comparison = passed
archive sha256 manifest = 19 files / passed
git diff --check = passed
```

本轮没有运行代码测试、CARLA 或训练；它们不属于 C0 文档收敛的验收范围。上述验证只证明
活动文档可导航、关键数字仍匹配冻结 artifact、归档可恢复且文本差异无格式错误，不能升级
任何算法 Evidence 状态。RELATED_WORK 的外部资料只用于定位，已按论文/官方页面复核，不是
本项目运行 Evidence。

## 冻结阶段结果

### H0 / H1

- H0 完成 H-only 路线收敛、旧路线归档和单机边界冻结；
- H1 在 Town03 完成真实 Classic + SimLingo 双候选 smoke；
- 两来源绑定同一 frame，VLA forward count 为 1；
- 两候选均 Guard PASS 且为 DISTINCT；
- selected/final/executed/applied identity 连贯；
- H1 Evidence 只证明候选与执行合同，不是驾驶性能结论。

### H2 paired outcomes

最终 gate-pass dataset：`h2-gatepass-20260813-routefix`。

```text
terminal = 120/120
valid/distinct = 108
decisive = 83
Expert wins = 51
VLA wins = 32
source-only baseline = 0.6144578313
dataset bytes = 1,480,172,014
whole-GPU peak = 8.3720703125 GiB
status = VERIFIED / GATE_PASSED / STOPPED
```

该数据证明两个独立来源存在真实选择空间。旧 gate-failed H2 dataset 同样保留，不能删除。

### H3 World development

最终运行：`h3-v2-20260815d-final`。

```text
OOF decisive = 91/91
reported best non-learning simple baseline = 84/91 = 0.9231
ECE = 0.00113
P99 = 13.89 ms
status = VERIFIED / GATE_PASSED / STOPPED
```

限制必须同时保留：

- learned candidate-only MLP 也达到 91/91；
- full-feature MLP 为 87/91；
- history sensitivity 部分由 hard scene gate 的结构保证；
- 因此 H3 证明小型开发集可区分，不证明上下文模型普遍优于所有 learned candidate-only
  baseline。

### H4 locked evaluation

最终运行：`h4-locked-20260816-final`。

```text
locked decisive = 64
World = 64/64
best simple = 57/64 = 0.890625
defer coverage = 63/64
status = VERIFIED / GATE_PASSED / STOPPED
```

限制必须同时保留：

- test set 小；
- simple baseline 已为 89.06%；
- temperature 到达 0.05 下边界，存在过度自信信号；
- microbenchmark 不能代表完整闭环尾延迟；
- 场景只覆盖三地图、有限 family/weather，不等于真实 OOD。

### H5 World on/off closed loop

最终运行：`h5-pilot-all2`，222/222 runs 完成。

```text
paired scenario roots = 74
World ON unsafe = 4
World OFF unsafe = 6
ON-only unsafe = 0
paired progress mean = +0.2549 m
bootstrap lower-95 = -0.0709 m
World ON switches = 14
World OFF switches = 0
scorer P99 = 47.2159 ms
deadline miss = 1
reset mismatch = 1
status = VERIFIED / GATE_FAILED / STOPPED
```

正式结论：World 没有在冻结 gate 上证明可复现闭环净收益。样本内没有 ON-only unsafe，但
未做出统计安全优越/非劣证明；进度置信区间跨 0，切换、资源和完整性门也未全部通过。不得
用 H3/H4 离线准确率改写该结论。

### H6 VLA-primary v1/v2

旧 seed 101 正式 pilot：`h6-vla90-formal-pilot-20260820-v1`。

```text
ticks = 600
World pair scored = 590/600
strict World VLA preference = 131/600 = 21.83%
VLA applied = 285/600 = 47.50%
Expert applied = 315/600
MRM applied = 0
VLA Guard = PASS 453 / REVIEW 147 / REJECT 0
World/Safety fallback to Expert = 18 ticks
RATO repair = 42
QP repair = 26
unsafe delta vs Classic = 0
paired progress lower-95 = +0.629 m
paired scenario roots = 12
scorer P99 = 9.12 ms
deadline miss = 0
switches = 31
ping-pong scenarios = 5
provenance failure = 10 missing World pair-score ticks（同一 Town01 aggressive-cut-in run）
status = MEASURED / GATE_FAILED / NOT_VERIFIED
```

正式低 VLA 占比的主因是 World 逐 tick 排名/校准，不是 Guard 或 Safety 大量拒绝 VLA。
Town03 free-flow 存在静态物体碰撞，Town03/Town05 存在红灯违规。seed 101 已消费，不能
调参后复用。

冻结 acceptance 的失败项为 actual VLA coverage、World VLA preference、switch rate、
ping-pong 和 provenance；paired progress 下界为正、deadline 为 0 不能覆盖这些失败。

H6 VLA75 v2 已实现 14-output 模型、A/B/C lineage、run-lock、acceptance 和 collector
hardening，但没有形成新 CUDA checkpoint、CORA 数据或正式 CARLA Evidence，因此仍是
`IMPLEMENTED / NOT_VERIFIED`。

## 2026-08-27 代码与数据审计确认的问题

以下为 2026-08-27 当时的问题记录；其中“当前”“必须修复”描述当时状态。
C1 已于 2026-08-30 完成对应工程加固，不能从本历史段落推断仍待修复。

### 1. 旧 H6 tickwise 反事实监督缺失

对本机 development dataset `h6-vla90-train-pilot-20260820-v2` 的 loader 审计：

| seed | tick rows | 双 outcome | 单 outcome | 双 executable | whole-policy pairs |
|---:|---:|---:|---:|---:|---:|
| 89 | 1200 | 0 | 1197 | 0 | 12 |
| 97 | 1200 | 0 | 1198 | 0 | 12 |

因此逐 tick pairwise mask 全为 0；主要比较监督来自每 seed 12 条 whole-policy row。旧
policy calibration 还使用第一条 decision 的特征监督整段 episode outcome。CORA 不允许
继续用该口径。

### 2. checkpoint validation 有常量占位

H6 `_vla75_validation_metrics` 存在 hardcoded masking/source-swap pass、`swap_error=0`、
`p99_ms=0`、`gpu_gib=0`，且参与 checkpoint selection。训练脚本后段虽计算部分真实指标，
readiness/selection 仍未完整绑定。C1 必须修复。

### 3. Group-DRO 语义错误

当前实现用标量 objective 与前 12 个异质输出做绝对差来构造 group loss，混合 progress、
variance、hazards、comfort、trust 等不同单位。C1 必须改为正确 per-head/per-sample loss。

### 4. temporal calibration/runtime 不一致

offline calibration 用稳定 source key；live EMA 使用 frame-scoped candidate id，跨 tick
可能重置。部分 hysteresis/emergency 条件会退化。C1 必须统一状态机并增加 trace parity。

### 5. single tick owner 违规

`scripts/h5_collect.py` 的清理路径存在直接 `world.tick()`。C1 必须移除或通过唯一 Runtime
推进。

### 6. demo/蒸馏宣传边界

- `carla_live_world_demo.py` 使用旧 H5 scorer，`--map` 未真正切图，并在 Safety 后强制
  最小 throttle；不能作为正式 Evidence；
- distilled scorer 的 risk loss 未加入总 loss，`val_data` 未实际用于验证；
- 某 latency 输出用 `<10ms` 判断却打印 `<4ms PASS`；
- `h5_ultimate_benchmark.py` 对随机未训练 World 做 latency 时仍输出质量式成功文案。

这些在 C1/C6 分别修复；当前不能作为简历正式结果。

## 历史离线验证（2026-08-27，非最新）

2026-08-27 全仓审计期间实际运行：

```text
python -m unittest discover -s tests -t . -v
436 tests: 435 passed, 1 skipped, 0 failed

python -m compileall -q safedrive_foundry scripts tests
passed

git diff --check
passed
```

跳过项是需要真实 GPU/checkpoint 的多次 SimLingo live forward；当前没有安装 coverage 模块，
因此 436 tests 不能解释为量化覆盖率。

该受限代理进程运行 `doctor/preflight` 时未访问到 GPU/CARLA。用户已明确确认本机 GPU 和
CARLA 可用；这次失败只代表该进程访问范围。任何后续真实任务仍必须在实际执行上下文中
重新记录 CUDA probe、preflight 和 CARLA Evidence，不能直接借用用户口头确认升级状态。

## 当前接管点

唯一下一任务为 [START_TASK](START_TASK.md) 的 C3：在同一阶段完成数据适配、真实 batch
与保存恢复检查、M0 离线基线、M1 常规 VLA 微调。验收后更新入口到 C4 并停止。
本轮文档修改不自动开始训练；后续只有 ROADMAP 的四个完整 C 阶段。
