# 项目当前进度

> 本文件只记录动态执行状态。任务编号、名称和依赖以 `ROADMAP.md` 为唯一来源；任务范围、验收和断点以对应任务文件为准。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前阶段 | **G3：CORE_MEASURED / FORMAL_CLOSE_PENDING** |
| 当前任务 | G3 pure VLA + constrained MPC 发布检查点；G3-05 Safety 正式复验仍待完成 |
| 当前状态 | **`MEASURED_WITH_LIMITS`**：真实 SimLingo path/speed → PathManager → MPC → CARLA 已有长测；不等于 G3 `VERIFIED` |
| 最近完成 | 官方 SimLingo contract、DX12 D3D 解法、CARLA 自动冷启动、Mercedes 几何、软连续性接入和单接缝 committed path |
| 推荐下一动作 | 发布后固定配置复验 Town03/Town15；随后恢复 G3-05 Safety，或经用户明确授权以 `PRE_G3_CLOSE` 准备 G4A |
| 最近更新 | 2026-07-19 |
| 发布目标 | `main` → `BaiYeGX/safedrive-foundry` |
| 过程日志 | `docs/project/G3_EXECUTION_JOURNAL.md` |
| 统一入口 | `docs/architecture/G3_VLA_MPC_RELEASE_GUIDE.md` |

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| G0 | `COMPLETED / FROZEN` | 冻结 |
| G1 | `COMPLETED_WITH_LIMITS` | 正式关闭 |
| G2 | `COMPLETED_WITH_LIMITS` | offline Safety Kernel |
| G3 | **`CORE_MEASURED / FORMAL_CLOSE_PENDING`** | pure VLA+MPC 已测；Demo 通过 ≠ G3-05 Safety VERIFIED |
| G4 | `PENDING` | **禁止**自动启动 |

## G3 Pure VLA + Constrained MPC 发布检查点（2026-07-19）

| 字段 | 当前事实 |
|---|---|
| 状态 | `MEASURED_WITH_LIMITS / POST_RELEASE_LIVE_REVALIDATION_REQUIRED` |
| 正式 demo 入口 | `tests/g3/run_g3_vla_mpc_minimal.py`（兼容入口，转到 `run_g3_vla_mpc_stable.py`） |
| 几何来源 | SimLingo 原生 20 点空间路径；官方 RoutePlanner 契约提供约 7.5–9.5m 相邻粗导航目标；地图中心线不作为 MPC 参考 |
| 跟踪器 | G3 专用 2s constrained MPC（舵角、舵速、舵加速度约束） |
| GPU 调度 | inference 与 CARLA tick 串行；runtime 内唯一 CUDA synchronize；50ms idle guard；debug draw 默认关 |
| 相机 | 官方契约 1024×512、x=-1.5m、z=2.0m、FOV=110°；BGRA→BGR→OpenCV JPEG→RGB→官方裁剪 |
| 速度语义 | VLA speed head 为主；`--v-ref` 仅为硬上限；无正速度 floor；VLA 制动立即生效 |
| 长测 | 600m coarse route 滚动续接；10s checkpoint + partial events；末20s运动比例门禁；碰撞/越线/离路/速度/VRAM evidence |
| D3D 隔离 | `camera-only` / `model-resident` / `forward-only` / `full` 四档；启动前持久化完整 run config |
| Large Map | Town11/12/13 hero + tile streaming；默认随机池排除；Town13 仍需 live 复验 |
| 路径执行 | 内禀无效/硬曲率/nav reverse 才硬拒；switch 类连续性问题软接入并刷新时间戳；短 prefix 后单次 old→latest blend，异常时 latest-only 回退 |
| 离线验证 | 见下方「停死修复 offline」；以本轮 `tests/g3` 实测为准 |
| 默认 RHI | **`DEFAULT_RHI=dx12`**（CLI / cold-start / carla_start.toml 一致） |
| VLA 速度输入 | 停止时传真实 0 m/s；`startup_speed_assist_mps` 仅未碰撞且未起步时可辅助；**碰撞后禁止伪 3 m/s** |
| 碰撞证据 | `CollisionEpisodeBook`：持续接触合并 episode；`collision_episodes.json` 含 first time/type/pose + 预碰撞 path/control |
| live A DX12 E0 | **PASS** 180s + **PASS** 300s；证据 `d3d_E0_A_dx12/`、`d3d_E0_A_dx12_300s/` |
| live E1 DX12 | 180s **无 D3D 崩溃**；驾驶 **DEMO_FAIL**（旧 raw collisions=272，待 episode 复测）；`d3d_E1_dx12_full/` |
| 发布说明 | `docs/architecture/G3_VLA_MPC_RELEASE_GUIDE.md` |
| 故障 runbook | `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md` |

### 2026-07-19 live：Mercedes Town03 120s@15 attach

| 字段 | 实测 |
|---|---|
| 命令 | Town03 · full · **--no-map-restart** · max-speed **15** · 120s · official 1024×512 · Mercedes 默认 |
| 证据 | `docs/architecture/evidence/g3-05/mercedes_town03_120s_15mps_attach/`（**JSON 不改写**） |
| 原始 result | **DEMO_FAIL**（`tail_moving_fraction_ge_0_50`=false；`verified=false`） |
| **离线归因更正** | **acceptance false negative / expected traffic-light stop**——**不是** VLA 异常停死 |
| 停车 | 约 **105.4s→120s**（~14.6s）；位置 ≈**(-102.265, 136.453)**；`first_long_stop` 前方为**红灯** |
| 对照 | 同图 Town03 300s attach 在 ≈**(-102.142, 136.382)** 红灯停 ~36.25s 后**正常起步** |
| 正确叙述 | VLA **正确识别红灯并减速停车**；测试窗口在红灯期间结束 |
| 车型 | requested/effective **`vehicle.mercedes.coupe_2020`** |
| 几何（当时） | 整包 `fallback_used`（steer 1.222 略超旧上限 1.20）——**已修为字段独立校验**（offline 本轮） |
| RHI | requested=dx12；effective=null（attach 无 cmdline）；**未**独立验证 DX12 |
| VLA | **153/160** accept |
| 里程 | distance **~373 m**；progress eff 1.0 |
| 碰撞/离路 | **0 / 0** |
| lane invasion | **3 episode / 4 raw**，**全部在开跑后 1.5–4.1s**；之后 ~116s 无侵线 → **初始对齐候选**，非全程不稳 |
| 跟踪 | cte_rms **~0.107 m**；sat=0；flips=13（门通过） |
| requery | **OFF** |
| 宣称 | **未宣称 G3 VERIFIED** |

### 2026-07-19 offline：stale-stop → relaunch 状态机（路口卡 0）

| 项 | 事实 |
|---|---|
| 归因 | 路口 switch 连拒 → age≥2.5s freshness **硬切 0** → 车停 → reanchor 后 VLA raw≈0.3&lt;0.35 → 永不起步；**非** MPC 求解/碰撞 |
| freshness | soft→hard 降到 **crawl 1.0 m/s**；仅 age≥**zero 5.0s** 才 0；`freshness_regime` 入 evidence |
| speed | stop/launch **迟滞**（0.35/0.50+确认帧）；reanchor/解除 execution stale 后 **recovery**（raw≥0.20×3 帧 + 命令层 floor 1.0 m/s，**不**伪 VLA 输入） |
| evidence | `vla_stop_source` / `execution_stale_latched` / `freshness_regime`；requery 仍默认 OFF |
| 测试 | `test_vla_speed_planner` + `test_vla_mpc_tracker`：**13/13 PASS** |
| 宣称 | **未跑 live 验证本修复；未宣称 G3 VERIFIED** |

### 2026-07-19 offline：尾段红灯验收 + 几何字段独立校验

| 项 | 事实 |
|---|---|
| 范围 | 交通灯 **oracle 仅 evidence/验收**；几何 per-field；**未** CARLA；**未**调 PathManager/MPC；requery 仍默认 OFF |
| 尾段门 | `tail_moving_or_expected_traffic_stop`；保留原始 `tail_motion.moving_fraction`；分类 moving / expected_red_light_stop / unexplained_stop / green_light_stuck |
| 几何 | max_steer 上限 **1.40 rad**；70°(1.222) 接受；单字段失败不丢弃其他字段 |
| 测试 | `test_vehicle_geometry` + `test_terminal_stop` + 相关 G3：**84 OK**；compileall 通过 |
| 宣称 | **未跑 live；未宣称 G3 VERIFIED** |

### 2026-07-19 offline：Mercedes + 车辆几何 + live 证据 schema

| 项 | 事实 |
|---|---|
| 范围 | 默认 Mercedes、yaw 不变量轴距、requery 默认关、lane episode/oracle、trace/RHI/prompt 证据；**未**启动 CARLA；**未改** PathManager/MPC 参数与验收阈值；无 Safety；无倒车 |
| 默认车型 | `vehicle.mercedes.coupe_2020`；缺失 → `RuntimeError`（禁 Audi/Tesla 静默回退） |
| 几何 | `vehicle_geometry.py`：PCA 主轴前后轴中心距；0/45/90/180° 轴距/轮距不变；2.5×1.5 在 yaw=90° 不把 1.5 当轴距 |
| requery | 默认 OFF；`--enable-stationary-requery` 才启用 |
| 新 evidence | `lane_invasion_episodes.json`；oracle RMS/P95/max；`committed_source_id` 计数；κ/steer multi-deadband flips；`\|Δsteer\|`/rate/accel；`control_seq.json` 20Hz |
| RHI/prompt | attach 无 cmdline `-dx12` 不宣称独立验证 DX12；official 实际 `command_text=None` |
| 测试 | `python -m unittest` geometry/lane/requery/runner/path_manager/mpc/speed/contract：**89 tests OK (1 skip)**；`compileall` 通过 |
| `git diff --check` | 本轮代码/runbook 无 trailing whitespace；`PROGRESS.md` 历史行既有 trailing whitespace（非本轮引入） |
| 宣称 | **未跑 live；未宣称 G3 VERIFIED** |

### 2026-07-19 停死修复 offline

| 项 | 事实 |
|---|---|
| 范围 | 默认 DX12；neural 停速；碰撞 episode；**未改** PathManager/MPC 参数；**无**倒车；**无** live |
| 修改 | `neural_policy.py`、`run_g3_vla_mpc_stable.py`、`test_vla_mpc_stable_runner.py`、runbook、PROGRESS |
| 测试 | `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`：**87/87 PASS** |
| 下一 live | 同配置 60s@6m/s **重复性** 复验（判定树：零碰撞+尾段运动） |
| 宣称 | **未宣称 G3 通过 / VERIFIED** |

### 2026-07-19 live first_collision_diagnostic_60s（max-speed=6）

| 字段 | 实测 |
|---|---|
| 命令 | Town04 · full · dx12 · **--max-speed 6** · 0.75s · guard 50ms · force-cold-start |
| 证据 | `docs/architecture/evidence/g3-05/first_collision_diagnostic_60s/` |
| result | **DEMO_PASS**（acceptance 全 true；`verified=false`） |
| RHI | requested/effective **dx12**；driver 610.74 |
| 碰撞 | episode=0 / raw=0 / lane=0 / offroad=0 |
| VLA | 80/80 accept；reject 无；path_age max **0.7s** |
| 运动 | distance **~271 m**；progress eff ~0.999；actual ~4.8 m/s；cap 6 |
| 跟踪 | cte_rms **~0.063 m**；sat=0；flips=0；mpc ~1197/1200 |
| 尾段 | moving_fraction **1.0**（末 20s 全动） |

判定：落入「零碰撞且尾段仍运动」→ 随后用户要求改为 **300s@15** 耐久重复性。

### 2026-07-19 live full_dx12_300s_15mps_repeat

| 字段 | 实测 |
|---|---|
| 命令 | Town04 · full · dx12 · **--max-speed 15** · 0.75s · guard 50ms · force-cold-start · 300s |
| 证据 | `docs/architecture/evidence/g3-05/full_dx12_300s_15mps_repeat/` |
| D3D | **满 300s 无 tick timeout / CrashContext**（DX12 稳定侧通过） |
| result | **DEMO_FAIL**（驾驶质量；`verified=false`） |
| 里程 | ~**410 m**；progress ~406 m；eff ~0.99 |
| VLA | accept **281/400**；reject：curvature 67+25 hard、lateral 22、heading 5 |
| path_age | p95 ~8.7 s；max ~**22.5 s**（有 reanchor，非旧 94s 死锁） |
| 尾段 | moving_fraction **0**（停死） |
| 碰撞 | **2 episode** / raw 1108；首碰 **~103.8 s** `static.guardrail`；(x,y)≈(369,-167) |
| 预碰撞 ep0 | speed≈0.94、throttle≈0.76、brake=0、steer≈0.36、path_age≈0.2、raw 20 点 / committed 94 点均有 |
| 二碰 ep1 | ~217.5 s `static.trafficsign`；raw 21 |
| 跟踪 | cte_rms ~0.84；sat ~0.13；offroad_frac ~0.19 |
| 相对 60s@6 | 低速短测 PASS；**15 m/s + 长时 + 弯道/护栏** 暴露接触与停死 |

归因笔记：episode 合并有效（1108 raw → 2 episode）。后续 VLA 事件复盘见下节 offline reanchor 时间线。

### 2026-07-19 offline：reanchor 加固 + driving_trace（无 live）

**300s 证据 n=70～150 时间线**（`full_dx12_300s_15mps_repeat`）

| 标记 | n / sim_s | 事实 |
|---|---|---|
| 最后正常段 | n=70–74 / ~55–58s | accepted，v_cmd~8–10 |
| 连续拒绝 | n=75–77 | curvature → lateral×2 |
| 错误单帧 reanchor | **n=78 / 60.8s** | `accepted_reanchor` jump5=1.51 hdg=15° |
| 停车 | n=80–118 | v_cmd≈0，stop_requested |
| 再 reanchor | n=125, 130, 134 | jump5 达 5.5，hdg 达 45° |
| 首次碰撞 | ~103.8s | `static.guardrail`（约 n=135 后） |

**代码**

1. `path_manager.py`：reanchor 需 **3 帧**一致；`reanchor_pending` → `accepted_reanchor`；`reanchor_nav_reverse`；纯 VLA 几何。
2. `run_g3_vla_mpc_stable.py`：`driving_trace.json`、offroad/reanchor 快照；debug-draw 绿/黄加粗。
3. `test_vla_path_manager.py`：单帧反向/pending/多帧确认/合法弧线。

**测试**：`python -m unittest discover -s tests/g3 -t . -v` → **91/91 PASS**。未 live；未改 MPC；无倒车；未宣称 VERIFIED。

### 2026-07-19 live reanchor_fix_120s_6mps_draw

| 字段 | 实测 |
|---|---|
| 流程 | preflight RETRYABLE → ensure 1× → preflight READY → run |
| 命令 | Town04 · full · dx12 · **max-speed 6** · 120s · **debug-draw** · force-cold-start |
| 证据 | `docs/architecture/evidence/g3-05/reanchor_fix_120s_6mps_draw/` |
| result | **DEMO_FAIL**（跑满 120s；`verified=false`） |
| collision episode | **0** |
| offroad_steps | **1**（frac~0.0004；VLA 时刻无 first_offroad 快照） |
| 尾段 | moving_fraction **0** |
| reanchor | **2× accepted_reanchor**，均经 `reanchor_pending`×2；**无单帧 reanchor** |
| 序列示例 | n=121–123：pending1→pending2→accepted_reanchor；n=156–158 同 |
| heading 切换 | n=123 预/后 committed 夹角 ~27°；n=158 ~44°（非 180° 反向） |
| 停死机制 | n=129+ `curvature_hard_limit` 连拒 → age 涨到 ~24s；pending 被硬曲率清零无法确认；v_cmd≈0 |
| accept | 79/160；lateral 45；curvature 20+11 hard；pending 5 |
| 门控结论 | **多帧 reanchor 生效**；失败主因弯道曲率硬拒 + 停车，不是一帧错误 reanchor |

### 2026-07-19 offline：n115–160 速度/曲率归因 + stationary_requery

**分析目录**：`docs/architecture/evidence/g3-05/reanchor_fix_120s_6mps_draw/offline_n115_160/`
（`speed_chain_n115_160.png`、`curvature_native_vs_dense_n115_160.png`、`diagnosis.json`）

| 问题 | 结论 |
|---|---|
| 停车谁压的 0？ | **VLA speed head 持续 raw≈0.15–0.35**（planner median）；**无** “raw>0.35 但 v_cmd=0” |
| n=121–123 反例 | 车已停但 raw 仍 4.3→2.2→2.2，说明**不单是** ego=0 反馈死锁 |
| n=129+ 停车 | stop_requested=True，v_cmd=0；与 curvature 连拒并存 |
| curvature_hard_limit | **11/11 帧 native max\|κ\|≤1，dense max\|κ\|>1** → **PCHIP 密化尖峰**（未改阈值） |

**代码（无 live）**

- `path_manager.compare_native_dense_curvature` 写入 trace
- `stationary_requery`：静止+连续 stop≥4 帧+无碰+导航/方向 OK → 用 1.5m/s **提示**重问 VLA；仅当新路径 accept 且正速度才采纳
- offroad：`road_surface`（lane_type、距 Driving 距离、heading 差、greenbelt 等），只作证据
- `first_long_stop_snapshot` + front RGB npy；完整 speed samples / resolved input speed

**测试**：`unittest discover -s tests/g3` → **96/96 PASS**。未放宽 hard_limit；未开 CARLA。

### 2026-07-19 offline：RTC + 横向模式 + coarse-target 归因（无 live）

文献对齐（SimLingo 提前变道 / Don't-Shake-the-Wheel / ACT / RTC）：三帧 reanchor 挡不住“连续稳定做错”；点序号 EMA 会错位平均；应 **世界坐标弧长对齐 + 近端承诺 + 远端最新 + 历史横向模式**。

**live 120s@6 requery trace 离线统计**（`offline_target_lateral_stats.json` + `offline_target_lateral_note.md`）：

| 指标 | 值 |
|---|---|
| \|path lat@10m\|>1.5 m | 53/160 (33%) |
| \|nav lat\|>2.0 m | 54/160 (34%) |
| 横向 mode flip（非零左右切换） | **25** |
| 本会触发 `early_lane_change`（nav≈直且 path 偏侧） | **仅 1 帧** |
| 主导拒绝 | `lateral_switch`×60 |
| 大 mismatch 集中 | route ≈ **280–300 m**（nav_lat 常 +6…+9 m） |

结论：本轮 sway 更吻合 **模式左右抖 + 弯道/长距 OOD**，而不是“nav 一直直、模型无故变道”。`early_lane_change` 仍保留（文献预变道）；`lateral_mode_flip` + RTC 近冻更针对 flip/jerk。

**代码**（未改 MPC / hard_limit）：

1. `path_manager.py`：`near_freeze_*` + `use_latest_only_for_far`；弧长 `relative_samples` 对齐；`lateral_mode_flip`；`early_lane_change`；二者禁止 reanchor 确认。
2. `test_vla_path_manager.py`：mode flip / early LC / near freeze 三测。

**测试**：`PYTHONPATH=safedrive_foundry python3 -m unittest tests.g3.test_vla_path_manager tests.g3.test_stationary_requery tests.g3.test_vla_mpc_stable_runner tests.g3.test_vla_mpc_tracker tests.g3.test_vla_speed_planner tests.g3.test_g3_native_path -v` → **54/54 PASS**（含 path_manager 15/15）。全量 `discover -s tests/g3` 中与本改无关的 pyarrow/torch 缺依赖失败未计为回归。未开 CARLA；未宣称 VERIFIED。

### 2026-07-19 live A/B：RTC+横向门 vs requery 基线（门控验证，非 G3 验收）

| 字段 | 实测 |
|---|---|
| 流程 | preflight RETRYABLE→ensure READY→preflight 曾掉线→ensure 再 READY→preflight READY→run force-cold-start |
| 命令 | Town04 · full · dx12 · **max-speed 6** · 120s · debug-draw · force-cold-start · seed=null（与基线 CLI 一致） |
| 新证据 | `docs/architecture/evidence/g3-05/rtc_lateral_gates_120s_6mps_draw/` |
| 基线 | `reanchor_requery_120s_6mps_draw/` |
| result | 两轮皆 **DEMO_FAIL** / `verified=false` |

| 指标 | 基线 | 本轮 | 判定 |
|---|---:|---:|---|
| distance_m | 306 | **212** | 恶化 >10% |
| steer_flip_rate | 0.54 Hz | **0.70 Hz** | 未改善 |
| path_age max | 5.2 s | **2.95 s** | 改善 |
| tail moving_frac | 0.56 | **1.0** | 改善 |
| cte_rms | 0.325 | 0.280 | 略好 |
| raw mode flips | 25 | **36** | raw 更抖 |
| committed mode flips | 16 | 15 | 黄线几乎无降 |
| `lateral_mode_flip` / `early_lane_change` | — | **0 / 0** | **不可归因** |
| 主导 reject | lateral_switch 60 | lateral_switch 67 | 同族 |
| freeze hdg p50/p95 | 7.3° / 23.8° | **14.5° / 31.6°** | 接缝航向变差 |

**结论（按约定判定）**：非成功门控验证——flip 上升且里程下降；亦非典型“门控锁死”（path age↓、尾段满运动、新门 0 触发）。下一刀优先 **冻结边界拼接连续性**；若修缝后黄线仍跟 raw/nav 共抖，再动 coarse target。未宣称 G3 VERIFIED；本轮 live **未再改代码/MPC**。

### 2026-07-19 offline：SimLingo 官方契约审计（无 CARLA）

**报告**：`docs/architecture/evidence/g3-05/simlingo_contract_audit_offline.md`
**单测**：`tests/g3/test_simlingo_contract_offline.py`

| P0 项 | 结论 |
|---|---|
| head 选择 `pred_route` vs `pred_speed_wps` | **接线正确**：路径用 `route`，速度用 `speed_wps`（与 `driving.py` / `control_pid` 一致） |
| 相机位姿 / FOV | **一致** (−1.5,0,2.0) / 110° |
| 传感器分辨率 | **不一致**：官方 1024×512 vs live 默认 **640×320** |
| JPEG 路径 | **不一致**：官方 BGR→cv2 JPEG→BGR2RGB vs 本仓库 RGB→PIL JPEG |
| 底边裁剪公式 | **一致**（~30% / 4.8/16） |
| target 两点 | **不一致**：官方 RoutePlanner **7.5 m** 后相邻点 vs 本仓库固定 **15 m / 30 m**（S 弯易放大 `nav_lat`） |
| map↔ego 符号 | **与官方 `Rᵀ(p−t)` 一致**；+x 前、+y=CARLA 右 |

**归因修正**：`lat@10m` 符号 / `lateral_mode_flip` **不能**当 S 弯变道判据；代理指标（低 CTE、flip 计数、accept 率）不能证明几何正确。
**P1**：暂停增强 `lateral_mode_flip`；RTC 接缝次优先于再加门。
**测试**：`python3 -m unittest tests.g3.test_simlingo_contract_offline -v` → **7/7 PASS**。未 live；未宣称 VERIFIED。

### 2026-07-19 实现 + live：Official SimLingo Contract 模式

**实现要点**

| 项 | 默认 |
|---|---|
| 相机 | **1024×512** / FOV110 / (−1.5,0,2.0) |
| 预处理 | BGRA→BGR→**cv2 JPEG**→RGB→crop 4.8/16→InternVL 448 |
| Target | densify~1m + **RoutePlanner(7.5,50)** 剩余 **[1],[2]** + Rᵀ |
| Heads | `pred_route`→PathManager/MPC；`pred_speed_wps`→速度 |
| `lateral_mode_flip` | **默认 OFF**（`--enable-lateral-mode-flip` 可开） |
| 旧配置 | `--no-official-contract` → 640/15·30/RGB-PIL/flip on |

**模块**：`simlingo_contract.py`；runner evidence 含 `official_contract` + `simlingo_contract` + gates。
**画线**：绿 raw route / 黄 committed / 青 speed_wps。

**离线**：相关 G3 套件 **65 pass + 1 skip**（20× forward 需 `SDF_CONTRACT_LIVE_FORWARD=1`）。
双 head 图：`docs/architecture/evidence/g3-05/official_contract_offline/dual_head.png`（固定合成图 20×；route_var 非 0 → 模型非严格确定性）。
报告：`official_contract_offline/OFFICIAL_CONTRACT_REPORT.md`。

**live** `official_contract_120s_6mps_draw`（Town04 · dx12 · max-speed 6 · 1024×512 · debug-draw）：

| 指标 | 值 |
|---|---:|
| result | **DEMO_FAIL**（仅 `steer_flip_rate_lt_0_5hz`） |
| vla_accepts | **160/160** |
| \|tp\| 典型 | **~7.5–9.5 m**（非 15/30） |
| distance_m | 240.8 |
| path_age max | **0.70 s** |
| collisions / offroad / lane_invasions | **0 / 0 / 0** |
| tail moving | **1.0** |
| cte_rms | 0.189 |
| steer_sign_flips | 104（~0.87 Hz） |
| peak VRAM | 2222 MB（无 OOM） |

未宣称 G3 VERIFIED；未改 MPC 参数。

### 2026-07-19 live 测量摘要（Town04，0.75s / 640×320，debug draw 关）

| 档 | 模式 | RHI | 结果 | 关键事实 |
|---|---|---|---|---|
| C | camera-only | dx11 | **PASS** 180s | RGB 241 帧；车静止；无模型 |
| D | model-resident | dx11 | **PASS** 180s | 模型驻留 ~1871MB CUDA；无 forward |
| E | full（旧） | dx11 | **FAIL** ~177s | D3D Assert / DXGI_INVALID_CALL |
| E0 | forward-only | dx11 | **FAIL** ~59s | 第~79 forward 后 tick timeout；无 CrashContext |
| **A E0** | forward-only | **dx12** | **PASS 180s** | 240/240 forward；车静止；infer P50/P95 ~147/175；VRAM 2222 |
| **A E0** | forward-only | **dx12** | **PASS 300s** | 400/400 forward；无崩溃 |
| **E1** | full | **dx12** | **无崩溃 / DEMO_FAIL** | 180s 跑完；distance~319m；accept 193/240；collisions=272；path_age max~27.7s（非旧 94s）；尾段 moving_fraction=0；cte_rms~0.62 |

归因：
- **DX12 解决 E0 的 CUDA forward↔D3D11 hang**（对比同配置 dx11 ~59s 崩）。
- E1 证明 **车辆运动+forward 在 DX12 上可撑满 180s**；失败项是驾驶质量（弯道/碰撞/尾段停车），不是 Server 假活。
- 路径重锚 live 生效（见 `accepted_reanchor`）；未再出现 131 连拒 / path_age~94s。
- **未宣称 G3 VERIFIED**；20/60/300 正式验收未跑；B offscreen 未需要。

### D3D 子事故关闭口径

- 状态：`MEASURED_RESOLVED_ON_DX12`，仅指当前 RTX 4080 / 驱动 610.74 /
  CARLA 0.9.16 的已测组合；DX11 是已知失败基线。
- 修复证据：DX12 E0 180s/240 forward、300s/400 forward，以及 DX12 E1 full
  180s/240 forward，合计 880 次真实 forward，均未复现 D3D hang/crash。
- 运行约束：后续 G3 live 使用默认 DX12，并核对 evidence 中
  `requested_rhi`、`effective_rhi` 和 Server 命令行；attach 到无 RHI 参数的既有进程时
  不得伪称“由命令行独立验证 DX12”。
- 边界：这不关闭碰撞、弯后卡死、CTE、转向饱和或尾段静止，G3 仍非
  `VERIFIED`。详细记录与复现/回归步骤见
  `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md`。
- 防回归：上述 runbook 已把 **DX12 cold-start + 显式 `--rhi dx12` +
  `run_config.json` 三项核验**写成标准解法，并记录不得再用 TdrDelay、控制调参或
  no-rendering 代替 RHI 修复。

## G3-05 历史 Visual Demo（已被当前稳定化入口取代）

| 字段 | 结果 |
|---|---|
| 状态 | **`DEMO_PASS`**（`not_g3_stage_verified=true`） |
| 证据 | `docs/architecture/evidence/g3-05/visual_demo/latest_demo_summary.json` |
| 入口 | `tests/g3/run_g3_vla_v0_visual_demo.py` |
| backend | `neural_simlingo` |
| mode | `VLA_SAFETY`（无 Classic 当前帧） |
| seed | 11，Town10HD_Opt，100 steps |
| n_track_approved | **100 / 100** |
| decision | **全部 ACCEPT** |
| distance_m | **~9.7 m**（speed 升至 ~9.5 m/s） |
| camera_frames | 100 |
| neural P50/P95 | **~124 / ~158 ms**（keep-on-GPU；&lt;250ms） |
| peak VRAM | ~2222 MB |
| force_throttle | false |
| restamp generated_time | false |
| sources_seen | `["vla_fast"]` |

### Demo 实现要点

1. **keep-on-GPU**：去掉每步 `model.cpu()` 回弹；warmup 后 P50 ~110–160ms
2. **同步推理**：推理时不 `world.tick`，避免 age 相对 G2 `max_candidate_age_s=0.25` 必 stale
3. **不 restamp** `generated_time_s`；仅对齐 set 级 `frame_id`/`simulation_time_s` 身份
4. **轨迹 reshape**：保留 SimLingo 路径形状，钳制 accel/κ/步长以过 Safety trackability
5. **Safety 绑定**：仅执行批准轨迹；EMERGENCY 硬刹

### 明确未做

- 未宣称 G3 阶段 `VERIFIED` / `COMPLETED`
- 未跑 dual-seed 正式关阶段 `assert_g3_close`
- 未启动 G4；未改 G2 配置

## 历史入口（仅保留证据，不再作为推荐动作）

- 原历史重跑命令：
  `SDF_VLA_KEEP_ON_GPU=1 python tests/g3/run_g3_vla_v0_visual_demo.py --seed 11 --max-steps 150`
- 当前不得用该结果替代 stable 管线的 20s/60s live 验收；正式 G3 关阶段仍需
  neural_live_v2 + assert_g3_close。

### 2026-07-19 offline：Town15 转弯陈旧死锁与黄线近端假曲率修复

**范围**：纯 VLA + PathManager + constrained MPC；未启动 CARLA，未改 Safety，
未启用倒车或 stationary requery，未宣称 G3 VERIFIED。

**已确认根因与修复**：

- `heading_switch` / `lateral_switch` / soft `curvature_limit` 原先作为硬拒绝，
  在 Town15 合法右转 n147–151 使 committed stamp 不刷新并触发 freshness 停车；
  现改为 `accepted_soft_<reason>`，硬拒绝只保留内禀无效、硬曲率和 nav reverse。
- committed 拼接从“latest→old→latest”双接缝改为短 prefix 后单次
  old→latest smoothstep。
- 不再每帧把 committed 首点强插到 ego；该做法在真实 trace 前 0.2–0.4m
  制造最高约 3.0 1/m 的假曲率。MPC 使用自身投影，blend 后检查近端曲率，
  必要时 latest-only 回退。
- execution recovery 只由已观测到的 MPC freshness 强制停车授权；普通路径接受/
  reanchor 不得覆盖 VLA 语义停车。可选 requery 保留官方 contract/image layout；
  停车图像证据按真实 RGB/BGR layout 命名。

**真实 trace 离线反事实**：重放
`mercedes_town15_300s_30mps_chasefix/driving_trace.json` n135–160：26/26
路径接纳、max path_age=0、committed max|κ|≈0.720 1/m、latest-only fallback=1。
该结果只证明旧 trace 在新路径逻辑下不再由 switch 门积累陈旧，不代替 live 闭环。

**验证**：

- `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`
  → **Ran 154，153 passed，1 skipped**（GPU 20× forward 显式 opt-in）。
