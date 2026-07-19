# G3 Pure VLA + Constrained MPC 稳定运行手册

**状态**：D3D 子事故 `MEASURED_RESOLVED_ON_DX12`；驾驶质量 `LIVE_PARTIAL / DEMO_FAIL`
**目标**：先验收直线和大半径弧线；路口、掉头随后单独验收。

## D3D 问题标准解法与防回归记录（2026-07-19）

这个记录是后续 G3 live 的运行基线，不得因为重新调 MPC、PathManager 或速度参数
而回退到 DX11。

### 标准解法（以后直接照做，不再重新排查）

1. CARLA 使用 **DX12 onscreen**：`carla_start.toml` 中同时保持 `rhi = "dx12"`
   且启动参数只有一个 `-dx12`，不能同时残留 `-dx11`。
2. 每条 G3 VLA live 命令显式添加 `--rhi dx12`；如果当前 CARLA 是用 DX11
   启动的，使用 `--force-cold-start` 进行一次完整冷启动，不能热切换 RHI。
3. 启动后检查 evidence 的 `run_config.json`：必须同时满足
   `requested_rhi=dx12`、`effective_rhi=dx12`，并确认 `server_command_line` 中是
   `-dx12`。
4. 保持已验证负载基线：Low、server 640×360、camera 640×320、VLA period
   0.75s、idle guard 50ms、debug draw 默认关闭。改变这些参数时只能视为新的
   未验证组合，不能覆盖本记录的已测结论。

**不要重复尝试的伪解法**：增加 Windows `TdrDelay`、继续降低 VLA 频率、把 MPC
调钝、修改路径门限、启用 no-rendering、仅靠减少显存或反复重启。它们都没有解决
已经隔离出的 DX11 + CUDA forward 冲突；其中 no-rendering 还会让视觉传感器失去
有效数据。

最短可靠启动示例：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 180 \
  --rhi dx12 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --force-cold-start
```

| 项 | 结论 |
|---|---|
| 已复现故障 | DX11 onscreen 下，真实 SimLingo CUDA forward 可单独触发 CARLA Server hang/crash |
| 已排除 | 不是 VRAM OOM；不是车辆运动、MPC、路径拒绝、debug draw 或仅模型驻留的必要结果 |
| 已验证修复 | CARLA 改用 DX12 onscreen；E0 forward-only 180s/240 次和 300s/400 次均通过 |
| full 交叉验证 | DX12 E1 full 完整运行 180s/240 次 forward，无 tick timeout、D3D crash 或新 CrashContext |
| 事故状态 | **在当前单机配置上按 DX12 运行时关闭；DX11 保留为已知失败基线** |
| 不在此关闭范围 | E1 的碰撞、弯后卡死、CTE 和尾段静止属于驾驶质量问题，G3 仍非 `VERIFIED` |

已测环境：RTX 4080 16GB、驱动 610.74、CARLA 0.9.16、Town04、Low、
server 640×360、camera 640×320、VLA period 0.75s、post-forward idle guard 50ms、
debug draw 关闭。三轮 DX12 共完成 880 次真实 forward，没有复现 D3D11 的约 60 秒
假活模式。

根因口径应写为：**当前证据支持“CARLA UE4 D3D11 渲染与同卡 CUDA forward 的
互操作/调度冲突”，DX12 RHI 是已实测有效的规避方案**。没有引擎或驱动级根因
转储，因此不要把它扩大表述成所有硬件上的通用 CARLA 缺陷。

原始证据：

- DX11 E0 失败：`docs/architecture/evidence/g3-05/d3d_E0_forward_only/`
- DX12 E0 180s：`docs/architecture/evidence/g3-05/d3d_E0_A_dx12/`
- DX12 E0 300s：`docs/architecture/evidence/g3-05/d3d_E0_A_dx12_300s/`
- DX12 E1 full 180s：`docs/architecture/evidence/g3-05/d3d_E1_dx12_full/`

后续所有 G3 VLA live 应使用 DX12。**runner CLI 默认 `DEFAULT_RHI=dx12`**（与
`carla_start.toml` 一致）；仍建议在关键复测中显式写 `--rhi dx12`，并在
`run_config.json` 核对 `requested_rhi=dx12`、`effective_rhi=dx12` 和命令行只有一个
`-dx12`。若再次出现 `world.tick` timeout/D3D 错误，先跑 E0 forward-only 180s 回归
并保存 CrashContext，不要先改 MPC、路径门限、VLA period、idle guard 或
Windows `TdrDelay`。

## 首次碰撞后停死：速度输入与碰撞证据（2026-07-19 代码轨）

**背景**：E1 DX12 full 180s 无 D3D 崩溃，但驾驶 DEMO_FAIL（多次接触/尾段静止）。
嫌疑之一是 `neural_policy` 在 `ego_v≤0.05` 时向 SimLingo **伪造 3 m/s**，使碰撞后
车已停住而模型仍看到“在动”，加剧无法重新起步。

**本轮代码修改（未调 PathManager/MPC 参数，无倒车）**：

1. **默认 RHI = DX12**：`DEFAULT_RHI` + argparse 默认 + cold-start 默认。
2. **`NeuralV0Policy.resolve_vla_input_speed_mps`**：
   - 默认传真实非负 `ego_v`（含 0）；
   - 仅当 `meta.startup_speed_assist_mps` 且 `has_collided=False` 且静止时允许首次起步辅助；
   - 碰撞后强制真实 0，禁止辅助。
3. **碰撞 episode**：持续接触合并为一次 episode（默认 gap 0.5s）；记录首次碰撞
   sim 时间、物体 type、车位姿；保存碰撞前 raw VLA 路径、committed 路径、速度、
   throttle/brake/steer；证据 `collision_episodes.json`。验收用 **episode 数** 而非
   原始 contact 计数（旧 E1 的 272 多半是持续接触刷事件）。

**离线验证**：`python -m unittest discover -s tests/g3 -t . -v`（须全绿）。

### Live first-collision 诊断（必须限速 6 m/s）

**不要**用默认 15 m/s 做首次碰撞因果诊断。用 `--max-speed 6`。

2026-07-19 实测一轮：`first_collision_diagnostic_60s` → **DEMO_PASS**，
0 collision episode，尾段 moving_fraction=1.0，cte_rms~0.06 m。
判定落入「零碰撞且尾段仍运动」→ 下一步是 **同配置 60s 重复性**，不要先调参。

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"
python scripts/sdf.py sim preflight --json
# RETRYABLE_FAILURE 时：python scripts/sdf.py sim ensure --rhi dx12 --startup-timeout 180 --json

# 第一轮（已 PASS）：first_collision_diagnostic_60s
# 第二轮重复性：
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 60 \
  --inference-mode full \
  --rhi dx12 \
  --max-speed 6 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/first_collision_diagnostic_60s_repeat
```

判读：`latest_summary` / `collision_episodes` / `vla_events` / `run_config`（dx12 + speed_cap=6）。
**不宣称 G3 VERIFIED**；勿倒车、勿 15 m/s、勿 300s，直到重复性明确。

## PathManager reanchor 加固（2026-07-19 offline）

300s@15 证据：n=75 起连续拒绝后 **n=78 单帧 `accepted_reanchor`**（大 jump），随后
长时间 v_cmd=0；n=130/134 再次错误 reanchor，约 104s 撞 guardrail。根因是
“陈旧路径+停车”时 **一次** VLA 候选即可丢弃 committed。

**现行为**

- 正常 accept 仍走 switch/曲率门。
- 仅当拒绝且 `path_age≥2.5s` 且 `speed≤0.5`：进入 reanchor 窗口。
- 候选须通过内禀几何 + **不得与 coarse nav target 明显反向**（`reanchor_nav_reverse`）。
- **连续 3 帧** heading/lateral 一致：`reanchor_pending` → `accepted_reanchor`。
- 几何始终来自 VLA，无 HD-map 中心线。

**证据**

- `driving_trace.json`：每次 VLA 的 ego/progress/targets/raw/committed/reason/age/control/offroad
- `reanchor_snapshots.json` / `first_offroad_snapshot.json`
- `--debug-draw`：绿=raw VLA，黄=committed→MPC（VLA 频率更新，reject 也画 raw）

### 下一 live（120s · 6 m/s · DX12 · debug-draw）

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"
python scripts/sdf.py sim preflight --json
# RETRYABLE 时：python scripts/sdf.py sim ensure --rhi dx12 --startup-timeout 180 --json

python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 120 \
  --inference-mode full \
  --rhi dx12 \
  --max-speed 6 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --force-cold-start \
  --debug-draw \
  --evidence-dir docs/architecture/evidence/g3-05/reanchor_fix_120s_6mps_draw
```

看：`reanchor_pending` 是否出现、是否仍有单帧 `accepted_reanchor`、黄/绿线是否可见、
`driving_trace` 中 path_age 与 offroad。MPC 参数未改；无倒车。

## n115–160 离线归因与 stationary_requery（2026-07-19）

120s@6 live 在弯后停死。离线复盘（**未改 hard_limit**）：

1. **速度**：n=129 后 `vla_speed_raw_mps`（samples 中位数）持续 ≤0.35 → planner
   `stop_requested` → `v_cmd=0`。**没有** “raw 为正却被 planner 压 0”。n=121–123 车已停
   仍有 raw≈2–4，故**不全是** ego=0 反馈死锁，但长期 stop 仍值得用 requery 打断。
2. **曲率**：`curvature_hard_limit` 共 11 帧，**全部** native max|κ|≤1 而 dense/PCHIP
   max|κ|>1 → **密化尖峰**（图：`offline_n115_160/curvature_native_vs_dense_*.png`）。
3. **trace 补强**：`resolved_vla_input_speed_mps`、完整 `vla_speed_samples_mps`、
   native/dense 曲率、`road_surface`（lane_type / 距 Driving / heading 差 / greenbelt 等）、
   `first_long_stop_snapshot` + RGB。
4. **stationary_requery**（历史实验路径）：原默认会在静止+连续 stop 时用 1.5 m/s
   提示重问。**自 2026-07-19 车辆几何轮起默认关闭**；仅
   `--enable-stationary-requery` 显式开启。默认关闭时不得额外 VLA forward、不得
   伪造 1.5 m/s 输入。

## 默认 ego 车辆、几何与观测证据（2026-07-19）

| 项 | 规则 |
|---|---|
| 默认 blueprint | **`vehicle.mercedes.coupe_2020`** |
| 缺失 blueprint | **`RuntimeError`**，禁止静默回退 Audi/Tesla |
| CLI 覆盖 | `--vehicle-blueprint <id>`；`run_config` 记 `requested_` / `effective_vehicle_blueprint` |
| 轴距估计 | **PCA/主轴投影 + 前后轴中心距**；兼容局部/世界坐标与 cm/m；**不**用 `max(x)-min(x)` |
| 字段独立 | wheelbase / track / max_steer **分别**校验；单字段失败不得整包丢弃；steer 上限约 **1.40 rad**（允许 Mercedes ~70°） |
| 不可信物理 | 仅该字段显式 **`mercedes_coupe_2020_validated_fallback`**；overall 可为 `mixed_physics_and_fallback` |
| stationary_requery | **默认 OFF**；实验开关 `--enable-stationary-requery` |
| lane oracle | 每控制周期记录 road/lane/junction/width/有符号横向误差；**`oracle_only=true`**；只进 evidence |
| 交通灯 oracle | `is_at_traffic_light` / state / id；**只**用于 terminal 分类与验收；**禁止**进 VLA/PathManager/MPC/速度 |
| 尾段验收 | 门：`tail_moving_or_expected_traffic_stop`；保留原始 `moving_fraction`；红灯合法停 → **不得**仅因 tail 静止判 FAIL |
| lane invasion | `lane_invasion_episodes.json`；短窗合并；Mercedes Town03 120s 的 3 episode 均在 **1.5–4.1s**（初始对齐候选） |
| RHI 证据 | `requested_rhi` vs `effective_rhi` 分开；attach 无 cmdline `-dx12` 时 **不得**宣称独立验证为 DX12 |
| prompt 证据 | official runtime 实际 `command_text=None` / `prompt_mode=target_point`，勿写 config 陈旧默认句 |

### Mercedes Town03 120s attach 归因更正（2026-07-19）

原始 evidence 目录：`docs/architecture/evidence/g3-05/mercedes_town03_120s_15mps_attach/`

原始 `result=DEMO_FAIL`（`tail_moving_fraction≈0.25`）**保留不改写**。

| 错误旧叙述 | 更正 |
|---|---|
| “VLA 中后段把速度压到 0 / 异常停死” | **错误** |
| 正确叙述 | VLA **正确识别红灯并减速停车**；约 105.4–120s 红灯停；窗口在红灯期间结束 |
| 诊断标签 | **acceptance false negative / expected traffic-light stop** |
| 对照 | 同图 300s 在几乎相同坐标红灯停 ~36s 后正常起步 |

代码修复后：末窗静止 + 红/黄灯 → `terminal_stop_classification=expected_red_light_stop`，尾段门可通过。

几何实现：`safedrive_foundry/driving_vla/runtime/vehicle_geometry.py`

车道证据：`safedrive_foundry/driving_vla/runtime/lane_evidence.py`

## 当前入口

```text
tests/g3/run_g3_vla_mpc_minimal.py
```

该文件是兼容入口，实际调用：

```text
tests/g3/run_g3_vla_mpc_stable.py
```

## 固定数据流

```text
CARLA RGB（与SimLingo训练标定一致）
→ SimLingo原生20点空间路径 + 独立speed waypoints
→ VLASpeedPlanner（VLA速度校准、15m/s硬上限、加速限速、制动立即生效）
→ VLAPathManager（弧长对齐、近端承诺、尾部融合、质量门）
→ ConstrainedVLAMPC（OSQP、2秒时域、短路径/陈旧路径/曲率限速）
→ carla.VehicleControl
```

地图路线只用于按官方 RoutePlanner 契约生成约 7.5m 后的相邻 coarse target，
不作为控制参考，也不提供车道中心线。

## Windows 重启前后

重启前无需继续启动 CARLA。2026-07-18 的一次规定恢复尝试返回
`BLOCKED_EXTERNAL / NEEDS_USER_ACTION`，错误为 WSL vsock 权限失败；不要继续循环
`ensure`。用户完成 NVIDIA 驱动更新并重启 Windows 后：

1. 确认 `carla_start.toml` 为 Town04、DX12、Low、640×360；runner 默认已是 DX12，
   关键复测仍建议显式 `--rhi dx12`。
2. 从 WSL 执行一次 `sdf sim preflight`。
3. 仅当结果为 `RETRYABLE_FAILURE` 时执行一次 `sdf sim ensure`，再执行一次 preflight。
4. `READY` 后先运行20秒验收。

```bash
source /home/sdf/.venvs/sdf/bin/activate
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 20 \
  --v-ref 6 \
  --vla-period-s 0.75 \
  --rhi dx12 \
  --force-cold-start
```

20秒结果为 `DEMO_PASS` 后，再运行60秒：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 60 \
  --v-ref 10 \
  --vla-period-s 0.75 \
  --rhi dx12
```

两轮都通过后，再跑用户要求的 5 分钟快速耐久测试。Town10HD 已有较好的短测
基础，且不像 Town13 那样依赖超大瓦片加载：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town10HD \
  --duration-s 300 \
  --max-speed 15 \
  --speed-gain 1.5 \
  --vla-period-s 0.75 \
  --rhi dx12 \
  --route-segment-m 600
```

`--max-speed 15`（兼容旧名 `--v-ref 15`）是绝对上限，不再把速度托到
12.75m/s。实际目标仍由 VLA speed
head 决定；校准后最高 15m/s。原生路径通常只有约 19m，因此 MPC 还会按可见
制动距离限速，直线实际可能先受约 10m/s 的路径视距上限约束。这是物理可停车
约束，不是强制慢速。

## PathManager：硬几何门 + 软连续性接入（2026-07-19 offline）

Town15 旧 trace 的直接因果链是：合法转弯连续触发 `heading_switch` /
`lateral_switch` → committed stamp 不刷新 → freshness 限速停车 → 停车输入使 speed
head 进一步降低。增加 stale 秒数或 reanchor 帧数只会延后死锁。

当前 `VLAPathManager` 行为：

1. **硬拒绝只用于内禀无效**：退化、过短、非前向、自交、点态硬曲率超过
   `hard_max_abs_curvature`，以及与 coarse nav 明显反向。硬拒绝时保留旧路径并允许
   新 VLA 速度降低目标。
2. **连续性指标是软诊断**：`curvature_limit`、`lateral_switch`、
   `heading_switch`、`lateral_mode_flip`、`early_lane_change` 不再把合法路口/S 弯
   变成陈旧路径；结果记为 `accepted_soft_<reason>`，并刷新 committed stamp。
3. **RTC 风格单接缝**：同一 ego 弧长站位上短暂保留已执行 prefix，之后用一段
   smoothstep 过渡到最新 VLA；禁止旧版“latest→old→latest”双切换。远端默认只取
   最新 VLA，不做不同模式的点序号平均。
4. **不强钉 ego 首点**：committed 从旧路径的 ego 投影点起，MPC 自己投影 ego。
   每帧硬插 ego 点会在前 0.2–0.4m 制造 1–3 1/m 的假曲率。
5. blend 后重新检查 MPC 实际会读到的近端曲率；拼接异常时回退到 latest-only，
   几何仍完全来自 VLA，不使用 HD-map 中心线。

Town15 `mercedes_town15_300s_30mps_chasefix` 的 n135–160 离线重放：旧逻辑在
n147–151 连拒并把 path age 推到 3.75s；新逻辑 26/26 接纳、最大 age=0，黄线
max|κ|≤0.721/m（仅 1 帧 latest-only 回退）。这是离线反事实，不等于 live 通过。

速度恢复只允许由“**MPC freshness 已实际强制停车** + 后续新鲜硬合法路径”授权；
普通路径接纳/reanchor 不能越过 VLA 的语义停车，避免把红灯停车误当死锁唤醒。
本改未修改 MPC 权重、未增加倒车、未启用 stationary requery。

## 可视化

- 默认不画线，避免长测中额外的 UE debug-render 负载。
- 需要看线时显式加 `--debug-draw`：绿色是当次 raw SimLingo 路径，
  黄色是实际交给 MPC 的 committed/ensembled 路径。两条线不是两个控制器。
- 路径只在新 VLA 结果到达时绘制，避免每个控制 tick 重画造成 UE GPU 压力。

## 结果判定

`latest_summary.json` 同时检查：

- VLA 路径接受率；
- MPC实际使用比例；
- CTE RMS；
- 方向饱和比例；
- 方向符号翻转频率；
- 路线进度/实际里程效率；
- 碰撞与离路比例；
- 是否达到与测试时长匹配的最小里程；
- 最后 20 秒至少 50% 时间仍在运动，防止“前半程开过、后半程停死”也算通过。

启动前先写 `run_config.json`；长测每 10 秒写 `progress_latest.json` 和
`vla_events_partial.json`。即使 CARLA 随后崩溃，也能恢复实际参数、接受/拒绝理由、
推理 P50/P95、VRAM、路径 age、速度、里程和道路事件。运行异常另写
`failure_latest.json`，其中 `last_operation=world_tick` 的 timeout 表示 CARLA Server 已不响应，
不是 MPC 异常。异常时还会只匹配本次启动后新生成的 Windows Unreal
CrashContext，复制为 `carla_crash_context_latest.xml` 并把 CrashType、D3D、驱动和 OOM
字段写入 failure JSON；旧崩溃记录不会被错配。

单纯“没有转圈”不再判定通过。

## D3D 诊断顺序

两轮 Town04 A/B 已证实：A（0.50s、768×384）在约 60s sim time 后是明确
`GPUCrash / 0x887A0020 INTERNAL_ERROR`；B（0.75s、640×320）运行到约 120s sim time，
但后半段被错误的“目标在车后”判断停车，因此不能证明 D3D 已解决。该导航判断
已改为按 route arc-length 判断，不再要求弯道中第二目标的 ego-x 必须更大。

2026-07-19 首轮 C/D 均180s PASS；旧 E full 在约177s sim time 后是
`CrashType=Assert / D3D11Query.cpp:356 / DXGI_ERROR_INVALID_CALL`，`bIsOOM=0`。
随后 E0（forward-only + 车辆静止）在约 59–61s / 第 ~79 次 forward 后挂死
（`world.tick` 30s timeout，无新 CrashContext），基本确认真实 CUDA forward 与
D3D11 冲突；车辆运动不是必要触发条件。显存峰值约 2.2GB，不是 VRAM OOM。

**禁止**用 no-rendering 模式做 VLA 视觉实验：CARLA 官方说明该模式下相机/GPU
传感器返回空数据（[rendering options](https://carla.readthedocs.io/en/0.9.8/adv_rendering_options/)）。
Offscreen（`-RenderOffScreen`）与 no-rendering 不同，可作单变量对照。

### RHI / Offscreen 可复现实验（事故复现与回归流程）

统一启动入口：`carla_start.toml` + `sdf sim ensure`（`ConnectionResolver`）。
运行器通过 `--rhi` / `--render-offscreen` 写入 toml 并在参数不一致时 cold-start，
**不**直接 `Popen CarlaUE4.exe`。

证据 `run_config.json` 必含：`requested_rhi`、`effective_rhi`、`render_offscreen`、
`server_command_line`、`driver_version`、`actual_map`、`server_resolution`。
失败时 `failure_latest.json` 必含：`n_forward_to_failure`、last forward/tick 时间、
camera 最后一帧、process/RPC 状态；无 CrashContext 的 tick timeout 分类为
`CARLA_SERVER_HANG_NO_CRASH_CONTEXT`。

#### 决策树（严格单变量）

1. **先跑 A（DX12 E0）180s**。
2. A PASS 后，**同配置跑 A 300s**。
3. A 不能启动或 E0 失败，才跑 **B（DX11 Offscreen E0）180s**。
4. B PASS 后，同配置跑 **B 300s**。
5. 只有某个 RHI 档位的 E0 **180s 与 300s 都通过**，才允许跑对应档位的 **E1 full 180s**。
6. 若 A/B 都失败：停止并保留证据；**不要**通过改 period、guard、TdrDelay 或控制参数掩盖失败。

#### A：DX12 E0（优先）

- Town04 · forward-only · 车辆静止 · 180s · vla-period 0.75s · gpu-idle-guard 50ms
- server 640×360 · camera 640×320 · Low · debug draw off · **dx12** · offscreen false

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

# 可选：显式 ensure 与请求 RHI 对齐（运行器 force-cold-start 也会写入 toml 并重启）
python scripts/sdf.py sim ensure --rhi dx12 --no-render-offscreen --startup-timeout 180 --json

python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 --duration-s 180 \
  --inference-mode forward-only \
  --rhi dx12 \
  --gpu-idle-guard-ms 50 \
  --vla-period-s 0.75 \
  --server-res-x 640 --server-res-y 360 \
  --cam-w 640 --cam-h 320 \
  --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_E0_A_dx12
```

A 180s PASS 后 300s：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 --duration-s 300 \
  --inference-mode forward-only \
  --rhi dx12 \
  --gpu-idle-guard-ms 50 \
  --vla-period-s 0.75 \
  --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_E0_A_dx12_300s
```

#### B：DX11 Offscreen E0（仅 A 失败后）

与 A 相同，仅 **dx11 + RenderOffScreen=true**。

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

python scripts/sdf.py sim ensure --rhi dx11 --render-offscreen --startup-timeout 180 --json

python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 --duration-s 180 \
  --inference-mode forward-only \
  --rhi dx11 \
  --render-offscreen \
  --gpu-idle-guard-ms 50 \
  --vla-period-s 0.75 \
  --server-res-x 640 --server-res-y 360 \
  --cam-w 640 --cam-h 320 \
  --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_E0_B_dx11_offscreen
```

B 180s PASS 后 300s：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 --duration-s 300 \
  --inference-mode forward-only \
  --rhi dx11 \
  --render-offscreen \
  --gpu-idle-guard-ms 50 \
  --vla-period-s 0.75 \
  --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_E0_B_dx11_offscreen_300s
```

#### E1 full（仅对应 RHI 的 E0 180+300 通过后）

```bash
# 示例：DX12 路径通过后
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 --duration-s 180 \
  --inference-mode full \
  --rhi dx12 \
  --gpu-idle-guard-ms 50 \
  --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_E1_dx12_full
```

### 历史负载隔离档（C / D / 旧 E0）

```bash
# C: CARLA + RGB，不加载模型，车保持制动
python tests/g3/run_g3_vla_mpc_minimal.py --map Town04 --duration-s 180 \
  --inference-mode camera-only --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_C_camera

# D: CARLA + RGB + 模型驻留显存，不做forward
python tests/g3/run_g3_vla_mpc_minimal.py --map Town04 --duration-s 180 \
  --inference-mode model-resident --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_D_resident

# 旧 E0（默认 dx11 onscreen）：已被上面 A/B 取代为当前主线
python tests/g3/run_g3_vla_mpc_minimal.py --map Town04 --duration-s 180 \
  --inference-mode forward-only --gpu-idle-guard-ms 50 --force-cold-start \
  --evidence-dir docs/architecture/evidence/g3-05/d3d_E0_forward_only
```

默认相机与 VLA 0.75s 周期对齐；runtime 内只做一次 CUDA synchronize，默认 50ms
host idle guard 再 tick。C 崩 → CARLA/驱动/相机；C 过 D 崩 → 模型驻留；
C/D 过 E0 崩 → CUDA forward↔D3D；E0 过 E1 崩 → 运动/场景与 forward 的组合。

Town11/12/13 是 Large Map：入口会把车辆设为 `role_name=hero`，先把 spectator
移到出生瓦片并预热 streaming。默认随机地图池不包含 Town11/12/13。

`TdrDelay` 不作为首要修复；只有事件日志确认 TDR 且 RHI/offscreen 隔离仍复现时，
才作为用户手动诊断变量。
