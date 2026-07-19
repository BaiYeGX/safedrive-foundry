# G3 Pure VLA + MPC 发布说明与后续路线

> 发布检查点：2026-07-19
> 适用环境：CARLA 0.9.16（Windows）+ WSL2 Ubuntu + RTX 4080 16GB
> 当前结论：**纯 VLA 轨迹与速度已经真实接入 CARLA，并由约束 MPC 执行；状态为
> `MEASURED_WITH_LIMITS`。正式 G3-05 的 VLA+Safety 阶段验收仍需新 live 证据。**

这份文档是当前 G3 的统一入口。历史故障细节保留在
[`G3_VLA_MPC_STABLE_RUNBOOK.md`](./G3_VLA_MPC_STABLE_RUNBOOK.md)，旧咨询稿仅作历史归因，
不要再从旧 demo 脚本开始。

## 1. 现在已经完成了什么

### 1.1 用户要求的核心闭环已经成立

当前主链为：

```text
CARLA 前视相机 + ego 状态 + coarse navigation target
  → SimLingo / InternVL2-1B 真实 CUDA forward
  → pred_route：20 点纯 VLA 空间路径
  → pred_speed_wps：独立 VLA 速度语义
  → VLAPathManager：硬几何门、软连续性接入、单接缝 committed path
  → ConstrainedVLAMPC：2 秒预测、舵角/舵速/舵加速度约束
  → carla.VehicleControl
```

地图只生成类似 GPS 的粗目标点，不把 CARLA 车道中心线当作 MPC 参考。绿色 raw path、
黄色 committed path 和最终方向/速度都来自 VLA 输出及其执行层条件化。

### 1.2 官方 SimLingo 输入/输出契约已对齐

默认 `--official-contract` 包含：

- 相机 `1024×512`、FOV 110°、安装位姿 `(-1.5, 0, 2.0)`；
- CARLA BGRA → BGR → OpenCV JPEG → RGB → 官方底部裁剪 → InternVL 448；
- 路线约 1m densify 后按官方 `RoutePlanner(7.5, 50)` 取相邻 `[1]`、`[2]` 目标；
- `pred_route` 只进入 PathManager/MPC；`pred_speed_wps` 只进入速度规划；
- 官方模式下不使用历史上会误伤 S 弯的 lateral-mode 硬门。

旧 `15m/30m` 目标、RGB/PIL 路径和错误的第二目标仍可用
`--no-official-contract` 做历史 A/B，但不是默认配置。

### 1.3 路径执行层已经从“拒绝到停死”改为连续更新

当前 PathManager 的原则是：

- **硬拒绝**：退化、过短、非前向、自交、硬曲率异常、与 coarse nav 明显反向；
- **软接入**：`curvature_limit`、`lateral_switch`、`heading_switch`、横向启发式；
- 软接入结果记录为 `accepted_soft_<reason>`，并刷新 committed timestamp；
- 短 committed prefix 后只做一次 old→latest smoothstep，不再构造
  latest→old→latest 双接缝；
- 不再把 ego 位置硬插为黄线首点；MPC 自己投影 ego，避免前 0.2–0.4m 假曲率；
- blend 产生异常曲率时回退到 latest-only，仍不使用 HD-map 中心线。

Town15 历史失败 trace 的 n135–160 离线反事实重放中，新逻辑 26/26 接纳、最大
path age 为 0、committed max|κ| 约 0.720 1/m。该结果证明旧 trace 不再因 switch 门
积累陈旧，但仍需新的 live 闭环验证。

### 1.4 VLA 速度、停车与恢复语义已分离

- `--max-speed` 是绝对上限，不是最低巡航速度；
- 默认 `speed-gain=1.0`，不再隐式乘 1.5；
- VLA 减速立即生效，提速受加速度斜率限制；
- 停车时向 VLA 输入真实 `0 m/s`；仅第一次起步允许显式 startup assist；
- 红灯停车与 execution freshness 停车分开记录；
- 只有“已经观测到 MPC freshness 强制停车 + 后续新鲜硬合法路径”才能授权恢复；
- 普通 path accept/reanchor 不能覆盖 VLA 的红灯式语义停车；
- `stationary_requery` 默认关闭，避免伪造常态 VLA 输入。

### 1.5 MPC 与车辆几何已针对真实车型校准

- 默认车辆为 `vehicle.mercedes.coupe_2020`；缺失时失败，不静默换车；
- wheelbase、track width、max steer 分字段读取和校验；
- Mercedes 约 70° 物理最大转角不会让整包几何回退；
- lateral MPC 按实际/短期可执行速度线性化，不按远高于实速的 speed cap 线性化；
- 路径视距、曲率、freshness 分别产生可审计的纵向速度上限；
- freshness 阈值必须满足 `soft <= hard <= zero`，配置错误直接失败。

### 1.6 CARLA 与 CUDA 同卡的 D3D 问题已有稳定解法

已隔离确认：DX11 下车辆静止但执行真实 CUDA forward，约第 79 次 forward 后仍可令
CARLA Server 假活；显存峰值约 2.2GB，不是 OOM。换成 DX12 后：

- forward-only 180s / 240 次 forward：完成；
- forward-only 300s / 400 次 forward：完成；
- full VLA+MPC 180s / 240 次 forward：完成且无 D3D crash。

因此当前本机基线固定为 DX12。这个结论只适用于已测的 CARLA 0.9.16、RTX 4080 和
当前驱动组合，不代表所有机器都已验证。

### 1.7 CARLA 自动冷启动已对齐手动双击行为

`sdf sim ensure` 默认使用 `default_engine` 模式：

- `Start-Process` 设置 CARLA 安装目录为 WorkingDirectory；
- 不传容易触发 shader pipeline 差异的额外 ArgumentList；
- 地图与 RHI 原子写入并复核 `DefaultEngine.ini`；
- `--map` 会传入共享启动器，不再由 toml 中旧 Town 覆盖意图；
- 进程第 0 秒退出会尽早分类为 shader fatal/process exited，而不是盲等 180s；
- 已有 READY 进程不会二次启动；地图不匹配明确返回 `MAP_MISMATCH`。

一次 Town03/DX12 自动冷启动 live 已在约 17.8s 后 READY。它证明当前默认路径可用，
不是对所有地图 shader cache 的无限保证。

### 1.8 Evidence 与验收数据已经可审计

runner 会生成：

- `run_config.json`：requested/effective map、RHI、车型、相机和官方契约；
- `progress_latest.json`：长测 checkpoint；
- `driving_trace.json` / `control_seq.json`：VLA 更新和 20Hz 控制；
- raw/committed 曲率、path age、全部速度限制来源、MPC solver 状态；
- collision episode、lane-invasion episode、off-road 与 lane oracle；
- traffic-light terminal stop 分类；
- CARLA hang/crash 分类和匹配本次运行的 CrashContext。

lane/traffic-light oracle 只进入 evidence 和验收，不进入 VLA、PathManager、MPC 或速度决策。

## 2. 已有 live 结果如何理解

| 运行 | 结果 | 能证明什么 |
|---|---|---|
| Town04 60s @ 6m/s | `DEMO_PASS`，约271m，0碰撞/0离路，CTE≈0.063m | 短直线/缓弯闭环可稳定运行 |
| Town03 300s，cap 20 | `DEMO_PASS`，约763m，0碰撞/0离路，steer flip≈0.17Hz | 官方契约 + 黄线/MPC 修复可长时间存活 |
| Town03 120s，Mercedes | 原始 `DEMO_FAIL`，但终点是红灯 | 这是旧验收 false negative，不是 VLA 异常停死 |
| Town15 旧 300s | 驾驶停死，随后 CARLA hang | 暴露 switch→stale 死锁；最新修复仅做了离线反事实，待 live |
| DX12 E0 300s | `DIAGNOSTIC_PASS` | 400 次真实 forward 未复现 DX11 D3D hang |

`DEMO_PASS` 是 stable runner 的驾驶门，不等于 G3 正式 `VERIFIED`。历史 JSON 保持原样，
不为了新解释重写结果。

进入版本库的最小证据集合与各文件适用边界见
[`RELEASE_EVIDENCE_INDEX.md`](./evidence/g3-05/RELEASE_EVIDENCE_INDEX.md)。大型逐帧 trace
保留在本机 evidence 目录，不和源码发布混在一起。

## 3. 从零开始运行

### 3.1 前提

1. Windows 已安装 CARLA 0.9.16，当前默认路径为 `E:\CARLA_0.9.16`；
2. WSL 项目路径为 `/mnt/e/autonomous driving`；
3. SimLingo 与 InternVL 权重按
   [`LOCAL_ASSETS.md`](../project/LOCAL_ASSETS.md) 登记放置；
4. 使用 `/home/sdf/.venvs/sdf`，不要用系统 Python 跑 VLA；
5. NVIDIA 驱动已更新并完成 Windows 重启。

### 3.2 激活环境与预检

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"
python scripts/sdf.py sim preflight --json
```

结果处理：

- `READY`：直接运行；
- `RETRYABLE_FAILURE`：只执行一次 ensure，再 preflight 一次；
- `NEEDS_USER_ACTION`：停止，按报告做 Windows/GUI 操作；
- 不循环 ensure，不在运行中反复切图。

CARLA 未运行时，推荐自动启动：

```bash
python scripts/sdf.py sim ensure \
  --map Town03 \
  --rhi dx12 \
  --startup-timeout 180 \
  --json
```

### 3.3 第一轮：60 秒可视 smoke

CARLA 已经在 Town03 READY 时：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town03 \
  --duration-s 60 \
  --inference-mode full \
  --no-map-restart \
  --max-speed 6 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --debug-draw \
  --evidence-dir docs/architecture/evidence/g3-05/release_smoke_town03_60s
```

线条含义：

- 绿色：当帧 raw `pred_route`；
- 黄色：真正交给 MPC 的 committed path；
- 青色：`pred_speed_wps` 几何，仅用于观察 speed head，**不控制方向**。

### 3.4 第二轮：180 秒驾驶质量验证

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town03 \
  --duration-s 180 \
  --inference-mode full \
  --no-map-restart \
  --max-speed 15 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --evidence-dir docs/architecture/evidence/g3-05/release_town03_180s_15mps
```

耐久测试默认关闭 debug draw，以减少 Unreal debug-render 负担。

### 3.5 第三轮：300 秒耐久

前两轮通过后再运行：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town03 \
  --duration-s 300 \
  --inference-mode full \
  --no-map-restart \
  --max-speed 20 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --evidence-dir docs/architecture/evidence/g3-05/release_town03_300s_20mps
```

`max-speed` 只是 cap。实际速度仍受 VLA speed head、路径视距、曲率和加速斜率限制。

### 3.6 复验 Town15 原失败点

Town15 不是第一轮 smoke 地图。Town03 稳定后再运行：

```bash
python scripts/sdf.py sim ensure --map Town15 --rhi dx12 --startup-timeout 180 --json

python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town15 \
  --duration-s 180 \
  --inference-mode full \
  --no-map-restart \
  --max-speed 15 \
  --vla-period-s 0.75 \
  --gpu-idle-guard-ms 50 \
  --debug-draw \
  --evidence-dir docs/architecture/evidence/g3-05/town15_soft_commit_180s_15mps
```

关键不是盯着总 accept 率，而是确认原转弯附近：

- 合法帧变为 `accepted` / `accepted_soft_*`；
- path age 不跨入 stale crawl/stop；
- 黄线无近端折角；
- 没有 `unexplained_stop`、持续 off-road 或碰撞卡死。

## 4. 离线验证

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

python -m unittest discover -s tests/g3 -t . -v
python -m unittest tests.g1.test_g1_02_connection -v
python -m compileall -q \
  safedrive_foundry/driving_vla \
  safedrive_foundry/runtime \
  tests/g3
git diff --check
```

发布前最近一次 G3 全量结果为 `Ran 154`、153 passed、1 skipped；skip 是需要真实 GPU
与 checkpoint 的显式 20× forward 方差测试，不是静默跳过业务逻辑。

## 5. 还没有完成什么

### 5.1 纯 VLA+MPC 实用闭环的剩余项

- 最新 soft-commit/单接缝实现尚未得到新的 Town15 live 证据；
- 高速能力尚未证明：cap 20 不代表实际达到 20m/s；
- 复杂路口、障碍物、碰撞后脱困和倒车没有完整解决；
- 无 Safety 的 pure demo 不保证避障或道路安全；
- raw VLA 仍可能产生短时波纹或错误意图；
- 多 seed、固定 spawn、跨地图重复性还未成为正式回归矩阵；
- CARLA DX12 大地图首次 shader 编译仍可能受本机 cache/驱动影响。

### 5.2 正式 G3-05 尚未关闭

当前发布主链故意是 **pure VLA + MPC，不带 Safety**。正式 G3-05 仍要求：

- VLA 轨迹进入 G2 Validator/Safety Kernel；
- 只执行 `executed_trajectory_id` 对应的批准/修复轨迹；
- timeout/stale/NaN 有可审计降级；
- 无 Classic 当前帧候选的 `VLA_SAFETY` 多 seed live；
- 新 Evidence 通过强化后的 `assert_g3_close`。

因此可以准确表述为：

> SafeDrive Foundry 已在单张 RTX 4080 上实现并测量 SimLingo 纯 VLA 轨迹与速度驱动
> CARLA、约束 MPC 执行和可审计 evidence；正式 VLA+Safety 阶段验收仍在收尾。

不能表述为“已证明真实道路自动驾驶安全”或“G3 全部 VERIFIED”。

## 6. 下一轮优化应按什么顺序

1. **先做 post-release live 复验**：Town03 180s + Town15 180s，不再改参数；
2. **固定回归身份**：把 map、spawn、route、seed 和 initial-state hash 固化；
3. **看道路真值而非只看 CTE-to-yellow**：lane-center oracle、侵线、junction 分段；
4. **舒适性门**：steer rate、steer acceleration、jerk 的 P95/P99；
5. **分离 raw 与执行层**：同一 raw route 离线比较 official PID 与当前 MPC；
6. **速度能力**：拆分 speed head、curve、horizon、freshness 四个 limit；
7. **复杂机动**：停车后重新规划、明确 recovery provenance，最后才考虑受限倒车；
8. **正式 Safety 闭环**：复用已有 G2，不在 pure runner 里再造第二套 Safety。

不要回到以下做法：用 HD-map 中心线替代 VLA、用 `set_transform` 把车拖回路面、降低门槛
掩盖碰撞、把 max speed 当速度下限、用普通 path accept 强行唤醒所有停车。

## 7. 如何进入 G4

G4 的正确第一步不是立即跑 MAP-Elites，而是把 G3 已经遇到的失败变成可重复场景。

### 7.1 推荐 G4A 顺序

1. **G4-01 Scenario Registry**：登记 schema、有效性、可解性与 suite；
2. **G4-02 Fixed Replay**：冻结 20–40 个场景、seed、spawn、route、initial-state hash；
3. **G4-04 Comparable K2**：同起点执行两条候选，计算 oracle best-of-K；
4. **G4-05 Acceptance**：输出 `ENTER_WORLD` / `WEAK_SELECTION_SPACE` /
   `NO_SELECTION_SPACE`；
5. G4-03 MAP-Elites 是 optional，不阻塞 G4A。

首批场景建议直接来自当前 G3 evidence：

- Town03 红灯停车/恢复；
- Town04 普通弯道与历史 guardrail/traffic-sign 接触；
- Town15 合法右转 switch/stale 回归；
- VLA timeout/stale/NaN；
- 路线耗尽、地图不匹配、CARLA hang 作为基础设施失败类，而非驾驶失败。

### 7.2 两条可选路线

**正式依赖路线（推荐）**：先恢复 G3-05，补 VLA+Safety live，再启动 G4-01。

```text
读取 START_TASK.md，恢复 G3-05。
```

**提前准备路线**：用户可明确授权只实现 G4-01/G4-02 的 registry/replay 基础设施，
状态标记为 `PRE_G3_CLOSE`，不得据此声称 G3 阶段关闭。

```text
读取 START_TASK.md，启动 G4-01；允许 PRE_G3_CLOSE 基础设施，但不宣称 G3 VERIFIED。
```

## 8. 关键代码与文档

| 路径 | 作用 |
|---|---|
| `tests/g3/run_g3_vla_mpc_minimal.py` | 推荐兼容入口 |
| `tests/g3/run_g3_vla_mpc_stable.py` | 主 live runner / evidence |
| `driving_vla/model/simlingo_contract.py` | 官方相机、JPEG、target 契约 |
| `driving_vla/model/neural_policy.py` | 真 SimLingo path/speed 接入 |
| `driving_vla/runtime/path_manager.py` | pure VLA 路径条件化 |
| `driving_vla/runtime/vla_speed_planner.py` | VLA 速度与停车/恢复 |
| `driving_vla/runtime/vla_mpc_tracker.py` | constrained MPC |
| `runtime/carla_connection.py` | preflight/ensure/冷启动分类 |
| `runtime/carla_engine_config.py` | DefaultEngine map/RHI 原子 pin |
| `G3_VLA_MPC_STABLE_RUNBOOK.md` | D3D、历史故障与详细操作手册 |
| `tasks/G3/G3-05_*.md` | 正式 G3 阶段验收标准 |

## 9. 发布口径

当前里程碑建议命名：

```text
G3 Pure VLA + Constrained MPC Demo — MEASURED_WITH_LIMITS
```

它代表最重要的工程目标已经达成：模型真实看图、输出路径与速度、CARLA 真正按其轨迹
行驶、MPC 约束执行、问题可从 evidence 复盘。它不等同于 Safety 阶段关闭，也不阻止
继续做正式 G3-05 或在明确标记下准备 G4 场景基础设施。
