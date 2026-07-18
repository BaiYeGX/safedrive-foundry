# 项目当前进度

> 本文件只记录动态执行状态。任务编号、名称和依赖以 `ROADMAP.md` 为唯一来源；任务范围、验收和断点以对应任务文件为准。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前阶段 | **G3：IN_PROGRESS** |
| 当前任务 | G3-05 pure VLA + constrained MPC 稳定化（非 Safety 阶段关闭） |
| 当前状态 | **`IMPLEMENTED_OFFLINE / LIVE_REVALIDATION_REQUIRED`**；新速度/长测修复尚未 live 验收 |
| 最近完成 | VLA 速度上限语义、短/陈旧路径制动、600m 路线续接、长测 evidence、Large Map hero/streaming；G3 离线 64 tests PASS |
| 推荐下一动作 | 用户更新驱动并重启 Windows；preflight READY 后依次跑 20s/60s/300s 验收 |
| 最近更新 | 2026-07-19 |
| 工作分支 | `codex/g3-vla-mpc-stabilization`（未提交、未 push） |
| 过程日志 | `docs/project/G3_EXECUTION_JOURNAL.md` |

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| G0 | `COMPLETED / FROZEN` | 冻结 |
| G1 | `COMPLETED_WITH_LIMITS` | 正式关闭 |
| G2 | `COMPLETED_WITH_LIMITS` | offline Safety Kernel |
| G3 | **`IN_PROGRESS`** | Demo 通过 ≠ 阶段 VERIFIED |
| G4 | `PENDING` | **禁止**自动启动 |

## G3-05 Pure VLA + Constrained MPC 稳定化（2026-07-18）

| 字段 | 当前事实 |
|---|---|
| 状态 | `IMPLEMENTED_OFFLINE / LIVE_REVALIDATION_REQUIRED` |
| 正式 demo 入口 | `tests/g3/run_g3_vla_mpc_minimal.py`（兼容入口，转到 `run_g3_vla_mpc_stable.py`） |
| 几何来源 | SimLingo 原生 20 点空间路径；地图仅提供约 15m/30m 粗导航目标 |
| 跟踪器 | G3 专用 2s constrained MPC（舵角、舵速、舵加速度约束） |
| GPU 调度 | inference 与 CARLA tick 串行；forward 后 CUDA synchronize；debug draw 降频 |
| 相机 | 与 SimLingo 标定统一为 x=-1.5m、z=2.0m、FOV=110°、2:1 输入 |
| 速度语义 | VLA speed head 为主；`--v-ref` 仅为硬上限；无正速度 floor；VLA 制动立即生效 |
| 长测 | 600m coarse route 滚动续接；60s checkpoint；碰撞/越线/离路/速度/VRAM evidence |
| Large Map | Town11/12/13 hero + tile streaming；默认随机池排除；Town13 仍需 live 复验 |
| 离线验证 | `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`：64/64 PASS |
| live preflight | `SERVER_NOT_RUNNING`；一次 `sim ensure` 返回 `BLOCKED_EXTERNAL / NEEDS_USER_ACTION`（WSL vsock permission） |
| runbook | `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md` |

本轮没有把旧 visual demo 的历史 `DEMO_PASS` 升格为新管线的通过，也没有宣称
G3 `VERIFIED`。Windows 手动恢复后须先跑 20 秒、60 秒，再跑 300 秒 15m/s 上限
耐久测试；任一项未达到新 summary 的完整门槛均为 `DEMO_FAIL`。

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
