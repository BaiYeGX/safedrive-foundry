# G3 Pure VLA + Constrained MPC 基线

本文是当前 K1 基线的唯一活动运行说明。详细历史诊断、长测日志和原始 Evidence 已
归档，索引见 `EVIDENCE.md`。

## 1. 当前结论

状态：

```text
K1 pure VLA + constrained MPC: MEASURED_WITH_LIMITS
real K2: REPAIR_REQUIRED
Safety live acceptance: optional / not verified
```

已经成立：

- SimLingo/InternVL2-1B 真实 CUDA forward；
- 原生 path 与 speed head；
- 官方相机/预处理/coarse-target 契约；
- PathManager 连续路径接入；
- constrained MPC 实际控制 CARLA；
- DX12 下完成多轮真实 forward 和闭环测量；
- run/config/轨迹/控制/碰撞证据可审计。

尚未成立：

- 真实 K2；
- World on/off；
- 多场景稳定效果；
- 完整 Safety 工程闭环；
- 实车或道路安全。

## 2. 运行链

```text
CARLA front RGB + ego + coarse target
  → SimLingo
  → pred_route + pred_speed_wps
  → VLAPathManager
  → ConstrainedVLAMPC
  → carla.VehicleControl
```

地图只提供 coarse navigation target，不把 HD-map 中心线当作 MPC 参考。

## 3. 冻结输入契约

```text
camera: 1024x512
FOV: 110 deg
pose: (-1.5, 0, 2.0)
preprocess: BGRA → BGR → OpenCV JPEG → RGB → official crop
navigation: RoutePlanner 7.5m semantic targets
RHI: dx12
```

`pred_route` 进入 PathManager/MPC；`pred_speed_wps` 进入速度规划。`--max-speed` 是
硬上限，不是最低速度。

## 4. 推荐复现

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

python scripts/sdf.py sim preflight --json

python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town03 \
  --duration-s 60 \
  --inference-mode full \
  --no-map-restart \
  --max-speed 6 \
  --debug-draw \
  --evidence-dir docs/runtime-evidence/g3-k1-smoke
```

该命令只复现 K1 基线，不是当前正式开发任务。新的运行证据写
`docs/runtime-evidence/`；冻结后再移入 archive。

## 5. PathManager 边界

硬拒绝：

- 退化/过短/非前向/自交；
- 硬曲率异常；
- 与 coarse navigation 明显反向；
- NaN/时间/身份错误。

连续性问题优先软接入：

- lateral/heading switch；
- 软曲率；
- old→latest 单接缝；
- blend 异常时 latest-only。

禁止用 HD-map 中心线替换 VLA、用 `set_transform` 拖车、移动阈值掩盖碰撞或用
普通 path accept 覆盖 VLA 语义停车。

## 6. 速度与停车

- VLA speed head 为主；
- 减速立即生效，提速受加速度斜率限制；
- 停车时给 VLA 真实 0m/s；
- startup assist 只允许首次起步；
- 红灯语义停车与 freshness 停车分开；
- 碰撞后禁止伪速度输入；
- stationary requery 默认关闭。

## 7. 已知限制

- K1 没有 World 可利用的候选选择空间；
- 高速、复杂路口、障碍物、倒车和脱困未完整解决；
- raw VLA 可能产生波纹或错误意图；
- 低速短测通过不能外推到高速长稳；
- 单张 4080 同时运行 CARLA 与 CUDA，需要严格 profile；
- DX11 是已知失败基线，当前只维护 DX12；
- 无 Safety 路径不保证避障或道路安全。

## 8. 故障诊断顺序

1. 查看 preflight 和有效 RHI；
2. 确认使用 sdf venv；
3. 检查 inference accept/reject、path age、freshness；
4. 区分 VLA stop、红灯 stop、execution stale 和碰撞后 stop；
5. 查看 collision episode 而不是 raw contact count；
6. 区分 D3D/Server 故障与驾驶质量失败；
7. 不在未归因前调 MPC 或放宽 Guard。

## 9. 当前下一步

不要继续优化 K1 演示。当前唯一动作是完成真实 K2：

```text
读取 START_TASK.md，开始当前任务。
```
