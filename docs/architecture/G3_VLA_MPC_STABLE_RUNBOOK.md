# G3 Pure VLA + Constrained MPC 稳定运行手册

**状态**：`IMPLEMENTED_OFFLINE / LIVE_REVALIDATION_REQUIRED`
**目标**：先验收直线和大半径弧线；路口、掉头随后单独验收。

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

地图路线只用于生成约15m/30m的粗导航目标，不作为控制参考。

## Windows 重启前后

重启前无需继续启动 CARLA。2026-07-18 的一次规定恢复尝试返回
`BLOCKED_EXTERNAL / NEEDS_USER_ACTION`，错误为 WSL vsock 权限失败；不要继续循环
`ensure`。用户完成 NVIDIA 驱动更新并重启 Windows 后：

1. 确认 `carla_start.toml` 是 Town04、DX11、Low、800×600。
2. 从 WSL 执行一次 `sdf sim preflight`。
3. 仅当结果为 `RETRYABLE_FAILURE` 时执行一次 `sdf sim ensure`，再执行一次 preflight。
4. `READY` 后先运行20秒验收。

```bash
source /home/sdf/.venvs/sdf/bin/activate
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 20 \
  --v-ref 6 \
  --vla-period-s 0.50
```

20秒结果为 `DEMO_PASS` 后，再运行60秒：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town04 \
  --duration-s 60 \
  --v-ref 10 \
  --vla-period-s 0.50
```

两轮都通过后，再跑用户要求的 5 分钟快速耐久测试。Town10HD 已有较好的短测
基础，且不像 Town13 那样依赖超大瓦片加载：

```bash
python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town10HD \
  --duration-s 300 \
  --max-speed 15 \
  --speed-gain 1.5 \
  --vla-period-s 0.50 \
  --route-segment-m 600
```

`--max-speed 15`（兼容旧名 `--v-ref 15`）是绝对上限，不再把速度托到
12.75m/s。实际目标仍由 VLA speed
head 决定；校准后最高 15m/s。原生路径通常只有约 19m，因此 MPC 还会按可见
制动距离限速，直线实际可能先受约 10m/s 的路径视距上限约束。这是物理可停车
约束，不是强制慢速。

## 可视化

- 绿色：当次 raw SimLingo 空间路径。
- 黄色：实际交给 MPC 的 committed/ensembled 路径。
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
- 是否达到与测试时长匹配的最小里程。

长测每 60 秒写入 `progress_latest.json`。即使 CARLA 随后崩溃，也能看到崩溃前
的实际速度、里程、VLA 接受数、碰撞和离路状态；运行异常另写
`failure_latest.json`。

单纯“没有转圈”不再判定通过。

## D3D 诊断顺序

如果重启后仍出现 `0x887A0020`：

1. CARLA only，运行10分钟；
2. CARLA + RGB camera，无模型；
3. CARLA + 模型常驻，无forward；
4. 当前默认串行2Hz forward；
5. 最后打开debug draw。

当前 demo 在 VLA forward 结束后执行 `torch.cuda.synchronize()`，再继续
`world.tick()`，避免单卡上 UE D3D 渲染与 CUDA forward 高峰重叠。

Town11/12/13 是 Large Map：入口会把车辆设为 `role_name=hero`，先把 spectator
移到出生瓦片并预热 streaming。它降低空瓦片/空指针风险，但 Town13 曾出现的
UE `EXCEPTION_ACCESS_VIOLATION 0x8` 只有重启后真实运行才能确认；因此默认随机
地图池不包含 Town11/12/13。它们只能显式 `--map Town13` 运行。

`TdrDelay` 不作为首要修复；只有事件日志确认 TDR 且驱动/负载隔离仍复现时，
才作为用户手动诊断变量。
