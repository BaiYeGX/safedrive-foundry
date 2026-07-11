# G0-05 确定性同步与环境诊断

## 已实现的契约

| 项目 | 固定约定 |
|---|---|
| 主键 | `episode_id + carla_frame` |
| Tick master | `sdf.g0-05.sync`，进程内 lease 防止第二个 owner |
| CARLA 时间 | `synchronous_mode=true`，`fixed_delta_seconds=0.05` |
| 子步 | `substepping=true`，`max_substep_delta_time=0.01`，`max_substeps=5` |
| ROS 时间 | `/clock` 由 sync driver 从同一 snapshot 的 `elapsed_seconds` 生成 |
| 帧字段 | `snapshot_frame == message_frame == clock_frame == carla_frame` |
| command lineage | `command_generated_frame <= command_planned_frame <= command_executed_frame` |
| 诊断状态 | `PASS`、`WARN`、`FAIL`、`BLOCKED` |

只读 `carla_status_bridge` 不调用 `world.tick()`，并拒绝重复、跳帧、过期或乱序 snapshot；`carla_sync_driver` 是唯一的 tick 与 `/clock` 发布者。

## 验证结果

已通过：

- `python -m compileall ...`：通过；
- `python -m unittest discover -s tests -v`：5 个 G0-05 单元测试通过；
- `sdf validate-g0`：14/14 通过，覆盖固定步长合法性、唯一 tick master、双跑一致性、重复 tick/缺帧/过期消息、CARLA 未启动/端口冲突/版本不匹配/GPU 不可见/磁盘不足故障分类、CARLA settings 校验和 checkpoint 恢复；
- seed `2026` 的两次 16 帧离线实验：frame、时间戳、事件顺序和关键状态 hash 比较通过；
- seed `303`：中断后以合法 checkpoint 恢复，与 clean run 的 10 帧 trace 比较通过。

## 当前现场门禁

2026-07-12 在当前 Windows 会话执行 `sdf doctor` 的结果为 `FAIL`，报告位于：

- [doctor.json](./evidence/g0-05/doctor.json)
- [doctor.md](./evidence/g0-05/doctor.md)

已识别的外部状态：RTX 4080、版本锁、磁盘、CARLA 安装路径和同步配置通过；CARLA RPC `172.30.80.1:2000` 未连通；当前 `wsl.exe` 没有可用注册发行版，因此 ROS 2 和现场 `/clock` 观察为 `BLOCKED`。因此现场 CARLA/ROS `/clock` 对齐尚未宣称通过，G0-05 保持 `BLOCKED_EXTERNAL`，恢复后必须继续同一任务，不得开始 G0-06。

恢复时先完成 WSL 发行版注册并启动 CARLA，再运行：

```powershell
python scripts/sdf.py doctor
python scripts/sdf.py sync-smoke --carla --steps 20 --run-id live-carla
```

在 WSL 中构建并运行 ROS driver，然后用 `ros2 topic echo /clock` 与 `/safedrive/carla/status` 比较同一批 `carla_frame`；完成后把现场结果追加到本文件和 G0-05 断点记录。
