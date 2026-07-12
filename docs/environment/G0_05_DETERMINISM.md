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

2026-07-12 现场门禁已通过，doctor 报告位于：

- [doctor.json](./evidence/g0-05/doctor.json)
- [doctor.md](./evidence/g0-05/doctor.md)

现场结果：

- `CarlaUE4`/`CarlaUE4-Win64-Shipping` 进程存在，TCP `172.30.80.1:2000/2001/2002` 全部可达；CARLA 0.9.16 client/server version 与 Town10HD world handshake 通过。
- `carla_sync_driver --steps 40` 返回码为 `0`；observer 收到 20 条 `/clock` 和 20 条 `/safedrive/carla/status`。
- `snapshot_frame == message_frame == clock_frame == carla_frame`、frame 严格递增、`clock_seconds` 步长 `0.05` 均通过；ROS `/clock` 与状态 `clock_seconds` 最大误差 `4.9965e-10 s`。
- ROS graph 显示 `/clock` 与 status 各只有一个 publisher，均为 `safedrive_carla_sync_driver`；状态消息的 tick master 均为 `sdf.g0-05.sync`。
- driver 退出后 `WorldSettings` 恢复为 `synchronous_mode=False`，无残留 sync driver、`/clock` 或 status publisher。

详细现场证据：

- `evidence/g0-05/live-20260712-030454/observer.json`：frame/clock 对齐与 20 条样本；
- `evidence/g0-05/live-owner-20260712-030603/observer.json`：唯一 tick master 与 publisher graph；
- `evidence/g0-05/live-owner-20260712-030603/preflight.txt`、`post_driver.txt`：进程、端口、API 与退出恢复。

`sdf doctor` 的 ROS 探测已改用 Jazzy 支持的 `ros2 --help`，不再把 `ros2 --version` 误报为不可用。G0-05 现场门禁完成，任务状态为 `COMPLETED`；本次不启动 G0-06。

若需复测，先启动 Windows CARLA，再运行（WSL shell 使用 Linux 路径覆盖 CARLA 安装路径）：

```bash
CARLA_ROOT=/mnt/e/CARLA_0.9.16 python3 scripts/sdf.py doctor
CARLA_ROOT=/mnt/e/CARLA_0.9.16 python3 scripts/sdf.py sync-smoke --carla --steps 20 --run-id live-carla
```

在 WSL 中构建并运行 ROS driver：`source /opt/ros/jazzy/setup.bash && source safedrive_foundry/ros_ws/install/setup.bash && export ROS_DOMAIN_ID=42 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && ros2 run safedrive_carla_bridge carla_sync_driver --host 172.30.80.1 --port 2000 --steps 20`；同时用 `ros2 topic echo /clock` 与 `ros2 topic echo /safedrive/carla/status` 观察并核对同一批 `carla_frame` 的时间和帧契约。复测时将结果追加到本文件和 G0-05 断点记录。
