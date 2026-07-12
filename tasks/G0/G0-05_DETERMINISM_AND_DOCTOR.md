# G0-05：确定性同步与环境诊断

**状态**：COMPLETED
**依赖**：G0-04

## 目标

完成固定步长、唯一Tick Master、`/clock`与frame契约，并建立`sdf doctor`和可恢复G0烟雾测试入口。

## 范围

- 定义`episode_id + carla_frame`主键和命令生成/计划/执行帧。
- 实现最小同步控制、丢帧/乱序/过期检测。
- 建立环境、版本、GPU、端口、CARLA、ROS、时钟、磁盘和路径诊断。
- 只覆盖G0诊断，不实现后续训练或场景搜索命令。

## 完成标准

- 全系统只有一个tick master，固定步长和substep参数合法。
- `/clock`、snapshot和ROS消息frame一致。
- 相同seed重复测试在规定容差内一致。
- `sdf doctor`输出PASS/WARN/FAIL/BLOCKED及JSON/Markdown报告。
- CARLA未启动、端口冲突、版本不符、GPU不可见和磁盘不足可识别。

## 验证方法

- 两次相同实验比较frame、时间戳、事件顺序和关键状态hash。
- 注入重复tick、缺帧、过期消息及至少三种环境故障。
- 验证中断后可安全重跑或从合法检查点恢复。
- 更新断点和进度后停止。

## 断点记录

| 字段 | 内容 |
|---|---|
| 最后状态 | 2026-07-12 `COMPLETED` |
| 已完成 | 冻结 `episode_id + carla_frame` 主键、command 生成/计划/执行帧和 snapshot/message/clock 对齐规则；实现固定步长与子步合法性校验、唯一 tick-master lease、重复 tick/缺帧/过期/乱序检测、原子 checkpoint 恢复、离线确定性 trace 比较、CARLA live sync driver 和 `sdf doctor` JSON/Markdown 诊断；修正 ROS 2 Jazzy 不支持 `ros2 --version` 导致的 doctor 误报；补装用户 site 的 CARLA 0.9.16 Linux client；修正 CARLA 0.9.16 `WorldSettings` 不支持 `copy.copy()` 的现场兼容性问题；完成现场 frame/clock/tick-owner 验证。 |
| 修改文件 | `safedrive_foundry/config/carla_ros.toml`、`safedrive_foundry/versions.lock`、`safedrive_foundry/README.md`、`safedrive_foundry/ros_ws/src/safedrive_carla_bridge/package.xml`、`setup.py`、`bridge_node.py`、`sync_contract.py`、`doctor.py`、`cli.py`、`sync_driver.py`、`scripts/sdf.py`、`sdf.cmd`、`sdf`、`tests/g0/test_g0_05.py`、`docs/environment/G0_05_DETERMINISM.md`、`docs/environment/evidence/g0-05/*`。 |
| 已运行验证 | `colcon build --symlink-install`：1 package finished；`python3 -m compileall -q ...` 通过；`python3 -m unittest discover -s tests -v`：5/5 通过；`python3 scripts/sdf.py validate-g0`：14/14 通过；seed 2026 两次 16 帧 trace 比较通过；seed 303 中断返回 75、恢复后与 clean run 的 10 帧 trace 比较通过；现场 doctor 的 WSL/ROS/GPU/CARLA path/RPC/version/world/2000/2001/2002 检查通过；40-tick driver 返回 0；20 条 `/clock` 与 20 条状态消息对齐通过，最大误差 `4.9965e-10 s`；唯一 publisher/tick master graph 检查通过；退出恢复检查通过。 |
| 阻塞 | 无。CARLA 进程、TCP 端口、ROS 2、frame/clock 对齐、唯一 tick master 和退出恢复均已现场验证。 |
| 恢复步骤 | 若需复测：启动 `E:\CARLA_0.9.16\CarlaUE4.exe`，确认 `CARLA_ROOT=/mnt/e/CARLA_0.9.16 python3 scripts/sdf.py doctor`，source Jazzy 与 `ros_ws/install`，设置 `ROS_DOMAIN_ID=42`/`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`，运行 `ros2 run safedrive_carla_bridge carla_sync_driver --host 172.30.80.1 --port 2000 --steps 20`；不要在本任务完成后自动开始 G0-06。 |
| 下一建议命令 | 等待用户明确指令后再执行 G0-06；本次不启动 G0-06。 |
