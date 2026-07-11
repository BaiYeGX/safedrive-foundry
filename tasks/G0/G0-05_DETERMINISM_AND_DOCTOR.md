# G0-05：确定性同步与环境诊断

**状态**：BLOCKED_EXTERNAL  
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
| 最后状态 | 2026-07-12 `BLOCKED_EXTERNAL` |
| 已完成 | 冻结 `episode_id + carla_frame` 主键、command 生成/计划/执行帧和 snapshot/message/clock 对齐规则；实现固定步长与子步合法性校验、唯一 tick-master lease、重复 tick/缺帧/过期/乱序检测、原子 checkpoint 恢复、离线确定性 trace 比较、CARLA live sync driver 和 `sdf doctor` JSON/Markdown 诊断。 |
| 修改文件 | `safedrive_foundry/config/carla_ros.toml`、`safedrive_foundry/versions.lock`、`safedrive_foundry/README.md`、`safedrive_foundry/ros_ws/src/safedrive_carla_bridge/package.xml`、`setup.py`、`bridge_node.py`、`sync_contract.py`、`doctor.py`、`cli.py`、`sync_driver.py`、`scripts/sdf.py`、`sdf.cmd`、`sdf`、`tests/g0/test_g0_05.py`、`docs/environment/G0_05_DETERMINISM.md`、`docs/environment/evidence/g0-05/*`。 |
| 已运行验证 | `python -m compileall ...` 通过；`python -m unittest discover -s tests -v`：5/5 通过；`python scripts/sdf.py validate-g0`：14/14 通过；seed 2026 两次 16 帧 trace 比较通过；seed 303 中断返回 75、恢复后与 clean run 的 10 帧 trace 比较通过；`sdf doctor` 输出 JSON/Markdown 并识别 CARLA 未启动、WSL 无发行版和 ROS `/clock` 阻塞。 |
| 阻塞 | 当前会话 `wsl.exe -l -q` 没有可用注册发行版，ROS 2 与现场 `/clock` 观察为 `BLOCKED`；配置的 CARLA `172.30.80.1:2000` 未启动/不可达。完成现场门禁需要外部 WSL 发行版注册/启动、CARLA Server 启动以及 ROS 2 GUI/跨系统环境操作；不能把离线验证替代现场对齐证据。 |
| 恢复步骤 | 1. 用户完成最小外部步骤：注册并启动 `Ubuntu-24.04` WSL 发行版；启动 `E:\CARLA_0.9.16\CarlaUE4.exe` 的 `2000/2001/2002` 端口。 2. 运行 `python scripts/sdf.py doctor`，确认 WSL、CARLA RPC、ROS 2 不再为 `BLOCKED/FAIL`。 3. 运行 `python scripts/sdf.py sync-smoke --carla --steps 20 --run-id live-carla`。 4. 在 WSL 构建并运行 `ros2 run safedrive_carla_bridge carla_sync_driver --steps 20`，同时记录 `/clock` 与 `/safedrive/carla/status` 的 frame 对齐。 5. 通过后更新本任务状态和 `PROGRESS.md` 为 `COMPLETED`；在此之前不要开始 G0-06。 |
| 下一建议命令 | `python scripts/sdf.py doctor` |
