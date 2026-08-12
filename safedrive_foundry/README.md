# SafeDrive Foundry runtime

活动 runtime 服务 H 路线：

```text
Observable → Classic Expert + nominal VLA → per-candidate Guard
           → World rank/defer → Safety → MPC/PID
```

当前已保留 CARLA/ROS 连接、确定性 runtime、Classic/Safety、nominal SimLingo policy、
轨迹 canonicalizer、PathManager 与 constrained MPC/PID。H1 的双来源 candidate set 与
逐候选 Guard 尚未实现；World Model 尚未实现。

权威入口：

- `../START_TASK.md`
- `../ROADMAP.md`
- `../docs/PROJECT.md`
- `../docs/HYBRID_CANDIDATES.md`
- `../docs/WORLD_MODEL.md`
- `../docs/ENVIRONMENT.md`
- `../docs/RESOURCES.md`

## 固定环境

- CARLA Server：Windows `E:\CARLA_0.9.16\CarlaUE4.exe`
- runtime/model：WSL2 Ubuntu 24.04
- ROS 2 Jazzy，`ROS_DOMAIN_ID=42`
- VLA venv：`/home/sdf/.venvs/sdf`
- 资源：RTX 4080 16GB 与 i5-13600KF
- host 不写死，由 `scripts/sdf.py sim` 动态解析
- 只允许已登记 runtime 成为 tick master

## 连接入口

```bash
cd "/mnt/e/autonomous driving"
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py sim status
python scripts/sdf.py sim preflight --json
```

只有 `READY` 才能继续真实 CARLA 任务。CARLA 未运行且返回 `RETRYABLE_FAILURE` 时只
允许一次：

```bash
python scripts/sdf.py sim ensure --map Town03 --rhi dx12 --startup-timeout 180 --json
```

随后只复查一次。需要 GUI/UAC、版本冲突、tick owner 冲突或依赖冲突时停止。

## ROS 2

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd "/mnt/e/autonomous driving/safedrive_foundry/ros_ws"
colcon build --symlink-install
source install/setup.bash
```

业务节点不得创建第二套 `carla.Client`、调用 `world.tick()` 或让模型直接控制底盘。
