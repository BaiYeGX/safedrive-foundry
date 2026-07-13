# G0-04 CARLA–ROS 2 跨系统连通证据

日期：2026-07-12（Asia/Singapore）  
状态：已完成验证（**下列正文为 G0 冻结现场记录**）

## 现行连接方式（2026-07-13 起，覆盖正文中的固定 IP 指引）

| 项 | 现行约定 |
|---|---|
| 入口 | WSL 内 `python3 scripts/sdf.py sim preflight` / `status` / `ensure` |
| Host | **禁止写死**；`ConnectionResolver` 动态解析 |
| 常见可达 | WSL **镜像网络**下 Windows CARLA 常在 `127.0.0.1:2000`；经典 NAT 模式则用 default gateway |
| 代理干扰 | `198.18.0.0/15` 等透明代理可能 TCP 假连通，必须以 RPC handshake 为准 |
| 路径 | `E:\CARLA_0.9.16` ↔ `/mnt/e/CARLA_0.9.16` |
| 官方地图 | `/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/*.xodr` |

正文中的 `172.30.80.1` 与“`127.0.0.1` 不可达”是 **G0-04 当日 NAT 模式** 的测量结果，**不得**再当作永久真理复制到新任务。

## 实际环境（G0-04 当日）

- Windows CARLA Server：`E:\CARLA_0.9.16`，server `0.9.16`
- WSL：`Ubuntu-24.04`，WSL2，Python `3.12.3`
- ROS 2：Jazzy，`ROS_DOMAIN_ID=42`，`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- WSL CARLA API：Linux `carla==0.9.16`，位于 `/home/sdf/.venvs/carla_ros`
- 当日 WSL 默认路由采样：`172.30.80.1`；`172.30.80.1:2000` TCP 探测成功（历史）
- ROS package：`safedrive_carla_bridge`，`carla_status_bridge`
- ROS topic：`/safedrive/carla/status`，类型 `std_msgs/msg/String`

## 构建证据

```text
Starting >>> safedrive_carla_bridge
Finished <<< safedrive_carla_bridge [约2 s]
Summary: 1 package finished
safedrive_carla_bridge carla_status_bridge
```

构建前通过 `setup.cfg` 固定 ament_python 的脚本安装路径：

```ini
[develop]
script_dir=$base/lib/safedrive_carla_bridge

[install]
install_scripts=$base/lib/safedrive_carla_bridge
```

## WSL → Windows CARLA 连接

```text
client= 0.9.16
server= 0.9.16
map= Carla/Maps/Town10HD_Opt
frame= 47426
```

G0-04 当日：`127.0.0.1:2000` 在**当时的 NAT 模式**下不可达；使用路由表 default gateway `172.30.80.1` 后连接成功。
**后续（含镜像网络）**：`127.0.0.1` 可能再次可达；请用 `sdf sim preflight` 记录实际 `host` / `host_source`，不要复制历史 IP。

## ROS 收发

bridge 启动日志：

```text
connected CARLA client=0.9.16 server=0.9.16
map=Carla/Maps/Town10HD_Opt endpoint=172.30.80.1:2000
topic=/safedrive/carla/status
```

`ros2 topic info`：

```text
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 0
```

两次独立 `ros2 topic echo --once` 均收到消息，示例字段如下：

```json
{
  "carla_frame": 147892,
  "delta_seconds": 0.007606,
  "endpoint": "172.30.80.1:2000",
  "episode_id": "a41eddcc686a4cf9b6...",
  "map": "Carla/Maps/Town10HD_Opt",
  "simulation_seconds": "present",
  "publisher_wall_time": "present"
}
```

早期实现错误地读取 `snapshot.timestamp.episode_id`；CARLA 0.9.16 的 Timestamp 没有该属性。现实现为 bridge 进程每次启动生成一个 session `episode_id`，并保留 CARLA `carla_frame`、仿真时间和 wall time。

## 诊断门禁

错误端口：

```text
CARLA connection failed at 172.30.80.1:2099; check server, port, firewall and Python API version
```

连接超时为配置的 2 秒，错误可读。

版本不匹配模拟（仅设置错误期望字符串 `CARLA_EXPECTED_VERSION=0.9.15`，未使用或恢复 0.9.15 文件）：

```text
CARLA version mismatch: expected=0.9.15, client=0.9.16, server=0.9.16
```

端口 2000 的 TCP 占用已由 `nc -zvw2 172.30.80.1 2000` 确认开放；bridge 连接日志同时记录 endpoint，便于区分端口占用/错误地址与 ROS 发现问题。

## 范围边界

- bridge 只读取 `world.get_snapshot()`，不调用 `world.tick()`。
- 不包含规划、控制、VLA、世界模型或 Safety Kernel。
- 固定步长和唯一 tick master 留给 G0-05。
