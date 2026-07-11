# G0-04：工程骨架与CARLA–ROS跨系统连通

**状态**：COMPLETED  
**依赖**：G0-02、G0-03

## 目标

创建不含业务算法的新工程骨架，并打通Windows CARLA、WSL2 Client和ROS 2的最小数据链路。

## 范围

- 创建`safedrive_foundry/`目录、README、配置约定、`.gitignore`和版本锁。
- 配置Windows/WSL地址、端口、防火墙与Python API路径。
- 从WSL读取CARLA world，并发布带frame/timestamp的最小ROS消息。
- 不实现完整bridge、规划、控制、VLA或世界模型。

## 完成标准

- 工程骨架符合总需求或有记录的调整。
- 大文件、凭据和生成数据默认忽略。
- WSL Client可重复连接Windows CARLA。
- 最小CARLA状态可发布和订阅为ROS消息。
- 断连、版本不匹配和端口占用有明确诊断。

## 验证方法

- 检查目录、Markdown链接、配置示例和忽略规则。
- 重复执行CARLA连接及ROS消息收发。
- 模拟server未启动并检查错误路径。
- 更新断点和进度后停止。

## 断点记录

| 字段 | 内容 |
|---|---|
| 最后状态 | 2026-07-12 `COMPLETED` |
| 已完成 | 创建 `safedrive_foundry/` 骨架、README、配置、版本锁、`.gitignore`、ament_python 包和最小 CARLA snapshot→ROS String bridge；修复 `setup.cfg` 安装路径与 CARLA 0.9.16 Timestamp 无 `episode_id` 的接口差异 |
| 修改文件 | `safedrive_foundry/README.md`、`safedrive_foundry/config/carla_ros.toml`、`safedrive_foundry/versions.lock`、`safedrive_foundry/.gitignore`、`safedrive_foundry/ros_ws/src/safedrive_carla_bridge/*`、`docs/environment/CARLA_ROS_CONNECTIVITY.md`、本任务文件、`PROGRESS.md` |
| 已运行验证 | WSL colcon 构建成功；WSL CARLA client 两次/多次连接读取 0.9.16 world；ROS topic 两次订阅收到 frame/timestamp/episode 消息；2099 错误端口快速诊断；错误期望版本给出 client/server mismatch；清理后 CARLA/UE4 进程和 2000～2002 端口均为 0 |
| 阻塞 | 无。WSL 当前 host gateway 为 `172.30.80.1`，已写入配置并提供动态获取命令 |
| 恢复步骤 | 复测时启动 `E:\CARLA_0.9.16`，在 WSL source ROS/setup 后设置 `CARLA_HOST` 为 default gateway，构建并运行 bridge；固定步长和唯一 tick master 留给 G0-05 |
| 下一建议命令 | `wsl -d Ubuntu-24.04 -- bash -lc 'source /opt/ros/jazzy/setup.bash; source /home/sdf/safedrive_foundry/ros_ws/install/setup.bash; ros2 pkg executables safedrive_carla_bridge'` |
