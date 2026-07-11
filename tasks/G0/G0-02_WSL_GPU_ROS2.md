# G0-02：WSL2、GPU 与 ROS 2 基础环境

**状态**：COMPLETED  
**依赖**：G0-01

## 目标

一次性建立冻结版本要求的 WSL2 Ubuntu、RTX 4080 计算环境和 ROS 2 基础，并验证重启后仍可使用。

## 范围

- 安装或调整冻结版本规定的 WSL2/Ubuntu 环境。
- 验证 WSL 内 GPU、CUDA/PyTorch 设备可见性。
- 安装 ROS 2、colcon 与基础开发工具。
- 验证 talker/listener、`ros2 doctor` 和最小 workspace 构建。
- 不安装 CARLA，不实现业务节点。

## 完成标准

- [x] WSL2 发行版与冻结方案一致。
- [x] WSL 中识别 RTX 4080 和 16GB 显存。
- [x] ROS 2 通信和 colcon 构建通过。
- [x] Python 环境、DDS、域 ID、QoS 和工作区路径约定有记录。
- [x] 需要管理员权限或重启的操作均已完成并重新验证。

详细基线与证据见 `docs/environment/WSL_ROS2_BASELINE.md`。

## 验证方法

- 保存 WSL 状态、Ubuntu 版本、GPU 可见性和 ROS 诊断结果。
- 在新终端及必要重启后重复 GPU 与 ROS 通信检查。
- 构建最小 colcon 工作区并确认包可发现。
- 更新断点和进度后停止。

## 断点记录

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-11（Asia/Singapore） |
| 最后状态 | COMPLETED |
| 已完成 | 启用 WSL/VirtualMachinePlatform 并完成 Windows 重启；安装 WSL 2.9.3.0 与 Ubuntu 24.04.4 WSL2；创建默认用户 `sdf`；安装 ROS 2 Jazzy Desktop、开发工具和 colcon；建立 Python venv 与 PyTorch CUDA 环境；建立 ROS/DDS/域 ID/QoS/工作区约定。 |
| 修改文件 | `versions.lock`、`docs/environment/WSL_ROS2_BASELINE.md`、本任务文件、`PROGRESS.md` |
| 已运行验证 | `wsl --version/status/-l -v`；Ubuntu 版本、内核和用户检查；`nvidia-smi`；PyTorch CUDA 张量实算；`ros2 doctor`；两轮 talker/listener（含 WSL terminate 后复测）；`colcon build --symlink-install`；`ros2 pkg prefix g0_smoke`；`dpkg --audit`。 |
| 结果摘要 | RTX 4080 16376 MiB 可见；PyTorch 2.12.1+cu126 CUDA 可用且实算正确；`ros2 doctor` 5/5；两轮 ROS 通信成功；最小工作区 1 个包构建成功。 |
| 失败或阻塞 | 无完成阻塞。`systemd-binfmt.service` 降级、Mihomo Fake-IP 网络限制及两个 ROS 包更新提示均已记录为非阻塞项。`torchaudio 2.12.1` 无 cu126 wheel，且项目无需音频，故未安装并修正冻结记录。 |
| 精确恢复步骤 | 无；本任务已完成。等待用户明确指定下一任务。 |
| 下一条建议命令 | 无；不要自动开始 G0-03。 |
