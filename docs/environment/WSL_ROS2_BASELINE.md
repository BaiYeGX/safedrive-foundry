# G0-02 WSL、GPU 与 ROS 2 基线

记录时间：2026-07-11（Asia/Singapore）  
状态：已验证

## 冻结环境

- Windows：Windows 11 Pro 25H2，build `26200.8655`
- WSL：`2.9.3.0`，内核 `6.18.35.2-1`
- 发行版：Ubuntu `24.04.4 LTS`，WSL2，默认用户 `sdf`（UID 1000）
- GPU：NVIDIA GeForce RTX 4080，`16376 MiB`，Windows 驱动 `591.86`
- Python：系统 Python `3.12.3`
- 虚拟环境：`/home/sdf/.venvs/sdf`
- PyTorch：`2.12.1+cu126`；torchvision `0.27.1+cu126`
- ROS 2：Jazzy Desktop；`ros-jazzy-desktop 0.11.0-1noble.20260616.084553`
- 开发工具：`ros-dev-tools 1.0.1`，`python3-colcon-common-extensions 0.3.0-100`
- 工作区：`/home/sdf/safedrive_ws`

`torchaudio 2.12.1` 没有官方 `cu126` wheel，且当前项目不需要音频能力，因此未安装；不得将不存在的构建写成已冻结依赖。

## ROS 约定

- `ROS_DOMAIN_ID=42`
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- 交互式 shell 通过 `/home/sdf/.bashrc` 自动加载 `/opt/ros/jazzy/setup.bash` 和上述变量。
- 控制、状态和离散事件默认采用 reliable + volatile；高频传感器流默认采用 best effort + volatile、depth 5。具体接口若有任务级定义，以接口定义为准。

## 验证证据

- `wsl --version`、`wsl -l -v`：发行版运行于 WSL2。
- WSL 内 `nvidia-smi`：识别 RTX 4080、16376 MiB 和驱动 591.86。
- CUDA 实算：`torch.cuda.is_available() == True`；在 GPU 上计算 `arange(1024).sum()` 得 `523776`。
- `ros2 doctor`：全部 5 项检查通过。报告提示两个仓库中存在更新版本（`point_cloud_transport`、`gz_cmake_vendor`），不影响当前基线验证，未擅自升级冻结环境。
- talker/listener：首次及 `wsl --terminate Ubuntu-24.04` 后均收到连续 `I heard: [Hello World: ...]`。
- colcon：创建并构建 `g0_smoke`，`1 package finished`；包前缀为 `/home/sdf/safedrive_ws/install/g0_smoke`。
- `dpkg --audit`：无输出，包数据库完整。

## 已知非阻塞项

- `systemd-binfmt.service` 在 WSL 中失败，使 systemd 显示 `degraded`；ROS、GPU、Python 和 colcon 均不依赖该服务，本任务验证未受影响。
- Mihomo/TUN Fake-IP 会把部分域名解析到 `198.18.0.0/15`，GitHub 路径曾返回 403；ROS apt 仓库安装和更新已成功。
- `sdf` 的密码保持锁定，避免自动设置凭据；需要管理操作时可由 Windows 侧使用 `wsl -d Ubuntu-24.04 -u root`。
