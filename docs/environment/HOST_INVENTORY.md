# 主机环境盘点

盘点时间：2026-07-11 15:15（Asia/Singapore）  
任务：G0-01  
方法：仅执行只读查询；未安装、升级、重启或修改系统设置。

> **重要（2026-07-13 状态覆盖）**
> 下表中 “WSL 未安装 / Ubuntu 未安装 / ROS 未发现 / CARLA 仅 0.9.15” 是 **G0-01 当日** 快照，**不是当前事实**。
> 当前已验证：WSL2 Ubuntu 24.04、ROS 2 Jazzy、CARLA **0.9.16**（`E:\CARLA_0.9.16` / `/mnt/e/CARLA_0.9.16`）、工程在 `/mnt/e/autonomous driving`。
> 以 `versions.lock`、`PROGRESS.md`、`WSL_ROS2_BASELINE.md`、`CARLA_SERVER_BASELINE.md` 与 `sdf doctor` / `sdf sim preflight` 为准。
> **禁止**再把本文件的 “WSL 未安装” 复制进新任务断点。

## 主机与操作系统

| 字段 | 实测值 | 证据/备注 |
|---|---|---|
| 主机名 | `BYGX` | `ipconfig /all` |
| Windows | Windows 11 Pro 25H2，build `26200.8655`，x64 | 注册表 `DisplayVersion=25H2`、`CurrentBuild=26200`、`UBR=8655`；`platform.platform()` 判定 Windows 11。注册表遗留 `ProductName` 仍显示 Windows 10，保留此不一致供后续复核。 |
| CPU | 13th Gen Intel Core i5-13600KF，约 3.494 GHz | `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`；核心/线程数因 CIM 权限拒绝为 `UNKNOWN`。 |
| 物理内存 | 34,185,818,112 bytes（约 31.84 GiB） | `Microsoft.VisualBasic.Devices.ComputerInfo`；采样时可用约 15.51 GiB。 |
| GPU | NVIDIA GeForce RTX 4080 Desktop，16,376 MiB | `nvidia-smi` |
| NVIDIA 驱动 | `591.86`，WDDM | `nvidia-smi` |
| 驱动报告的最高 CUDA | `13.1` | `nvidia-smi`；这是驱动能力，不代表已安装 CUDA Toolkit。 |
| CUDA Toolkit | `UNKNOWN` / 未在 PATH 中发现 | 本任务未发现 `nvcc` 证据；不得用 `nvidia-smi` 的 CUDA 字段替代 Toolkit 版本。 |
| 固件虚拟化 | `UNKNOWN` | CIM 查询被拒绝；需在 G0-02 用任务管理器/BIOS 或提权命令确认。 |
| Hyper-V / VirtualMachinePlatform 功能状态 | `UNKNOWN` | 非提权 DISM 返回 740；WSL 命令提示未安装。 |

## 存储

| 卷 | 文件系统 | 总容量 | 可用空间 | 风险 |
|---|---|---:|---:|---|
| `C:` | NTFS | 929.80 GiB | 596.28 GiB | 充足 |
| `D:` | NTFS | 3.64 TiB | 61.61 GiB | 可用比例低，不建议作为新数据主盘 |
| `E:` | NTFS | 931.50 GiB | 102.67 GiB | 可容纳 CARLA，但训练/回放数据增长前需设置容量门禁 |

原始字节值来自 `[System.IO.DriveInfo]::GetDrives()`。现有项目和 CARLA 位于 `E:`，该卷剩余空间是近期主要容量风险。

## 网络

- 主物理网卡：Intel Ethernet Controller I225-V，DHCP，IPv4 联网正常。
- 存在 Mihomo/Wintun 与 Tailscale 虚拟隧道；它们可能影响 Windows–WSL 端口路由或防火墙诊断。
- CARLA 默认 TCP `2000/2001` 是否放行：`UNKNOWN`，应在 G0-04 连通测试中实测。
- 文档不保存完整 MAC、外网 IPv6 等非必要标识。

## 现有软件

| 软件 | 状态/版本 | 证据与影响 |
|---|---|---|
| WSL | 未安装或未启用 | `wsl --status`、`wsl --version`、`wsl -l -v` 均返回安装提示。 |
| Ubuntu on WSL | 未安装 | 无发行版列表。 |
| ROS 2 | 未发现 | `ros2` 不在 PATH。 |
| CARLA Server | 已存在 `E:\CARLA`，版本 `0.9.15` | 目录含 `CARLA_0.9.15.zip` 和 `CarlaUE4.exe`。本任务未启动服务器。 |
| CARLA Python API | 仅有 CPython 3.7 Windows wheel/egg | `carla-0.9.15-cp37-cp37m-win_amd64.whl`；不能直接用于当前 Python 3.12。 |
| ScenarioRunner | 未发现 | 后续必须与 CARLA 使用同版本分支/tag。 |
| Python | Anaconda CPython `3.12.7` | `E:\Anaconda\python.exe`；`py` launcher 无已注册解释器。 |
| PyTorch | `UNKNOWN` | `pip show torch` 查询超时，未获得可靠版本证据。 |
| Git | `2.51.0.windows.1` | `C:\Program Files\Git\cmd\git.exe`。 |
| Docker | 未安装或不在 PATH | `docker` 命令不存在；G0 首选路径不要求 Docker。 |

## 只读命令与结果摘要

执行过：`Get-ComputerInfo`、Windows 版本注册表查询、CPU 注册表查询、`nvidia-smi`、`wsl --status/--version/-l -v`、`git --version`、`python --version`、`py -0p`、`docker --version/info`、`Get-Command ros2/CarlaUE4.exe`、固定路径 CARLA 检索、`ipconfig /all`、`DriveInfo` 与 `ComputerInfo` 查询。

受限项：CIM、`Get-Volume`、`Get-NetAdapter` 被拒绝访问；DISM 功能查询要求管理员权限；这些失败没有改变系统状态，并已用非提权接口补齐可获取字段。

## G0-02 前置事项

1. `BLOCKED_EXTERNAL`：安装/启用 WSL2 和 Virtual Machine Platform 需要管理员权限，通常还需要重启。
2. 用户需确认 BIOS/UEFI 虚拟化已启用；当前为 `UNKNOWN`。
3. 安装后验证 WSL 内 `nvidia-smi`，且不要在 WSL 安装 Linux NVIDIA 显示驱动。
4. 优先把 WSL 工程与高频小文件放在 Linux ext4 虚拟磁盘；数据归档位置需考虑 `E:` 仅余约 102.67 GiB。
5. 现有 `E:\CARLA` 不删除、不覆盖；作为 0.9.15 回退资产保留。
