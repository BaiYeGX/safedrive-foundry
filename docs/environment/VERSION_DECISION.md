# G0 版本决策

决策日期：2026-07-11  
状态：冻结初稿；安装与烟雾测试由 G0-02/G0-03 验证，失败时按本文回退，不静默换版。

## 决策摘要

| 组件 | 首选组合 | 回退组合 |
|---|---|---|
| Windows Host | 当前 Windows 11 Pro 25H2 x64 | 同左，不降级主机系统 |
| WSL | WSL 2（安装时可用稳定版） | 同左 |
| Ubuntu | 24.04 LTS amd64 | 22.04 LTS amd64 |
| ROS 2 | Jazzy Jalisco Desktop，apt 二进制包 | Humble Hawksbill Desktop，apt 二进制包 |
| CARLA | 0.9.16 Windows packaged build | 0.9.15 Windows packaged build（现有 `E:\CARLA`） |
| ScenarioRunner | tag/branch `v0.9.16` / `0.9.16` | tag/branch `v0.9.15` / `0.9.15` |
| Python | CPython 3.12.x（Ubuntu 系统 Python） | CPython 3.10.x（Ubuntu 系统 Python） |
| PyTorch | 2.12.1 + `cu126` wheel | 2.7.1 + `cu126` wheel |
| CUDA Toolkit | 不作为 PyTorch wheel 运行前置；如需编译，固定 `12.6` toolkit-only | 同左 |
| NVIDIA 驱动 | 保留当前 Windows `591.86`，G0 不升级 | 同左 |

## 选择依据与兼容证据

### Windows、WSL 与 GPU

- Microsoft 的 WSL 安装要求是 Windows 10 2004/build 19041 以上或 Windows 11；当前 build 26200 满足版本门槛。[Microsoft WSL 安装文档](https://learn.microsoft.com/en-us/windows/wsl/install)
- NVIDIA 说明 WSL2 使用 Windows 主机驱动映射出的 `libcuda.so`，不得在 WSL 内安装 Linux NVIDIA 显示驱动；R495+ 支持 CUDA on WSL，当前 591.86 高于门槛。[NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- 当前 WSL/虚拟化尚未启用，因此这里只能判定“无已知版本硬冲突”，GPU 透传必须在 G0-02 实测。

### Ubuntu 与 ROS 2

- ROS 2 Jazzy 将 Ubuntu 24.04 amd64 列为 Tier 1，并支持到 2029-05；这比 Kilted（支持到 2026-11）更适合作为长周期研究基线。[REP-2000](https://www.ros.org/reps/rep-2000.html)
- 回退使用 Humble + Ubuntu 22.04；官方 Humble 二进制文档明确支持 Ubuntu Jammy 22.04 x86_64。[ROS 2 Humble Ubuntu 文档](https://docs.ros.org/en/humble/Installation/Alternatives/Ubuntu-Install-Binary.html)
- Ubuntu 24.04 的系统 Python 为 3.12，Ubuntu 22.04 的系统 Python 为 3.10，因此两个组合避免替换 ROS 依赖的系统解释器。

### CARLA、ScenarioRunner 与 Python

- CARLA 官方发布页将 0.9.16 标为最新 0.9.x packaged release，并提供 Windows/Ubuntu 资产。[CARLA releases](https://github.com/carla-simulator/carla/releases)
- CARLA 0.9.16 快速开始文档声明 packaged build 支持 Windows 10/11、Ubuntu 20.04/22.04，以及 Python 3.7–3.12；首选 Python 3.12 位于该范围内。[CARLA 0.9.16 quick start](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
- ScenarioRunner 官方仓库要求版本与 CARLA 匹配：0.9.16 对 0.9.16，0.9.15 对 0.9.15；0.9.16 还修复了 Python 3.12 移除 `distutils` 带来的问题。[ScenarioRunner](https://github.com/carla-simulator/scenario_runner)
- CARLA Server 继续运行在 Windows，ROS 2/Python 客户端运行在 WSL2；因此 CARLA 文档未列 Ubuntu 24.04 packaged server 不构成首选组合的硬冲突。跨系统 TCP 与 Python API 必须在 G0-04 实测。

### PyTorch 与 CUDA

- PyTorch 官方提供 2.12.1 的 CUDA 12.6 wheel；回退版本 2.7.1 也提供 `cu126` wheel。[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- `nvidia-smi` 报告当前驱动最高支持 CUDA 13.1；使用自带 CUDA 12.6 runtime 的 wheel 低于驱动能力，避免把系统 Toolkit 与 PyTorch runtime 绑定。
- 首选 2.12.1 较新，生态包兼容风险高于回退 2.7.1；训练任务正式开始前仍须用实际模型依赖锁文件验证。

## 已知风险与门禁

| 风险 | 当前判断 | 必须验证/处置 |
|---|---|---|
| WSL2/固件虚拟化未启用 | 外部阻塞 | G0-02 由用户批准管理员操作和重启；随后检查 `wsl -l -v`。 |
| CARLA 0.9.16 Windows server 与 WSL Python 3.12 client | 官方版本范围支持，但未在本机实测 | G0-03 独立 server/client 烟雾测试。失败则启用 0.9.15 回退，不混用 API。 |
| ROS 2 Jazzy 与 CARLA ROS 接口 | CARLA 支持 ROS 2，但具体 Jazzy 集成未形成官方逐版本矩阵 | G0-04 编译、topic、控制与时钟烟雾测试；必要时记录补丁 commit。 |
| 旧 CARLA 0.9.15 Python API 仅有 cp37 wheel | 与首选 Python 不兼容 | 回退时从 PyPI/官方 0.9.15 资产取得匹配 Python API，严禁复用现有 cp37 wheel。 |
| Windows 版本字段不一致 | 注册表遗留字段显示 Windows 10 | G0-02 用 `winver` GUI 或提权系统查询确认展示名称；build 号作为权威机器字段。 |
| 存储余量 | `E:` 约 102.67 GiB | 安装前估算 CARLA/WSL/data，设置低空间停止门禁，不删除用户文件。 |
| 网络隧道/防火墙 | 存在 Mihomo/Tailscale | G0-04 实测 2000/2001 与 ROS DDS；仅在证据表明冲突后请求用户调整。 |

## 回退触发条件

满足任一项即暂停首选组合并记录证据，再切换完整回退组合：

1. CARLA 0.9.16 Windows server 与 Python 3.12 client 无法完成官方最小 smoke test，且问题可复现、非路径错误。
2. CARLA ROS 接口在 Jazzy 上无法构建或存在阻断同步/控制的上游不兼容，且合理补丁无法在 G0-04 范围内解决。
3. 关键依赖尚不支持 Python 3.12，并且隔离环境不能解决。

不得只降一个组件造成 CARLA/ScenarioRunner 或 Ubuntu/ROS 2 混配。回退后仍须重新执行完整 G0 门禁。

## 尚需实测或用户授权

- 管理员权限：启用 WSL、Virtual Machine Platform、安装发行版；可能需要重启。
- 用户确认：BIOS/UEFI 虚拟化状态，以及是否允许保留首选与回退两套 CARLA 资产占用空间。
- 实测：WSL GPU、CARLA headless/窗口启动、Python API、ScenarioRunner、ROS 2 DDS、`/clock`、固定步长与 frame 对齐。
- 本任务不安装上述组件，也不宣称烟雾测试已通过。
