# G0-03：CARLA Server安装与独立验证

**状态**：COMPLETED  
**依赖**：G0-01、G0-02

## 目标

在Windows侧安装冻结版本CARLA，配置匹配的Python API，并在不依赖ROS的条件下完成独立启动、连接、tick和退出测试。

## 范围

- 安装CARLA及匹配Python API。
- 固化启动命令、安装路径、端口、Low/No Rendering配置和日志位置。
- 编写或使用最小client检查server/client版本、地图和若干tick。
- 不连接ROS，不实现业务规划。

## 完成标准

- CARLA可通过可记录命令启动和关闭。
- Client可连接、读取版本与地图并完成固定tick数。
- 重复测试无残留关键Actor或僵尸进程。
- 常见下载、端口、版本和启动故障有恢复说明。

## 验证方法

- 独立烟雾测试至少重复两次。
- 模拟server未启动或端口错误，确认错误快速可读。
- 保存日志和版本证据，更新断点与进度后停止。

## 断点记录

| 字段 | 内容 |
|---|---|
| 最后状态 | 2026-07-12 `COMPLETED` |
| 已完成 | 安装/核验 CARLA 0.9.16 与 cp312 API；固定启动参数；在 `carla0916` Conda 环境完成版本、地图、world、tick、错误端口和关闭清理验证 |
| 修改文件 | `scripts/g0/carla_server_smoke.py`、`docs/environment/CARLA_SERVER_BASELINE.md`、`docs/environment/evidence/g0-03/*`、本任务文件、`PROGRESS.md`、`versions.lock` |
| 已运行验证 | `carla0916` Python 3.12 环境四轮正式 smoke；每轮 10 个严格递增 frame，actor 数量 23→23 且无新增；直接 `get_world`/`wait_for_tick`；2099 错误端口快速失败；关闭后 CARLA 进程和 2000～2002 端口均为 0 |
| 阻塞 | 无。此前的 `UnicodeDecodeError` 已确认来自错误 Python 环境，不是 CARLA 0.9.16 或中文 Windows 不兼容 |
| 恢复步骤 | 若复测，激活 `carla0916` 后按基线文档启动 server 和 smoke；不得使用未安装 CARLA wheel 的 `(base)` 环境 |
| 下一建议命令 | `conda activate carla0916; python .\scripts\g0\carla_server_smoke.py` |
