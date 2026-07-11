# G0-01：环境盘点与版本冻结

**状态**：COMPLETED  
**依赖**：无

## 目标

以只读方式盘点主机与现有软件状态，随后依据官方兼容信息确定Windows、WSL Ubuntu、ROS 2、CARLA、ScenarioRunner、Python和PyTorch的首选及回退版本组合。

## 范围

- 查询Windows、WSL2、RTX 4080、驱动/CUDA、CPU、内存、磁盘、虚拟化和网络状态。
- 查询Git、Python、Docker、CARLA、ROS 2现有安装。
- 研究官方版本兼容信息，形成版本决策。
- 创建`docs/environment/HOST_INVENTORY.md`、`docs/environment/VERSION_DECISION.md`和`versions.lock`初稿。
- 禁止安装、升级、重启或修改系统设置。

## 完成标准

- 主机关键字段完整；无法获取的内容明确标记`UNKNOWN`。
- 每个冻结版本有选择依据、兼容证据和风险。
- 定义一个首选组合和一个回退组合。
- 明确需要实测确认、管理员权限或用户决策的事项。

## 验证方法

- 复查盘点命令与记录一致。
- 检查版本组合不存在已知硬冲突。
- 确认本任务未改变系统环境。
- 更新本任务断点和`PROGRESS.md`后停止。

## 断点记录

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-11 15:15–15:35（Asia/Singapore） |
| 最后状态 | COMPLETED |
| 已完成 | 只读盘点 Windows、CPU、内存、磁盘、网络、WSL、GPU、驱动、Git、Python、Docker、CARLA、ROS 2；研究 Microsoft、NVIDIA、ROS、CARLA、ScenarioRunner、PyTorch 官方兼容信息；冻结首选与回退组合。 |
| 修改文件 | `docs/environment/HOST_INVENTORY.md`、`docs/environment/VERSION_DECISION.md`、`versions.lock`、本任务文件、`PROGRESS.md` |
| 已运行验证 | 复核主机记录与盘点输出；检查 CARLA/ScenarioRunner 严格同版、Ubuntu/ROS 2 官方平台对应、Python 支持范围和 PyTorch cu126 官方 wheel；检查任务仅发生仓库文件写入，未执行安装、升级、重启或系统设置修改。 |
| 阻塞 | G0-01 无阻塞；G0-02 存在 `BLOCKED_EXTERNAL` 前置：启用/安装 WSL2 需要管理员权限且通常需要重启，固件虚拟化状态仍为 `UNKNOWN`。 |
| 未解决问题 | ROS 2 Jazzy 与 CARLA ROS 接口缺少逐版本官方矩阵，必须在 G0-04 实测；Windows ProductName 遗留字段不一致；`E:` 仅余约 102.67 GiB。 |
| 恢复步骤 | 无需恢复 G0-01；等待用户明确指定 G0-02 后，按其任务文件执行。 |
| 下一建议命令 | 不自动执行；用户批准 G0-02 后先复核管理员权限、重启窗口与 BIOS/UEFI 虚拟化。 |
