# G1-01：正式数据契约、运行身份与多速率 Profile

**状态**：COMPLETED  
**依赖**：G0-06  
**阶段**：G1 统一运行时与经典专家

## 目标与原因

把 G0 的最小 JSON 状态链路升级为后续规划、控制、VLA、World 和 Safety 共用的强类型契约，并解决随机 `episode_id`、进程内 tick ownership 和 50ms/20ms Profile 边界问题。若先做规划算法，接口债会扩散到全部阶段。

## 输入与范围

- 输入：G0 frame/clock 契约、`versions.lock`、`EXECUTION_ARCHITECTURE.md`。
- 定义 `sdf_interfaces`、坐标系、单位、枚举、schema version 和稳定运行身份。
- 建立 `throughput_20hz`、`control_50hz` 配置与合法性检查。
- 保留 G0 topic 兼容读取，不修改 G0 证据。

## 明确不做

不接入 ScenarioRunner，不生成 Actor，不实现地图、规划或控制算法。

## 允许修改路径

`safedrive_foundry/ros_ws/src/sdf_interfaces/**`、`safedrive_foundry/config/**`、`safedrive_foundry/runtime/**`、`tests/g1/**`、`docs/architecture/**`、本任务与 `PROGRESS.md`。

## 交付物与完成标准

- 消息/服务定义覆盖 Frame、Ego、Actor、Route、Trajectory、Policy、Safety、Control、RunEvent。
- Runtime 生成的身份可跨进程复用且不依赖 bridge UUID。
- 0.05s 与 0.02s 配置均通过 CARLA substep 合法性测试；错误组合快速失败。
- JSON 兼容适配器有 schema 校验；正式接口不再依赖自由字符串。

## 验证

运行接口构建、序列化 round-trip、坐标/单位属性测试、重复身份/跨 run 隔离测试和双 Profile 配置测试；保存构建及 schema 清单。

## 资源与断点

不启动训练；CARLA 仅做最小 Profile smoke。中断时记录最后成功 schema、构建命令和兼容性失败。

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-12 |
| 最后状态 | COMPLETED |
| 已完成 | 新增 `sdf_interfaces` ROS 2 强类型消息/服务包；新增独立 runtime 稳定 `RunIdentity`、G0 JSON schema 兼容适配器和双 Profile 合法性检查；记录坐标、单位、schema 与接口 hash 清单。G0 bridge、任务与冻结证据未修改。 |
| 修改文件 | `safedrive_foundry/ros_ws/src/sdf_interfaces/**`、`safedrive_foundry/runtime/**`、`safedrive_foundry/config/runtime_profiles.toml`、`tests/g1/**`、`docs/architecture/G1_01_SYSTEM_CONTRACTS.md`、`docs/architecture/G1_01_INTERFACE_MANIFEST.md`、本任务与 `PROGRESS.md`。 |
| 已运行验证 | 初次从 `ros_ws` 执行 `python3 -m unittest discover -s tests/g1 -v` 因相对路径错误快速失败（未执行任何测试）；随后在项目根执行 `python3 -m unittest discover -s tests/g1 -t . -v`：4/4 PASS；`source /opt/ros/jazzy/setup.bash && cd safedrive_foundry/ros_ws && colcon build --symlink-install --packages-select sdf_interfaces`：1 package finished；CDR `serialize_message`/`deserialize_message` 的 `ControlCommand` round-trip PASS；`git diff --check` PASS。构建有非阻塞 clock-skew 警告。 |
| 结果摘要 | `throughput_20hz`（0.05s，5×0.01s）与 `control_50hz`（0.02s，2×0.01s）均通过 CARLA 子步容量规则；错误子步组合在构造期失败。稳定 run_id 仅取决于显式 runtime 输入，改变 attempt 会隔离 run；正式接口不依赖 G0 的随机 bridge `episode_id`。 |
| 失败或阻塞 | 无。首次测试命令路径错误已在同一轮修正；ROS 构建的时钟偏移警告未影响生成或验证。 |
| 精确恢复步骤 | 如需审计，先校验 `docs/architecture/G1_01_INTERFACE_MANIFEST.md` 的 hash，再执行已记录的 unittest、colcon build 和 CDR round-trip 命令。 |
| 下一条建议命令 | 等待用户明确指定下一项任务；不得自动开始 G1-02。 |
