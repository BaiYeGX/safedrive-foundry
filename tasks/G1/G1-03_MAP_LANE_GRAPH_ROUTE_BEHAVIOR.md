# G1-03：地图、Lane Graph、路由与行为层

**状态**：COMPLETED
**依赖**：G1-02

## 目标

从 OpenDRIVE/Waypoint 构建可验证地图语义、Lane Graph、多目标 A* Route Corridor 和可审计行为状态机。

## 范围

覆盖 road/lane/junction、前后继、换道、停止线、信号、限速、路线代价，以及巡航、跟车、停车、让行、换道、绕障和最小风险行为目标。分别标记 Oracle 与 Observable 所需输入。

## 不做与允许修改

不生成局部轨迹、不控制车辆。允许修改 `safedrive_foundry/classic_stack/map/**`、`safedrive_foundry/classic_stack/route/**`、`safedrive_foundry/classic_stack/behavior/**`、`safedrive_foundry/ros_ws/src/**` 中的对应 ROS adapter、`safedrive_foundry/config/**`、`tests/g1/**`、`docs/architecture/**` 和可视化证据。

本轮恢复额外授权（用户批准计划）：超 ~1500 行体量；W0 修改 `safedrive_foundry/runtime/carla_connection.py`、`doctor.py`、`tests/g1/test_g1_02_connection.py` 与连接相关配置/文档。

## 完成标准

- 至少三张 CARLA 地图可构图、缓存、查询并检测拓扑异常。
- Route Corridor 合法、确定且包含道路/换道/路口语义。
- 行为切换具备进入、保持、退出、超时、抑制和事件原因。
- 直行、转弯、换道、红灯、无保护让行路线通过固定 seed 测试。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。

## 验证与断点

运行图连通性/属性测试、路线重复性、不可达目标、行为时间线和地图可视化检查；记录地图 hash、route ID 与失败节点。允许修改路径内形成 Evidence，完成后停止。

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-13 |
| 最后状态 | COMPLETED |
| 已完成/验证 | W0：多候选 host（loopback 优先、198.18 代理殿后）、RPC 真值、`running_in_wsl` doctor、Windows 路径 `/mnt/e/CARLA_0.9.16`、ensure 启动。G1-03：官方 OpenDRIVE Town01/03/10HD 构图/缓存/路由；合成 fixture 固定 seed 行为与语义回归；ROS adapter colcon 通过。 |
| 地图真源 | `/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/{Town01,Town03,Town10HD}.xodr` → `fixtures/carla/` + evidence；SHA 见 manifest。Town01 API `to_opendrive` 曾与官方文件同 hash 后因连续 `load_world` 触发 Fatal Error，用户关闭 Server；验收改用官方 xodr，不再依赖切图导出。 |
| 已运行验证 | `PYTHONPATH=safedrive_foundry python3 -m unittest tests.g1.test_g1_03_map_route_behavior tests.g1.test_g1_02_connection -v`：20/20 PASS；`python3 -m compileall -q safedrive_foundry/runtime safedrive_foundry/classic_stack` PASS；`colcon build --symlink-install --packages-select safedrive_classic_stack` PASS；`sdf sim preflight` 在 Server 运行且镜像网络时 READY（host=127.0.0.1）。 |
| 阻塞 | 无（Server 已关闭时 preflight 预期 RETRYABLE，不标 WSL 缺失）。 |
| 未做 | 不生成局部轨迹、不控制车辆；不启动 G1-04。 |
| 修改文件 | `classic_stack/**`、`config/classic_stack/**`、`ros_ws/src/safedrive_classic_stack/**`、`runtime/carla_connection.py`、`doctor.py`、`carla_start.toml`、`tests/g1/test_g1_0{2,3}_*`、`docs/architecture/**`、`docs/project/CODEX_GROK_TROUBLESHOOTING.md`、本任务、`PROGRESS.md` |
| 下一条建议命令 | 用户明确指定后：`执行G1-04，读取START_TASK.md。` |
