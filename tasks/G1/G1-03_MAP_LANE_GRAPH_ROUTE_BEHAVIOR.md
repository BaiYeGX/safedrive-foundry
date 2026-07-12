# G1-03：地图、Lane Graph、路由与行为层

**状态**：PENDING  
**依赖**：G1-02

## 目标

从 OpenDRIVE/Waypoint 构建可验证地图语义、Lane Graph、多目标 A* Route Corridor 和可审计行为状态机。

## 范围

覆盖 road/lane/junction、前后继、换道、停止线、信号、限速、路线代价，以及巡航、跟车、停车、让行、换道、绕障和最小风险行为目标。分别标记 Oracle 与 Observable 所需输入。

## 不做与允许修改

不生成局部轨迹、不控制车辆。允许修改 `safedrive_foundry/classic_stack/map/**`、`safedrive_foundry/classic_stack/route/**`、`safedrive_foundry/classic_stack/behavior/**`、`safedrive_foundry/ros_ws/src/**` 中的对应 ROS adapter、`safedrive_foundry/config/**`、`tests/g1/**`、`docs/architecture/**` 和可视化证据。

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
| 最后状态 | PENDING |
| 已完成/验证 | 无 |
| 阻塞 | 无 |
| 恢复步骤 | 从最近地图 hash、route ID 和失败 case 继续 |
