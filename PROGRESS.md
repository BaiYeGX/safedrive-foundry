# 项目当前进度

> 本文件只记录动态执行状态。任务编号、名称和依赖以 `ROADMAP.md` 为唯一来源；任务范围、验收和断点以对应任务文件为准。

## 当前指针

| 字段 | 当前值 |
|---|---|
| 当前阶段 | G1：经典专家、风险自适应规划与鲁棒实时控制（进行中） |
| 当前任务 | G1-02：Simulation Runtime 与场景生命周期 |
| 当前状态 | `COMPLETED` |
| 最近完成 | G1-02：Simulation Runtime 与场景生命周期 |
| 推荐下一任务 | `G1-03 MAP_LANE_GRAPH_ROUTE_BEHAVIOR`（仅推荐，不自动开始） |
| 最近更新 | 2026-07-12 |

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| G0 | `COMPLETED / FROZEN` | 6项完成，证据包已冻结 |
| G1 | `IN_PROGRESS` | 9项，G1-01、G1-02已完成 |
| G2 | `PENDING` | 6项 |
| G3 | `PENDING` | 7项 |
| G4 | `PENDING` | 6项 |
| G5 | `PENDING` | 6项 |
| G6 | `PENDING` | 6项 |
| G7 | `PENDING` | 5项 |
| G8 | `PENDING` | 6项 |

## 当前阻塞与决策

- G1-02 原完成证据和本轮 connection hardening evidence 均已验证；真实 10/10 lifecycle 防回归、source-free status/preflight/ensure、27/27 测试和协议同步全部通过。
- G0保持只读；G1-01、G1-02已按实际仓库完成正式消息、目录、连接和生命周期校准，不回写G0。
- G0的0.05s smoke不是全链路20ms结论；G1-06/G1-08必须测量50Hz控制的P50/P95/P99和deadline miss。
- 经典算法优化只保留RACE-Plan、RACE-Control与G2 RATO-SCP三项主贡献；未证明净收益的改进不进入默认配置。
- 后续任务51项；连同冻结G0共57项。任务结构只在`ROADMAP.md`维护。

## 最近更新

| 日期 | 变更 |
|---|---|
| 2026-07-12 | 完成任务目录与管理文档全量一致性修复：ROADMAP/活动任务 57/57、阶段数量与依赖检查通过；非 G0 任务结构和路径归一化，新增 `scripts/maintenance/task_catalog_check.py`、维护测试与 `docs/project/TASK_CATALOG_AUDIT.md`。未启动 G1-03；G0 任务、代码、配置和历史证据保持冻结。 |
| 2026-07-12 | 完成统一 `safedrive_foundry/runtime/carla_connection.py`、现有 `sdf sim status/preflight/ensure` 路由、动态 endpoint 与错误/任务状态映射；新 shell 无环境变量、无 source 真实验证通过，ensure 前后 CARLA 进程数 2→2，连接层接入后 NPC+camera 10/10 仍通过。管理协议已按真实实现同步，G1-02 恢复 COMPLETED。 |
| 2026-07-12 | 获得明确范围扩展授权，恢复 G1-02 为 `IN_PROGRESS`；先实现统一 CARLA connection 与现有 sdf sim 路由，真实验证通过后再更新 AGENTS/START_TASK/PROGRESS 协议。原 lifecycle 10/10 evidence 保持不变。 |
| 2026-07-12 | 启动 G1-02 post-completion hardening 审计；发现请求修改 `AGENTS.md`/`START_TASK.md` 超出当时任务声明的允许范围，按协议标记 `DECISION_REQUIRED` 并停止。原 G1-02 10/10 evidence 保持不变；现行路径以 `tasks/G1/G1-02_SCENARIO_RUNTIME_AND_ACTOR_LIFECYCLE.md` 为准。 |
| 2026-07-12 | 完成 G1-02：修复 cleanup 前错误登记 COMPLETED、补充 FINALIZING/CLEANUP_FAILED/CRASHED、线程安全幂等关闭、sensor callback drain、TM/NPC 销毁竞态；新增父子进程 supervisor。真实 crash injection 与 10 次 NPC+camera live acceptance 全部通过，证据见 `docs/runtime/evidence/g1-02/live-isolated-10-03/`。 |
| 2026-07-12 | G1-02 live 验收部分通过：动态 CARLA 0.9.16 连接、唯一 lease、空场景、单 Ego control/tick 与清理已实测；修复真实 Transform config hash。NPC+camera cleanup 连续三次 native SIGABRT，任务标记 BLOCKED，现场保存于 `docs/runtime/evidence/g1-02/`。 |
| 2026-07-12 | 恢复 G1-02 环境：动态解析 WSL 默认路由的 Windows host，TCP/RPC 2000 和 CARLA client/server `0.9.16` handshake 通过（Town10HD_Opt）。新增可 source 的运行环境入口，后续任务不再复制历史 gateway。 |
| 2026-07-12 | G1-02 实现受控 Python Scenario Runtime、跨进程 tick lease、确定性 Actor/Sensor 生命周期、Traffic Manager 同步/seed、传感器 barrier 和 SQLite Run Registry；本地生命周期回归 10/10 PASS。真实 CARLA 诊断 `172.30.80.1:2000` 超时，任务暂停等待模拟器可用。 |
| 2026-07-12 | 完成 G1-01：新增 `sdf_interfaces` 强类型 ROS 接口、稳定 runtime 身份、G0 JSON 兼容适配器和 20Hz/50Hz Profile；Python 测试4/4、接口构建和 CDR round-trip 通过，接口 hash 清单见 `docs/architecture/G1_01_INTERFACE_MANIFEST.md`。 |
| 2026-07-12 | 新增固定启动入口`START_TASK.md`；用户以后只需给出任务ID并要求读取该文件，Codex自动定位任务、检查依赖、按需读取、验证、更新断点与进度后停止。 |
| 2026-07-12 | 合并管理文档：执行规则与任务标准统一到`AGENTS.md`；任务列表只留在`ROADMAP.md`；本文件精简为动态指针；主张改为`docs/project/CLAIMS.md`；专项算法决策移入`docs/project/decisions/`。 |
| 2026-07-12 | 完成经典算法优化选型，新增G1-07 RACE-Plan和G1-08 RACE-Control，经典专家验收顺延为G1-09；未启动G1。 |
| 2026-07-12 | G0-01～G0-06全部完成；验收与SHA-256 manifest位于`docs/environment/evidence/g0-06/`。 |

## 恢复提示

新对话固定读取本文件和当前任务文件。若当前任务不是“无”，按任务末尾断点验证外部状态后继续；完成后更新本文件并停止，不自动开始下一任务。
