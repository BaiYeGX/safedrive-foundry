# G1-02：Simulation Runtime 与场景生命周期

**状态**：COMPLETED  
**依赖**：G1-01

## 目标与原因

建立唯一 CARLA tick owner 和可恢复的 Scenario/Actor/Sensor/Control 生命周期，使后续算法只通过正式接口参与闭环。

## 范围

- 接入匹配 CARLA 0.9.16 的 ScenarioRunner 或受控 Python scenario adapter。
- Runtime 负责加载地图、spawn/destroy、传感器 barrier、control apply、tick、清理和 Run Registry。
- 设置 Traffic Manager 同步模式、seed、端口和 actor spawn 顺序。
- 建立命令超时、CARLA 断连、残留 Actor、重复 Runtime 和安全重跑路径。

## 不做与允许修改

不实现驾驶决策。允许修改 `safedrive_foundry/runtime/**`、`safedrive_foundry/scenario/**`、`safedrive_foundry/ros_ws/src/sdf_runtime/**`、`safedrive_foundry/config/runtime/**`、`tests/g1/**`、`docs/runtime/**`、本任务、`PROGRESS.md`，以及本轮已授权的 `AGENTS.md`、`START_TASK.md`。

## 完成标准

- 第二个 Runtime 无法取得 tick lease；所有业务节点没有直接 tick 路径。
- 相同配置可重复 spawn 相同 actor 清单，退出后无残留。
- Camera、Ego、Actor、Control 与 `/clock` 使用同一 FrameHeader。
- 中断后可从 scenario attempt 重跑，不把半成品登记为成功。

## 验证

运行空场景、单 Ego、NPC+Traffic Manager、传感器超时、错误端口、强制中断与重复 Runtime 测试；保存 actor/sensor 生命周期和清理证据。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-12 |
| 最后状态 | COMPLETED |
| 已完成 | 已实现受控 Python Scenario Runtime（不依赖 ScenarioRunner）：唯一 tick lease、两阶段 Registry 终态、callback admission/in-flight/queue/worker drain、sensor→NPC→Ego 清理、TM/WorldSettings 恢复、幂等线程安全 close、清理验证和 CARLA Transform hash 兼容。新增父子进程真实 supervisor；native 非零退出登记 `CRASHED` 并补偿清理。 |
| 修改文件 | `safedrive_foundry/runtime/scenario_runtime.py`、`safedrive_foundry/runtime/__init__.py`、`safedrive_foundry/config/runtime/scenario_runtime.toml`、`safedrive_foundry/config/runtime/carla_environment.sh`、`tests/g1/test_g1_02_scenario_runtime.py`、`tests/g1/run_g1_02_live.py`、`tests/g1/run_g1_02_live_child.py`、`tests/g1/run_g1_02_live_isolated.py`、`docs/runtime/G1_02_SCENARIO_RUNTIME.md`、`docs/runtime/evidence/g1-02/**`、本任务与 `PROGRESS.md`。 |
| 已运行验证 | `python3 -m unittest discover -s tests/g1 -t . -v`：18/18 PASS；`python3 -m py_compile`（Runtime、3 个 live harness）PASS；`git diff --check` PASS。动态环境下 CARLA RPC 2000、client/server `0.9.16`、map `Carla/Maps/Town10HD_Opt` 通过。真实 crash injection `live-isolated-crash-02`：child `-6`、faulthandler 保存线程栈、父级 CRASHED、补偿清理无残留。真实连续 10 次 `live-isolated-10-03`：10/10 return code 0、stderr 无内容、Registry 10/10 COMPLETED、无 SIGABRT、无残留；最终 actor count 23，world async/fixed delta null。 |
| 结果摘要 | Registry 顺序已修复为 `RUNNING → FINALIZING → COMPLETED`；cleanup/verification 失败只能是 `CLEANUP_FAILED`，child native 非零只能是 `CRASHED`。根因是 TM 在 autopilot NPC destroy 时仍发控制命令，触发 `trying to operate on a destroyed actor`；现已在 NPC destroy 前撤销 autopilot，并对 CARLA actor-list RPC 传播做 bounded verification。 |
| 本轮审计记录 | 实际 `carla.Client` 创建点包括 G0 `scripts/g0/carla_server_smoke.py`、bridge `sync_driver.py`/`bridge_node.py`、bridge `doctor.py`/`sync_contract.py`/`cli.py`，以及 G1 live harness 三个脚本；G1-02 Runtime 本身通过调用方 client 接入。硬编码/历史 host 出现在 G0 `safedrive_foundry/config/carla_ros.toml`、G0 文档/证据和 README；这些不属于本轮可改范围且 G0 必须保持冻结。独立动态 gateway 解析目前只有 `safedrive_foundry/config/runtime/carla_environment.sh`，它仅解析/导出环境变量，要求手工 source，不做 RPC、进程、版本、地图、WorldSettings、tick-owner 或状态持久化。当前代码未发现已实现的 `sdf sim status/preflight/ensure` 子命令；已验证 CARLA 启动路径记录在 `docs/environment/CARLA_SERVER_BASELINE.md`，但本轮未调用启动。拟修改文件为 `safedrive_foundry/runtime/carla_connection.py`（统一 resolver/handshake）、现有 CLI 入口、`tests/g1/**`、`docs/runtime/**`、本任务与 `PROGRESS.md`；`AGENTS.md`/`START_TASK.md` 的协议修改需先获范围决策。 |
| 失败或阻塞 | 无当前外部阻塞；post-completion hardening 代码、CLI、真实连接和 10/10 防回归已通过。CARLA 未运行时的真实自动启动未执行，因为当前已运行实例不能为本轮强行停止；已用注入式有限等待测试验证，不将该限制误报为外部阻塞。 |
| 精确恢复步骤 | 审计时先运行 `sdf sim preflight`，读取本轮 connection evidence 和原 `live-isolated-10-03` lifecycle evidence；不得开始 G1-03，除非用户另行指定。 |
| 下一条建议命令 | 等待用户检查；不要自动进入 G1-03。 |

### Post-completion hardening（本轮新增）

- 状态机：`RUNNING → FINALIZING → COMPLETED`；cleanup/verification 失败为 `CLEANUP_FAILED`，子进程非零退出为 `CRASHED`。
- 统一连接模块：`safedrive_foundry/runtime/carla_connection.py`；现有 CLI 路由：`safedrive_foundry/ros_ws/src/safedrive_carla_bridge/safedrive_carla_bridge/cli.py`；shell 仅为兼容 adapter。
- 新命令：`sdf sim status`、`sdf sim preflight`、`sdf sim ensure`。status/preflight 只读，不创建 Actor、不改 WorldSettings、不获取 tick lease；ensure 已运行实例不重复启动。
- 报告字段覆盖 host/source、previous/resolved host、TCP/RPC、client/server version、map、WorldSettings、process、tick owner、error/retry/recovery；`RETRYABLE_FAILURE`、`BLOCKED_EXTERNAL/NEEDS_USER_ACTION`、`FAILED_FINAL` 映射稳定。
- 真实验证：新 shell、未设置 `CARLA_HOST/PORT` 且未 source adapter 时 status/preflight/ensure READY；ensure 进程数 2→2；连接层接入后的 `live-isolated-10-hardening-01` 10/10 无 SIGABRT、无 stderr、无残留、Registry 全 COMPLETED。connection + lifecycle 全量测试 27/27 PASS。
- 限制：CARLA 未运行时的真实自动启动未执行，以免停止当前已验证实例；已用已验证启动配置的注入式 bounded timeout/NEEDS_USER_ACTION 测试覆盖，限制不标为外部阻塞。
- 本轮实际新增/修改：`safedrive_foundry/runtime/carla_connection.py`、`safedrive_foundry/config/runtime/carla_start.toml`、`safedrive_foundry/config/runtime/carla_environment.sh`、`safedrive_foundry/runtime/__init__.py`、`scripts/sdf.py`、现有 `.../safedrive_carla_bridge/cli.py`、`tests/g1/test_g1_02_connection.py`、三个 G1 live harness 的 resolver 接入、`docs/runtime/evidence/g1-02/connection-hardening.md`、`AGENTS.md`、`START_TASK.md`、`PROGRESS.md`；FUTURE_PROJECT_VISION.md、G0 文件和旧 evidence 未改写。
