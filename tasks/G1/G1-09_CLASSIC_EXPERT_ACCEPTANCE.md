# G1-09：Classic Expert 双轨集成、算法消融与阶段验收

**状态**：`COMPLETED_WITH_LIMITS`
**依赖**：G1-01～G1-08

## 目标

集成 Runtime、Route、Behavior、Frenet/ST、Hybrid A*、RACE-Plan、基础/自适应 MPC 和 PID，形成 Classic-Oracle 教师与 Classic-Observable 公平基线；冻结专家数据质量门禁，并判断哪些算法优化有真实净收益。

## 范围与允许修改

只做 G1 集成、小型缺陷修复、场景套件、算法配对消融、指标、Run/Data Registry 和 Evidence Bundle；实质缺陷退回对应任务。允许修改 `safedrive_foundry/runtime/**`、`safedrive_foundry/classic_stack/**`、`safedrive_foundry/ros_ws/src/**` 中的 G1 adapter、`safedrive_foundry/validation/g1/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/datasets/g1/**`、文档、本任务和 `PROGRESS.md`。

## 完成标准

- 城市路线、红灯、跟车、切入、换道、绕障、路口让行和复杂机动闭环可重复运行。
- 公布 Basic Classic、RACE-Plan、RACE-Control 和 Full RACE 的配对结果，分别报告安全、进度、舒适、规划/控制尾延迟、资源和失败类型。
- Oracle/Observable 结果分开；正专家数据必须无碰撞/严重违规、动力学可行、舒适合格且无连续超时，失败进入失败库。
- 只有跨 seed/场景产生稳定净收益的改进进入默认专家；局部有效或负贡献保留适用边界。
- 两个运行 Profile、重复性、资源和阶段 Evidence hash/link/schema 检查通过；形成 G2 所需 CandidateTrajectory、RiskField、ActorObservation 和控制基线。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。

## 验证方法与断点

从干净终端执行固定 G1 suite、多 seed 配对消融和算法退化测试；中断记录缺失矩阵单元、run_id、默认配置选择依据和负结果，不自动开始 G2。

---

## 验收结果与证据映射（追加，不替代上文标准）

权威 offline：`docs/architecture/evidence/g1-09/repair-20260713/summary.json`（`SDF_WRITE_G1_09_EVIDENCE=1` 生成；单测默认写 tempfile，不污染权威目录）。
权威 live（有限集成）：`docs/architecture/evidence/g1-live/latest_success.json`（schema **v4**，`run_id=g1-live-full-11-1783946091`）。
**禁止**将旁路 603m waypoint 或旧根目录 `evidence/g1-09/summary.json`（若存在历史 pointer）当作本轮无限制关闭证据。

| # | 完成标准条款 | 结果 | 证据映射 / 说明 |
|---|---|---|---|
| 1a | 城市路线闭环可重复 | **部分** | Offline Frenet/Hybrid 场景套件可重复；live 短城市场景 ~182m Town10HD **arrived** |
| 1b | 红灯 / 跟车 / 切入 / 换道 / 绕障 / 路口让行 | **未 VERIFIED（限制）** | 离线有对应场景族；**权威 live 为 `no_traffic`、`junctions=0`**，不覆盖交通参与者与信号 |
| 1c | 复杂机动（含 U-turn） | **部分** | Offline Hybrid/RS 四机动 + partial 审计；live **dense U-turn**（`hybrid_in_loop=false`），**不**作 Hybrid/RS live 证明 |
| 2 | Basic / RACE-Plan / RACE-Control / Full 配对 | **通过（offline）** | g1-07/08/09 repair：systems 表 + admission；报告 success、candidates、p99、miss、promote |
| 3 | Oracle/Observable 分轨 + 专家门禁 | **通过（offline gate）** | g1-09：`oracle_plan`/`observable_plan` 分表；gate：plan positives≥5、CTE&lt;5、非全 brake |
| 4 | 仅稳定净收益进默认 | **通过（负结论保留）** | plan `promote_full=false` recommend p1；不伪造 full 净收益 |
| 5a | Profile / 重复性 / 资源 | **部分** | control **50Hz offline plant** 有 P50/P95/P99；**无** live 50Hz 正式 VERIFIED 面板；**无**第二 Profile 全套 live |
| 5b | Evidence hash/link/schema | **通过（repair 轨）** | g1-05～09 `repair-20260713/manifest.json`；g1-04 已补 manifest；live schema `safedrive.g1.live_stack.v4` |
| 5c | G2 导出接口登记 | **通过（路径登记）** | g1-09 evidence `g2_exports`：CandidateTrajectory / RiskField / ActorObservation / ControlBaseline |

### 阶段限制清单（写入 `COMPLETED_WITH_LIMITS` 的硬边界）

1. Solver **不是** OSQP/SQP-RTI（`constrained_gradient_ltv_bicycle`）。
2. **无** CARLA live 50Hz P50/P95/P99 / deadline miss 正式 VERIFIED。
3. RACE admission **仅 offline**；**无** CARLA A/B 收益。
4. 权威 live **不是** Hybrid/RS 进环证明（dense uturn；`hybrid_calls=0`）。
5. 权威 live **无** 交通/红灯/路口让行覆盖（`no_traffic`，`junctions=0`）。
6. 长 600m full **零碰撞非 VERIFIED**。
7. 工作区历史 `docs/environment/*` 等 diff **不纳入** 关闭条件。

### 验证与断点

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-13 |
| 最后状态 | `COMPLETED_WITH_LIMITS` |
| 离线验证 | `PYTHONPATH=safedrive_foundry python3 -m unittest discover -s tests/g1 -t . -v` → **78 OK** |
| 离线权威证据 | `docs/architecture/evidence/g1-09/repair-20260713/`（`summary.json` + `manifest.json`） |
| Live 权威证据 | `docs/architecture/evidence/g1-live/latest_success.json`（v4，arrived，dense uturn，RaceControl+identify） |
| 旧状态废止 | 禁止使用任务内历史 **`COMPLETED`** 与仅指向根 `evidence/g1-09/summary.json` 的旧断点作为关闭依据 |
| 停止 | **G1 阶段正式结束**（`COMPLETED_WITH_LIMITS`）；**不**自动开始 G2；下一阶段仅用户指令 `执行G2-01，读取START_TASK.md` |
