# G3-05：VLA+Safety 闭环与阶段验收

**状态**：ACCEPTANCE_FAILED
**依赖**：G3-01～G3-04
**阶段角色**：必做（G3 关闭）
**一句话**：证明 VLA 真能进 Safety→控制闭环；稳与可降级优先，分数其次。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；
2. `docs/project/CLAIMS.md` C1（主）、C3（Safety 护栏交叉）；
3. `docs/project/SDF_VLA_WORLD_SYSTEM_ARCHITECTURE.md` 运行模式相关节；
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 1～7、10、12 节；
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、5、7、8 节；
6. G2-05 与 G3-01～G3-04 最终产物、失败 Evidence、断点。

## 项目成功口径（本任务）

- **C1 最小成功**：单机上 VLA 产出完整轨迹并完成无 Classic 当前帧候选的可重复闭环（可 `WITH_LIMITS`）。
- Hybrid 作为工程稳健对照，**不替代** VLA_SAFETY 主证明。
- 不要求全面超过 Classic；超过了是加分，没超过诚实写。

## 目标

在 Runtime + G2 Safety + MPC 上对比并固化：

| 配置 | 目的 |
|---|---|
| Classic-Observable | 系统下界/对照 |
| 非语言基线 | 隔离轨迹头 |
| Raw VLA | 无/弱 Safety 行为（慎用，短测） |
| VLA+Validator | 硬预筛 |
| VLA+完整 Safety | 主配置 |
| HYBRID | 工程稳健 |

记录拒绝/修复、超时、接管、资源与失败类型。

## 实现范围与边界

### 必做

- 固定场景 + 多种子（至少小集合）；
- 控制 20ms 环不等待 GPU；
- unavailable/timeout/stale → 既定降级；
- 模型卡 + Evidence 目录。

### 明确不做

- 不开始 G4 大规模搜索；不训 World；
- 不把 live 未跑写成 VERIFIED。

## 完成标准与验证

### 最小通过

- `VLA_SAFETY` 固定简单路线可重复跑通；
- 故障注入至少覆盖 timeout/stale/NaN 之一并正确降级；
- Evidence hash/link 自检。

### 诚实记录

- 碰撞/完成/舒适/时延表，负结果保留；
- G2 live 限制若仍在，写入本阶段限制清单。

### 建议验证命令

```text
sdf sim preflight
python3 -m unittest discover -s tests/g3 -t . -v
# live 验收脚本（实现后登记）
```

## 允许修改

G3 小型缺陷修复、`runtime` adapter、`validation/g3`、`registry`、`artifacts/g3`、`tests/g3`、`reports`、本任务、`PROGRESS.md`。
通过后停止，不自动开 G4。

## 断点记录

**2026-07-14 曾标 COMPLETED_WITH_LIMITS — 2026-07-16 复审作废**

### 旧 Live 目录（**无效验收证据**，勿再引用为通过）
- `docs/architecture/evidence/g3-05/neural_live/latest_live_summary.json`
  - seed 11/13：`decision_tail` 末 15 帧 **全部 `EMERGENCY`**，却 distance≈140–148m
  - 根因：`run_g3_vla_safety_live.py` 强制油门 + 无批准轨迹仍驱动 + 开环转向
- Fault timeout：`sources_seen=[]` 却 distance≈138m；顶层摘要硬编码 `["vla_fast"]`
- `assert_g3_close.py` 仅检查 `all_ok`/步数类字段 → **门禁不足，PASS 作废**

### 2026-07-16 验收复审 → ACCEPTANCE_FAILED
阻塞（须修复后重跑 live，生成新 Evidence）：
1. 删除 Safety/MPC 后的强制油门与「无 accepted 仍给 thr」旁路。
2. 仅执行 `executed_trajectory_id` 对应轨迹，或 Safety 明确输出的 MRM/Emergency 控制；禁止 first-available 回退绕过裁决。
3. `COMPLETED` 不得仅 `steps>=80`；须约束决策谱系（禁止长距离全程 EMERGENCY 却算成功）。
4. `assert_g3_close` 必须校验：Safety 决策分布、执行谱系、`sources_seen` 与 per-result 一致、证据哈希、VLA_SAFETY 下无 Classic 当前帧且运动可归因于批准路径。
5. timeout 故障：须证明降级路径正确且运动来源可审计，禁止硬编码 `sources_seen`。

G3 阶段在本任务未通过前保持 **NOT_VERIFIED**；**禁止**启动 G4。

### 2026-07-18 Pure VLA + MPC 稳定化断点

- 用户本轮只要求先证明不带 Safety 的 `VLA spatial path + MPC`，不改变本任务最终
  `VLA_SAFETY` 阶段关闭标准。
- 已新增原生空间路径管理、G3 专用 2 秒约束 MPC、相机标定修复和稳定 live 入口；
  旧 `run_g3_vla_mpc_minimal.py` 仅作为兼容入口。
- 离线验证：
  `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`
  共 53 项，全部通过。
- live 首次 preflight：`RETRYABLE_FAILURE / SERVER_NOT_RUNNING`；按协议执行的唯一一次
  `sdf sim ensure` 返回 `BLOCKED_EXTERNAL / NEEDS_USER_ACTION`，错误包含
  `WSL UtilBindVsockAnyPort: socket failed 1`。
- 精确恢复：用户完成 Windows 驱动更新与重启；执行一次 preflight；仅 READY 后按
  `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md` 跑 20 秒与 60 秒 Town04 验收。
- 当前状态保持 `LIVE_NOT_VERIFIED`，不得据此启动 G4。

### 2026-07-19 Speed / endurance / Large Map 修复断点

- `--v-ref` 已改为纯最大速度上限，删除 `max(vla_speed, 0.85 * cap)` 强制速度下限；
  VLA speed head 经可配置 gain 校准，减速立即生效，仅提速限斜率。
- MPC 新增长度可停车、路径陈旧和稳健曲率限速；拒绝的 VLA 几何仍可降低速度，
  但不能以新路径时间戳刷新旧几何。
- 固定 160m 导航路线改为 600m 滚动续接；路线耗尽不再伪造正前方目标。
- 5 分钟 evidence 新增 60s checkpoint、实际/目标速度、VRAM、碰撞、越线、离路和
  路线进度。长测不再使用对环形/弯曲路线无意义的 distance/displacement 门槛。
- Town11/12/13 增加 hero actor、spectator 预流式和 tile/actor active distance；
  默认随机地图排除 Large Maps。Town13 的 UE 0x8 crash 仍须 live 复验。
- 离线验证：
  `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`
  共 64 项，全部通过。
- 精确恢复：Windows 驱动更新并重启后，按
  `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md` 依次运行 20s、60s、300s。
  当前仍是 `LIVE_REVALIDATION_REQUIRED`，不是 G3 阶段通过。
