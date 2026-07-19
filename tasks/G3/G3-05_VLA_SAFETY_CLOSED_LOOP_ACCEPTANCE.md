# G3-05：VLA+Safety 闭环与阶段验收

**状态**：REVALIDATION_REQUIRED（pure VLA+MPC 已 `MEASURED_WITH_LIMITS`；正式 Safety 关闭未通过）
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

### 2026-07-19 Pure VLA + MPC 发布检查点

- 统一说明：`docs/architecture/G3_VLA_MPC_RELEASE_GUIDE.md`。
- 已实现并实测真实 SimLingo 相机输入、`pred_route`/`pred_speed_wps`、纯 VLA
  PathManager、constrained MPC、Mercedes 几何、DX12 D3D 解法和 CARLA 自动冷启动。
- 已有 Town03 300s stable runner `DEMO_PASS` 等 evidence；Town15 最新 switch/stale
  修复只有旧 trace 离线反事实，仍需 post-release live。
- 当前结论只能写 `G3 Pure VLA + Constrained MPC — MEASURED_WITH_LIMITS`。
- 本任务的阶段关闭条件仍是 VLA+Safety 多种子/故障链新证据与
  `assert_g3_close`；历史无效 evidence 不恢复效力。
- 下一步：先固定配置复验 Town03/Town15，再恢复本任务的 Safety 正式验收；未完成前
  不得宣称 G3 `VERIFIED`。

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

### 2026-07-19 Town04 A/B 复审与 D3D 加固断点

- A 在约60s sim time 后的 Windows CrashContext 明确为
  `GPUCrash / 0x887A0020 INTERNAL_ERROR / bIsOOM=0`；后续 `world.tick` 和 actor destroy
  timeout 是 CARLA Server 死亡后的结果。
- B 运行到约120s sim time，但后半段因导航目标用 ego-x 单调性判断
  而误停车；因此 B 不能证明 D3D 已解决。
- 弯道导航目标改为 route arc-length/剩余路程判断；相机默认从固定10Hz
  改为与0.75s VLA周期对齐，默认640×320，CARLA冷启动默认640×360，
  debug draw默认关闭，并移除runner中重复CUDA synchronize。
- 新增 `camera-only` / `model-resident` / `full` 三档D3D隔离模式；启动前写
  `run_config.json`，每10s写checkpoint与partial VLA events，异常记录最后操作。
- 新增末20s运动比例门禁，避免停车长测被误判为通过。
- 离线验证：
  `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`，
  70/70 PASS；`py_compile` 和定向 `git diff --check` PASS。
- 精确恢复：Windows/CARLA可用后，按
  `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md` 先跑三档180s隔离，再做
  20s/60s/300s full验收。当前仍为 `LIVE_REVALIDATION_REQUIRED`。

### 2026-07-19 C/D/E live 后续修复断点

- C `camera-only` 与 D `model-resident` 均180s `DIAGNOSTIC_PASS`；旧 E `full`
  在约177s sim time 后崩溃，CrashContext 为
  `Assert / D3D11Query.cpp:356 / DXGI_ERROR_INVALID_CALL`，`bIsOOM=0`。
- 复核发现 C/D 均保持车辆静止，而 E 同时加入 forward 和车辆运动；
  因此旧三档证据存在混杂，不足以将 CUDA forward 判为唯一原因。
- 新增 `forward-only`：执行真实 SimLingo forward 和路径条件化，但车辆全程制动；
  默认 post-forward idle guard 从10ms增至50ms。下轮先跑 E0 forward-only，
  通过后再跑 E1 full。
- CrashContext 解析改为 `.//<field>` 嵌套查找，并将 D3D Assert 分类为
  `CARLA_D3D_CRASH`；同时记录模型驻留 VRAM，partial event 新增 raw 20点路径。
- 旧 E 另有确定的路径死锁：69/227 接受、最长131次连续拒绝、
  `path_age~94s`、末20s静止。PathManager 现使用曲率 90% 稳健分位+绝对
  硬上限，避免单个 PCHIP 插值尖峰误拒；仅在旧路径≥2.5s且车速≤0.5m/s
  时从当前纯 VLA 路径重新锚定，正常行驶中的大跳变仍拒绝。
- 离线验证：
  `/home/sdf/.venvs/sdf/bin/python -m unittest discover -s tests/g3 -t . -v`，
  74/74 PASS。未跑修复后 live，仍为 `LIVE_REVALIDATION_REQUIRED`。

### 2026-07-19 D3D 子事故关闭与驾驶问题解耦

- DX11 E0 `forward-only` 在车辆全程制动时仍于约 59–61s、第约 79 次真实
  forward 后令 CARLA Server 假活，证明车辆运动、MPC 和路径管理不是必要触发条件；
  显存峰值约 2.2GB，不是 OOM。
- 换用 DX12 后，同一 E0 隔离档分别完成 180s/240 次和 300s/400 次 forward；
  随后的 E1 full 也完成 180s/240 次 forward，均无 tick timeout、D3D crash 或新
  CrashContext。因此本机 D3D 子事故记为 `MEASURED_RESOLVED_ON_DX12`，DX11
  保留为已知失败基线。
- 后续 G3 live 必须显式传 `--rhi dx12` 并核对 `run_config.json` 的 requested/effective
  RHI；当前 runner CLI 默认仍为 DX11，不能只依赖 toml 当前值。
- E1 的结果仍是 `DEMO_FAIL`：约319m后弯道碰撞并卡在标志牌附近，另有
  `collisions=272`、`cte_rms~0.62m`、转向饱和约19%、末20s不运动。这些属于驾驶
  质量问题，与 D3D 事故分轨处理；G3 继续保持 `NOT_VERIFIED`。
- 详细结论、证据路径和回归流程见
  `docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md`。
