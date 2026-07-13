# G3-03：F0 可行性 + VLA-V0 上游接入与最小闭环

**状态**：PENDING  
**依赖**：G3-02  
**阶段角色**：必做（**全项目最关键工程门**）  
**一句话**：把 SimLingo/InternVL2-1B 真加载起来，canonicalize 成项目轨迹，再经 Safety 最小闭环；稳比炫优先。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/LOCAL_ASSETS.md` 与 `safedrive_foundry/config/vla/local_assets.toml`（路径 + **必须用 `/home/sdf/.venvs/sdf`**）；  
3. `docs/project/SDF_VLA_1B_DESIGN.md` 第 1、4、10 节（**含 F0**）与 G3 映射；  
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3～7、12 节；  
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1～5、7、8 节；  
6. `docs/project/CLAIMS.md` C1；  
7. G3-01 数据契约、G3-02 基线接口与断点。

## 项目成功口径（本任务）

- **F0 不过 → 禁止正式后训练与“已验证可运行”表述。**  
- 不上实车；仿真研究 lineage 即可。  
- 目标是**稳定跑通**，不是刷 Leaderboard。  
- 禁止直接换 3B/7B；SimLingo 不可用时才走干净 InternVL2-1B + 同接口（`SDF-VLA-1B-IVL`）。

## 目标

1. 完成 **F0**（见下）；  
2. 完成 **VLA-V0**：Observation adapter + 原生 path/speed + 确定性 TrajectoryCanonicalizer → `K=1/T=10/dt=0.25s/horizon=2.5s`；  
3. 无 Classic **当前帧候选** 的 `VLA + G2 Safety + MPC/PID` 最小闭环（固定简单路线即可）。

## 实现范围与边界

### 必做

- 冻结上游代码/权重/预处理/许可证/hash；  
- 独立环境加载；固定样本 path/speed 可复现；  
- BF16（及可选 8-bit）前向；显存与 P50/P95/P99、30 分钟稳定性；  
- 20～100 step resource smoke；save/restore；  
- ObservationBundle → CandidateSet → Validator；  
- 唯一 tick owner；禁止第二 `carla.Client` / 直接 `world.tick()`。

### 明确不做

- 不训练 K2、Router、复杂 OOD、LoRA（留给后续）；  
- 不把 2.5s 外推成 3s；  
- 不移植上游 Leaderboard/ScenarioRunner 作为 tick master。

## 完成标准与验证

### F0 清单（全部要有实测记录）

| 项 | 通过标准 |
|---|---|
| 加载 | checkpoint 成功，版本记录完整 |
| 固定样本 | path/speed 确定性复现（允许文档登记的数值容差） |
| 精度 | 至少一种部署精度前向成功 |
| 资源 | 显存峰值、时延分位、deadline miss 有表 |
| 稳定 | ≥30 min 或等价 step 无 OOM/崩溃（环境限制则记实际时长+原因） |
| 链路 | CandidateSet→Validator 无 shape/NaN 泄漏 |
| 仿真边界 | 无第二 tick master |

### V0 闭环

- 固定简单路线可重复完成一段 episode；  
- VLA 超时/失败 → Safety/MRM 链，**不卡死 50Hz 控制**；  
- P95 首门槛目标 ≤200ms（约 5Hz）；达不到则记 `COMPLETED_WITH_LIMITS` 并给实测值，**不得改合同装通过**。

### 建议验证命令

```text
sdf sim preflight          # 需要 live 时
python3 -m unittest discover -s tests/g3 -t . -v
# 任务规定的 offline smoke / live 脚本（实现后写入断点）
```

## 允许修改

`safedrive_foundry/driving_vla/model`、`adapter`、`config`、`runtime`（VLA 边界）、`tests/g3`、`docs`、本任务、`PROGRESS.md`。  
通过后停止，不自动开 G3-04。

## 断点记录

尚未开始。恢复时优先核验权重 hash、CUDA/WSL 可用性与上次 F0 失败项。
