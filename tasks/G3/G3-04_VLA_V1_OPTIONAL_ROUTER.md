# G3-04：VLA-V1 最小增强与可选 VLA-V2 路由

**状态**：IMPLEMENTED / NOT_VERIFIED
**依赖**：G3-03
**阶段角色**：必做 V1；V2 optional
**一句话**：在 V0 稳定后加上低维 history 与 K=2，给 World 留出两条可排序候选。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；
2. `docs/project/SDF_VLA_1B_DESIGN.md` 第 4～6、10 节；
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 5、6、10、12 节；
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1～4、7、8 节；
5. `docs/project/CLAIMS.md` C1；
6. G3-03 的 F0 证据、V0 配置与断点；G3-01 split。

## 项目成功口径（本任务）

- **K=2 是为 World 服务的硬需求**（作品要 VLA+World，至少要有两条候选）。
- 训练以 heads-only / 小适配为主；24h 训不完就减数据，不换大模型。
- V2 FAST/REASON 可 `OPTIONAL_NOT_RUN`，不挡 G3-05。

## 目标

### VLA-V1（必做）

- 当前单图 + 当前 ego + 至多 4 个低维 history 时刻；
- 小型 Ego/History Adapter；
- `K=2` nominal / conservative（或等价双候选），`T=10/dt=0.25s/horizon=2.5s`；
- 简单候选概率与 top-1/top-2 margin。

### VLA-V2（optional）

- Runtime Guard（DEFER）与 FAST/REASON 路由；无净收益则保留 always FAST。

## 实现范围与边界

### 必做

- 锚定残差优先，避免候选塌缩；
- 输出完整进 CandidateSet；非法/过期拒绝；
- 训练不可读 Regression 标签。

### 明确不做

- 完整多帧图像历史堆叠、K4、3s horizon、复杂 ensemble 非完成条件；
- 不重新基础模型选型；不 3B/7B。

## 完成标准与验证

### 最小通过

- 固定 split 上 K2 形状/时间/坐标正确，可复现；
- top-1 与 oracle best-of-K **可计算**（为 G4 准备）；
- 无 OOM 的小训练或适配 smoke 通过。

### 诚实记录

- 覆盖/坍塌/margin/资源表；
- 语言路由若无收益 → 负结论 + always FAST。

### 建议验证命令

```text
python3 -m unittest discover -s tests/g3 -t . -v
```

## 允许修改

`safedrive_foundry/driving_vla/training`、`losses`、`uncertainty`、`evaluation`、`config`、`registry`、`tests/g3`、本任务、`PROGRESS.md`。

## 断点记录

**2026-07-14 实现落地（后被复审回退）**
- V1：`V1Policy` / residual K2 接口存在；V2：`OPTIONAL_NOT_RUN`

**2026-07-16 验收复审 → IMPLEMENTED / NOT_VERIFIED**
缺口：
1. `_apply_residual` 实际 `lateral_bias=0`，两候选 **空间轨迹相同**，仅改速度。
2. 不重积分位置 / 加速度 / 曲率 → 运动学自相矛盾。
3. 基于位置 ADE 的 oracle best-of-K **无法区分** 候选 → 不满足 G4 选择空间前置。
V2 仍为 `OPTIONAL_NOT_RUN`（不挡 G3-05，但 V1 本身未 VERIFIED）。
