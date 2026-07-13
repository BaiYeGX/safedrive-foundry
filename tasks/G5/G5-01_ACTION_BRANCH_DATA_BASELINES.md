# G5-01：动作分支数据与简单世界基线（WE0 标注）

**状态**：PENDING  
**依赖**：G4-05  
**阶段角色**：必做（本项目 World 必上）  
**一句话**：为 World-V0 准备 K2 动作分支数据，并钉住 CV/CTRV/Reward 等简单基线。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_WORLD_MODEL_DESIGN.md` 第 1～3 节、WE0；  
3. `docs/project/EXECUTION_ARCHITECTURE.md` 身份/Oracle/Observable/Registry；  
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3、5、8、13、14 节；  
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1～4、6～8 节；  
6. G4-05 标签与场景/反事实产物；G3/G2 候选与安全接口。

## 项目成功口径（本任务）

- **无论 G4-05 是 ENTER / WEAK / NO_SELECTION，本任务都执行。**  
- WE0 结果写入数据卡，只影响 C2 表述强度，不取消实现。  
- 简单基线必须存在，防止 World “必赢空气”。

## 目标

- 冻结 ActionBranchDatasetV0（K2、可比起点）；  
- 实现 persistence / CV / CTRV / 规则或 IDM / Reward MLP 等**至少两类**简单基线；  
- 明确 train 标签不含 runtime Oracle 泄漏。

## 实现范围与边界

### 必做

- 分支轨迹 chunk、actor 集合上限 N≤8 的数据约定；  
- hash、split、resource smoke。

### 明确不做

- 不训完整 World-V0（G5-02）；不在线控车分叉。

## 完成标准与验证

### 最小通过

- 数据集可重建；基线可跑；  
- action 置换后标签/特征关系可测；  
- G4-05 标签原样登记到 manifest。

### 建议验证命令

```text
python3 -m unittest discover -s tests/g5 -t . -v
```

## 允许修改

`world_model/data`、`baselines`、`config`、`tests/g5`、`registry`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
