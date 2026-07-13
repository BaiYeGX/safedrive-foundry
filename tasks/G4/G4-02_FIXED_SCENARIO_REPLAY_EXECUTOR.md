# G4-02：G4A 固定困难场景、确定性 Replay 与可恢复执行器

**状态**：PENDING  
**依赖**：G4-01  
**阶段角色**：必做（G4A）  
**一句话**：冻结 20～40 场景 + seed + initial-state hash，保证重跑不漂移、可断点恢复。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_WORLD_MODEL_DESIGN.md` 第 3 节（固定困难包前置）；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、6～8 节；  
5. `docs/project/CLAIMS.md` C4；  
6. G4-01 schema/门禁/suite manifest 与断点。

## 项目成功口径（本任务）

- **可复现 > 场景数量**。宁少而稳。  
- Random/LHS 主动搜索属 G4B，**不是**完成条件。

## 目标

- 冻结 20～40 困难场景 manifest；  
- 固定 seed、initial-state hash、确定性 replay；  
- 可暂停/恢复队列、幂等 `run_id`、重试分类；  
- 为同一起点 K2 分支（G4-04）准备执行器。

## 实现范围与边界

### 必做

- 风险分项可解释，不压成黑盒单分；  
- INVALID/仿真异常计入成本但不污染成功统计；  
- 首次失败 vs 独立失败去重规则。

### 明确不做

- 不把 optional 搜索结果写进 G4A 必做交付。

## 完成标准与验证

### 最小通过

- 固定 seed 重放一致；断点恢复不重复 rollout；  
- 样本数/有效数/预算表一致；  
- baseline manifest + 失败窗口可归档。

### 建议验证命令

```text
sdf sim preflight   # live 时
python3 -m unittest discover -s tests/g4 -t . -v
```

## 允许修改

`scenario_search/baselines`、`checkpoint`、`executor`、`config/search`、`tests/g4`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
