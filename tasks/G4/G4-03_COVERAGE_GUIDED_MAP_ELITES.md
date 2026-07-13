# G4-03：覆盖引导 MAP-Elites 主动失败搜索（G4B Optional）

**状态**：PENDING  
**依赖**：G4-02  
**阶段角色**：**Optional / 默认不执行**  
**一句话**：锦上添花的自动搜场景；不挡 G4A、G5、G8 与作品演示。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10 节；  
3. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、6～8 节；  
4. `docs/project/CLAIMS.md` C4（C4B optional）；  
5. G4-01/G4-02 执行器、预算与断点。

## 项目成功口径（本任务）

- 未执行 → 写 `OPTIONAL_NOT_RUN`，**完全合法**。  
- 不得用 G4B 结果冒充 G4A 固定包门禁通过。

## 目标

实现单一 MAP-Elites/archive：descriptor/bin/quality，Random/LHS 初始化，mutation 后重过合法性。

## 实现范围与边界

### 若执行则必做

- 与 Random/LHS 同预算公平比；  
- INVALID 不更新 archive；checkpoint 可恢复。

### 明确不做

- 并行多套 QD/CMA-ES 作为完成条件；  
- 不阻塞 G4-04/G4-05。

## 完成标准与验证

- 执行则：覆盖/独立失败/成本表 + 负结果可保留。  
- 不执行则：断点写明 `OPTIONAL_NOT_RUN` 与原因（时间/优先级）。

## 允许修改

`scenario_search/qd`、`archive`、`config/search`、`tests/g4`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

默认不执行。若启动，在此记录预算与结果路径。
