# G7-03：确定性工作流与 Agent 红队（Optional）

**状态**：PENDING  
**依赖**：G7-01、G7-02  
**阶段角色**：**Optional**  
**一句话**：证明关 LLM 也能 data/release；Agent 只出草案。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` C6；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10、11 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2、5～8 节；  
5. G7-01～G7-02 产物。

## 项目成功口径（本任务）

- C6 负收益 → Agent 默认关，仍 OK。  
- 不阻塞 G8。

## 目标

确定性 data/release 流程 + Agent 红队；重复请求同 manifest。

## 完成标准与验证

- LLM-off 全流程；越权/投毒/中断红队；Evidence 自检。

## 允许修改

`agents/workflows`、`data_release_adapter`、`validation/g7`、`registry`、`artifacts/g7`、`tests/g7`、本任务、`PROGRESS.md`。

## 断点记录

默认不执行。
