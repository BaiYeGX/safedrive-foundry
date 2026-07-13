# G7-02：场景/失败研究助手（Optional）

**状态**：PENDING  
**依赖**：G7-01  
**阶段角色**：**Optional**  
**一句话**：受控助手提场景假设与根因草案，证据必须可指回 run id。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10、11 节；  
3. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2、5～8 节；  
4. G7-01 工具与审计；G4/G5/G6 只读接口。

## 项目成功口径（本任务）

- 无 G7-01 不启动。  
- 幻觉/缺证据必须 abstain。

## 目标

覆盖空洞/失败簇 → 可证伪场景草案 + 根因候选 + 复现实验建议；执行仍归治理层。

## 实现范围与边界

- 场景须过 G4 schema；  
- 区分事实/推断/未知；不改真值标签。

## 完成标准与验证

- 与模板同预算对照；注入/伪证据检测。

## 允许修改

`agents/research_assistant`、`evidence_retriever`、`validation/g7`、`tests/g7`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

默认不执行。
