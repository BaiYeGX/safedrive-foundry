# G5-03：风险头、动作敏感性与两候选排序

**状态**：PENDING  
**依赖**：G5-02  
**阶段角色**：必做  
**一句话**：输出 collision/TTC/off-road 与 pairwise 排序，并证明模型真的吃 ego action。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_WORLD_MODEL_DESIGN.md` 风险与排序节；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3、5、9、10、13 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2～4、7、8 节；  
5. `docs/project/CLAIMS.md` C2；  
6. G5-01/G5-02 产物与断点。

## 项目成功口径（本任务）

- World **永远不是 Safety 真值**。  
- 排序可差，但异常必须返回 invalid/unavailable/timeout，**禁止静默零风险**。  
- 无动作条件对照更强 → 不得宣称动作条件有效。

## 目标

形成 `WorldRolloutBatchV0`：actor future + collision + TTC + off-road + 两候选排序 + 异常标志；完成 action shuffle / no-action 检验。

## 实现范围与边界

### 必做

- pairwise ranking 与 regret 记录；  
- 与简单基线同 split 对比。

### 明确不做

- M3、复杂 ensemble OOD、独立大神经 risk 总分为完成条件。

## 完成标准与验证

### 最小通过

- 安全/擦碰/碰撞/off-road 样例可输出分项风险；  
- action 敏感性测试有报告；  
- 失败路径不输出假安全。

### 诚实记录

- 优于基线或负结论二选一，不得空表。

## 允许修改

`world_model/heads`、`losses`、`calibration`、`evaluation`、`interventions`、`config`、`tests/g5`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
