# G6-04：抗遗忘与前后对照回归

**状态**：PENDING  
**依赖**：G6-03  
**阶段角色**：必做  
**一句话**：同一冻结集上对比训练前后：困难切片、正常集、OOD、Safety 与时延。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` C5；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10、12～13 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` G6 字段；  
5. G6-01～G6-03 产物与断点。

## 项目成功口径（本任务）

- 允许困难集小改善、正常集持平或轻微波动；  
- **禁止**用平均分掩盖关键场景崩坏；  
- 无改善 → 负结论，仍算任务完成。

## 目标

配对 seed 比较 posttrain 前后在：目标失败、相邻簇、正常、OOD、故障、实时性上的变化。

## 实现范围与边界

### 必做

- 预登记提升/不可接受回归/无结论区间；  
- 四类指标：安全、效率舒适、泛化、资源实时。

### 明确不做

- 不在本任务宣称 G6 阶段完成（留给 G6-05）。

## 完成标准与验证

- 效应与区间表；全谱系；结果只进入 G6-05。

## 允许修改

`validation/g6`、`registry`、`artifacts/g6`、`tests/g6`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
