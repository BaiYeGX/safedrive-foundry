# G6-01：失败窗口、Hard Case 与数据门禁

**状态**：PENDING  
**依赖**：G4-05  
**阶段角色**：必做（G6 起点；不依赖 World 有增益）  
**一句话**：把 VLA 失败前 2～4 秒切成可审计窗口，供一轮监督适配使用。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/EXECUTION_ARCHITECTURE.md` 身份/时间/Registry；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3、8、10、12、13 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2～4、6～8 节；  
5. `docs/project/CLAIMS.md` C5；  
6. G4-05 与 G3-05 产物；  
7. 仅当 G5 已完成时附加读 G5-05 误排/事件 schema。

## 项目成功口径（本任务）

- G5 未完成或 World 负收益 **不阻塞** 本任务。  
- 窗口质量与防泄漏优先于数量。

## 目标

- EventWindow / HardCase schema、去重、场景平衡；  
- provenance/hash；Regression 隔离；  
- 可选附加 World 误排事件。

## 实现范围与边界

### 必做

- 失败前 2～4s + 恢复段规则；  
- 泄漏/冲突/不可比窗口拒绝。

### 明确不做

- 不在本任务开训 adapter。

## 完成标准与验证

- schema/manifest 可重建；单元测试拒绝泄漏；谱系可回溯 run/frame/candidate。

## 允许修改

`data_pipeline/events`、`hardcase`、`schema`、`registry`、`config`、`tests/g6`、`docs`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
