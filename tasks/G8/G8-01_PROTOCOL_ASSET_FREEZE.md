# G8-01：主张、协议与资产冻结

**状态**：PENDING  
**依赖**：G6-05  
**阶段角色**：必做（G8 起点）  
**一句话**：冻结“能演示什么、不能吹什么”，以及代码/模型/数据/场景 hash。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` 全文；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 10～14 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 全文；  
5. G2-05、G3-05、G4-05、G5-05、G6-05 结果；G7 仅若已执行。

## 项目成功口径（本任务）

四槽位（World **必须有可运行第三槽**）：

1. Classic（+Safety）  
2. VLA+Safety  
3. **VLA+World+Safety**（作品主展示）  
4. PostTrained…（无 posttrain 则标记缺失/limits，不虚构）

不上实车；CARLA ≠ 道路安全证明。

## 目标

冻结 C0～C6（C6 仅 G7 执行时）、指标/护栏、四配置 manifest、训练与最终测试隔离。

## 实现范围与边界

### 必做

- 每项主张：对照、主指标、护栏、统计单位、停止规则；  
- World/Agent 负结果配置显式化；  
- hash 重建 smoke。

### 明确不做

- 不在本任务跑全矩阵（G8-02）。

## 完成标准与验证

- 干净工作区 hash 核对；最终标签对训练不可见。

## 允许修改

`validation/g8/protocol`、`registry/evidence`、`artifacts/g8/freeze`、`docs`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
