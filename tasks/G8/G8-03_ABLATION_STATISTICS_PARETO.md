# G8-03：核心消融、统计与 Pareto

**状态**：PENDING  
**依赖**：G8-02  
**阶段角色**：必做（范围可收缩）  
**一句话**：只做能支撑叙事的核心消融，不做笛卡尔积。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` 全文；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 10～14 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 全文；  
5. G8-01/G8-02 产物。

## 项目成功口径（本任务）

### 必做消融（作品向最小集）

- VLA vs 非语言/Classic 对照（能跑的）；  
- World on vs off（同一 VLA/Safety）；  
- Safety 链关键档位（若数据允许）；  
- posttrain before/after（若 G6 有权重）。

### Optional（仅已执行时）

- G4B 搜索；G7 Agent；V2 REASON；Active CARLA。

负贡献必须标出，不得删表。

## 目标

配对消融 + 区间/bootstrap（能力范围内）+ Pareto（安全/效率/资源）。

## 完成标准与验证

- 表与 registry 一致；无从次要 slice 反推全局。

## 允许修改

`validation/g8/stats`、`artifacts/g8`、`tests/g8`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
