# G4-04：G4A 可比 K2 分支与 Oracle Best-of-K

**状态**：PENDING  
**依赖**：G4-02  
**阶段角色**：必做（G4A）  
**一句话**：同一起点跑两条 VLA 候选，算出 oracle best-of-K，给 World 科学标签与演示数据。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_WORLD_MODEL_DESIGN.md` 第 3 节（四项选择空间）；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、13、14 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、6～8 节；  
5. G4-01/G4-02 产物；G3-05 模型；G2 SafetyEvent 接口。  
   G4-03 非依赖；已跑可只读对照。

## 项目成功口径（本任务）

- oracle 只用于**评价**，不进在线控制。  
- 无论选择空间强或弱，**G5 仍会实现 World**（作品要求）；本任务结果只决定 C2 能吹多响。  
- 最小化/聚类属 optional，非完成条件。

## 目标

- 相同 initial-state hash/seed 下执行 VLA-V1 两条候选；  
- 记录 proposed / accepted / executed；  
- 汇总 top-1 vs oracle best-of-K；  
- 不可比分支明确打标。

## 实现范围与边界

### 必做

- Ego/关键 Actor/信号/路线容差检查；  
- 完整 provenance；  
- 不可比结果禁止进入“World 有效”或 preference 主张。

### 明确不做

- 当前帧 CARLA 在线分叉控车。

## 完成标准与验证

### 最小通过

- K2 分支可重复；  
- 冻结困难切片上 top-1/oracle 表可生成；  
- Registry 可查询重放。

### 诚实记录

- 同步失败率、坍塌、选择空间强弱标签。

## 允许修改

counterfactual/分支执行相关实现、`tests/g4`、`registry`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
