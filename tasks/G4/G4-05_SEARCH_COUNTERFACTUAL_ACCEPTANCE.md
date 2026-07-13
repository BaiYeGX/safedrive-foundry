# G4-05：G4A 验收与选择空间科学标注

**状态**：PENDING  
**依赖**：G4-01、G4-02、G4-04  
**阶段角色**：必做（G4 关闭）  
**一句话**：验收固定包与 oracle 表，输出选择空间标签；**不否决**后续 World 实现。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_WORLD_MODEL_DESIGN.md` 第 3 节与 WE0；  
3. `docs/project/CLAIMS.md` C4；  
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10 节；  
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` G4 字段；  
6. G4-01、G4-02、G4-04 最终产物与失败清单。

## 项目成功口径（本任务）

输出三态科学标签（写入 Evidence）：

| 标签 | 含义 | 对 G5 实现 |
|---|---|---|
| `ENTER_WORLD` | 四项选择空间条件通过 | 实现 World，C2 可冲正收益 |
| `WEAK_SELECTION_SPACE` | 部分满足/不稳定 | **仍实现 World**；C2 默认谨慎/可能负 |
| `NO_SELECTION_SPACE` / `IMPROVE_VLA` | 几乎无排序空间 | **仍实现 World 作作品模块**；C2 记负或“无增益”；演示可 World on + 对照 off |

旧文 `SKIPPED_BY_GATE` **不再用于跳过本项目 G5 实现**；仅可在严格发表附录中讨论“若追求净收益可暂缓”。

## 目标

- 验收 Registry、20～40 场景、replay、K2 谱系、top-1 vs oracle；  
- 固化 C4 对照与统计单位；  
- 为 G5 提供数据指针与标签。

## 实现范围与边界

### 必做

- 复现率、候选同步失败、资源/wall-time；  
- G4B 未执行 → `OPTIONAL_NOT_RUN`。

### 明确不做

- 不自动开始 G5 实现（仍需用户口令）；  
- 不因弱选择空间删除 World 计划。

## 完成标准与验证

- 标签与原始 run 一致；Evidence 自检通过；  
- 完成后停止。

## 允许修改

`validation/g4`、`registry`、`artifacts/g4`、`tests/g4`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
