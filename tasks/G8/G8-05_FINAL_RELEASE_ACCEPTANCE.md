# G8-05：最终审计与版本准出

**状态**：PENDING  
**依赖**：G8-04  
**阶段角色**：必做（项目关闭）  
**一句话**：确认作品可答辩：VLA+World 真上了、能稳跑、限制写清、不夸大。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` 全文；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 全文；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 全文；  
5. G8-01～G8-04 产物与限制。

## 项目成功口径（本任务）

准出检查清单：

| 项 | 要求 |
|---|---|
| VLA | 闭环证据存在 |
| World | 接入证据存在，on/off 对照存在 |
| Safety | 降级与硬约束未被学习覆盖 |
| 稳定 | 演示配置可重复跑 |
| 诚实 | 负结果/limits 已登记 |
| 实车 | **明确不宣称** |
| G4B/G7 | 可为 `OPTIONAL_NOT_RUN` |
| G5 | **不得**用 SKIPPED 逃避实现（本作品路径） |

## 目标

输出发布候选：叙事、限制、复现索引、配置表；只修发布阻塞。

## 实现范围与边界

### 必做

- 需求→任务→测试→Evidence 追踪表；  
- 不扩大实验、不移动阈值换结论。

### 明确不做

- 自动 commit/push/merge main。

## 完成标准与验证

- 清单勾选完成；从零抽检通过；  
- 完成后停止。

## 允许修改

发布阻塞级最小修复、`validation/g8`、`docs`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
