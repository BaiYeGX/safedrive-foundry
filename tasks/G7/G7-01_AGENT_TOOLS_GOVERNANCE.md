# G7-01：Agent 白名单工具与治理（Optional）

**状态**：PENDING  
**依赖**：G6-05  
**阶段角色**：**Optional / After Release；默认不执行**  
**一句话**：若要做研发 Agent，先做沙箱与审计；不是作品主线。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/EXECUTION_ARCHITECTURE.md` 权限/Registry/审计；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10、11 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2、5～8 节；  
5. G6-05 资产与确定性脚本入口。

## 项目成功口径（本任务）

- 未执行 → `OPTIONAL_NOT_RUN`，**不影响 G8 准出**。  
- Agent 永不进实时控制环。

## 目标

typed tools、路径沙箱、预算、超时、dry-run、幂等、审批、审计、停止与恢复。

## 实现范围与边界

### 若执行则必做

- 禁止任意 shell/第二 tick master；  
- 不改冻结 Safety/split；  
- 审计可重放。

## 完成标准与验证

- 红队：越权/超时/注入；LLM off 时工具仍可由脚本调用。

## 允许修改

`agents/tools`、`governance`、`audit`、`tests/g7`、`docs`、本任务、`PROGRESS.md`。

## 断点记录

默认 `OPTIONAL_NOT_RUN`。
