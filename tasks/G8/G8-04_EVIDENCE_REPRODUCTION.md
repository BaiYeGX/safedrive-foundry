# G8-04：Evidence Bundle、演示与冷启动复现

**状态**：PENDING  
**依赖**：G8-02、G8-03  
**阶段角色**：必做  
**一句话**：别人按清单能复现你的演示与正/负结果，而不是只有截图。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` 全文；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 10～14 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 全文；  
5. G8-01～G8-03 产物。

## 项目成功口径（本任务）

- 演示入口必须覆盖 **VLA+World+Safety**。  
- 负结果回放与正结果同等重要。  
- 简历数字只能引用 VERIFIED 或明确标注 MEASURED。

## 目标

Evidence manifest：原始 run、指标、图、视频/轨迹、失败回放、复现脚本；冷启动抽检。

## 实现范围与边界

### 必做

- 四配置（存在的）演示步骤；  
- hash 链接检查；损坏/缺失检测。

### 明确不做

- 选择性删除失败 run。

## 完成标准与验证

- 全新工作目录复现代表性结果；  
- 主张→协议→日志可追溯。

## 允许修改

`validation/g8/evidence`、`artifacts/g8`、`docs`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
