# G8-02：统一回归与长稳（含演示配置）

**状态**：PENDING  
**依赖**：G8-01  
**阶段角色**：必做  
**一句话**：四配置（能跑的都跑）在固定矩阵上回归；稳定与可定位失败优先。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` 全文；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 1～7、9～14 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 全文；  
5. G8-01 冻结协议与矩阵。

## 项目成功口径（本任务）

- **VLA+World+Safety** 必须出现在矩阵且稳定段通过。  
- 矩阵可按预登记规则缩小，但不得偷偷拿掉 World 槽。  
- 长稳数字只报实测。

## 目标

在常规/长尾/历史失败/OOD/故障子集上跑发布配置；记录 20ms deadline、模型周期、GPU/CPU/内存/磁盘、恢复。

## 实现范围与边界

### 必做

- 串行加载配置（16GB 共卡）；  
- 失败可定位到模块；  
- 关键 case 抽查轨迹/事件。

### 明确不做

- 预写达标；扩展未冻结场景刷分。

## 完成标准与验证

- 矩阵完整或按规则标缺失；  
- World on 配置无系统性崩塌；  
- 资源峰值表。

## 允许修改

`validation/g8/regression`、`stress`、`collector`、`artifacts/g8`、最小发布阻塞修复、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
