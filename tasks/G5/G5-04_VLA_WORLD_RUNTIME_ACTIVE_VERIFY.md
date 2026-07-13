# G5-04：World 软排序运行时接入与确定性降级

**状态**：PENDING  
**依赖**：G5-03  
**阶段角色**：必做（**作品演示关键接线**）  
**一句话**：把 World 接到预筛后的 K2 候选上做软排序，超时就关掉 World，车还得能开。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_VLA_WORLD_SYSTEM_ARCHITECTURE.md` 在线链；  
3. `docs/project/EXECUTION_ARCHITECTURE.md` 时间、tick owner、Profile、权限；  
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 1～7、9、12～14 节；  
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、5、7、8 节；  
6. G2-05、G3-05、G5-01～G5-03 产物。

## 项目成功口径（本任务）

- 演示默认应能打开 **VLA+World+Safety**。  
- World 失败不得拖死 50Hz 控制。  
- Active CARLA 异步队列 **optional**，非第一版完成条件。

## 目标

- 仅对 G2 预筛合法的 VLA-V1 两候选做 collision/off-road/TTC 优先软排序；  
- 排序后再进最终 Validator/Safety；  
- 实现开关：`world_enabled=true/false`。

## 实现范围与边界

### 必做

- VLA P95 目标 ≤200ms；World 目标约 5Hz（实测登记）；  
- timeout/OOM/NaN/invalid → 跳过排序，退回 VLA+Safety；  
- `VLA_SAFETY` 无 Classic 当前帧候选；  
- World 不能解除硬拒绝或改 Safety 阈值。

### 明确不做

- 当前帧在线 CARLA 分叉控车；World 成为 tick owner。

## 完成标准与验证

### 最小通过

- 同场景开关 World 可复现两条 trace；  
- 注入 World 超时 → 降级成功；  
- 无第二 tick master。

### 诚实记录

- 排序收益/误排/时延/显存/降级率；差也如实写。

## 允许修改

`world_model/runtime`、`verification_queue`、`cache`、arbitration adapter、`tests/g5`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
