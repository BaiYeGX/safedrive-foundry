# G5-02：World-V0 轻量动作条件对象中心模型

**状态**：PENDING  
**依赖**：G5-01  
**阶段角色**：必做  
**一句话**：训练约 4M～8M 的动作条件 World，预测他车/环境对 Ego 候选的响应（非视频生成）。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_WORLD_MODEL_DESIGN.md` World-V0 结构与规模；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3、5、6、9、13 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1～4、7、8 节；  
5. `docs/project/CLAIMS.md` C2；  
6. G5-01 数据集与基线。

## 项目成功口径（本任务）

- **先稳定前向与 checkpoint**，再谈是否优于 CV/CTRV。  
- 不生成像素视频；不做 tick owner。  
- 单卡 16GB 训不动就减层/宽/样本，不换服务器叙事。

## 目标

训练 World-V0：

| 项 | 值 |
|---|---|
| K | 2 |
| T / dt / horizon | 10 / 0.25s / 2.5s |
| N actors | ≤8 |
| M modes | 1 |
| 参数 | 约 4M～8M（实施后登记实数） |

条件：ego、相关 actor、简化车道/可行驶区域、候选轨迹。

## 实现范围与边界

### 必做

- action conditioning；mask/missing actor 行为；  
- mixed precision、OOM 降级、精确恢复。

### 明确不做

- Frozen visual teacher 非完成条件；  
- 不接 Safety 硬阈值。

## 完成标准与验证

### 最小通过

- 不同 Ego 候选产生可区分 future；  
- action permutation / 小样本过拟合测试通过；  
- 参数量、显存、吞吐、时延有实测表。

### 诚实记录

- 弱于简单基线 → 负结论，**模块仍保留**进入 G5-03/04。

## 允许修改

`world_model/model`、`training`、`config`、`tests/g5`、`registry`、`docs`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
