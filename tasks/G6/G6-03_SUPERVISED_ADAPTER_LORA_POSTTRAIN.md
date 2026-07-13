# G6-03：一轮监督 Adapter / LoRA 后训练

**状态**：PENDING  
**依赖**：G6-02  
**阶段角色**：必做（单轮；允许负收益）  
**一句话**：只训一个监督适配器，用失败窗口纠偏；不搞 RL 全家桶。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_VLA_1B_DESIGN.md` G6 与预算；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 5、6、10、12、13 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1～4、7、8 节；  
5. `docs/project/CLAIMS.md` C5；  
6. G6-01/G6-02 manifest；G3-04 base checkpoint。

## 项目成功口径（本任务）

- **训练指标 ≠ 闭环收益**；闭环留给 G6-04/05。  
- 禁止 PPO/GRPO/多种 preference/同时训 World。  
- 16GB 训爆就减步数/样本，不换大模型。

## 目标

用失败前窗口 + corrective 轨迹，训练**一个** adapter 或少量 LoRA checkpoint，谱系完整。

## 实现范围与边界

### 必做

- 单一预登记配方；可恢复；NaN/OOM 审计；  
- base/adapter/data/config/seed hash。

### 明确不做

- 多轮自动飞轮；多目标黑盒总损失不可消融。

## 完成标准与验证

- 训练跑完或诚实记资源失败；  
- checkpoint 可加载进同一 VLA 接口；  
- 进入 G6-04 前后对比，不在本任务宣称飞轮成功。

## 允许修改

`driving_vla/posttrain`、`losses`、`config/posttrain`、`registry`、`tests/g6`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
