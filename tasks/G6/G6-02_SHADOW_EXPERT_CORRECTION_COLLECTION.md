# G6-02：Shadow 专家修正采集

**状态**：PENDING  
**依赖**：G6-01  
**阶段角色**：必做  
**一句话**：影子对照采集 corrective 轨迹，不给 Shadow 控车权。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/SDF_VLA_1B_DESIGN.md` G6 映射；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 1～5、8、12、13 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4～8 节；  
5. `docs/project/CLAIMS.md` C5；  
6. G6-01 schema/门禁/断点。

## 项目成功口径（本任务）

- 这是 **DAgger 风格标签采集**，不是多 preference 主线。  
- Shadow 永不抢 tick、不延迟 50Hz。  
- World 仅可选对照源。

## 目标

并行 VLA / Classic / Safety（+可选 World），在分歧/风险/OOD 时触发查询，产出：

`proposed / accepted / executed / corrective` 全谱系样本。

## 实现范围与边界

### 必做

- 执行/影子/反事实隔离；  
- query budget 与原因审计；  
- 绑定 Observable 与 candidate_id。

### 明确不做

- 多种 preference pair 必做交付；  
- 训练写 Regression。

## 完成标准与验证

- 样本可回放到 run/frame；  
- 中断不重复；  
- timeout/OOD 触发行为正确；  
- 至少够 G6-03 开训的最小集合（数量在断点登记）。

## 允许修改

`data_pipeline/dagger`、runtime shadow adapter、`config/collection`、`tests/g6`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
