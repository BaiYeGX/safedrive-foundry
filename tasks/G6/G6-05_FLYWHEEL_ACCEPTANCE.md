# G6-05：一轮失败驱动适配阶段验收

**状态**：PENDING  
**依赖**：G6-01～G6-04  
**阶段角色**：必做（G6 关闭）  
**一句话**：验收“失败→窗口→修正采集→单轮监督→回归”全链真实跑完；增益可负。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` C5、发布配置；  
3. `docs/project/SDF_VLA_1B_DESIGN.md` G6；  
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8、10、12～13 节；  
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` G6；  
6. G6-01～G6-04 产物与 A/B。

## 项目成功口径（本任务）

- 成功 = 飞轮**真发生**且可追溯，不是必须 SOTA。  
- 不要求多轮自动飞轮、RL、多 preference。  
- 有 PostTrained 权重则进入 G8 第四槽；没有则 limits。

## 目标

固化 C5 证据：数据效率、成本、泄漏阻断、谱系；输出是否采用 posttrain 权重的建议（可否）。

## 实现范围与边界

### 必做

- 完整 Evidence；不自动开 G7。

### 明确不做

- 扩大实验矩阵到不可维护。

## 完成标准与验证

- 谱系可重建；护栏未恶意放宽；C5 正/负皆可。

## 允许修改

`validation/g6`、`registry`、`artifacts/g6`、`tests/g6`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
