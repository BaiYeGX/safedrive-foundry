# G4-04：场景搜索与反事实综合评测
**状态**：PENDING  
**依赖**：G4-01～G4-03

## 目标
完成Manual/Random/LHS/CMA-ES公平评测，验证失败发现、复现、最小化和反事实证据链。

## 范围
只做G4综合实验与门禁，不训练世界模型或Agent。

## 完成标准
- 报告首次失败rollout、固定预算独立失败、最严重风险、INVALID率、复现率和最小反例距离。
- 稀有事件含置信区间。
- Regression Case和Counterexample Registry完成。
- G4 Evidence Bundle完整。

## 验证方法
按冻结实验协议批量运行并独立复查代表性失败证据。

## 断点记录
最后状态：PENDING；恢复时从缺失实验单元和run registry继续。
