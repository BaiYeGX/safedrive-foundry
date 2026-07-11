# G2-04：Safety Kernel集成评测
**状态**：PENDING  
**依赖**：G2-01～G2-03

## 目标
集成Validator、RATO、Shadow、仲裁、回退与故障注入，证明安全改善没有被过度保守掩盖。

## 范围
完成Safety消融、Pareto分析和G2门禁；不训练VLA。

## 完成标准
- Raw、Validator、Hard Fallback、RATO模式公平对照。
- 同时报碰撞/TTC、完成率、无故停车、舒适性、修改量、成功率和时延。
- 固定场景和故障回归通过。
- G2 Evidence Bundle完整。

## 验证方法
多seed运行安全套件，报告均值、离散度、失败分类和Pareto前沿。

## 断点记录
最后状态：PENDING；恢复时检查消融矩阵和缺失run_id。

