# G4-01：场景注册与搜索基线
**状态**：PENDING  
**依赖**：G3-04

## 目标
建立Engineered、Failure-derived、System-generated三路场景注册、故障schema、有效性规则和Manual/Random/LHS基线。

## 范围
参数化核心逻辑场景与多目标风险指标；不实现CMA-ES和Agent搜索。

## 完成标准
- Logical/Concrete/Regression/Minimal Counterexample层级稳定。
- INVALID场景规则明确且不计风险发现。
- 同预算搜索协议、seed和结果登记完成。
- 至少覆盖代表性切入、急刹、路口、VRU和故障场景。

## 验证方法
schema校验、合法/非法采样、Random/LHS覆盖与复现测试。

## 断点记录
最后状态：PENDING；恢复时读取scenario registry和未完成场景族。

