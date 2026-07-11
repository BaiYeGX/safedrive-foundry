# G3-03：VLA监督训练与Open-loop评测
**状态**：PENDING  
**依赖**：G3-01、G3-02

## 目标
完成VLT/SFT多任务训练、稳定基线、校准和open-loop消融。

## 范围
训练轨迹、NLL、行为、风险、关键Actor、不确定性、证据与平滑目标；不做闭环后训练。

## 完成标准
- 训练可断点续训并自动登记模型/数据/配置。
- 与无语言、单轨迹、简单视觉/行为基线比较。
- 报告ADE/FDE/NLL、行为、风险、Actor、grounding、ECE/Brier和资源。
- 选择闭环候选checkpoint有可追溯依据。

## 验证方法
独立验证/测试集评测、消融、错误切片和重复训练一致性检查。

## 断点记录
最后状态：PENDING；恢复时从registry中的最近合法checkpoint继续。

