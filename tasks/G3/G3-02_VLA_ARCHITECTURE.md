# G3-02：轻量驾驶VLA架构
**状态**：PENDING  
**依赖**：G3-01

## 目标
在RTX 4080 16GB约束下实现视觉、Ego、Route和Language融合的轻量VLA及多任务输出头。

## 范围
冻结/轻量微调视觉与语言骨干；实现Behavior、K-mode Trajectory/Action、Risk、Critical Actor、Uncertainty和Evidence头。

## 完成标准
- 模型输入输出与Policy Adapter一致。
- 轨迹并行生成，不逐点语言解码。
- 支持LoRA/QLoRA、混合精度、checkpoint和OOM降级。
- 参数量、显存、吞吐和模块测试记录完整。

## 验证方法
合成与真实batch前向/反向、保存恢复、数值稳定、显存峰值和输出约束测试。

## 断点记录
最后状态：PENDING；恢复时从checkpoint、配置hash和最近模型测试继续。

