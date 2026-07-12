# G3-04：轻量 Fast/Slow 驾驶 VLA 架构

**状态**：PENDING  
**依赖**：G3-03

## 目标与范围

在 RTX 4080 16GB 上实现前视视觉、Ego、Route 和可选语言融合，使用并行 action/trajectory queries 输出 K 候选、Behavior、Risk、Critical Actor、Uncertainty 与 Evidence。Fast 路径常态生成轨迹，Slow 路径只在 OOD/分歧/复杂交互触发。

允许修改 `safedrive_foundry/driving_vla/model/**`、`safedrive_foundry/driving_vla/adapter/**`、`safedrive_foundry/config/**`、`tests/g3/**` 和 `docs/architecture/**`；不做完整训练和闭环。

## 完成标准与验证

- 不自回归逐点生成连续轨迹；输出动力学/数值约束可检查。
- 支持冻结骨干、LoRA/QLoRA、混合精度、checkpoint 和 OOM 降级。
- 合成/真实 batch 前后向、保存恢复、显存、吞吐与 trigger 测试通过。
- 大模型仅为可选离线教师，不成为运行依赖。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时记录模型配置、参数量、显存、失败 batch 和 checkpoint。
