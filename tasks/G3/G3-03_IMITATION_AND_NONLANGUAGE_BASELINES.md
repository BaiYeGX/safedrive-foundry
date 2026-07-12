# G3-03：模仿学习与非语言基线

**状态**：PENDING  
**依赖**：G3-02

## 目标与范围

在引入 VLA 之前建立可复现下限：constant/route follower、Ego+Route MLP、视觉单轨迹、多模态无语言轨迹和 behavior-conditioned planner。统一 Policy Adapter、训练/评测与资源记录。

允许修改 `safedrive_foundry/driving_vla/baselines/**`、`safedrive_foundry/driving_vla/training/**`、`safedrive_foundry/config/**`、`tests/g3/**`、`safedrive_foundry/validation/g3/**` 和报告；不实现语言推理。

## 完成标准与验证

- 所有基线共享数据、split、轨迹格式和训练预算。
- 报告 ADE/FDE/NLL、可执行率、行为、闭环短套件和资源。
- checkpoint 可恢复并登记 model/data/config hash。
- 选出 G3-04 的明确基础模型，不凭单一 open-loop 指标决定。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从 Model Registry 最近合法 checkpoint 与缺失基线继续。
