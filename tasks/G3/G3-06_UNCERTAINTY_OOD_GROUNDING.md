# G3-06：不确定性、OOD 与证据一致性

**状态**：PENDING  
**依赖**：G3-05

## 目标与范围

完成温度/概率校准、候选分歧、embedding OOD、选择性驾驶、Fast/Slow trigger，以及语言—Actor—风险—轨迹的反事实 grounding 检查。

允许修改 `safedrive_foundry/driving_vla/uncertainty/**`、`safedrive_foundry/driving_vla/evaluation/grounding/**`、`safedrive_foundry/config/**`、`tests/g3/**` 与报告；不更改 G2 安全阈值。

## 完成标准与验证

- 报告 ECE/Brier、Risk-Coverage、OOD Recall、误/漏接管和 Slow 触发成本。
- 删除/移动声称关键 Actor 时，风险与轨迹变化可测；流畅但错误解释判失败。
- 视觉退化、路线冲突、未见 Town/天气和候选分歧用例通过。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取 calibration artifact、OOD split 和未通过 grounding case。
