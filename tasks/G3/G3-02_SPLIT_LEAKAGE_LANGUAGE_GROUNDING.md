# G3-02：数据划分、特权泄漏与语言 Grounding

**状态**：PENDING  
**依赖**：G3-01

## 目标与范围

按 Town/Route/场景族/天气/时间和失败簇冻结 ID、OOD、Regression 划分；构建由真值模板生成并校验的行为、关键 Actor、规则、风险时域和结构化证据标签。

允许修改 `safedrive_foundry/data_pipeline/vla/audit/**`、`safedrive_foundry/data_pipeline/vla/split/**`、`safedrive_foundry/data_pipeline/vla/labeler/**`、`safedrive_foundry/data_pipeline/schema/**`、`safedrive_foundry/validation/g3/**`、`tests/g3/**` 和数据卡；不训练正式模型。

## 完成标准与验证

- CARLA 隐藏未来、真实 TTC、碰撞结果和测试专家轨迹不能进入 Policy Input。
- 重复帧、近重复路线、同失败簇跨 split 泄漏可检测。
- 语言标签与 actor_id/conflict_zone/risk_horizon/trajectory_id 对齐；无证据标签拒收。
- 故意泄漏和错误 grounding 注入测试通过，冻结 split manifest。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取数据版本、split hash、审计报告和争议标签。
