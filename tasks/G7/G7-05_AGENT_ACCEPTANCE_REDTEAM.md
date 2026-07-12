# G7-05：Agent 净收益、Grounding 与红队验收

**状态**：PENDING  
**依赖**：G7-01～G7-04

## 目标与完成标准

在相同任务和预算下比较人工固定脚本、模板流程与 Agent，报告成功率、独立失败、诊断准确、证据率、幻觉、时间、算力/API 成本和人工干预。执行越权、提示注入、证据伪造、资源耗尽和中断恢复红队；形成自动/需审批/禁止清单。未证明净收益不发布 Agent 主张。

允许修改 G7 小缺陷、`safedrive_foundry/validation/g7/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/artifacts/g7/**`、报告、Evidence、本任务和 `PROGRESS.md`。最后状态：PENDING；不自动开始 G8。

## 验证方法

执行同预算基线对照、工具调用审计、grounding 抽检、权限红队、提示注入与中断恢复测试。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时读取实验预算、审计日志、红队失败队列与最近命令。
