# G3-07：VLA 闭环与阶段验收

**状态**：PENDING  
**依赖**：G3-01～G3-06

## 目标与范围

将 VLA 以 5～10Hz trajectory chunk 接入 Runtime、MPC 与 Safety，比较 Classic-Observable、非语言基线、Raw VLA、VLA+Validator 和 VLA+Safety。只做 G3 集成、小型修复、实验与 Evidence。

## 完成标准与验证

- 正式推理无特权字段，超时/stale/OOD/接管完整记录。
- ID、OOD 和退化视觉多 seed 闭环报告安全、完成、舒适、grounding、时延和资源。
- 对比 open-loop 与 closed-loop 排名差异，明确失败类型和适用边界。
- G3 Evidence 自检通过，不自动开始 G4。

## 明确不做

- 不改变既定路线、直接依赖、Safety 硬约束、数据划分或冻结实验协议，不提前实现后续任务；验收发现的实质缺陷退回原任务。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 允许修改与断点

允许修改 `safedrive_foundry/runtime/**` 中的 G3 adapter、`safedrive_foundry/validation/g3/closed_loop/**`、`safedrive_foundry/registry/**`、报告、本任务和 `PROGRESS.md`。最后状态：PENDING；恢复时从缺失实验单元、策略版本和 run_id 继续。
