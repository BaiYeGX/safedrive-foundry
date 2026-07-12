# G2-03：二维 RATO-SCP 与可行走廊

**状态**：PENDING  
**依赖**：G2-02

## 目标与范围

在 Frenet/局部坐标中实现二维序列凸优化，对横纵轨迹做最小修复。包括车辆动力学、道路边界、静动态障碍、凸安全走廊、交通规则、信赖域、slack、迭代收敛与 warm start；复用 G1 `RiskField` 的预测包络，但保持独立 Safety 权限和更保守的约束下界。

## 不做与允许修改

不训练风险模型。允许修改 `safedrive_foundry/safety_kernel/rato/scp/**`、`safedrive_foundry/safety_kernel/geometry/**`、`safedrive_foundry/safety_kernel/solver/**`、`safedrive_foundry/config/**`、`tests/g2/**` 和 `docs/architecture/**`。

## 完成标准与验证

- 每次迭代保存线性化点、约束、slack、信赖域和终止原因。
- 与纵向 QP 在切入、静态绕障、狭窄通道和换道冲突中公平比较。
- 超时、振荡、不可行按已冻结降级链处理。
- 报告安全裕度、修改量、进度、舒适、成功率与时延分位数。
- 与 G1 规划风险代价严格区分：风险代价可以权衡，Safety 硬约束/冻结 slack 上限不得被规划器或模型修改。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；恢复时从最近 solver checkpoint、失败场景和信赖域参数继续。
