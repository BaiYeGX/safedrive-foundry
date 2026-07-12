# G1-04：Frenet Lattice 与 S-T 速度规划

**状态**：PENDING  
**依赖**：G1-03

## 目标

完成常规道路的空间候选、动态占用和带时间轨迹，使跟车、换道、停车与常规绕行形成可执行参考轨迹；冻结固定采样基础版本，作为 G1-07 算法优化不可变的公平基线。

## 范围

实现 Frenet 采样、道路/动力学/静态碰撞约束、候选代价、动态 Actor 基线预测、S-T 占用、DP/QP 速度走廊和 jerk 平滑；输出统一 `t,x,y,yaw,kappa,v,a,jerk`。

## 不做与允许修改

不实现倒车/掉头，不实现底层 MPC。允许修改 `safedrive_foundry/classic_stack/planning/frenet/**`、`safedrive_foundry/classic_stack/planning/speed/**`、`safedrive_foundry/classic_stack/geometry/**`、`safedrive_foundry/config/**`、`safedrive_foundry/scenario/**`、`tests/g1/**` 和 `docs/architecture/**`。

## 完成标准

- 跟车、停车、换道、切入制动和常规绕行均有合法解或可诊断无解。
- 轨迹满足道路、曲率、速度、加速度和 jerk 约束。
- 保存候选数量、拒绝原因、代价分项和 P50/P95/P99。
- 与 centerline+constant-speed 简单基线公平比较。
- 固定采样网格、代价权重、预测模型和候选预算写入版本化配置，禁止 G1-07 通过悄然改变基线获得虚假提升。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 验证与断点

固定场景多 seed 比较成功率、碰撞裕度、进度、舒适和耗时；中断记录场景 ID、候选摘要和 solver 状态。

| 字段 | 内容 |
|---|---|
| 最后状态 | PENDING |
| 已完成/验证 | 无 |
| 恢复步骤 | 校验 G1-03 Route Corridor 后从失败场景继续 |
