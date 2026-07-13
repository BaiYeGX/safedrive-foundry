# G1-05：Hybrid A*–Reeds–Shepp 复杂机动

**状态**：`COMPLETED_WITH_LIMITS`
**依赖**：G1-03

## 目标

为道路阻断、狭窄绕障、掉头、倒车和脱困提供独立于 Frenet 假设的复杂机动规划能力，并冻结基础搜索版本供 G1-07 公平优化。

## 范围

实现 SE(2) 离散、运动原语、非完整约束、解析 Reeds–Shepp expansion、碰撞检查、启发式、档位/转向切换代价、超时和部分解。复用 G1-04 的车辆模型与轨迹接口。

## 不做与允许修改

不替代常规 Frenet 主规划器，不实现控制器。允许修改 `safedrive_foundry/classic_stack/planning/hybrid_astar/**`、`safedrive_foundry/classic_stack/geometry/**`、`safedrive_foundry/config/**`、`tests/g1/**` 和 `docs/architecture/**`。

## 完成标准

- 阻断绕行、三点掉头、倒车入位与死胡同脱困可运行。
- 与 grid A*+Dubins 基线比较路径、曲率、换挡、节点和耗时。
- 超时、无解、partial solution 和碰撞拒绝原因可审计。
- 规划器选择规则不依赖场景名称硬编码。
- 基础启发式、离散分辨率、解析扩展策略和节点预算版本化；G1-07 不得用不同地图或 deadline 伪造节点/时延收益。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。

## 验证与断点

| 字段 | 内容 |
|---|---|
| 时间 | 2026-07-13 |
| 最后状态 | `COMPLETED_WITH_LIMITS` |
| 已完成 | Hybrid A*（自行车原语 + RS 解析扩展 + 部分解）、grid A*+Dubins 基线、特征选择器、四类离线机动场景、配置冻结；partial ⇒ `ok=false` |
| config_hash | 见 `docs/architecture/evidence/g1-05/config_hash.txt` |
| 权威证据 | `docs/architecture/evidence/g1-05/repair-20260713/`（含 `manifest.json`） |
| 已运行验证 | `unittest tests.g1.test_g1_05_hybrid_astar` PASS；阶段全量 `tests/g1` **78 OK** |
| 限制 | 以离线机动为主；权威 G1 live **非** Hybrid/RS 进环证明（dense uturn） |
| 停止 | **任务关闭**；阶段状态见 `PROGRESS.md` / G1-09 |
