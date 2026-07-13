# G3-05：VLA+Safety 闭环与阶段验收

**状态**：PENDING  
**依赖**：G3-01～G3-04  
**阶段角色**：必做（G3 关闭）  
**一句话**：证明 VLA 真能进 Safety→控制闭环；稳与可降级优先，分数其次。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` C1（主）、C3（Safety 护栏交叉）；  
3. `docs/project/SDF_VLA_WORLD_SYSTEM_ARCHITECTURE.md` 运行模式相关节；  
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 1～7、10、12 节；  
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、5、7、8 节；  
6. G2-05 与 G3-01～G3-04 最终产物、失败 Evidence、断点。

## 项目成功口径（本任务）

- **C1 最小成功**：单机上 VLA 产出完整轨迹并完成无 Classic 当前帧候选的可重复闭环（可 `WITH_LIMITS`）。  
- Hybrid 作为工程稳健对照，**不替代** VLA_SAFETY 主证明。  
- 不要求全面超过 Classic；超过了是加分，没超过诚实写。

## 目标

在 Runtime + G2 Safety + MPC 上对比并固化：

| 配置 | 目的 |
|---|---|
| Classic-Observable | 系统下界/对照 |
| 非语言基线 | 隔离轨迹头 |
| Raw VLA | 无/弱 Safety 行为（慎用，短测） |
| VLA+Validator | 硬预筛 |
| VLA+完整 Safety | 主配置 |
| HYBRID | 工程稳健 |

记录拒绝/修复、超时、接管、资源与失败类型。

## 实现范围与边界

### 必做

- 固定场景 + 多种子（至少小集合）；  
- 控制 20ms 环不等待 GPU；  
- unavailable/timeout/stale → 既定降级；  
- 模型卡 + Evidence 目录。

### 明确不做

- 不开始 G4 大规模搜索；不训 World；  
- 不把 live 未跑写成 VERIFIED。

## 完成标准与验证

### 最小通过

- `VLA_SAFETY` 固定简单路线可重复跑通；  
- 故障注入至少覆盖 timeout/stale/NaN 之一并正确降级；  
- Evidence hash/link 自检。

### 诚实记录

- 碰撞/完成/舒适/时延表，负结果保留；  
- G2 live 限制若仍在，写入本阶段限制清单。

### 建议验证命令

```text
sdf sim preflight
python3 -m unittest discover -s tests/g3 -t . -v
# live 验收脚本（实现后登记）
```

## 允许修改

G3 小型缺陷修复、`runtime` adapter、`validation/g3`、`registry`、`artifacts/g3`、`tests/g3`、`reports`、本任务、`PROGRESS.md`。  
通过后停止，不自动开 G4。

## 断点记录

尚未开始。
