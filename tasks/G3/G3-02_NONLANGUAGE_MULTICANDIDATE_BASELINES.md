# G3-02：非语言单轨迹与多候选公平基线

**状态**：PENDING  
**依赖**：G3-01  
**阶段角色**：必做  
**一句话**：用便宜、可复现的非语言基线钉住“多候选轨迹”的对照标尺，避免后来把一切收益都算给 VLA。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3～5、12 节；  
3. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2～4、7、8 节；  
4. `docs/project/CLAIMS.md` C1 与统一指标；  
5. G3-01 数据 schema、冻结 split、泄漏审计与断点。

## 项目成功口径（本任务）

- 基线用于**对照**，不要求超过未来 VLA。  
- 基础模型选型已冻结为 SimLingo/InternVL2-1B；**本任务不重新选型**。  
- 输出必须能进同一套 CandidateSet / Safety 解析链。

## 目标

在统一数据、split、轨迹格式与计算预算下，实现并可评测：

1. Route/Ego 类简单策略或 MLP 基线；  
2. 时序视觉 **单轨迹** 基线（对齐 V0 K1 合同）；  
3. 时序视觉 **多候选** 基线接口（对齐 V1 K2 预算，可先 stub 后填）。

Classic-Observable 仅作系统参照，不额外喂特权信息给“公平学习基线”。

## 实现范围与边界

### 必做

- 统一 Policy Adapter：输入 schema、输出 `K/T/dt/horizon`、坐标系、概率占位；  
- V0 合同：`K=1,T=10,dt=0.25s,horizon=2.5s`；为 V1 `K=2` 预留接口与预算表；  
- checkpoint 保存/恢复与 model/data/config hash；  
- open-loop 指标 + 可选极短闭环 smoke（不阻塞本任务，若环境未就绪记 `NOT_RUN`）。

### 明确不做

- 不加载完整 SimLingo 训练；不做 LoRA；不做 World；  
- 不把随机噪声伪造成“模型候选”；  
- 不根据单一 ADE 决定“换基础模型”。

## 接口与交付物

| 类型 | 内容 |
|---|---|
| 代码 | `safedrive_foundry/driving_vla/baselines`、`adapter`、`config` |
| 报告 | ADE/FDE/覆盖/可执行率/Safety 拒绝率/资源表 |
| 测试 | `tests/g3` 形状、hash、恢复、非法轨迹拒绝 |

## 完成标准与验证

### 最小通过

- 三基线（或 1+2 已实现且 K2 接口冻结）在固定 split 上可跑通推理；  
- 输出可被 Validator 解析；非法/NaN 被拒；  
- 单轨迹 vs 多候选在**相同 encoder 预算声明**下可比。

### 诚实记录

- 负结果与失败 slice 保留；  
- 未跑 live 闭环不得写闭环 VERIFIED。

### 建议验证命令

```text
python3 -m unittest discover -s tests/g3 -t . -v
```

## 允许修改

`safedrive_foundry/driving_vla/baselines`、`training`（仅基线）、`adapter`、`config`、`tests/g3`、`validation/g3`、`reports`、本任务、`PROGRESS.md`。  
当前任务通过后停止。

## 断点记录

尚未开始。
