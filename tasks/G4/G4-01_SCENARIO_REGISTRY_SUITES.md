# G4-01：场景注册表、核心长尾套件、覆盖与可解性

**状态**：PENDING  
**依赖**：G3-05  
**阶段角色**：必做（G4A 基础）  
**一句话**：把“能复现的困难场景”管成 Registry，而不是一堆散落脚本。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/EXECUTION_ARCHITECTURE.md` Registry、身份、Oracle/Observable；  
3. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 8 节；  
4. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 2、4、6～8 节；  
5. `docs/project/CLAIMS.md` C4；  
6. G3-01 场景身份/split 与 G3-05 发布模型、失败清单、断点。

## 项目成功口径（本任务）

- 优先 **20～40 个可复现困难场景** 的质量，不追求自动搜索覆盖全世界。  
- 场景要同时服务：VLA 压测、oracle 对比、World 数据、演示录像。

## 目标

扩展 G3 场景身份为可注册层级（Functional→Logical→Concrete→Regression→可选 Minimal Counterexample），统一 Engineered / Failure-derived / System-generated 来源元数据，并定义合法性与可解性门禁。

## 实现范围与边界

### 必做

- 场景族至少覆盖：切入/急刹、换道冲突、无保护左转、VRU/行人、遮挡、阻断或 U-turn、天气/视觉退化、传感器或模型故障类之一；  
- 参数域、有效性、预期事件、termination、版本字段；  
- 正常 / 长尾 / 故障 / Regression 套件隔离；  
- 与 G3 split 兼容的 `scenario_id` / provenance。

### 明确不做

- 不实现 MAP-Elites（G4-03）；  
- 不在本任务训练 VLA/World。

## 完成标准与验证

### 最小通过

- 非法 spawn、不可达、重叠、先天不可解 → `INVALID`；  
- Classic-Oracle **仅可解性**，不进 Observable 策略；  
- 固定 seed 属性可复现；schema/registry 可查询。

### 建议验证命令

```text
python3 -m unittest discover -s tests/g4 -t . -v
```

## 允许修改

`safedrive_foundry` 下 `scenario`、`solvability`、`registry/scenario`、`config/scenarios`、`tests/g4`、`docs`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
