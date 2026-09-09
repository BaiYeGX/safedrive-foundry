# SafeDrive Foundry 代理操作规范

本文件只保存长期规则。当前唯一任务见 `START_TASK.md`，后续顺序见 `ROADMAP.md`，
已确认事实见 `PROGRESS.md`。

## 1. 项目边界

- 项目是 CARLA–ROS 2 纯软件在环，不面向实车或公共道路。
- CARLA Server 在 Windows；ROS 2、客户端、模型和训练在 WSL2 Ubuntu。
- 固定硬件为 RTX 4080 16GB 和 i5-13600KF，不得假设第二张 GPU 或远程服务器。
- 唯一活动研究链为 `H0 → H1 → H2 → H3 → H4 → H5 → H6`。
- 本周后续只用 C3、C4、C5、C6 四个完整阶段；普通工程检查不拆小数编号阶段。
  两次训练、一次小型开发闭环与结题材料为当前范围，具体合同见 ROADMAP。
- 核心在线链为 `Observable → Classic Expert + nominal VLA → per-candidate Guard →
  World rank/defer → Safety → MPC/PID`。
- Classic Expert 与 nominal VLA 各自独立提出一条轨迹；学习模块不得伪造第二候选。
- nominal VLA 可使用阶段锁定的原始或微调权重；C4 联合学习的 World 辅助梯度应进入共享
  LoRA。真实未来/控制只作离线 label；预测执行只在模型内部使用，无控制或 tick 权限。
- 硬件限制属于实验设置，不作为论文标题或创新贡献；工程完成与学习收益分开验收。
- `Guard eligible` 固定指 `PASS` 或 `REVIEW`；`REJECT` 候选不得进入 World。
- World 只在通过 Guard 的候选之间排序或放弃判断，不能生成轨迹、覆盖硬安全约束、
  获得无约束底盘控制或 tick 权限。
- World 的 `source-blind` 固定指在线 feature schema 不含 source/slot/order/provenance 元数据；
  不伪称物理轨迹本身不可能暴露 planner 风格，后者必须用审计和反事实平衡诊断。
- Oracle 只允许离线标注；Regression、特权未来和场景答案不得进入在线输入或训练特征。
- `archive/` 是只读可恢复历史区，不是活动需求或实现来源。

## 2. 文档优先级

```text
用户本轮明确指令
> START_TASK.md
> AGENTS.md
> ROADMAP.md
> docs/PROJECT.md
> 其他活动设计文档
> archive 中的历史材料
```

活动权威文档：

| 文件 | 职责 |
|---|---|
| `README.md` | H 路线入口与目录 |
| `START_TASK.md` | 当前唯一任务、验收和停止点 |
| `ROADMAP.md` | H0–H6 最短路线 |
| `PROGRESS.md` | 已确认动态事实 |
| `docs/PROJECT.md` | 系统边界、成功口径和证据合同 |
| `docs/HYBRID_CANDIDATES.md` | Expert/VLA 候选与 Guard 合同 |
| `docs/COUNTERFACTUAL_DATA.md` | SFT/World 训练视图、同锚点双分支与 split 合同 |
| `docs/WORLD_MODEL.md` | VLA 联合学习、执行后果 World 与在线选择合同 |
| `docs/RELATED_WORK.md` | VLA、World、反事实与选择性决策的活动技术定位 |
| `docs/SHOWCASE.md` | 结题演示、简历和面试主干—枝干合同 |
| `docs/RESOURCES.md` | 本机资产与资源预算 |
| `docs/ENVIRONMENT.md` | 环境与运行入口 |
| `docs/EVIDENCE.md` | 活动 Evidence 与归档索引 |

活动文档冲突时停止并报告；不得从 archive 自动恢复旧路线、阈值或任务。

## 3. 单任务循环

每轮严格执行：

1. 读取 `START_TASK.md`、`PROGRESS.md` 和任务直接引用的文档。
2. 检查当前分支及 `git status --short`，识别已有改动。
3. 明确目标、允许路径、验收、验证命令与停止点。
4. 只实现当前任务。
5. 运行直接相关的最小测试。
6. 检查 `git diff`、`git diff --stat` 和 `git diff --check`。
7. 更新 `PROGRESS.md` 后停止，不自动进入下一 H 阶段。

同一问题最多一次初始实现和两次有实质差异的修复；连续两次无进展即停止。

## 4. 文件与归档

- 当前任务可修改必要的非冻结代码、配置、测试和文档。
- 使用 `rg`/`rg --files` 搜索，文本编辑使用 `apply_patch`。
- 保留无关改动；遇到来源不明且重叠的修改时停止。
- 不做计划外重构、重复实现、假数据、空实现或用删除测试掩盖失败。
- 新行为必须有测试；大范围修改分批验证。
- 临时输出进入临时目录；冻结后按 `docs/EVIDENCE.md` 归档。
- 归档前记录原路径、原因和恢复位置；优先移动，不改写冻结 Evidence。
- 归档后更新 `docs/EVIDENCE.md` 或归档 README。

## 5. 代码与安全

- 不硬编码 CARLA host；使用 `python scripts/sdf.py sim ...`。
- 不复制 gateway、不创建第二 tick master、不直接调用 `world.tick()`。
- 正式 collector/实验使用 `ScenarioRuntime` 持有 tick；ROS `carla_sync_driver` 只允许作为互斥的
  G0/bridge bring-up 模式。两者不得在同一 CARLA endpoint 同时运行，任务开始前必须解析唯一 owner。
- Guard 必须逐候选执行，World 只能接收通过 Guard 的候选。
- World 不是安全真值；它不能修改 slack、MRM、Emergency 或回退权限。
- 候选生成、Guard、World、Safety、控制器的 provenance 必须可追踪。
- 不执行未授权任意 shell，不修改冻结 split/阈值、系统配置、防火墙、凭据或 Git 全局配置。

## 6. Git

- 开始前检查分支和状态，不回滚、覆盖或删除用户无关改动。
- 禁止 `git reset --hard`、`git clean -fd`、force checkout/rebase/push。
- 默认不 commit、不 push、不合并；只有用户明确要求时执行。
- 不在 `main` 做无人值守代码开发；用户明确授权的纯文档/整理除外。

## 7. 验证

任务文件中的命令优先；否则选择最小相关集合：

```text
python scripts/sdf.py doctor
python scripts/sdf.py sim preflight --json
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry
colcon build --symlink-install
git diff --check
```

真实 CARLA 任务只有 preflight `READY` 才继续；`RETRYABLE_FAILURE` 只 ensure 一次再
复查一次；`NEEDS_USER_ACTION`、版本冲突、tick 冲突或依赖冲突立即停止。离线文档、
schema、单测和训练不强制 preflight。报告只能写实际运行过的命令与结果。

## 8. Evidence 与停止

证据状态固定为：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

只有 `VERIFIED` 数字可无保留引用。必须保留失败、负收益、尾延迟、deadline miss、
资源与限制。达到验收、停止点、两次修复无进展、外部权限/GUI/登录阻塞、测试环境
不可靠、未知重叠修改、破坏性 Git、实车/生产/密钥范围或活动文档冲突时立即停止。

完成报告必须包含状态、修改、测试、未解决问题和用户接管事项。
