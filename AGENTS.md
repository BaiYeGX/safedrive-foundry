# SafeDrive Foundry 代理操作规范

本文件约束仓库内所有代理行为。它只保留长期规则；当前任务见 `START_TASK.md`，
未来顺序见 `ROADMAP.md`，动态事实见 `PROGRESS.md`。

## 1. 项目边界

- 项目是 CARLA–ROS 2 纯软件在环（SIL），不是实车或公共道路系统。
- CARLA Server 在 Windows；ROS 2、客户端、模型和训练在 WSL2 Ubuntu。
- 固定硬件为 RTX 4080 16GB 和 i5-13600KF；不得假设服务器或第二张 GPU。
- 核心研究链是 `VLA K2 → Contract Guard → World on/off → MPC/PID`。
- VLA 只提出轨迹，World 只排序，模型不得直接获得无约束底盘控制或 tick 权限。
- Classic 是基线/标签/可选 Hybrid；完整 Safety 是独立工程扩展。
- G0、冻结 Evidence 和归档内容默认只读；只有用户明确要求修复或整理时才能移动，
  且必须保留可恢复副本。

## 2. 文档优先级

```text
用户本轮明确指令
> START_TASK.md
> AGENTS.md
> ROADMAP.md
> docs/PROJECT.md
> 其他设计文档
> archive 中的历史文档
```

活动权威文档：

| 文件 | 职责 |
|---|---|
| `README.md` | 项目入口与目录 |
| `START_TASK.md` | 当前唯一任务、验收和停止点 |
| `ROADMAP.md` | 后续最短路线 |
| `PROGRESS.md` | 已确认动态事实 |
| `docs/PROJECT.md` | 系统边界、成功口径和证据合同 |
| `docs/R1_REAL_K2.md` | 当前 R1 的代码分析、实现设计和分层验收 |
| `docs/VLA.md` | VLA/K2 设计 |
| `docs/WORLD_MODEL.md` | G4A/G5 World 设计 |
| `docs/RESOURCES.md` | 本机资产与资源预算 |
| `docs/ENVIRONMENT.md` | 环境与运行入口 |
| `docs/G3_BASELINE.md` | 当前 K1 基线运行说明 |
| `docs/EVIDENCE.md` | Evidence 与归档索引 |

archive 只保存历史，不作为新需求来源。若活动文档冲突，停止并报告。

## 3. 单任务循环

每轮严格执行：

1. 读取 `START_TASK.md`、`PROGRESS.md` 和当前任务直接引用的核心文档。
2. 检查当前分支和 `git status --short`，识别用户已有改动。
3. 明确目标、允许路径、验收、验证命令和停止点。
4. 只实现当前任务。
5. 运行直接相关的最小测试。
6. 检查 `git diff`、`git diff --stat` 和 `git diff --check`。
7. 更新 `PROGRESS.md` 后停止，不自动进入 ROADMAP 下一阶段。

同一问题最多 1 次初始实现和 2 次有实质差异的修复。连续两次无进展即停止。

## 4. 范围与文件

- 仓库不设置单轮文件数或改动行数上限。
- 当前任务可修改所有必要的非冻结代码、配置、测试和文档，无需逐目录追加授权。
- 大范围修改必须分批验证，但规模本身不是停止条件。
- 使用 `rg`/`rg --files` 搜索；简单文件编辑使用 `apply_patch`。
- 保留用户无关改动；与当前目标重叠且来源不明时停止。
- 不做计划外重构、重复实现、临时代码、假数据或空实现。
- 新行为必须有测试；不得删除或弱化测试掩盖失败。
- 临时运行输出不得长期散落根目录：任务中写临时目录，冻结后按 `docs/EVIDENCE.md`
  移入 archive。

## 5. Archive 规则

- archive 是本机可恢复历史区，不是活动代码或活动文档目录。
- 归档前确认目标、原路径、原因和恢复位置。
- 优先移动而不是删除；不得改写冻结 Evidence 内容。
- 归档完成后更新 `docs/EVIDENCE.md` 或对应 archive README。
- 不从 archive 自动恢复旧路线、旧阈值或旧任务。

## 6. 代码与安全

- 不硬编码 CARLA host；使用 `python scripts/sdf.py sim ...`。
- 不复制 G0 gateway、创建第二 tick master 或直接调用 `world.tick()`。
- Oracle 与 Observable 严格隔离；Regression 不进入训练。
- World 不能成为安全真值，不能覆盖 Safety 硬约束。
- Safety 扩展启用后，学习模块不能修改 slack、MRM、Emergency 或回退权限。
- Agent 不进入实时控制环，不执行未授权任意 shell，不修改冻结 split/阈值。
- 不修改系统配置、防火墙、凭据或 Git 全局配置。

## 7. Git

- 开始前检查分支与状态。
- 不回滚、覆盖或删除用户无关改动。
- 禁止 `git reset --hard`、`git clean -fd`、force checkout/rebase/push。
- 默认不 commit、不 push、不合并；只有用户明确要求时执行。
- 不在 `main` 做无人值守代码开发；纯文档/仓库整理按用户明确授权执行。

## 8. 验证

任务文件中的命令优先；没有时使用已存在的最小相关命令：

```text
python scripts/sdf.py doctor
python scripts/sdf.py sim preflight --json
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry
colcon build --symlink-install
git diff --check
```

报告必须写实际命令、通过/失败/未运行和原因。没有运行的测试不得写通过。

真实 CARLA 任务：

1. preflight `READY` 才继续；
2. `RETRYABLE_FAILURE` 只 ensure 一次，再 preflight 一次；
3. `NEEDS_USER_ACTION` → `BLOCKED_EXTERNAL` 并停止；
4. version/tick/dependency conflict → `BLOCKED` 或 `DECISION_REQUIRED`；
5. 不需要 CARLA 的文档、schema、单测和训练不强制 preflight。

## 9. Evidence 与进度

只有确认事实才能进入 `PROGRESS.md`。证据状态：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

只有 `VERIFIED` 数字可作无保留量化表述。必须保留失败、负收益、P50/P95/P99、
deadline miss、资源和限制。

中断时记录：

- 当前状态；
- 已完成内容；
- 修改文件；
- 已运行验证与结果；
- 阻塞；
- 精确恢复步骤；
- 下一条命令。

## 10. 停止条件

当前验收通过、达到 START_TASK 停止点、两次修复无进展、需要外部权限/GUI/登录、
测试环境不可靠、出现重叠未知改动、需要破坏性 Git 操作、涉及实车/生产/密钥或活动
文档冲突时，立即停止并报告。

完成报告必须包含状态、修改、测试、未解决问题和需要用户接管的事项。
