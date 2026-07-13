# SafeDrive Foundry 代理操作规范

本文件适用于仓库根目录及其子目录，是长期有效的代理行为规范。它约束代理如何读取任务、修改文件、验证结果和停止工作；不替代具体任务的验收标准。

## 1. 项目背景与真实边界

SafeDrive Foundry 是基于 CARLA–ROS 2 的纯软件在环（SIL）自动驾驶研发平台，目标是把经典专家、轻量驾驶 VLA、动作条件世界模型、独立 Safety Kernel、反事实场景搜索和可审计 Agent 组织成可验证闭环。

当前已知技术边界：

- CARLA Server 运行在 Windows；ROS 2、客户端和研发组件运行在 WSL2 Ubuntu。
- 固定单机资源为 RTX 4080 16GB、Intel i5-13600KF；不得把服务器、多 GPU 或未验证的外部服务当作必需依赖。
- 经典规划与控制承担基线、专家标签、Shadow、修复、回退和安全下限；学习策略不能绕过 Validator、Safety Kernel、MPC/PID 或紧急降级。
- 端到端/VLA 输出轨迹或动作块，不直接获得未经约束的车辆控制权；世界模型用于动作条件预测和候选排序，不是安全真值；Agent 不进入实时控制环。
- G0 已验收冻结。除非用户明确要求修复 G0 缺陷，否则不得修改 G0 任务、配置、代码和历史证据。

## 2. 文档职责与优先级

各文档职责如下：

1. `AGENTS.md`：长期不变的开发规则和代理行为规范。
2. `START_TASK.md`：当前唯一允许执行的任务和停止点。
3. `tasks/`：具体任务要求、验收条件和实施约束。
4. `ROADMAP.md`：里程碑顺序和中期规划，不代表当前全部要做。
5. `FUTURE_PROJECT_VISION.md`：长期设想，只用于理解方向，不得据此提前开发。
6. `PROGRESS.md`：只记录已经确认完成的事实，不作为需求来源。

冲突处理顺序为：

```text
用户本轮明确指令 > START_TASK.md > 对应 tasks/ 任务文件
> AGENTS.md > ROADMAP.md > FUTURE_PROJECT_VISION.md
```

若文档之间存在实质冲突，停止实施并报告冲突位置、影响和需要用户决定的事项，不得自行选择一种解释。用户本轮明确要求的文件范围同样必须遵守；本轮未授权的文件不得修改。

## 3. 单任务执行循环

每轮严格执行以下顺序：

1. 从用户指令或 `START_TASK.md` 确认唯一、最小、未完成步骤。
2. 明确目标、相关文件、允许/禁止修改范围、验收条件、测试命令和停止点。
3. 实现目标并运行相关测试。
4. 检查真实 `git diff`、`git diff --stat` 和测试结果，不只相信文字总结。
5. 验收失败时只处理具体修复项；修复后重新验证。
6. 当前步骤通过或达到停止点后立即停止，不自动开始下一步骤。

每个步骤最多 1 次初始实现和 2 次修复。同一问题连续两次没有实质进展时立即停止并报告。

## 4. 每轮读取与上下文控制

固定先读取：

- `START_TASK.md`（若用户要求按入口执行）；
- `PROGRESS.md`；
- 唯一匹配的当前任务文件 `tasks/GX/GX-XX_*.md`。

按需读取任务直接引用、直接依赖的最终产物，以及相关接口/配置/断点。仅在阶段切换或依赖不清时读取 `ROADMAP.md` 对应章节；仅在需求冲突、重大范围变更或用户明确要求时读取 `FUTURE_PROJECT_VISION.md` 对应章节。不得为了保险通读整个仓库、全部任务或所有管理文档；已确认且未变化的内容不重复读取。

只读取必要路径、约束和验收条件，不复制无关聊天历史。若预计扫描范围很大、修改超过 20 个文件或有效改动约超过 1500 行，先停止并报告范围。

## 5. 代码、文件与安全规则

- 只修改当前任务允许的路径，保留用户已有和无关改动；发现不明来源工作区改动时停止并报告。
- 遵循现有架构、命名、类型、格式和模块边界，禁止计划外重构、重复实现、临时代码、假数据和空实现。
- 新行为必须有对应测试；不得删除、跳过、弱化或改写测试来掩盖失败。
- 不确定公共接口是否允许变化时保持兼容并停止询问；不得覆盖用户无关修改。
- 不删除用户数据、归档或冻结证据，不擅自升级冻结版本，不修改系统配置、防火墙或 WSL 安全设置。
- 后续业务不得硬编码 CARLA host、复制 G0 历史 gateway、创建第二套 `carla.Client`/tick master 或直接调用 `world.tick()`；统一使用已验证的 `sdf sim` 入口。
- Agent 只能调用任务明确允许的白名单工具，不得执行任意 shell/Python；MCP 相关实现保留至 G7-01。
- 学习模块只能调整规划软风险代价和允许的模型输出，不能覆盖 Safety 硬约束、slack 上限、回退权限或紧急动作。
- Oracle 与 Observable 结果不得混用，训练/测试/回归集不得泄漏；预期指标不得写成真实结果。

## 6. Git 规则

- 开始工作前检查 `git status --short` 和当前分支；记录不明改动。
- 不在 `main` 分支进行无人值守开发；如需代码开发，应由用户或工作流明确允许分支策略。
- 禁止 `git reset --hard`、`git clean -fd`、强制 checkout、rebase、force push 及其他破坏性操作。
- 默认不创建提交、不 push、不合并 `main`；只有 `START_TASK.md` 或用户明确允许时才创建本地提交。
- 不修改 Git 全局配置，不回滚或删除不是当前代理产生的更改。

## 7. 验证与真实命令

任务文件中的验证命令优先。若任务文件没有明确命令，先从仓库现有脚本、配置和历史证据确认，再运行与改动直接相关的最小验证；不得猜测或编造命令。

仓库中已确认存在、可按任务需要使用的命令包括：

```text
sdf doctor
sdf sim preflight
sdf sim ensure
colcon build --symlink-install
python3 -m unittest discover -s tests -t . -v
python3 -m compileall -q <已确认的代码路径>
python3 scripts/sdf.py validate-g0
git diff --check
```

具体任务可要求其他命令；必须先在任务、脚本或配置中确认其存在。报告测试时写明实际执行的完整命令、通过/失败/未运行、失败原因和客观限制；没有运行的测试不得写成通过。

涉及真实 CARLA 的任务必须先执行：

1. `sdf sim preflight` 返回 `READY` 才能继续；
2. `RETRYABLE_FAILURE` 时只允许执行一次 `sdf sim ensure`，成功后只重新执行一次 preflight；
3. `needs_user_action=true` 且 `error_code=NEEDS_USER_ACTION` 时标记 `BLOCKED_EXTERNAL` 并停止；
4. 版本不匹配、tick owner 冲突或依赖冲突标记 `BLOCKED` 或 `DECISION_REQUIRED`，不得自动换版本。

连接记录应保留 `error_code`、`retry_count`、`previous_host`、`resolved_host`、`recovery_action` 和 `recovery_result`（若该连接层提供这些字段）。

## Windows 与 WSL 执行环境

- Codex 主进程可能运行在 Windows，但项目的 Linux 工具链位于 WSL2 的 `Ubuntu-24.04`。
- Windows 项目路径 `E:\autonomous driving` 对应 WSL 路径 `/mnt/e/autonomous driving`。
- Linux 命令必须通过以下形式执行：
  `wsl -d Ubuntu-24.04 -- /usr/bin/bash -lic '<command>'`
- PowerShell 中找不到某个 Linux 命令，只代表该命令不在 Windows PATH 中，不得据此判断 WSL 或 Linux 环境不存在。
- 需要 Shell 初始化、别名、函数或环境激活时，使用 `bash -lic`。
- 不得绕过项目规定的统一命令入口。入口无法找到时，先读取环境初始化文档并激活规定环境。

## 8. PROGRESS、断点与证据

只有验收且测试结果已确认的事实才能写入 `PROGRESS.md`。每次更新仅记录完成任务及内容、关键修改文件、实际测试结果、分支/提交、已知限制和下一停止点；不记录“计划完成”“大概可用”等结论。

任务证据状态统一为：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

只有 `VERIFIED` 结果可用于简历量化表述，必须保留 P50/P95/P99、deadline miss、资源、失败和限制；没有稳定收益时保留负结论，不得事后移动阈值。

中断、额度不足或未完成时，在任务末尾记录：时间、最后状态、已完成内容、修改文件、已运行验证、结果摘要、失败/阻塞、精确恢复步骤和下一条建议命令。恢复时先验证文件、版本和外部状态仍成立，不重复已验证工作。

## 9. 外部阻塞与停止协议

遇到以下任一情况，不得重复尝试、自动扩大范围或伪造结果：

- 需要管理员权限、UAC、系统服务、重启、GUI、登录、许可证接受或用户选择；
- 沙盒禁止访问/写入目标路径且没有项目范围内的安全替代方案；
- 网络、代理、凭据、外部服务或第三方工具不可用，重试不会改变外部状态；
- Codex 无法完成交互式、硬件、视觉确认或权限授予；
- 同一阻塞原因连续两次验证仍存在；
- 继续操作可能影响用户数据、冻结证据、系统配置或项目范围外资源。

处理顺序：做一次最小只读诊断，记录错误原文、完整命令、环境和尝试次数；将任务标记为 `BLOCKED_EXTERNAL`、`BLOCKED` 或 `DECISION_REQUIRED`；在任务断点和 `PROGRESS.md` 写出用户需要执行的最小动作；立即停止，不自动切换路径、不扩大权限、不开始下一任务、不循环重试。只有用户完成外部动作或明确给出新授权/决策后，才能恢复同一任务。

## 10. 停止条件与完成定义

满足以下任一条件必须停止并报告：当前验收通过；达到 `START_TASK.md` 停止点；同一问题修复两次仍失败；需要改变架构或超出允许目录；测试环境无法可靠运行；出现未知改动；需要破坏性 Git 操作；涉及真实车辆、生产环境、密钥或高风险权限；任务或文档存在实质冲突。

停止报告必须包含：

1. 当前完成状态；
2. 修改文件；
3. 实际测试命令和结果；
4. 未解决问题；
5. 需要用户决定或手动接管的事项。

只有在验收条件全部满足、实际测试通过、范围外差异可解释、测试标准未降低、没有临时实现、`PROGRESS.md` 与真实状态一致且未自动 push/合并 main 时，才能声明当前步骤完成。

## 11. 仓库结构速览

以下是当前已存在且与执行相关的目录；新增目录必须由任务授权：

```text
tasks/G0..G8/       分阶段任务文件
tests/              Python 测试与 live/maintenance 验证入口
scripts/            sdf 入口、CARLA smoke 与维护脚本
tools/              Python 3.12 embedded runtime 与 CARLA 0.9.16 wheel
safedrive_foundry/  ROS 2/运行时/桥接/配置等项目代码
docs/               架构、环境、运行时和证据文档
archive/            归档内容，默认只读
versions.lock       版本锁定信息
START_TASK.md       任务启动路由
ROADMAP.md          阶段顺序与任务依赖
PROGRESS.md         已确认进度事实
FUTURE_PROJECT_VISION.md  冻结的长期总设想
```

本速览不构成任务授权；实际允许修改范围、依赖和验收条件始终以当前任务文件和 `START_TASK.md` 为准。
