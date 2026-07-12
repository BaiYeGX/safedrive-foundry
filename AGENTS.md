# Codex 单任务执行协议

本文件适用于项目根目录及全部子目录，是 Codex 执行规则、任务标准和通用证据规则的唯一来源。

`FUTURE_PROJECT_VISION.md` 是冻结的原始总需求；`ROADMAP.md` 是任务编号和依赖的唯一来源；`PROGRESS.md` 是当前执行状态的唯一来源。G0 已验收冻结，除非用户明确要求修复 G0 缺陷，否则不得修改 G0 任务与证据。

用户通过根目录`START_TASK.md`启动指定任务；该文件只负责定位和按需读取路由，不覆盖本协议。

## 1. 单任务原则

1. 每次对话只能执行用户明确指定、或 `PROGRESS.md` 标记为当前任务的一项任务。
2. 未完成当前任务时不得自动开始下一任务。
3. 完成后必须验证、更新任务断点和 `PROGRESS.md`，随后停止等待用户指令。
4. 前置依赖未完成时，将当前任务标记为 `BLOCKED`，不得绕过。
5. 任务是可独立验收的垂直能力切片，不得缩减为占位文件，也不得顺带实现后续任务。

## 2. 每轮读取规则

### 固定必读

1. `PROGRESS.md`；
2. 当前任务 `.md` 文件。

根目录 `AGENTS.md` 由 Codex 自动应用，不要求用户反复提示读取。

### 按需读取

- 当前任务列出的直接引用、接口、配置、断点和直接依赖任务的最终产物；
- 涉及跨组件接口、时间、身份、Profile、Oracle/Observable、Registry 或权限边界时，读取 `docs/project/EXECUTION_ARCHITECTURE.md` 对应章节；
- G1-04～G2-06读取 `docs/project/decisions/CLASSIC_ALGORITHM_OPTIMIZATION.md`；
- 阶段验收、消融、量化结论或简历证据任务读取 `docs/project/CLAIMS.md`；
- 只有阶段切换、依赖不清时读取 `ROADMAP.md` 对应阶段；
- 只有需求冲突、重大范围变更或用户明确要求时读取 `FUTURE_PROJECT_VISION.md` 对应章节。

禁止为了“保险”每轮通读全部管理文档或所有前序任务。

## 3. 任务文件最低标准

任务开始前必须能明确找到以下语义；章节允许合并：

1. 状态与直接依赖；
2. 目标及当前执行原因；
3. 输入事实和前置产物；
4. 范围与明确不做；
5. 允许修改路径；
6. 交付物；
7. 可测量完成标准；
8. 验证命令、场景或指标；
9. 资源与自动化边界；
10. 断点记录。

缺少允许修改范围、交付物或可测量标准时标记 `DECISION_REQUIRED`，不得自行扩大范围。

## 4. 任务粒度

一个任务通常同时包含实现、测试、最小真实运行和证据，但只解决一个核心研究或工程问题。

必须继续拆分的情况：

- 同时训练两个不同核心模型；
- 同时实现两套无共享求解结构的算法；
- 既设计数据协议又完成大规模训练；
- 验收依赖尚未实现的后续模块；
- 中断后无法精确描述恢复点。

不应拆分的情况：

- 单个小函数无法独立证明能力；
- 文档与对应实现/验证被人为拆开；
- 同一接口的实现和单元测试被拆成两个任务。

## 5. 执行边界

- 只修改当前任务允许的文件；保留用户已有和无关改动。
- 不删除用户数据、不覆盖归档、不擅自升级冻结版本。
- 管理员、GUI、重启、登录、许可或其他外部步骤标记 `BLOCKED_EXTERNAL`，并给出最小操作说明。
- 需求冲突或重大方案选择标记 `DECISION_REQUIRED`。
- 不把预期指标写成真实结果，不因文件存在就判定完成。
- Oracle和Observable结果不得混用；训练、测试与回归集不得泄漏。
- 规划软风险代价不能修改Safety硬阈值、slack上限或回退权限。

## 6. 完成与证据规则

证据状态统一为：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

- `IMPLEMENTED`：实现存在且规定测试通过；
- `MEASURED`：冻结协议下产生真实结果；
- `VERIFIED`：结果、统计、产物链接、限制和复核完整。

任务只有在以下条件全部满足时才能标记 `COMPLETED`：

1. 规定测试和真实门禁通过；
2. 代码、配置、数据、run和输出路径已登记；
3. 命令、结果、资源、失败和限制已记录；
4. 断点记录完整；
5. `PROGRESS.md` 已更新；
6. 未自动开始下一任务。

只有 `VERIFIED` 结果可以进入简历量化表述。必须报告P50/P95/P99、deadline miss和资源，不能用平均值掩盖尾部退化；没有稳定收益时保留负结论，不得改写基线或事后移动阈值。

## 7. 断点记录

中断、额度不足或尚未完成时，任务文件必须记录：

| 字段 | 内容 |
|---|---|
| 时间 | 待填写 |
| 最后状态 | PENDING |
| 已完成 | 无 |
| 修改文件 | 无 |
| 已运行验证 | 无 |
| 结果摘要 | 无 |
| 失败或阻塞 | 无 |
| 精确恢复步骤 | 校验依赖与外部状态后继续 |
| 下一条建议命令 | 待填写 |

恢复时先验证断点中的文件、版本和外部状态仍成立，不重复已经验证的工作。

## 8. 外部阻塞与停止协议

遇到以下情况不得陷入重复尝试、反复重试或自动扩大范围：

- 明确需要管理员权限、UAC确认、系统服务、重启、GUI、登录、许可证接受或用户选择；
- 沙盒禁止访问/写入目标路径，且没有项目范围内的安全替代方案；
- 网络、代理、凭据、外部服务或第三方工具不可用，重试不会改变外部状态；
- Codex自身无法完成的交互式操作、硬件操作、视觉确认或权限授予；
- 同一阻塞原因连续两次验证仍然存在；
- 继续操作可能修改用户数据、冻结证据、系统配置或项目范围之外的资源。

处理顺序：

1. 做一次最小、只读的诊断，确认不是命令或路径错误；
2. 记录错误原文、命令、环境、已尝试次数和影响范围；
3. 将任务标记为 `BLOCKED_EXTERNAL`、`BLOCKED` 或 `DECISION_REQUIRED`；
4. 在 `PROGRESS.md` 和任务断点中写出用户需要执行的最小动作；
5. 立即停止，不自动切换替代方案、不开始下一任务、不循环重试。

只有用户完成外部动作或明确给出新授权/新决策后，才能恢复同一任务。不得把外部阻塞包装成任务完成，也不得因为额度不足或工具失败伪造验证结果。

## 9. 状态枚举

```text
PENDING
CURRENT
IN_PROGRESS
PAUSED
BLOCKED
BLOCKED_EXTERNAL
DECISION_REQUIRED
VALIDATING
COMPLETED
FAILED_FINAL
```

## 10. 任务状态与单次运行状态

任务状态和单次运行状态是两层，不得混用：

任务级状态：`PENDING`、`CURRENT`、`IN_PROGRESS`、`PAUSED`、`BLOCKED`、`BLOCKED_EXTERNAL`、`DECISION_REQUIRED`、`VALIDATING`、`COMPLETED`、`FAILED_FINAL`。

单次运行级状态：`PENDING`、`PRECHECK`、`RUNNING`、`FINALIZING`、`CLEANING_UP`、`VALIDATING`、`COMPLETED`、`RETRYABLE_FAILURE`、`CRASHED`、`CLEANUP_FAILED`、`FAILED_FINAL`。

- `RETRYABLE_FAILURE` 只表示一次运行遇到已有的安全、有限自动恢复路径；任务通常保持 `IN_PROGRESS`。
- `RETRYABLE_FAILURE` 不得直接改写为 `BLOCKED_EXTERNAL`；重试预算耗尽后按实际原因转为 `BLOCKED`、`BLOCKED_EXTERNAL` 或 `FAILED_FINAL`。
- `CRASHED`、`CLEANUP_FAILED` 或未完成 cleanup verification 的运行不得登记 `COMPLETED`。
- `BLOCKED_EXTERNAL` 只用于确需 UAC、GUI、登录、许可证、管理员权限、重启或物理操作的外部动作。

## 11. CARLA 任务预检与恢复

明确依赖真实 CARLA 的任务，统一通过已验证的 `sdf sim` 入口预检；业务代码和任务不得自行解析 gateway、复制历史 host、创建第二套 `carla.Client` 或自行重试。

1. 先运行 `sdf sim preflight`。
2. `READY` 继续当前任务。
3. `RETRYABLE_FAILURE` 先运行一次 `sdf sim ensure`，成功后只重新运行一次 `sdf sim preflight`。
4. `ensure` 返回 `needs_user_action=true` 且 `error_code=NEEDS_USER_ACTION`，才将任务标记 `BLOCKED_EXTERNAL` 并停止。
5. 版本不匹配、tick owner 冲突或依赖冲突标记 `BLOCKED` 或 `DECISION_REQUIRED`，不得自动换版本。
6. 同一故障不得无限重试；不得自动扩大权限、修改防火墙/WSL/系统安全设置或启动未知 CARLA 路径。

`RETRYABLE_FAILURE`、`BLOCKED_EXTERNAL`、`FAILED_FINAL` 是稳定的运行/任务语义；连接层必须同时记录 `error_code`、`retry_count`、`previous_host`、`resolved_host`、`recovery_action` 和 `recovery_result`。G0 冻结配置与历史证据中的地址仅作历史记录，不得被 G1 及后续代码复制。

## 12. 验收任务修复边界

验收任务允许修复小型集成缺陷，但边界必须按影响判断：

- **小型集成缺陷**：不改变接口、算法、阈值、数据划分或实验协议，且只修复使既定验收无法运行、记录或复现的接线、适配、清理、日志和报告问题；可在验收任务内修复并重新运行受影响验证。
- **实质缺陷**：改变接口、算法行为、安全阈值、数据划分、模型结构或冻结协议；必须退回原任务，更新原任务断点，并重新验证所有受影响结果，验收任务不得就地吸收。
- 任务级状态与单次运行级状态始终分层；`RETRYABLE_FAILURE` 只能表示有限、安全的运行级恢复路径，不得直接等同于 `BLOCKED_EXTERNAL`。
- 验收任务不得破坏上一轮 G1-02 已完成的连接和生命周期加固，包括统一连接解析、tick lease、生命周期终态、清理验证、`CRASHED`/`CLEANUP_FAILED` 登记和已有证据。

## 13. 共享运行规则

- G1 及后续任务不得硬编码 CARLA host、复制 G0 历史 gateway，或各自解析 WSL gateway；统一使用已验证的 `sdf sim` 连接/预检入口和版本化运行配置。
- 后续业务不得自行创建第二套 `carla.Client`、tick master 或 tick lease；业务节点不得直接调用 `world.tick()`。
- Agent 只能调用白名单工具，不得执行任意 shell、Python 或其他未治理命令。
- 学习模块只能调整规划软风险代价和允许的模型输出，不能覆盖 Safety 硬约束、slack 上限、回退权限或紧急动作。
- MCP 相关实现保留到 G7-01；当前阶段不实现 MCP，也不把 MCP/API 调用包装成算法交付物。
- `172.30.80.1` 只允许作为 G0 冻结配置、任务和历史证据中的环境记录；后续代码、配置和新任务证据不得复制该固定地址。
