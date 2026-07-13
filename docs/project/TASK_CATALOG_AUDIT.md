# 任务目录与管理文档一致性审计

审计日期：2026-07-12（Asia/Singapore）  
审计范围：`ROADMAP.md`、`PROGRESS.md`、`AGENTS.md`、`START_TASK.md`、活动 `tasks/G0`～`tasks/G8`、维护检查器与维护测试。  
执行边界：本轮只做目录、依赖、文档结构、路径和维护自检修复；未启动 G1-03 或其他路线任务。

## 1. 任务统计

| 阶段 | 活动任务数 | 状态摘要 |
|---|---:|---|
| G0 | 6 | 6 项 `COMPLETED / FROZEN` |
| G1 | 9 | 审计当日：G1-01、G1-02 `COMPLETED`；G1-03～G1-09 当时为 `PENDING`。**现行见 `PROGRESS.md`（G1-03 已完成）** |
| G2 | 6 | 6 项 `PENDING` |
| G3 | 7 | 7 项 `PENDING` |
| G4 | 6 | 6 项 `PENDING` |
| G5 | 6 | 6 项 `PENDING` |
| G6 | 6 | 6 项 `PENDING` |
| G7 | 5 | 5 项 `PENDING` |
| G8 | 6 | 6 项 `PENDING` |
| **合计** | **57** | ROADMAP 57 行、活动文件 57 个、唯一 ID 57 个 |

检查器结果：`roadmap_rows=57`、`active_task_files=57`、`roadmap_unique_ids=57`、`active_unique_ids=57`，每阶段数量为 `6/9/6/7/6/6/6/5/6`。

## 2. 一致性修复

### 2.1 清单、重复与归档

- ROADMAP 补回 G0-01～G0-06 明细，使 G0～G8 全部 57 项均有路线条目。
- 活动目录中没有缺失任务、重复 ID、错误文件名、错误目录或正文首标题 ID 不一致；因此没有移动文件，也没有创建 `archive/task_duplicates/`。
- 检查器明确排除路径中含 `archive` 的 Markdown；相同内容重复会报告 `identical`，内容不同重复会报告 `DECISION_REQUIRED`，不会静默合并。

### 2.2 依赖

- 57 个任务文件的直接依赖已与 ROADMAP 逐项相等。
- 无不存在依赖、无自依赖、无依赖环、无跨阶段错误、无未来任务反向依赖。
- 修正了 G8-03～G8-06 的路线依赖，使其与任务文件的并行回归、消融、证据和最终验收关系一致；没有改变阶段顺序或研究范围。

### 2.3 任务结构

- 对 G1-02、G1-03～G1-06、G1-09、G2～G8 的缺项做了最小补充；48 个非 G0 任务获得明确的 `交付物` 与 `资源与自动化边界` 语义，17 个缺少明确不做边界的任务补充了 `明确不做`。
- 目标、范围、允许修改路径、完成标准、验证方法、断点与恢复步骤均保留原任务语义；未新增算法、性能指标或实验结果。
- G0 任务未按统一模板重写，因为 G0 任务、配置、代码、README 和历史证据已冻结。

### 2.4 路径

- 活动 G1～G8 任务中的实现路径统一到真实工程根 `safedrive_foundry/` 下。
- 配置路径统一为现有的 `safedrive_foundry/config/`；ROS 工作区统一为现有的 `safedrive_foundry/ros_ws/`。
- 未来尚未实现的 `classic_stack`、`safety_kernel`、`driving_vla`、`world_model`、`scenario_search`、`counterfactual`、`data_pipeline`、`agents`、`validation`、`registry` 和 `artifacts` 仅在任务允许修改路径中归一化到 `safedrive_foundry/`，没有为了匹配文档创建目录。
- 冻结的 `FUTURE_PROJECT_VISION.md` 工程树、G0 文档和 G0 历史证据未改写；其中的旧目录/环境记录作为冻结历史保留，当前活动任务以本审计后的路径为准。

## 3. 状态与共享规则

- **审计当日（2026-07-12）**：`PROGRESS` 为 G1-01/G1-02 完成，推荐 G1-03 未启动。
- **现行（2026-07-13，以 `PROGRESS.md` 为准）**：G1-01～**G1-03 已完成**；推荐 **G1-04**；本审计报告不自动重跑检查器，状态以 `PROGRESS.md` 与任务文件为准。
- `AGENTS.md` 已明确验收任务的小型集成缺陷与实质缺陷边界，要求实质缺陷退回原任务并重验受影响结果。
- `AGENTS.md` 已明确任务状态与单次运行状态分层，`RETRYABLE_FAILURE` 不得直接等同于 `BLOCKED_EXTERNAL`，并保留 G1-02 连接与生命周期加固。
- 已补充统一连接/预检、无硬编码 CARLA host、无独立 gateway 解析、无第二 tick master、业务节点不得直接 `world.tick()`、Agent 不得任意 shell、学习模块不得覆盖 Safety 硬约束，以及 MCP 保留至 G7-01 的规则。
- G0 历史 `172.30.80.1` 未修改；后续任务使用 `sdf sim preflight` 动态 host（含 loopback / 非代理网关）。

## 4. 仍需决策的问题

1. `FUTURE_PROJECT_VISION.md` 是冻结原始总需求，其中工程树仍保留 `configs/`、`ros2_ws/` 等原始目录示意；如需修改冻结总需求，必须由用户单独给出范围决策。本轮使用真实现有路径修复活动任务，不创建兼容目录。
2. 本轮未发现需要移入 `archive/task_duplicates/` 的重复文件；若未来出现内容不同的活动重复文件，必须先完成 `DECISION_REQUIRED`，不能自动合并。
3. G0 历史任务文件中仍存在旧的结构表达和历史环境路径，这是冻结限制下的已知问题，不影响活动任务目录检查结果。

## 5. 修改文件与验证

本轮维护修改：

- 管理文档：`AGENTS.md`、`START_TASK.md`、`ROADMAP.md`、`PROGRESS.md`。
- 活动任务文档：`tasks/G1/G1-02`、`tasks/G1/G1-03`～`G1-09`，以及 `tasks/G2`～`tasks/G8` 全部任务文件；G0 文件未写入。
- 自动检查：`scripts/maintenance/task_catalog_check.py`。
- 自动测试：`tests/maintenance/__init__.py`、`tests/maintenance/test_task_catalog_check.py`。
- 审计报告：本文件。

已运行并通过：

```text
python3 scripts/maintenance/task_catalog_check.py       PASS
python3 scripts/maintenance/task_catalog_check.py --json PASS; ok=true, errors=[]
python3 -m unittest discover -s tests/maintenance -t . -v  PASS; 10/10
python3 -m py_compile scripts/maintenance/task_catalog_check.py PASS
git diff --check                                      PASS; 仅有既有 sdf.cmd LF→CRLF 提示
```

工作树状态以本轮结束时的终端输出为准；不包含任何 Git 提交。
