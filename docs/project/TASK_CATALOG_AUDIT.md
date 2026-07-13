# 任务目录与管理文档一致性审计

审计日期：2026-07-13（Asia/Singapore）
审计范围：`ROADMAP.md`、`PROGRESS.md`、`START_TASK.md`、`FUTURE_PROJECT_VISION.md`、`CLAIMS.md`、活动 `tasks/G0`～`tasks/G8`、维护检查器与维护测试。
执行边界：用户授权重构 G2～G8 路线和全部相关任务；G0 冻结，G1 已关闭且实现/证据不改，本轮未启动任何 G2 实现。

## 1. 任务统计

| 阶段 | 活动任务数 | 状态摘要 |
|---|---:|---|
| G0 | 6 | 6 项 `COMPLETED / FROZEN` |
| G1 | 9 | G1-01～G1-09 已关闭，具体限制见 `PROGRESS.md` |
| G2 | 5 | 5 项 `PENDING` |
| G3 | 5 | 5 项 `PENDING` |
| G4 | 5 | 5 项 `PENDING` |
| G5 | 5 | 5 项 `PENDING` |
| G6 | 5 | 5 项 `PENDING` |
| G7 | 3 | 3 项 `PENDING` |
| G8 | 5 | 5 项 `PENDING` |
| **合计** | **48** | ROADMAP 48 行、活动文件 48 个、唯一 ID 48 个 |

目标检查结果：`roadmap_rows=48`、`active_task_files=48`、`roadmap_unique_ids=48`、`active_unique_ids=48`，每阶段数量为 `6/9/5/5/5/5/5/3/5`；实际命令结果见第 5 节。

## 2. 一致性修复

### 2.1 清单、重复与归档

- G0/G1 共 15 项保持不变；G2～G8 从原 42 项去冗余后定稿为 33 项，全项目 48 项均有唯一路线条目。
- 活动目录中没有缺失任务、重复 ID、错误文件名、错误目录或正文首标题 ID 不一致；因此没有移动文件，也没有创建 `archive/task_duplicates/`。
- 检查器明确排除路径中含 `archive` 的 Markdown；相同内容重复会报告 `identical`，内容不同重复会报告 `DECISION_REQUIRED`，不会静默合并。

### 2.2 依赖

- 48 个任务文件的直接依赖必须与 ROADMAP 逐项相等。
- 无不存在依赖、无自依赖、无依赖环、无跨阶段错误、无未来任务反向依赖。
- 重构 G2～G8 直接依赖：Safety→VLA→场景/反事实→World→后训练→研发助手→四配置准出；VLA 和 World 保持核心必做。

### 2.3 任务结构

- G2～G8 的 33 个任务统一包含启动读取清单、目标、实现边界、完成标准、验证、允许路径、交付物和断点；目标频率均明确为待实测门禁，不是结果。
- 删除重复的数据/验收拆分、CMA-ES 与多套 QD 并行、PPO 前置和五 Agent 角色；强化 Fast/Slow VLA、动作条件结构化 World、两级 Safety 和确定性准出。
- 将纵向 QP/RATO-SCP、反事实实现/搜索验收、抗遗忘/飞轮验收、统计/Evidence/最终准出拆成独立可验证边界，使 33 项仍适合逐项单机执行。
- `START_TASK.md` 包含全部 48 个文件路由；检查器要求每个活动任务都被入口覆盖，全部 G2～G8 任务具有 `启动读取清单`，且清单引用的 `docs/*.md` 真实存在。
- G0 任务未按统一模板重写，因为 G0 任务、配置、代码、README 和历史证据已冻结。

### 2.4 路径

- 活动 G1～G8 任务中的实现路径统一到真实工程根 `safedrive_foundry/` 下。
- 配置路径统一为现有的 `safedrive_foundry/config/`；ROS 工作区统一为现有的 `safedrive_foundry/ros_ws/`。
- 未来尚未实现的 `classic_stack`、`safety_kernel`、`driving_vla`、`world_model`、`scenario_search`、`counterfactual`、`data_pipeline`、`agents`、`validation`、`registry` 和 `artifacts` 仅在任务允许修改路径中归一化到 `safedrive_foundry/`，没有为了匹配文档创建目录。
- 冻结的 `FUTURE_PROJECT_VISION.md` 工程树、G0 文档和 G0 历史证据未改写；其中的旧目录/环境记录作为冻结历史保留，当前活动任务以本审计后的路径为准。

## 3. 状态与共享规则

- **现行（2026-07-13，以 `PROGRESS.md` 为准）**：G1-01～G1-09 已关闭，G2 未启动；固定推荐口令为 `读取 START_TASK.md，启动 G2-01。`。
- `AGENTS.md` 已明确验收任务的小型集成缺陷与实质缺陷边界，要求实质缺陷退回原任务并重验受影响结果。
- `AGENTS.md` 已明确任务状态与单次运行状态分层，`RETRYABLE_FAILURE` 不得直接等同于 `BLOCKED_EXTERNAL`，并保留 G1-02 连接与生命周期加固。
- 已补充统一连接/预检、无硬编码 CARLA host、无独立 gateway 解析、无第二 tick master、业务节点不得直接 `world.tick()`、Agent 不得任意 shell、学习模块不得覆盖 Safety 硬约束，以及 MCP 保留至 G7-01 的规则。
- G0 历史 `172.30.80.1` 未修改；后续任务使用 `sdf sim preflight` 动态 host（含 loopback / 非代理网关）。

## 4. 仍需决策的问题

1. `FUTURE_PROJECT_VISION.md` 是冻结原始总需求，其中工程树仍保留 `configs/`、`ros2_ws/` 等原始目录示意；如需修改冻结总需求，必须由用户单独给出范围决策。本轮使用真实现有路径修复活动任务，不创建兼容目录。
2. 本轮未发现需要移入 `archive/task_duplicates/` 的重复文件；若未来出现内容不同的活动重复文件，必须先完成 `DECISION_REQUIRED`，不能自动合并。
3. G0 历史任务文件中仍存在旧的结构表达和历史环境路径，这是冻结限制下的已知问题，不影响活动任务目录检查结果。

## 5. 修改文件与验证

本轮路线重构修改：

- 管理文档：`START_TASK.md`、`ROADMAP.md`、`PROGRESS.md`、`FUTURE_PROJECT_VISION.md`、`docs/project/CLAIMS.md`、`docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md`、`docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md`。
- 活动任务文档：重写 `tasks/G2`～`tasks/G8` 全部任务文件；G0/G1 任务、实现和证据未写入。
- 自动检查：`scripts/maintenance/task_catalog_check.py`。
- 自动测试：`tests/maintenance/__init__.py`、`tests/maintenance/test_task_catalog_check.py`。
- 审计报告：本文件。

已运行并通过：

```text
python3 scripts/maintenance/task_catalog_check.py       PASS
python3 scripts/maintenance/task_catalog_check.py --json PASS; ok=true, errors=[]
python3 -m unittest discover -s tests/maintenance -t . -v  PASS; 19/19
python3 -m py_compile scripts/maintenance/task_catalog_check.py PASS
python3 -m compileall -q scripts/maintenance tests/maintenance PASS
git diff --check                                      PASS
```

工作树状态以本轮结束时的终端输出为准；不包含任何 Git 提交。
