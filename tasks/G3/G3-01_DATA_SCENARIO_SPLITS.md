# G3-01：场景身份、VLA 数据契约、冻结划分与泄漏审计

**状态**：IMPLEMENTED / NOT_VERIFIED
**依赖**：G2-05
**阶段角色**：必做（G3 起点）
**一句话**：先把身份、输入分层和 split 钉死，再谈 VLA 训练与闭环。

## 启动读取清单

启动本任务前，除 `START_TASK.md`、`PROGRESS.md`、`ROADMAP.md` 对应阶段和本任务全文外，按顺序读取：

1. `docs/project/PROJECT_SUCCESS_PROFILE.md` 全文；
2. `docs/project/LOCAL_ASSETS.md`（路径 + **`/home/sdf/.venvs/sdf` venv 约定**）；
3. `docs/project/EXECUTION_ARCHITECTURE.md` 第 2～8 节；
4. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 3、5、8、12 节；
5. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1～4、6～8 节；
6. `docs/project/SDF_VLA_1B_DESIGN.md` 数据与许可相关小节；
7. G2-05 发布接口、SafetyEvent/执行结果 schema、最终断点与 Evidence。

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 项目成功口径（本任务）

- 本阶段不追求驾驶分数，只保证后续 VLA/World **共用同一套 frame/run 身份**。
- 数据可少而干净；禁止泄漏到 Regression。
- 许可与 `deployment_scope` 必须可追溯（仿真研究默认可）。

## 目标

在任何 VLA 训练/大规模采集前，统一：

- `run_id` / `frame_id` / `scenario_id` / `attempt_id` 等身份；
- 策略输入 vs 特权标签 vs 仅评测字段；
- ID / OOD / Regression 冻结划分与泄漏审计；
- 最小 `scenario` 元数据（family、parameter_hash），避免 G4 返工。

## 实现范围与边界

### 必做

- 对齐前视图像、Ego（及低维 history 占位）、Route/导航、可选语言标签、Classic 候选（标签/对照用）、Safety/执行结果；
- 四层字段：`policy_input`、`privileged_label`、`evaluation_only`、`regression_frozen`；
- 存储：Parquet 或 DuckDB 分片 + content hash + 断点续采；
- 按 Town / Route / 场景族 / 天气 / 失败簇划分 train/val/test 与 Regression；
- 数据卡：来源、许可、Oracle 字段列表、容量上限。

### 明确不做

- 不实现 VLA 模型、不训练、不接 World；
- 不下载完整上游大数据集（仅预留 manifest 与最小样本策略）；
- 不改 G2 Safety 阈值；不新建第二 tick master。

## 接口与交付物

| 类型 | 内容 |
|---|---|
| Schema | Observation/样本/split/manifest 版本化定义 |
| 代码路径 | `safedrive_foundry/data_pipeline/vla`、`collector`、`schema`、最小 `scenario/schema`、`registry` |
| 测试 | `tests/g3` 身份一致性、泄漏拒绝、hash 稳定 |
| 文档 | 数据卡 + 本任务断点；更新 `PROGRESS.md` |

## 完成标准与验证

### 最小通过（稳定/契约）

- 固定 seed 下样本摘要与 hash 可复现；中断续采不重复分片。
- 故意注入：跨 split 泄漏、错帧、Regression 写入、近重复 → **必须拒绝**。
- 单元/契约测试通过；无真实 CARLA 也可完成本任务主体。

### 诚实记录

- 语言标签若启用，仅限 behavior / critical_actor / conflict / risk_horizon / intended_action 等受限集合。
- 记录当前磁盘配额与预计 VLA 活跃工作集上限。

### 建议验证命令

```text
python3 -m unittest discover -s tests/g3 -t . -v
# 若任务引入 compile 目标：
python3 -m compileall -q safedrive_foundry/data_pipeline
```

## 允许修改

相对仓库根：`safedrive_foundry/` 下本任务直接需要的 data/schema/registry/config、`tests/g3`、`docs/`（非 G0 冻结证据）、本任务文件、`PROGRESS.md`。
不得提前实现 G3-02 以后的模型训练或 World。涉及真实 CARLA 采集时先 `sdf sim preflight`。

## 停止条件

验收通过即停；依赖接口不清、磁盘策略冲突、需改 G2 冻结契约 → 记录 `DECISION_REQUIRED` 并停。

## 断点记录

**2026-07-14 实现落地（后被复审回退）**
- 实现：`safedrive_foundry/data_pipeline/vla/*`（schema/split/leakage/store/datacard）
- 测试：`tests/g3/test_g3_01_data_contracts.py`（单测绿 ≠ 任务 VERIFIED）
- 证据：`docs/architecture/evidence/g3-01/datacard.json`

**2026-07-16 验收复审 → IMPLEMENTED / NOT_VERIFIED**
缺口：
1. 存储为 JSONL，任务要求 Parquet 或 DuckDB。
2. `near_dup_hash_prefix` 未使用，仅精确 payload hash 去重，非近重复。
3. split 分桶 hash 未包含 `failure_cluster`（datacard 声明 axes 含该项）。
不得将本任务标为 COMPLETED，直至缺口关闭并重验。
