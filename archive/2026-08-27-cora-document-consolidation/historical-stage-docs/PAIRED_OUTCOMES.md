# H2 Paired Outcomes 合同

## 1. 范围与隔离

H2 只比较同一 observable anchor 上 H1 生成的 Expert/VLA 候选。live collector 不导入离线
Oracle；Oracle 不进入 CARLA、Guard、Safety 或控制进程。H3 仍未授权。

活动实现：

```text
safedrive_foundry/data_pipeline/h2/contracts.py       pair/branch/outcome 合同
safedrive_foundry/data_pipeline/h2/live_contract.py   live-only hash/指标辅助
safedrive_foundry/data_pipeline/h2/store.py           原子 Parquet/CAS/manifest
safedrive_foundry/data_pipeline/h2/oracle.py          仅离线标签
safedrive_foundry/data_pipeline/h2/quality.py         pilot/full 固定门
scripts/h2_collect.py                                 单地图物化、smoke、采集
scripts/h2_label_audit.py                             离线 label/audit
scripts/h2_run.py                                     三地图固定顺序编排
```

## 2. 固定矩阵

```text
maps     = Town01, Town03, Town05
families = free_flow, slow_lead, stopped_lead, cut_in, red_light_hold
seeds    = 0..3
weather  = ClearNoon, CloudyNoon
total    = 120
pilot    = seed=0 / ClearNoon = 15
```

矩阵由 `h2-fixed-matrix-v1` 确定性物化。Expert slot 和 branch order 都从 pair hash 排名
得到，完整矩阵严格 60/60。每张地图先冷重启，运行一个单 tick Runtime smoke，再物化 40
个物理场景；三部分使用同一 HEAD、tracked diff hash 和 untracked manifest hash 后，才冻结
`scenario_manifest.json/parquet`。代码、配置或 schema 变化必须使用新 dataset id。

## 3. Pair 流程

1. 清洁场景中生成 20 个 20 Hz observable history tick，最后一帧为 exact anchor；
2. Classic 与 nominal SimLingo 各生成一次，VLA forward count 必须等于 1；
3. 独立 Guard 后，只有两个 `PASS` 且 H1 标记 `DISTINCT` 才执行分支；
4. capture 清理后，按冻结 hash order 分别重建两个初态；
5. 重用 capture 的原始 candidate object、points 和 canonical hash，仅重绑 Safety
   run/frame/simulation time；
6. Safety 每次只接收被强制执行的一条候选；orphan、跨 source fallback 或不可解析执行
   id 都使分支无效；
7. 一个 `ScenarioRuntime` 是唯一 tick master，MPC/PID 在 20 Hz 执行 50 ticks；每 4 ticks
   只刷新同一 Safety final trajectory 的 buffer stamp；
8. 保存 ego timeline、actor future、collision/lane events、红灯/走廊、进度、舒适、deadline、
   GPU 与 cleanup。

Reset 的 role、route、weather、light 和 script hash 必须完全相同；位置、yaw、speed 的包含
边界分别为 `0.05 m / 0.5 deg / 0.10 m/s`。不通过 reset 时不执行分支。

## 4. 存储与恢复

数据位于被 Git 忽略的：

```text
generated/h2/paired-outcomes/<dataset-id>/
```

- `pairs/<pair-id>.parquet`：每 pair 一个不可变 live shard；
- `images/sha256/<prefix>/<hash>.png`：按内容去重；
- `timelines/`、`actor-future/`、`events/`：Zstd Parquet；
- `labels/<pair-id>.parquet`：离线标签，与 live shard 物理分离；
- `manifest.json`：排序后的路径、字节数和 SHA256。

文件先写同目录临时文件、`fsync`，再 `os.replace`。恢复只跳过内容 hash、dataset id 和
pair id 都通过的 shard；同 id 不同内容拒绝覆盖。正式 Evidence 位于同样被忽略的
`docs/runtime-evidence/h2/<dataset-id>/`。

## 5. H3 feature view

`h3_feature_view` 只构造以下新对象，不从完整 pair 做事后字段删除：

```text
anchor + observable_history + route + candidate trajectory/hash
```

它物理不包含 source、slot、branch order、actor future、outcome、Oracle、Regression、label
或 winner。Oracle 标签、future 和 provenance 仍保存在 H2 store，供离线审计和目标构造，
不得成为 H3 scorer 输入。

## 6. 离线 Oracle 与门

Oracle 规则与阈值以 [START_TASK](../START_TASK.md) 为准。实现只比较两个
`BranchOutcome`，winner 仅引用 candidate id/hash，不读取 source、slot 或执行顺序；交换
两个 branch 必须得到完全相同的 label/reason。

Pilot 通过后才扩大到 120。完整矩阵失败时保留全部 terminal/negative 结果，不补挑场景、
不改 family/seed/weather 或标签门。全门、失败状态和 H3 停止条件同样以 START_TASK 为准。

## 7. 安全冷重启

`python scripts/sdf.py sim restart --map TownXX --rhi dx12` 仅在 READY、world async、tick
owner free、恰好一个 Windows CARLA 进程且 PID/executable/command/config 都匹配时运行。
它只请求 `CloseMainWindow` 并同时验证 PID 与 TCP 已退出；不使用 force kill。退出后原子
pin `DefaultEngine.ini`（保留备份），bounded ensure，最后只做一次 preflight recheck。
未知、多进程、路径/配置不匹配或正常退出失败都返回 `NEEDS_USER_ACTION`。
