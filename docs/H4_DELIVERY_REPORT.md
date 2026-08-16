# H4 Locked Evaluation 完整交付与实施记录

## 1. 目标

H4 是在 H3 冻结产物上的唯一一次 locked test 盲测。核心要求：

- 不训练、不调参、不修改 H3 冻结 checkpoint；
- 在读取 test 标签前冻结 H4 脚本、阈值、split、checkpoint 和校准；
- 报告 AUROC/accuracy、regret、calibration、defer coverage、两来源胜率、P50/P95/P99 与显存；
- World 不优于简单基线则如实停止，不进入 H5。

## 2. 发现并修复的问题

### 2.1 H3 raw ensemble uncertainty 导致 runtime 全 defer

现象：H3 的 5 个 final seed 模型 utility logit 尺度差异很大，直接平均后的 ensemble
uncertainty 在 dev 上全部超过 `max_uncertainty=0.35`，导致 `WorldScorer.score_pair()`
对 112/112 dev 样本全部 `defer_low_confidence`。

修复：在 H4 冻结流程中增加 **dev-only per-model utility normalization**：

```text
normalized_utility_i = (utility_i - mean_i) / std_i
```

其中 `mean_i` / `std_i` 只从 H3 dev folds 计算，不使用 test 标签。归一化后：

- dev coverage：`91/91`，accuracy `1.0000`
- locked test coverage：`63/64`，ranked accuracy `1.0000`
- uncertainty 均值从 `1.16` 降至 `0.065`

该 normalization 写入 H4 evidence，作为 H4 冻结校准的一部分。

### 2.2 H3 `defer_curve` 与 runtime 语义不一致

原 `defer_curve` 使用未 cap 的 uncertainty，且 margin 条件恒真。已修复为：

- uncertainty 按 runtime 方式 cap 到 `[0,1]`；
- 使用冻结 `defer_margin` 判断 margin；
- 与 `NormalizedWorldScorer` 的 defer 逻辑一致。

### 2.3 H3 文档数字矛盾

统一为最终 Evidence 的权威值：

- H3 P99：`13.89 ms`
- action/history sensitivity：`28.57pp`（不是 `0.2857 pp`）

### 2.4 H3 Evidence 缺少 provenance 绑定

新增 sidecar：

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/provenance.json
docs/runtime-evidence/h3/h3-v2-20260815d-final/split_provenance.json
```

不改写冻结的 `final-delivery.json`，因此不破坏 H3 evidence SHA。

### 2.5 H4 缺失能力补齐

新增：

```text
safedrive_foundry/data_pipeline/h4/
  contracts.py
  locked_dataset.py
  metrics.py
  runtime.py
scripts/h4_locked_eval.py
scripts/h4_acceptance.py
tests/hybrid/test_h4_locked_eval.py
```

实现内容：

- locked test loader（只允许 `split=test`）；
- test/dev lineage 与 feature payload 隔离审计；
- AUROC、Brier、NLL、ECE；
- runtime-faithful defer coverage；
- end-to-end selector（World + h1_soft_selector fallback）；
- Expert/VLA 分层胜率；
- normalized 5-seed ensemble runtime；
- bootstrap 95% CI；
- swap consistency；
- 增量显存 benchmark。

### 2.6 H3 `_resource_benchmark` 增量显存口径修正

原 H3 benchmark 把 `peak_reserved_gib` 直接当作增量显存。已修正为：

- 在模型加载前记录 `pre_reserved_gib`；
- 在 benchmark 后计算 `incremental_gpu_gib = peak_reserved_gib - pre_reserved_gib`；
- 资源门使用真正的增量显存。

修改文件：`scripts/h3_train_eval_v2.py`。

### 2.7 H3 split manifest 支持 Challenge 身份绑定

`build_split_manifest` 新增可选参数：

```text
challenge_physical_manifest_sha256
challenge_store_manifest_sha256
```

当传入 Challenge store 身份时，split manifest 会同时记录 H2 与 Challenge 的 physical/store manifest，
避免再出现 combined split 只绑定 H2 的 provenance 缺口。

修改文件：

```text
safedrive_foundry/data_pipeline/h3/dataset.py
scripts/h3_train_eval_v2.py
```

## 3. H4 最终运行

```text
run_id = h4-locked-20260816-final
mode   = evaluate
```

Evidence：

```text
docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json
evidence_sha256 = 35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4
gate_status     = GATE_PASSED
failures        = []
```

实测：

```text
test valid rows          74
test decisive            64
World accuracy           1.0000 (64/64)
AUROC                    1.0000
best simple baseline     candidate_only 0.890625 (57/64)
bootstrap delta          0.109375
bootstrap lower 95       0.03125
defer coverage           0.984375 (63/64)
ranked accuracy          1.0000
end-to-end accuracy      0.984375
h1_soft_selector         0.890625
Expert/VLA wins          39 / 25
ECE                      0.00884
Brier                    0.00129
NLL                      0.00666
P50 / P95 / P99          12.22 / 14.31 / 15.23 ms
deadline misses          0
incremental GPU          0.03125 GiB
swap max error           0.0
isolation                passed
```

## 4. 门结论

```text
GATE_PASSED
H5_AUTHORIZED = true
H5 still NOT_AUTHORIZED until a separate task explicitly authorizes closed-loop.
```

## 5. 回归验证

```text
python -m unittest discover -s tests/hybrid -t . -v
python -m unittest discover -s tests -t .
python -m compileall -q safedrive_foundry scripts tests
git diff --check
```

全量测试：`352 tests: 351 passed, 1 skipped, 0 failed`。

## 6. 停止边界

- H4 已完成并冻结；
- H5 不自动进入；
- 不把 H4 离线准确率解释为闭环收益；
- 不因 H4 通过而放松 Safety/Guard/Oracle 边界。
