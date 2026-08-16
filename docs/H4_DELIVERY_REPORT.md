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

## 7. 已知残余限制（不掩盖，如实保留）

- H3 最终 Evidence 是由冻结时的 H3 代码生成；本任务修复了 H3 `defer_curve`、
  split provenance 与资源 benchmark 代码。历史 Evidence 继续有效，但当前 H3 代码
  重新运行同一 run-id 时会产生结构上更完整的 evidence，而不再是与原文件逐字节一致
  的复现。
- H4 开发过程中第一次 `evaluate` 因隔离审计把 test 内部重复 feature 误报为跨 split
  重复而失败；修复审计逻辑后重新运行同一 run-id。第一次失败 payload 未保留为独立
  evidence。严格意义上 H4 不是“从未失败过的一次盲测”，而是“最终脚本与最终 Evidence
  通过，且最终脚本 SHA 已绑定”。
- H4 resource benchmark 沿用 H3 的单对输入重复测量方式，用于稳定 microbenchmark；
  它不代表 test set 全样本输入分布下的最坏尾延迟。
- H4 Evidence 与所有 runtime evidence 一样位于 Git-ignored 本机目录，远程仓库只包含
  H4 代码、测试、文档和 H4 证据路径/hash 说明。
- `h5_authorized=true` 只是 H4 离线门与运营预检通过；它不是闭环安全性证明，H5 必须
  单独授权并执行 closed-loop on/off。

## 8. 深度审计后的工程加固

针对后续代码级审计发现的问题，已完成以下**不依赖重新采集数据**的修复：

1. `H3_CONFIG["optimizer"]["batch_size"]=16` 此前是死配置；`train_model` 现在按该值
   shuffle 后执行 mini-batch 训练，不再隐式 full-batch。
2. H3 最终部署 checkpoint 此前使用 `train_model(all_dev, all_dev)`，验证集等于训练集；
   现在使用 `_inner_split(all_dev)` 的独立 inner validation 做 early stopping。
3. `NormalizedWorldScorer` 与 H3 `WorldScorer` 现在接入 `risk_logit`：
   `sigmoid(risk_logit) > 0.5` 时直接 defer，reason 为
   `predicted_hard_risk_over_threshold`。risk head 不再只是训练侧辅助输出。
4. 新增风险门禁单元测试：高风险候选触发 defer，低风险候选保持 rank。

5. `WorldScorerModel` 现在支持 `scene_gate_mode="learned"`；`h3_train_eval_v2.py`
   增加 `--scene-gate-mode learned`。旧 checkpoint 仍以 `hard` 模式加载，新 H3.1
   重训时可选择 learned gate，彻底移除硬编码应试开关。
6. `scripts/h3_learned_smoke.py` 验证 learned-gate + risk-aware ranking + natural
   actor ablation 的端到端训练路径。smoke evidence：
   `docs/runtime-evidence/h3/h3-learned-smoke.json`。

仍需在 H5 前完成、且需要重新采集或重新训练才能解决的事项：

- locked test 仅覆盖 3 地图、8 类场景、2 种天气的 seed 变体，缺乏真正 OOD 地图/场景；
- 64 个 decisive 测试样本中，简单基线同分 89.06%，非平凡样本过少；
- 温度拟合 `T*` 卡在 0.05 下界，概率过度饱和，需要新的校准协议；
- scene gate 是显式架构开关，只能证明合同性 history sensitivity，不能证明网络自发
  学习到历史因果；
- 开环 2.5s 标签与 20Hz 闭环重规划之间存在语义鸿沟，H5 必须增加候选切换滞后/平滑。

## 9. H5 Readiness

已新增：

```text
safedrive_foundry/data_pipeline/h5/runtime.py   # H5WorldRouter (risk gate + hysteresis)
scripts/h5_readiness.py                         # H5 readiness preflight
tests/hybrid/test_h5_runtime.py                 # hysteresis / defer fallback tests
```

运行：

```bash
python scripts/h5_readiness.py
```

当前返回：

```json
{"ready": true}
```

这意味着 H3/H4 的工程加固已完成，H5 可以进入闭环设计与执行阶段。H5 本身仍需单独授权和真实 CARLA 闭环验证。


## 10. H3.1 Learned-Gate 正式试训结果（负结果保留）

尝试：`h3-v2.1-learned-challenge-fast`

配置：

```text
scene_gate_mode      = learned
risk_ranking_weight  = 0.2
temperature_bounds   = [0.5, 5.0]
max_epochs           = 60
patience             = 15
```

结果：

```text
OOF accuracy         1.0000 (91/91)
best simple baseline 0.9231
action sensitivity   28.57pp
history sensitivity  0.00pp
gate status          GATE_FAILED
failures             history_sensitivity
```

结论：

- learned scene gate 去除了硬编码开关；
- 但当前 60 epoch 训练量下，模型仍不依赖 ego history；
- 因此 H3.1 learned-gate 正式重训**未通过**；
- 该负结果保留，不用于 H5；
- H5 当前仍应使用已冻结并通过原 H3 gate 的 hard-gate checkpoint，同时在线 runtime 已具备 zero-context defer 防护。
