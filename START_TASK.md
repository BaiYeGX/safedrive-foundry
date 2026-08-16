# 当前唯一任务：H4 Locked Evaluation（已完成）

## 状态

```text
H0 route consolidation = VERIFIED / STOPPED
H1 hybrid candidate contract = VERIFIED / STOPPED
H2 paired outcomes = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H3 World scorer development = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H4 locked evaluation = COMPLETED / VERIFIED / GATE_PASSED / STOPPED
H5 World on/off closed loop = NOT_AUTHORIZED
Online Oracle = PROHIBITED
```

## H4 目标

在 H3 冻结的模型、split、阈值和 checkpoint 基础上，对从未参与 H3 训练的 locked test
split 进行唯一一次盲测，报告排序、校准、defer、资源与两来源胜率，并判断 World 是否
优于冻结简单基线。H4 不训练、不调参、不读 test 标签后修改脚本。

## H4 最终 Evidence

```text
docs/runtime-evidence/h4/h4-locked-20260816-final/final-delivery.json
evidence_sha256 35e28958ddd98d9df7a980ffd707bf6049efb9685e22d335082f69916974e6e4
gate_status     GATE_PASSED
gate.failures   []
```

关键实测：

```text
test valid rows         74
test decisive           64
World accuracy          1.0000 (64/64)
AUROC                   1.0000
best simple baseline    candidate_only 0.890625 (57/64)
bootstrap delta         0.109375, lower_95 0.03125
defer coverage          0.984375 (63/64 ranked)
ranked accuracy         1.0000
end-to-end accuracy     0.984375 vs h1_soft_selector 0.890625
Expert/VLA wins         39 / 25
ECE                     0.00884
P99 latency             15.23 ms
incremental GPU         0.03125 GiB
deadline misses         0
swap max error          0.0
isolation               passed
```

H5 授权预检：

```text
h5_authorized = true
```

H5 工程预检已通过（`scripts/h5_readiness.py` 返回 ready=true），但仍需新任务单独授权执行 closed-loop on/off；进入前必须使用 risk-gated runtime 与 H5 hysteresis router，并重做温度校准。

## H4 冻结输入

- H3 evidence：`h3-v2-20260815d-final`，SHA `f475309aca...`
- H3 split manifest：`17dedd305aaf...`
- H2 store：physical `6e74a789...` / store `22d11961...`
- Challenge store：physical `7b27cc14...` / store `a87871c9...`
- 5 个 H3 final checkpoints（seed 11/23/37/53/71）
- 温度 `T*=0.05000531921999756`
- defer 阈值：`max_uncertainty=0.35`, `defer_margin=0.05`
- 资源门：`P99<=50ms`, `deadline_misses=0`, `incremental_vram<=1.5GiB`

## H4 严格禁止

- 打开 test 标签后修改 H4 脚本、阈值、seed 或 checkpoint；
- 使用 test 标签做任何模型选择或调参；
- 因为结果不理想就重新运行或降低门槛；
- 将 H4 离线结果解释为闭环收益。

## H3 冻结记录（历史）

H3 最终运行 `h3-v2-20260815d-final` 已冻结为
`H3 COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。H3 Evidence 与 sidecar：

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/final-delivery.json
docs/runtime-evidence/h3/h3-v2-20260815d-final/provenance.json
docs/runtime-evidence/h3/h3-v2-20260815d-final/split_provenance.json
```

H3 关键门：OOF accuracy `1.0000`，best simple baseline `candidate_only 0.9231`，
bootstrap lower 95 `0.0330`，action/history sensitivity `28.57pp`，ECE `0.00113`，
P99 `13.89ms`，增量显存 `0.0566GiB`，0 deadline misses。
