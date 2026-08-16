# H5 实验矩阵（已冻结）

## 目标

在相同 candidates / Guard / Safety / controller / seed / 场景下，比较：

| 臂 | selector |
|---|---|
| off | FrozenH1Router |
| on | H5WorldRouter（risk gate + hysteresis + force_defer 关闭） |
| defer | H5WorldRouter 强制 defer（等价 off，用于开销对照） |

## 场景矩阵

- 来源：H4 locked test split 的 `split=test & valid_pair=true` 物理场景。
- 数量：74 个场景，3 臂 × 74 = 222 次运行。
- 物理 manifest：
  - H2：`generated/h2/paired-outcomes/h2-gatepass-20260813-routefix/scenario_manifest.json`
  - H3 Challenge：`generated/h3/carla-challenge-v2/h3-challenge-v2-20260815d-dev/scenario_manifest.json`
- 不读取 H4 test labels / Oracle / Regression。

## Pilot

- 12 个场景，按 map/family 平衡选取，用于验证协议完整性。
- 只有 pilot 协议门通过后才运行 full。

## 每臂指标

```text
collision / violation
route completion / progress（净进度）
comfort（jerk / lateral accel）
candidate switch count / chattering
fallback / takeover count
P50 / P95 / P99 latency（World scorer）
GPU memory
deadline miss
```

## 停止条件

```text
on 臂安全指标劣于 off 臂
World 覆盖 Safety 回退
chattering 未改善
资源超限
CARLA 状态不稳定
```

## 最终结果

- 已执行 222/222 runs（74 scenarios × 3 arms）。
- Evidence：`docs/runtime-evidence/h5/h5-pilot-all2/final-delivery.json`
- 状态：`GATE_FAILED`，World 闭环收益不可复现，负结果保留。

## 运行入口

```bash
python scripts/h5_readiness.py
python scripts/h5_calibrate_risk.py
python scripts/h5_calibrate_temperature.py
python scripts/h5_run.py --dataset-id h5-<UTC> --pilot --full --accept
```

单场景冒烟：

```bash
python scripts/h5_collect.py --dataset-id h5-smoke \
  --map Town03 --scope pilot --pair-id 'Town03__emergency_lead_brake__s0__ClearNoon'
```
