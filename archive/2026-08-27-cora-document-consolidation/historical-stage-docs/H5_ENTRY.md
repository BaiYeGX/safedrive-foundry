# H5 Entry Checklist

## 1. 进入 H5 的前置条件

```text
H3 Evidence: GATE_PASSED
H4 Evidence: GATE_PASSED
H5 Readiness: ready=true
CARLA Preflight: READY
```

当前验证命令：

```bash
python scripts/h5_readiness.py
python scripts/h5_calibrate_risk.py
python scripts/h5_calibrate_temperature.py
python scripts/h5_run.py --preflight
```

风险阈值校准 artifact：

```text
generated/h5/risk_calibration.json
generated/h5/temperature_calibration.json
```

## 2. H5 必须使用的新组件

- `H5WorldRouter`：
  - risk-gated defer；
  - probability temperature floor 0.5；
  - minimum hold ticks；
  - hysteresis margin；
  - emergency_switch_margin；
  - force_defer 模式；
  - scorer deadline defer；
  - fallback 到 FrozenH1Router。
- 路径：
  - `safedrive_foundry/data_pipeline/h5/runtime.py`

## 3. H5 实验设计

| 臂 | selector | 说明 |
|---|---|---|
| off | FrozenH1Router | 非学习基线 |
| on | H5WorldRouter | World + risk gate + hysteresis |
| defer | H5WorldRouter 强制 defer | 等价于 off，用于检查系统开销 |

场景矩阵：H4 locked test valid rows（74 个），3 臂共 222 次运行。

主指标：

- collision / violation；
- route completion / progress；
- comfort（jerk / lateral accel）；
- candidate switching count / chattering；
- takeover / fallback count；
- P50/P95/P99 latency；
- GPU memory；
- deadline miss。

## 4. 停止条件

- 任一臂 Safety 回退被 World 覆盖：立即停止；
- on 臂安全指标劣于 off 臂：停止并记录负结果；
- chattering 未改善：停止并报告；
- 资源超限：停止。

## 5. 当前最终状态

- H5 工程代码已完成：矩阵、collector、acceptance、orchestrator。
- 正式 pilot + full 已一次性执行：222/222 runs。
- 最终 Evidence：`docs/runtime-evidence/h5/h5-pilot-all2/final-delivery.json`。
- 结果：`GATE_FAILED`，World 闭环收益不可复现，负结果已冻结。
