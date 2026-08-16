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
python scripts/h5_run.py --preflight
```

风险阈值校准 artifact：

```text
generated/h5/risk_calibration.json
```

## 2. H5 必须使用的新组件

- `H5WorldRouter`：
  - risk-gated defer（runtime 默认 `0.35` 保守阈值；dev 校准中点 `0.370274`）；
  - probability temperature floor `0.5`（避免胜率概率饱和）；
  - minimum hold ticks；
  - hysteresis margin；
  - emergency_switch_margin（强优势可提前打破 hold）；
  - zero-context defer（防止 scene gate 静默输出）；
  - fallback 到 FrozenH1Router。
- 路径：
  - `safedrive_foundry/data_pipeline/h5/runtime.py`

## 3. H5 实验设计

建议三臂：

| 臂 | selector | 说明 |
|---|---|---|
| off | FrozenH1Router | 非学习基线 |
| on | H5WorldRouter | World + risk gate + hysteresis |
| defer | H5WorldRouter 强制 defer | 等价于 off，用于检查系统开销 |

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

## 5. 尚未完成

实际 closed-loop on/off 需要：

- 新任务授权；
- 冻结 H5 场景矩阵；
- 运行真实 CARLA 闭环实验；
- 生成 H5 Evidence。

## 6. 当前 live smoke 状态

`scripts/h5_smoke.py` 已成功执行一次 Town03 live smoke：

```text
run_id              h5-town03-1786855295418892365
routing reason      h5_world_ranked
selector            h5_world_v1
selected            expert
world               RANKED
vla_forward_count   1
ok                  true
```

Evidence：

```text
docs/runtime-evidence/h5/h5-smoke-20260816-ok/h5_smoke.json
docs/runtime-evidence/h5/h5-smoke-3t-20260816-ok/h5_smoke.json
```

3-tick smoke 验证了 per-frame candidate id 变化下 switch_count 正确按 source 统计：
`decisions=3, switch_count=0, defer_count=0`。

之前因 CARLA 冷重启要求导致的失败也已保留：

```text
docs/runtime-evidence/h5/failed/h5-smoke-20260816-cold-restart-required/h5_smoke.json
```
