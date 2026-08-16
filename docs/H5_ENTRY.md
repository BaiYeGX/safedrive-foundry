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
python scripts/h5_run.py --preflight
```

## 2. H5 必须使用的新组件

- `H5WorldRouter`：
  - risk-gated defer；
  - minimum hold ticks；
  - hysteresis margin；
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

`scripts/h5_smoke.py` 已实现并通过编译/测试，但首次 live 运行因 CARLA 冷重启要求中断：

```text
RuntimeViolation:map_mismatch ... cold restart required
```

失败 Evidence 已保留：

```text
docs/runtime-evidence/h5/failed/h5-smoke-20260816-cold-restart-required/h5_smoke.json
```

这不是 H5 代码失败，而是当前 CARLA 进程需要一次受控冷重启。进入完整 H5 前应先执行：

```bash
python scripts/sdf.py sim restart --map Town03 --rhi dx12 --json
python -m scripts.h5_smoke --map Town03 --evidence-dir <new-evidence-dir>
```
