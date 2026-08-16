# H5 实验矩阵草案

## 目标

在相同 candidates / Guard / Safety / controller / seed / 场景下，比较：

| 臂 | selector |
|---|---|
| off | FrozenH1Router |
| on | H5WorldRouter（risk gate + hysteresis） |
| defer | H5WorldRouter 强制 defer（等价 off，用于开销对照） |

## 建议场景

初始阶段（Town03）：

```text
3 maps x 4 families x 4 seeds x 2 weather
优先：
  Town01 / Town03 / Town05
  emergency_lead_brake
  aggressive_cut_in
  red_light_dilemma
  cross_traffic_conflict
```

## 每臂指标

```text
collision / violation
route completion / progress
comfort（jerk / lateral accel）
candidate switch count / chattering
fallback / takeover count
P50 / P95 / P99 latency
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

## 运行入口

```bash
python -m scripts.h5_smoke --map Town03 --ticks 20 --evidence-dir <dir>
```

完整 H5 需要单独授权与场景矩阵确认。
