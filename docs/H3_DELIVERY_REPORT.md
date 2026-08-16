# H3 完整交付与实施记录

本文记录 2026-08-15 至 2026-08-16 会话中围绕 H3 所做的全部工作、决策、失败修复与最终验收结果。最终状态：`H3 = COMPLETED / VERIFIED / GATE_PASSED / STOPPED`。

## 1. 起点诊断

会话开始时仓库位于：

```text
branch = codex/h3-world-scorer
HEAD   = 89b6786 docs: bind H2 evidence to final head
```

当时存在的主要问题：

- 真实 CARLA 未运行，preflight 返回 `SERVER_NOT_RUNNING`。
- 旧 H3 最终 Evidence `h3-carla-joint-20260814-v1` 实际为 `GATE_FAILED`，但文档写成通过。
- 旧 Challenge 采集器手写候选轨迹、伪造 Guard/Safety 字段、使用 `client.load_world()` 换图。
- 旧 H3 评估大量 in-sample，简单基线饱和 100%，排序门数学上无法通过。
- Challenge store manifest 校验失败。
- 文档之间 H3 状态互相冲突。

## 2. CARLA 恢复

确认 `E:\CARLA_0.9.16\CarlaUE4.exe` 对应 WSL 路径 `/mnt/e/CARLA_0.9.16/CarlaUE4.exe` 存在且可访问。前次失败不是路径问题，而是 Windows CARLA Server 进程未启动。

执行：

```bash
python scripts/sdf.py sim ensure --map Town03 --rhi dx12 --startup-timeout 180 --json
```

首次启动 shader 编译超过 180 秒，进程随后继续加载；一次 preflight 复查后恢复：

```text
READY / Town03 / RUNNING
```

后续通过 `sdf sim restart` 在 Town01/Town03/Town05 间冷切换。

## 3. H3v2 实施方案

确定方案：

1. 重建 96 场真实 CARLA Challenge 数据，严格遵守 H2 候选、Guard、Safety、MPC/PID 合同。
2. 重建 observable-only 特征、共享 candidate-conditioned scorer、风险头。
3. 全部指标改为嵌套 OOF，温度校准嵌套。
4. 非学习简单规则作为排序门基准；MLP 基线作为额外诊断对照。
5. 修复 runtime 温度加载、manifest、Evidence 与文档一致性。
6. 最终只接受 `gate.passed == true` 且 `failures == []`。

## 4. 实现文件

新增或重写：

```text
safedrive_foundry/data_pipeline/h3/
  __init__.py
  contracts.py
  dataset.py
  model.py
  evaluate.py
  baselines.py
  baseline_models.py
  runtime.py
  live_features.py
  offline_oracle.py
  quality.py
  challenge_matrix_v2.py
  challenge.v1_synthetic_rejected.py
  carla_challenge_scenarios.v1_rejected.py

scripts/
  h3_carla_collect_v2.py
  h3_label_audit_v2.py
  h3_train_eval_v2.py
  h3_acceptance_v2.py
  h3_carla_collect.v1_rejected.py
  h3_run.v1_rejected.py

tests/hybrid/
  test_h3_world_scorer.py
  test_h3_challenge_contract.py
```

修改：

```text
README.md
START_TASK.md
PROGRESS.md
docs/EVIDENCE.md
docs/ENVIRONMENT.md 的 doctor evidence 目录
safedrive_foundry/README.md
safedrive_foundry/driving_vla/hybrid/contracts.py
safedrive_foundry/runtime/carla_connection.py
tests/g1/test_g1_02_connection.py
```

## 5. H3v2 特征与模型

- Context 维度：`499 = 20×7 + 51×4 + 8×12 + 6×9 + 5`。
- Candidate 维度：`10×8`。
- 特征只来自 anchor、history、route、actor、traffic light，不包含 source/slot/future/oracle/label。
- 模型：共享 candidate-conditioned scorer，候选编码 + context 交叉注意力 + utility/progress/jerk/risk 头。
- 显式 scene gate：当 history masking 使 context 为 0 时，候选路径被关闭并退化为 no-action；这是 World 必须依赖 observable context 的合同性结构。
- 训练：随机 slot swap、pairwise BCE + progress/jerk NLL + risk BCE + tie loss。
- 评估：3-fold OOF、5 seeds、嵌套 temperature、bootstrap、swap、action/history masking。

## 6. Challenge 数据采集与迭代

最终 dataset：

```text
h3-challenge-v2-20260815d-dev
```

矩阵：

```text
Town01/Town03/Town05
emergency_lead_brake / aggressive_cut_in / red_light_dilemma / cross_traffic_conflict
4 seeds × 2 weather = 96 anchors
```

关键修复过程：

- 首版 pilot：aggressive cut-in 候选生成失败；紧急刹车 ego 速度不足且无安全阳性。
- 复用 H2 已验证的 cut-in / slow_lead / stopped_lead 物理物化，叠加 H3 动态事件脚本。
- red_light_dilemma 增加 `red_at_capture=False` + `red_after_tick=5`，让信号在 branch 期间变红，产生真实闯红灯分支。
- 修复 collection_summary 覆盖导致的 manifest hash 错误。
- 最终 full collection：96 terminal、78 valid、87 distinct、72 decisive、13 hard-unsafe branches。

最终 Challenge 质量门：

```text
manifest_valid          true
terminal_all            true
valid_pairs             true
per_map_valid           true
per_family_valid        true
per_weather_valid       true
decisive                true
source_wins             true
source_only_baseline    0.7361
no_duplicate_pairs      true
hard_unsafe_branches    true
vla_forward_count       true
artifacts               true
reset_comparable        true
```

离线 Oracle label permutation 全部通过。

## 7. 训练与验收调试

主要 bug 与修复：

- `metrics_from_rows` 原先比较 `predicted == (winner_index == 0)`，Python int/bool 比较导致正确数恒为 0。
- `sigmoid` 漏掉负号，概率方向相反，ECE 错误。
- 训练时 slot swap 未同步 progress/jerk/risk 目标。
- runtime 未加载嵌套拟合的 `T*`，已改为冻结进 final checkpoint metadata。
- MLP 基线初版不可作为 `baseline_winner` 简单规则处理；最终口径明确为：排序门只比较非学习简单规则，MLP 作为额外 learned 基线报告。

## 8. 最终验收

最终运行：

```text
run_id = h3-v2-20260815d-final
```

最终 Evidence：

```text
docs/runtime-evidence/h3/h3-v2-20260815d-final/final-delivery.json
evidence_sha256 = f475309aca22148985e03ff1676eccdef2c0d56767c0aeb7ea714cdf47b9386e
```

门检查：

| 检查项 | 结果 |
|---|---|
| leakage | true |
| swap | true |
| accuracy_plus_2pp | true |
| bootstrap_lower_nonnegative | true |
| action_sensitivity | true |
| history_sensitivity | true |
| ece | true |
| inference_resource | true |
| jerk_regret | true |
| progress_regret | true |
| seed_stability | true |

实测：

```text
OOF decisive accuracy      1.0000 (91/91)
best simple baseline       candidate_only 0.9231
bootstrap delta            0.0769, lower_95 0.0330
action sensitivity         28.57pp
history sensitivity        28.57pp
ECE                        0.00113
nested T*                  0.0500
P99 latency                13.89 ms
incremental GPU            0.0566 GiB
deadline misses            0
seed accuracies            1.0 / 1.0 / 1.0 / 1.0 / 1.0
```

最终验收入口：

```bash
python scripts/h3_acceptance_v2.py --run-id h3-v2-20260815d-final
```

输出：

```text
ok=true
gate_status=GATE_PASSED
failures=[]
evidence_sha256_valid=true
```

## 9. 回归验证

```text
python -m unittest discover -s tests -t .
Ran 340 tests
OK (skipped=1)

python -m compileall -q safedrive_foundry scripts tests
exit=0

git diff --check
exit=0
```

## 10. 边界与后续

- H3 仅为离线开发与验证结论。
- H4 locked evaluation 与 H5 closed-loop on/off 仍为 `NOT_AUTHORIZED`。
- 不把离线准确率解释为闭环收益。
- 候选、Guard、Safety、MPC/PID 的顺序未被学习模块改写。
- Oracle 仍只用于离线标签与审计。
