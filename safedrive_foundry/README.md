# SafeDrive Foundry runtime

本目录承载 H0–H6 的活动代码。当前结题主线为 H6-CORA：复用现有 SimLingo VLA、Classic
Expert、Guard、Safety、MPC/PID 和 exact-reset collector，修复 World 的反事实标签、
source shortcut、uncertainty calibration 与 offline/live selector 一致性。

```text
Observable
  ├── pretrained SimLingo VLA ── candidate_vla ───── Guard ──┐
  └── Classic Frenet/ST Expert ─ candidate_expert ── Guard ──┤
                                                               ▼
                             candidate-conditioned Outcome World
                                                               ▼
                                  calibrated choose/hold/defer
                                                               ▼
                                   Safety → executable → MPC/PID
```

## 权威文档

- [START_TASK](../START_TASK.md)：当前唯一代码任务与停止点；
- [ROADMAP](../ROADMAP.md)：H6-CORA C0–C6 顺序；
- [PROGRESS](../PROGRESS.md)：冻结结果和当前已确认问题；
- [PROJECT](../docs/PROJECT.md)：系统和研究边界；
- [HYBRID_CANDIDATES](../docs/HYBRID_CANDIDATES.md)：候选、Guard、身份链；
- [COUNTERFACTUAL_DATA](../docs/COUNTERFACTUAL_DATA.md)：同锚点双分支数据；
- [WORLD_MODEL](../docs/WORLD_MODEL.md)：outcome model、loss、calibration、router；
- [ENVIRONMENT](../docs/ENVIRONMENT.md)、[RESOURCES](../docs/RESOURCES.md)、
  [EVIDENCE](../docs/EVIDENCE.md)：运行、预算和证据。

## 代码边界

- `driving_vla/`：nominal VLA、双候选 contracts/generators/pipeline；
- `classic_stack/`：Classic planning/control adapters；
- `data_pipeline/h2-h6/`：paired data、World training/evaluation/runtime；
- `safety_kernel/`：最终硬验证、repair、fallback/MRM；
- `runtime/`：CARLA connection、registry、single tick owner；
- `ros_ws/`：ROS 2 status/tick synchronization bridge；
- `config/`：机器可读路径、CARLA/VLA/实验配置。

World 在线 feature 禁止 source、slot、branch order、Guard verdict、rollout future、outcome、
Oracle、Regression 和 formal answer。World 不能生成轨迹、复活 REJECT、修改 Safety 或直接
输出 throttle/brake/steer。

这里的 source-blind 是 metadata schema 约束；允许的轨迹几何仍可能暴露 planner 风格，必须
通过 trajectory-to-source probe 和反事实平衡诊断，不能删掉物理 feature 伪造随机结果。

## 固定环境

```text
CARLA Server: Windows E:\CARLA_0.9.16\CarlaUE4.exe
runtime/model: WSL2 Ubuntu 24.04
ROS 2: Jazzy, ROS_DOMAIN_ID=42
venv: /home/sdf/.venvs/sdf
hardware: RTX 4080 16GB + i5-13600KF
```

用户已确认 GPU/CARLA 资产可用；每次真实任务仍必须在实际执行上下文 probe/preflight。

## 运行入口

```bash
cd "/mnt/e/autonomous driving"
source /home/sdf/.venvs/sdf/bin/activate
python scripts/sdf.py sim status
python scripts/sdf.py sim preflight --json
```

只有 `READY` 才继续 live task。业务、模型、collector 和 cleanup 不得新建第二 tick master
或直接调用 `world.tick()`。

正式 collector 使用 `ScenarioRuntime` 持有 tick lease；ROS `carla_sync_driver` 是互斥的 G0/
bridge bring-up owner。二者不得同时推进同一 endpoint；只读 status bridge 不拥有 tick。

离线验证：

```bash
python -m unittest discover -s tests -t . -v
python -m compileall -q safedrive_foundry scripts tests
git diff --check
```

测试通过只说明对应代码合同，不等于 CORA 数据、GPU checkpoint 或 CARLA formal 已验证。
