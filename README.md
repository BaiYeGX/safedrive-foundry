# SafeDrive Foundry

SafeDrive Foundry 是一个运行在单机上的 CARLA–ROS 2 纯软件在环自动驾驶研发平台。
项目把经典专家、轻量驾驶 VLA、独立 Safety Kernel、约束 MPC、场景回归和后续动作条件
World Model 组织成可运行、可降级、可审计的闭环。

## 当前状态

- G0：环境与确定性基线已冻结；
- G1：经典专家与运行时已 `COMPLETED_WITH_LIMITS`；
- G2：独立 Safety Kernel 已完成 offline 主体；
- G3：**SimLingo 纯 VLA 路径 + VLA 速度 + constrained MPC 已真实接入 CARLA，
  状态为 `MEASURED_WITH_LIMITS`**；
- G3-05 正式 VLA+Safety live 验收仍需新证据；
- G4/G5 尚未正式启动。

当前最重要的实现链：

```text
CARLA RGB + ego + coarse target
  → SimLingo / InternVL2-1B
  → raw VLA path + VLA speed
  → PathManager
  → constrained MPC
  → CARLA vehicle
```

MPC 不跟踪 CARLA 车道中心线；地图只提供类似 GPS 的 coarse navigation target。

## 从哪里开始

完整能力说明、环境启动、短测/长测命令、三条可视化线、D3D 解法、已知限制、优化顺序
和 G4 入口统一见：

- [G3 Pure VLA + MPC 发布说明与后续路线](docs/architecture/G3_VLA_MPC_RELEASE_GUIDE.md)
- [G3 稳定运行与故障排查手册](docs/architecture/G3_VLA_MPC_STABLE_RUNBOOK.md)
- [G3 精选发布证据索引](docs/architecture/evidence/g3-05/RELEASE_EVIDENCE_INDEX.md)
- [当前进度](PROGRESS.md)
- [任务入口](START_TASK.md)
- [项目路线图](ROADMAP.md)

最短启动流程：

```bash
source /home/sdf/.venvs/sdf/bin/activate
cd "/mnt/e/autonomous driving"

python scripts/sdf.py sim preflight --json
# CARLA 未运行且结果为 RETRYABLE_FAILURE 时只 ensure 一次：
python scripts/sdf.py sim ensure --map Town03 --rhi dx12 --startup-timeout 180 --json

python tests/g3/run_g3_vla_mpc_minimal.py \
  --map Town03 \
  --duration-s 60 \
  --inference-mode full \
  --no-map-restart \
  --max-speed 6 \
  --debug-draw \
  --evidence-dir docs/architecture/evidence/g3-05/release_smoke_town03_60s
```

## 关键边界

- 当前是 CARLA SIL，不是实车或公共道路安全证明；
- VLA/World 不能绕过 Validator、Safety Kernel 或执行层硬约束；
- pure VLA+MPC demo 当前不带 Safety，不具备完整避障保证；
- 单张 RTX 4080 同时承担 CARLA DX12 与 CUDA，当前稳定基线为 DX12；
- SimLingo 代码与模型权重是外部大资产，不纳入 Git，路径见
  [LOCAL_ASSETS.md](docs/project/LOCAL_ASSETS.md)；
- 运行产生的大型 evidence 默认应本地保存，只选择关键摘要进入版本库。

## 离线验证

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -m unittest discover -s tests/g3 -t . -v
python -m unittest tests.g1.test_g1_02_connection -v
```

最近一次 G3 全量离线结果：`Ran 154`，153 passed，1 个真实 GPU 20× forward 测试按
显式环境开关跳过。
