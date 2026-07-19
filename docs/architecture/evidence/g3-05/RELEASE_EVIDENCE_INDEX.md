# G3 Pure VLA + MPC 精选证据索引

> 检查点：2026-07-19
> 发布口径：`MEASURED_WITH_LIMITS`，不是 G3-05 Safety `VERIFIED`

本索引只登记能够支撑当前发布说明的最小 JSON。多 MB 的逐帧 trace、模型输入数组、
console log 和 partial checkpoint 留在本机 evidence 目录，不进入源码发布；历史结果保持
原样，不因后续归因变化而改写。

## D3D/CUDA 隔离

| Evidence | 结果 | 结论 |
|---|---|---|
| [`d3d_E0_forward_only/failure_latest.json`](./d3d_E0_forward_only/failure_latest.json) | DX11，静止车，约第 79 次 forward 后 Server 无响应 | 车辆运动、MPC 和显存 OOM 不是必要条件 |
| [`d3d_E0_A_dx12_300s/latest_summary.json`](./d3d_E0_A_dx12_300s/latest_summary.json) | DX12，300s / 400 次 forward 完成 | 当前硬件/驱动组合的稳定基线 |
| [`d3d_E0_A_dx12_300s/run_config.json`](./d3d_E0_A_dx12_300s/run_config.json) | requested/effective RHI 均为 DX12 | 固定运行配置 |
| [`d3d_E1_dx12_full/latest_summary.json`](./d3d_E1_dx12_full/latest_summary.json) | DX12 full 180s 完成，无 D3D crash；驾驶门失败 | D3D 稳定与驾驶质量分轨判断 |
| [`d3d_E1_dx12_full/run_config.json`](./d3d_E1_dx12_full/run_config.json) | full runner 的 DX12 配置 | 复现配置 |

## 驾驶闭环

| Evidence | 结果 | 结论 |
|---|---|---|
| [`official_contract_town03_300s_20mps_attach/latest_summary.json`](./official_contract_town03_300s_20mps_attach/latest_summary.json) | `DEMO_PASS`，300s、约 763m、0 碰撞、0 离路 | 官方 contract 与 constrained MPC 的长测能力 |
| [`official_contract_town03_300s_20mps_attach/run_config.json`](./official_contract_town03_300s_20mps_attach/run_config.json) | 1024×512、speed gain 1.0、DX12、Town03 | 该长测的输入与运行配置 |
| [`mercedes_town15_300s_30mps_chasefix/failure_latest.json`](./mercedes_town15_300s_30mps_chasefix/failure_latest.json) | Town15 历史失败 | 保留负结果，支撑 switch/stale 修复归因 |
| [`mercedes_town15_300s_30mps_chasefix/run_config.json`](./mercedes_town15_300s_30mps_chasefix/run_config.json) | Mercedes 几何和 attach/RHI 边界 | 不把 attach 的未知 RHI 伪报为独立验证 |

Town15 最新 PathManager 修复目前只有上述历史 trace 的离线反事实重放：n135–160 为
26/26 接纳、max path age 0、committed max|κ| 约 0.720 1/m。它不是新的 live 结果，
因此发布后仍需按
[`G3_VLA_MPC_RELEASE_GUIDE.md`](../../G3_VLA_MPC_RELEASE_GUIDE.md) 的顺序复验。

## 证据使用规则

- `verified=false` 必须原样解释，不能从文件名或 `DEMO_PASS` 推导 G3 阶段关闭。
- lane/traffic-light oracle 只用于报告和验收，不得进入控制。
- 不用旧 DX11 失败反推所有机器必然失败；当前结论限定 CARLA 0.9.16、RTX 4080、
  驱动 610.74 和已测配置。
- 新 live evidence 应写入新目录，不覆盖这里登记的历史 JSON。
