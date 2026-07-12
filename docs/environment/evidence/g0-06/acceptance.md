# G0-06 集成验收报告

日期：2026-07-12（Asia/Singapore）  
状态：**COMPLETED**  
范围：汇总 G0-01～G0-05 产物并从干净终端复现，不新增业务功能。

## 结论

G0 集成门禁通过。Windows/WSL2/CARLA/ROS 2、版本锁、固定步长、唯一 tick master、frame/clock 对齐、重复性、checkpoint 恢复、诊断入口和退出清理均有可追溯证据。G0-06 不启动 G1；阶段关闭仍等待用户确认。

## 前置任务

| 任务 | 状态 | 关键证据 |
|---|---|---|
| G0-01 环境盘点与版本冻结 | COMPLETED | [`HOST_INVENTORY.md`](../../HOST_INVENTORY.md)、[`VERSION_DECISION.md`](../../VERSION_DECISION.md)、[`versions.lock`](../../../../versions.lock) |
| G0-02 WSL2、GPU 与 ROS 2 | COMPLETED | [`WSL_ROS2_BASELINE.md`](../../WSL_ROS2_BASELINE.md) |
| G0-03 CARLA Server | COMPLETED | [`CARLA_SERVER_BASELINE.md`](../../CARLA_SERVER_BASELINE.md) |
| G0-04 骨架与跨系统连通 | COMPLETED | [`CARLA_ROS_CONNECTIVITY.md`](../../CARLA_ROS_CONNECTIVITY.md) |
| G0-05 确定性与诊断 | COMPLETED | [`G0_05_DETERMINISM.md`](../../G0_05_DETERMINISM.md)、[`g0-05/`](../g0-05/) |

## 本次复现

所有离线命令均在无 profile、无 rc 文件的 bash 中执行；需要访问 Windows interop、GPU 和 WSL 的现场命令在外部权限下运行，并显式保留系统 Windows/WSL 路径。

| 验收项 | 结果 | 证据 |
|---|---|---|
| G0-05 离线契约 | PASS，14/14 | [`validation.md`](validation.md) |
| 相同 seed 重复 | PASS，seed 2026，16 帧×2，差异为空 | [`smoke/repeat-1/trace.json`](smoke/repeat-1/trace.json)、[`smoke/repeat-2/trace.json`](smoke/repeat-2/trace.json) |
| checkpoint 恢复 | PASS；中断命令返回75，恢复后10帧与 clean run 差异为空 | [`smoke/recover/trace.json`](smoke/recover/trace.json)、[`smoke/recover-clean/trace.json`](smoke/recover-clean/trace.json) |
| CARLA live tick smoke | PASS，client/server均为0.9.16，20帧 `10431–10450` | [`smoke/live/trace.json`](smoke/live/trace.json) |
| `sdf doctor` | PASS 基础检查；报告整体为 WARN，仅因运行前/运行间未捕获到 `/clock`，该项由 ROS observer 单独验证 | [`doctor.json`](doctor.json)、[`doctor.md`](doctor.md)、[`doctor-live.json`](doctor-live.json) |
| ROS frame/clock | PASS，200条 `/clock` 与200条 status；前20帧 `40937–40956` 严格递增，四字段相等，时间步长0.05，最大误差 `4.933440322929528e-10 s` | [`ros-observer.json`](ros-observer.json)；前序原始现场证据 [`g0-05/live-20260712-030454/observer.json`](../g0-05/live-20260712-030454/observer.json) |
| 唯一 tick master | PASS，所有观察到的 status 声明 `sdf.g0-05.sync` | [`ros-observer.json`](ros-observer.json)、[`g0-05/live-owner-20260712-030603/observer.json`](../g0-05/live-owner-20260712-030603/observer.json) |
| 退出清理 | PASS，driver返回0；退出后无 sync driver、`/clock` publisher或status publisher；CARLA `synchronous_mode=false` | [`post-driver.json`](post-driver.json)、[`g0-05/live-owner-20260712-030603/post_driver.txt`](../g0-05/live-owner-20260712-030603/post_driver.txt) |

## 版本与环境一致性

`doctor.json` 对实际运行环境给出 PASS：Python `3.12.3`、RTX 4080/`16376 MiB`、WSL `2.9.3.0`、Ubuntu-24.04、ROS 2 Jazzy、CARLA 安装路径、2000/2001/2002 端口和 client/server `0.9.16` handshake 均与根 [`versions.lock`](../../../../versions.lock) 及 G0-02/G0-03 基线一致。PyTorch `2.12.1+cu126` 与 torchvision `0.27.1+cu126` 的实测记录见 [`WSL_ROS2_BASELINE.md`](../../WSL_ROS2_BASELINE.md)。

## 复现命令摘要

```bash
./sdf validate-g0 --json-out docs/environment/evidence/g0-06/validation.json --markdown-out docs/environment/evidence/g0-06/validation.md
./sdf sync-smoke --seed 2026 --steps 16 --run-dir docs/environment/evidence/g0-06/smoke/repeat-1
./sdf sync-smoke --seed 2026 --steps 16 --run-dir docs/environment/evidence/g0-06/smoke/repeat-2
./sdf compare docs/environment/evidence/g0-06/smoke/repeat-1/trace.json docs/environment/evidence/g0-06/smoke/repeat-2/trace.json
./sdf sync-smoke --seed 303 --steps 10 --run-dir docs/environment/evidence/g0-06/smoke/recover --interrupt-after 4  # expected rc=75
./sdf sync-smoke --seed 303 --steps 10 --run-dir docs/environment/evidence/g0-06/smoke/recover --resume
./sdf sync-smoke --seed 303 --steps 10 --run-dir docs/environment/evidence/g0-06/smoke/recover-clean
./sdf compare docs/environment/evidence/g0-06/smoke/recover/trace.json docs/environment/evidence/g0-06/smoke/recover-clean/trace.json
CARLA_ROOT=/mnt/e/CARLA_0.9.16 ./sdf doctor
CARLA_ROOT=/mnt/e/CARLA_0.9.16 ./sdf sync-smoke --carla --steps 20 --carla-host 172.30.80.1 --carla-port 2000 --run-dir docs/environment/evidence/g0-06/smoke/live
```

现场 ROS driver 命令及外部权限说明见 [`G0_05_DETERMINISM.md`](../../G0_05_DETERMINISM.md)；本次新增 observer 和 post-driver JSON 只保存汇总结果，不把运行期临时日志纳入 Git。

## 限制与阶段边界

- `sdf doctor` 的 `/clock` 检查是瞬时 topic discoverability 检查，driver快速退出时可返回预期 `WARN`；不能以该 WARN 推翻 ROS observer 的现场对齐证据。
- 当前证据证明 G0 的软件在环基础，不证明 G1 规划、控制或任何后续学习模块。
- G0 阶段关闭和 G1-01 启动仍需用户明确确认，符合 [`G0-06_ACCEPTANCE.md`](../../../../tasks/G0/G0-06_ACCEPTANCE.md) 与 [`ROADMAP.md`](../../../../ROADMAP.md) 的阶段关闭规则。
