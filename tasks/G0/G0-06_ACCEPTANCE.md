# G0-06：G0集成验收与证据包

**状态**：COMPLETED
**依赖**：G0-01～G0-05全部完成

## 目标

从干净终端复现G0完整流程，确认环境、CARLA、ROS、跨系统连通、确定性、版本锁和诊断入口满足门禁，并形成可追溯Evidence Bundle。

## 范围

- 汇总和验证已有G0产物，不新增业务功能。
- 只修复小型文档或配置问题；实质缺陷退回对应任务。
- 形成验收报告和拆分G1所需的事实基线。

## 完成标准

- Windows/WSL2/CARLA/ROS 2烟雾测试通过。
- 固定步长、tick master和frame对齐通过重复性验证。
- `versions.lock`与实际环境一致。
- `sdf doctor`可复现检查结果。
- G0任务、断点、日志和证据路径完整。
- 用户确认关闭G0后才允许拆分G1。

## 验证方法

- 从干净终端按README执行完整复现流程。
- 重复确定性测试并比较结果。
- 校验Evidence Bundle中的版本、hash、日志、报告和链接。
- 更新`PROGRESS.md`，汇报后停止。

## 断点记录

| 字段 | 内容 |
|---|---|
| 最后状态 | 2026-07-12 `COMPLETED` |
| 已完成 | 汇总G0-01～G0-05并从无profile/rc终端复现离线契约、重复性、checkpoint恢复、CARLA live tick、doctor、ROS frame/clock对齐和退出清理；生成验收报告、JSON证据与SHA-256 manifest。 |
| 修改文件 | `docs/environment/evidence/g0-06/acceptance.md`、`manifest.json`、`doctor*.json/md`、`validation.json/md`、`ros-observer.json`、`post-driver.json`、`smoke/*`；本任务文件、`PROGRESS.md`。前序G0-05未提交改动均保留。 |
| 已运行验证 | `colcon build --symlink-install`：1 package finished；`validate-g0` 14/14；seed 2026 两次16帧 compare PASS；seed 303 中断返回75、恢复与clean 10帧 compare PASS；CARLA 0.9.16 live 20帧 PASS；doctor版本锁/GPU/WSL/ROS/CARLA/端口检查 PASS（仅瞬时`/clock`为预期WARN）；ROS observer 200/200、frame/clock全对齐、最大误差`4.933440322929528e-10 s`、driver rc=0；退出后无节点/topic publisher且WorldSettings恢复；compileall、unittest 5/5、manifest hash和Markdown链接检查通过。 |
| 阻塞 | 无完成阻塞。G0阶段关闭仍需用户确认，这是路线图流程门禁，不自动开始G1。 |
| 恢复步骤 | 无需恢复；如需审计，先校验`docs/environment/evidence/g0-06/manifest.json`，再按`acceptance.md`复现实验。用户确认关闭G0后，另行明确启动G1-01。 |
| 下一建议命令 | 等待用户确认关闭G0；不要自动执行G1-01。 |
