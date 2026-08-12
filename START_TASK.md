# 当前唯一任务：H0 路线合并与可恢复归档

## 状态

```text
H0 route consolidation = VERIFIED / STOPPED
H1 hybrid candidate contract = NOT_IMPLEMENTED
World training/runtime = NOT_AUTHORIZED
CARLA live run = NOT_REQUIRED
```

## 目标

把活动项目收敛为一套 H 路线：

1. 活动入口、设计文档与进度只使用 H0–H6；
2. 保留 nominal VLA、Classic Expert、Guard/Safety、MPC/PID 的可复用实现；
3. 旧候选生成、旧 World 实验、旧脚本/测试/配置和冻结 Evidence 移入
   `archive/2026-08-12-h-route-consolidation/`；
4. 归档保留原路径说明、工作树快照、校验值和恢复方法；
5. 不实现 H1，不启动训练或 CARLA。

## 允许范围

- 根入口文档、`docs/`、维护脚本；
- `safedrive_foundry/driving_vla/` 中的路线收敛与失效依赖移除；
- `tests/hybrid/` 的 nominal VLA/控制器回归整理；
- `archive/2026-08-12-h-route-consolidation/` 的可恢复归档。

## 验收

- 活动路线文档只列 H0–H6；
- 活动源码不再导入已归档候选生成或旧 World 模块；
- 项目入口不存在指向已归档活动文件的链接；
- `tests/hybrid` 通过；`compileall` 与 `git diff --check` 通过；
- 旧路线标识扫描除地图 road node fixture 外无活动命中；
- `PROGRESS.md` 与 `docs/EVIDENCE.md` 记录归档事实与验证结果。

## 停止点

验收已完成，本轮在此停止。下一步是 `H1`，但必须由后续任务明确授权；本轮不得
自动实施。
