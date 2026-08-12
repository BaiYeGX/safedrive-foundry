# H Evidence 与归档索引

## 1. 活动 Evidence 规则

H 路线只承认新建且 provenance 完整的 Evidence：

```text
PLANNED → IMPLEMENTED → MEASURED → VERIFIED
```

每个正式 artifact 必须绑定 worktree/commit、config、split、seed、CARLA/模型版本、输入
observation、candidate、Guard、selector、Safety、executed trajectory、outcome、延迟与资源。
没有实际运行的数字不能进入 `MEASURED`，没有冻结复核不能进入 `VERIFIED`。

H0 仅是仓库收敛，不产生驾驶性能数字。H1 尚未开始，当前没有 H World 数据、checkpoint
或 on/off 结果。

## 2. 2026-08-12 路线收敛归档

归档根：

```text
archive/2026-08-12-h-route-consolidation/
```

| 内容 | 状态 | 恢复方式 |
|---|---|---|
| 旧设计文档 | Git 版本化的可恢复历史 | 按 archive README 的原相对路径复制 |
| runtime Evidence/checkpoint | 本机只读，不进入普通 Git | 从本机 `legacy-active/docs/runtime-evidence/` 恢复 |
| 旧候选生成/训练/World 源码 | Git 版本化，不参与活动 import | 从 `legacy-active/source/` 恢复 |
| 旧脚本、配置与测试 | Git 版本化，不是活动入口 | 从对应 `legacy-active/` 子目录恢复 |
| 本轮开始时 tracked dirty worktree | patch 快照 | `recovery/tracked-worktree.patch` |
| 本轮开始时 untracked 文件 | tar 快照 | `recovery/untracked-worktree.tar.gz` |
| 本地生成环境/runtime 输出 | 本机只读，可恢复或重建 | `generated/` |

恢复前必须先阅读
[`archive/2026-08-12-h-route-consolidation/README.md`](../archive/2026-08-12-h-route-consolidation/README.md)，
在临时目录验证，不得直接覆盖活动 H 文件。

工作树快照校验：

```text
tracked-worktree.patch
sha256 1cd44ae9f0f5bcea4589dcbf1f90087259e9d969817110b0521d9155436272aa

untracked-worktree.tar.gz
sha256 0d1fde09623dceb11527bae5cb33ed817630b1a417e5ba78a79011259c828fe7
```

完整归档冻结边界：25097 个常规 payload、3100957261 bytes，另有 868 个符号链接。
`MANIFEST.sha256` 覆盖全部常规 payload；`SYMLINKS.tsv` 冻结符号链接路径与目标：

```text
MANIFEST.sha256
sha256 a313ccaf5e2d37f14901b9830ed29125243d9710f3a7de6b4ceb905fd86dc251

SYMLINKS.tsv
sha256 bff2e26a9c7ff86a55ddb38fd4e2e09482468587e4dfc53c827ec9e4d00c53fb
```

可移植归档为 272 个 legacy 文件、8860549 bytes；
20122 个 runtime Evidence 文件（2960996577 bytes）和 4572 个 generated 文件
（126980647 bytes）只保存在本机。

干净 clone 可以恢复可移植旧源码/文档/配置/测试和两个工作树快照，但不会包含 3GB
历史 runtime Evidence。`evaluation/` 与 `evaluation-remaining/` 的合并恢复规则，以及
迁移到 `tests/hybrid/` 的回归边界，以 archive README 为准。

## 3. 历史材料的解释边界

- archive 保存失败、负收益、冻结阈值和旧实现，内容不重写；
- archive 不是活动任务、接口、数字或门槛来源；
- 历史候选生成失败只支持“该旧方法停止”，不证明 H World 有效；
- 历史 World 无收益只支持“旧数据/旧条件化不足”，不等同于 H3 的结果；
- 若未来需要引用历史事实，必须同时引用原 artifact、状态和限制，不能改名为 H 指标。

## 4. H 当前状态

| 阶段 | Evidence 状态 |
|---|---|
| H0 route consolidation | `VERIFIED / STOPPED` |
| H1 independent candidates | `NOT_IMPLEMENTED` |
| H2 paired outcomes | `NOT_STARTED` |
| H3 World development | `NOT_STARTED` |
| H4 locked evaluation | `NOT_STARTED` |
| H5 World on/off | `NOT_STARTED` |
| H6 closure | `NOT_STARTED` |
