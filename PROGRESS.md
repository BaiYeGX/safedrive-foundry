# SafeDrive Foundry 进度

## 2026-08-12 — H0 路线收敛

状态：`H0_CONSOLIDATED / VERIFIED / STOPPED`。

用户明确要求活动项目只保留 H 路线，并把不再需要的文件移入 `archive/`。本轮已完成：

- 冻结本轮开始时的 tracked patch 与 untracked tar，并记录 SHA256；
- 将旧路线文档、runtime Evidence、候选生成/训练源码、World 源码、阶段脚本、配置和
  测试移动到 `archive/2026-08-12-h-route-consolidation/`；
- 保留 nominal SimLingo policy、通用 trajectory contract、Classic/Safety 与 MPC/PID；
- 移除 nominal runtime 对已归档 learned-candidate 特征/权重导出的依赖；
- 新建 `tests/hybrid/`，把仍有效的 nominal VLA 与控制器回归迁入；
- 将活动入口统一为 H0–H6，并新增 HybridCandidateSet 与 candidate-conditioned World 合同。

归档结果：25097 个常规 payload、3100957261 bytes，另有 868 个符号链接。Git 版本化
272 个可移植 legacy 文件、归档说明、完整 manifest、符号链接表和两个工作树恢复快照；
20122 个 runtime Evidence 文件与 4572 个 generated 文件继续作为本机只读归档，不进入
普通 Git。`MANIFEST.sha256` 的 SHA256 为
`a313ccaf5e2d37f14901b9830ed29125243d9710f3a7de6b4ceb905fd86dc251`，符号链接表为
`bff2e26a9c7ff86a55ddb38fd4e2e09482468587e4dfc53c827ec9e4d00c53fb`。恢复映射、快照
hash、clone 边界与恢复约束已写入归档 README。

验证结果：

- 完整归档 `sha256sum -c MANIFEST.sha256`：25097/25097 个常规 payload 通过；
  `SYMLINKS.tsv` 冻结 868 个符号链接；
- 可移植归档、tracked patch 和解包后的 untracked tar 高置信度凭据扫描：0 个命中文件；
- `PYTHONDONTWRITEBYTECODE=1 /home/sdf/.venvs/sdf/bin/python -m unittest discover
  -s tests/hybrid -t . -v`：81 tests passed，1 个真实 GPU 20 次 forward 按环境门跳过；
- `/home/sdf/.venvs/sdf/bin/python -m compileall -q safedrive_foundry scripts tests`：通过；
- 活动文档本地链接检查：`BROKEN_LOCAL_LINKS 0`；
- 活动树旧路线标识、旧候选/World import 与 Python cache 扫描：无命中；地图测试中的
  OpenDRIVE road-node 主键明确保留；
- `git diff --check`：通过；冻结归档使用 archive 专用 whitespace 属性，未为通过检查而
  改写历史源码或 recovery patch；
- 全活动测试 `PYTHONDONTWRITEBYTECODE=1 /home/sdf/.venvs/sdf/bin/python -m unittest
  discover -s tests -t .`：293 tests 中 291 passed、1 skipped、1 failed。失败是未修改的
  Safety/RATO 静态障碍测试在整套负载下以 47.52 ms 命中 solver deadline；该测试单独
  复跑通过，因此记录为既有时序敏感问题，未通过放宽 deadline 或改测试掩盖。

当前研究事实：

```text
H0 = VERIFIED / STOPPED
H1 = NOT_IMPLEMENTED
H2 = NOT_STARTED
H3 = NOT_STARTED
H4 = NOT_STARTED
H5 = NOT_STARTED
H6 = NOT_STARTED
```

限制：本轮没有启动 CARLA、没有训练模型、没有采集 H 数据，也没有把 archive 中的历史
结果重命名为 H Evidence。完整活动测试仍有上述一个时序敏感失败；H 专项回归与静态
验证均通过。

下一步只允许在新任务中实施 H1；不得自动进入 H2 或 World 训练。
