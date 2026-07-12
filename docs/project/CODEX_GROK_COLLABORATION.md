# Codex + Grok 协作执行规范

## 目标与角色

本方案采用两个彼此独立的本地客户端，不依赖第三方编排插件：

- Codex（当前对话使用 GPT-5.6 Sol、medium）负责读取需求、确定唯一 `GX-XX`、生成任务清单、审查真实差异、独立复测和最终验收。
- Grok CLI 只在隔离 Git worktree 中修改代码并执行任务要求的局部测试。
- Git、PowerShell、WSL 和仓库测试入口是事实来源。Grok 的总结不能替代 `git diff`、测试日志和 Codex 复核。

模型名称和推理强度由 Codex 客户端会话选择，不写进仓库脚本，也不通过非官方接口远程操纵当前 Codex Desktop。开始任务前应在 Codex 客户端确认当前模型为 GPT-5.6 Sol、推理强度为 medium。

## 安全边界

1. 当前主工作区只用于 Codex 分析和验收，不让 Grok 直接修改。
2. 每个任务使用一个 `grok/gx-xx-slug` 分支和一个独立 worktree。
3. `TaskManifest` 必须由 Codex 在 Grok worktree 外创建；其中的 `allowed_paths` 和复测命令在 Grok 执行前冻结。
4. Grok 以单次无交互模式运行，禁用其子流程和跨会话记忆。
5. 独立复测脚本从控制工作区运行，并拒绝 `main`/`master`、范围外文件、G0 冻结路径和 `git diff --check` 错误。
6. 只有 Codex 阅读真实 diff、查看 Grok 日志并确认独立复测通过后，任务才可验收。
7. 脚本不自动提交、不 push、不合并，也不删除 worktree。

## 文件

| 文件 | 用途 |
|---|---|
| `.collab/config.json` | WSL、worktree、Grok 和冻结路径配置 |
| `.collab/task-template.json` | 单任务清单模板 |
| `scripts/collab/Test-CollabEnvironment.ps1` | 环境和真实命令入口诊断 |
| `scripts/collab/New-GxWorktree.ps1` | 创建任务分支和隔离 worktree |
| `scripts/collab/Invoke-GrokTask.ps1` | 受控调用 Grok CLI并保存原始输出 |
| `scripts/collab/Invoke-IndependentVerification.ps1` | Codex 独立范围检查和复测 |
| `.collab/runs/` | 本地运行日志，已被 Git 忽略 |

## 首次检查

在 Windows PowerShell 中运行：

```powershell
Set-Location 'E:\autonomous driving'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\collab\Test-CollabEnvironment.ps1
```

预期 JSON 至少包含 Git 版本、当前分支、Grok 绝对路径、Grok 版本和 `wsl: READY`。如果刚安装 Grok 后 Codex Desktop 找不到命令，重启 Codex Desktop；脚本也会优先使用 `%USERPROFILE%\.grok\bin\grok.exe`，因此不依赖旧 PATH。

Codex CLI 与 Codex Desktop 是不同客户端。此协作方案由当前 Codex Desktop 对话编排，不要求后台再次调用 Codex CLI。需要在 WSL 终端单独使用 Codex CLI 时，应按 OpenAI 官方方式安装并单独登录，不能执行 Desktop 应用目录内的内部 `codex.exe`。

## 标准 GX-XX 流程

### 1. Codex 准备任务清单和提示文件

复制 `.collab/task-template.json` 到控制工作区之外的可信位置，例如：

```text
C:\tmp\safedrive-task-packets\G1-03\task.json
C:\tmp\safedrive-task-packets\G1-03\prompt.md
```

清单必须明确：

- 唯一 `task_id`；
- 基础提交 `base_ref`，必须是创建 worktree 时输出的 40 位 immutable commit SHA，不能写 `HEAD` 或可移动分支名；
- Grok 可修改的 `allowed_paths`；
- Windows 或 WSL 中需要由 Codex 再次执行的完整验证命令；
- 绝对 `prompt_file` 路径。

提示文件应包含目标、输入事实、允许和禁止范围、验收条件、测试命令及停止点。不要把聊天历史、令牌或凭据写入提示文件。

### 2. 创建隔离 worktree

```powershell
$created = powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\collab\New-GxWorktree.ps1 `
  -TaskId G1-03 -Slug map-route -BaseRef 733a0d9494451795dbef89b6efe2342560f9d923 |
  ConvertFrom-Json

$created.worktree
$created.base_sha
```

脚本拒绝重复分支、重复目录和不存在的基础提交。worktree 默认位于 `C:\tmp\safedrive-worktrees`，可在 `.collab/config.json` 修改。

### 3. Grok 编码和局部测试

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\collab\Invoke-GrokTask.ps1 `
  -TaskManifest 'C:\tmp\safedrive-task-packets\G1-03\task.json' `
  -Worktree $created.worktree
```

脚本要求 Grok worktree 在开始时干净、HEAD 与任务清单的 immutable base SHA 一致，拒绝 `main`/`master`，固定使用 `--prompt-file`、JSON 输出、`--no-subagents`、`--no-memory` 和最大回合数。Grok 原始 stdout、stderr、退出码、最终 HEAD 及工作区状态保存在 `.collab/runs/<run-id>/`。如果 Grok 自行提交，运行记录会标记 `commit_detected`，独立验证仍按 base SHA 比较完整差异。

Grok 退出成功只代表编码回合结束，不代表任务通过。

### 4. Codex 检查真实修改

Codex 必须亲自执行并检查：

```powershell
git -C $created.worktree status --short
git -C $created.worktree diff --stat <base-sha> --
git -C $created.worktree diff <base-sha> --
```

检查内容至少包括：任务范围、接口兼容性、安全边界、测试是否被弱化、是否存在临时实现、意外生成文件以及 Grok 报告与真实差异是否一致。

### 5. 独立复测

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\collab\Invoke-IndependentVerification.ps1 `
  -TaskManifest 'C:\tmp\safedrive-task-packets\G1-03\task.json' `
  -Worktree $created.worktree
```

复测脚本先检查实际修改范围和冻结目录，再执行 `git diff --check`，最后逐条运行清单中的 Windows/WSL 命令。任何一条失败都会返回非零退出码，详细 stdout/stderr 保留在独立日志目录。

### 6. 修复与验收

如果审查或复测失败，Codex把具体缺陷、文件位置和重现命令写成新的修复提示。因为已有修改，`Invoke-GrokTask.ps1` 默认拒绝直接开始新回合；Codex应先确认这些修改属于上一回合，再显式提交到任务分支，或创建新的干净修复 worktree。不要为了继续而删除或隐藏差异。

通过后仍由用户或既有项目流程决定是否提交、合并或移除 worktree。本套脚本不会自动执行这些操作。

## WSL 命令约定

任务清单中的 WSL 命令只写 Linux 命令本身，例如：

```json
{
  "environment": "wsl",
  "command": "python3 -m unittest discover -s tests -t . -v",
  "timeout_seconds": 1200
}
```

独立复测脚本负责转换当前 worktree 路径，并按项目约定调用：

```text
wsl.exe -d Ubuntu-24.04 -- /usr/bin/bash -lic '<command>'
```

不得因为 PowerShell 中找不到 Linux 命令，就判断 WSL 中不存在该工具。

## 信任链

```text
用户任务 / START_TASK
  -> Codex 冻结 task.json + prompt.md
  -> Git 隔离分支/worktree
  -> Grok 修改并运行局部测试
  -> Codex 阅读真实 diff 和原始日志
  -> 独立脚本重新执行测试
  -> Codex 按任务验收条件决定通过或修复
```

这条链路的核心不是让两个模型互相口头确认，而是让它们通过不可省略的 Git 差异、独立命令和日志证据交接。
