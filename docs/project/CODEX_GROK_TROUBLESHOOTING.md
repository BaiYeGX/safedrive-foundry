# Windows、WSL、Codex 与 Grok 常见问题

## 手动输入 `grok` 可用，但 Codex 后台找不到

本机实际发生过的原因是：Grok 已安装到 `%USERPROFILE%\.grok\bin\grok.exe`，用户级 PATH 也已经更新，但 Codex Desktop 是在安装前启动的，仍保留旧的进程环境变量。新开的终端读取新 PATH，所以表现不同。

处理：

1. 关闭并重新启动 Codex Desktop；
2. 用 `Test-CollabEnvironment.ps1` 检查；
3. 自动化脚本固定优先检查 `%USERPROFILE%\.grok\bin\grok.exe`，不把 PATH 当成唯一依据。

不要反复重装 Grok，也不要把程序复制进项目目录。

## `Get-Command codex` 有路径，但执行返回 Access denied

如果路径类似：

```text
C:\Program Files\WindowsApps\OpenAI.Codex_*\app\resources\codex.exe
```

它是 Codex Desktop 随应用打包的内部文件，不应当作独立 CLI 入口。WindowsApps 的应用权限和沙盒可能允许解析路径，却拒绝普通后台进程直接执行。

处理：

- 当前方案直接由 Codex Desktop 对话负责审查和编排，不需要让脚本递归启动另一个 Codex；
- 如需独立 Codex CLI，在 WSL 中按 OpenAI 官方文档安装 `@openai/codex` 并单独登录；
- 不修改 WindowsApps ACL，不获取该目录所有权，也不建立指向内部二进制的包装脚本。

## Codex 调用 WSL 挂起，但用户终端中的 WSL 正常

这通常说明 WSL 本身已安装，但调用发生在不同的后台进程、沙盒或登录上下文中。也可能是发行版首次启动、shell 初始化脚本、映射盘访问或交互提示阻塞。

按以下顺序诊断：

```powershell
wsl -l -v
wsl -d Ubuntu-24.04 -- /usr/bin/bash -lc 'printf READY'
wsl -d Ubuntu-24.04 -- /usr/bin/bash -lic 'pwd; command -v git; command -v python3'
```

- 第二条失败：检查发行版名称、首次启动和 WSL 服务状态；
- 第二条成功而第三条挂起：检查 `~/.bashrc`、`~/.profile` 等初始化文件是否执行了交互命令；
- 用户终端成功而 Codex 后台失败：重启 Codex Desktop，再运行环境脚本；不要据此重新安装 WSL；
- 命令需要登录 shell 环境时使用 `bash -lic`，纯健康检查使用 `bash -lc`，避免无谓加载复杂初始化。

自动化命令必须设置超时，不能无限等待 WSL。

## PowerShell 读取中文 Markdown 出现乱码

Windows PowerShell 5.1 对无 BOM UTF-8 文件的默认解码可能不正确。文件通常没有损坏，只是读取方式错误。

使用：

```powershell
Get-Content -LiteralPath .\AGENTS.md -Encoding UTF8
```

脚本读取 JSON 和 Markdown 时应显式指定 `-Encoding UTF8`。不要看到终端乱码就批量重写文档，否则可能真的破坏编码。

## Grok 返回成功，是否代表任务完成

不代表。成功退出只能证明 Grok 进程结束。仍必须：

1. 查看 `git status --short`；
2. 查看相对固定基础提交的完整 `git diff`；
3. 检查修改是否越过 `allowed_paths` 或 G0 冻结路径；
4. 由控制工作区中的独立脚本重新执行验证；
5. 由 Codex 对照任务验收条件给出结论。

Grok 的文字总结、自己生成的测试结果或自己修改过的验证脚本都不能单独成为验收证据。

任务清单中的 `base_ref` 必须固定为 40 位 commit SHA。使用 `HEAD` 会在 Grok 自行提交后随 worktree 移动，导致比较基线不再可靠；脚本会直接拒绝这种清单。

## Grok 启动时报告 `settings.json` 格式损坏

本机烟雾测试还观察到 Grok 的用户级启动警告：

```text
malformed settings.json ... expected value at line 1 column 1
```

这指向 `C:\Users\张周易\.claude\settings.json`，不属于 SafeDrive Foundry 仓库，也不是 Grok CLI 本体的版本检查失败。它可能导致用户级 hook 或命令加载失败，但本次 Grok 单回合仍然能够返回结果并保持 worktree 干净。

处理时先在用户终端备份该文件，再用 JSON 校验器定位首个非法字符；不要把空文件、日志或命令输出直接当作 `settings.json`。如果不依赖这些用户级 hook，可以暂时移走它后重启 Grok；如果依赖它，按 Grok/Claude 配置格式逐项恢复。项目协作脚本不会修改此文件。

同一启动日志中的 `skill name does not match expected name from path` 是用户级命令文件命名/声明不一致，和项目 worktree 隔离无关；需要单独修正命令名或其声明，不应通过放宽项目验证器来掩盖。

## 为什么不用未知插件或模型内部子流程

该项目需要可复现、可审计并能在 Windows/WSL 上恢复。Git worktree、PowerShell、WSL、官方 Codex 客户端和已安装的 Grok CLI 都有明确边界；第三方编排插件会额外引入权限、维护状态和协议漂移风险。因此控制层只调用本地 CLI 和标准 Git，不依赖来源不明的插件，也禁用 Grok 自己继续派生执行流程。

## 日志在哪里

Grok 和独立复测日志默认保存在控制工作区的：

```text
.collab/runs/<run-id>/
```

该目录被 Git 忽略。日志可能包含代码片段和本地路径，分享前应检查；提示文件和日志中不得放入 API key、OAuth token 或其他凭据。

## 恢复检查清单

中断后不要直接重新运行 Grok。先检查：

```powershell
git -C <worktree> branch --show-current
git -C <worktree> status --short
git -C <worktree> diff --stat <base-sha> --
git worktree list
```

然后读取最近的 `.collab/runs/*/grok-run.json` 或 `verification.json`。确认基础提交、任务清单、允许路径和外部环境仍然成立，再决定复测、修复或人工接管。
