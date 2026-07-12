[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][ValidatePattern('^G[1-8]-\d{2}$')][string]$TaskId,
    [Parameter(Mandatory)][ValidatePattern('^[a-z0-9][a-z0-9-]*$')][string]$Slug,
    [string]$BaseRef = 'HEAD',
    [string]$RepositoryRoot = ''
)

. "$PSScriptRoot\Common.ps1"
if (-not $RepositoryRoot) { $RepositoryRoot = Get-RepositoryRoot -StartPath $PSScriptRoot }
$config = Get-CollabConfig -RepositoryRoot $RepositoryRoot
$branch = "$($config.branch_prefix)$($TaskId.ToLowerInvariant())-$Slug"
$worktree = Join-Path ([Environment]::ExpandEnvironmentVariables([string]$config.worktree_root)) "$TaskId-$Slug"

& git -C $RepositoryRoot rev-parse --verify "$BaseRef^{commit}" *> $null
if ($LASTEXITCODE -ne 0) { throw "Base ref does not resolve to a commit: $BaseRef" }
$baseSha = ((& git -C $RepositoryRoot rev-parse "$BaseRef^{commit}" 2>&1) -join '').Trim()
& git -C $RepositoryRoot show-ref --verify --quiet "refs/heads/$branch"
if ($LASTEXITCODE -eq 0) { throw "Branch already exists: $branch" }
if (Test-Path -LiteralPath $worktree) { throw "Worktree path already exists: $worktree" }

$parent = Split-Path -Parent $worktree
if ($PSCmdlet.ShouldProcess($worktree, "Create branch $branch from $BaseRef and add worktree")) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    & git -C $RepositoryRoot worktree add -b $branch $worktree $BaseRef
    if ($LASTEXITCODE -ne 0) { throw 'git worktree add failed.' }
}

[ordered]@{ task_id=$TaskId; branch=$branch; worktree=$worktree; base_ref=$BaseRef; base_sha=$baseSha } | ConvertTo-Json
