[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$TaskManifest,
    [Parameter(Mandatory)][string]$Worktree,
    [string]$ControlRepositoryRoot = ''
)

. "$PSScriptRoot\Common.ps1"
if (-not $ControlRepositoryRoot) { $ControlRepositoryRoot = Get-RepositoryRoot -StartPath $PSScriptRoot }
$config = Get-CollabConfig -RepositoryRoot $ControlRepositoryRoot
$manifestPath = (Resolve-Path -LiteralPath $TaskManifest).Path
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$worktree = (Resolve-Path -LiteralPath $Worktree).Path

if ([string]$manifest.task_id -notmatch '^G[1-8]-\d{2}$') { throw 'Manifest task_id must match G1-01 through G8-99.' }
if ([string]$manifest.base_ref -notmatch '^[0-9a-fA-F]{40}$') { throw 'Manifest base_ref must be an immutable 40-character Git commit SHA.' }
$baseSha = ((& git -C $ControlRepositoryRoot rev-parse "$($manifest.base_ref)^{commit}" 2>&1) -join '').Trim()
if ($baseSha -ne ([string]$manifest.base_ref).ToLowerInvariant()) { throw "Manifest base_ref does not resolve to itself: $($manifest.base_ref)" }
$branch = ((& git -C $worktree branch --show-current 2>&1) -join '').Trim()
if (-not $branch) { throw "Not a Git worktree: $worktree" }
if ($branch -in @('main','master')) { throw "Refusing to run Grok on protected branch: $branch" }
$worktreeHead = ((& git -C $worktree rev-parse HEAD 2>&1) -join '').Trim()
if ($worktreeHead.ToLowerInvariant() -ne $baseSha.ToLowerInvariant()) { throw "Worktree HEAD does not match manifest base_ref: $worktreeHead" }
$dirty = @(& git -C $worktree status --porcelain)
if ($dirty.Count -gt 0) { throw 'Grok worktree must be clean before a new run.' }

$promptPath = (Resolve-Path -LiteralPath ([string]$manifest.prompt_file)).Path
$grok = Resolve-GrokExecutable -Config $config
$runId = "{0}-{1}" -f $manifest.task_id, (Get-Date -Format 'yyyyMMdd-HHmmss')
$runRoot = Join-Path $ControlRepositoryRoot ([string]$config.run_directory)
$runDir = Join-Path $runRoot $runId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$stdout = Join-Path $runDir 'grok.stdout.json'
$stderr = Join-Path $runDir 'grok.stderr.log'
$maxTurns = if ($manifest.max_turns) { [int]$manifest.max_turns } else { [int]$config.grok.max_turns }
if ($maxTurns -lt 1 -or $maxTurns -gt 100) { throw "max_turns must be between 1 and 100: $maxTurns" }
$args = @('--cwd', $worktree, '--prompt-file', $promptPath, '--output-format', [string]$config.grok.output_format,
    '--permission-mode', [string]$config.grok.permission_mode, '--max-turns', [string]$maxTurns)
if ([bool]$config.grok.disable_subagents) { $args += '--no-subagents' }
if ([bool]$config.grok.disable_memory) { $args += '--no-memory' }
if ([bool]$config.grok.always_approve) { $args += '--always-approve' }

$exitCode = Invoke-CapturedProcess -FilePath $grok -ArgumentList $args -WorkingDirectory $worktree `
    -TimeoutSeconds 7200 -StdoutPath $stdout -StderrPath $stderr
$changed = @(& git -C $worktree status --porcelain)
$headAfter = ((& git -C $worktree rev-parse HEAD 2>&1) -join '').Trim()
$stopReason = ''
try {
    $grokJson = Get-Content -LiteralPath $stdout -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    $stopReason = [string]$grokJson.stopReason
} catch {
    $stopReason = 'UNPARSEABLE_OUTPUT'
}
$result = [ordered]@{ run_id=$runId; task_id=$manifest.task_id; base_sha=$baseSha; branch=$branch; worktree=$worktree; head_after=$headAfter; commit_detected=($headAfter -ne $baseSha); exit_code=$exitCode; stop_reason=$stopReason; changed=$changed; stdout=$stdout; stderr=$stderr }
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'grok-run.json') -Encoding UTF8
$result | ConvertTo-Json -Depth 6
if ($exitCode -ne 0 -or $stopReason -notin @('EndTurn','Stop')) { exit 1 }
