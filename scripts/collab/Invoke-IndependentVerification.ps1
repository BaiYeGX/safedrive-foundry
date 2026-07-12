[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$TaskManifest,
    [Parameter(Mandatory)][string]$Worktree,
    [string]$ControlRepositoryRoot = ''
)

. "$PSScriptRoot\Common.ps1"
if (-not $ControlRepositoryRoot) { $ControlRepositoryRoot = Get-RepositoryRoot -StartPath $PSScriptRoot }
$config = Get-CollabConfig -RepositoryRoot $ControlRepositoryRoot
$manifest = Get-Content -LiteralPath (Resolve-Path -LiteralPath $TaskManifest).Path -Raw -Encoding UTF8 | ConvertFrom-Json
$worktree = (Resolve-Path -LiteralPath $Worktree).Path
$branch = ((& git -C $worktree branch --show-current 2>&1) -join '').Trim()
if ($branch -in @('main','master')) { throw "Refusing to verify mutable work on protected branch: $branch" }

$baseRef = if ($manifest.base_ref) { [string]$manifest.base_ref } else { '' }
if ($baseRef -notmatch '^[0-9a-fA-F]{40}$') { throw 'Manifest base_ref must be an immutable 40-character Git commit SHA.' }
$changedFiles = @(& git -C $worktree diff --name-only $baseRef --; & git -C $worktree ls-files --others --exclude-standard)
$changedFiles = @($changedFiles | Where-Object { $_ } | Sort-Object -Unique)
if ($changedFiles.Count -eq 0) { throw 'No changed files found to verify.' }

$violations = @()
foreach ($path in $changedFiles) {
    if (-not (Test-RelativePathAllowed -RelativePath $path -AllowedPaths $manifest.allowed_paths)) { $violations += "outside allowed_paths: $path" }
    if (Test-RelativePathAllowed -RelativePath $path -AllowedPaths $config.protected_paths) { $violations += "protected path: $path" }
}
if ($violations.Count -gt 0) { throw "Scope verification failed:`n$($violations -join "`n")" }

& git -C $worktree diff --check $baseRef --
if ($LASTEXITCODE -ne 0) { throw 'git diff --check failed.' }

$runId = "verify-{0}-{1}" -f $manifest.task_id, (Get-Date -Format 'yyyyMMdd-HHmmss')
$runDir = Join-Path (Join-Path $ControlRepositoryRoot ([string]$config.run_directory)) $runId
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$results = @()
$index = 0
foreach ($check in $manifest.verification) {
    $index++
    $stdout = Join-Path $runDir ("check-{0:D2}.stdout.log" -f $index)
    $stderr = Join-Path $runDir ("check-{0:D2}.stderr.log" -f $index)
    $timeout = if ($check.timeout_seconds) { [int]$check.timeout_seconds } else { 600 }
    if ([string]$check.environment -eq 'windows') {
        $code = Invoke-CapturedProcess -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-NonInteractive','-Command',[string]$check.command) `
            -WorkingDirectory $worktree -TimeoutSeconds $timeout -StdoutPath $stdout -StderrPath $stderr
    } elseif ([string]$check.environment -eq 'wsl') {
        $wslPath = Convert-ToWslPath -WindowsPath $worktree
        $quotedPath = "'" + $wslPath.Replace("'", "'\''") + "'"
        $bashCommand = "cd $quotedPath && $([string]$check.command)"
        $code = Invoke-CapturedProcess -FilePath 'wsl.exe' -ArgumentList @('-d',[string]$config.wsl_distribution,'--','/usr/bin/bash','-lic',$bashCommand) `
            -WorkingDirectory $worktree -TimeoutSeconds $timeout -StdoutPath $stdout -StderrPath $stderr
    } else { throw "Unknown verification environment: $($check.environment)" }
    $results += [ordered]@{ environment=$check.environment; command=$check.command; exit_code=$code; stdout=$stdout; stderr=$stderr }
    if ($code -ne 0) { break }
}

$passed = ($results.Count -eq @($manifest.verification).Count) -and (@($results | Where-Object { $_.exit_code -ne 0 }).Count -eq 0)
$summary = [ordered]@{ run_id=$runId; task_id=$manifest.task_id; branch=$branch; base_ref=$baseRef; changed_files=$changedFiles; passed=$passed; checks=$results }
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runDir 'verification.json') -Encoding UTF8
$summary | ConvertTo-Json -Depth 8
if (-not $passed) { exit 1 }
