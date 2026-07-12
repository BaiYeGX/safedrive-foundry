[CmdletBinding()]
param([string]$RepositoryRoot = '')

. "$PSScriptRoot\Common.ps1"
if (-not $RepositoryRoot) { $RepositoryRoot = Get-RepositoryRoot -StartPath $PSScriptRoot }
$config = Get-CollabConfig -RepositoryRoot $RepositoryRoot
$results = [ordered]@{}

$results.powershell = $PSVersionTable.PSVersion.ToString()
$results.git = (& git --version 2>&1 | Select-Object -First 1)
$results.branch = (& git -C $RepositoryRoot branch --show-current 2>&1 | Select-Object -First 1)
$results.grok_path = Resolve-GrokExecutable -Config $config
$results.grok_version = (& $results.grok_path --version 2>&1 | Select-Object -First 1)
$results.wsl_distribution = [string]$config.wsl_distribution

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'safedrive-collab'
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
$wslStdout = Join-Path $tempRoot 'wsl-health.stdout.log'
$wslStderr = Join-Path $tempRoot 'wsl-health.stderr.log'
try {
    $wslExit = Invoke-CapturedProcess -FilePath 'wsl.exe' `
        -ArgumentList @('-d',[string]$config.wsl_distribution,'--','/usr/bin/bash','-lc','printf READY') `
        -WorkingDirectory $RepositoryRoot -TimeoutSeconds 20 -StdoutPath $wslStdout -StderrPath $wslStderr
    $wslOutput = if (Test-Path -LiteralPath $wslStdout) { Get-Content -LiteralPath $wslStdout -Raw -ErrorAction SilentlyContinue } else { '' }
    $wslError = if (Test-Path -LiteralPath $wslStderr) { Get-Content -LiteralPath $wslStderr -Raw -ErrorAction SilentlyContinue } else { '' }
    $results.wsl = if ($wslExit -eq 0 -and $wslOutput -eq 'READY') { 'READY' } else { "FAILED (exit $wslExit): $wslError" }
} catch {
    $results.wsl = "FAILED: $($_.Exception.Message)"
}
$results | ConvertTo-Json -Depth 5

if ($results.wsl -ne 'READY') { exit 2 }
