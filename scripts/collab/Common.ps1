Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RepositoryRoot {
    param([string]$StartPath = (Get-Location).Path)
    $root = (& git -C $StartPath rev-parse --show-toplevel 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Not inside a Git repository: $StartPath`n$root" }
    return [System.IO.Path]::GetFullPath(($root | Select-Object -First 1))
}

function Get-CollabConfig {
    param([string]$RepositoryRoot)
    $path = Join-Path $RepositoryRoot '.collab\config.json'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing collaboration config: $path" }
    return Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Resolve-GrokExecutable {
    param($Config)
    foreach ($candidateValue in $Config.grok_candidates) {
        $candidate = [Environment]::ExpandEnvironmentVariables([string]$candidateValue)
        if ([System.IO.Path]::IsPathRooted($candidate)) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
            continue
        }
        $command = Get-Command $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $command) { return $command.Source }
    }
    throw 'Grok CLI was not found. Restart the terminal after installation or update .collab/config.json.'
}

function Convert-ToWslPath {
    param([Parameter(Mandatory)][string]$WindowsPath)
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') { throw "Cannot convert path to WSL form: $full" }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

function Test-RelativePathAllowed {
    param([string]$RelativePath, [object[]]$AllowedPaths)
    $normalized = $RelativePath.Replace('\', '/').TrimStart('./')
    foreach ($allowedValue in $AllowedPaths) {
        $allowed = ([string]$allowedValue).Replace('\', '/').TrimStart('./')
        if ($allowed.EndsWith('/')) {
            if ($normalized.StartsWith($allowed, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
        } elseif ($normalized.Equals($allowed, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Invoke-CapturedProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][int]$TimeoutSeconds,
        [Parameter(Mandatory)][string]$StdoutPath,
        [Parameter(Mandatory)][string]$StderrPath
    )
    function ConvertTo-ProcessArgument([string]$Value) {
        if ($Value -notmatch '[\s"]') { return $Value }
        $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
        $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
        return '"' + $escaped + '"'
    }

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Arguments = (($ArgumentList | ForEach-Object { ConvertTo-ProcessArgument ([string]$_) }) -join ' ')
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { throw "Could not start process: $FilePath" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        & taskkill.exe /PID $process.Id /T /F *> $null
        $process.WaitForExit()
        $stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $StdoutPath -Encoding UTF8
        $stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $StderrPath -Encoding UTF8
        $process.Dispose()
        throw "Process timed out after $TimeoutSeconds seconds: $FilePath"
    }
    $process.WaitForExit()
    $stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $StdoutPath -Encoding UTF8
    $stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $StderrPath -Encoding UTF8
    $exitCode = [int]$process.ExitCode
    $process.Dispose()
    return $exitCode
}
