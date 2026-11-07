#requires -Version 7.0
# Retrieve completed evidence only. Never launch a solver or profiler.
# OutputDirectory and LocalWslRoot must refer to the same local directory.
param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [Parameter(Mandatory=$true)][string]$RemoteRoot,
    [Parameter(Mandatory=$true)][string]$RemotePython,
    [Parameter(Mandatory=$true)][string]$SshExecutable,
    [Parameter(Mandatory=$true)][string]$ScpExecutable,
    [Parameter(Mandatory=$true)][string]$SshConfig,
    [Parameter(Mandatory=$true)][string]$WslExecutable,
    [Parameter(Mandatory=$true)][string]$WslDistribution,
    [Parameter(Mandatory=$true)][string]$LocalWslPython,
    [Parameter(Mandatory=$true)][string]$LocalWslHelper,
    [Parameter(Mandatory=$true)][string]$LocalWslRoot,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
if (-not $Execute) {
    Write-Output 'PLAN ONLY: read-only checkpoint retrieval, at most40minutes; no network or GPU work'
    return
}
if ($SshHost -notmatch '^[A-Za-z0-9_.@\[\]:-]+$') { throw 'Invalid SSH host' }
$RemoteRoot = $RemoteRoot.TrimEnd('/')
$taskOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
$remoteHelper = "$RemoteRoot/tools/optimization/secondary_checkpoints.py"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.Directory]::CreateDirectory($taskOutput) | Out-Null
$taskWatch = [Diagnostics.Stopwatch]::StartNew()

function ConvertTo-RemoteArgument([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Invoke-BoundedProgram([string]$Executable, [string[]]$Arguments, [int]$TimeoutSeconds) {
    $taskRemainingMs = [Math]::Floor((2400 - $taskWatch.Elapsed.TotalSeconds) * 1000)
    $taskTimeoutMs = [Math]::Min($TimeoutSeconds * 1000, $taskRemainingMs)
    if ($taskTimeoutMs -le 0) { throw 'Bounded checkpoint watch expired' }
    $taskInfo = [Diagnostics.ProcessStartInfo]::new()
    $taskInfo.FileName = $Executable
    $taskInfo.UseShellExecute = $false
    $taskInfo.CreateNoWindow = $true
    $taskInfo.RedirectStandardOutput = $true
    $taskInfo.RedirectStandardError = $true
    foreach ($taskArgument in $Arguments) { [void]$taskInfo.ArgumentList.Add($taskArgument) }
    foreach ($taskName in @($taskInfo.Environment.Keys)) {
        if ($taskName -match '(?i)VAST|API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY') {
            [void]$taskInfo.Environment.Remove($taskName)
        }
    }
    $taskProcess = [Diagnostics.Process]::new()
    $taskProcess.StartInfo = $taskInfo
    $taskStarted = $false
    try {
        [void]$taskProcess.Start()
        $taskStarted = $true
        $taskOutputRead = $taskProcess.StandardOutput.ReadToEndAsync()
        $taskErrorRead = $taskProcess.StandardError.ReadToEndAsync()
        if (-not $taskProcess.WaitForExit([int]$taskTimeoutMs)) {
            throw 'Bounded checkpoint command timed out'
        }
        [void]$taskErrorRead.GetAwaiter().GetResult()
        if ($taskProcess.ExitCode -ne 0) { throw 'Checkpoint command failed' }
        return $taskOutputRead.GetAwaiter().GetResult()
    } finally {
        if ($taskStarted -and -not $taskProcess.HasExited) {
            $taskProcess.Kill($true)
            [void]$taskProcess.WaitForExit(5000)
        }
        $taskProcess.Dispose()
    }
}

$taskSshOptions = @(
    '-F', $SshConfig, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
    '-o', 'StrictHostKeyChecking=yes', '-o', 'ForwardAgent=no', '-o', 'SendEnv=-*'
)
$afterCount = 0
$watchState = @{ phase = 'watching'; verified_evaluations = 0; started_at = [DateTime]::UtcNow.ToString('o') }

while ($taskWatch.Elapsed.TotalSeconds -lt 2400) {
    try {
        $taskRemoteArguments = @($RemotePython, '-B', $remoteHelper, 'snapshot', '--root', $RemoteRoot, '--after', [string]$afterCount)
        $taskRemoteCommand = ($taskRemoteArguments | ForEach-Object { ConvertTo-RemoteArgument $_ }) -join ' '
        $taskResponse = Invoke-BoundedProgram $SshExecutable ($taskSshOptions + @($SshHost, $taskRemoteCommand)) 120
        $metadata = $taskResponse | ConvertFrom-Json
        $watchState['remote_phase'] = $metadata.phase
        $watchState['observed_evaluations'] = $metadata.completed_evaluations
        if ($metadata.archive) {
            if (-not $metadata.archive.StartsWith("$RemoteRoot/") -or $metadata.archive -notmatch '^[A-Za-z0-9_./-]+\.tar\.gz$' -or $metadata.archive.Split('/') -contains '..') {
                throw 'Checkpoint archive must be inside the supplied remote root'
            }
            $archiveName = [System.IO.Path]::GetFileName($metadata.archive)
            $localArchive = Join-Path $taskOutput $archiveName
            [void](Invoke-BoundedProgram $ScpExecutable ($taskSshOptions + @("${SshHost}:$($metadata.archive)", $localArchive)) 300)
            $stream = [System.IO.File]::OpenRead($localArchive)
            try {
                $algorithm = [System.Security.Cryptography.SHA256]::Create()
                $actual = ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
            } finally { $stream.Dispose() }
            if ($actual -ne $metadata.sha256) { throw 'Checkpoint archive checksum mismatch' }
            $folder = $archiveName.Replace('.tar.gz', '')
            $taskVerifyArguments = @('-d', $WslDistribution, '--', $LocalWslPython, $LocalWslHelper, 'verify', '--archive', "$LocalWslRoot/$archiveName", '--out', "$LocalWslRoot/$folder", '--sha256', $metadata.sha256)
            $verification = Invoke-BoundedProgram $WslExecutable $taskVerifyArguments 120
            $proof = $verification | ConvertFrom-Json
            $afterCount = [int]$proof.completed_evaluations
            $watchState['verified_evaluations'] = $afterCount
            $watchState['latest_checkpoint'] = $proof.directory
            $watchState['latest_archive_sha256'] = $metadata.sha256
            $watchState['last_error'] = $null
            if ($metadata.phase -in @('complete', 'failed', 'qualification_failed', 'timed_out')) {
                $watchState['phase'] = 'terminal_evidence_verified'
                $watchState['finished_at'] = [DateTime]::UtcNow.ToString('o')
            }
        }
        $watchState['updated_at'] = [DateTime]::UtcNow.ToString('o')
        [System.IO.File]::WriteAllText((Join-Path $taskOutput 'watcher-status.json'), ($watchState | ConvertTo-Json -Depth 8), $utf8)
        if ($watchState.phase -eq 'terminal_evidence_verified') { break }
    } catch {
        $watchState['last_error'] = $_.Exception.GetType().Name
        $watchState['updated_at'] = [DateTime]::UtcNow.ToString('o')
        [System.IO.File]::WriteAllText((Join-Path $taskOutput 'watcher-status.json'), ($watchState | ConvertTo-Json -Depth 8), $utf8)
    }
    $taskRemainingSeconds = [Math]::Floor(2400 - $taskWatch.Elapsed.TotalSeconds)
    if ($taskRemainingSeconds -gt 0) { Start-Sleep -Seconds ([Math]::Min(300, $taskRemainingSeconds)) }
}
if ($watchState.phase -ne 'terminal_evidence_verified') {
    $watchState['phase'] = 'bounded_watch_expired'
    [System.IO.File]::WriteAllText((Join-Path $taskOutput 'watcher-status.json'), ($watchState | ConvertTo-Json -Depth 8), $utf8)
    exit 1
}
