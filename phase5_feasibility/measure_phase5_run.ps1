# Phase 5 feasibility measurement harness.
#
# STRICTLY ISOLATED from the LongCat parity investigation. This script only
# launches an existing llama-debug binary and samples OS/GPU counters. It never
# touches LongCat arithmetic, parity instrumentation, frozen artifacts, model
# files, or the authoritative capture procedure.
#
# Counters use Win32_PerfRawData_* CIM classes rather than Get-Counter paths,
# because Get-Counter path names are localized and this machine is not on an
# English locale.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]   $Label,
    [Parameter(Mandatory = $true)][string]   $Exe,
    [Parameter(Mandatory = $true)][string[]] $ExeArgs,
    [Parameter(Mandatory = $true)][string]   $OutDir,
    [string] $DumpDir      = $null,
    [int]    $SampleMs     = 250,
    [int]    $TimeoutSec   = 5400
)

$ErrorActionPreference = 'Stop'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stdoutPath  = Join-Path $OutDir "$Label.stdout.log"
$stderrPath  = Join-Path $OutDir "$Label.stderr.log"
$samplePath  = Join-Path $OutDir "$Label.samples.csv"
$reportPath  = Join-Path $OutDir "$Label.report.json"

function Get-MemSnapshot {
    $os  = Get-CimInstance Win32_OperatingSystem
    $mem = Get-CimInstance Win32_PerfRawData_PerfOS_Memory
    [pscustomobject]@{
        # Win32_OperatingSystem reports KB.
        PhysTotalMB   = [math]::Round($os.TotalVisibleMemorySize / 1KB, 1)
        PhysFreeMB    = [math]::Round($os.FreePhysicalMemory     / 1KB, 1)
        CommitTotalMB = [math]::Round($os.TotalVirtualMemorySize / 1KB, 1)
        CommitFreeMB  = [math]::Round($os.FreeVirtualMemory      / 1KB, 1)
        PageReads     = [uint64]$mem.PageReadsPerSec      # cumulative count
        PagesInput    = [uint64]$mem.PagesInputPerSec     # cumulative count
    }
}

function Get-DiskReadBytes {
    $d = Get-CimInstance Win32_PerfRawData_PerfDisk_LogicalDisk -Filter "Name='C:'"
    if ($null -eq $d) { return [uint64]0 }
    return [uint64]$d.DiskReadBytesPerSec   # cumulative in PerfRawData
}

function Get-GpuMem {
    param([int]$TargetPid)
    $total = 0; $proc = 0
    try {
        $g = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        if ($g) { $total = [int]($g | Select-Object -First 1) }
        $apps = & nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>$null
        foreach ($line in $apps) {
            $parts = $line -split ',\s*'
            if ($parts.Count -ge 2 -and [int]$parts[0] -eq $TargetPid) { $proc = [int]$parts[1] }
        }
    } catch { }
    return [pscustomobject]@{ TotalMB = $total; ProcMB = $proc }
}

# ---------------------------------------------------------------- baseline
$gpuIdle  = Get-GpuMem -TargetPid 0
$memStart = Get-MemSnapshot
$diskStart = Get-DiskReadBytes

Write-Host "=== Phase 5 run: $Label ===" -ForegroundColor Cyan
Write-Host "exe  : $Exe"
Write-Host "args : $($ExeArgs -join ' ')"
Write-Host "idle GPU used = $($gpuIdle.TotalMB) MiB ; phys free = $($memStart.PhysFreeMB) MB"

if ($DumpDir) { $env:LONGCAT_HIDDEN_DUMP_DIR = $DumpDir }

$sw = [System.Diagnostics.Stopwatch]::StartNew()

$proc = Start-Process -FilePath $Exe -ArgumentList $ExeArgs -PassThru -NoNewWindow `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

# Touching .Handle caches the process handle so .ExitCode is still readable
# after the process exits; without this PowerShell returns $null.
$null = $proc.Handle

$samples = New-Object System.Collections.Generic.List[object]
$peak = [pscustomobject]@{
    WorkingSetMB = 0.0; PrivateMB = 0.0; GpuProcMB = 0; GpuTotalMB = 0
    CommitUsedMB = 0.0; PhysUsedMB = 0.0
}

while (-not $proc.HasExited) {
    if ($sw.Elapsed.TotalSeconds -gt $TimeoutSec) {
        Write-Host "TIMEOUT after $TimeoutSec s - killing" -ForegroundColor Red
        try { $proc.Kill() } catch { }
        break
    }

    try { $proc.Refresh() } catch { break }

    $ws = 0.0; $pv = 0.0
    try {
        $ws = [math]::Round($proc.WorkingSet64        / 1MB, 1)
        $pv = [math]::Round($proc.PrivateMemorySize64 / 1MB, 1)
    } catch { }

    $m   = Get-MemSnapshot
    $gpu = Get-GpuMem -TargetPid $proc.Id

    $commitUsed = [math]::Round($m.CommitTotalMB - $m.CommitFreeMB, 1)
    $physUsed   = [math]::Round($m.PhysTotalMB   - $m.PhysFreeMB,   1)

    if ($ws        -gt $peak.WorkingSetMB) { $peak.WorkingSetMB = $ws }
    if ($pv        -gt $peak.PrivateMB)    { $peak.PrivateMB    = $pv }
    if ($gpu.ProcMB  -gt $peak.GpuProcMB)  { $peak.GpuProcMB    = $gpu.ProcMB }
    if ($gpu.TotalMB -gt $peak.GpuTotalMB) { $peak.GpuTotalMB   = $gpu.TotalMB }
    if ($commitUsed  -gt $peak.CommitUsedMB) { $peak.CommitUsedMB = $commitUsed }
    if ($physUsed    -gt $peak.PhysUsedMB)   { $peak.PhysUsedMB   = $physUsed }

    $samples.Add([pscustomobject]@{
        t_s          = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        ws_mb        = $ws
        private_mb   = $pv
        gpu_proc_mb  = $gpu.ProcMB
        gpu_total_mb = $gpu.TotalMB
        phys_used_mb = $physUsed
        commit_mb    = $commitUsed
        page_reads   = $m.PageReads
        pages_input  = $m.PagesInput
        disk_read_b  = Get-DiskReadBytes
    })

    Start-Sleep -Milliseconds $SampleMs
}

try { $proc.WaitForExit() } catch { }
$sw.Stop()

if ($DumpDir) { Remove-Item Env:LONGCAT_HIDDEN_DUMP_DIR -ErrorAction SilentlyContinue }

$memEnd   = Get-MemSnapshot
$diskEnd  = Get-DiskReadBytes

$samples | Export-Csv -Path $samplePath -NoTypeInformation -Encoding ASCII

# ------------------------------------------------- parse llama.cpp timings
$loadSec = $null; $promptSec = $null; $promptTps = $null; $nEval = $null
if (Test-Path $stderrPath) {
    foreach ($line in (Get-Content $stderrPath)) {
        if ($line -match 'load time\s*=\s*([0-9.]+)\s*ms')            { $loadSec   = [double]$Matches[1] / 1000.0 }
        if ($line -match 'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second') {
            $promptSec = [double]$Matches[1] / 1000.0
            $nEval     = [int]$Matches[2]
            $promptTps = [double]$Matches[3]
        }
    }
}

$report = [ordered]@{
    label                 = $Label
    timestamp             = (Get-Date).ToString('o')
    exe                   = $Exe
    args                  = ($ExeArgs -join ' ')
    dump_dir              = $DumpDir
    exit_code             = $proc.ExitCode
    completed             = ($proc.ExitCode -eq 0)
    wall_seconds          = [math]::Round($sw.Elapsed.TotalSeconds, 2)
    llama_load_seconds    = $loadSec
    llama_prompt_seconds  = $promptSec
    llama_prompt_tokens   = $nEval
    llama_prompt_tps      = $promptTps
    peak_gpu_proc_mib     = $peak.GpuProcMB
    peak_gpu_total_mib    = $peak.GpuTotalMB
    gpu_idle_baseline_mib = $gpuIdle.TotalMB
    peak_working_set_mb   = $peak.WorkingSetMB
    peak_private_mb       = $peak.PrivateMB
    peak_phys_used_mb     = $peak.PhysUsedMB
    phys_total_mb         = $memStart.PhysTotalMB
    peak_commit_used_mb   = $peak.CommitUsedMB
    commit_limit_mb       = $memStart.CommitTotalMB
    hard_page_reads       = [int64]($memEnd.PageReads  - $memStart.PageReads)
    pages_input           = [int64]($memEnd.PagesInput - $memStart.PagesInput)
    disk_read_bytes       = [int64]($diskEnd - $diskStart)
    disk_read_gib         = [math]::Round(($diskEnd - $diskStart) / 1GB, 3)
    sample_count          = $samples.Count
}

if ($report.wall_seconds -gt 0) {
    $report.disk_read_avg_mib_s = [math]::Round(($diskEnd - $diskStart) / 1MB / $report.wall_seconds, 1)
}

$report | ConvertTo-Json -Depth 4 | Set-Content -Path $reportPath -Encoding ASCII

Write-Host ""
Write-Host "--- $Label ---" -ForegroundColor Green
$report.GetEnumerator() | ForEach-Object { "{0,-24} {1}" -f $_.Key, $_.Value }
Write-Host ""
Write-Host "samples -> $samplePath"
Write-Host "report  -> $reportPath"
