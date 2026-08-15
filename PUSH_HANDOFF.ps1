param(
    [string]$Repo = "D:\llama.cpp-longcat-pre-gate4",
    [string]$Branch = "handoff/longcat-parity-diagnostics-20260815",
    [string]$CommitMessage = "WIP: LongCat 512-token parity diagnostics handoff"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2
Set-Location $Repo

Write-Host "===== LONGCAT SAFE HANDOFF PUSH ====="
if (-not (Test-Path ".git")) { throw "STOP: not a Git working tree: $Repo" }

$origin = git remote get-url origin
if ($LASTEXITCODE -ne 0) { throw "STOP: origin remote is missing" }
Write-Host "origin: $origin"
Write-Host "starting HEAD: $(git rev-parse HEAD)"
Write-Host "current branch: $(git branch --show-current)"

if ((git branch --show-current) -eq "handoff/longcat-sparse-gate4-wip-20260814") {
    throw "STOP: do not put parity diagnostics on the existing Gate4 recovery branch"
}

git show-ref --verify --quiet ("refs/heads/" + $Branch)
if ($LASTEXITCODE -eq 0) {
    git switch $Branch
} else {
    git switch -c $Branch
}
if ($LASTEXITCODE -ne 0) { throw "STOP: failed to select handoff branch" }

Write-Host "===== PRE-STAGE STATUS ====="
git status --short

# Tracked modifications/deletions: preserve the actual diagnostic tree.
git add -u
if ($LASTEXITCODE -ne 0) { throw "STOP: git add -u failed" }

# Root-level diagnostic/handoff text/code only. No binary captures/models/build outputs.
Get-ChildItem -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match 'longcat|handoff|CLAUDE|NEXT_ACTION|MEMORANDUM' -and
        $_.Extension -in @('.py','.ps1','.md','.json','.txt') -and
        $_.Length -lt 10MB
    } |
    ForEach-Object {
        git add -- $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "STOP: failed to stage $($_.FullName)" }
    }

Write-Host "===== STAGED FILES ====="
git diff --cached --name-status

$staged = @(git diff --cached --name-only)
if ($staged.Count -eq 0) { throw "STOP: nothing staged" }

$forbidden = @()
foreach ($p in $staged) {
    if ($p -match '(?i)(\.bin$|\.log$|\.gguf$|\.safetensors$|\.pt$|\.pth$|\.zip$|(^|/)build/|cmake-build|\.cache/)') {
        $forbidden += $p
    }
}
if ($forbidden.Count -gt 0) {
    $forbidden | ForEach-Object { Write-Host "FORBIDDEN: $_" }
    throw "STOP: binary/model/build output is staged"
}

$tooLarge = @()
foreach ($p in $staged) {
    if (Test-Path $p) {
        $size = (Get-Item $p).Length
        if ($size -gt 50MB) { $tooLarge += ("{0} ({1:N1} MiB)" -f $p,($size/1MB)) }
    }
}
if ($tooLarge.Count -gt 0) {
    $tooLarge | ForEach-Object { Write-Host "OVERSIZED: $_" }
    throw "STOP: staged file exceeds 50 MiB"
}

git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw "STOP: staged diff fails git diff --cached --check" }

git commit -m $CommitMessage
if ($LASTEXITCODE -ne 0) { throw "STOP: git commit failed" }

$newHead = git rev-parse HEAD
Write-Host "new HEAD: $newHead"

git push -u origin $Branch
if ($LASTEXITCODE -ne 0) { throw "STOP: git push failed" }

Write-Host "PUSHED BRANCH: $Branch"
Write-Host "PUSHED COMMIT: $newHead"
Write-Host "LONGCAT SAFE HANDOFF PUSH: PASS"
