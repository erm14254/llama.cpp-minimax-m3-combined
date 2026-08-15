param([string]$Repo = "D:\llama.cpp-longcat-pre-gate4")

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2

function Sha256([string]$Path) {
    (Get-FileHash $Path -Algorithm SHA256).Hash.ToLower()
}

Set-Location $Repo
Write-Host "===== LONGCAT HANDOFF BOOTSTRAP ====="

if (-not (Test-Path ".git")) { throw "STOP: not a Git working tree: $Repo" }

$modelSource = "src\models\longcat-flash-ngram.cpp"
$debugSource = "common\debug.cpp"
$expectedModel = "aaff66b65e5fc4ca245cfe6b379a60b6bfae268b94cf5b69f0dfd7ca10486cf1"
$expectedDebug = "ee673463453c3c7f39de4d43a778551c7db97f8ee42bd0e936ddffd3994c3fc4"

foreach ($p in @($modelSource,$debugSource)) {
    if (-not (Test-Path $p)) { throw "STOP: required source missing: $p" }
}

$modelSha = Sha256 $modelSource
$debugSha = Sha256 $debugSource

Write-Host "model expected: $expectedModel"
Write-Host "model actual:   $modelSha"
Write-Host "debug expected: $expectedDebug"
Write-Host "debug actual:   $debugSha"

if ($modelSha -ne $expectedModel) { Write-Warning "Model SHA differs. Inspect local diff/history; do not blindly restore." }
if ($debugSha -ne $expectedDebug) { Write-Warning "Debug SHA differs. Inspect local diff/history; do not blindly restore." }

Write-Host ""
Write-Host "===== GIT ====="
git branch --show-current
git remote -v
git status --short

Write-Host ""
Write-Host "===== IMPORTANT PATHS ====="
@(
    "D:\LongCat-Flash-Lite-Sparse",
    "D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved",
    "D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16",
    "D:\llama.cpp-longcat-pre-gate4-build-cuda132\bin\Release\llama-debug.exe",
    "D:\llama.cpp-longcat-pre-gate4\prompt_512_a.txt"
) | ForEach-Object { Write-Host ((Test-Path $_).ToString().PadRight(5) + " " + $_) }

Write-Host ""
Write-Host "===== LONGCAT ROOT SCRIPTS ====="
Get-ChildItem -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match 'longcat' -and $_.Extension -in @('.py','.ps1','.md','.json') } |
    Sort-Object Name |
    ForEach-Object { Write-Host ("{0}  {1}" -f $_.Name,(Sha256 $_.FullName)) }

Write-Host ""
Write-Host "Read CLAUDE.md + HANDOFF_MEMORANDUM_2026-08-15.md + NEXT_ACTION.md."
Write-Host "Do not patch more MLA arithmetic before intermediate HF/C++ stage capture."
Write-Host "LONGCAT HANDOFF BOOTSTRAP: PASS"
