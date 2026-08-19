# HF 2050 first-owner semantic-capture runner (Run B -> Run A).
# Windows PowerShell 5.1, fail-fast: any gate failure throws.
#
#   .\run_longcat_hf_lsa_2050_capture.ps1
#
# Run B = canonical final-row logits through the byte-frozen Gate-3 core
# (bb82bcb6...) via capture_longcat_sparse_hf_2050_raw_ids.py, with its
# own fail-closed sparse-engagement proof (observation-only shim).
# Run A = capture_longcat_hf_lsa_2050_firstowner.py: first-owner indexer
# surfaces + secondary logits byte-gated against Run B (instrumentation-
# inertness). select()/torch.topk execute exactly once per forward; no
# second top-k exists anywhere.
#
# MEASUREMENT ONLY. No C++ execution, no arithmetic change, no Gate-4
# criterion; Gate 4 remains NOT RUN. The offline comparator
# (analyze_longcat_lsa_hf_cpp_blockers_2050.py) is a SEPARATE later step
# and is NOT invoked by this runner.
param(
    [string]$TokensBin = ''
)
$ErrorActionPreference = 'Stop'

$repo     = 'D:\llama.cpp-longcat-claude'
$py       = Join-Path $repo '.venv\Scripts\python.exe'
$modelDir = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved'
$coreScript = 'D:\llama.cpp-longcat-mtp\capture_longcat_sparse_hf_gate3_logits.py'

$runBScript = Join-Path $repo 'capture_longcat_sparse_hf_2050_raw_ids.py'
$runAScript = Join-Path $repo 'capture_longcat_hf_lsa_2050_firstowner.py'
$cmpScript  = Join-Path $repo 'analyze_longcat_lsa_hf_cpp_blockers_2050.py'

$logitsDir  = Join-Path $repo 'hf_logits_2050_v1'
$captureDir = Join-Path $repo 'hf_lsa_2050_capture'

# Protocol-commit script SHAs (authoring-time frozen; any drift aborts).
# Hashes are of the canonical CRLF checkout form (core.autocrlf=true on
# this machine; the committed blobs are LF-normalized -- same class as
# the historical d267bf29/dfda8836 line-ending twin).
$expectedScripts = @{
    $runBScript = '3d5b93316d82aab7d19237b5c98de1251bcfd7fbf1c3bfaa915475762affdea9'
    $runAScript = '18fcc5e191e39bf23489e4848ad6ec7659c638c341cd8a919b96c36bd9b9e18f'
    $cmpScript  = '0b9206426182ea3810136afda938b5c689a1a28cba780a59fbe7ec5bd4bd4e45'
}
$expectedCoreSha    = 'bb82bcb6c3bc1d21685221a884dac3b39dc7af06f54fea6187f606dddf4213cb'
$expectedRuntimeSha = 'a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428'
$expectedConfigSha  = '116a80c97e4215bf26668d93b4efd6043b2990e26f9157f2697cffeac17027d5'
$expectedTokenSha   = 'eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed'
$nTokens            = 2050
$expectedTokenId    = 483

# Expected Run A inventory: name -> exact bytes.
$expectedRunABins = [ordered]@{
    'hf_attn_norm0.bin'               = 3072 * 2050 * 4
    'hf_q_a_norm0.bin'                = 1536 * 2050 * 4
    'hf_indexer_k_proj.bin'           =  128 * 2050 * 4
    'hf_indexer_k_norm.bin'           =  128 * 2050 * 4
    'hf_indexer_k.bin'                =  128 * 2050 * 4
    'hf_indexer_q_proj.bin'           = 2048 * 2050 * 4
    'hf_indexer_q.bin'                = 2048 * 2050 * 4
    'hf_indexer_weights_prescale.bin' =   16 * 2050 * 4
    'hf_rope_cos.bin'                 =   64 * 2050 * 4
    'hf_rope_sin.bin'                 =   64 * 2050 * 4
    'hf_weight_k_norm.bin'            =  128 * 4
    'hf_weight_wk.bin'                =  128 * 3072 * 4
    'hf_weight_wq_b.bin'              = 2048 * 1536 * 4
    'hf_weight_weights_proj.bin'      =   16 * 3072 * 4
    'hf_logits_2050_runA.bin'         = 131072 * 4
}
foreach ($ownerIl in 0..13) {
    $expectedRunABins[('hf_top_k_owner{0:d2}.bin' -f (2 * $ownerIl))] = 2048 * 2050 * 4
}

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

Write-Host '== preflight (HF 2050 first-owner capture) =='

# Environment fail-closes.
if ($env:TORCH_ALLOW_TF32_CUBLAS_OVERRIDE) { throw 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set' }
if ($env:PYTHONPATH) { throw "PYTHONPATH is set ('$env:PYTHONPATH') - it could shadow the pinned transformers" }
if ($env:PYTHONHOME) { throw 'PYTHONHOME is set' }

# Pinned interpreter + repo state.
if (-not (Test-Path $py)) { throw "venv python missing: $py" }
Push-Location $repo
try {
    $gitHead = (& git rev-parse HEAD).Trim()
    $dirty = & git status --porcelain=v1 | Where-Object { $_ -notmatch '^\?\?' }
    if ($dirty) { throw "tracked tree not clean:`n$($dirty -join "`n")" }
} finally { Pop-Location }
Write-Host "git_head=$gitHead"

# Runner self-hash (computed dynamically at runtime -- a self-referential
# expected value cannot be embedded; the offline comparator reverifies
# this provenance field against the actual runner file on disk).
$runnerSelfSha = Get-Sha256 $PSCommandPath
Write-Host "runner_self_sha256=$runnerSelfSha ($PSCommandPath)"

# Script SHA + py_compile gates.
foreach ($s in $expectedScripts.Keys) {
    if (-not (Test-Path $s)) { throw "script missing: $s" }
    $sha = Get-Sha256 $s
    if ($sha -ne $expectedScripts[$s]) { throw "script SHA mismatch: $s`n  got      $sha`n  expected $($expectedScripts[$s])" }
    & $py -m py_compile $s
    if ($LASTEXITCODE -ne 0) { throw "py_compile failed: $s" }
    Write-Host "script OK: $(Split-Path $s -Leaf) $sha"
}

# Comparator self-test in the pinned venv (cheap; confirms the actual
# numpy/python execution environment of the round BEFORE any GPU work).
& $py $cmpScript --self-test
if ($LASTEXITCODE -ne 0) { throw "comparator --self-test failed in the pinned venv (exit $LASTEXITCODE)" }
Write-Host 'comparator --self-test: PASS (pinned venv)'

# Frozen core + frozen HF runtime gates (re-gated inside the wrappers too).
foreach ($pair in @(
        @($coreScript, $expectedCoreSha),
        @((Join-Path $modelDir 'modeling_longcat_flash_sparse.py'), $expectedRuntimeSha),
        @((Join-Path $modelDir 'configuration_longcat_flash_sparse.py'), $expectedConfigSha))) {
    $p = $pair[0]; $want = $pair[1]
    if (-not (Test-Path $p)) { throw "frozen file missing: $p" }
    $sha = Get-Sha256 $p
    if ($sha -ne $want) { throw "frozen file SHA mismatch: $p`n  got      $sha`n  expected $want" }
    Write-Host "frozen OK: $(Split-Path $p -Leaf) $sha"
}

# Token stream: default = the on-disk P1 artifact; regenerate 2050 x i32
# 483 if absent. Either path is protected by the frozen stream SHA.
if (-not $TokensBin) {
    $TokensBin = Join-Path $repo 'cpp_logits_2050_P1\llamacpp-LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008-tokens.bin'
}
if (-not (Test-Path $TokensBin)) {
    Write-Host "tokens bin missing; regenerating frozen 2050x483 stream"
    $TokensBin = Join-Path $repo 'tokens_2050_a.bin'
    $bytes = New-Object byte[] (4 * $nTokens)
    for ($i = 0; $i -lt $nTokens; $i++) {
        [BitConverter]::GetBytes([int]$expectedTokenId).CopyTo($bytes, 4 * $i)
    }
    [IO.File]::WriteAllBytes($TokensBin, $bytes)
}
$tokSha = Get-Sha256 $TokensBin
if ($tokSha -ne $expectedTokenSha) { throw "token stream SHA mismatch: $tokSha" }
if ((Get-Item $TokensBin).Length -ne 4 * $nTokens) { throw 'token stream length mismatch' }
Write-Host "tokens OK: $TokensBin $tokSha"

# Fresh output dirs (abort-if-exist contract).
foreach ($d in @($logitsDir, $captureDir)) {
    if (Test-Path $d) { throw "output dir already exists (fresh-dir contract): $d" }
}
New-Item -ItemType Directory -Path $logitsDir | Out-Null

$canonicalBin  = Join-Path $logitsDir 'hf_sparse_2050_v1.bin'
$canonicalJson = Join-Path $logitsDir 'hf_sparse_2050_v1.json'
$proofJson     = Join-Path $logitsDir 'sparse_engagement_proof.json'

function Invoke-Capture([string]$tag, [string]$script, [string[]]$scriptArgs) {
    $outLog = Join-Path $repo ("hf_lsa_2050_" + $tag + ".out.log")
    $errLog = Join-Path $repo ("hf_lsa_2050_" + $tag + ".err.log")
    foreach ($l in @($outLog, $errLog)) {
        if (Test-Path $l) { throw "log already exists (fresh-log contract): $l" }
    }
    Write-Host "== $tag start: $(Get-Date -Format o) =="
    $argList = @($script) + $scriptArgs
    $proc = Start-Process -FilePath $py -ArgumentList $argList `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -NoNewWindow -Wait -PassThru
    $code = $proc.ExitCode
    Write-Host "== $tag exit: $code $(Get-Date -Format o) =="
    if ($code -ne 0) {
        Get-Content $outLog -Tail 40 | ForEach-Object { Write-Host "  out| $_" }
        Get-Content $errLog -Tail 40 | ForEach-Object { Write-Host "  err| $_" }
        throw "$tag failed with exit code $code (logs: $outLog / $errLog)"
    }
    return @{
        tag = $tag; exit_code = $code
        stdout_log = $outLog; stdout_log_sha256 = (Get-Sha256 $outLog)
        stdout_log_bytes = (Get-Item $outLog).Length
        stderr_log = $errLog; stderr_log_sha256 = (Get-Sha256 $errLog)
        stderr_log_bytes = (Get-Item $errLog).Length
    }
}

# ---- Run B (canonical logits + own sparse-engagement proof) ----
$runBInfo = Invoke-Capture 'runB' $runBScript @(
    '--model-dir', $modelDir,
    '--tokens-bin', $TokensBin,
    '--out-bin', $canonicalBin,
    '--out-json', $canonicalJson,
    '--proof-json', $proofJson
)

foreach ($p in @($canonicalBin, $canonicalJson, $proofJson)) {
    if (-not (Test-Path $p)) { throw "Run B artifact missing: $p" }
}
if ((Get-Item $canonicalBin).Length -ne 131072 * 4) { throw 'canonical logits size mismatch' }
$proof = Get-Content $proofJson -Raw | ConvertFrom-Json
if ($proof.engagement_proof -ne 'PASS') {
    throw "Run B sparse-engagement proof != PASS - canonical artifact REJECTED (see $proofJson)"
}
if (@($proof.collector).Count -ne 28) { throw "Run B collector has $(@($proof.collector).Count) records != 28" }
$canonicalSha = Get-Sha256 $canonicalBin
Write-Host "Run B canonical logits: $canonicalSha (engagement proof PASS)"

# ---- Run A (surfaces + inertness-gated secondary logits) ----
$runAInfo = Invoke-Capture 'runA' $runAScript @(
    '--model-dir', $modelDir,
    '--tokens-bin', $TokensBin,
    '--canonical-logits-bin', $canonicalBin,
    '--out-dir', $captureDir
)

# Inventory + independent A==B re-verification.
foreach ($name in $expectedRunABins.Keys) {
    $p = Join-Path $captureDir $name
    if (-not (Test-Path $p)) { throw "Run A artifact missing: $name" }
    $len = (Get-Item $p).Length
    if ($len -ne $expectedRunABins[$name]) {
        throw "Run A artifact size mismatch: $name is $len, expected $($expectedRunABins[$name])"
    }
}
foreach ($aux in @('summary.json', 'SHA256SUMS.txt')) {
    if (-not (Test-Path (Join-Path $captureDir $aux))) { throw "Run A aux missing: $aux" }
}

# Manifest verification: exact expected inventory, every listed artifact
# rehashed against its manifest value.
$sumsPath = Join-Path $captureDir 'SHA256SUMS.txt'
$manifest = @{}
Get-Content $sumsPath | Where-Object { $_.Trim() } | ForEach-Object {
    if ($_ -match '^([0-9a-f]{64})\s+(.+)$') {
        if ($manifest.ContainsKey($Matches[2])) { throw "duplicate SHA256SUMS.txt entry: $($Matches[2])" }
        $manifest[$Matches[2]] = $Matches[1]
    }
    else { throw "malformed SHA256SUMS.txt line: $_" }
}
$expectedNames = @($expectedRunABins.Keys) + @('summary.json')
foreach ($name in $expectedNames) {
    if (-not $manifest.ContainsKey($name)) { throw "SHA256SUMS.txt missing entry: $name" }
}
foreach ($name in @($manifest.Keys)) {
    if ($expectedNames -notcontains $name) { throw "SHA256SUMS.txt unexpected entry: $name" }
    $actual = Get-Sha256 (Join-Path $captureDir $name)
    if ($actual -ne $manifest[$name]) {
        throw "manifest rehash mismatch: $name`n  got      $actual`n  manifest $($manifest[$name])"
    }
}
Write-Host "Run A manifest verified: $($manifest.Count) entries rehashed OK"

$runALogitsSha = Get-Sha256 (Join-Path $captureDir 'hf_logits_2050_runA.bin')
if ($runALogitsSha -ne $canonicalSha) {
    throw "Run A logits != Run B canonical (runner re-verification): $runALogitsSha vs $canonicalSha"
}
Write-Host "A==B logits re-verified by runner: $runALogitsSha"

$runAManifestSha = Get-Sha256 $sumsPath
$runASummarySha  = Get-Sha256 (Join-Path $captureDir 'summary.json')
$runBCoreJsonSha = Get-Sha256 $canonicalJson
$runBProofSha    = Get-Sha256 $proofJson

# ---- provenance ----
$prov = [ordered]@{
    purpose             = 'HF 2050 first-owner capture round: Run B (canonical logits + engagement proof) then Run A (surfaces). No C++ execution; no Gate-4 criterion; Gate 4 NOT RUN.'
    git_head            = $gitHead
    model_dir           = $modelDir
    tokens_bin          = $TokensBin
    tokens_bin_sha256   = $tokSha
    core_script_sha256  = $expectedCoreSha
    runB_script_sha256  = $expectedScripts[$runBScript]
    runA_script_sha256  = $expectedScripts[$runAScript]
    cmp_script_sha256   = $expectedScripts[$cmpScript]
    runner_script_sha256 = $runnerSelfSha
    canonical_logits_sha256 = $canonicalSha
    runA_logits_sha256  = $runALogitsSha
    a_equals_b          = $true
    engagement_proof    = 'PASS'
    runA_manifest_sha256 = $runAManifestSha
    runA_summary_sha256  = $runASummarySha
    runB_core_json_sha256 = $runBCoreJsonSha
    runB_engagement_proof_sha256 = $runBProofSha
    interception_seams  = $proof.meta.interception_seams
    runB                = $runBInfo
    runA                = $runAInfo
    python              = $py
    timestamp           = (Get-Date -Format o)
}
$provPath = Join-Path $logitsDir 'run_provenance.json'
$prov | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 $provPath
Write-Host "provenance: $provPath"
Write-Host 'HF 2050 CAPTURE ROUND: ALL RUNNER GATES PASS'
