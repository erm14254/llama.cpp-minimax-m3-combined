# 2050 determinism protocol, Type S (S1/S2/S3): eval-callback LSA surface
# capture at the first sparse-active length. Windows PowerShell 5.1,
# fail-fast: any gate failure throws. MEASUREMENT ONLY -- no HF comparison,
# no arithmetic claims, no Gate-4 criterion. Authored per the reviewed 2050
# protocol plan (amendments 1-4 + implementation-hygiene additions).
#
#   .\run_longcat_lsa_2050_typeS.ps1 -Tag S1 -EstablishPlacement
#   .\run_longcat_lsa_2050_typeS.ps1 -Tag S2 -BaselineRunDir <path to cpp_lsa_2050_S1>
#   .\run_longcat_lsa_2050_typeS.ps1 -Tag S3 -BaselineRunDir <path to cpp_lsa_2050_S1>
#
# Geometry (frozen, never retuned): -c 4608 -b 4608 -ub 2304 -fa off
# -ctk f32 -ctv f32 --no-warmup -fitt 4096 -v. The standing (29, 15, ATTN)
# placement tuple belongs to -ub 512 ONLY: S1 establishes the -ub 2304
# fit/placement tuple; S2/S3 (and both Type-P runners via -BaselineRunDir)
# must reproduce it exactly or the family is invalid.
#
# The child gets exactly ONE dump family (LONGCAT_LSA_DUMP_DIR); every other
# diagnostic env stays unset (44-name parent sweep). --save-logits is NOT
# passed (it would disable the eval callback). Real-sparse structural
# evidence is gated on n_kv=2304 owner/reuse audit lines plus the single
# real-data mask line; n_kv=4608 lines are reserve/fit builds and are never
# proof of sparse execution.
param(
    [Parameter(Mandatory = $true)][ValidateSet('S1', 'S2', 'S3')][string]$Tag,
    [switch]$EstablishPlacement,
    [string]$BaselineRunDir
)
$ErrorActionPreference = 'Stop'

$repo    = 'D:\llama.cpp-longcat-claude'
$binDir  = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf    = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt  = Join-Path $repo 'prompt_2050_a.txt'
$cuda132 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$runDir  = Join-Path $repo ("cpp_lsa_2050_" + $Tag)

$expectedPromptSha      = 'e2791fac7561166c1e4865db64db8726d2ccd499ccfd891efd78d5fd2c42b310'
# Frozen 2050 token stream (2050 x i32 483), recovered full hash verified
# identical across four independent read-only historical run dirs.
$expectedTokenStreamSha = 'eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed'
$expectedCublasVer      = '6.14.11.1330'
$nTokens                = 2050
$expectedTokenId        = 483

# LSA measurement-apparatus instrumentation build (checkpoint 09e42fc14).
$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = '39bffc906c03a59af82931cb2505735e3c8ad4e99fc24c121b6113cf77e62bd2'
    'llama.dll'        = '37431a1916e5118af619defe864db63e96d2b5dd290580fa205c36737d4e2d5b'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}

# Expected capture inventory at n_tokens=2050: 22 bins (exact bytes) + 22
# sidecars, nothing else. Sizes = ne0 * 2050 * 4 (full-sequence f32 dumps).
$expectedDumps = [ordered]@{
    'lsa_anchor_attn_norm0_full.bin' = 3072 * 2050 * 4
    'lsa_anchor_q_a_norm0_full.bin'  = 1536 * 2050 * 4
    'lsa_indexer_k_proj_full.bin'    =  128 * 2050 * 4
    'lsa_indexer_k_norm_full.bin'    =  128 * 2050 * 4
    'lsa_indexer_k_full.bin'         =  128 * 2050 * 4
    'lsa_indexer_q_proj_full.bin'    = 2048 * 2050 * 4
    'lsa_indexer_q_full.bin'         = 2048 * 2050 * 4
    'lsa_indexer_weights_full.bin'   =   16 * 2050 * 4
}
foreach ($ownerIl in 0..13) {
    $expectedDumps[('lsa_top_k_owner{0:d2}_full.bin' -f (2 * $ownerIl))] = 2048 * 2050 * 4
}

# The 22 tensor names whose writer confirmation lines must appear in stdout.
$expectedDumpTensors = @(
    'attn_norm-0', 'q_a_norm-0',
    'lsa_indexer_k_proj-0', 'lsa_indexer_k_norm-0', 'lsa_indexer_k_2d-0',
    'lsa_indexer_q_proj-0', 'lsa_indexer_q_2d-0', 'lsa_indexer_weights-0'
) + (1..27 | Where-Object { $_ % 2 -eq 1 } | ForEach-Object { "lsa_top_k_reuse-$_" })

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}
function Get-Sha256Bytes([byte[]]$bytes) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { (($sha.ComputeHash($bytes)) | ForEach-Object { $_.ToString('x2') }) -join '' }
    finally { $sha.Dispose() }
}

Write-Host "== preflight (lsa 2050 Type S $Tag) =="

# Placement-mode parameter contract: S1 establishes, S2/S3 reproduce.
if ($EstablishPlacement -and $BaselineRunDir) { throw "pass either -EstablishPlacement or -BaselineRunDir, not both" }
if (-not $EstablishPlacement -and -not $BaselineRunDir) { throw "pass -EstablishPlacement (S1) or -BaselineRunDir (S2/S3)" }
if ($Tag -eq 'S1' -and -not $EstablishPlacement) { throw "S1 must run with -EstablishPlacement" }
if ($Tag -ne 'S1' -and $EstablishPlacement) { throw "$Tag must run with -BaselineRunDir (only S1 establishes placement)" }
$baseline = $null
if ($BaselineRunDir) {
    $baseProv = Join-Path $BaselineRunDir 'run_provenance.json'
    if (-not (Test-Path $baseProv)) { throw "baseline provenance missing: $baseProv" }
    $baseline = Get-Content $baseProv -Raw | ConvertFrom-Json
    foreach ($f in @('placement_line', 'offloaded_line', 'id_dense_start', 'n_kv_lid_real')) {
        if ($null -eq $baseline.$f) { throw "baseline provenance lacks field: $f" }
    }
    Write-Host "baseline placement loaded from $baseProv"
}

foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match the recorded instrumentation build"

# Environment-cleanliness sweep - the audited 44-name contract list, aligned
# with run_longcat_resid_walk_512.ps1. The parent session must be clean;
# LONGCAT_LSA_DUMP_DIR is then set child-only below.
$sweep = @(
    'LONGCAT_HIDDEN_DUMP_DIR','LONGCAT_ROPE_INJECT_DIR','LONGCAT_ROPE_ORACLE_DIR',
    'LONGCAT_RESID_WALK_DUMP_DIR','LONGCAT_RESID_INJECT_DIR','LONGCAT_ATTN_NORM2_INJECT_DIR',
    'LONGCAT_PROJ_INJECT_DIR','LONGCAT_NORM_INJECT_DIR','LONGCAT_FFN_INP2_INJECT_DIR',
    'LONGCAT_LSA_DUMP_DIR',
    'GGML_CUDA_ALLREDUCE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DEVICES',
    'GGML_CUDA_DISABLE_FUSION','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_ENABLE_UNIFIED_MEMORY',
    'GGML_CUDA_GRAPH_OPT','GGML_CUDA_NO_PINNED','GGML_CUDA_P2P','GGML_CUDA_PDL',
    'GGML_CUDA_REGISTER_HOST','GGML_CUDA_VALIDATE_MUL_MAT_ID',
    'GGML_CUDA_AR_BF16_THRESHOLD','GGML_CUDA_AR_COPY_CHUNK_BYTES','GGML_CUDA_AR_COPY_THRESHOLD',
    'GGML_OP_OFFLOAD_MIN_BATCH','GGML_CPU_DISABLE_FUSION','GGML_BACKEND_PATH',
    'GGML_TOTAL_THREADS','GGML_SCHED_DEBUG','GGML_SCHED_DEBUG_REALLOC',
    'LLAMA_ATTN_ROT_DISABLE','LLAMA_GRAPH_REUSE_DISABLE','LLAMA_BATCH_DEBUG',
    'LLAMA_GRAPH_INPUT_DEBUG','LLAMA_GRAPH_RESULT_DEBUG','LLAMA_KV_CACHE_DEBUG',
    'LLAMA_DSV4_COMPRESS_DEBUG','LLAMA_TRACE',
    'CUBLAS_LOGINFO_DBG','CUBLAS_LOGDEST_DBG','CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_FILE',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'
)
foreach ($v in $sweep) {
    if (Test-Path "Env:$v") { throw "env sweep FAIL: $v is set in the session" }
}
Write-Host "env sweep: $($sweep.Count)/$($sweep.Count) clean"

if (-not (Test-Path $gguf)) { throw "GGUF missing: $gguf" }
$promptSha = Get-Sha256 $prompt
if ($promptSha -ne $expectedPromptSha) { throw "prompt SHA FAIL: $promptSha" }
Write-Host "prompt OK ($expectedPromptSha)"

$gitHead = (& git -C $repo rev-parse HEAD)
$dirty = @(& git -C $repo status --porcelain | Where-Object { $_ -notmatch '^\?\?' })
if ($dirty.Count -ne 0) { throw "git tracked tree not clean: $($dirty -join '; ')" }
Write-Host "git head $gitHead, tracked tree clean"

if (Test-Path $runDir) { throw "run dir already exists: $runDir (refusing to overwrite; fresh dir per run)" }
New-Item -ItemType Directory $runDir | Out-Null

# Tensor filter aligned 1:1 with the 22 captured surfaces (dumps fire from
# the exact-name spec regardless; the filter makes intent legible and prints
# the matching stat lines). lsa_indexer_kq / lsa_indexer_score / direct mask
# dumps are deliberately EXCLUDED (deferred contingencies; the first two
# carry +/-inf mask terms that would trip the dump writer's non-finite
# abort).
$filter = '(attn_norm-0|q_a_norm-0|lsa_indexer_k_proj-0|lsa_indexer_k_norm-0|lsa_indexer_k_2d-0|lsa_indexer_q_proj-0|lsa_indexer_q_2d-0|lsa_indexer_weights-0|lsa_top_k_reuse-(11|13|15|17|19|21|23|25|27|1|3|5|7|9))$'
$args = @(
    '-m', ('"' + $gguf + '"'),
    '-f', ('"' + $prompt + '"'),
    '-c', '4608', '-b', '4608', '-ub', '2304',
    '-fa', 'off', '-ctk', 'f32', '-ctv', 'f32',
    '--no-warmup', '-fitt', '4096', '-v',
    '--tensor-filter', ('"' + $filter + '"')
) -join ' '

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = Join-Path $binDir 'llama-debug.exe'
$psi.Arguments = $args
$psi.WorkingDirectory = $repo
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.EnvironmentVariables['PATH'] = $cuda132 + ';' + $env:PATH
$psi.EnvironmentVariables['LONGCAT_LSA_DUMP_DIR'] = $runDir

Write-Host "== launch =="
Write-Host ("llama-debug.exe " + $args)
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
[void]$proc.Start()
$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()

# Live cuBLAS module verification from the running process.
$cublasPath = $null
$deadline = (Get-Date).AddSeconds(180)
while (-not $proc.HasExited -and -not $cublasPath -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 1500
    try {
        $m = (Get-Process -Id $proc.Id -ErrorAction Stop).Modules |
             Where-Object { $_.ModuleName -ieq 'cublas64_13.dll' }
        if ($m) { $cublasPath = $m[0].FileName }
    } catch { }
}
if (-not $cublasPath) {
    if (-not $proc.HasExited) { $proc.Kill() }
    throw "cuBLAS module never observed in live process"
}
$vi = (Get-Item $cublasPath).VersionInfo
$cublasVer = '{0}.{1}.{2}.{3}' -f $vi.FileMajorPart, $vi.FileMinorPart, $vi.FileBuildPart, $vi.FilePrivatePart
Write-Host "live cuBLAS: $cublasPath ($cublasVer)"
if ($cublasPath -notlike ($cuda132 + '*')) {
    if (-not $proc.HasExited) { $proc.Kill() }
    throw "cuBLAS PATH-pin FAIL: loaded from $cublasPath"
}
if ($cublasVer -ne $expectedCublasVer) {
    if (-not $proc.HasExited) { $proc.Kill() }
    throw "cuBLAS version FAIL: $cublasVer != $expectedCublasVer"
}

$proc.WaitForExit()
$outLog = Join-Path $repo ("cpp_lsa_2050_" + $Tag + ".out.log")
$errLog = Join-Path $repo ("cpp_lsa_2050_" + $Tag + ".err.log")
[IO.File]::WriteAllText($outLog, $stdoutTask.Result)
[IO.File]::WriteAllText($errLog, $stderrTask.Result)
Write-Host ("exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) { throw "llama-debug exit code $($proc.ExitCode)" }

Write-Host "== postflight gates =="
$err = Get-Content $errLog -Raw
$out = Get-Content $outLog -Raw

# Placement: parse the LAST fit line generically (form is branch-dependent:
# with or without overflow_type; ATTN/GATE/UP). NEVER assert the -ub 512
# literal (29, 15, ATTN) here.
$placeMatches = [regex]::Matches($err, 'set ngl_per_device\[\d+\]\.\(n_layer, n_part(?:, overflow_type)?\)=\([^\)]*\), id_dense_start=\d+')
if ($placeMatches.Count -lt 1) { throw "placement gate FAIL: no fit/placement line found" }
$placementLine = $placeMatches[$placeMatches.Count - 1].Value
$idDenseStart  = [int][regex]::Match($placementLine, 'id_dense_start=(\d+)').Groups[1].Value
$offloadMatches = [regex]::Matches($err, 'offloaded \d+/\d+ layers')
if ($offloadMatches.Count -lt 1) { throw "offload gate FAIL: no offloaded-layers line" }
$offloadedLine = $offloadMatches[$offloadMatches.Count - 1].Value
Write-Host "placement: $placementLine"
Write-Host "offload:   $offloadedLine"
if ($EstablishPlacement) {
    Write-Host "PLACEMENT ESTABLISHED (S1): gate all later runs on the tuple above via -BaselineRunDir"
} else {
    if ($placementLine -ne $baseline.placement_line) { throw "placement gate FAIL: '$placementLine' != baseline '$($baseline.placement_line)'" }
    if ($offloadedLine -ne $baseline.offloaded_line) { throw "offload gate FAIL: '$offloadedLine' != baseline '$($baseline.offloaded_line)'" }
    if ($idDenseStart -ne [int]$baseline.id_dense_start) { throw "id_dense_start gate FAIL: $idDenseStart != baseline $($baseline.id_dense_start)" }
    Write-Host "placement/offload/id_dense_start match the S1 baseline"
}

if ($err -notmatch 'graphs reused =\s+0') { throw "graphs-reused gate FAIL" }
Write-Host "graphs reused = 0 OK"

if ($err -notmatch [regex]::Escape('llama_kv_cache_dsa: creating main KV cache, size = 4608 cells')) {
    throw "DSA main-cache gate FAIL"
}
if ($err -notmatch [regex]::Escape('creating indexer KV cache, size = 4608 cells')) {
    throw "DSA indexer-cache gate FAIL"
}
Write-Host "DSA two-cache creation lines OK"

# No injector may have fired (colon suffix = actual activity lines; the
# case-sensitive match ignores lowercase tokenizer special-token names).
if ($err -cmatch 'LONGCAT_(RESID_INJECT|ATTN_NORM2_INJECT|PROJ_INJECT|NORM_INJECT|ROPE_INJECT|FFN_INP2_INJECT):') {
    throw "unexpected injector activity in stderr"
}
Write-Host "injector absence OK"

# Real-sparse structural evidence. Reserve/fit graph builds print n_kv=4608
# owner/reuse lines in every -c 4608 run and are NOT evidence; the real
# single-ubatch 2050 decode must contribute exactly one complete alternating
# owner/reuse sweep at n_kv=2304 with per-pair tensor-pointer pairing.
$auditMatches = [regex]::Matches($err, 'LONGCAT_LSA_AUDIT (owner|reuse) block=(\d+)(?: owner_block=(\d+))? n_kv=(\d+) top_k=(\d+) tensor=(\S+)')
$real = @()
$reserveCount = 0
foreach ($m in $auditMatches) {
    $nkv = [int]$m.Groups[4].Value
    if ($nkv -eq 2304) { $real += , $m }
    elseif ($nkv -eq 4608) { $reserveCount++ }
    else { throw "audit gate FAIL: unexpected n_kv=$nkv audit line" }
}
if ($real.Count -ne 28) { throw "real-decode audit gate FAIL: $($real.Count) n_kv=2304 lines (expected exactly 28: 14 owner + 14 reuse)" }
for ($p = 0; $p -lt 14; $p++) {
    $o = $real[2 * $p]
    $r = $real[2 * $p + 1]
    if ($o.Groups[1].Value -ne 'owner') { throw "audit pairing FAIL: entry $(2*$p) is not an owner line" }
    if ($r.Groups[1].Value -ne 'reuse') { throw "audit pairing FAIL: entry $(2*$p+1) is not a reuse line" }
    $ob = [int]$o.Groups[2].Value
    $rb = [int]$r.Groups[2].Value
    if ($ob -ne 2 * $p) { throw "audit pairing FAIL: owner block $ob != $(2*$p)" }
    if ($rb -ne 2 * $p + 1) { throw "audit pairing FAIL: reuse block $rb != $(2*$p+1)" }
    if ([int]$r.Groups[3].Value -ne $ob) { throw "audit pairing FAIL: reuse owner_block $($r.Groups[3].Value) != $ob" }
    if ([int]$o.Groups[5].Value -ne 2048) { throw "audit gate FAIL: owner block $ob top_k != 2048" }
    if ([int]$r.Groups[5].Value -ne 2048) { throw "audit gate FAIL: reuse block $rb top_k != 2048" }
    if ($o.Groups[6].Value -ne $r.Groups[6].Value) { throw "audit pointer pairing FAIL at pair il=$ob/$rb" }
}
Write-Host "real-decode audit: 14 owner + 14 reuse at n_kv=2304, pointer pairing 14/14 ($reserveCount reserve-class n_kv=4608 lines ignored)"

$maskLiteral = 'LONGCAT_LSA_AUDIT mask seq=0 query_pos=2049 visible=2050 forced=1040 init_pos=[0,15] local_pos=[1026,2049]'
$maskCount = [regex]::Matches($err, [regex]::Escape($maskLiteral)).Count
if ($maskCount -ne 1) { throw "mask evidence FAIL: $maskCount occurrences of the real-data mask line (expected exactly 1)" }
Write-Host "real-data mask line OK (query_pos=2049 visible=2050 forced=1040)"

# Tokenization gate (amendment 1): callback mode has no *-tokens.bin, so the
# established stdout token listing is parsed, the i32-LE stream is
# reconstructed in memory from the PARSED ids, and its SHA256 must equal the
# frozen 2050 stream hash. This proves S/P input identity directly.
$outLines = $out -split "`r?`n"
$hdrIdx = -1
$hdrCount = -1
for ($i = 0; $i -lt $outLines.Count; $i++) {
    $m = [regex]::Match($outLines[$i], '^Token ids \((\d+)\):\s*$')
    if ($m.Success) {
        if ($hdrIdx -ge 0) { throw "tokenization gate FAIL: multiple 'Token ids' headers" }
        $hdrIdx = $i
        $hdrCount = [int]$m.Groups[1].Value
    }
}
if ($hdrIdx -lt 0) { throw "tokenization gate FAIL: 'Token ids (N):' header not found in stdout" }
if ($hdrCount -ne $nTokens) { throw "tokenization gate FAIL: header token count $hdrCount != $nTokens" }
$ids = New-Object 'System.Collections.Generic.List[int]'
$j = $hdrIdx + 1
while ($j -lt $outLines.Count -and $outLines[$j] -match '^\s*(\S+\(\d+\)\s*)+$') {
    foreach ($m in [regex]::Matches($outLines[$j], '\((\d+)\)')) {
        $ids.Add([int]$m.Groups[1].Value)
    }
    $j++
}
if ($ids.Count -ne $nTokens) { throw "tokenization gate FAIL: parsed $($ids.Count) ids != $nTokens" }
$nonTarget = 0
foreach ($id in $ids) { if ($id -ne $expectedTokenId) { $nonTarget++ } }
if ($nonTarget -ne 0) { throw "tokenization gate FAIL: $nonTarget ids != $expectedTokenId" }
$tokBytes = New-Object byte[] ($nTokens * 4)
for ($i = 0; $i -lt $nTokens; $i++) {
    [Array]::Copy([BitConverter]::GetBytes([int32]$ids[$i]), 0, $tokBytes, $i * 4, 4)
}
$tokStreamSha = Get-Sha256Bytes $tokBytes
if ($tokStreamSha -ne $expectedTokenStreamSha) { throw "tokenization gate FAIL: reconstructed stream SHA $tokStreamSha != $expectedTokenStreamSha" }
Write-Host "tokenization: $nTokens ids, all $expectedTokenId, reconstructed stream SHA $tokStreamSha OK"

# Writer confirmation lines (stdout): one full-sequence dump per surface.
foreach ($t in $expectedDumpTensors) {
    if ($out -cnotmatch ('LONGCAT_HIDDEN_VECTOR_DUMP tensor=' + [regex]::Escape($t) + ' ')) {
        throw "dump confirmation line missing for $t"
    }
}
Write-Host "writer confirmation lines: $($expectedDumpTensors.Count)/$($expectedDumpTensors.Count) present"

# Inventory: exactly the 22 expected bins (exact sizes) + 22 sidecars,
# nothing else in the run dir at this point (provenance/manifest come later).
foreach ($name in $expectedDumps.Keys) {
    $p = Join-Path $runDir $name
    if (-not (Test-Path $p)) { throw "capture dump missing: $name" }
    if ((Get-Item $p).Length -ne $expectedDumps[$name]) { throw "capture dump size FAIL: $name is $((Get-Item $p).Length) B (expected $($expectedDumps[$name]))" }
    $sc = [IO.Path]::ChangeExtension($p, '.json')
    if (-not (Test-Path $sc)) { throw "capture sidecar missing: $name" }
}
$expectedNames = @($expectedDumps.Keys) + @($expectedDumps.Keys | ForEach-Object { [IO.Path]::ChangeExtension($_, '.json') })
$expectedNames = $expectedNames | Sort-Object
$actualNames = @(Get-ChildItem $runDir -File | ForEach-Object { $_.Name }) | Sort-Object
if ($actualNames.Count -ne 44) { throw "inventory FAIL: $($actualNames.Count) files in run dir (expected exactly 44 before provenance/manifest)" }
if (($actualNames -join "`n") -ne ($expectedNames -join "`n")) { throw "inventory FAIL: run dir file set differs from the expected 22 bins + 22 sidecars" }
Write-Host "inventory: 22 bins (sizes exact) + 22 sidecars, nothing else"

# Cryptographic log binding (hygiene 1): the logs are final (child exited,
# both files fully written above and never appended afterwards).
$outLogSha = Get-Sha256 $outLog
$errLogSha = Get-Sha256 $errLog
$outLogBytes = (Get-Item $outLog).Length
$errLogBytes = (Get-Item $errLog).Length
Write-Host "log binding: stdout $outLogSha ($outLogBytes B); stderr $errLogSha ($errLogBytes B)"

# Provenance FIRST, then the manifest that covers it (amendment 4).
$placementMode = 'BASELINE'
if ($EstablishPlacement) { $placementMode = 'ESTABLISH' }
$baselineDirRecord = ''
if ($BaselineRunDir) { $baselineDirRecord = $BaselineRunDir }
$prov = @{
    tag = $Tag
    purpose = '2050 determinism protocol Type S: eval-callback LSA surface capture (no HF comparison)'
    git_head = $gitHead
    arithmetic_head = 'bec291558383fe3184b82a44ea888556a52bfe2d'
    binaries = $expectedBins
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    env_sweep_names = $sweep
    tensor_filter = $filter
    invocation = ("llama-debug.exe " + $args)
    exit_code = $proc.ExitCode
    run_dir = $runDir
    placement_mode = $placementMode
    baseline_run_dir = $baselineDirRecord
    placement_line = $placementLine
    offloaded_line = $offloadedLine
    id_dense_start = $idDenseStart
    n_kv_lid_real = 2304
    audit_real_owner_lines = 14
    audit_real_reuse_lines = 14
    audit_reserve_lines = $reserveCount
    mask_line = $maskLiteral
    token_count = $nTokens
    token_ids_all_483 = $true
    token_stream_sha256_reconstructed = $tokStreamSha
    stdout_log = $outLog
    stdout_log_sha256 = $outLogSha
    stdout_log_bytes = $outLogBytes
    stderr_log = $errLog
    stderr_log_sha256 = $errLogSha
    stderr_log_bytes = $errLogBytes
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $runDir 'run_provenance.json')
Write-Host "provenance written"

# Manifest: every data artifact + run_provenance.json, excluding the
# manifest itself -- exactly 45 lines for Type S.
$manifest = Join-Path $runDir 'SHA256SUMS.txt'
$manifestEntries = @(Get-ChildItem $runDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name)
if ($manifestEntries.Count -ne 45) { throw "manifest FAIL: $($manifestEntries.Count) entries (expected exactly 45)" }
$manifestEntries | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
Write-Host "manifest written: $manifest (45 entries)"

Write-Host ("LSA 2050 TYPE S (" + $Tag + "): ALL GATES PASS")
Write-Host ("placement tuple of record: " + $placementLine + " | " + $offloadedLine)
