# 2050 determinism protocol, Type P (P1/P2/P3): production-style
# --save-logits run at the first sparse-active length. Windows PowerShell
# 5.1, fail-fast: any gate failure throws. MEASUREMENT ONLY -- the final
# logits are hashed and finiteness-gated but compared to NOTHING here (the
# V-logit determinism verdict is the offline comparator's job; there is no
# HF comparison and no Gate-4 criterion in this round).
#
#   .\run_longcat_lsa_2050_typeP.ps1 -Tag P1 -BaselineRunDir <path to cpp_lsa_2050_S1>
#
# --save-logits disables the eval callback (mutually exclusive by design),
# so Type P carries ZERO LONGCAT_* env and must produce zero dump activity.
# Geometry/runtime flags are identical to Type S; the S1-established
# -ub 2304 placement tuple is a hard equality gate (S/P mismatch invalidates
# attribution/determinism). Real-sparse structural evidence (n_kv=2304
# owner/reuse audit sweep + the single real-data mask line) is gated here
# exactly as in Type S -- the audit lines are callback-independent.
param(
    [Parameter(Mandatory = $true)][ValidateSet('P1', 'P2', 'P3')][string]$Tag,
    [Parameter(Mandatory = $true)][string]$BaselineRunDir
)
$ErrorActionPreference = 'Stop'

$repo    = 'D:\llama.cpp-longcat-claude'
$binDir  = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf    = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt  = Join-Path $repo 'prompt_2050_a.txt'
$cuda132 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$outDir  = Join-Path $repo ("cpp_logits_2050_" + $Tag)

$expectedPromptSha      = 'e2791fac7561166c1e4865db64db8726d2ccd499ccfd891efd78d5fd2c42b310'
# Frozen 2050 token stream (2050 x i32 483), recovered full hash verified
# identical across four independent read-only historical run dirs.
$expectedTokenStreamSha = 'eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed'
$expectedCublasVer      = '6.14.11.1330'
$expectedTokensBytes    = 2050 * 4
$expectedLogitsBytes    = 131072 * 4

# LSA measurement-apparatus instrumentation build (checkpoint 09e42fc14).
$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = '39bffc906c03a59af82931cb2505735e3c8ad4e99fc24c121b6113cf77e62bd2'
    'llama.dll'        = '37431a1916e5118af619defe864db63e96d2b5dd290580fa205c36737d4e2d5b'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

Write-Host "== preflight (lsa 2050 Type P $Tag) =="

$baseProv = Join-Path $BaselineRunDir 'run_provenance.json'
if (-not (Test-Path $baseProv)) { throw "baseline provenance missing: $baseProv" }
$baseline = Get-Content $baseProv -Raw | ConvertFrom-Json
foreach ($f in @('placement_line', 'offloaded_line', 'id_dense_start', 'n_kv_lid_real')) {
    if ($null -eq $baseline.$f) { throw "baseline provenance lacks field: $f" }
}
Write-Host "baseline placement loaded from $baseProv"

foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match the recorded instrumentation build"

# Environment-cleanliness sweep - the audited 44-name contract list, aligned
# with run_longcat_resid_walk_512.ps1. Type P authorizes NO diagnostic env
# at all: the child gets only the CUDA v13.2-first PATH pin.
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

if (Test-Path $outDir) { throw "output dir already exists: $outDir (refusing to overwrite; fresh dir per run)" }
New-Item -ItemType Directory $outDir | Out-Null

$args = @(
    '-m', ('"' + $gguf + '"'),
    '-f', ('"' + $prompt + '"'),
    '-c', '4608', '-b', '4608', '-ub', '2304',
    '-fa', 'off', '-ctk', 'f32', '-ctv', 'f32',
    '--no-warmup', '-fitt', '4096', '-v',
    '--save-logits', '--logits-output-dir', ('"' + $outDir + '"')
) -join ' '

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName  = Join-Path $binDir 'llama-debug.exe'
$psi.Arguments = $args
$psi.WorkingDirectory = $repo
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError  = $true
$psi.EnvironmentVariables['PATH'] = $cuda132 + ';' + $env:PATH

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
$outLog = Join-Path $repo ("cpp_logits_2050_" + $Tag + ".out.log")
$errLog = Join-Path $repo ("cpp_logits_2050_" + $Tag + ".err.log")
[IO.File]::WriteAllText($outLog, $stdoutTask.Result)
[IO.File]::WriteAllText($errLog, $stderrTask.Result)
Write-Host ("exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) { throw "llama-debug exit code $($proc.ExitCode)" }

Write-Host "== postflight gates =="
$err = Get-Content $errLog -Raw
$out = Get-Content $outLog -Raw

# Placement: parse the LAST fit line generically and gate equality against
# the S1-established baseline. NEVER assert the -ub 512 literal here.
$placeMatches = [regex]::Matches($err, 'set ngl_per_device\[\d+\]\.\(n_layer, n_part(?:, overflow_type)?\)=\([^\)]*\), id_dense_start=\d+')
if ($placeMatches.Count -lt 1) { throw "placement gate FAIL: no fit/placement line found" }
$placementLine = $placeMatches[$placeMatches.Count - 1].Value
$idDenseStart  = [int][regex]::Match($placementLine, 'id_dense_start=(\d+)').Groups[1].Value
$offloadMatches = [regex]::Matches($err, 'offloaded \d+/\d+ layers')
if ($offloadMatches.Count -lt 1) { throw "offload gate FAIL: no offloaded-layers line" }
$offloadedLine = $offloadMatches[$offloadMatches.Count - 1].Value
Write-Host "placement: $placementLine"
Write-Host "offload:   $offloadedLine"
if ($placementLine -ne $baseline.placement_line) { throw "placement gate FAIL: '$placementLine' != baseline '$($baseline.placement_line)'" }
if ($offloadedLine -ne $baseline.offloaded_line) { throw "offload gate FAIL: '$offloadedLine' != baseline '$($baseline.offloaded_line)'" }
if ($idDenseStart -ne [int]$baseline.id_dense_start) { throw "id_dense_start gate FAIL: $idDenseStart != baseline $($baseline.id_dense_start)" }
Write-Host "placement/offload/id_dense_start match the S1 baseline"

if ($err -notmatch 'graphs reused =\s+0') { throw "graphs-reused gate FAIL" }
Write-Host "graphs reused = 0 OK"

if ($err -notmatch [regex]::Escape('llama_kv_cache_dsa: creating main KV cache, size = 4608 cells')) {
    throw "DSA main-cache gate FAIL"
}
if ($err -notmatch [regex]::Escape('creating indexer KV cache, size = 4608 cells')) {
    throw "DSA indexer-cache gate FAIL"
}
Write-Host "DSA two-cache creation lines OK"

# Diagnostic silence: no injector activity, no dump-family activity, and no
# dump artifacts. (Case-sensitive activity-line matches; tokenizer
# special-token names in the verbose load log stay ignored.)
if ($err -cmatch 'LONGCAT_(HIDDEN_DUMP|RESID_WALK|RESID_INJECT|ATTN_NORM2_INJECT|PROJ_INJECT|NORM_INJECT|ROPE_INJECT|ROPE_ORACLE|FFN_INP2_INJECT)') {
    throw "unexpected LONGCAT_ diagnostic activity in stderr"
}
if ($out -cmatch 'LONGCAT_HIDDEN_VECTOR_DUMP') {
    throw "unexpected dump-writer activity in stdout (Type P must have zero dumps)"
}
$lsaFiles = @(Get-ChildItem $outDir -File -Filter 'lsa_*')
if ($lsaFiles.Count -ne 0) { throw "unexpected lsa_* artifacts in Type P output dir" }
Write-Host "diagnostic silence OK (no injector, no dump activity, no lsa_* files)"

# Real-sparse structural evidence -- identical gates to Type S (the audit
# lines are callback-independent). Reserve/fit n_kv=4608 lines are ignored.
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

# Token stream gate: the saved stream must equal the frozen 2050 stream.
$tokensBin = @(Get-ChildItem $outDir -File -Filter '*-tokens.bin')
if ($tokensBin.Count -ne 1) { throw "tokens bin gate FAIL: $($tokensBin.Count) *-tokens.bin files (expected exactly 1)" }
if ($tokensBin[0].Length -ne $expectedTokensBytes) { throw "tokens bin size FAIL: $($tokensBin[0].Length) B (expected $expectedTokensBytes)" }
$tokSha = Get-Sha256 $tokensBin[0].FullName
if ($tokSha -ne $expectedTokenStreamSha) { throw "token-stream SHA FAIL: $tokSha != $expectedTokenStreamSha" }
Write-Host "token stream: $tokSha OK (2050 tokens)"

# Logits artifact: exactly one, exactly 524,288 B, SHA recorded only.
$logitsBin = @(Get-ChildItem $outDir -File -Filter '*.bin' |
    Where-Object { $_.Name -notlike '*-tokens.bin' })
if ($logitsBin.Count -ne 1) { throw "logits bin gate FAIL: $($logitsBin.Count) candidate bins (expected exactly 1)" }
if ($logitsBin[0].Length -ne $expectedLogitsBytes) { throw "logits size FAIL: $($logitsBin[0].Length) B (expected $expectedLogitsBytes)" }
$logitsSha = Get-Sha256 $logitsBin[0].FullName
Write-Host "logits: $($logitsBin[0].Name) ($logitsSha) -- recorded, compared to nothing in this round"

# Logits-finiteness hard gate (amendment 2): all 131,072 f32 finite. Any
# NaN/Inf is protocol-invalid, never a determinism datum.
$logitsBytes = [IO.File]::ReadAllBytes($logitsBin[0].FullName)
if ($logitsBytes.Length -ne $expectedLogitsBytes) { throw "logits reread size FAIL" }
$nonfinite = 0
for ($i = 0; $i -lt 131072; $i++) {
    $v = [BitConverter]::ToSingle($logitsBytes, $i * 4)
    if ([single]::IsNaN($v) -or [single]::IsInfinity($v)) { $nonfinite++ }
}
if ($nonfinite -ne 0) { throw "logits finiteness FAIL: $nonfinite nonfinite values of 131072 (protocol-invalid run)" }
Write-Host "logits finiteness: 131072/131072 finite"

# Inventory: exactly the four --save-logits artifacts at this point.
$allFiles = @(Get-ChildItem $outDir -File)
if ($allFiles.Count -ne 4) { throw "inventory FAIL: $($allFiles.Count) files in output dir (expected exactly 4 before provenance/manifest)" }
$promptTxt = @($allFiles | Where-Object { $_.Name -like '*-prompt.txt' })
$plainTxt  = @($allFiles | Where-Object { $_.Name -like '*.txt' -and $_.Name -notlike '*-prompt.txt' })
if ($promptTxt.Count -ne 1) { throw "inventory FAIL: $($promptTxt.Count) *-prompt.txt files (expected 1)" }
if ($plainTxt.Count -ne 1) { throw "inventory FAIL: $($plainTxt.Count) logits .txt files (expected 1)" }
Write-Host "inventory: prompt.txt + tokens.bin + logits.bin + logits.txt, nothing else"

# Cryptographic log binding (hygiene 1): the logs are final (child exited,
# both files fully written above and never appended afterwards).
$outLogSha = Get-Sha256 $outLog
$errLogSha = Get-Sha256 $errLog
$outLogBytes = (Get-Item $outLog).Length
$errLogBytes = (Get-Item $errLog).Length
Write-Host "log binding: stdout $outLogSha ($outLogBytes B); stderr $errLogSha ($errLogBytes B)"

# Provenance FIRST, then the manifest that covers it (amendment 4).
$prov = @{
    tag = $Tag
    purpose = '2050 determinism protocol Type P: production-style --save-logits run (no HF comparison)'
    git_head = $gitHead
    arithmetic_head = 'bec291558383fe3184b82a44ea888556a52bfe2d'
    binaries = $expectedBins
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    env_sweep_names = $sweep
    invocation = ("llama-debug.exe " + $args)
    exit_code = $proc.ExitCode
    run_dir = $outDir
    placement_mode = 'BASELINE'
    baseline_run_dir = $BaselineRunDir
    placement_line = $placementLine
    offloaded_line = $offloadedLine
    id_dense_start = $idDenseStart
    n_kv_lid_real = 2304
    audit_real_owner_lines = 14
    audit_real_reuse_lines = 14
    audit_reserve_lines = $reserveCount
    mask_line = $maskLiteral
    tokens_sha256 = $tokSha
    logits_sha256 = $logitsSha
    logits_count = 131072
    logits_nonfinite = 0
    stdout_log = $outLog
    stdout_log_sha256 = $outLogSha
    stdout_log_bytes = $outLogBytes
    stderr_log = $errLog
    stderr_log_sha256 = $errLogSha
    stderr_log_bytes = $errLogBytes
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $outDir 'run_provenance.json')
Write-Host "provenance written"

# Manifest: every data artifact + run_provenance.json, excluding the
# manifest itself -- exactly 5 lines for Type P.
$manifest = Join-Path $outDir 'SHA256SUMS.txt'
$manifestEntries = @(Get-ChildItem $outDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name)
if ($manifestEntries.Count -ne 5) { throw "manifest FAIL: $($manifestEntries.Count) entries (expected exactly 5)" }
$manifestEntries | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
Write-Host "manifest written: $manifest (5 entries)"

Write-Host ("LSA 2050 TYPE P (" + $Tag + "): ALL GATES PASS")
Write-Host ("logits sha256 of record: " + $logitsSha)
