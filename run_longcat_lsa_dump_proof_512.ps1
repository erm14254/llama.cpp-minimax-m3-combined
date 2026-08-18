# LSA measurement-apparatus round: <=512 owner-K + attribution-anchor
# dumpability proof (Type S), Windows PowerShell 5.1. Fail-fast: any gate
# failure throws. Serialization/surface validation ONLY -- no arithmetic
# claims, no HF value comparison (that is the reviewed >2048 round's job).
#
#   .\run_longcat_lsa_dump_proof_512.ps1 -Tag <tag>
#
# The run enables exactly ONE dump family (LONGCAT_LSA_DUMP_DIR) on the
# child; the two standing dump families and every injector stay disabled.
# The tensor filter names the same five surfaces the LSA spec table serves
# at <=512, so filter, spec, and inventory describe one experiment:
#   attn_norm-0, q_a_norm-0 (attribution anchors)
#   lsa_indexer_k_proj-0, lsa_indexer_k_norm-0, lsa_indexer_k_2d-0 (owner K)
# The sparse-only spec entries (q_proj/q_2d/weights/top_k) must NOT fire at
# <=512 -- their absence is a structural negative control gate.
param(
    [Parameter(Mandatory = $true)][string]$Tag
)
$ErrorActionPreference = 'Stop'

$repo    = 'D:\llama.cpp-longcat-claude'
$binDir  = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf    = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt  = Join-Path $repo 'prompt_512_a.txt'
$cuda132 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$runDir  = Join-Path $repo ("cpp_lsa_dump_proof_" + $Tag + "_512")
$venvPy  = Join-Path $repo '.venv\Scripts\python.exe'
$analyze = Join-Path $repo 'analyze_longcat_lsa_dump_proof_512.py'

$expectedPromptSha = 'd3c44b156c85427176e7038c4b8f902101424097bb3ce51095333e59e52e5aca'
$expectedCublasVer = '6.14.11.1330'

# LSA measurement-apparatus instrumentation build (M-a/M-b/M-c dump views +
# the LONGCAT_LSA_DUMP_DIR family in common/debug.cpp). Only llama.dll and
# llama-common.dll moved from the de-clobber build; exe/ggml-cuda are the
# promotion-set binaries.
$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = '39bffc906c03a59af82931cb2505735e3c8ad4e99fc24c121b6113cf77e62bd2'
    'llama.dll'        = '37431a1916e5118af619defe864db63e96d2b5dd290580fa205c36737d4e2d5b'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}

# Expected proof inventory: 5 bins + 5 sidecars, nothing else.
$expectedDumps = @{
    'lsa_anchor_attn_norm0_full.bin' = 3072 * 512 * 4
    'lsa_anchor_q_a_norm0_full.bin'  = 1536 * 512 * 4
    'lsa_indexer_k_proj_full.bin'    =  128 * 512 * 4
    'lsa_indexer_k_norm_full.bin'    =  128 * 512 * 4
    'lsa_indexer_k_full.bin'         =  128 * 512 * 4
}
# Sparse-only spec entries that must NOT materialize at <=512.
$forbiddenDumps = @(
    'lsa_indexer_q_proj_full.bin','lsa_indexer_q_full.bin',
    'lsa_indexer_weights_full.bin'
) + (0..13 | ForEach-Object { 'lsa_top_k_owner{0:d2}_full.bin' -f (2 * $_) })

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

Write-Host "== preflight (lsa dump proof $Tag) =="

foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match the recorded instrumentation build"

# Environment-cleanliness sweep - the audited 44-name list, aligned with
# run_longcat_resid_walk_512.ps1 (the names are the contract). The parent
# session must be clean; LONGCAT_LSA_DUMP_DIR is then set child-only below.
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
if (-not (Test-Path $analyze)) { throw "analyzer missing: $analyze" }
& $venvPy -m py_compile $analyze
if ($LASTEXITCODE -ne 0) { throw "analyzer py_compile FAIL" }
$analyzeSha = Get-Sha256 $analyze
Write-Host "prompt OK; analyzer py_compile OK ($analyzeSha)"

if (Test-Path $runDir) { throw "run dir already exists: $runDir (refusing to overwrite)" }
New-Item -ItemType Directory $runDir | Out-Null

# Launch with child-only env: PATH pin + the single authorized dump var.
$filter = '(attn_norm-0|q_a_norm-0|lsa_(indexer_k_proj|indexer_k_norm|indexer_k_2d)-0)$'
$args = @(
    '-m', ('"' + $gguf + '"'),
    '-f', ('"' + $prompt + '"'),
    '-c', '4608', '-b', '4608', '-ub', '512',
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
$outLog = Join-Path $repo ("cpp_lsa_dump_proof_" + $Tag + "_512.out.log")
$errLog = Join-Path $repo ("cpp_lsa_dump_proof_" + $Tag + "_512.err.log")
[IO.File]::WriteAllText($outLog, $stdoutTask.Result)
[IO.File]::WriteAllText($errLog, $stderrTask.Result)
Write-Host ("exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) { throw "llama-debug exit code $($proc.ExitCode)" }

Write-Host "== postflight gates =="
$err = Get-Content $errLog -Raw
$out = Get-Content $outLog -Raw

if ($err -notmatch [regex]::Escape('(n_layer, n_part, overflow_type)=(29, 15, ATTN), id_dense_start=0')) {
    throw "placement gate FAIL: (29, 15, ATTN) not found"
}
Write-Host "placement: (29, 15, ATTN), id_dense_start=0 OK"

if ($err -notmatch 'graphs reused =\s+0') { throw "graphs-reused gate FAIL" }
Write-Host "graphs reused = 0 OK"

if ($err -notmatch [regex]::Escape('offloaded 29/30 layers')) { throw "offload gate FAIL" }
Write-Host "offloaded 29/30 OK"

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

# Writer confirmation lines (stdout): one full-sequence dump per surface.
foreach ($t in @('attn_norm-0','q_a_norm-0','lsa_indexer_k_proj-0','lsa_indexer_k_norm-0','lsa_indexer_k_2d-0')) {
    if ($out -cnotmatch ('LONGCAT_HIDDEN_VECTOR_DUMP tensor=' + [regex]::Escape($t) + ' ')) {
        throw "dump confirmation line missing for $t"
    }
}
Write-Host "writer confirmation lines: 5/5 present"

# Inventory: exactly the 5 expected bins (exact sizes) + 5 sidecars.
foreach ($name in ($expectedDumps.Keys | Sort-Object)) {
    $p = Join-Path $runDir $name
    if (-not (Test-Path $p)) { throw "proof dump missing: $name" }
    if ((Get-Item $p).Length -ne $expectedDumps[$name]) { throw "proof dump size FAIL: $name" }
    $j = [IO.Path]::ChangeExtension($p, '.json')
    if (-not (Test-Path $j)) { throw "proof sidecar missing: $name" }
}
foreach ($name in $forbiddenDumps) {
    if (Test-Path (Join-Path $runDir $name)) {
        throw "NEGATIVE CONTROL FAIL: sparse-only dump $name materialized at <=512"
    }
}
$binCount  = @(Get-ChildItem $runDir -File -Filter '*.bin').Count
$jsonCount = @(Get-ChildItem $runDir -File -Filter '*.json').Count
if ($binCount -ne 5)  { throw "inventory FAIL: $binCount bins (expected exactly 5)" }
if ($jsonCount -ne 5) { throw "inventory FAIL: $jsonCount sidecars (expected exactly 5)" }
Write-Host "inventory: 5 bins (sizes OK) + 5 sidecars, negative control clean"

# Offline analyzer: sidecar fields, nope-half identity, BF16-lattice checks.
& $venvPy $analyze --run-dir $runDir
if ($LASTEXITCODE -ne 0) { throw "analyzer FAIL (exit $LASTEXITCODE)" }
Write-Host "analyzer: PASS"

# Manifest + provenance.
$manifest = Join-Path $runDir 'SHA256SUMS.txt'
Get-ChildItem $runDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
Write-Host "manifest written: $manifest"

$prov = @{
    tag = $Tag
    purpose = 'LSA <=512 owner-K + attribution-anchor dumpability proof (Type S)'
    git_head = (& git -C $repo rev-parse HEAD)
    arithmetic_head = 'bec291558383fe3184b82a44ea888556a52bfe2d'
    binaries = $expectedBins
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    analyzer_sha256 = $analyzeSha
    exit_code = $proc.ExitCode
    env_sweep_names = $sweep
    tensor_filter = $filter
    invocation = ("llama-debug.exe " + $args)
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $runDir 'run_provenance.json')

Write-Host ("LSA DUMP PROOF (" + $Tag + "): ALL GATES PASS")
