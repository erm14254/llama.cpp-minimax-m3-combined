# Production-style frozen-512 endpoint run + frozen comparator, reconstructed
# from the recorded postAB parameters (WIN11_HANDOFF_2026-08-17_FROZEN512.md;
# the --logits-output-dir literal was unrecorded and is reconstructed here,
# recorded in this script and the provenance sidecar). Eval callback stays
# off (--save-logits path; debug.cpp makes them mutually exclusive); zero
# LONGCAT_*/diagnostic env; child-only CUDA v13.2-first PATH pin with
# live-process module verification.
#
# Endpoint decision rule (reviewed plan; frozen criterion NEVER widened):
#   PASS only at 0 violations + top-1 agreement.
#   top-1 change or violations > 40  -> STOP_FOR_REVIEW (exit 2). With all
#     stage local gates passed this is NOT stage disproof by itself - the
#     review decides (error cancellation is known in this project).
#   violations <= 40 with top-1 agreement -> RECORD (exit 0), a measurement.
param(
    [Parameter(Mandatory = $true)][string]$Tag
)
$ErrorActionPreference = 'Stop'

$repo    = 'D:\llama.cpp-longcat-claude'
$binDir  = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf    = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt  = Join-Path $repo 'prompt_512_a.txt'
$cuda132 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$outDir  = Join-Path $repo ("cpp_logits_512_" + $Tag)
$venvPy  = Join-Path $repo '.venv\Scripts\python.exe'

$hfOracle    = 'D:\llama.cpp-longcat-pre-gate4\hf_sparse_512_v4.bin'
$comparator  = 'D:\llama.cpp-longcat-mtp\compare_longcat_sparse_gate3_logits.py'
$expectedHfOracleSha   = '8825d92d7d9cdea42a4ea3aa2e3df5766bdf880323b1f48ea8c17ff63f3c5ecf'
$expectedComparatorSha = '6976fbc035c60692406a02cc1a6706b2702bbb16f579829e44c29dcdcc57bc93'
$expectedPromptSha     = 'd3c44b156c85427176e7038c4b8f902101424097bb3ce51095333e59e52e5aca'
$expectedTokensSha     = '4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c'
$expectedCublasVer     = '6.14.11.1330'
$standingViolations    = 40   # the pre-change standing FAIL count; not an acceptance threshold

# Standing state = stage A (stage B reverted by 11d93b56a after the 96-
# violation endpoint review): source byte-identical to the 458a03685 +
# 923fad90d pair. The recompiled llama.dll is 99ad8993... (MSVC timestamp
# embedding; original stage-A build was c890671e...); functional identity is
# proven by the endpoint reproducing the stage-A logits 9d8583e3... byte-exact.
# exe/ggml-cuda byte-identical to the b98070666 instrumentation set.
$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = '9367c541149a0969c2f495e5b4f13cbe883967fc4f5df06663c78e73e2ea4888'
    'llama.dll'        = '84012cc489d8864dadca84e1d3f1e426507e5fa5bea9f5fe5283e5b5ead7c343'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

Write-Host "== preflight (frozen512 $Tag) =="

foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match the recorded stage build"

# Environment-cleanliness sweep - the audited 42-name list, verbatim from
# run_longcat_resid_walk_512.ps1 (the names are the contract).
$sweep = @(
    'LONGCAT_HIDDEN_DUMP_DIR','LONGCAT_ROPE_INJECT_DIR','LONGCAT_ROPE_ORACLE_DIR',
    'LONGCAT_RESID_WALK_DUMP_DIR','LONGCAT_RESID_INJECT_DIR','LONGCAT_ATTN_NORM2_INJECT_DIR',
    'LONGCAT_PROJ_INJECT_DIR','LONGCAT_NORM_INJECT_DIR',
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
$hfSha = Get-Sha256 $hfOracle
if ($hfSha -ne $expectedHfOracleSha) { throw "HF logits oracle SHA FAIL: $hfSha" }
$cmpSha = Get-Sha256 $comparator
if ($cmpSha -ne $expectedComparatorSha) { throw "comparator SHA FAIL: $cmpSha" }
Write-Host "prompt/oracle/comparator SHAs OK"

if (Test-Path $outDir) { throw "output dir already exists: $outDir (refusing to overwrite)" }
New-Item -ItemType Directory $outDir | Out-Null

$args = @(
    '-m', ('"' + $gguf + '"'),
    '-f', ('"' + $prompt + '"'),
    '-c', '4608', '-b', '4608', '-ub', '512',
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
$outLog = Join-Path $repo ("cpp_logits_512_" + $Tag + ".out.log")
$errLog = Join-Path $repo ("cpp_logits_512_" + $Tag + ".err.log")
[IO.File]::WriteAllText($outLog, $stdoutTask.Result)
[IO.File]::WriteAllText($errLog, $stderrTask.Result)
Write-Host ("exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) { throw "llama-debug exit code $($proc.ExitCode)" }

Write-Host "== postflight gates =="
$err = Get-Content $errLog -Raw
if ($err -notmatch [regex]::Escape('(n_layer, n_part, overflow_type)=(29, 15, ATTN), id_dense_start=0')) {
    throw "placement gate FAIL: (29, 15, ATTN) not found"
}
Write-Host "placement: (29, 15, ATTN), id_dense_start=0 OK"
if ($err -notmatch 'graphs reused =\s+0') { throw "graphs-reused gate FAIL" }
Write-Host "graphs reused = 0 OK"
if ($err -notmatch [regex]::Escape('offloaded 29/30 layers')) { throw "offload gate FAIL" }
Write-Host "offloaded 29/30 OK"
# Case-sensitive, prefix-specific: the tokenizer's special-token names
# (<longcat_pad>, <longcat_system>, ...) legitimately appear in the verbose
# load log and must not trip this. Only actual diagnostic activity lines count.
if ($err -cmatch 'LONGCAT_(HIDDEN_DUMP|RESID_WALK|RESID_INJECT|ATTN_NORM2_INJECT|PROJ_INJECT|NORM_INJECT|ROPE_INJECT|ROPE_ORACLE)') {
    throw "unexpected LONGCAT_ diagnostic activity in stderr"
}

$tokensBin = Get-ChildItem $outDir -Filter '*-tokens.bin' | Select-Object -First 1
if (-not $tokensBin) { throw "tokens bin missing in $outDir" }
$tokSha = Get-Sha256 $tokensBin.FullName
if ($tokSha -ne $expectedTokensSha) { throw "token-stream SHA FAIL: $tokSha" }
Write-Host "token stream: $expectedTokensSha OK"

$logitsBin = Get-ChildItem $outDir -Filter '*.bin' |
    Where-Object { $_.Name -notlike '*-tokens.bin' } | Select-Object -First 1
if (-not $logitsBin) { throw "logits bin missing in $outDir" }
if ($logitsBin.Length -ne 524288) { throw "logits size FAIL: $($logitsBin.Length)" }
$logitsSha = Get-Sha256 $logitsBin.FullName
Write-Host "logits: $($logitsBin.Name) ($logitsSha)"

Write-Host "== frozen comparator =="
$cmpJson = Join-Path $outDir ("frozen512_" + $Tag + ".json")
& $venvPy $comparator --hf-bin $hfOracle --cpp-bin $logitsBin.FullName --out-json $cmpJson
# The comparator exits 1 on a FAIL verdict (a legitimate measurement) and
# also on hard STOP errors; the verdict JSON existing and parsing with a
# 'passed' field distinguishes them. Only a missing/unparseable JSON is an
# infrastructure failure here - the decision rule below owns the verdict.
if (-not (Test-Path $cmpJson)) { throw "comparator produced no verdict JSON (exit $LASTEXITCODE)" }
$r = Get-Content $cmpJson -Raw | ConvertFrom-Json
if ($null -eq $r.passed) { throw "comparator JSON lacks 'passed' (exit $LASTEXITCODE)" }

# Manifest + provenance before the verdict (results are recorded regardless).
$manifest = Join-Path $outDir 'SHA256SUMS.txt'
Get-ChildItem $outDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
$prov = @{
    tag = $Tag
    instrumentation_head = '923fad90d4d34388a14e2a6c83cf1b7dff9b4ba8'
    arithmetic_head = 'f136453d3f001009c5ee039e37b120f352a5e89d'
    binaries = $expectedBins
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    hf_oracle_sha256 = $expectedHfOracleSha
    comparator_sha256 = $expectedComparatorSha
    tokens_sha256 = $tokSha
    logits_sha256 = $logitsSha
    exit_code = $proc.ExitCode
    env_sweep_names = $sweep
    invocation = ("llama-debug.exe " + $args)
    standing_violations_reference = $standingViolations
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $outDir 'run_provenance.json')

Write-Host "== endpoint decision rule =="
Write-Host ("violations = " + $r.violations + " / 131072 (standing pre-change: " + $standingViolations + ")")
Write-Host ("top1: hf=" + $r.hf_top1 + " cpp=" + $r.cpp_top1 + " agree=" + $r.top1_agree)
Write-Host ("worst_ratio=" + $r.worst_tolerance_ratio + " max_abs=" + $r.max_abs_error + " rmse=" + $r.rmse + " cosine=" + $r.cosine_similarity)
if (-not $r.top1_agree) {
    Write-Host "VERDICT: STOP_FOR_REVIEW (top-1 disagreement - mandatory review; frozen criterion unchanged)"
    exit 2
}
if ($r.violations -gt $standingViolations) {
    Write-Host "VERDICT: STOP_FOR_REVIEW (violations exceed the standing $standingViolations - mandatory review; NOT by itself stage disproof if all local gates passed)"
    exit 2
}
if ($r.violations -eq 0) {
    Write-Host "VERDICT: PASS under the frozen criterion (0 violations, top-1 agrees)"
} else {
    Write-Host ("VERDICT: RECORD (FAIL under the frozen criterion at " + $r.violations + " violations, top-1 agrees; interim measurement, not a new acceptance threshold)")
}
exit 0
