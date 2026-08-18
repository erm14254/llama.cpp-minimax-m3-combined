# Project Gate-3 regression: the established 4-token exact/full-attention
# parity check (HANDOFF_MEMORANDUM_2026-08-15.md lines 163-183), re-run
# against the standing production arithmetic. The memo-era invocation flags
# were not recorded; per the reviewed plan this reconstruction uses the
# standard production-style parameters with the 4-token input and records
# itself. The criterion is the frozen one: all 131,072 logits within
# atol 0.5 / rtol 0.05 AND top-1 = 444 on both sides, 0 violations - Gate 3
# must PASS or this script throws.
#
# Historical references (read-only mtp tree, hashed 2026-08-18):
#   prompt "Hello, world!" -> token ids [20769, 235, 3121, 224]
#   historical tokens.bin  ad9883df7c21de340e1fea799c2c9746afb5e4097fd0df7a596fda68f634fb0f
#   HF oracle              longcat_sparse_gate3_hf_v4_logits.bin = 2c178ea5... (524,288 B)
param(
    [Parameter(Mandatory = $true)][string]$Tag
)
$ErrorActionPreference = 'Stop'

$repo    = 'D:\llama.cpp-longcat-claude'
$binDir  = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf    = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt  = Join-Path $repo 'prompt_gate3_helloworld.txt'
$cuda132 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$outDir  = Join-Path $repo ("cpp_logits_gate3_" + $Tag)
$venvPy  = Join-Path $repo '.venv\Scripts\python.exe'

$hfOracle    = 'D:\llama.cpp-longcat-mtp\longcat_sparse_gate3_hf_v4_logits.bin'
$comparator  = 'D:\llama.cpp-longcat-mtp\compare_longcat_sparse_gate3_logits.py'
$expectedHfOracleSha   = '2c178ea5384d9b8ef59755658ecce2dfba33528edc7bf58964f23db81a26e050'
$expectedComparatorSha = '6976fbc035c60692406a02cc1a6706b2702bbb16f579829e44c29dcdcc57bc93'
$expectedPromptSha     = '315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3'
$expectedTokensSha     = 'ad9883df7c21de340e1fea799c2c9746afb5e4097fd0df7a596fda68f634fb0f'
$expectedCublasVer     = '6.14.11.1330'

# Standing state = stage A (stage B reverted by 11d93b56a). llama.dll is the
# post-revert recompile 99ad8993... (functional identity to the stage-A
# build proven by the stageArevert endpoint reproduction).
$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = '261f08a5d3a4db5f0d699b0b99f4d2dfba4f74d11967d6574b5ce68db2ca9894'
    'llama.dll'        = '15543e91e1dd3048263b29d6f1ee83d66d49150ebfb79844ef711246faec0bb9'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

Write-Host "== preflight (gate3 $Tag) =="
foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match the recorded standing build"

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
if ((Get-Sha256 $prompt) -ne $expectedPromptSha) { throw "prompt SHA FAIL" }
if ((Get-Sha256 $hfOracle) -ne $expectedHfOracleSha) { throw "gate3 HF oracle SHA FAIL" }
if ((Get-Sha256 $comparator) -ne $expectedComparatorSha) { throw "comparator SHA FAIL" }
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
$outLog = Join-Path $repo ("cpp_logits_gate3_" + $Tag + ".out.log")
$errLog = Join-Path $repo ("cpp_logits_gate3_" + $Tag + ".err.log")
[IO.File]::WriteAllText($outLog, $stdoutTask.Result)
[IO.File]::WriteAllText($errLog, $stderrTask.Result)
Write-Host ("exit code: " + $proc.ExitCode)
if ($proc.ExitCode -ne 0) { throw "llama-debug exit code $($proc.ExitCode)" }

Write-Host "== postflight gates =="
$err = Get-Content $errLog -Raw
if ($err -cmatch 'LONGCAT_(HIDDEN_DUMP|RESID_WALK|RESID_INJECT|ATTN_NORM2_INJECT|PROJ_INJECT|NORM_INJECT|ROPE_INJECT|ROPE_ORACLE)') {
    throw "unexpected LONGCAT_ diagnostic activity in stderr"
}
$tokensBin = Get-ChildItem $outDir -Filter '*-tokens.bin' | Select-Object -First 1
if (-not $tokensBin) { throw "tokens bin missing" }
$tokSha = Get-Sha256 $tokensBin.FullName
if ($tokSha -ne $expectedTokensSha) { throw "token-stream SHA FAIL: $tokSha (expected the historical 4-token stream)" }
Write-Host "token stream: 4 tokens [20769,235,3121,224] reproduced ($expectedTokensSha) OK"

$logitsBin = Get-ChildItem $outDir -Filter '*.bin' |
    Where-Object { $_.Name -notlike '*-tokens.bin' } | Select-Object -First 1
if (-not $logitsBin) { throw "logits bin missing" }
if ($logitsBin.Length -ne 524288) { throw "logits size FAIL" }
$logitsSha = Get-Sha256 $logitsBin.FullName
Write-Host "logits: $($logitsBin.Name) ($logitsSha)"

Write-Host "== gate-3 comparator =="
$cmpJson = Join-Path $outDir ("gate3_" + $Tag + ".json")
& $venvPy $comparator --hf-bin $hfOracle --cpp-bin $logitsBin.FullName --out-json $cmpJson
if (-not (Test-Path $cmpJson)) { throw "comparator produced no verdict JSON (exit $LASTEXITCODE)" }
$r = Get-Content $cmpJson -Raw | ConvertFrom-Json
if ($null -eq $r.passed) { throw "comparator JSON lacks 'passed'" }

$manifest = Join-Path $outDir 'SHA256SUMS.txt'
Get-ChildItem $outDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
$prov = @{
    tag = $Tag
    purpose = 'project Gate-3 regression (4-token, established criterion)'
    arithmetic_head = 'bec291558383fe3184b82a44ea888556a52bfe2d'
    binaries = $expectedBins
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    hf_oracle_sha256 = $expectedHfOracleSha
    comparator_sha256 = $expectedComparatorSha
    tokens_sha256 = $tokSha
    logits_sha256 = $logitsSha
    invocation = ("llama-debug.exe " + $args)
    reconstruction_note = 'memo-era flag set unrecorded; standard production-style parameters used and recorded per the reviewed plan'
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $outDir 'run_provenance.json')

Write-Host "== gate-3 verdict =="
Write-Host ("violations = " + $r.violations + "  top1: hf=" + $r.hf_top1 + " cpp=" + $r.cpp_top1)
if (-not $r.passed) { throw "PROJECT GATE-3 REGRESSION FAIL: violations=$($r.violations) top1_agree=$($r.top1_agree)" }
if ($r.hf_top1 -ne 444) { throw "gate-3 oracle top1 unexpected: $($r.hf_top1)" }
Write-Host "PROJECT GATE-3: PASS (0 violations, top-1 444 both sides)"
exit 0
