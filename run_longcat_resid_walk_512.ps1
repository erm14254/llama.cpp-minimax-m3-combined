# Causal-reset experiment run harness (control / inject), Windows PowerShell 5.1.
# Fail-fast: any gate failure throws. Measurement-only; no arithmetic changes.
#
#   .\run_longcat_resid_walk_512.ps1 -Mode control
#   .\run_longcat_resid_walk_512.ps1 -Mode inject
#
# Preflight: binary-set SHA verification (instrumentation HEAD 2f827a91e),
# environment-cleanliness sweep, prompt/oracle SHA gates, fresh run dir.
# Child process: CUDA v13.2-first PATH pin (child env only, parent untouched),
# LONGCAT_HIDDEN_DUMP_DIR + LONGCAT_RESID_WALK_DUMP_DIR (+ LONGCAT_RESID_INJECT_DIR
# in inject mode). Live cuBLAS module verification from the running process.
# Postflight: exit code, placement/graph-reuse/injection log gates, final-row
# regression hashes vs the committed attnpath manifest, landing gate (inject),
# SHA256SUMS.txt over all dumps.
param(
    [Parameter(Mandatory=$true)][ValidateSet('control','inject','inject2','inject3','inject4','injectffn')] [string]$Mode,
    [string]$Suffix = ''
)
$ErrorActionPreference = 'Stop'

$repo      = 'D:\llama.cpp-longcat-claude'
$binDir    = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf      = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt    = Join-Path $repo 'prompt_512_a.txt'
$oracleDir = 'D:\lc_resid_walk_512'
$cuda132   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$tag       = $Mode
if ($Suffix -ne '') { $tag = $Mode + '_' + $Suffix }
$runDir    = Join-Path $repo ("cpp_resid_walk_" + $tag + "_512")
# Prior-generation run dir: every applicable hash it recorded must reproduce
# byte-exactly in this run (except run_provenance.json), proving the extended
# instrumentation inert on all previously captured surfaces. For inject2 the
# prior is the single-reset inject_b1 run and reproduction is restricted to
# the upstream-of-attn_norm-2 subset (everything downstream changes by
# design).
if ($Mode -in @('inject2','inject3','inject4')) {
    $priorDir = Join-Path $repo 'cpp_resid_walk_inject_b1_512'
} elseif ($Mode -eq 'injectffn') {
    # Prior generation for injectffn is the ffn_norm-experiment run; only
    # the invariant subset is gated against it (see $allow below).
    $priorDir = Join-Path $repo 'cpp_resid_walk_injectffn_ffnNorm_512'
} else {
    $priorDir = Join-Path $repo ("cpp_resid_walk_" + $Mode + "_512")
}

# Instrumentation HEAD ac8010739 (ffn_inp-2 injector + ffn_norm-2 dump spec
# in llama-common.dll 261f08a5...). Production arithmetic = standing stage A
# (stage B and both bisect variants reverted): llama.dll 84012cc4... is the
# post-bisect recompile of the 0-diff stage-A source, functional identity
# proven by the byte-exact 9d8583e3... endpoint reproduction.
# exe/ggml-cuda byte-identical to the b98070666 instrumentation set.
$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = '261f08a5d3a4db5f0d699b0b99f4d2dfba4f74d11967d6574b5ce68db2ca9894'
    'llama.dll'        = '84012cc489d8864dadca84e1d3f1e426507e5fa5bea9f5fe5283e5b5ead7c343'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}
$expectedOracle5Sha = '4c9792430fee2716b573ccf365617e537adf8305571e2a5a0b1a881c0c4de340'
$expectedOracle6Sha = 'c91991eb459352ec407aebcee5ee2b12e7b25db0bafd3e0462955a8f8144df6b'
$oracle2Dir = 'D:\lc_block1_stages_512'
$expectedOracle2Sha = 'afa16c6c3324387e9261c708cae044b8fcb08acda8c8f6315d2ba8d39a8f0fd7'
$oracle34Dir = 'D:\lc_block2_mla_512'
$expectedOracle3Sha = '32173b18459358494f943288b974ef7df70eb540ff9e366c720c14f250407a96'
$expectedOracle4Sha = '28ea5b52221a94ddf780f04507f11aee7b6fc8617974f53d558424d41c470f3f'
$expectedPromptSha = 'd3c44b156c85427176e7038c4b8f902101424097bb3ce51095333e59e52e5aca'
$expectedCublasVer = '6.14.11.1330'

# Final-row regression sets, hashes from the committed
# cpp_attn0_mla_attnpath_512/SHA256SUMS.txt (byte-identical to expB).
$upstreamRegression = @{
    'inp_embd_ngram.bin'       = 'd0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f'
    'logical0_attn0_norm.bin'  = 'a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af'
    'q_a_proj.bin'             = 'ddf69fe4d372184806d5941ca87c7a629149b15cdc1fa7f2ae1f81792dcf95ed'
    'q_a_layernorm.bin'        = '956bd3e87b02a89ad1e3dd71801decffd10103d37bade7c490836aedd384dd37'
    'q_b_proj.bin'             = '4f3b647b62c60475fc03f023ce46a5c01951c45847ced2557b5692b2ed3e79b1'
    'kv_a_proj_with_mqa.bin'   = '513390418c9877fa46286d397db7c9c9fb6408852836fb7827106acd183ceecc'
    'kv_a_layernorm.bin'       = 'b44cc101b03b11d96c0d9c52613f7469141dd7786b8128f93e3b7e912c550373'
    'kv_cmpr_scaled.bin'       = '909b7ee75366b0ee1d5a912c103762563236cd07c6fd8385ceb1e549f2a86ce8'
    'q_pe_rope.bin'            = 'a783bf7b6c120f5b415f8120da32c09d369f0da8edab0220d019769b711d7bed'
    'k_pe_rope.bin'            = '529175d6df630e53ea9acbdc5ada96e818788f90c2f5248cd57e3c85c0a05029'
    'kqv_out.bin'              = 'fce2bb9840c8eb35977b54f5a3cd56bb31c3da1c3a17620a56087a03383f40ce'
    'o_proj.bin'               = 'ac91a8310515ffcbf8802d761028705a371865e553bfc2a9d4dad8f7f416bf3f'
    'logical0_attn0_resid.bin' = '7e05940b5c1b6b8f3bcaf210ac8938a44ea3006052c1f16e20c855df6452f109'
    'logical0_mlp0_resid.bin'  = 'de18420a5b2e0d4e2575d1f597cc67c01ccd6e1b17c03dccfd5a20d687b31cb7'
}
# Pre-registered expected-moved surface under the stage-A il>=1 arithmetic:
# ffn_inp-1 (physical block 1, il=1) is the single il>=1-dependent member of
# the historical 15-file upstream set. Recorded (old committed hash kept for
# reference), NOT gated; every il==0/pre-layer surface above stays a hard gate.
$expectedMovedSurfaces = @{
    'logical0_attn1_resid.bin' = 'af49add8343451d0f9379dd874a5cd9f59cbe43f95179a7a4fca69cce3da0e12'
}
$downstreamRegression = @{
    'logical_00.bin'  = 'fa813b529fba809778497da7b43a5c5dc653dcb5906f9078fa08e8c9b35f1e3b'
    'logical_01.bin'  = 'f6e9e0685a7c7b45f1f85521b641f3dee3b6febeb87963a428fb78a07f5411c0'
    'logical_02.bin'  = '55e28ae734163814c28a2c592725a86ea6a357d71cfde2cb98aad5144bd74887'
    'logical_03.bin'  = 'a45e1965854f2120228f0712957db0aa047cf70a3b022afba24d713a75681844'
    'logical_04.bin'  = 'f6ea6919e819f67f9f4cd7ff66d3d49d001b8897cb2d0584d98835a6b8d4195d'
    'logical_05.bin'  = '850956c621cdf86901b087b4f078640bdcdce043b3026608467f6d15922f33d2'
    'logical_06.bin'  = 'f25ed6547a455bdbb62eee0cc265269107c47cc2aa40345ca687bcaad42154f2'
    'logical_07.bin'  = '43c9475a5d5bedfe7675c2d36d3d583b38ff5dee6ff188bb1bf1742437c75f3b'
    'logical_08.bin'  = '02bb900289a8d58572b14aa2c4daf75d7703045bb9111b3aa496d050e42d05b2'
    'logical_09.bin'  = 'e1dff304465998848ae5528699f6ad6d7c362e1ac49fca3709c001744f756482'
    'logical_10.bin'  = '6c9bc832266c587ceb7868cafcd7f02b57201327615034407177234a0afb1f44'
    'logical_11.bin'  = '570ce9efdf243695440b739c1a6b3902395c760c375753ce6455d571521e19eb'
    'logical_12.bin'  = '6cab13fd58a75683fd86b783440d21d7b4adfb55ffa4fe61ca203a9c205b79e0'
    'result_norm.bin' = 'cdca61ccf103d19ec064759970f2e2f84b725a1eb52e713ae73144c505cfead3'
}
# Final-row dump of the injected node must equal row 511 of the oracle, whose
# committed hash is the hf_hidden_512_v4 logical_00 final-row oracle.
$hfLogical00FinalRowSha = '5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff'

function Get-Sha256([string]$path) {
    (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
}

Write-Host "== preflight ($tag) =="

# 1. Binary set (identical recorded set for both runs).
foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match the recorded stage build (see `$expectedBins comment)"

# 2. Environment-cleanliness sweep (parent session must be clean).
# Source-audited fail-closed name list (wrapper-aware derivation 2026-08-17:
# bare getenv + ggml_cuda_ar_env_u64 wrapper reads across common/, src/,
# ggml/src core+CPU+CUDA; backends not compiled on this CUDA/Windows build
# excluded; cuBLAS/cuBLASLt logging vars are the library's own contract;
# TORCH override is python-side, swept defensively). Count is descriptive.
$sweep = @(
    # LONGCAT diagnostics (6)
    'LONGCAT_HIDDEN_DUMP_DIR','LONGCAT_ROPE_INJECT_DIR','LONGCAT_ROPE_ORACLE_DIR',
    'LONGCAT_RESID_WALK_DUMP_DIR','LONGCAT_RESID_INJECT_DIR','LONGCAT_ATTN_NORM2_INJECT_DIR',
    'LONGCAT_PROJ_INJECT_DIR','LONGCAT_NORM_INJECT_DIR','LONGCAT_FFN_INP2_INJECT_DIR',
    # GGML_CUDA bare getenv (12; incl. P2P, missed by earlier subdir-limited greps)
    'GGML_CUDA_ALLREDUCE','GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DEVICES',
    'GGML_CUDA_DISABLE_FUSION','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_ENABLE_UNIFIED_MEMORY',
    'GGML_CUDA_GRAPH_OPT','GGML_CUDA_NO_PINNED','GGML_CUDA_P2P','GGML_CUDA_PDL',
    'GGML_CUDA_REGISTER_HOST','GGML_CUDA_VALIDATE_MUL_MAT_ID',
    # GGML_CUDA wrapper reads via ggml_cuda_ar_env_u64 (3)
    'GGML_CUDA_AR_BF16_THRESHOLD','GGML_CUDA_AR_COPY_CHUNK_BYTES','GGML_CUDA_AR_COPY_THRESHOLD',
    # ggml core / CPU / dispatch (6)
    'GGML_OP_OFFLOAD_MIN_BATCH','GGML_CPU_DISABLE_FUSION','GGML_BACKEND_PATH',
    'GGML_TOTAL_THREADS','GGML_SCHED_DEBUG','GGML_SCHED_DEBUG_REALLOC',
    # llama core toggles/debug (8)
    'LLAMA_ATTN_ROT_DISABLE','LLAMA_GRAPH_REUSE_DISABLE','LLAMA_BATCH_DEBUG',
    'LLAMA_GRAPH_INPUT_DEBUG','LLAMA_GRAPH_RESULT_DEBUG','LLAMA_KV_CACHE_DEBUG',
    'LLAMA_DSV4_COMPRESS_DEBUG','LLAMA_TRACE',
    # cuBLAS / cuBLASLt library logging (4)
    'CUBLAS_LOGINFO_DBG','CUBLAS_LOGDEST_DBG','CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_FILE',
    # python-side (1)
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'
)
foreach ($v in $sweep) {
    if (Test-Path "Env:$v") { throw "env sweep FAIL: $v is set in the session" }
}
Write-Host "env sweep: $($sweep.Count)/$($sweep.Count) clean (source-audited list)"

# 3. Inputs.
if (-not (Test-Path $gguf))   { throw "GGUF missing: $gguf" }
$promptSha = Get-Sha256 $prompt
if ($promptSha -ne $expectedPromptSha) { throw "prompt SHA FAIL: $promptSha" }
Write-Host "prompt: $expectedPromptSha OK"

$oracleSha = $null
$oracle2Sha = $null
$oracleFfnSha = $null
if ($Mode -in @('inject','inject2','inject3','inject4')) {
    $oraclePath = Join-Path $oracleDir 'logical_00_oracle.bin'
    if (-not (Test-Path $oraclePath)) { throw "oracle missing: $oraclePath" }
    if ((Get-Item $oraclePath).Length -ne 6291456) { throw "oracle size FAIL" }
    $oracleSha = Get-Sha256 $oraclePath
    $sums = Get-Content (Join-Path $oracleDir 'SHA256SUMS.txt')
    $line = $sums | Where-Object { $_ -match 'logical_00_oracle\.bin$' }
    if (-not $line) { throw "oracle not in HF SHA256SUMS" }
    $recorded = ($line -split '\s+')[0].ToLower()
    if ($recorded -ne $oracleSha) { throw "oracle SHA FAIL: disk $oracleSha != recorded $recorded" }
    Write-Host "oracle: $oracleSha OK (matches HF capture manifest)"
}
if ($Mode -in @('inject2','inject3','inject4')) {
    $oracle2Path = Join-Path $oracle2Dir 'attn0_norm.bin'
    if (-not (Test-Path $oracle2Path)) { throw "oracle2 missing: $oracle2Path" }
    if ((Get-Item $oracle2Path).Length -ne 6291456) { throw "oracle2 size FAIL" }
    $oracle2Sha = Get-Sha256 $oracle2Path
    if ($oracle2Sha -ne $expectedOracle2Sha) { throw "oracle2 SHA FAIL: $oracle2Sha != $expectedOracle2Sha" }
    $sums2 = Get-Content (Join-Path $oracle2Dir 'SHA256SUMS.txt')
    $line2 = $sums2 | Where-Object { $_ -match 'attn0_norm\.bin$' }
    if (-not $line2) { throw "oracle2 not in HF block1 SHA256SUMS" }
    $recorded2 = ($line2 -split '\s+')[0].ToLower()
    if ($recorded2 -ne $oracle2Sha) { throw "oracle2 manifest mismatch" }
    Write-Host "oracle2: $oracle2Sha OK (attn0_norm, matches HF block1 manifest)"
}
if ($Mode -eq 'injectffn') {
    # ffn_norm causal experiment: the exact-predecessor oracle is the HF
    # layer-1 attn0_resid full-sequence capture (injected at ffn_inp-2).
    $oFfn = Join-Path $oracle2Dir 'attn0_resid.bin'
    if (-not (Test-Path $oFfn)) { throw "ffn oracle missing: $oFfn" }
    if ((Get-Item $oFfn).Length -ne 6291456) { throw "ffn oracle size FAIL" }
    $oracleFfnSha = Get-Sha256 $oFfn
    if ($oracleFfnSha -ne '4718460be4d2bb0243c4b9bcf76e20ca4b8d5a0f35ec3717ca6b8dd5cb5f73c3') {
        throw "ffn oracle SHA FAIL: $oracleFfnSha"
    }
    $sumsFfn = Get-Content (Join-Path $oracle2Dir 'SHA256SUMS.txt')
    $lineFfn = $sumsFfn | Where-Object { $_ -match 'attn0_resid\.bin$' }
    if (-not $lineFfn) { throw "ffn oracle not in HF block1 SHA256SUMS" }
    if ((($lineFfn -split '\s+')[0].ToLower()) -ne $oracleFfnSha) { throw "ffn oracle manifest mismatch" }
    Write-Host "ffn oracle: $oracleFfnSha OK (attn0_resid, matches HF block1 manifest)"
}
$oracle3Sha = $null
$oracle4Sha = $null
if ($Mode -in @('inject3','inject4')) {
    $o3 = Join-Path $oracle34Dir 'q_a_proj.bin'
    $o4 = Join-Path $oracle34Dir 'kv_a_proj_with_mqa.bin'
    if (-not (Test-Path $o3)) { throw "oracle3 missing: $o3" }
    if (-not (Test-Path $o4)) { throw "oracle4 missing: $o4" }
    if ((Get-Item $o3).Length -ne 3145728) { throw "oracle3 size FAIL" }
    if ((Get-Item $o4).Length -ne 1179648) { throw "oracle4 size FAIL" }
    $oracle3Sha = Get-Sha256 $o3
    $oracle4Sha = Get-Sha256 $o4
    if ($oracle3Sha -ne $expectedOracle3Sha) { throw "oracle3 SHA FAIL: $oracle3Sha" }
    if ($oracle4Sha -ne $expectedOracle4Sha) { throw "oracle4 SHA FAIL: $oracle4Sha" }
    $sums34 = Get-Content (Join-Path $oracle34Dir 'SHA256SUMS.txt')
    $l3 = $sums34 | Where-Object { $_ -match '\sq_a_proj\.bin$' }
    $l4 = $sums34 | Where-Object { $_ -match 'kv_a_proj_with_mqa\.bin$' }
    if (-not $l3 -or (($l3 -split '\s+')[0].ToLower() -ne $oracle3Sha)) { throw "oracle3 manifest mismatch" }
    if (-not $l4 -or (($l4 -split '\s+')[0].ToLower() -ne $oracle4Sha)) { throw "oracle4 manifest mismatch" }
    Write-Host "oracle3: $oracle3Sha OK (q_a_proj)"
    Write-Host "oracle4: $oracle4Sha OK (kv_a_proj_with_mqa)"
}
$oracle5Sha = $null
$oracle6Sha = $null
if ($Mode -eq 'inject4') {
    $o5 = Join-Path $oracle34Dir 'q_a_layernorm.bin'
    $o6 = Join-Path $oracle34Dir 'kv_a_layernorm.bin'
    if (-not (Test-Path $o5)) { throw "oracle5 missing: $o5" }
    if (-not (Test-Path $o6)) { throw "oracle6 missing: $o6" }
    if ((Get-Item $o5).Length -ne 3145728) { throw "oracle5 size FAIL" }
    if ((Get-Item $o6).Length -ne 1048576) { throw "oracle6 size FAIL" }
    $oracle5Sha = Get-Sha256 $o5
    $oracle6Sha = Get-Sha256 $o6
    if ($oracle5Sha -ne $expectedOracle5Sha) { throw "oracle5 SHA FAIL: $oracle5Sha" }
    if ($oracle6Sha -ne $expectedOracle6Sha) { throw "oracle6 SHA FAIL: $oracle6Sha" }
    $sums56 = Get-Content (Join-Path $oracle34Dir 'SHA256SUMS.txt')
    $l5 = $sums56 | Where-Object { $_ -match '\sq_a_layernorm\.bin$' }
    $l6 = $sums56 | Where-Object { $_ -match 'kv_a_layernorm\.bin$' }
    if (-not $l5 -or (($l5 -split '\s+')[0].ToLower() -ne $oracle5Sha)) { throw "oracle5 manifest mismatch" }
    if (-not $l6 -or (($l6 -split '\s+')[0].ToLower() -ne $oracle6Sha)) { throw "oracle6 manifest mismatch" }
    Write-Host "oracle5: $oracle5Sha OK (q_a_layernorm)"
    Write-Host "oracle6: $oracle6Sha OK (kv_a_layernorm)"
}

# 4. Fresh run dir.
if (Test-Path $runDir) { throw "run dir already exists: $runDir (refusing to overwrite)" }
New-Item -ItemType Directory $runDir | Out-Null

# 5. Launch with child-only env (parent env untouched).
$filter = '^(q_a_proj|q_a_norm|q_b_proj|q_pe_rope|kv_cmpr_pe|k_pe_rope|kv_a_norm|kv_cmpr_scaled|kqv_out|attn_out|ffn_inp)-0$'
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
$psi.EnvironmentVariables['LONGCAT_HIDDEN_DUMP_DIR']     = $runDir
$psi.EnvironmentVariables['LONGCAT_RESID_WALK_DUMP_DIR'] = $runDir
if ($Mode -in @('inject','inject2','inject3','inject4')) {
    $psi.EnvironmentVariables['LONGCAT_RESID_INJECT_DIR'] = $oracleDir
}
if ($Mode -eq 'injectffn') {
    $psi.EnvironmentVariables['LONGCAT_FFN_INP2_INJECT_DIR'] = $oracle2Dir
}
if ($Mode -in @('inject2','inject3','inject4')) {
    $psi.EnvironmentVariables['LONGCAT_ATTN_NORM2_INJECT_DIR'] = $oracle2Dir
}
if ($Mode -in @('inject3','inject4')) {
    $psi.EnvironmentVariables['LONGCAT_PROJ_INJECT_DIR'] = $oracle34Dir
}
if ($Mode -eq 'inject4') {
    $psi.EnvironmentVariables['LONGCAT_NORM_INJECT_DIR'] = $oracle34Dir
}

Write-Host "== launch =="
Write-Host ("llama-debug.exe " + $args)
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $psi
[void]$proc.Start()
$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()

# 6. Live cuBLAS module verification from the running process.
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
# Locale-independent version (FileVersion string renders with ',' on some
# locales): compose from the numeric parts.
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
$outLog = Join-Path $repo ("cpp_resid_walk_" + $tag + "_512.out.log")
$errLog = Join-Path $repo ("cpp_resid_walk_" + $tag + "_512.err.log")
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

if ($Mode -in @('inject','inject2','inject3','inject4')) {
    if ($err -notmatch [regex]::Escape('LONGCAT_RESID_INJECT: l_out-1 <- logical_00_oracle.bin (6291456 bytes)')) {
        throw "injection log-line gate FAIL"
    }
    Write-Host "injection log line OK"
}
if ($Mode -eq 'injectffn') {
    if ($err -notmatch [regex]::Escape('LONGCAT_FFN_INP2_INJECT: ffn_inp-2 <- attn0_resid.bin (6291456 bytes)')) {
        throw "ffn injection log-line gate FAIL"
    }
    Write-Host "ffn injection log line OK"
}
if ($Mode -in @('inject2','inject3','inject4')) {
    if ($err -notmatch [regex]::Escape('LONGCAT_ATTN_NORM2_INJECT: attn_norm-2 <- attn0_norm.bin (6291456 bytes)')) {
        throw "attn_norm2 injection log-line gate FAIL"
    }
    Write-Host "attn_norm2 injection log line OK"
}
if ($Mode -in @('inject3','inject4')) {
    if ($err -notmatch [regex]::Escape('LONGCAT_PROJ_INJECT: q_a_proj-2 <- q_a_proj.bin (3145728 bytes)')) {
        throw "q_a_proj injection log-line gate FAIL"
    }
    if ($err -notmatch [regex]::Escape('LONGCAT_PROJ_INJECT: kv_cmpr_pe-2 <- kv_a_proj_with_mqa.bin (1179648 bytes)')) {
        throw "kv_cmpr_pe injection log-line gate FAIL"
    }
    Write-Host "projection injection log lines OK"
}
if ($Mode -eq 'inject4') {
    if ($err -notmatch [regex]::Escape('LONGCAT_NORM_INJECT: q_a_norm-2 <- q_a_layernorm.bin (3145728 bytes)')) {
        throw "q_a_norm injection log-line gate FAIL"
    }
    if ($err -notmatch [regex]::Escape('LONGCAT_NORM_INJECT: kv_a_norm-2 <- kv_a_layernorm.bin (1048576 bytes)')) {
        throw "kv_a_norm injection log-line gate FAIL"
    }
    Write-Host "norm injection log lines OK"
}

# Final-row regression hashes.
$regressionSet = $upstreamRegression.Clone()
if ($Mode -eq 'control') {
    foreach ($k in $downstreamRegression.Keys) { $regressionSet[$k] = $downstreamRegression[$k] }
}
$failed = 0
foreach ($name in ($regressionSet.Keys | Sort-Object)) {
    $p = Join-Path $runDir $name
    if (-not (Test-Path $p)) { Write-Host "MISSING $name"; $failed++; continue }
    $h = Get-Sha256 $p
    if ($h -ne $regressionSet[$name]) { Write-Host "MISMATCH $name $h"; $failed++ }
}
if ($failed -gt 0) { throw "final-row regression FAIL: $failed surface(s)" }
Write-Host ("final-row regression: " + $regressionSet.Count + "/" + $regressionSet.Count + " match committed attnpath manifest (il==0/pre-layer invariants)")

# Expected-moved surfaces: hash and record, never gate. A surface that has
# NOT moved off its old-arithmetic hash is reported (informational) - it
# would indicate the il>=1 change did not take effect where expected.
$movedRecord = @{}
foreach ($name in ($expectedMovedSurfaces.Keys | Sort-Object)) {
    $p = Join-Path $runDir $name
    if (-not (Test-Path $p)) { throw "expected-moved surface missing: $name" }
    $h = Get-Sha256 $p
    $movedRecord[$name] = $h
    if ($h -eq $expectedMovedSurfaces[$name]) {
        Write-Host "expected-moved surface UNMOVED (informational): $name still $h"
    } else {
        Write-Host "expected-moved surface recorded: $name $h (old-arithmetic was $($expectedMovedSurfaces[$name]))"
    }
}

if ($Mode -in @('inject','inject2','inject3','inject4')) {
    # Landing gates: full-seq dump of the injected node == oracle, and its
    # final-row dump == committed HF logical_00 final-row oracle.
    $landing = Get-Sha256 (Join-Path $runDir 'logical_00_full.bin')
    if ($landing -ne $oracleSha) { throw "landing gate FAIL: logical_00_full $landing != oracle $oracleSha" }
    Write-Host "landing gate: logical_00_full.bin == oracle ($oracleSha) OK"
    $fr = Get-Sha256 (Join-Path $runDir 'logical_00.bin')
    if ($fr -ne $hfLogical00FinalRowSha) { throw "landing final-row gate FAIL: $fr" }
    Write-Host "landing final-row gate: logical_00.bin == HF final-row oracle OK"
}
if ($Mode -eq 'injectffn') {
    # ffn landing gate: the walk dump of the injected ffn_inp-2 node must
    # equal the HF attn0_resid oracle byte-exactly.
    $landingFfn = Get-Sha256 (Join-Path $runDir 'block1_attn0_resid_full.bin')
    if ($landingFfn -ne $oracleFfnSha) { throw "ffn landing gate FAIL: block1_attn0_resid_full $landingFfn != oracle $oracleFfnSha" }
    Write-Host "ffn landing gate: block1_attn0_resid_full.bin == attn0_resid oracle ($oracleFfnSha) OK"
}
if ($Mode -in @('inject2','inject3','inject4')) {
    # Second landing gate: the walk dump of the injected attn_norm-2 node
    # must equal the HF attn0_norm oracle byte-exactly.
    $landing2 = Get-Sha256 (Join-Path $runDir 'block1_attn0_norm_full.bin')
    if ($landing2 -ne $oracle2Sha) { throw "landing2 gate FAIL: block1_attn0_norm_full $landing2 != oracle2 $oracle2Sha" }
    Write-Host "landing2 gate: block1_attn0_norm_full.bin == attn0_norm oracle ($oracle2Sha) OK"
}
if ($Mode -in @('inject3','inject4')) {
    $landing3 = Get-Sha256 (Join-Path $runDir 'block2_q_a_proj_full.bin')
    if ($landing3 -ne $oracle3Sha) { throw "landing3 gate FAIL: block2_q_a_proj_full $landing3" }
    Write-Host "landing3 gate: block2_q_a_proj_full.bin == HF q_a_proj oracle OK"
    $landing4 = Get-Sha256 (Join-Path $runDir 'block2_kv_a_proj_full.bin')
    if ($landing4 -ne $oracle4Sha) { throw "landing4 gate FAIL: block2_kv_a_proj_full $landing4" }
    Write-Host "landing4 gate: block2_kv_a_proj_full.bin == HF kv_a_proj_with_mqa oracle OK"
}
if ($Mode -eq 'inject4') {
    $landing5 = Get-Sha256 (Join-Path $runDir 'block2_q_a_norm_full.bin')
    if ($landing5 -ne $oracle5Sha) { throw "landing5 gate FAIL: block2_q_a_norm_full $landing5" }
    Write-Host "landing5 gate: block2_q_a_norm_full.bin == HF q_a_layernorm oracle OK"
    $landing6 = Get-Sha256 (Join-Path $runDir 'block2_kv_a_norm_full.bin')
    if ($landing6 -ne $oracle6Sha) { throw "landing6 gate FAIL: block2_kv_a_norm_full $landing6" }
    Write-Host "landing6 gate: block2_kv_a_norm_full.bin == HF kv_a_layernorm oracle OK"
}

# Full-seq dump inventory.
$expectedFull = @('result_norm_full.bin') + (0..13 | ForEach-Object { 'logical_{0:d2}_full.bin' -f $_ })
if ($Suffix -ne '') {
    $expectedFull += @(
        'block1_attn0_norm_full.bin','block1_attn0_out_full.bin',
        'block1_attn0_resid_full.bin','block1_ffn0_norm_full.bin',
        'block1_mlp0_resid_full.bin',
        'block1_attn1_norm_full.bin','block1_attn1_resid_full.bin',
        'block2_q_a_proj_full.bin','block2_q_a_norm_full.bin',
        'block2_q_b_proj_full.bin','block2_q_scaled_full.bin',
        'block2_kv_a_proj_full.bin',
        'block2_kv_a_norm_full.bin','block2_kv_cmpr_scaled_full.bin',
        'block2_q_pe_rope_full.bin','block2_k_pe_rope_full.bin',
        'block2_kqv_out_full.bin'
    )
}
$fullSeqSizes = @{
    'block2_q_a_proj_full.bin'       = 1536 * 512 * 4
    'block2_q_a_norm_full.bin'       = 1536 * 512 * 4
    'block2_q_b_proj_full.bin'       = 6144 * 512 * 4
    'block2_q_scaled_full.bin'       = 6144 * 512 * 4
    'block2_kv_a_proj_full.bin'      =  576 * 512 * 4
    'block2_kv_a_norm_full.bin'      =  512 * 512 * 4
    'block2_kv_cmpr_scaled_full.bin' =  512 * 512 * 4
    'block2_q_pe_rope_full.bin'      = 2048 * 512 * 4
    'block2_k_pe_rope_full.bin'      =   64 * 512 * 4
    'block2_kqv_out_full.bin'        = 4096 * 512 * 4
}
foreach ($name in $expectedFull) {
    $p = Join-Path $runDir $name
    if (-not (Test-Path $p)) { throw "full-seq dump missing: $name" }
    $expectSize = 6291456
    if ($fullSeqSizes.ContainsKey($name)) { $expectSize = $fullSeqSizes[$name] }
    if ((Get-Item $p).Length -ne $expectSize) { throw "full-seq dump size FAIL: $name" }
}
Write-Host "full-seq inventory: $($expectedFull.Count) dumps present, sizes OK"

# Dual-reset manifest reproduction, re-scoped for the il>=1 arithmetic era
# (pre-registered in the reviewed stage-A plan): the historical 81/81
# full-manifest gate is invalid by design once il>=1 arithmetic changes -
# every surface downstream of an il>=1 operator legitimately moves. The gate
# now covers exactly the invariant subset (il==0/pre-layer upstream dumps +
# injected-node landing witnesses); everything else in the committed prior
# manifest is counted and reported as expected-divergent, never gated.
$dualPrior = Join-Path $repo 'cpp_resid_walk_inject2_b1_512'
if ($Mode -eq 'inject2' -and $runDir -ne $dualPrior -and (Test-Path (Join-Path $dualPrior 'SHA256SUMS.txt'))) {
    $dualInvariantAllow = @($upstreamRegression.Keys) + @(
        'logical_00.bin','logical_00_full.bin','logical_00_full.json',
        'block1_attn0_norm_full.bin','block1_attn0_norm_full.json'
    )
    $dpFailed = 0
    $dpCount = 0
    $dpSkipped = 0
    foreach ($line in (Get-Content (Join-Path $dualPrior 'SHA256SUMS.txt'))) {
        if ($line.Trim() -eq '') { continue }
        $parts = $line.Trim() -split '\s+', 2
        $sha = $parts[0].ToLower(); $name = $parts[1].Trim()
        if ($name -eq 'run_provenance.json') { continue }
        if ($dualInvariantAllow -notcontains $name) { $dpSkipped++; continue }
        $dpCount++
        $p = Join-Path $runDir $name
        if (-not (Test-Path $p)) { Write-Host "DUAL-PRIOR MISSING $name"; $dpFailed++; continue }
        if ((Get-Sha256 $p) -ne $sha) { Write-Host "DUAL-PRIOR MISMATCH $name"; $dpFailed++ }
    }
    if ($dpFailed -gt 0) { throw "dual-reset invariant-subset reproduction FAIL: $dpFailed/$dpCount" }
    Write-Host "dual-reset invariant-subset reproduction: $dpCount/$dpCount byte-identical ($dpSkipped prior surfaces expected-divergent under il>=1 arithmetic, recorded in this run's own manifest)"
}

# Prior-generation manifest reproduction: every file the previous run of this
# mode recorded (except run_provenance.json) must hash identically here,
# proving the extended instrumentation inert on all previous surfaces.
if ($Suffix -ne '' -and (Test-Path (Join-Path $priorDir 'SHA256SUMS.txt'))) {
    $allow = $null
    if ($Mode -in @('inject','inject2','inject3','inject4')) {
        # Invariant subset only: the il==0/pre-layer upstream final-row
        # dumps + the injected-node witnesses (the l_out-1 landing makes the
        # logical_00 files oracle-valued in prior and current runs alike).
        # ffn_inp-1 (il=1) is in $expectedMovedSurfaces - recorded above,
        # not gated here. Everything downstream of il>=1 arithmetic differs
        # by design across arithmetic stages and is data, not a gate.
        $allow = @($upstreamRegression.Keys) + @(
            'logical_00.bin','logical_00_full.bin','logical_00_full.json'
        )
    }
    if ($Mode -eq 'injectffn') {
        # Same invariant-subset principle; the injectffn landing witness is
        # the ffn_inp-2 dump (oracle-valued in prior and current runs).
        $allow = @($upstreamRegression.Keys) + @(
            'block1_attn0_resid_full.bin','block1_attn0_resid_full.json'
        )
    }
    $priorFailed = 0
    $priorCount = 0
    foreach ($line in (Get-Content (Join-Path $priorDir 'SHA256SUMS.txt'))) {
        if ($line.Trim() -eq '') { continue }
        $parts = $line.Trim() -split '\s+', 2
        $sha = $parts[0].ToLower(); $name = $parts[1].Trim()
        if ($name -eq 'run_provenance.json') { continue }
        if ($null -ne $allow -and $allow -notcontains $name) { continue }
        $priorCount++
        $p = Join-Path $runDir $name
        if (-not (Test-Path $p)) { Write-Host "PRIOR MISSING $name"; $priorFailed++; continue }
        if ((Get-Sha256 $p) -ne $sha) { Write-Host "PRIOR MISMATCH $name"; $priorFailed++ }
    }
    if ($priorFailed -gt 0) { throw "prior-manifest reproduction FAIL: $priorFailed/$priorCount" }
    Write-Host "prior-manifest reproduction: $priorCount/$priorCount byte-identical (subset for inject2: upstream of attn_norm-2)"
}

# Manifest over everything in the run dir.
$manifest = Join-Path $runDir 'SHA256SUMS.txt'
Get-ChildItem $runDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
Write-Host "manifest written: $manifest"

# Provenance sidecar.
$prov = @{
    mode = $Mode
    suffix = $Suffix
    instrumentation_head = 'ac8010739a5081ca94fad1363b5d276eb06c90ae'
    arithmetic_head = 'f136453d3f001009c5ee039e37b120f352a5e89d'
    oracle_ffn_sha256 = $oracleFfnSha
    binaries = $expectedBins
    moved_surfaces = $movedRecord
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    oracle_sha256 = $oracleSha
    oracle2_sha256 = $oracle2Sha
    oracle3_sha256 = $oracle3Sha
    oracle4_sha256 = $oracle4Sha
    oracle5_sha256 = $oracle5Sha
    oracle6_sha256 = $oracle6Sha
    exit_code = $proc.ExitCode
    env_sweep_names = $sweep
    invocation = ("llama-debug.exe " + $args)
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $runDir 'run_provenance.json')

Write-Host ("RESID WALK RUN (" + $Mode + "): ALL GATES PASS")
