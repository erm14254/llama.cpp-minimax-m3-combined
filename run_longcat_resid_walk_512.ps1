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
    [Parameter(Mandatory=$true)][ValidateSet('control','inject')] [string]$Mode
)
$ErrorActionPreference = 'Stop'

$repo      = 'D:\llama.cpp-longcat-claude'
$binDir    = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release'
$gguf      = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$prompt    = Join-Path $repo 'prompt_512_a.txt'
$oracleDir = 'D:\lc_resid_walk_512'
$cuda132   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64'
$runDir    = Join-Path $repo ("cpp_resid_walk_" + $Mode + "_512")

$expectedBins = @{
    'llama-debug.exe'  = 'df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0'
    'llama-common.dll' = 'aa646eb5221e2f60553507915beb05a6f8daa9a14e390b00b81b5254f7b23dca'
    'llama.dll'        = '93466c40380729857eb43f7d4ccfa4cf7f336d634cec0b44bb359d2411465dc3'
    'ggml-cuda.dll'    = '502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48'
}
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

Write-Host "== preflight ($Mode) =="

# 1. Binary set (identical recorded set for both runs).
foreach ($name in $expectedBins.Keys) {
    $p = Join-Path $binDir $name
    $h = Get-Sha256 $p
    if ($h -ne $expectedBins[$name]) { throw "binary SHA FAIL: $name $h != $($expectedBins[$name])" }
}
Write-Host "binary set: 4/4 match instrumentation build (HEAD 2f827a91e)"

# 2. Environment-cleanliness sweep (parent session must be clean).
$sweep = @(
    'LONGCAT_HIDDEN_DUMP_DIR','LONGCAT_ROPE_INJECT_DIR','LONGCAT_ROPE_ORACLE_DIR',
    'LONGCAT_RESID_WALK_DUMP_DIR','LONGCAT_RESID_INJECT_DIR',
    'GGML_CUDA_ALLREDUCE','GGML_CUDA_AR_COPY_CHUNK_BYTES','GGML_CUDA_AR_COPY_THRESHOLD',
    'GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DEVICES','GGML_CUDA_DISABLE_FUSION',
    'GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_ENABLE_UNIFIED_MEMORY','GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_NO_PINNED','GGML_CUDA_PDL','GGML_CUDA_REGISTER_HOST','GGML_CUDA_VALIDATE_MUL_MAT_ID',
    'CUBLAS_LOGINFO_DBG','CUBLAS_LOGDEST_DBG','CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_FILE',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'
)
foreach ($v in $sweep) {
    if (Test-Path "Env:$v") { throw "env sweep FAIL: $v is set in the session" }
}
Write-Host "env sweep: $($sweep.Count)/$($sweep.Count) clean"

# 3. Inputs.
if (-not (Test-Path $gguf))   { throw "GGUF missing: $gguf" }
$promptSha = Get-Sha256 $prompt
if ($promptSha -ne $expectedPromptSha) { throw "prompt SHA FAIL: $promptSha" }
Write-Host "prompt: $expectedPromptSha OK"

$oracleSha = $null
if ($Mode -eq 'inject') {
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
if ($Mode -eq 'inject') {
    $psi.EnvironmentVariables['LONGCAT_RESID_INJECT_DIR'] = $oracleDir
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
$outLog = Join-Path $repo ("cpp_resid_walk_" + $Mode + "_512.out.log")
$errLog = Join-Path $repo ("cpp_resid_walk_" + $Mode + "_512.err.log")
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

if ($Mode -eq 'inject') {
    if ($err -notmatch [regex]::Escape('LONGCAT_RESID_INJECT: l_out-1 <- logical_00_oracle.bin (6291456 bytes)')) {
        throw "injection log-line gate FAIL"
    }
    Write-Host "injection log line OK"
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
Write-Host ("final-row regression: " + $regressionSet.Count + "/" + $regressionSet.Count + " match committed attnpath manifest")

if ($Mode -eq 'inject') {
    # Landing gates: full-seq dump of the injected node == oracle, and its
    # final-row dump == committed HF logical_00 final-row oracle.
    $landing = Get-Sha256 (Join-Path $runDir 'logical_00_full.bin')
    if ($landing -ne $oracleSha) { throw "landing gate FAIL: logical_00_full $landing != oracle $oracleSha" }
    Write-Host "landing gate: logical_00_full.bin == oracle ($oracleSha) OK"
    $fr = Get-Sha256 (Join-Path $runDir 'logical_00.bin')
    if ($fr -ne $hfLogical00FinalRowSha) { throw "landing final-row gate FAIL: $fr" }
    Write-Host "landing final-row gate: logical_00.bin == HF final-row oracle OK"
}

# Full-seq dump inventory.
$expectedFull = @('result_norm_full.bin') + (0..13 | ForEach-Object { 'logical_{0:d2}_full.bin' -f $_ })
foreach ($name in $expectedFull) {
    $p = Join-Path $runDir $name
    if (-not (Test-Path $p)) { throw "full-seq dump missing: $name" }
    if ((Get-Item $p).Length -ne 6291456) { throw "full-seq dump size FAIL: $name" }
}
Write-Host "full-seq inventory: $($expectedFull.Count) dumps present, sizes OK"

# Manifest over everything in the run dir.
$manifest = Join-Path $runDir 'SHA256SUMS.txt'
Get-ChildItem $runDir -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-Sha256 $_.FullName), $_.Name
} | Out-File -Encoding ascii $manifest
Write-Host "manifest written: $manifest"

# Provenance sidecar.
$prov = @{
    mode = $Mode
    instrumentation_head = '2f827a91e2853ce15fb52dab0cb3321e7b888000'
    binaries = $expectedBins
    cublas_module = $cublasPath
    cublas_version = $cublasVer
    oracle_sha256 = $oracleSha
    exit_code = $proc.ExitCode
    invocation = ("llama-debug.exe " + $args)
}
$prov | ConvertTo-Json | Out-File -Encoding ascii (Join-Path $runDir 'run_provenance.json')

Write-Host ("RESID WALK RUN (" + $Mode + "): ALL GATES PASS")
