# Win11 Blackwell — Authoritative C++ Block-0 MLA Stage Capture

**Status: executed 2026-08-17. All three frozen gates passed byte-exact.**
Results in `STATUS_2026-08-17.md`; artifacts in `cpp_attn0_mla_blackwell_512/`.

**Purpose:** produce the six C++ block-0 MLA stage surfaces on the RTX PRO 6000
Blackwell machine, where the frozen `2c804a35…` residual baseline was made, so
the capture clears the Phase 3a regression gate and the HF↔C++ comparison
becomes authoritative.

---

## Corrections to the pre-execution draft

The original section 3 of this runbook **did not reproduce the baseline**. It was
reconstructed by inference before anyone had run it. Corrected below, with the
evidence for each change. Kept visible because the wrong version is quotable
from git history.

| Draft said | Reality | Evidence |
|---|---|---|
| `-ngl 99` | **not passed** | [common/fit.cpp:378](common/fit.cpp:378) throws `"n_gpu_layers already set by user … abort"` — the fit cannot complete if `-ngl` is given, yet all five frozen logs end `common_fit_params: successfully fit params to free device memory` |
| `-ot ffn_*=CUDA_Host` | **not passed** | no tensor-override line appears anywhere in the frozen logs; the MoE split comes from auto-fit's `n_part` mechanism |
| "offloaded 30/30 layers" | **29/30** | the 30/30 line is the fit's *initial probe* pass; every final `load_tensors:` in the frozen logs reads `offloaded 29/30`, `CUDA0 model buffer size = 88936.14 MiB` |
| (absent) | **`-fitt 4096` required** | default is 1024 MiB ([common/common.h:473](common/common.h:473)); the frozen log records `cannot meet free memory target of 4096 MiB`. Without it the budget rises from 91250 to 94322 MiB, the `LAYER_FRACTION_UP` candidate is accepted, and placement lands on `(29,15,UP)` instead of `(29,15,ATTN)` |
| (absent) | **`-b 4608` required** | frozen log records `n_batch = 4608`; the default is 2048. Verified numerically inert, but it aligns the log for comparison |
| (absent) | **`-v` required** | `LOG_TRC` needs verbosity ≥ 4 and `LOG_DBG` ≥ 5; the default is `LOG_LEVEL_INFO = 3` ([common/log.cpp:29](common/log.cpp:29)). Without it the fit trace is **not emitted at all**, so placement cannot be verified |
| (absent) | **cuBLAS must be pinned to CUDA v13.2** | see section 5 — otherwise the gates fail silently |

`-ctk f32 -ctv f32` is confirmed by arithmetic, not just by the draft:
4608 × 576 × **4 B** = 10.125 MiB/layer, matching the frozen log's
`CPU KV 10.12 MiB` (1 layer) + `CUDA0 KV 283.50 MiB` (28 layers) = 29 layers.

---

## 1. Source state

Active checkout `D:\llama.cpp-longcat-claude`, branch `claude/longcat-win11`.
The callback-only instrumentation is already committed:

| File | SHA256 | Bytes |
|---|---|---|
| `src\models\longcat-flash-ngram.cpp` | `7f726913922e58bbffbc008eba0067e328543683094904107af4f7501b2e9a47` | 46446 |
| `common\debug.cpp` | `9cbaddc5ed7eb3413ddbd5cf276da83a4c4a50d6aea3c06ddf821fe52497a4a7` | 20184 |

Both changes are callback-only — no arithmetic is altered. Every non-`cb()`
line in the `longcat-flash-ngram.cpp` diff against `c24ad1fcd` is a comment.

`src\models\longcat-flash-ngram.cpp` — six `cb()` sites:

- three colliding block-0 `cb(q, "q", il)` → `q_a_proj` / `q_a_norm` / `q_b_proj`
- post-norm `cb(kv_cmpr, "kv_cmpr", il)` → `kv_a_norm`
- **new** `cb(cur, "attn_out", il)` after `build_attn` returns, before the residual add

That last one matters: `build_attn` emits `kqv_out` *before* applying `wo`, so
`kqv_out-0` is 4096-wide pre-projection output, not HF `o_proj` (3072-wide).

`common\debug.cpp` — dump helper only: width generalized off the hardcoded 3072,
full-sequence token-major mode with JSON sidecars for the six MLA surfaces
(existing surfaces keep final-row 12288-byte layout), stride-aware reads via
`nb[]`, and dump targets requested at ask time so `--tensor-filter` need not
cover them.

Also stage `prompt_512_a.txt` (untracked, 1024 B,
`d3c44b156c85427176e7038c4b8f902101424097bb3ce51095333e59e52e5aca`) — it lives
only in the read-only `D:\llama.cpp-longcat-pre-gate4` tree.

## 2. Configure and build

Build directory `D:\llama.cpp-longcat-claude-build-cuda132`. Use the repo-local
venv cmake (**4.4.2**, matching the historical pip cmake); the system cmake is
4.3.2. Pin the CUDA toolset explicitly — `nvcc` on `PATH` resolves to **13.0**.

```powershell
$CMAKE = 'D:\llama.cpp-longcat-claude\.venv\Scripts\cmake.exe'
$CUDA  = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
& $CMAKE -S 'D:/llama.cpp-longcat-claude' -B 'D:/llama.cpp-longcat-claude-build-cuda132' `
  -G 'Visual Studio 17 2022' -A x64 -T "cuda=$CUDA" `
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real `
  -DCUDAToolkit_ROOT="$CUDA" -DLLAMA_BUILD_TESTS=OFF
& $CMAKE --build 'D:/llama.cpp-longcat-claude-build-cuda132' --config Release --target llama-debug -- /m:8 /v:minimal
```

Those four `-D` values are the only non-default settings in the historical
`CMakeCache.txt`. Verified: **182/182** reproducibility-relevant cache entries
match the historical build; the 17 differences are generated or path-dependent.

## 3. Run

```powershell
$ErrorActionPreference = 'Stop'
$REPO   = 'D:\llama.cpp-longcat-claude'
$EXE    = 'D:\llama.cpp-longcat-claude-build-cuda132\bin\Release\llama-debug.exe'
$GGUF   = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$OUT    = Join-Path $REPO 'cpp_attn0_mla_blackwell_512'
$LOGOUT = Join-Path $REPO 'cpp_attn0_mla_blackwell_512.out.log'   # stdout
$LOGERR = Join-Path $REPO 'cpp_attn0_mla_blackwell_512.err.log'   # stderr
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$argList = @(
  '-m', $GGUF, '-f', (Join-Path $REPO 'prompt_512_a.txt'),
  '-c','4608', '-b','4608', '-ub','512', '-n','0',
  '-fa','off', '-ctk','f32', '-ctv','f32', '--no-warmup', '-v', '-fitt','4096',
  '--tensor-filter','(q_a_proj-0|q_a_norm-0|q_b_proj-0|kv_cmpr_pe-0|kv_a_norm-0|attn_out-0|ffn_inp-0)$'
)

$origPath = $env:PATH
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;' + $origPath   # section 5
$env:LONGCAT_HIDDEN_DUMP_DIR = $OUT
try {
    $p = Start-Process -FilePath $EXE -ArgumentList $argList -NoNewWindow -PassThru `
                       -RedirectStandardOutput $LOGOUT -RedirectStandardError $LOGERR
    Write-Host "PID = $($p.Id)"
    $p.WaitForExit()
} finally {
    $env:LONGCAT_HIDDEN_DUMP_DIR = ''      # not Remove-Item Env: — see notes
    $env:PATH = $origPath
}
```

Runtime is roughly 45 s.

Notes:

- **No `-ngl`, no `-ot`.** Placement comes from auto-fit. Verify it converged to
  `(29, 15, ATTN)` / `id_dense_start=0` before trusting anything downstream; the
  trace prints ~2.5 s in, long before the model loads, so it makes a cheap
  preflight.
- **`--no-warmup` is mandatory.** Warmup is on by default and would run a dummy
  graph that overwrites the dumps.
- **No `--save-logits`** — the callback is not installed when it is active
  ([examples/debug/debug.cpp:231](examples/debug/debug.cpp:231)). This also means
  `-tokens.bin` cannot be produced in the same run, so the tokenization gate
  reads the log.
- **Use `Start-Process` with real redirection.** PowerShell `2>&1` on a native
  command wraps stderr in `NativeCommandError` records and loses lines — visible
  in the frozen logs, whose first lines are a mangled error block.
- **`$env:X = ''`, not `Remove-Item Env:X`.** The dump path points into the repo
  and `Remove-Item` on it can trip path-protection tooling. An empty value is
  equivalent — `common/debug.cpp` tests `dump_dir[0] == '\0'`.
- There is deliberately **no early-abort env var.** One was attempted and
  removed: returning false from a ggml eval callback only breaks the current
  split's node loop, not the graph, and it skips the rest of that split so
  anything downstream is computed from incomplete state.

## 4. Gate and verify

Four checks. **`logical0_attn0_resid.bin` = `2c804a35…` is the arbiter.**

```powershell
$anchors = @{
  'inp_embd_ngram.bin'       = 'd0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f'
  'logical0_attn0_norm.bin'  = 'a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af'
  'logical0_attn0_resid.bin' = '2c804a35a0397e380d77f08d2a7ffd11fc6e4672722c31a98f7f96620c2a4a4e'
}
foreach ($k in $anchors.Keys) {
  $h = (Get-FileHash -Algorithm SHA256 (Join-Path $OUT $k)).Hash.ToLower()
  if ($h -eq $anchors[$k]) { Write-Host "PASS  $k" -ForegroundColor Green }
  else { Write-Host "FAIL  $k`n  exp $($anchors[$k])`n  got $h" -ForegroundColor Red }
}
```

**The two log-based gates read different files.**
[common/log.cpp:89-93](common/log.cpp:89) routes `GGML_LOG_LEVEL_NONE` to
**stdout** and everything else to **stderr**, and the token-piece output is
`LOG(...)` at level `NONE` ([common/log.h:111](common/log.h:111)):

| Gate | File | Expected |
|---|---|---|
| Placement | `$LOGERR` | `(29, 15, ATTN)`, `id_dense_start=0`, `offloaded 29/30`, `CUDA0 model buffer size = 88936.14 MiB`, `CPU KV 10.12` + `CUDA0 KV 283.50 MiB`, compute `1654.00` / `24.03 MiB` |
| Tokenization | `$LOGOUT` | `Token ids (512)`, all pieces `(483)` |

Required sizes for the six MLA surfaces, each with a matching `.json` sidecar:
`q_a_proj` 3,145,728 · `q_a_layernorm` 3,145,728 · `q_b_proj` 12,582,912 ·
`kv_a_proj_with_mqa` 1,179,648 · `kv_a_layernorm` 1,048,576 · `o_proj` 6,291,456.

## 5. cuBLAS must be pinned to CUDA v13.2

`ggml-cuda.dll` imports **`cublas64_13.dll` by bare name**, and this machine's
`PATH` lists `CUDA\v13.0\bin\x64` **before** `v13.2`. Unpinned, the process loads
cuBLAS **6.14.11.1300** (v13.0) instead of the **6.14.11.1330** (v13.2) the build
targets, and `logical0_attn0_resid` comes out
`49d729e16b09b7d113e1d774851364b4c31ee1ff98608505cbe52122824fc928`.

This was diagnosed by running the **historical** `llama-debug.exe` — the exact
binary that produced `2c804a35…` on 2026-08-15 — which reproduced the same
`49d729e1…`, clearing the rebuild of any fault. Prepending the v13.2 directory
restored `2c804a35…` byte-exact.

The signature is diagnostic: custom elementwise kernels (`inp_embd_ngram`,
`attn_norm-0`) stay byte-exact while only the GEMM-dependent residual moves —
what a cuBLAS kernel-selection change looks like. **Without the pin the frozen
gates fail silently and any comparison is invalid.**

## 6. Compare

```powershell
$PY = 'D:\llama.cpp-longcat-claude\.venv\Scripts\python.exe'
& $PY compare_longcat_attn0_mla_stages.py `
    --hf-dir  D:/lc_mla_blackwell `
    --cpp-dir D:/llama.cpp-longcat-claude/cpp_attn0_mla_blackwell_512 `
    --json-out cpp_attn0_mla_blackwell_512/comparison_authoritative.json
& $PY analyze_longcat_attn0_mla_bf16_boundary.py `
    --hf-dir  D:/lc_mla_blackwell `
    --cpp-dir D:/llama.cpp-longcat-claude/cpp_attn0_mla_blackwell_512 `
    --json-out cpp_attn0_mla_blackwell_512/bf16_boundary_authoritative.json
```

`--hf-dir` is `D:\lc_mla_blackwell` — `_external_artifacts\hf_mla_blackwell_20260816\`
holds only sidecars, since `*.bin` is gitignored.

**Do not pass `--noise-floor`.** 0.00172 was the *Ampere-vs-Blackwell* floor and
does not apply now that both sides are Blackwell. Byte-exactness remains the hard
requirement for the frozen gates, but it is **not** asserted as the standard for
the six MLA intermediates: HF/PyTorch and llama.cpp/CUDA can differ legitimately
on the same GPU through kernel selection and order of operations. Classify those
with the full metrics, the first-divergent-token index, and BF16-rounding
analysis.

Then generate `SHA256SUMS.txt` for the capture directory:

```powershell
Push-Location $OUT
Get-ChildItem -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | Sort-Object Name |
  ForEach-Object { "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.Name } |
  Set-Content -Encoding ascii 'SHA256SUMS.txt'
Pop-Location
```

---

## Outcome

Executed 2026-08-17. All three gates passed. The Q path is byte-exact to the HF
oracles; `kv_a_proj_with_mqa` is a pure BF16 output boundary; the first genuine
divergence is **`kv_a_layernorm`**, explained exactly by HF rounding the
normalized activation to BF16 before the weight multiply. Full analysis in
`STATUS_2026-08-17.md`.
