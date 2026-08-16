# Win11 Blackwell — Authoritative C++ Block-0 MLA Stage Capture

**Purpose:** produce the six C++ block-0 MLA stage surfaces on the RTX PRO 6000
Blackwell machine, where the frozen `2c804a35…` residual baseline was made, so
the capture clears the Phase 3a regression gate and the HF↔C++ comparison
becomes authoritative.

**Why Win11:** on the Win10 RTX 3090 the run reproduces `inp_embd_ngram`
(`d0e9edc8…`) and `logical0_attn0_norm` (`a1c4c20c…`) **byte-exact**, but
`logical0_attn0_resid` comes out `88defe30…` instead of `2c804a35…`. That is
the same Ampere-vs-Blackwell signature seen on the HF side, so per the standing
rule the local C++ stage captures are indicative only.

**Claude Code is not required on Win11.** Everything below is PowerShell.

---

## 1. Files to copy Win10 → Win11

Two source files. **Both changes are callback-only — no arithmetic is altered.**
`longcat-flash-ngram.cpp` changes only `cb()` labels plus one added `cb()`;
`debug.cpp` changes only the dump helper.

| From (Win10, `…\llama.cpp-longcat-claude\`) | To (Win11, `D:\llama.cpp-longcat-pre-gate4\`) |
|---|---|
| `src\models\longcat-flash-ngram.cpp` | `src\models\longcat-flash-ngram.cpp` |
| `common\debug.cpp` | `common\debug.cpp` |

| File | SHA256 | Bytes |
|---|---|---|
| `longcat-flash-ngram.cpp` | `7f726913922e58bbffbc008eba0067e328543683094904107af4f7501b2e9a47` | 46446 |
| `common\debug.cpp` | `9cbaddc5ed7eb3413ddbd5cf276da83a4c4a50d6aea3c06ddf821fe52497a4a7` | 20184 |

The files being replaced are the frozen handoff versions
(`aaff66b6…86cf1` and `ee673463…c3fc4`). Both are in git, so back them up or
rely on `git diff` to restore.

### What changed

`src\models\longcat-flash-ngram.cpp` — six `cb()` sites:

- three colliding block-0 `cb(q, "q", il)` → `q_a_proj` / `q_a_norm` / `q_b_proj`
- post-norm `cb(kv_cmpr, "kv_cmpr", il)` → `kv_a_norm` (the pre-norm view already used `kv_cmpr`)
- **new** `cb(cur, "attn_out", il)` after `build_attn` returns, before the residual add

That last one matters: `build_attn` emits `kqv_out` *before* applying `wo`, so
`kqv_out-0` is 4096-wide pre-projection output, not HF `o_proj`. `attn_out-0`
is the post-`wo`, pre-residual surface.

`common\debug.cpp` — dump helper only: width generalized off the hardcoded 3072,
full-sequence token-major mode with JSON sidecars for the six MLA surfaces
(existing surfaces keep final-row 12288-byte layout), stride-aware reads via
`nb[]`, and dump targets requested at ask time so `--tensor-filter` need not
cover them.

---

## 2. Rebuild

```powershell
$ErrorActionPreference = 'Stop'
$REPO  = 'D:\llama.cpp-longcat-pre-gate4'
$BUILD = 'D:\llama.cpp-longcat-pre-gate4-build-cuda132'
$expect = @{
  (Join-Path $REPO 'src\models\longcat-flash-ngram.cpp') = '7f726913922e58bbffbc008eba0067e328543683094904107af4f7501b2e9a47'
  (Join-Path $REPO 'common\debug.cpp')                   = '9cbaddc5ed7eb3413ddbd5cf276da83a4c4a50d6aea3c06ddf821fe52497a4a7'
}
foreach ($p in $expect.Keys) {
  $h = (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower()
  if ($h -ne $expect[$p]) { throw "SHA mismatch: $p`n  got $h" }
  Write-Host "OK  $(Split-Path $p -Leaf)" -ForegroundColor Green
}
cmake --build $BUILD --config Release --target llama-debug -- /m:8 /v:minimal
Write-Host "BUILD EXITCODE=$LASTEXITCODE"
```

If `cmake` is not on PATH, it may live in the venv as
`…\llama-cpp-env\Scripts\cmake.exe` (that is where it was on Win10).

---

## 3. Run

The flags below are reconstructed from your own frozen log
`cpp_attn0_hf_rmsnorm_mlaeps1e6_qbf16_512.log`, which recorded
`offloaded 30/30 layers to GPU`, `n_ctx = 4608`, `n_batch = 4608`,
`n_ubatch = 512`, `flash_attn = disabled`, and `-ot` overrides sending the
`ffn_*` expert / shared-expert / gate-inp tensors to `CUDA_Host`.

**The `2c804a35…` gate in step 4 is the arbiter.** If it fails, the placement
does not match what produced the frozen baseline — adjust `-ot`/`-ngl` to match
your original invocation rather than accepting the result.

```powershell
$ErrorActionPreference = 'Continue'
$EXE    = Join-Path $BUILD 'bin\Release\llama-debug.exe'
$GGUF   = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf'
$PROMPT = Join-Path $REPO 'prompt_512_a.txt'
$OUT    = Join-Path $REPO 'cpp_attn0_mla_blackwell_512'
$LOG    = Join-Path $REPO 'cpp_attn0_mla_blackwell_512.log'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$ot = 'ffn_(down|gate|up)_exps\.weight=CUDA_Host,ffn_(down|gate|up)_shexp\.weight=CUDA_Host,ffn_gate_inp\.weight=CUDA_Host'
$env:LONGCAT_HIDDEN_DUMP_DIR = $OUT
& $EXE -m $GGUF -f $PROMPT -c 4608 -ub 512 -n 0 -ngl 99 -ot $ot -fa off -ctk f32 -ctv f32 --no-warmup --tensor-filter '(q_a_proj-0|q_a_norm-0|q_b_proj-0|kv_cmpr_pe-0|kv_a_norm-0|attn_out-0|ffn_inp-0)$' 2>&1 | Tee-Object -FilePath $LOG | Out-Null
Write-Host "EXITCODE=$LASTEXITCODE"
Remove-Item Env:LONGCAT_HIDDEN_DUMP_DIR
```

Notes:

- `--no-warmup` is required. Warmup is on by default and would run a dummy
  graph that overwrites the dumps with garbage.
- No `--save-logits` — the callback is not installed when `save_logits` is active.
- There is deliberately **no early-abort env var.** One was attempted and
  removed: returning false from a ggml eval callback only breaks the current
  split's node loop, not the graph, and it skips the rest of that split so
  anything downstream is computed from incomplete state. A full 512-token
  forward completes fine.

---

## 4. Gate and verify

```powershell
$anchors = @{
  'inp_embd_ngram.bin'          = 'd0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f'
  'logical0_attn0_norm.bin'     = 'a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af'
  'logical0_attn0_resid.bin'    = '2c804a35a0397e380d77f08d2a7ffd11fc6e4672722c31a98f7f96620c2a4a4e'
}
foreach ($k in $anchors.Keys) {
  $p = Join-Path $OUT $k
  if (-not (Test-Path $p)) { Write-Host "MISSING $k" -ForegroundColor Red; continue }
  $h = (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower()
  if ($h -eq $anchors[$k]) { Write-Host "PASS  $k" -ForegroundColor Green }
  else { Write-Host "FAIL  $k`n  exp $($anchors[$k])`n  got $h" -ForegroundColor Red }
}
Get-ChildItem $OUT -Filter '*.bin' | Sort-Object Name | ForEach-Object { "{0,-30} {1,10}" -f $_.Name, $_.Length }
```

Required sizes for the six MLA surfaces: `q_a_proj` 3,145,728 ·
`q_a_layernorm` 3,145,728 · `q_b_proj` 12,582,912 · `kv_a_proj_with_mqa`
1,179,648 · `kv_a_layernorm` 1,048,576 · `o_proj` 6,291,456. Each must have a
matching `.json` sidecar.

**`logical0_attn0_resid.bin` = `2c804a35…` is the gate.** All three anchors
should pass; the first two already pass on Ampere, so a failure there means
something drifted in the rebuild.

---

## 5. Copy back Win11 → Win10

| Win11 source | Win10 destination |
|---|---|
| `D:\llama.cpp-longcat-pre-gate4\cpp_attn0_mla_blackwell_512\` (whole directory) | `…\llama.cpp-longcat-claude\_external_artifacts\cpp_mla_blackwell\` |
| `D:\llama.cpp-longcat-pre-gate4\cpp_attn0_mla_blackwell_512.log` | same directory |

Please include a `SHA256SUMS.txt` as you did for the HF capture:

```powershell
Push-Location $OUT
Get-ChildItem -File | Where-Object { $_.Name -ne 'SHA256SUMS.txt' } | ForEach-Object { "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower(), $_.Name } | Set-Content -Encoding ascii 'SHA256SUMS.txt'
Pop-Location
Get-Content (Join-Path $OUT 'SHA256SUMS.txt')
```

The whole directory is ~27 MB of `.bin` plus the small final-row files and
sidecars. The log is worth including — it records the offload placement, which
is what makes the `2c804a35…` gate meaningful.

---

## What this unblocks

Once the Blackwell C++ surfaces are back, the comparison against the
authoritative Blackwell HF oracles becomes apples-to-apples, and
`compare_longcat_attn0_mla_stages.py` produces the authoritative answer. The
indicative Ampere run already points hard at `o_proj`; this run is what makes
that conclusion citable.
