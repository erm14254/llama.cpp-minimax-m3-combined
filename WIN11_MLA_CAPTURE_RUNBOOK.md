# Win11 Blackwell — Authoritative HF Block-0 MLA Stage Capture

**Purpose:** produce the six HF block-0 MLA stage surfaces on the original
RTX PRO 6000 Blackwell machine, where the frozen oracles were made, so the
capture clears all three byte-exact gates.

**Why this must run on Win11:** on the Win10 RTX 3090 the harness clears gates 1
and 2 byte-exact but misses gate 3 by ~1 BF16 ULP on 19.8% of elements
(rel-RMSE 0.00172, cosine 0.9999985, 2464/3072 elements bit-identical).
transformers 5.14.1 and 5.15.0 produce byte-identical output there, so the
library version is excluded; the remaining variable is Ampere-vs-Blackwell
cuBLAS BF16 kernel selection.

**Claude Code is not required on Win11.** Every step below is plain PowerShell.

---

## 1. Files to copy Win10 → Win11

Exactly **one** file. Do not copy the model, the GGUF, or any oracle — the
originals under `D:\` are used in place.

| From (Win10) | To (Win11) |
|---|---|
| `C:\Users\Alan-PC\Downloads\llama.cpp-longcat-claude\capture_longcat_hf_attn0_mla_stages.py` | `D:\llama.cpp-longcat-pre-gate4\capture_longcat_hf_attn0_mla_stages.py` |

---

## 2. SHA256 values

**File you are copying:**

| File | SHA256 | Bytes |
|---|---|---|
| `capture_longcat_hf_attn0_mla_stages.py` | `4aaec929452402fddec4c56f043799ba46ef424b0754e88723816d1fa9d27db0` | 20305 |

**Inputs already on `D:\` — verified, not copied:**

| File | SHA256 |
|---|---|
| `D:\LongCat-...-LSA-Preserved\modeling_longcat_flash_sparse.py` | `a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428` |
| `D:\llama.cpp-longcat-pre-gate4\sparse_512_fa_off\...-tokens.bin` | `4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c` |
| `D:\llama.cpp-longcat-pre-gate4\hf_logical0_stages_512_v4\attn0_resid.bin` | `2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177` |

The script re-verifies the runtime and token SHAs itself and aborts on mismatch.

---

## 3. PowerShell commands (Win11)

### Step 3.1 — Set paths and verify every input

```powershell
$ErrorActionPreference = 'Stop'
$MODEL   = 'D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved'
$REPO    = 'D:\llama.cpp-longcat-pre-gate4'
$SCRIPT  = Join-Path $REPO 'capture_longcat_hf_attn0_mla_stages.py'
$TOKENS  = Join-Path $REPO 'sparse_512_fa_off\llamacpp-LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008-tokens.bin'
$ORACLE  = Join-Path $REPO 'hf_logical0_stages_512_v4\attn0_resid.bin'
$OUTDIR  = Join-Path $REPO 'hf_attn0_mla_stages_512'
$SCRATCH = 'D:\lcscratch'
$LOG     = Join-Path $REPO 'hf_attn0_mla_stages_512.log'
$expect = @{
  $SCRIPT = '4aaec929452402fddec4c56f043799ba46ef424b0754e88723816d1fa9d27db0'
  (Join-Path $MODEL 'modeling_longcat_flash_sparse.py') = 'a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428'
  $TOKENS = '4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c'
  $ORACLE = '2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177'
}
$bad = 0
foreach ($p in $expect.Keys) {
  if (-not (Test-Path $p)) { Write-Host "MISSING  $p" -ForegroundColor Red; $bad++; continue }
  $h = (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower()
  if ($h -ne $expect[$p]) { Write-Host "MISMATCH $p`n  got $h" -ForegroundColor Red; $bad++ }
  else { Write-Host "OK       $(Split-Path $p -Leaf)" -ForegroundColor Green }
}
if ($bad -gt 0) { throw "$bad input file(s) failed verification - do not run the capture" }
Write-Host 'All inputs verified.' -ForegroundColor Green
```

### Step 3.2 — Select the Python interpreter

Use the environment that produced the frozen artifacts (`torch 2.13.0+cu132`).
**Do not install or upgrade anything.** Point `$PY` at that interpreter, e.g.
its venv `Scripts\python.exe`, then confirm:

```powershell
$PY = 'D:\llama.cpp-longcat-pre-gate4\llama-cpp-env\Scripts\python.exe'
if (-not (Test-Path $PY)) { throw "Set `$PY to the interpreter that has torch + transformers" }
& $PY -c "import torch,transformers,numpy,safetensors,sys;print('python      ',sys.version.split()[0]);print('torch       ',torch.__version__);print('cuda avail  ',torch.cuda.is_available());print('device      ',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE');print('transformers',transformers.__version__);print('numpy       ',numpy.__version__);print('safetensors ',safetensors.__version__)"
```

Expect `torch 2.13.0+cu132`, CUDA available, and an RTX PRO 6000 device name.
`transformers` 5.14.1 or 5.15.0 are both fine — proven byte-identical.

### Step 3.3 — Run the capture

```powershell
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $SCRATCH | Out-Null
& $PY $SCRIPT --model-dir $MODEL --tokens-bin $TOKENS --out-dir $OUTDIR --scratch-dir $SCRATCH --attn-impl sdpa --oracle-resid $ORACLE 2>&1 | Tee-Object -FilePath $LOG
Write-Host "EXITCODE=$LASTEXITCODE"
```

**Exit code 0 and `HF ATTN0 MLA STAGE CAPTURE: PASS` means success.**

Keep `$SCRATCH` short (`D:\lcscratch`). A long scratch path can push the
runtime package copy past Windows `MAX_PATH`.

### Step 3.4 — Verify the outputs

```powershell
Get-ChildItem $OUTDIR | Sort-Object Name | ForEach-Object { "{0,-26} {1,10}  {2}" -f $_.Name, $_.Length, (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower() }
```

### If it fails

- **Gate 1 or 2 fails** — unexpected on Win11; stop and send the log. Do not
  change the script or the expected hashes.
- **Gate 3 fails** — the log prints max_abs / RMSE / rel_RMSE / cosine /
  exact_elems against the frozen oracle. Send those numbers; if they match the
  Win10 figures (rel-RMSE 0.00172, 2464/3072 exact) the cause is not the GPU
  after all. Do not widen the gate.
- **`--attn-impl`** — leave it at `sdpa`. `eager` and `flex_attention` are
  strictly worse on Win10 and are only diagnostic alternatives.

---

## 4. Python environment requirements

Nothing to install. The script imports only what the frozen environment
already has:

- `torch` 2.13.0+cu132 (CUDA), `transformers` (5.14.1 or 5.15.0),
  `numpy`, `safetensors`

Runtime characteristics, measured on the RTX 3090:

- peak VRAM well under 2 GB — only block-0 weights plus `embed_tokens` load
- the 58.5 GiB N-gram tables are **not** materialized; 36 unique rows are read
  by `safe_open(...).get_slice(...)`
- weight load ≈ 2.2 s, forward ≈ 0.01 s
- the canonical model directory is opened read-only; the frozen runtime is
  copied into `$SCRATCH` and SHA-verified there

---

## 5. Expected gate SHAs

All three must print `PASS`:

| Gate | Surface | Required SHA256 |
|---|---|---|
| 1 | block-0 input, row 511 | `d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f` |
| 2 | `input_layernorm[0]` output, row 511 | `a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af` |
| 3 | `input + o_proj` residual, row 511 | `2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177` |

The log should also show `lsa_mode = full-owner` (the exact full-attention path
at 512 tokens) and `n-gram rows read = 36`.

---

## 6. Files to copy back Win11 → Win10

Copy the whole output directory plus the log — **27,394,048 bytes** of `.bin`
plus small JSON.

| Win11 source | Win10 destination |
|---|---|
| `D:\llama.cpp-longcat-pre-gate4\hf_attn0_mla_stages_512\` (entire directory) | `C:\Users\Alan-PC\Downloads\llama.cpp-longcat-claude\hf_attn0_mla_stages_512\` |
| `D:\llama.cpp-longcat-pre-gate4\hf_attn0_mla_stages_512.log` | same directory as above |

Expected directory contents — 13 files:

| File | Bytes |
|---|---|
| `q_a_proj.bin` | 3,145,728 |
| `q_a_layernorm.bin` | 3,145,728 |
| `q_b_proj.bin` | 12,582,912 |
| `kv_a_proj_with_mqa.bin` | 1,179,648 |
| `kv_a_layernorm.bin` | 1,048,576 |
| `o_proj.bin` | 6,291,456 |
| `q_a_proj.json`, `q_a_layernorm.json`, `q_b_proj.json`, `kv_a_proj_with_mqa.json`, `kv_a_layernorm.json`, `o_proj.json` | small |
| `summary.json` | small |

Every `.bin` is canonical **token-major `[512, width]` little-endian F32**.
The `.json` sidecars carry shape, dtype, whole-tensor SHA256, final-row SHA256,
and min/max — the comparator reads shape from these and hard-fails on mismatch
rather than inferring it from file length.

The log matters as much as the vectors: it records the Win11 torch/transformers
versions, `lsa_mode`, and the three gate SHAs that make the capture
authoritative.
