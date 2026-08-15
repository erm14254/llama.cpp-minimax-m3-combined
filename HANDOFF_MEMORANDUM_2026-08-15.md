# LongCat-Flash-Lite-Sparse / Heretic / llama.cpp Handoff Memorandum

**Date:** 2026-08-15

## Purpose

This handoff preserves the current engineering state for adapting Heretic abliteration to Meituan `LongCat-Flash-Lite-Sparse`, preserving actual LongCat Sparse Attention (LSA), CLI reuse, native 3-step MTP behavior, compact Safetensors export, and native llama.cpp/GGUF support.

This is a **diagnostic handoff**, not a production-ready release state. The current local parity tree intentionally contains experimental precision patches used to localize numerical differences between the frozen Hugging Face reference and llama.cpp.

The recommended continuation environment is Claude Code Desktop operating directly on the user's Windows filesystem and local Git repository so that it can inspect/edit files, build, run diagnostics, and summarize results without manual command copy/paste.

## Non-negotiable constraints

- Preserve actual LongCat Sparse Attention / learned sparse selection. Do not replace it with a dense surrogate for release runtime.
- Preserve CLI owner/reuse behavior exactly.
- Preserve native MTP semantics: **one physical parameterized MTP block reused for three conceptual MTP steps**. Never conflate physical block count and conceptual step count.
- Do not set `NEXTN_PREDICT_LAYERS=3`; the checkpoint has one physical MTP block.
- Do not enable tokenizer regex fixes unless a concrete tokenizer defect is proven. Validated behavior uses `fix_mistral_regex=False` semantics.
- Never overwrite canonical/hard-linked source Safetensors.
- Never widen the frozen C++↔HF parity criterion to make a failure pass.
- No repaired 2050-token run until the 512-token common-path numerical issue is resolved.
- No production Flash Attention patch/commit yet. The broad Q prescale is diagnostic only.
- Do not rewrite/reset the already-pushed Gate4 WIP handoff branch.
- Generated Python scripts: SHA256 + `python -m py_compile` before execution.
- Prefer bounded fail-fast Windows PowerShell 5.1 commands.

## Architecture/checkpoint facts

- 14 logical decoder layers / 28 physical attention sublayers.
- hidden size 3072.
- dense FFN 6144.
- expert FFN 1024.
- 256 routed experts + 128 identity/zero expert slots.
- top-k experts 12.
- sparse indexer `index_topk=2048`.
- index local tokens 1024.
- index init/sink tokens 16.
- CLI factor 2.
- MTP = 3 conceptual steps.
- `mtp_replicate_modules=true`.
- `dsa_mtp_cli=true`.
- Pristine source: 26 Safetensors.
- 56 main indexer tensors.
- 21 physical MTP tensors.
- 60 total `.indexer.` tensors, therefore 4 physical MTP indexer tensors.

### LSA semantics

At total KV length `<=2048`, the frozen reference takes the mathematically exact full-attention path. Above 2048, the learned indexer selects a fixed top-K of 2048. First 16 init/sink tokens and the local 1024-token region are forced **inside** that fixed top-K via `+inf`; they do not increase K beyond 2048. CLI owner/reuse is exact.

For conceptual MTP, step 1 owns/computes sparse top-K and conceptual steps 2/3 reuse the exact step-1 top-K when `dsa_mtp_cli=true`.

## Frozen HF / Heretic baseline

Frozen public runtime candidate v4:

- file: `modeling_longcat_flash_sparse.py`
- SHA256: `a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428`

Do not modify frozen v4.

Fresh-study Heretic patches:

- `main.py` v2 SHA256: `345454b1df7617a8065259c51ad382c434b494370f4a24e73182cee827097f0b`
- `model.py` v2 SHA256: `8a621498bf1fd67862c17f445694b509b98edf56055cbbdea1201c48b08523d7`

HF Gates 1–4 previously passed and should not be rerun unless a code change invalidates them.

Heretic optimization/evaluation stayed in the `<=2048` exact full-attention region (max template input 49, max requested context 149).

## Model source/provenance

Canonical pristine:

`D:\LongCat-Flash-Lite-Sparse`

Equivalent hard-linked source:

`D:\LongCat-Flash-Lite-Sparse-MTP`

Do not overwrite those source shards.

Actual Trial283 Heretic export:

`D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved`

Engineering BF16 GGUF:

`D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16`

Trial283 export is intentionally 31 shards due Heretic resharding. It is not a semantic failure versus pristine 26 shards.

Trial283 export facts:

- corrected indexed size ~128.761 GiB.
- Safetensors size ~128.762 GiB.
- keys source/export 11220/11220, missing 0, added 0.
- split experts gate/up/down 3584/3584/3584.
- MTP preserved 21/21.
- LSA indexers preserved 60/60.

## Heretic outcomes

### Trial283 — Low-KL

Parameters:

- `direction_index=per layer`
- `attn.o_proj.max_weight=2.18`
- `attn.o_proj.max_weight_position=9.23`
- `attn.o_proj.min_weight=0.68`
- `attn.o_proj.min_weight_distance=3.70`
- `mlp.down_proj.max_weight=1.06`
- `mlp.down_proj.max_weight_position=12.88`
- `mlp.down_proj.min_weight=0.66`
- `mlp.down_proj.min_weight_distance=5.69`

Metrics: study KL 0.0157, post-export ~0.0156. More refusals, lower unrelated drift.

### Trial3 — Low-Refusal

KL 0.0779. Human review found almost no genuine refusals, but unrelated drift risk is higher. Exact Trial3 parameter vector/post-export literal count is not preserved here; do not invent it.

Recommended names: `...-Low-KL`, `...-Low-Refusal`.

## llama.cpp repositories/branches

Fork:

`https://github.com/erm14254/llama.cpp-minimax-m3-combined`

Primary implementation checkout:

`D:\llama.cpp-longcat-mtp`

Local branch there: `longcat-sparse`.

Important commits:

- Gate2 loader: `484db978356bcff6e2c53f7bca6fa09f5aa8087d`
- Gate3 BF16 cache fix: `cb7729cf18088ae6cd6d9cac52e3ee536be02dc4`
- pushed WIP recovery commit: `98f5dd1ccbf484bc8e95dbc47b49a64238131d2a`

Already-pushed recovery branch:

`handoff/longcat-sparse-gate4-wip-20260814`

Do not rewrite it.

### Current parity diagnostic checkout

`D:\llama.cpp-longcat-pre-gate4`

Build directory:

`D:\llama.cpp-longcat-pre-gate4-build-cuda132`

Push this diagnostic tree to a **separate branch**, recommended:

`handoff/longcat-parity-diagnostics-20260815`

## GGUF gates

- Gate1 PASS
- Gate2 PASS
- Gate3 PASS
- Gate4 OPEN
- Gate5 pending: conceptual 3-step MTP owner/reuse/reuse
- Gate6 pending: generation crossing 2048
- Gate7 pending: quantized GGUF

Frozen Gate3 parity criterion — never widen:

For **all** 131072 logits:

`abs(cpp-hf) <= 0.5 + 0.05*abs(hf)`

and top-1 must agree.

Gate3 exact raw token IDs: `[20769,235,3121,224]`.

Gate3 passed with max abs 0.7879, mean 0.1207, RMSE 0.15094, cosine 0.99984249, violations 0, top1 both 444.

## Gate4 structural LSA status

A 2050-token structural run already proved:

- owners even physical blocks 0..26: 14.
- reuse odd physical blocks: 14.
- fixed top-K 2048.
- final query position 2049 saw all 2050 causal keys.
- forced set size 1040 = first 16 + local 1026..2049.
- sparse LSA structure PASS.

Logits were all NaN with Flash Attention enabled, which led to the numerical investigation below.

## Flash Attention NaN root cause — proven

Exact 512-token diagnostic prompt:

`D:\llama.cpp-longcat-pre-gate4\prompt_512_a.txt`

It is `(" a"*512)` and tokenizes to 512 copies of token ID 483.

Token stream SHA256:

`4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c`

Observed on the clean pre-Gate4 tree:

- FA on -> NaN.
- FA off -> finite.
- BF16/F32 KV and ubatch changes did not eliminate FA NaN.
- first NaN localized to physical block 0 `FLASH_ATTN_EXT` with finite Q/K/V immediately before it.
- Q head 18 max abs ~157318.
- FP16 max is 65504.

CUDA MMA-F16 narrowed F32 Q to half before multiplying by scale, so the cast overflowed before scaling. A broad diagnostic `half(scale*tmp)` change removes the NaN, proving a real narrow FA overflow defect.

Do not promote the broad prescale patch to production yet. A likely production direction is an overflow-only fallback that preserves the old in-range path.

## HF 512 oracle / finite common-path parity

HF 512 logits oracle:

`D:\llama.cpp-longcat-pre-gate4\hf_sparse_512_v4.bin`

SHA256:

`8825d92d7d9cdea42a4ea3aa2e3df5766bdf880323b1f48ea8c17ff63f3c5ecf`

HF top1 = 483.

Best early finite common-path baseline was FA-off + F32 KV:

- max abs 1.8465769
- mean 0.2416432
- RMSE 0.3069317
- cosine 0.999293288
- frozen-criterion violations 2122
- top1 both 483
- top20 20/20

Therefore BF16 cache precision is not the only parity issue.

## Output-row audit — exact

HF `lm_head.weight` vs GGUF `output.weight` rows at indices:

`483, 15626, 15777, 25433, 39590, 112084, 122091`

were decoded exact and raw-BF16 exact. Output-head conversion is not the source of the recurrent logit outliers; the error is upstream.

## N-gram arithmetic root cause — proven

Static N-gram data was exact:

- all 6144 hash IDs match.
- selected token embedding row exact.
- 12 selected N-gram rows exact.
- 12 projection matrices exact.

Standalone reconstruction showed original C++ matched full-F32 arithmetic, while HF requires BF16 intermediate rounding. A diagnostic patch added BF16 boundaries around N-gram projection/accumulation/final division and losslessly restored F32 where the graph requires it.

Current diagnostic input is byte-exact HF:

`inp_embd_ngram` SHA256:

`d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f`

## Logical-layer-0 localization

Frozen HF logical-0 stage SHAs:

- input: `d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f`
- attn0_resid: `2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177`
- mlp0_resid: `cf48a0ad3001e82ae41020675458df66219ea929caa1168ffddf64196d70404f`
- attn1_resid: `b4c1e5f684afefcec4129e3e6ec095a38d9b7f880115f819f78f8a698fe14431`
- logical0_out: `5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff`

After exact N-gram input, the first material divergence was already at `attn0_resid`, localizing the next issue to physical attention block 0.

## Main block-0 attention RMSNorm — proven exact after diagnostic patch

HF standalone block-0 input RMSNorm oracle SHA:

`a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af`

HF weight: `model.layers.0.input_layernorm.0.weight`.

GGUF weight: `blk.0.attn_norm.weight`.

GGUF stores the norm weight as F32, but decoded F32 values are exactly HF BF16 values expanded to F32. Decoded weight SHA on both sides:

`f4de45bb014ce6cb200ba0e94285fe0ae7d757b81d69d8b16a36a9bb96f3af30`

Transformers LongCat RMSNorm computes normalization in F32, rounds normalized activation to input dtype BF16, multiplies by BF16 weight, and returns BF16. Generic llama.cpp was effectively all-F32.

A block-0 diagnostic patch reproduced HF precision boundaries and made C++ `attn_norm-0` **byte-exact**:

`a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af`

With exact input and exact main RMSNorm, the remaining `attn0_resid` baseline is:

SHA256 `8ea9b911d4810982af4186e66562cb5f316e7a0a9c2439101f6654eb10887dfd`

- max abs 0.0122458
- RMSE 0.000454813
- rel-RMSE 0.0146032

Therefore another independent mismatch exists inside MLA.

## MLA epsilon experiment

HF `q_a_layernorm` and `kv_a_layernorm` use LongCat RMSNorm default epsilon `1e-6`. Generic C++ used global model RMS epsilon `1e-5`.

Block-0-only change to `1e-6` produced residual SHA:

`c2b8473b9d044ba50a978e7249a694b81f111cd5bc434b585ecd776a922c2199`

rel-RMSE 0.0148132, about 1.44% worse RMSE than the best exact-main-norm baseline. The semantic correction is real, but epsilon alone is not the dominant error.

## Q-side BF16 semantics experiment

A block-0-only Q diagnostic emulated BF16 boundaries through `wq_a`, Q-A RMSNorm, `wq_b`, and Q scaling while retaining eps `1e-6`.

Current local `src/models/longcat-flash-ngram.cpp` SHA:

`aaff66b65e5fc4ca245cfe6b379a60b6bfae268b94cf5b69f0dfd7ca10486cf1`

Current `common/debug.cpp` SHA:

`ee673463453c3c7f39de4d43a778551c7db97f8ee42bd0e936ddffd3994c3fc4`

Current Q-BF16 residual SHA:

`2c804a35a0397e380d77f08d2a7ffd11fc6e4672722c31a98f7f96620c2a4a4e`

rel-RMSE 0.0152371. This is ~4.34% worse RMSE than the best exact-main-norm baseline.

Therefore do **not** generalize more BF16 patches by guesswork. Current source is a diagnostic state, not the best-parity state.

## Current diagnostic source contents

`src/models/longcat-flash-ngram.cpp` currently contains:

- proven N-gram BF16 parity diagnostic + restore-F32 widening.
- proven block-0 main attention RMSNorm HF precision diagnostic.
- block-0 Q-A/KV-A epsilon `1e-6` diagnostic.
- block-0 Q BF16 semantic diagnostic.
- inherited diagnostic/WIP code in this parity tree.

`common/debug.cpp` contains hidden-vector dump support and logical-0 mappings including `inp_embd_ngram`, `ffn_inp-0`, `l_out-0`, `ffn_inp-1`, `l_out-1`, and `attn_norm-0`.

When using the debug callback, run **without `--save-logits`** because the callback is not installed when `save_logits` is active.

## Important generated-script SHAs preserved in the conversation

- HF 512 raw wrapper: `d267bf29f4df5a52ab1d0495a5df78ec6b2d677476cf94baba368bb4629f1dc2`
- HF hidden capture: `5485211665c3dc942dcf93955a6dc607c673f9890a90216a5b9845729651b54e`
- C++ hidden-dump applicator: `cc60eba1e1a390a86c6f1f405677503fd58c336c0eefd9aac355c4af3a0c156c`
- N-gram inventory: `ac59dcddabaacebe127e8660888e83a5b5214e734eb6e26dee997a13e73d28d4`
- N-gram hash audit: `5ed21d61807ce3df6dab370547830b98918df3642b26e7b7365ff043d9c79005`
- N-gram selected-weight audit: `3bc07c72721ba4d6609f15e2ae504a6a13fc063903a7b3b967a07407b182cd72`
- N-gram BF16 applicator: `9eb2fc2d63a2f1d9adb4e3a0358ef8f6ec797817b7a75c78bbd4f61bb1a1ae4f`
- restore-F32 v2 applicator: `635e4a53a052c5c42e614d9b56ff9e099ed6fb8f183e7b3731fd08dcc30f89ac`
- hidden-after-N-gram comparator: `d29c57c18ced3d43c4cd6c7f1d1bce36c1c63b4799dad01621344f489d290afd`
- logical0 stage dump applicator: `b8e50f3e90512d35493a429d910909b023768de1aea39610b18be77706cb2dcd`
- logical0 stage comparator: `ee4d07fc7c49cd233b55cd490d751ae8e5167332493f1b2c6c75c25e95dd38dc`
- attn0 norm oracle: `ce5077317dce5c3a2d126345a5f0b1e65a99a9a3f54e4187db1fbbcd1a9500b5`
- attn0 norm dump applicator: `e4cc566f565b995f6c4d99f1b4f675beb8e365d7c4e0aff317b164c43307f1d8`
- attn0 norm comparator: `097c01f9a90eb91c774b24951845ab8ba4ab77933f356eff2daf690dad30d78f`
- BF16-rounding diagnostic: `e40cb0f6fc6b8dcaeee01521f7b0d5a8753b295c50c312cc4b031b998b8e5c28`
- RMSNorm semantic audit: `7148a461d82bac19d411bbce308b28a5caf6c3efda93a2b8c85a6ab7de6cee91`
- main attn0 HF-RMSNorm applicator: `92dbae3ae1cbc7d66b96494162da04cfcc947c3c73b7a4a8ed7fd03f4ad88b29`
- patched RMSNorm comparator: `d752223f619b8958ba898ef51dfe3e95372b2240eb87e64fc7b493b75cc0b9cb`
- MLA eps v2 applicator: `be2c332dcf3e32f2da17560f80dec1d7750d259ce7f630f42a0288b58b1e95cd`
- MLA eps comparator: `6cb689cc191a69457bdec7fcadb5be77de2b8c30280ac0b062b8755a45598010`
- Q-BF16 comparator: `f2c64e2e15c2e4f98d330f3062533504fed434fa13979dd4fa7bb698d94574fc`

The Q-BF16 applicator SHA was not pasted back after creation. Retrieve it from the local file if needed; do not invent it.

## Immediate next engineering step

**Do not patch another precision boundary yet.**

Capture actual Hugging Face full-sequence 512-token MLA intermediate surfaces for physical attention block 0, then add callback-only matching C++ dumps.

HF block-0 attention surfaces to capture:

- `q_a_proj`
- `q_a_layernorm`
- `q_b_proj`
- `kv_a_proj_with_mqa`
- `kv_a_layernorm`
- `o_proj`

Use exact 512 raw tokens = 512 copies of token 483, frozen v4 runtime, BF16 CUDA model, TF32 disabled, `use_cache=False`, and save final-token vectors as little-endian F32.

A draft script name from the previous session was:

`capture_longcat_hf_attn0_mla_stages.py`

It had **not yet been run at handoff time**. If a local file exists, inspect/hash/`py_compile` it before executing; otherwise generate a bounded equivalent.

After HF capture, add **callback-only** C++ surfaces for matching Q/KV boundaries. Do not alter arithmetic in the same step. The first real stage divergence determines the next investigation.

## Diagnostic interpretation after stage capture

- divergence at `q_a_proj`: audit Q-A weights/input dtype/GEMM rounding.
- `q_a_proj` matches but Q-A norm diverges: isolate epsilon + BF16 norm boundaries.
- Q path matches through `q_b_proj`: move to compressed KV.
- Q/KV pre-attention stages match: inspect RoPE, scaling, absorption, attention kernel, output projection.
- Do not move to MLP/MoE until attention0 residual is explained.

## Diagnostic build/run baseline

Executable:

`D:\llama.cpp-longcat-pre-gate4-build-cuda132\bin\Release\llama-debug.exe`

Use for common-path diagnostics:

- `--flash-attn off`
- `--cache-type-k f32`
- `--cache-type-v f32`
- exact 512 prompt
- no `--save-logits` when using hidden-dump callback

## Hardware/performance context

Windows workstation:

- RTX PRO 6000 Blackwell Workstation Edition.
- ~95.59 GiB physical VRAM.
- 256 GB RAM.
- WDDM shared-memory spillover possible.

Runtime expands checkpoint to ~149.761 GiB across 482 unique CUDA storages because routed experts are padded from 256 checkpoint experts to 384 runtime slots. This exceeds physical VRAM and can spill through WDDM.

The earlier 2050-token ~951.6 s anomaly was WDDM residency/thrashing, not intrinsic LSA cost. Under healthy observed residency, frozen v4 was ~18.4 s for the 2050 target and ~2.0 s for MTP. Treat these as observed conditions, not universal benchmarks.

## Git handoff strategy

Do not push the parity diagnostic tree onto `handoff/longcat-sparse-gate4-wip-20260814`.

Create a new branch from `D:\llama.cpp-longcat-pre-gate4`, recommended:

`handoff/longcat-parity-diagnostics-20260815`

Commit tracked source modifications, root-level LongCat diagnostic scripts, and these handoff files.

Do not commit model shards, GGUF, `.bin` vectors, `.log` files, build directories, or large generated artifacts.

Use the included `PUSH_HANDOFF.ps1` for conservative staging and a size/extension guard.

## Release status at handoff

Proven:

- HF Heretic edit/export/reload integrity.
- native LSA and physical MTP tensors preserved.
- GGUF structure and llama.cpp loader support.
- 4-token Gate3 parity pass.
- Gate4 sparse owner/reuse structure above 2048.
- FA NaN root cause.
- N-gram BF16 root cause and exact input parity.
- main block-0 RMSNorm root cause and exact norm parity.

Still open:

- 512-token common-path MLA numerical parity.
- production-quality FA overflow fix.
- repaired 2050 run after common path is resolved.
- conceptual 3-step MTP Gate5.
- generation crossing 2048 Gate6.
- quantized GGUF Gate7.
- production cleanup/rebase/PR organization.
