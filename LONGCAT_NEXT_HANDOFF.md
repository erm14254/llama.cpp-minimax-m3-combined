# LongCat-Next llama.cpp handoff — 2026-08-08

## Scope

This is a handoff checkpoint for the LongCat-Next work in:

- Repository: `erm14254/llama.cpp-minimax-m3-combined`
- Draft PR: `#9 Add LongCat-Next Stage-1 text-core support`
- PR branch: `codex/longcat-next-core-cpp-spike`
- Last remote PR head before this handoff: `fafdf80fa19a426a0eca2820dfa76832bcbfe8c5`

The existing PR implements the Stage-1 **text core**, not complete LongCat-Next multimodal/image-generation support. Vision/audio/image-generation integration remains substantial separate work.

## Current local WIP state

At handoff time the working tree contains three modified files:

- `src/llama-graph.cpp`
- `src/models/longcat-flash-ngram.cpp`
- `tests/longcat-next-capture.cpp`

The local state combines:

1. Production-candidate-v1 numerical changes:
   - Python-contract BF16 n-gram fusion.
   - LongCat MLA Wv pre-expansion before attention accumulation.
2. Diagnostic LongCat trunk weighted-RMSNorm change:
   - BF16 boundary on the normalized activation before norm-weight multiplication.
3. Capture-only instrumentation:
   - unique block-0 Q-LoRA and compressed-KV callback names for internal localization.

This state is **diagnostic/WIP, not accepted production behavior**.

## Important preserved patches / hashes

Production candidate v1:

`longcat-next-bf16-production-candidate-v1.patch`

SHA-256:

`c3f49e532762aec862df3c1c996542d69926f472e0bca9279b91cab7cd91901a`

LongCat trunk pre-weight RMSNorm diagnostic:

`longcat-trunk-rmsnorm-preweight-bf16-diagnostic-v1.patch`

SHA-256:

`6a043ccc434cfc0e814d67067fa0ef30cc68eb8b1c11728792e3f50f52fbe634`

Capture-only Q/KV stage instrumentation v3:

`longcat-layer0-qkv-internal-capture-only-v3.patch`

SHA-256:

`b7efec3f3067391853a7a94a7598a28c559aa4fe7c3ddacc661332a4b0856396`

## What was established

### 1. N-gram BF16 fusion

The original C++ path accumulated projected n-gram values in F32 and rounded late.

The Python reference behavior is equivalent to:

- BF16-round each projected n-gram source;
- sequential BF16 accumulation;
- BF16 result through the division/output boundary.

Using the Python-style contract made the fused input and block-0 input exact.

### 2. Wv placement in MLA

Using identical BF16 attention probabilities and compressed BF16 V:

- applying Wv before attention accumulation produced Python's robust block-2 token-1 router expert `199`;
- applying Wv after compressed-value accumulation produced `207`.

The actual C++ pre-expanded-Wv diagnostic changed the first canonical surface at `attn_out-0` while Q/K/cache/KQ/softmax were unchanged between those Wv controls.

This is operational mechanism evidence, not proof that Wv is the exclusive cause of all remaining parity failures.

### 3. LongCat trunk weighted RMSNorm BF16 boundary

Across blocks 0–5, both attention and FFN trunk norms showed the same local arithmetic contract:

Python:

`BF16 input -> F32 RMS normalization -> BF16 normalized activation -> BF16 weight multiply -> BF16 result`

Original C++:

`BF16-grid input -> F32 RMS normalization -> F32 weight multiply -> BF16 result`

A LongCat-specific diagnostic moved the BF16 boundary before the norm-weight multiplication. Under that diagnostic, block-0 `attn_norm-0` became bit-exact to both Python-default and Python-math.

This did **not** make the whole model pass. The first block-level failure remained physical block 4.

### 4. Corrected norm re-opened the block-0 attention core

With the corrected, Python-exact `attn_norm-0`:

- authentic Python block-0 attention reproduced the Python downstream path exactly;
- captured C++ `attn_out-0` still differed;
- replaying captured C++ attention output remained sufficient to produce downstream block-4 failure.

Therefore, in the current diagnostic source state:

`same_normalized_input_attention_core_residual_required = true`

The older conclusion that the block-0 attention core was not required applied only to the previous, non-exact attention-norm trajectory.

### 5. Latest Q/KV internal-stage localization

Capture-only instrumentation exposed:

Q branch:

`attn_norm -> q_a_proj -> q_a_norm -> q_b_proj -> q_scaled`

KV branch:

`attn_norm -> kv_cmpr_pe -> kv_cmpr_pre_norm -> kv_cmpr_norm -> kv_cmpr_scaled`

The official Python BF16 attention endpoint was reconstructed bit-exactly for both Python-default and Python-math before interpreting these stages.

Latest findings:

- `q_a_proj-0`
  - raw C++ is extremely close to an F32 Python diagnostic (`RMS ~2.0e-08`);
  - BF16-rounding C++ matches Python BF16 at `7677 / 7680` elements;
  - only 3 BF16 elements remain different.
- `q_a_norm-0`
  - diverges materially after the projection.
- `kv_cmpr_pe-0`
  - raw C++ is extremely close to F32 Python;
  - BF16-rounding C++ reproduces the official Python BF16 projection **exactly**.
- `kv_cmpr_pre_norm-0`
  - same: BF16-rounding C++ reproduces Python exactly.
- `kv_cmpr_norm-0`
  - diverges again after the exact projection boundary.
- C++ `q_b_proj -> q_scaled` scaling relation is exact F32.
- C++ `kv_cmpr_norm -> kv_cmpr_scaled` scaling relation is exact F32.

The next diagnostic that was prepared but not yet treated as an accepted result is:

`replay-longcat-block0-internal-norm-contract-preweight-rms-v1.py`

SHA-256:

`a8017120ae326f68b4fa93617df5f438965755a49b2fc5ad37ac43b717c74e7a`

Its purpose is to isolate the internal `q_a_layernorm` and `kv_a_layernorm` contracts using exact captured projection inputs and authentic Python modules.

## Acceptance status

The implementation has **not** passed the required frozen full:

**10-case / 433-array BF16 CPU parity gate**

Tolerances were not widened.

The current diagnostic source state must not be described as production-ready or parity-complete.

## Multimodal / image-generation status

The current PR is Stage-1 text-core work.

LongCat-Next image/audio functionality is not implemented by this work. The PR intentionally deferred visual/audio tensors and multimodal integration, including image/audio understanding/generation paths, modality tokenizers/detokenizers, decoder/refiner/vocoder work, and MTMD/product integration.

Anyone taking over specifically for LongCat-Next image generation should treat this branch as a text-backbone foundation, not as a nearly-complete multimodal implementation.

## Guardrails used during the investigation

- Python-default is the primary numerical oracle.
- Python sdpa-math is a sensitivity control only.
- Do not widen tolerances based on observed C++ output.
- Sufficiency does not establish exclusivity/root causality.
- Do not infer a Q/K/router/MLP/etc. production fix merely from a downstream mismatch.
- Preserve diagnostic changes separately from accepted production changes where possible.
- Full acceptance gates remain mandatory after any eventual production change.

## Suggested takeover strategy

A practical takeover does not need to repeat the entire investigation.

1. Start from the handoff WIP commit/branch.
2. Read this note and the latest internal-stage report.
3. Decide whether to finish the internal Q/KV RMSNorm numerical work or park exact BF16 parity.
4. If the actual goal is image generation, separately inventory and implement the deferred multimodal/image stack instead of assuming text parity work completes it.
5. Keep any further diagnostic-only changes clearly marked as WIP until validated.
