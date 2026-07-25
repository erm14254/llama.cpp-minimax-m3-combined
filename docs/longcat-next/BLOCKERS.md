# LongCat-Next blockers and gates

## Hard blockers

| Scope | Blocker | Exit evidence |
|---|---|---|
| Native MTP | Published Next model and sidecars contain no MTP weights. | Official revision-matched MTP sidecar, or separately trained and validated weights. |
| Multimodal correctness | MTMD embedding-only ubatches do not provide tokens to `llm_graph_input_ngram::set_input`. | Token-aware embedding override contract plus mixed prompt/multi-sequence parity tests. |
| Text correctness | No official golden fixtures have been committed for hashes, embeddings, layers, and logits. | Deterministic PyTorch fixtures covering BOS, zero, EOS, all 53 ignored IDs, and rollback. |
| Modal generation | llama.cpp has no eight-level head/controller with visual CFG and audio guidance state. | Raw modal-code parity and sequence lifecycle tests. |
| Output products | No native refiner/flow/HiFT graphs or bounded binary server outputs. | Independent image-generation and speech-generation exit suites. |

## Evidence gaps closed by this verification

The Work audit called external checkpoint headers a P0 hard gate. That gate is now
closed for metadata:

* image sidecar: 1,771 exact BF16 entries, including previously omitted
  `visual_model.*`;
* HiFT: 328 exact F32 entries and explicit weight-normalized pairs.

Still needed are full-payload hashes, shared-codebook equality checks, folded HiFT
reference tensors, and numerical decoder goldens. Those require payload execution,
not additional architecture speculation.

## Not correctness blockers, but possible product blockers

| Gap | Correct fallback | Production risk |
|---|---|---|
| Residual-VQ kernel | `||x||² + ||e||² - 2x·e`, tiled at graph/host level | Workspace/bandwidth at maximum grids. |
| CUDA F16/BF16 ConvTranspose1d | Cast to F32, or matmul + `ggml_col2im_1d` | Copies, bandwidth, and latency. |
| Three-axis reset RoPE helper | Split head ranges and compose existing operations | Graph size/backend overhead and parity complexity. |
| Fused Mish | Compose exp/log/tanh/mul | Flow-decoder latency. |
| Binary output streaming | Buffer a bounded artifact first | Time-to-first-byte and memory. |

## Resource gates

Fixed calculations are trustworthy, forecasts are not:

* n-gram BF16 tables: exactly 58.500 GiB;
* full-context one-sequence F16 KV in the fork layout: exactly 7.4375 GiB;
* main payload: exactly 140.47 GiB;
* image sidecar: exactly 9.54 GiB file size;
* 96 GiB VRAM fit, 256 GiB host conversion fit, and all latency numbers: unmeasured.

A deployment claim requires peak allocator telemetry for text, maximum image input,
default image generation, maximum audio input, and long speech output; Q4/Q5 quality
ablations; 8k/32k/131k KV tests; one- and two-slot tests; cold sidecar load/eviction;
and end-to-end cancellation.

## Branching decision

Use four branches after the shared text spike:

1. image understanding (processor → visual encoder/bridge/RVQ → embeddings),
2. image generation (visual head/controller → decoder/refiner/VAE → artifact),
3. audio understanding (frontend → encoder/bridge/RVQ → embeddings),
4. speech generation (audio head/controller → codec/flow/HiFT → waveform).

This maximizes independent parity and allows either understanding path to ship
without accepting the generation scheduler, memory, or API risk.

