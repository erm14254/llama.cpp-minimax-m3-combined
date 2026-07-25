# LongCat-Next reuse map

## Reuse with no algorithmic change

| Area | Fork anchor | Reuse decision |
|---|---|---|
| 14 logical → 28 physical blocks | `conversion/longcat_flash_ngram.py`, `_remap_double_block` | Reuse mapping and accounting. |
| Expert stacking | same converter, expert remap/stack path | Reuse streaming design; source names match. |
| MLA trunk | `src/models/longcat-flash-ngram.cpp`, `graph::graph` | Reuse absorbed MLA math, KV-B split/layout, scaling, and compressed cache. |
| Paired residual schedule | same graph | Reuse delayed MoE shortcut and even/odd block structure. |
| Routing | `llm_graph_build_longcat_moe_route` in `src/llama-graph.cpp` | Reuse correction-bias selection, unbiased weights, identity aggregation. |
| CUDA duplicate IDs | fork changes in CUDA `mul_mat_id` paths | Carry forward only after upstream rebase and backend tests. |
| GGUF mechanics | split files, mmap, quantization, scheduler/offload | Reuse generic infrastructure. |
| Tests | `tests/test-longcat-router.cpp`, `tests/test-longcat-ngram.cpp` | Retain as Flash regression base. |

The structural basis is strong: hidden/rank/head/FFN/expert topology is equal and
all 11,143 Next trunk/text tensor names occur in Lite. “Reuse” does not mean binary
compatibility with a Lite GGUF.

## Reuse after mandatory adaptation

| Area | Required Next change |
|---|---|
| Architecture/schema | New `longcat-next`; three explicit extents; no MTP key. |
| Embeddings/output | Slice text input to 131,125, retain/extract modal rows from 282,624, output 131,125. |
| N-grams | Hash base 131,072; ignored `[131072,131125)` IDs; zero/EOS boundaries; masked hash zero; conditional `/13`. |
| RoPE/context | Plain base 10,000,000 and context 131,072; do not inherit Lite YaRN/base 5M. |
| Tokenizer | Preserve 53 added control IDs and official chat template. |
| Lifecycle | Store n-gram/modal state per sequence through copy/remove/shift/reset/speculation. |
| Tests | Add Next goldens for ignored IDs, boundaries, multimodal transitions, and three extents. |
| MTMD | Carry original placeholder token IDs alongside externally supplied media embeddings. |

## Building blocks only, not reusable implementations

* MTMD byte ingestion, image/audio decoding, and marker parsing.
* Qwen vision window/patch graph patterns; LongCat requires bicubic preprocessing,
  its own ordering, bridge, and residual VQ.
* Whisper audio graph, FFT, Slaney filter, and resampling helpers; exact padding,
  centered STFT, drop-last, chunking, and normalization differ.
* GGML Conv1d/2d, im2col/col2im, attention, norms, FFT/ISTFT, and host sampling.

## Entirely new

* Visual and audio encoders/bridges/RVQ converters and graphs.
* Eight-level visual/audio depth-head execution and sampling.
* Per-sequence visual CFG and audio guidance/code state machines.
* Coarse image decoder, diffusion refiner, VAE, and scheduler.
* Codec decoder, flow prenet/estimator, HiFT, and segment combiner.
* Lazy sidecar dependency manifests and hash validation.
* Image/audio output API objects, routes, artifacts, limits, and cancellation.

## Explicit non-reuse

The 17 Lite `model.mtp.*` weights must never be copied into Next. Their existence
and compatible-looking dimensions do not make them trained for Next's checkpoint,
vocabulary, RoPE, or hidden distribution. Generic prompt n-gram speculation is a
separate weight-free feature and must be disabled during modal state.

