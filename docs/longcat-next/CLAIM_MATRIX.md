# LongCat-Next claim matrix

Status vocabulary is exactly: **Confirmed**, **Confirmed with correction**,
**Plausible but unverified**, **Incorrect**, or **Blocked by unavailable evidence**.

| ID | Work claim | Status | Independent evidence / correction |
|---|---|---|---|
| C01 | Next has zero `model.mtp.*`; Lite has 17 | Confirmed | Counts from both official `model.safetensors.index.json` files; HF Next sibling list has no MTP sidecar. |
| C02 | Native Next MTP is unavailable | Confirmed | A trained auxiliary graph cannot be reconstructed from absent weights; Lite weights are checkpoint-specific. |
| C03 | Vocabulary extents are 131072/131125/282624 | Confirmed | Exact `config.json` fields and optimized `modules/nmm_flash.py` slicing. |
| C04 | There are twelve learned n-gram tables | Confirmed | Orders 2–4 times four splits in `modeling_longcat_ngram.py`; 12 embedder and 12 projection names in index. |
| C05 | IDs 131072…131124 are ignored | Confirmed | `NgramCache._ignore_token_ids` maps the configured half-open interval to zero. |
| C06 | Zero and EOS delimit history | Confirmed | `_shift_right_ignore_eos` masks across both zero and EOS boundaries. |
| C07 | Table `i` has `78*131072+2*i+1` rows | Confirmed | Source constructor expression and indexed tensor family agree. |
| C08 | Hash zero contributes a zero vector | Confirmed | `EmbeddingWithMask` masks the lookup output; it does not rely on learned row zero. |
| C09 | Ordinary positions divide by 13; ignored positions do not | Confirmed | Forward combines base plus 12 projections and conditionally divides only the non-ignored result. |
| C10 | Main index has 13,450 names | Confirmed | Direct JSON `len(weight_map)`. |
| C11 | Family counts are 11143/425/71/1740/71 | Confirmed | Prefix classifier over all index keys; sums to 13,450. |
| C12 | Main payload is 150,825,367,872 bytes | Confirmed | Index metadata; independently reconciles from HF dtype parameter counts. |
| C13 | Parameter count is 74,257,230,752 | Confirmed | Official HF API; dtype subtotals sum exactly. |
| C14 | Every Next text tensor name exists in Lite | Confirmed | Set comparison after excluding Next modal families and Lite MTP. |
| C15 | Flash trunk conversion/graph is reusable | Confirmed with correction | Structural/tensor equality supports reuse, but it is source reuse—not drop-in reuse—because extents, RoPE, history semantics, and lifecycle differ. |
| C16 | CUDA duplicate expert-ID fixes are reusable | Plausible but unverified | Fork changes/tests exist; no CUDA hardware/backend matrix was run in this audit. Rebase necessity must be rechecked. |
| C17 | Existing n-gram tests are reusable | Confirmed with correction | Algorithmic tests are useful; Next needs new ignored-ID, EOS/zero, `/13`, and embedding-batch cases. |
| C18 | Current MTMD embedding batches lose token identity | Confirmed | MTMD builds embedding batches; fork n-gram input returns on `!ubatch->token`. |
| C19 | Existing GGML ops suffice for prototypes | Confirmed with correction | Ops express the math; backend support, peak memory, exact padding/RoPE compositions, and parity remain unproven. |
| C20 | Residual-VQ fused search is only a performance gap | Confirmed with correction | Squared-distance matmul fallback is correct; its workspace may itself become a practical resource blocker at maximum grids. |
| C21 | Mixed-precision CUDA ConvTranspose1d is only a performance gap | Confirmed | F32 conversion or matmul+`ggml_col2im_1d` is a correctness fallback; CUDA source exposes F32/F32 only. |
| C22 | Chat Completions supports typed image/audio input | Confirmed | `server-common.cpp` handles `image_url` and `input_audio`. |
| C23 | Responses supports image but not audio input | Confirmed | `server-chat.cpp` accepts only input text/image/file in that conversion path. |
| C24 | Server output is text-only | Confirmed | `output_modalities` is `['text']`; no images/speech routes are registered. |
| C25 | `/audio/transcriptions` is directly reusable | Confirmed with correction | Transport/schema are reusable; dispatch must still support the LongCat model path and its MTMD encoder. |
| C26 | N-gram tables occupy 58.500 GiB BF16 | Confirmed | Exact product and binary-unit conversion. |
| C27 | Full 131k F16 KV is 7.4375 GiB/sequence | Confirmed | Fork caches 576 K plus 512 V elements for each of 28 blocks. |
| C28 | Image sidecar inventory was unavailable | Incorrect | Range-fetched header gives all 1,771 names/shapes/dtypes; see inventory. |
| C29 | Image sidecar contains only listed decoder/refiner families | Confirmed with correction | It additionally contains 385 `visual_model.*` BF16 tensors (631,975,680 parameters). |
| C30 | Image sidecar file size is 10,248,311,818 bytes | Confirmed | HF metadata/file response; header offsets reconcile payload plus header framing. |
| C31 | HiFT exact inventory was unavailable | Incorrect | The 83 MB state dict was downloaded and metadata-parsed: 328 F32 tensors. |
| C32 | HiFT requires weight-normalization handling | Confirmed | Stored convolution names include paired `.weight_g` and `.weight_v`. |
| C33 | HiFT is stored under a `generator` key | Incorrect | Root object is the state dictionary itself. Loader compatibility is unaffected. |
| C34 | Visual input is a feasible prototype | Plausible but unverified | Architecture is expressible; no processor/RVQ golden was run and maximum-grid workspace is unknown. |
| C35 | Deterministic image decoder is feasible | Plausible but unverified | Exact weights/topology now available; no GGML graph or parity fixture exists. |
| C36 | Refiner is latency rather than correctness blocked | Plausible but unverified | Operations exist, but 3-axis RoPE, scheduler/RNG, VAE upcast, memory, and 84-pass latency need measurement. |
| C37 | Audio understanding is feasible | Plausible but unverified | Source topology maps to existing ops; frontend and VQ parity have not been demonstrated. |
| C38 | Speech generation is feasible | Plausible but unverified | Exact HiFT metadata removes an evidence gap, but codec/flow/vocoder parity and latency remain open. |
| C39 | Arbitrary music/SFX should not be promised | Confirmed | Official claims/examples center on speech conversation and voice cloning. |
| C40 | 96 GiB VRAM is viable | Plausible but unverified | Fixed arithmetic is sound; quantization, scratch, fragmentation, concurrency, and latency are estimates. |
| C41 | 256 GiB RAM is viable | Plausible but unverified | Mmap/streaming plan is credible but no conversion or pressure test was run. |
| C42 | Four modality capabilities should be separate branches | Confirmed | They have separable dependencies and exit gates; generation should not block understanding. |

No important reviewed claim was classified **Blocked by unavailable evidence** after
the sidecar metadata fetch. Numerical parity and performance claims are instead
**Plausible but unverified**, because the evidence needed is future execution rather
than inaccessible published metadata.

