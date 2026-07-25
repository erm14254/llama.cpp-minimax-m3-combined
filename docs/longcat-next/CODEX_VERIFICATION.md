# Independent LongCat-Next feasibility verification

**Audit date:** 2026-07-25  
**Scope:** documentation and metadata inspection only; no inference code was changed.

## Verdict

Proceed, but only as staged, separately gated work. The text/trunk spike is well
founded; native MTP is impossible with the published weights. Image understanding,
image generation, audio understanding, and speech generation should be four
separate implementation branches after the text spike. Input and output paths have
different graphs, state machines, risks, and independently packaged weights.

The Work audit is unusually accurate. Independent checks confirmed its central
architecture, inventory, n-gram, reuse, MTMD, operator, API, and memory conclusions.
Two material corrections are required:

1. The image sidecar is no longer “header unavailable.” An HTTP range request
   retrieved its 226,408-byte safetensors header without downloading the 10.25 GB
   payload. It has **1,771 BF16 tensors / 5,124,042,701 parameters**: 558
   `image_decoder.*`, 828 `image_refiner.*`, and 385 `visual_model.*`. The last
   family is a duplicated visual-model family that the Work report did not list.
2. The 96 GiB VRAM plan is a capacity scenario, not a verified fit. Weight payload
   arithmetic and the 7.4375 GiB full-context KV calculation are correct, but the
   stated scratch, quantized-core, and latency allowances have not been measured.

## Source lock and independent method

The repository revisions independently resolved to the same commits used by the
Work audit: LongCat-Next `49dc7181`, LongCat-Next-inference `70ab100b`, official HF
model `0cf06318`, and upstream llama.cpp `555881eb`. The fork was inspected at its
current `work` branch; its LongCat baseline is the changes relative to
`555881eb`.

Evidence was deliberately limited to Git source, model indexes/configuration,
the HF API, a safetensors header range, and the 83 MB HiFT state dictionary. No
main checkpoint shard or image-sidecar payload was downloaded.

Core commands (run from the repository root):

```bash
cat docs/longcat-next/WORK_FEASIBILITY_AUDIT.md
git clone --depth 1 https://github.com/meituan-longcat/LongCat-Next.git /tmp/lc-next
git clone --depth 1 https://github.com/meituan-longcat/LongCat-Next-inference.git /tmp/lc-inf
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama-up
curl -L https://huggingface.co/meituan-longcat/LongCat-Next/resolve/main/model.safetensors.index.json -o /tmp/next-index.json
curl -L https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/resolve/main/model.safetensors.index.json -o /tmp/lite-index.json
curl -L --range 0-1048575 https://huggingface.co/meituan-longcat/LongCat-Next/resolve/main/image_decoder/image_decoder.safetensors -o /tmp/image-head.bin
curl -L https://huggingface.co/meituan-longcat/LongCat-Next/resolve/main/cosy24k_vocoder/hift.pt -o /tmp/hift.pt
```

## High-impact findings

### Checkpoints, vocabulary, and n-grams

* **Confirmed:** the Next index has 13,450 names and no `model.mtp.*`; the Lite
  index has 11,160 names and exactly 17 `model.mtp.*`. The HF repository sibling
  list exposes no MTP sidecar. This proves absence from published files, not that
  Meituan never trained an unpublished MTP.
* **Confirmed:** `config.json` independently contains `text_vocab_size=131072`,
  `text_vocab_plus_multimodal_special_token_size=131125`, and
  `vocab_size=282624`. The optimized loader slices the first 131,125 embedding
  rows while modal offsets address the full table.
* **Confirmed with precision:** `NgramCache`, `_shift_right_ignore_eos`,
  `EmbeddingWithMask`, and `NgramEmbedding.forward` implement three-token history,
  replacement of IDs `[131072,131125)` by zero, zero/EOS segment boundaries,
  twelve order-2/3/4 split tables, sizes `78*131072 + 2*i + 1`, masked hash zero,
  and `(base + 12 projections)/13` only at ordinary text positions. Modal/special
  positions retain the unscaled replacement embedding.

### Inventory and storage

* **Confirmed:** index-family counts are 11,143 trunk/text, 425 visual tokenizer,
  71 visual head, 1,740 audio tokenizer/decoder/flow, and 71 audio head.
* **Confirmed:** index metadata reports 150,825,367,872 payload bytes. HF metadata
  reports 74,257,230,752 parameters: 73,101,777,568 BF16 and 1,155,453,184 F32.
  These reconcile exactly: `2*BF16 + 4*F32 = 150,825,367,872`.
* **Confirmed with correction:** the image sidecar header is now inventoried in
  `TENSOR_INVENTORY.md`. All 1,771 tensors are BF16 and their offsets total
  10,248,085,402 payload bytes; the file is 10,248,311,818 bytes including its
  226,408-byte header and 8-byte length.
* **Confirmed with correction:** `hift.pt` is a plain 328-tensor F32 state dict,
  not a nested `generator` dictionary. It stores weight normalization as
  `weight_g`/`weight_v` for applicable convolutions, so conversion must fold or
  reproduce weight norm. Exact representative names and shapes are inventoried.

### Reuse and runtime interfaces

* **Confirmed:** trunk configuration and all 11,143 text-family names match Lite;
  conversion remapping, paired 28-block MLA/MoE graph, routing behavior, expert
  stacking, KV layout, and most tests are genuine reuse. RoPE, context, vocabulary
  handling, ignored IDs, modal state, and all modality graphs require adaptation.
* **Confirmed:** `llm_graph_input_ngram::set_input()` returns immediately when
  `ubatch->token` is null. MTMD's embedding batches use embedding data rather than
  token IDs. Therefore current embedding-only media injection cannot update Next's
  token-history contract; this is a correctness blocker for multimodal parity.
* **Confirmed with qualification:** current GGML operations can express a
  correctness prototype, including `ggml_col2im_1d` and `ggml_im2col_3d`. This is
  an expressibility conclusion, not proof that every proposed composition has
  backend coverage, acceptable memory, or parity.
* **Confirmed:** CUDA ConvTranspose1d currently dispatches F32/F32. A conversion or
  matmul+col2im fallback makes mixed precision a performance gap for a prototype.
  Residual-VQ can likewise use the standard squared-distance identity. Both become
  production risks because unfused intermediates can be large.
* **Confirmed:** Chat Completions parses `image_url` and `input_audio`; Responses
  converts `input_image` but rejects `input_audio`; transcription routes exist;
  model metadata hard-codes text-only output; no image-generation or speech route
  is registered.

## Memory finding

The Work audit's n-gram table calculation is exact: 31,406,985,216 BF16 parameters
are 58.500 GiB. Its MLA cache arithmetic is also exact for this fork's storage
contract: `28*(576 K + 512 V)*2 = 60,928 bytes/token`, hence 7.4375 GiB at 131,072
tokens for one sequence. Image sidecar disk/payload numbers are now exact. Audio
family byte totals are derivations from header metadata and are credible.

The conclusion “96 GiB VRAM / 256 GiB RAM is viable” remains **plausible but
unverified** because quantization quality, allocator fragmentation, simultaneous
graphs, backend work buffers, page-cache behavior, and latency were not measured.
Treat 32k context, one slot, lazy sidecars, and phase eviction as spike assumptions,
not product guarantees.

## True hard blockers

1. Native Next MTP: no published checkpoint weights.
2. Multimodal correctness: token identity and media embeddings cannot currently be
   carried together through the MTMD/ubatch contract.
3. Reference parity: no committed official golden fixtures for n-gram boundaries,
   modal RVQ IDs, state-machine transitions, or decoder intermediates.
4. Production image/speech: unimplemented multi-stage controllers and server output
   contracts; these are architecture work, not missing scalar GGML primitives.
5. Deployment claims: no measured peak/latency or quantization-quality results on
   the proposed 96/256 GiB target.

External decoder metadata is **not** still a blocker: it was obtained in this
verification. Payload hashes and numerical goldens remain future gates.

## Recommendation

Proceed with the text-only `FIRST_SPIKE.md`, then branch independently:

* `longcat-next-image-understanding`
* `longcat-next-image-generation`
* `longcat-next-audio-understanding`
* `longcat-next-speech-generation`

Do not merge those branches merely by modality: generation requires depth heads,
outer state, decoders, iterative schedulers, binary outputs, and much larger memory
surfaces absent from understanding. Do not advertise native MTP or a verified
96-GiB production profile.

