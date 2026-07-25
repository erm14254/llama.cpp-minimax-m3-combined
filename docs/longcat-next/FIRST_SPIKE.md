# First implementation spike (do not implement in this audit)

## Objective

Prove **text-only LongCat-Next BF16/F16 correctness with mandatory learned n-gram
embedding**, using the existing LongCat-Flash-Lite trunk. Do not include MTP, MTMD,
modal heads, quantization tuning, server output, or decoder sidecars.

This is the smallest meaningful spike because a loader-only experiment cannot test
the defining n-gram semantics, while a modality spike would depend on an unproven
trunk.

## Narrow deliverables

1. A metadata/header inventory harness that asserts 13,450 source names, expected
   families, zero MTP, exact extents, and total bytes.
2. A provisional `longcat-next` core converter that maps only the 11,143 trunk/text
   names, slices `embed_tokens[:131125]`, rejects/dockets modal families, and never
   emits `nextn_predict_layers`.
3. Loader metadata for 131,072 hash/tokenizer extent, 131,125 embedding/head extent,
   282,624 source joint extent, plain 10M RoPE, and 131,072 context.
4. Adapted Flash trunk graph with exact ignored-ID/zero/EOS/masked-zero/`/13`
   learned n-gram behavior.
5. Official PyTorch golden fixture generator and small checked fixtures; fixtures
   need hashes and embeddings for short sequences, not the whole checkpoint.
6. CPU parity tests first; one available accelerator backend only after CPU passes.

## Required cases

* BOS and left zero padding.
* EOS at every position in a four-token history window.
* literal token zero.
* ordinary maximum BPE ID 131071.
* every ignored ID 131072…131124, especially image/audio pad and transition IDs.
* multi-token prompt versus token-at-a-time decode.
* two independent sequence IDs in one ubatch.
* sequence copy, removal, reset, KV shift, and speculative reject/rollback.
* output shape exactly 131,125 logits; source embedding extent recorded as 282,624.

## Exit criteria

* Every main-index name is classified mapped, intentionally sliced, deferred modal,
  or explicitly dropped training state; counts reconcile to 13,450.
* Hash IDs exactly match official Python for all cases.
* Pre-trunk embeddings and selected hidden/logit tensors match within declared
  BF16/F16 tolerance.
* Greedy text continuation matches on a fixed prompt corpus.
* Flash-Lite text, router, n-gram, and MTP regressions remain green.
* Peak memory is recorded at 8k and 32k; no 96-GiB claim is made from arithmetic.

## Stop conditions

Stop before modal work if any of these persists: source tensor accounting mismatch,
three extents cannot coexist without unsafe global `n_vocab` assumptions, n-gram
history diverges after EOS/rollback, or trunk logits fail reference parity.

## Next work after success

First design and test the token-aware MTMD embedding override contract in isolation.
Then start image understanding and audio understanding as separate branches. Start
visual/audio generation branches only after their respective raw encoder/codebook
fixtures exist; do not let image refinement or HiFT block understanding support.

