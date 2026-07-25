# LongCat-Next Stage 1 C++ core spike

This provisional architecture implements only the LongCat-Next text core. It has a
distinct `longcat-next` GGUF architecture, no MTP metadata or graph, and defers all
2,307 visual/audio tensors. Conversion requires the exact 13,450-name source
inventory, maps 11,143 core names, slices `model.embed_tokens.weight` to 131,125
rows, retains the 131,125-row language head, and rejects any `model.mtp.*` name.

The GGUF records all three vocabulary extents: hash/text base 131,072, core
input/output 131,125, and source joint embedding 282,624. It also records ignored
start/count 131,072/53. The loader requires these exact values and all twelve learned
n-gram tables and projections.

The CPU graph reuses the reviewed LongCat-Flash 28-physical-block MLA/MoE trunk.
Hashing uses base 131,072; ignored control IDs hash as zero and preserve their
unscaled base embedding, while ordinary positions retain conditional division by 13.
The LongCat memory wrapper keeps position-aware history synchronized with clear,
remove, copy, keep, positive/negative shifts, division, save/read, and graph-level
speculative rollback.

## Local-only conversion and parity

The checkpoint and accepted references are not committed. Convert locally, then run
the inventory gate with paths such as:

```bat
python tests\test-longcat-next-local-parity.py ^
  --bf16-reference D:\LongCat-Next-reference\bf16-candidate-15b7fe8c ^
  --f16-reference D:\LongCat-Next-reference\f16-candidate-15b7fe8c
```

The accepted reference contract is 433 arrays per precision, two byte-identical
official repeats, identical greedy continuations, zero repeat differences, and null
cross-implementation tolerances. This repository does not choose or widen those
tolerances. Numerical parity cannot be claimed until locally converted BF16 and F16
GGUF results are compared with the accepted fixtures after tolerance review.

Image/audio inputs and heads, MTMD, server integration, MTP, decoders, quantization
tuning, custom accelerator kernels, and performance claims remain out of scope.
