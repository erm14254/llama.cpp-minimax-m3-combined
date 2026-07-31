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
the real C++ capture/comparator with paths such as:

```bat
python tests\test-longcat-next-local-parity.py --model D:\LongCat-Next\longcat-next-bf16.gguf ^
  --reference-dir D:\LongCat-Next-reference\bf16-candidate-15b7fe8c --precision bf16 ^
  --output-dir D:\LongCat-Next-reference\cpp-bf16 --capture-exe build\bin\longcat-next-capture.exe ^
  --n-gpu-layers 0 --threads 0
```

After CPU parity succeeds, the accelerator form changes only placement controls:

```bat
python tests\test-longcat-next-local-parity.py --model D:\LongCat-Next\longcat-next-bf16.gguf --reference-dir D:\LongCat-Next-reference\bf16-candidate-15b7fe8c --precision bf16 --output-dir D:\LongCat-Next-reference\cpp-bf16-gpu --capture-exe build\bin\longcat-next-capture.exe --n-gpu-layers 99 --threads 8
```

The accepted reference contract is 433 arrays per precision, two byte-identical
official repeats, identical greedy continuations, and zero repeat differences. The
Stage-0 reference metadata retains its historical null tolerance fields; the checked
Stage-1 policy below now supplies the pre-execution cross-implementation thresholds.
Numerical parity cannot be claimed until locally converted BF16 and F16 GGUF results
are compared with the accepted fixtures.

The callback mapping is stable and explicit: `inp_embd` is the base embedding,
`ngram_proj-0..11` are the masked raw projections, `inp_embd_ngram` is the fused
pre-trunk embedding, `l_out-0/1/2/27` are physical blocks 0, 1, 2, and 27, and
`h_nextn` is the all-token final normalized hidden state. `result_norm` is not used
because it is output-row selected. Captures are lossless raw arrays
with a dtype/shape manifest. The machine-readable report records errors, the frozen per-array tolerances, normalized
violations, and pass/fail results.

The Python driver creates one case manifest directly from the accepted NPZ integer
arrays. One capture-process invocation loads the GGUF once, creates an isolated
context per case, and uses a runtime context sized to the longest fixture plus eight
generated tokens and a 16-token safety margin. Masked leading padding is assigned to
isolated auxiliary sequence IDs; attended tokens use sequence zero and the exact
reference positions. Embedding surfaces compare every position. Physical-block and
final hidden surfaces compare only positions selected by `attention_mask`.

`tests/fixtures/longcat-next/stage1-tolerances.json` freezes the BF16/F16 combined
absolute-relative criterion before local execution. A failure requires diagnosis;
the checked thresholds must not be widened after inspecting C++ results.

The documented command is CPU-first (`--n-gpu-layers 0 --threads 0`). A subsequent
accelerator run uses the same harness with, for example, `--n-gpu-layers 99`; a
positive `--threads` value sets both decode and batch thread counts.

Image/audio inputs and heads, MTMD, server integration, MTP, decoders, quantization
tuning, custom accelerator kernels, and performance claims remain out of scope.
