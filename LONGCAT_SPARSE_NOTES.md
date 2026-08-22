# LongCat Flash Lite Sparse — implementation handoff

Status notes for a fresh instance. Written 2026-08-22 at HEAD `8160ab4a2`,
production arithmetic restored (`-use_fast_math` on, `CUBLAS_TF32_TENSOR_OP_MATH`
on), tracked tree clean.

## Status

The sparse (LSA) path is verified working end to end:

- model loads and generates coherently below and above the 2048 boundary;
- the sparse path engages exactly when `n_kv > index_topk` (2048) — `n_kv` pads
  to a multiple of 256, so N=2048 runs dense (`n_kv` 2048) and N=2050 runs
  sparse (`n_kv` 2304);
- the GGUF chat template applies correctly with no control-token leakage;
- generation verified on real prose prompts, including a 2685-token prompt
  producing 300 tokens of coherent original output, served via `llama-server`
  across three sequential requests in one process;
- the sparse path is confirmed active on that prompt at `n_kv` 2560/2816/3072
  (`LONGCAT_LSA_AUDIT`). Those audit lines come from the `-v` `llama-cli` runs
  in `gen_out/`; the `llama-server` throughput runs were made without `-v` and
  carry no audit lines.

## HF logit parity

Frozen criterion `err > 0.5 + 0.05*|hf|`; cosine ≈ 0.9997; top-1 correct (483)
at every length. Violations out of 131072:

| N | production (TF32 on, fast-math on) | TF32 off + fast-math off |
|---|---|---|
| 512  | 0   | 32 |
| 1024 | 1   | 2  |
| 1536 | 7   | 0  |
| 2048 | 298 | 0  |
| 2050 | 126 | 43 |

Dense parity is achievable. **The both-off column is diagnostic only — not a
proposed production configuration.** The two knobs interact strongly rather than
adding: fast-math-off alone is much worse at short context (512: 0→495), and
most of that regression disappears once TF32 is also off.

## Residual 2050 gap

Attributed to near-tie indexer scores at the top-k cutoff. Membership disagrees
with HF in **17** index slots under production arithmetic and **19** under IEEE,
with the identity of the disagreeing owners shifting rather than reducing.
Contested members cluster in a narrow index band. Likely irreducible between the
two implementations.

Note the sparse-vs-dense difference on our side is only that two additional
valid cells are masked; llama.cpp computes attention over the full padded
`n_kv` (2304) with `-inf` masking, whereas HF gathers a compact 2048 and expands
via `kv_b_proj`. That factorization difference is present identically at 2048
and 2050, so it does not explain a sparse-specific residual.

## Greedy reproducibility caveat

Enabling `--spec-type draft-mtp` forks greedy generation at the same character
offset (327) as vanilla **even with zero drafts delivered**
(`--spec-draft-n-max 1 --spec-draft-n-min 2`).

Target `cparams`, KV geometry, layer placement (30/30 on CUDA0) and target graph
node count (3290) are all identical. What differs is `ggml_backend_sched` split
counts: `66/27/48/45/50/52` → `41/2/29/32/34/31`. That is the **leading candidate
mechanism, not proven** — `draft_mtp::process()` (`common/speculative.cpp`
1427–1544) also runs in that configuration and reads the target's `h_nextn`, and
it was not isolated.

`need_n_rs_seq()` is inert here: LongCat is absent from
`llm_arch_supports_rs_rollback`, so `n_rs_seq` is clamped to 0.

## MTP

Executes. Enabled with `--spec-type draft-mtp` (CLI only — `--mtp` is a
download helper). No separate draft model: the MTP context is built on the
target model.

| n_max | acceptance | tok/s |
|---|---|---|
| 1 | 0.463 | 16.2 |
| 3 | 0.232 | 12.4 |
| vanilla | — | 17.5 |

Reuses the dense `graph_mtp` unmodified — nothing in it is conditioned on the
sparse arch, LSA, or the indexer, and `mtp.dsa_cli` metadata is not consulted.
Unlike other MTP models this arch never checks `ml.load_mtp`, so block 28 always
loads.

## Performance

Measured in a single `llama-server` process, model loaded once,
`cache_prompt:false` on every request. Generation throughput is flat regardless
of context: **17.18 tok/s** (53-token prompt), **17.14 tok/s** (2694-token
prompt), **18.36 tok/s** (53-token prompt repeated). The earlier 2.3–26 tok/s
spread across separate processes was a cold-start artefact, not a
context-length or paging effect — host working set grew +25.7 GiB, then
+8.0 GiB, then +14 MiB across the three runs as mmap'd weights faulted in, and
prompt processing on an identical prompt went 9.91 → 34.10 tok/s over the same
warming.

Hardware: RTX PRO 6000 Blackwell Workstation Edition, **97,887 MiB** dedicated
VRAM (registry and `nvidia-smi`; llama.cpp reported 95,346 MiB *free* at load),
~128 GiB WDDM shared, 256 GB host RAM.

The BF16 model does **not** fit in VRAM and is not fully resident. `-fitt 4096`
places 29/30 layers on GPU with **15 of 29 overflowing**
(`overflow_type=ATTN` — only the attention part on device), giving
`CUDA0 model buffer = 88,936 MiB` and `CPU_Mapped model buffer = 131,069 MiB`.
llama.cpp's own fit summary is `90,887 MiB used, 4,458 MiB free`. Measured
device peak during generation is 92,756 MiB, flat to within 57 MiB across all
three runs. "No paging" holds only in the *disk* sense once the host pages are
warm — the CPU-mapped half is still read over PCIe every token, and that is
what bounds the ~17 tok/s.

**Headroom is the constraint.** ~4.4 GiB spare at BF16 by llama.cpp's fit
accounting. KV cache is **35.0 KiB/token** measured: `141.75 MiB` MLA K-only
over 28 trunk layers plus `15.75 MiB` indexer LID over 14 owner layers, at 4608
cells, both bf16 (the loader promotes F16→BF16 for absorbed MLA). That is
~0.27 GiB at 8K but ~4.4 GiB at 128K, so long context does not currently fit.

## Suggested next steps, in order

1. **Quantized GGUF — for VRAM headroom.** Measured `llama-quantize --dry-run`
   sizes: Q8_0 68.42 GiB, Q6_K 52.85, Q5_K_M 44.97, Q4_K_M 37.57. Two
   LongCat-specific notes. `blk.*.attn_k_b.weight` has `ne[0]=128`, so no
   K-quant is legal — the automatic fallback picks Q5_1/Q5_0 below Q6_K, so
   override it to `q8_0`. And llama.cpp's indexer quantization exclusion
   matches MiniMax tensor names only, so LongCat's `indexer.proj` (F32 in
   source), `indexer.attn_k` and `indexer.attn_q_b` are unprotected —
   protecting all three costs 19.8 MiB and matters because the indexer
   determines top-k membership. Recommended: Q8_0 base with `ngram_embd=q6_K`
   plus indexer protection → 61.34 GiB. Note `ngram_embd.*` alone is 45% of the
   model at BF16 (58.5 GiB) and is a pure lookup table, hence the most
   quantization-tolerant part. Expect a throughput gain as well as headroom:
   at Q8_0 the whole model fits in VRAM, removing the 131,069 MiB CPU-mapped
   spill that currently bounds generation — **untested**.
2. ~~`llama-server` smoke test~~ — **done**; see Performance above.
3. Optional: isolate the MTP divergence by skipping `draft_mtp::process()` in the
   zero-draft configuration.

Do not reopen the frozen R1 top-k apparatus unless explicitly requested.

## Reference

- GGUF: `scratchpad/gate4-gguf/LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved-BF16.gguf`
  (a second sharded artifact, `…-283-Low-KL-BF16-00001-of-00008.gguf`, is
  metadata- and tensor-identical and produces bit-identical logits).
- HF oracles: `hf_logits_2050_v1/`, `hf_logits_N_v1/` (512/1024/1536/2048).
- Comparator: `D:\llama.cpp-longcat-mtp\compare_longcat_sparse_gate3_logits.py`.
- Build tree: `D:\llama.cpp-longcat-claude-build-cuda132`.
- Runs must pin cuBLAS to CUDA v13.2 on the child PATH; an unpinned run silently
  loads 13.0 and changes results.
- Plain (non-thinking) chat mode needs
  `--chat-template-kwargs "{\"enable_thinking\": null}"`; the default resolves to
  `true` and injects `/think_on` + an open `<longcat_think>`.
- Non-interactive completion is `llama-completion`; `llama-cli` is the chat
  client and is the only one accepting `--spec-type`.
