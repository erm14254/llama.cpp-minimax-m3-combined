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
- the GGUF chat template applies correctly with no control-token leakage.

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

138 GB BF16 against a 96 GB card, so throughput is paging-bound and varies
strongly with context: roughly 2.3 tok/s at short context up to ~26 tok/s at
long. **No single figure is intrinsic model speed.**

## Suggested next steps, in order

1. **Quantized GGUF** — the largest practical win; removes the paging bottleneck.
2. **`llama-server` smoke test.**
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
