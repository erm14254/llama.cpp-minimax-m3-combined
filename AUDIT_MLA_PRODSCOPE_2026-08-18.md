# Audit — Standard-Path MLA Mechanism Scope for the Reviewed Production-Arithmetic Plan (2026-08-18)

Source audit required by the recorded decision in `NEXT_ACTION.md` /
`WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md` (option (a)): determine
exactly which standard-path operations/layers share the three independently
measured MLA mechanisms **before** proposing any code change. Global il≥1
scope is established below **by source structure on both sides**, not by
assumption from the block-2 measurements alone. Read-only audit; no
arithmetic was changed by this commit.

Audited states: C++ `src/models/longcat-flash-ngram.cpp` at the delta-handoff
tree (1396 lines, SHA256
`cb6265974996afe1cd2f9f6e8439f75871bb6357f99772c0d75df8df4834f15a`, working
tree == HEAD `5e06842c3`); HF frozen runtime
`modeling_longcat_flash_sparse.py` (SHA `a3bc3161…`, verified) importing
norm/MLA primitives from installed transformers 5.15.0
`modeling_longcat_flash.py` (SHA
`bf7aa6387cf5bdf6c80b4a0f1b7bdd4878809fe33763323247c5fb73c4018659`, matching
`config.json` `transformers_version`). Line refs: `[T:n]` = transformers
file, `[S:n]` = sparse file, bare `n` = the C++ model file.

## The three measured mechanisms (evidence base, unchanged by this audit)

1. **Missing BF16 representation boundaries at MLA projection/scale
   outputs** — block-2 causal frontier: `q_a_proj-2` 786,432/786,432 and
   `kv_cmpr_pe-2` 294,912/294,912 (MLA walk), `q_b_proj-2`
   3,145,728/3,145,728 and `kv_cmpr_scaled-2` 262,144/262,144 (hex reset)
   all HF-equivalent at the BF16 output boundary from all-exact
   inputs/parameters; raw C++ F32 outputs off-lattice.
2. **HF LoRA-norm BF16 cast semantics** `bf16(bf16(x·rsqrt(var+eps))·w)` —
   quad reset: cast ordering dominant (~2.3e-3 under either eps); HF
   mechanism byte-closed for `q_a_norm` (D6), near-closure (7 one-ulp)
   for `kv_a_norm`.
3. **LoRA-norm eps mismatch** — C++ il≥1 runs `f_norm_rms_eps` = 1e-5; HF
   LoRA norms use the class default 1e-6 (source + runtime-module gate).

## C++ site map (trunk layer loop, line 594, il = 0..27; n_layer = 28)

| Mechanism site | il==0 (corrected, diagnostic stack) | il≥1 (uncorrected) |
|---|---|---|
| trunk attn_norm (OUT of scope) | 600–625 | 627–632 plain `build_norm` |
| q_a proj `wq_a` | 649–658 (input cast + BF16-operand GEMM + output round) | 704–707 plain F32 GEMM, no round |
| q_a_layernorm | 666–682 (`ggml_rms_norm(…, 1.0e-6f)` + cast chain) | 714–719 `build_norm` = 1e-5, F32-kept |
| q_b proj `wq_b` | 685–690 (+ output round) | 722–725 plain, no round |
| q scale `mla_scale_q` | 699–701 (+ post-scale roundtrip) | 729 plain scale |
| kv_a proj `wkv_a_mqa` (752, shared op) | 753–768 full-576 output roundtrip (766–767) | nothing |
| kv_a_layernorm | 900–926 (1e-6 + cast chain) | 928–933 `build_norm` = 1e-5, F32-kept |
| kv scale `mla_scale_kv` (941, shared) | 942–953 post-scale roundtrip | nothing |
| RoPE / attention core / `wo` | out of scope; no il-conditional arithmetic (979–981; verified into `llama-graph.cpp`) | same |

- `build_norm` has no per-call eps: always `hparams.f_norm_rms_eps`
  (`llama-graph.cpp:1790`), loaded once from GGUF
  `…attention.layer_norm_rms_epsilon` (model file line 79) = 1e-5. No
  second eps hparam exists for LoRA norms; the 1e-6 literal exists only at
  667 and 900 (il==0). Hence 54/56 trunk LoRA-norm instances + both MTP
  LoRA norms (1228, 1314) run 1e-5 vs HF 1e-6.
- Trunk `attn_norm`/`ffn_norm`/`output_norm` at 1e-5 **match** HF
  `config.rms_norm_eps` — no eps issue there (cast semantics for trunk
  norms remain a separately prohibited follow-up).
- **MTP is a separate un-corrected copy**: `graph_mtp` (1134–1396) shares
  no MLA builder with the trunk `graph` (482–1131); dispatch at 474–480;
  trunk loop bound excludes the MTP block (`model.layers[28]`). Trunk edits
  do NOT propagate to MTP. (User decision: MTP excluded this round.)
- cb-label continuity: block-2 walk names exist on the il≥1 path via the
  02d9687dc plumbing (712, 720, 726); `q_scaled` (730), `kv_a_norm` (937),
  `kv_cmpr_scaled` (955) are labeled at all il.

## HF structural-uniformity map

- RMSNorm `[T:47–62]`: `__init__(hidden_size, eps=1e-6)`; forward = fp32
  variance, `self.weight * hidden_states.to(input_dtype)` — exactly
  mechanism 2. `@use_kernel_forward_from_hub` inert (`kernels` not
  installed).
- eps split follows **tensor width, never layer index** (135 modules):
  `q_a_layernorm` `[T:359]` and `kv_a_layernorm` `[T:367]` take the class
  default 1e-6 at every layer (28 trunk + 1 MTP each); all
  hidden-size-3072 norms use `config.rms_norm_eps` = 1e-5
  (`config.json:55`, the only epsilon key — 1e-6 is invisible from config,
  which is why the GGUF carries only 1e-5).
- Layer uniformity: trunk layers constructed in one index-free
  comprehension `[S:1393–1398]`; `layer_idx` used only for KV-cache slot
  ids and fixed indexer ownership; **no arithmetic distinguishes any layer
  from layer 0/1** `[S:959–978]`. Dense (≤2048 KV, the frozen-512 regime)
  and sparse paths share identical eps/scale/dtype semantics (runtime
  branch `[S:877]`, not per-layer).
- BF16 landing points: every MLA `nn.Linear` output and both LoRA scale
  multiplies land bf16 (`q_states` chain `[T:415]`, `compressed_kv`
  `[T:419]`, `k_pass` `[T:421]`, scales `[T:424–426]`); scales are fp64
  Python floats on bf16 tensors (weak promotion keeps bf16);
  `mla_scale_q_lora = √2`, `mla_scale_kv_lora = √6` `[T:382–383]`;
  `k_rot` never scaled. C++ scales full q pre-split — elementwise
  equivalent (block-0 Q/R1 byte-exact gates).
- MTP `[S:1054–1065]` inherits `LongcatFlashMLA.__init__` → same LoRA
  norms (1e-6), same scales — structural sharing exists but the C++ twin
  is a separate copy (above) and is excluded this round.

## Scope conclusion (what the approved plan changes)

Trunk blocks il = 1..27 only; il==0 branches preserved literally; MTP
untouched; GEMM operands stay F32 (the exact measured configuration — the
BF16-weight GEMMs already convert F32 src1→BF16 internally per the closed
attention-core dispatch mechanism, so only the **output** boundary is
missing):

- **Stage A (five BF16 output-boundary sites):** A1 `wq_a` output round;
  A2 `wq_b` output round; A3 post-`mla_scale_q` roundtrip
  (**source-audit-derived + block-0-known-answer-supported**, not
  independently causal-frontier-measured at block 2); A4 full-576
  `kv_cmpr_pe` roundtrip (new else-branch); A5 post-`mla_scale_kv`
  roundtrip (new else-branch).
- **Stage B (two LoRA-norm sites):** replace `build_norm` at 714–719 and
  928–933 with `ggml_rms_norm(…, 1.0e-6f)` → BF16 → F32 → `ggml_mul(w)` →
  BF16 → F32, mirroring the byte-proven il==0 chains (667–682, 900–926).
  eps as a literal mirroring the il==0 sites and the HF class default
  (GGUF has no second eps key; no converter/GGUF change).

Out of scope (unchanged): trunk-norm cast semantics at il≥1, `wk_b`
absorption, concat, attention core, `wo` output boundary, residual adds,
MLP/MoE, output_norm, lm_head, production RoPE, MTP, LSA indexer.

## Graph/backend mechanics (verified this session)

1. All 28 trunk attention blocks execute on CUDA0 in the frozen-512
   placement (runtime cuBLAS probe: 28 occurrences per attention GEMM
   signature); CUDA `GGML_OP_CAST` F32↔BF16 proven in production by the
   il==0 chain.
2. Graph-node budget measured: `LLM_ARCH_LONGCAT_FLASH_SPARSE` is **not**
   in the enlarged `n_tokens*40` list (`llama-context.cpp:2306–2315`);
   capacity = `max(1024, 8·n_tensors)` = 8×539 = 4312 nodes; current graph
   2192 (`sched_reserve` in the inject4 log); projected post-change ≈2516.
   Ample headroom; no reservation edit required; overflow would assert
   loudly.
3. Placement sensitivity negligible (il==0 already carries the per-layer
   pattern; fit decisions are GB-granular); placement is a hard per-run
   gate regardless.
4. CUDA graphs: single-eval prompt runs, `graphs reused = 0` hard gate,
   baseline-identical env; every prior node addition of this class
   reproduced byte-identically (81/81, 69/69).
5. `ggml_rms_norm` eps is per-call; no global/il-dependent state.
6. Fusion: ggml-cuda fuses `RMS_NORM+MUL(+ROPE)` only on adjacent ops
   (`ggml-cuda.cu:2682+`, `:3057`); Stage B's casts break adjacency → the
   unfused sequence byte-proven at il==0. Stage A leaves norm code
   untouched.
7. Both roundtrip sites restore F32 before the `ggml_view_*` splits
   (738–748, 774–783) — no stride/type hazard.
8. Cast/rms_norm CUDA kernels are non-atomic; byte-reproducibility is
   established project-wide; build flags unchanged (182/182 CMakeCache
   parity).

## Validation references established for the plan

- Gate-3 4-token HF oracle located + hashed:
  `D:\llama.cpp-longcat-mtp\longcat_sparse_gate3_hf_v4_logits.bin` =
  `2c178ea5384d9b8ef59755658ecce2dfba33528edc7bf58964f23db81a26e050`
  (524,288 B); comparator is path-agnostic (`--hf-bin`/`--cpp-bin`).
- Quad-run committed dumps for Stage-A known-answer/target inputs
  (`cpp_resid_walk_inject3_b2_512/SHA256SUMS.txt`):
  `block2_q_a_norm_full.bin` = `2b600082…`, `block2_kv_a_norm_full.bin` =
  `93d7442a…`, `block2_q_b_proj_full.bin` =
  `23257a1e6891f5daaee74e9cdab16e314c37d86da3605c25c6f6018a03d0b60b`,
  `block2_kv_cmpr_scaled_full.bin` =
  `7450dd4a91683330d2c213370702223abd13e2758995d85fe8fe24dd215dc3a8`.
- Harness gate impact pre-registered: `$upstreamRegression` entry 15
  (`logical0_attn1_resid.bin` = `ffn_inp-1`, il=1) and its 18-name
  allowlist twin are expected-to-move; the 81/81 inject2_b1 reproduction
  gate re-scopes to the invariant subset; all landing gates survive.

The full reviewed plan (change set, five Stage-A local gates, endpoint
decision rules, abort rules) is recorded in the session plan file and will
be restated in the STATUS addenda as the stages execute.
