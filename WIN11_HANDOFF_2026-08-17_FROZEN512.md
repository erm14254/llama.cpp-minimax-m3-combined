# LongCat Sparse Parity — Frozen-512 Checkpoint Handoff (2026-08-17)

Self-sufficient handoff for a fresh session. Read this file, `NEXT_ACTION.md`,
`STATUS_2026-08-17.md` (full evidence trail), and `CLAUDE.md` (base guardrails,
partially superseded by this file where noted). No conversation history is
required.

## Git state

- **Branch:** `claude/longcat-win11` (base `35ad2e63a`, migration checkpoint).
- **Measurement binary/source HEAD: `b009d6f68`** — the source state of the
  binaries used for the frozen-512 measurement. Their exact SHA256
  (re-hashed from disk at handoff time, unchanged since the run):
  `llama-debug.exe` `df2a57f6f99428d0735ceea88af2fdd8d8c59f7453b0b994d869020f007eddb0`,
  `llama.dll` `93466c40380729857eb43f7d4ccfa4cf7f336d634cec0b44bb359d2411465dc3`,
  `ggml-cuda.dll` `502e50e8855d5fc4f23758afa9c4ba277be3339b4159527ff1ae41268f7c1d48`
  (build dir `D:\llama.cpp-longcat-claude-build-cuda132`).
- **Measurement-result / pre-handoff base HEAD: `6bf5ee61f`.** Tracked tree
  clean at that commit (untracked working artifacts only: capture dirs' bins,
  logs, `prompt_512_a.txt`, probe dirs — all intentional, gitignored classes).
- **Relevant scientific commit sequence, derived from Git
  (`git log 35ad2e63a..6bf5ee61f`): actual count = 17** —
  `eb46d2bbd` authoritative Blackwell block-0 MLA capture · `517445bce`
  runbook correction · `b3ac16665` cuBLAS contract docs · `3d16341dc` Exp A
  arithmetic · `668f400cc` Exp A gates pass · `5b408206c` Exp B arithmetic ·
  `b49f817ca` Exp B gates pass · `5dda40c50` attention-path dump plumbing ·
  `0ebc5bc0a` localization: first divergence Q/K RoPE · `9a7de8a76` R0
  instrumentation · `2137155a5` R0 control result · `1cba1760b` R1 rotation ·
  `e1a9777ae` R1 gates pass · `2a9ded17c` H3/H4 excluded · `6f5fa1305`
  attention-core mechanism closed · `b009d6f68` runtime dispatch confirmed ·
  `6bf5ee61f` frozen-512 result.
- **This documentation handoff commit** is the commit containing this file,
  recoverable with:
  `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-17_FROZEN512.md`.
  It is docs-only and must never be associated with the measurement binaries.

## Frozen 512-token end-to-end criterion — current standing (commit `6bf5ee61f`)

**Criterion (frozen, never widen):** for **all 131,072** final-position
logits, `abs(cpp − hf) ≤ 0.5 + 0.05·abs(hf)` (ATOL 0.5 / RTOL 0.05) **and**
top-1 equality. Comparator (verbatim, reuse as-is):
`D:\llama.cpp-longcat-mtp\compare_longcat_sparse_gate3_logits.py`
(SHA `6976fbc035c60692406a02cc1a6706b2702bbb16f579829e44c29dcdcc57bc93`).

**Frozen inputs:** HF logits oracle
`D:\llama.cpp-longcat-pre-gate4\hf_sparse_512_v4.bin` = **`8825d92d7d9cdea42a4ea3aa2e3df5766bdf880323b1f48ea8c17ff63f3c5ecf`**
(524,288 B, final-position row, vocab 131,072); prompt `prompt_512_a.txt`
(`d3c44b15…`, 512 × token 483); token stream **`4893d787…`**. Run parameters:
`-c 4608 -b 4608 -ub 512 -fa off -ctk f32 -ctv f32 --no-warmup -fitt 4096`,
placement `(29, 15, ATTN)` / offloaded 29/30 / CUDA0 88936.14 MiB,
`--save-logits` (eval callback off; no `LONGCAT_*` env vars).

**Historical baseline — corrected identity (provenance trap, re-verified from
disk at handoff time):**
- **Correct F32-KV baseline:**
  `D:\llama.cpp-longcat-pre-gate4\sparse_512_fa_off_f32\llamacpp-…-00008.bin`
  = `1a8e37e294890dd4c873b28f9c42f521c1b25226fd248175f6274cd6c7ed9a3e` —
  reproduces the memorandum's numbers digit-for-digit: **2,122 violations**,
  max_abs **1.84657693**, mean 0.241643069, RMSE 0.306931687, cosine
  0.999293288, top1 483, top20 20/20.
- **Similarly named WRONG artifact (BF16-cache variant, do not use as the
  memo baseline):**
  `D:\llama.cpp-longcat-pre-gate4\sparse_512_fa_off\llamacpp-…-00008.bin`
  = `f39f77b6bd01cfe608a44b2282f0f65972c6108b975823bc1b42b3720850f060`
  (scores 12,609 violations).

**Current result (`e14d95bfaaa0fea2977ed4ac852b7a631427e27f79eeb17c04a5a70c824660df`):
FAIL under the frozen criterion — the criterion must not be widened.**

| Frozen metric | Baseline (`1a8e37e2…`) | Current (`e14d95bf…`) |
|---|---|---|
| violations | 2,122 / 131,072 | **40 / 131,072** |
| worst_tolerance_ratio | 2.78345282 | **1.33129834** (@ token 100151) |
| max_abs | 1.84657693 | **0.851543427** (@ 73210) |
| mean / RMSE | 0.241643069 / 0.306931687 | **0.130625596 / 0.164743901** |
| cosine | 0.999293288 | **0.999799076** |
| top-1 / top-20 / top-100 | 483 ✓ · 20/20 · 92/100 | 483 ✓ · 20/20 · **96/100** |

**Attribution:** the 2,122 → 40 delta belongs to the **aggregate current
block-0 corrective stack** (N-gram BF16/restore-F32, block-0 RMSNorm HF
semantics, Q-side BF16 chain, eps 1e-6, Experiment A D1+D2, Experiment B D3)
relative to the clean pre-diagnostic baseline — **not to A+B alone**.
A+B-specific causal results stand only on the dedicated block-0 captures
(`cpp_attn0_mla_expA_512/`, `_expB_512/`).

## Scientific state — what is explained vs open

**Block-0 attention is diagnosed to the current measurement resolution.**
Confirmed (all byte-exact-gated or collapse-proven; see STATUS addenda):
1. Upstream anchors byte-exact: `inp_embd_ngram` `d0e9edc8…`, block-0
   `attn_norm` `a1c4c20c…`.
2. Q trio byte-exact to HF oracles: `ddf69fe4…`/`956bd3e8…`/`4f3b647b…`.
3. KV path byte-exact after A+B: `kv_a_proj_with_mqa` = `513390…`,
   `kv_a_layernorm` = `b44cc101…`, `kv_cmpr_scaled` = `909b7ee7…` (S2b).
   Mechanisms: full-576 BF16 output boundary; HF RMSNorm cast semantics
   `bf16(bf16(x·rsqrt(var+1e-6))·w)` (eps from source); post-scale bf16.
4. RoPE: first genuine attention-path divergence (H1). HF uses BF16 cos/sin +
   BF16 elementwise (unique exact ordering C2). Canonical targets
   `c8b9b6bf…` (q) / `3ed6f4e7…` (k), generator
   `make_longcat_rope_targets.py`. **R1 proved real ggml BF16 graph
   arithmetic reproduces HF rotary byte-exactly given exact angles**
   (env-gated `LONGCAT_ROPE_ORACLE_DIR`). **R0 proved exact HF rope moves
   downstream only marginally** (S3 0.003869→0.003846; o_proj
   0.004759→0.004738). **Production `ggml_rope_ext` angle generation remains
   a known exact-parity gap** (ggml F32 trig provably rounds to different
   BF16 in 3,377/16,384 sin values) — with a measured small block-0
   downstream contribution.
5. Attention core: H3 (softmax cast) and H4 (ordering) **excluded** by
   orthogonal pairs (V3↔V5 delta rel 1.09e-6). The C++↔HF difference is
   **mechanistically explained to reduction noise** by the corrected V7″
   model (0.98× the measured reference reduction discrepancy): unrounded F32
   GEMM outputs (`prefer_f32_output`, ggml-cuda.cu:1510-1519), TF32
   quantization of the two off-lattice operands (q_abs, probs) under the
   handle-global `CUBLAS_TF32_TENSOR_OP_MATH`, and the lossy latent-context
   src1→BF16 conversion into `wv_b`. Runtime cuBLAS API log directly
   confirmed dispatch/types (`runtime_dispatch_confirmation.json`).
   **Do not reopen the attention-core mechanism without contradictory
   evidence.**
6. wo boundary semantics analytically byte-exact (plain bf16 linear on the
   oracle context); FA NaN root cause (half overflow pre-scale) known from
   the memorandum, production fix not designed.

## Runtime & environment contract (unchanged, mandatory)

- **cuBLAS child-runtime contract:** authoritative C++ parity runs must
  resolve `cublas64_13.dll` to **CUDA v13.2 / cuBLAS 6.14.11.1330** via a
  child-process v13.2-first PATH pin (never machine-wide), verified against
  the **live process module list** (path + version). Wrong-runtime signature:
  anchors pass, residual `49d729e1…`.
- **`.venv`** at `D:\llama.cpp-longcat-claude\.venv` is validated (Py 3.12.10,
  torch 2.13.0+cu132, transformers 5.15.0) — **do not mutate** (no
  pip install/upgrade) without reported cause + approval. Native toolchain
  (CUDA 13.2.0, MSVC 14.44.35207, driver) must not be upgraded.
- **Diagnostic env overrides stay unset** unless an experiment explicitly
  requires them: `LONGCAT_HIDDEN_DUMP_DIR`, `LONGCAT_ROPE_INJECT_DIR` (R0),
  `LONGCAT_ROPE_ORACLE_DIR` (R1); all twelve `GGML_CUDA_*` getenv overrides
  incl. `GGML_CUDA_CUBLAS_COMPUTE_TYPE`; `CUBLAS_LOGINFO_DBG`,
  `CUBLAS_LOGDEST_DBG`, `CUBLASLT_LOG_LEVEL`, `CUBLASLT_LOG_FILE`. Sweep
  fail-closed before production-style runs.
- PyTorch analysis scripts: `torch.backends.cuda.matmul.fp32_precision =
  "ieee"` (2.13 API only), fail-closed on
  `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`.

## Paths & roles

Writable: `D:\llama.cpp-longcat-claude` (repo),
`D:\llama.cpp-longcat-claude-build-cuda132` (build). Read-only reference:
`D:\llama.cpp-longcat-pre-gate4` (+ its build dir), `D:\llama.cpp-longcat-mtp`
(comparator home), `D:\LongCat-Win11-Migration-20260816`,
`D:\lc_mla_blackwell` (block-0 HF oracles), `D:\lc_mla_blackwell_attn`
(attention-path HF oracles incl. `rope_cos/sin`, `attn_o_input`),
`D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved`
(checkpoint + frozen runtime `a3bc3161…`),
`D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16` (GGUF).
Never overwrite canonical Safetensors/GGUF/frozen `.bin` oracles/historical
artifacts. Do not commit `.bin`/`.log`/GGUF/Safetensors/build outputs.

## Standing rules for the next phase

**Newly authorized:** measurement-only inspection/comparison/localization of
downstream MLP/MoE boundaries (this supersedes the previous blanket MLP/MoE
prohibition for measurement only). **Still forbidden without separate
approval:** arithmetic changes to MLP/MoE (pending a reviewed plan);
production FA changes; the 2050-token run; widening the frozen logit
criterion (or any frozen gate); production RoPE changes; any other new
arithmetic. Preserve all established byte-exact gates as regressions.

## Next scientific objective (defined, NOT begun)

> Localize the first remaining unexplained downstream divergence responsible
> for the residual frozen 512-token full-model gap, **without assuming that
> the remaining 40 violations originate in block-0 attention.**

**Narrowest measurement-only starting point:** downstream boundary walking
immediately after the diagnosed block-0 attention path — block-0 MLP/MoE →
block-0 output → attn1 → per-logical-block — using existing artifacts, zero
new runs for the first pass.

- **Attribution-correct artifact selection (mandatory first step):** the
  40-violation production-style run used production `ggml_rope_ext` angles
  (all R0/R1 env unset). `cpp_attn0_mla_r1_512/` used captured HF cos/sin —
  a *different* block-0 arithmetic state. First determine whether an existing
  **production-angle A+B capture** contains the needed downstream boundaries;
  candidates: `cpp_attn0_mla_expB_512/` and `cpp_attn0_mla_attnpath_512/`
  (both pre-R0/R1-env; attnpath was Class-2 gated byte-identical to expB).
  Prefer such a capture for direct localization of the production-style gap.
  If only R1 contains a needed boundary, it may serve as a **clean-RoPE
  diagnostic walk** that does **not** directly attribute the 40
  production-style violations. If direct production-state attribution is
  required and no suitable artifact exists, **stop and design the narrowest
  production-angle capture** — never treat R1 as equivalent silently.
- HF-side comparanda: `pre-gate4\hf_logical0_stages_512_v4\` (`mlp0_resid`
  `cf48a0ad…`, `attn1_resid` `b4c1e5f6…`, `logical0_out` `5292e88a…`) and
  `pre-gate4\hf_hidden_512_v4\` (`logical_01…logical_12`, `result_norm`).
- **Mandatory pre-comparison verification for every C++/HF pair:** existence,
  SHA, shape/dtype, token/row representation (full-sequence vs final-row),
  logical-vs-physical block mapping, and semantic boundary equivalence —
  never inferred from filenames. If a boundary is absent or
  representation-incompatible, stop and design the narrowest new capture.

## Fresh-session opening prompt (self-contained)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Active writable checkout:
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`); build dir:
> `D:\llama.cpp-longcat-claude-build-cuda132`. Everything else under `D:\`
> related to this project is read-only reference (pre-gate4, mtp, migration
> package, `lc_mla_blackwell*` HF oracle dirs, checkpoint, GGUF — never
> overwrite; never commit bins/logs/GGUF). Use the validated repo-local
> `.venv` python; do not mutate it or the native CUDA 13.2/MSVC toolchain.
> First read `WIN11_HANDOFF_2026-08-17_FROZEN512.md`,
> `ARTIFACT_SHA256_20260817_CHECKPOINT.txt`, `NEXT_ACTION.md`, and
> `STATUS_2026-08-17.md`; verify branch, the pre-handoff base commit
> `6bf5ee61f`, and a clean tracked tree, and report any discrepancy before
> changes. Runtime contract: every authoritative C++ run pins the child PATH
> v13.2-first so `cublas64_13.dll` resolves to cuBLAS 6.14.11.1330, verified
> from the live process module list (wrong-runtime signature: anchors pass,
> residual `49d729e1…`); keep all diagnostic env overrides listed in the
> handoff unset. Scientific state: block-0 attention is diagnosed to current
> measurement resolution (Q/KV byte-exact after Experiments A+B; RoPE
> divergence explained, R1 proved BF16 rotation exact given exact angles, R0
> proved small downstream contribution; attention-core difference
> mechanistically explained — TF32 + F32-output + wv_b BF16-conversion — do
> not reopen without contradictory evidence; production RoPE angle generation
> remains an exact-parity gap). The frozen 512-token criterion (ATOL 0.5 /
> RTOL 0.05 over all 131,072 final logits + top-1; oracle `8825d92d…`;
> comparator `compare_longcat_sparse_gate3_logits.py` in the read-only mtp
> tree; correct historical baseline `sparse_512_fa_off_f32` = `1a8e37e2…`,
> NOT the bf16-cache `sparse_512_fa_off` = `f39f77b6…`) currently stands at
> **FAIL: 40/131,072 violations** (worst ratio 1.331, max_abs 0.852, cosine
> 0.999799, top1 483 agree), down from 2,122 — a delta owed to the aggregate
> block-0 corrective stack, not A+B alone. Never widen the criterion.
> Immediate objective (measurement-only; arithmetic changes to anything
> remain forbidden pending reviewed plans): localize the first remaining
> unexplained downstream divergence responsible for the residual frozen-512
> gap, without assuming the 40 violations originate in block-0 attention.
> Start with the zero-new-run downstream boundary walk described in the
> handoff's "Next scientific objective" section, honoring its
> attribution-correct artifact selection (production-angle captures
> expB/attnpath preferred; R1 is clean-RoPE-diagnostic only) and its
> mandatory per-pair artifact verification protocol. Start in Plan mode;
> propose the shortest rigorous path before executing.

## Key committed analysis scripts (py_compile-clean; SHAs in the checkpoint manifest)

`analyze_longcat_attn0_mla_bf16_boundary.py` ·
`analyze_longcat_attn0_kv_a_norm_semantics.py` ·
`analyze_longcat_attn_path_localization.py` ·
`analyze_longcat_attn_core_attribution.py` ·
`analyze_longcat_attn_core_mechanism.py` · `make_longcat_rope_targets.py` ·
extended `capture_longcat_hf_attn0_mla_stages.py` ·
`compare_longcat_attn0_mla_stages.py`.
