# LongCat Sparse Parity — Delta Checkpoint: Pre-Gate-4 Causal Chain (98429981e → 9d80e0898)

Self-sufficient **delta** handoff for a fresh session, covering the
scientific work performed after the frozen-512 checkpoint handoff
(`98429981e`). The base workstation/toolchain/runtime handoff
`WIN11_HANDOFF_2026-08-17_FROZEN512.md` **remains authoritative and is not
duplicated here**.

A fresh session must read and reconcile, in order:

1. `WIN11_HANDOFF_2026-08-17_FROZEN512.md` — base handoff (machine, runtime
   contract, frozen criterion, paths & roles).
2. **This delta** — everything since `98429981e`.
3. `STATUS_2026-08-17.md` — the full evidence trail (eight addenda were
   appended during this delta: downstream walk, causal reset, block-1
   localization, dual reset, block-2 MLA walk, quad reset, hex reset, plus
   wording-correction banners).
4. `NEXT_ACTION.md` — current decision state.
5. `ARTIFACT_SHA256_20260817_CHECKPOINT.txt` — the frozen-512 manifest
   (unchanged by this delta).
6. `CLAUDE.md` — base guardrails (partially superseded where the base
   handoff and this delta say so).

## Git state

- **Branch:** `claude/longcat-win11`. Tracked tree clean (untracked working
  artifacts only — capture bins/logs, gitignored classes, intentional).
- **Last scientific HEAD: `9d80e0898`** ("hex reset - both frontier
  operators HF-equivalent at the BF16 output boundary"). **This delta
  document's own commit is docs-only and sits on top of it**; recover it
  with `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
  and verify it is the current HEAD, with `9d80e0898` in its parent
  history.
- **Scientific commit sequence `98429981e..9d80e0898` — exactly 24
  commits** (Git-derived, verbatim):
  `5fc07c8f0` downstream boundary walk · `551ecb7c0` walk wording
  (observational, origin unresolved) · `2f827a91e` causal-reset
  instrumentation · `2fce13c6a` HF full-sequence residual capture ·
  `42ae149be` causal reset at logical_00 · `14fed6e55` non-additivity
  wording · `d95939f49` block-1 sub-boundary dump specs · `20c206510` HF
  layer-1 stages capture · `cc16e83cf` block-1 localization (RMSNorm cast
  class) · `7342efe88` weight-validation wording · `bcbc3846d` attn_norm-2
  injector + attn_out-2 dump · `32968b6b2` HF stages re-capture with
  attn0_out · `aa07c29a1` dual reset (block-2 attention implicated) ·
  `ee1af769e` S3-closure scoping · `02d9687dc` block-2 MLA walk plumbing ·
  `83ac7b758` HF block-2 MLA internals capture · `0dad46a92` block-2
  offline targets (block-0 known-answer gated) · `d04672f2f` block-2 MLA
  walk (root GEMMs pure output boundaries; eps mismatch recorded) ·
  `5f45bccad` interpretation distinctions · `28b5bec64` projection-output
  injector · `c02b56bd7` quad reset (norm 2×2) · `301b78be4` scope
  qualification + KV near-closure label · `b98070666` norm-output
  injector · `9d80e0898` hex reset (frontier holds).
- **Instrumentation source HEAD `b98070666`**; current binary set
  (`D:\llama.cpp-longcat-claude-build-cuda132\bin\Release`):
  `llama-debug.exe` `df2a57f6…` (unchanged from checkpoint),
  `llama-common.dll` `9476e31f…` (carries all env-gated debug.cpp
  instrumentation), `llama.dll` `97592bb7…` (carries only the il==2
  name/dump plumbing from `02d9687dc`, inertness proven by 81/81
  manifest reproduction), `ggml-cuda.dll` `502e50e8…` (unchanged).

## Causal findings through `9d80e0898` (each with committed artifacts)

All measurement-only; production arithmetic untouched throughout; every
C++ run under the pinned live-verified cuBLAS 6.14.11.1330 with
`graphs reused = 0`, placement `(29, 15, ATTN)`.

1. **Observational downstream walk** (`downstream_walk_512/`, comparator
   `6c600599…`): no single divergence boundary; smooth ×1.17–2.64/step
   growth, direction churn; **observational — cannot distinguish inherited
   propagation from local generation** (corrected wording committed).
2. **Causal reset at `logical_00`** (`resid_inject_walk_512/`, comparator
   `f5429368…`; full-sequence [512,3072] HF oracle injection, causal-cut
   proof embedded): the downstream trunk **locally regenerates the dominant
   divergence share under this intervention** — `logical_01` ratio 0.679,
   endpoint `result_norm` ratio 0.773 — **non-additive, not an
   upstream/downstream partition**. First-raw and 0.01/0.10/0.50 crossings
   coincide at `logical_01`.
3. **Block-1 sub-boundary localization** (`block1_stages_512/`, comparator
   `85bc1b40…`): first-raw = first-BF16-irreducible = `attn_norm-2` with
   byte-exact input → **operator-attributed: the trunk RMSNorm
   cast-ordering class**. HF mechanism byte-closed:
   `bf16(bf16(x·rsqrt(var+1e-5))·w)` 1,572,864/1,572,864 (trunk-norm eps
   byte-identified **1e-5**); C++ = plain F32-kept norm to ≤4-ulp reduction
   noise. GGUF norm weight functionally validated (not raw-compared).
4. **Dual reset (`logical_00` + `attn_norm-2`)** (`block2_attn_reset_512/`,
   comparator `71bc3ca6…`): with both block-2 attention operands exact in
   value, pre-add `attn_out-2` is **BF16-irreducible (26.7% bf16-match,
   rel 6.13e-3) → block-2 attention causally implicated including its
   F32-carrier/dtype/kernel semantics**. HF residual-add mechanism
   **closed for that frozen capture by exact-input S3**
   (`bf16(f32(HF_attn_out)+f32(resid))` byte-exact); S1 count-equality is
   supporting evidence only.
5. **Block-2 MLA-internal walk** (`block2_mla_walk_512/`, comparator
   `c8f86663…`; all-input exactness rule incl. operator parameters): both
   root projection GEMMs (weights raw-BF16 bit-identical) are
   **HF-equivalent at the BF16 output boundary** (786,432/786,432 and
   294,912/294,912 after rounding) but their raw F32 outputs are **not
   pipeline-equivalent** — a missing/different representation boundary,
   not a GEMM arithmetic failure. **Direct semantic parameter mismatch
   discovered: C++ il≥1 LoRA norms run eps = 1e-5 (`f_norm_rms_eps`) while
   HF uses 1e-6** (GGUF metadata + source + runtime-instantiated-module
   gate all verified).
6. **Quad reset (+ exact projection outputs)** (`block2_norms_512/`,
   comparator `9c586534…`): **scoped to the exact-predecessor intervention
   in this frozen 512-token capture** — both LoRA norms bf16-irreducible
   under byte-exact inputs and widening-verified weights → **norm operator
   composite attributed**; non-additive 2×2: **HF BF16 cast ordering is
   the dominant measured factor (~2.3e-3 under either eps); the eps
   mismatch (~1e-4 F32-regime → ~8.7e-4 BF16-cast-regime) and the missing
   projection BF16 boundary (65.0→73.8% / 60.0→73.5% bf16-match) are
   smaller but independently real**. `q_a_norm` HF mechanism byte-closed
   by D6 (eps 1e-6); `kv_a_norm` = **near-closure/model residue
   (262,137/262,144, 7 one-BF16-ulp misses), not exact closure**.
7. **Hex reset (+ exact norm outputs)** (`block2_frontier2_512/`,
   comparator `e9fd0a27…`): `q_b_proj-2` (weight bit-identical
   9,437,184/9,437,184, bias absent both sides) and `kv_cmpr_scaled-2`
   (scale constant f32-bit-identical `0x401cc471`) are both
   **BF16-EQUIVALENT** (3,145,728/3,145,728 and 262,144/262,144) —
   representation-boundary differences; being raw-different, **neither
   causally advances the frontier**; the narrowest exact-output
   predecessor-reset designs are recorded, not executed.

**Emerging picture (descriptive, capture-scoped):** every operator brought
to all-exact inputs+parameters so far is HF-equivalent at the BF16 output
boundary; the causally established block-2 divergence machinery is the
**missing BF16 representation boundaries** at projection/scale outputs plus
the **norm composite** (cast dominant, eps secondary) — mirroring the
block-0 A+B findings without transferring any mechanism by assumption.

## Proven vs open

**Proven (per-operator, capture-scoped):** complete causal cut at
`logical_00` (both dataflows); block-1/2 semantic pairings and layouts;
projection GEMMs, `wq_b` GEMM, and the KV scale HF-equivalent at the BF16
boundary from exact inputs; norm composite decomposed (cast/eps/predecessor
factors, non-additive); HF residual-add and `q_a_norm` mechanisms
byte-closed (scoped); root/q_b weights raw-BF16 bit-identical; norm weights
exact widenings; eps values proven on both sides (runtime gate + GGUF
metadata); rope target generator regenerates block-0 canonical targets
byte-exact.

**Open:** `kv_a_norm` exact closure (near-closure label); production RoPE
angle generation (composite rule — R1 remains the standing proof that
rotation arithmetic is exact given exact angles); the attention core at
il≥1 (all-input rule never satisfied; the block-0 V7″/TF32 mechanism is
closed **for block 0 only**); MLP/MoE at il≥1 unmeasured per-operator;
blocks 3+ not individually walked; and the **strict frozen-512
production-style parity state is unchanged: FAIL 40/131,072 with top-1
agreement at the `6bf5ee61f` production state** (none of this delta's
env-gated instrumentation altered production arithmetic).

## Project gates (original Gate 1–7 terminology, per `HANDOFF_MEMORANDUM_2026-08-15.md`)

- **Project Gate 3 (≤2048 exact/full-attention regression): PASS under its
  established criterion.** The **strict frozen-512 production-style HF
  parity criterion is a separate, stricter measure** (comparator
  `6976fbc0…`) and currently stands at **FAIL 40/131,072, top-1 agrees** —
  it is never to be phrased as a Gate-3 state.
- **Project Gate 4 (>2048-token true LongCat Sparse Attention run): NOT
  RUN.** All work through `9d80e0898` is **pre-Gate-4 parity hardening**.

## Guardrails delta (on top of the base handoff)

- **`LONGCAT_*` env vars — 8 names, source-derived (count descriptive):**
  `LONGCAT_HIDDEN_DUMP_DIR`, `LONGCAT_RESID_WALK_DUMP_DIR` (dump dirs);
  `LONGCAT_ROPE_INJECT_DIR`, `LONGCAT_ROPE_ORACLE_DIR` (R0/R1, block-0);
  `LONGCAT_RESID_INJECT_DIR`, `LONGCAT_ATTN_NORM2_INJECT_DIR`,
  `LONGCAT_PROJ_INJECT_DIR`, `LONGCAT_NORM_INJECT_DIR` (the causal-reset
  injectors). All env-gated in `common/debug.cpp` (plus the R1 oracle path
  in the model file); **all must be unset** for production-style runs.
- **Environment hygiene is bound to the explicit authoritative name list in
  `run_longcat_resid_walk_512.ps1`** (source-audited, wrapper-aware;
  reconciliation 19→21→23→39→42 recorded in STATUS and the committed
  JSONs; the names are the contract, the count is descriptive).
- **Run harness:** `run_longcat_resid_walk_512.ps1` modes
  `control|inject|inject2|inject3|inject4` (progressively: no reset →
  `logical_00` → +`attn_norm-2` → +both projections → +both norms), with
  preflight binary/oracle SHA gates, child-only CUDA v13.2-first PATH pin,
  live-process cuBLAS module verification, landing/inertness/reproduction
  postflight gates.
- **Oracle/artifact locations:** HF full-seq oracles `D:\lc_resid_walk_512`,
  `D:\lc_block1_stages_512`, `D:\lc_block2_mla_512` (bins gitignored;
  sidecars mirrored under `_external_artifacts/`); offline targets
  `block2_mla_targets/` (known-answer-gated generator
  `make_longcat_block2_mla_targets.py`); per-run manifests + provenance
  sidecars committed in each `cpp_resid_walk_*` dir.
- Standing prohibitions unchanged: no arithmetic changes anywhere pending
  the reviewed plan; no generalization of block-0 A+B or the trunk RMSNorm
  semantics; no production FA patch; no Gate-4/2050-token run; never widen
  the frozen criterion or any byte-exact gate; never overwrite canonical
  artifacts; no `.bin`/`.log`/GGUF commits; `.venv`/toolchain untouched.

## Next objective (a plan question, NOT a pre-approved patch; NOT begun)

Prepare a **reviewed narrow production arithmetic plan** grounded in the
proven findings — the measured BF16 representation boundaries at
projection/scale outputs, the HF norm cast semantics, and the C++ 1e-5 vs
HF 1e-6 LoRA-norm epsilon mismatch — with the explicit requirement that
**the plan must first audit the source to determine exactly which
standard-path operations/layers share these mechanisms before proposing
any code change; global il≥1 scope must NOT be assumed solely from the
block-2 measurements.** No arithmetic until that plan is reviewed and
approved. The recorded predecessor-reset measurement designs (exact
`q_b_proj-2` / `kv_cmpr_scaled-2` output resets toward RoPE-composite /
absorption / attention-core judgment) remain available if review requests
more evidence first.

## Fresh-session opening prompt (self-contained)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. First read and
> reconcile, in order: `WIN11_HANDOFF_2026-08-17_FROZEN512.md` (base
> machine/runtime handoff), `WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
> (this delta), `STATUS_2026-08-17.md`, `NEXT_ACTION.md`,
> `ARTIFACT_SHA256_20260817_CHECKPOINT.txt`, `CLAUDE.md`. Verify the
> branch, a clean tracked tree, and the Git state: the **current HEAD must
> be the docs-only commit containing this delta file** — recover it with
> `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
> — and separately verify that **`9d80e0898` is the recorded last
> scientific checkpoint in its parent history** (24 scientific commits
> since `98429981e`). Report any discrepancy before changes. Runtime
> contract: every authoritative C++ run pins the child PATH v13.2-first so
> `cublas64_13.dll` resolves to cuBLAS 6.14.11.1330, verified from the
> live process module list; all eight `LONGCAT_*` env vars and the full
> audited hygiene list in `run_longcat_resid_walk_512.ps1` stay unset
> except as a harness mode sets them. Scientific standing: the pre-Gate-4
> causal chain through the hex reset is complete (see the delta's findings
> table); project Gate 3 (≤2048 exact/full-attention regression) is PASS;
> the separate strict frozen-512 production-style parity criterion stands
> at FAIL 40/131,072 with top-1 agreement; project Gate 4 (>2048 true
> LongCat Sparse Attention) has NOT been run. Immediate objective: prepare
> the **reviewed narrow production arithmetic plan** from the
> independently measured il≥1 MLA findings (BF16 representation
> boundaries, norm cast semantics, the 1e-5-vs-1e-6 LoRA-norm eps
> mismatch), beginning with a source audit of exactly which
> standard-path operations/layers share these mechanisms — do NOT assume
> global il≥1 scope from block-2 measurements alone, and do NOT change
> any arithmetic until the plan is reviewed and approved. Start in Plan
> mode; propose the audit-and-plan path before executing anything.
