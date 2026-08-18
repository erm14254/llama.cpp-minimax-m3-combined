# Audit — Project Gate-4 Readiness (2026-08-18) — VERDICT: BLOCKED

Read-only readiness audit performed at checkpoint `774c5dc98` (verified:
branch `claude/longcat-win11`, HEAD = the N2-promotion delta commit,
promotion `bec291558` in history, tracked tree clean, model diff vs
`458a03685` = the promotion reference `fed0370e…`, promotion binary set on
disk, standing artifacts `b8067779…` / `8852bd5b…` present). Standing:
production arithmetic = Stage A + N2; strict frozen-512 PASS (0/131,072,
top-1 483); Project Gate 3 PASS (0, top-1 444); **Gate 4 NOT RUN**. No
Gate-4 execution and no arithmetic changes were performed by this audit.

## VERDICT: BLOCKED

**Gate 4 cannot be executed or protocol-frozen against the standing
branch**: the feature is absent, with additional unclosed indexer-semantic
and determinism blockers on top. (The conditional protocol below is a
review-visible future path, not an approved or executable Gate-4
protocol.)

## Primary finding (verified first-hand): the LSA implementation is absent from the standing branch

`98f5dd1cc` ("WIP: LongCat Sparse Attention, Gate 4 in progress" — the
`longcat-sparse` branch, checked out at the read-only mtp tree) is **NOT
an ancestor** of `claude/longcat-win11`; `git merge-base` = `cb7729cf1`,
the commit immediately before it. The entire parity-hardening program was
built on a fork that predates the LSA code. On the standing branch: the
DSA two-cache (`kv_mla`+`kv_lid`) is never constructed for
`LLM_ARCH_LONGCAT_FLASH_SPARSE` (the create_memory DSA case lists
GLM_DSA/DEEPSEEK32 only); the indexer exists solely in the tensor loaders
(`indexer_k_norm_eps` loaded and validated ==1e-6 but consumed nowhere);
`set_input_longcat_lsa_mask` does not exist in `llama-kv-cache.cpp`.

**Consequence: a >2048-token run on the standing build today would
silently run FULL DENSE attention — finite, plausible, semantically wrong
logits: the worst failure mode for a parity gate.** Nothing surfaced
earlier because every measured run in the program was ≤2048, where the
LSA design is dense-identical.

## What the mtp-branch LSA code (the future transplant source) contains

- Mechanism: graph-level mask unmask (`ggml_fill -inf` → `ggml_set_rows`
  zeros at top-K → `ggml_add` true causal mask; the final add makes
  boundary-straddling ubatches exact); owner/reuse via graph-scope
  `prev_top_k` with fail-closed asserts; forced set (16 init + 1024 local
  ≤ 2048) built by valid-token rank with `GGML_ASSERT`s.
- Dense↔sparse switch: `n_kv_lid > index_topk` on the padded cell count —
  crossing at 2049 real tokens (2048 stays dense, matching HF).
  **Trap: `-c 2048` makes sparsity unreachable (silent dense); `-c 4608`
  is fine.**
- **FA-off is structurally supported by the port source and avoids the
  recorded FA-on half/narrowing overflow path**: the sparse mechanism is
  mask-level, FA-off KQ uses a `GGML_PREC_F32` mul_mat,
  `-ctk f32 -ctv f32` is permitted (LID cache widens losslessly), and the
  LID-mask F32 override (`longcat_lsa → flash_attn=false` in the input
  copy) is load-bearing. **This establishes structural viability of the
  FA-off path, not yet HF semantic correctness at >2048.**
- `indexer_k_norm_eps` (1e-6) is consumed correctly in the mtp LSA graph
  — the wiring exists only where the LSA exists.
- **CUDA `ggml_top_k`/CUB does not guarantee deterministic ordering.**
  Guaranteed exact-`+inf` forced-entry ties and exact-`-inf` padding ties
  exist, but those ties alone do not prove selected-set nondeterminism:
  all 1040 forced entries fit inside K=2048 regardless of tie resolution,
  and padding should remain outside the selected finite set. A repeat-run
  probe must compare top-K membership sets, ordering, resulting masks,
  and final logits separately; byte-exact Gate-4 reproducibility is
  blocked by this issue only if membership, masks, or logits actually
  vary — ordering-only variation is insufficient.
- Surviving instrumentation on the standing branch: the
  `LONGCAT_GATE4_NAN_AUDIT` first-NaN abort harness (exit 86). The
  `LONGCAT_LSA_AUDIT` structural logs exist only in the transplant source.
- Config mechanics: standing `-b 4608` ≥ 2050 ✓; `-ub 512` crosses with a
  final 2-token ubatch (minimal indexer exercise) vs `-ub ≥ 2304` (wide
  exercise) — a future protocol decision.

## Artifact/oracle inventory

- **No Gate-4 criterion run exists on either side.** The memo-era
  "structural run" = the FA-on all-NaN 2050 run
  (`longcat_sparse_gate4_crossing_audit/`, logits 131,072/131,072 NaN)
  plus decoded `LONGCAT_LSA_AUDIT` evidence (14 owners + 14 reuse, top-K
  2048, `query_pos=2049 visible=2050 forced=1040 init=[0,15]`). The
  "gate4_short_regression" dir is the Gate-3 4-token run re-executed.
- A finite FA-off 2050 C++ run exists (`pre_gate4_2050_fa_off`, top-1
  483) — **provenance: the historical LSA-bearing/pre-parity
  implementation (the `longcat-sparse` branch lineage); it demonstrates
  finite FA-off behavior only for that historical code. Reference-only;
  it provides NO evidence that the current standing branch exercises
  sparse attention — the current standing branch has no LSA
  implementation at all.**
- **The HF >2048 logits oracle does NOT exist** and must be created. The
  capture machinery is proven: the Gate-3 core is length-agnostic
  internally (`bb82bcb6…`, frozen-v4-gated, fail-closed non-finite,
  final-row) and the 512 raw-ids wrapper pattern (`d267bf29…`) shows the
  SHA/count-pinned token-stream injection; a 2050 sibling wrapper
  (pins `eb04e101…`/2050) is the required new piece. Caveat: the
  `use_cache=False` sparse-owner path of that script family is
  unexercised. HF auto-engages sparse at `total_kv_len > 2048`
  (source-verified; reuse-side refuses dense fallback loudly).
- Frozen 2050 assets exist, SHA-stable across 6 historical runs: prompt
  `" a"×2050` = `e2791fac…` (mtp tree only — a committed repo copy is
  needed), token stream 2050×483 = `eb04e101…`.
- Comparator `6976fbc0…` is length-agnostic and reusable verbatim
  (final-row 131,072 f32; fails closed on NaN; frozen ATOL 0.5/RTOL 0.05
  + top-1).
- **No Gate-4 acceptance criterion has ever been recorded.** The
  universal frozen formula exists (CLAUDE.md, never Gate-scoped); any
  future protocol must propose the criterion explicitly as a review
  decision.
- FA-on at 2050 remains all-NaN (half overflow pre-scale; production FA
  patch prohibited; broad prescale diagnostic-only) → **any eventual
  Gate-4 PASS certifies the FA-off path only** (mandatory phrasing).

## Resolution path (each its own reviewed plan; NONE begun in this round)

**Primary: a surgical LSA semantic transplant onto the current
Stage-A+N2 standing source — NOT a generic port/cherry-pick of
`98f5dd1cc`.** The old LSA commit predates the parity-hardening program
and must not overwrite current arithmetic. The transplant round must
isolate the LSA-specific delta (DSA two-cache construction/wiring, the
graph input + `set_input_longcat_lsa_mask` builder, the indexer graph +
owner/reuse/top-K logic, `filter_lid`/memory-params pieces) and
transplant only those feature-specific changes, leaving every hardened
arithmetic path byte-untouched. **Hard transplant gates (≤2048 byte-exact
inertness proving preservation of the standing state):** frozen-512
reproduces `b8067779…` byte-exact; Gate-3 reproduces `8852bd5b…`
byte-exact; the `injectffn` operator gate `9815422f…` and all 15 standing
invariants + landings hold; binary provenance discipline throughout.

**Secondary blockers — all unclosed indexer semantics (close by direct
source proof where possible; the remainder folds into ONE coherent
first-owner indexer capture banking the K-norm, layout, score, and
selection surfaces in a single >2048 GPU experiment):**
1. Indexer K-norm BF16 cast ordering — transplant source runs an
   F32-kept norm (bf16 round after the weight multiply); HF's indexer
   `k_norm` is a `LongcatFlashRMSNorm` instance, the twice-byte-closed
   cast class `bf16(bf16(x·rsqrt(var+1e-6))·w)`, upstream of every top-K
   decision. Offline A/D reconstruction vs the captured HF K surface;
   quantify top-K membership impact. No arithmetic until measured and
   reviewed.
2. Indexer rope/nope split & layout equivalence (transplant source places
   rope at offset 0, inverted vs main attention) — prove equivalence to
   HF from source or include the post-rope indexer-K surface in the
   capture.
3. Indexer YaRN `attn_factor` behavior (transplant source applies the
   full YaRN set incl. mscale to the indexer RoPE) — verify vs HF source
   or include the relevant surface in the capture.
4. Top-K determinism characterization — the pre-registered repeat-run
   probe (same input, N runs) comparing membership sets, ordering,
   masks, and final logits separately; blocked-status applies only if
   membership/masks/logits vary.

**Conditional Gate-4 protocol (review-visible future path; executable
only after the transplant round + secondary-blocker resolutions pass
review):** frozen 2050 prompt/tokens (repo copy committed); HF oracle via
the new 2050 wrapper over the proven capture core (sparse-owner-path
caveat gated with mode checks); C++ side at the standing runner pattern
`-c 4608 -b 4608 -fa off -ctk f32 -ctv f32` with the `-ub` decision made
at review; comparator verbatim; acceptance criterion proposed as the
universal frozen formula + top-1 (explicitly a review decision);
structural gates = the transplanted `LONGCAT_LSA_AUDIT` evidence
reproduced under the criterion run's config; FA-off-only certification
phrasing mandatory.

**Flag (data, not doctrine):** an uncommitted, unreferenced planning file
`D:\LongCat-Flash-Lite-Sparse roadmap.txt` (drive root) argues for a
richer Gate 4 (boundary cases around 2048, long-position RoPE sanity,
direct selection validation). No project standing; adopt-or-discard is a
review decision.
