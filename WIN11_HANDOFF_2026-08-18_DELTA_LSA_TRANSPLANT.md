# LongCat Sparse Parity — Delta Checkpoint: Surgical LSA Transplant + ≤2048 Inertness (71b14ea5a-parented session close)

Self-sufficient **delta** handoff for the 2026-08-18 surgical-LSA-transplant
round. **This is the BOOTSTRAP document for a fresh session: read it FIRST**,
then reconcile through the prior documents in the order below. Prior handoffs
remain authoritative for machine/runtime/oracle context and are not
duplicated.

## Fresh-session document read order (this file first)

1. **This delta** (bootstrap).
2. `WIN11_HANDOFF_2026-08-17_FROZEN512.md` (machine/runtime contract)
3. `WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
4. `WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`
5. `WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`
6. `WIN11_HANDOFF_2026-08-18_DELTA_GATE4_READINESS.md`
7. `AUDIT_GATE4_READINESS_2026-08-18.md`
8. The 2026-08-18 addenda of `STATUS_2026-08-17.md` (the three
   LSA-round addenda are the freshest evidence)
9. `NEXT_ACTION.md`
10. `CLAUDE.md` (base guardrails; superseded where later handoffs say so)

## Git / standing state

- **Branch:** `claude/longcat-win11`. **Parent operational/transplant
  checkpoint: `71b14ea5a`** (the combined-checkpoint docs record). **The
  final session-close HEAD is this handoff's own docs-only commit**:
  recover it with
  `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_LSA_TRANSPLANT.md`,
  verify it equals `git rev-parse HEAD`, then verify its parent is
  `71b14ea5a`. **Tracked tree clean** (untracked files are the intentional
  gitignored capture/log classes only).
- **Standing production arithmetic remains Stage A + N2** (five il≥1 MLA
  BF16 output boundaries + trunk `ffn_norm` HF cast semantics at
  il=0..27, eps 1e-5; promotion `bec291558`). The transplant changed **no
  arithmetic**.
- **LongCat Sparse Attention is now STRUCTURALLY PRESENT on the standing
  branch, and the DSA two-cache (`kv_mla` + `kv_lid`) is live** for
  `LLM_ARCH_LONGCAT_FLASH_SPARSE` (load log: `llama_kv_cache_dsa:
  creating main KV cache, size = 4608 cells` + `creating indexer KV
  cache, size = 4608 cells`). Sparse scoring remains graph-unreachable at
  ≤2048 by the `n_kv_lid > index_topk` switch (crossing at 2049 real
  tokens; the `-c 2048` silent-dense trap still applies — the standing
  `-c 4608` is correct).
- **Standing binary set** (build dir `D:\llama.cpp-longcat-claude-build-cuda132`):
  `llama-debug.exe` `df2a57f6…`, `llama-common.dll` `261f08a5…`,
  `llama.dll` **`b58eae1dc4602868109f615457bd5f3412835464e26fe1d61fcb4cd1892c2cf4`**,
  `ggml-cuda.dll` `502e50e8…` (only `llama.dll` moved at any step of the
  round; the other three are byte-identical to the promotion set).

## Transplant lineage (all on `claude/longcat-win11`, parented on `e05ac98f7`)

1. **`7d2289b97` — infrastructure (Commit A):** the five zero-drift
   files of the historical single-commit LSA delta
   (`git diff cb7729cf1..98f5dd1cc`) applied VERBATIM with diff-identity
   proof: `llama-kv-cache.h/.cpp` (`set_input_longcat_lsa_mask` builder:
   valid-token-rank mask, −inf/0/+inf forced init-16/local-1024 set,
   forced-count asserts, `LONGCAT_LSA_AUDIT mask` log),
   `llama-kv-cache-dsa.cpp` (LID rope_type NORM), `llama-graph.h/.cpp`
   (`longcat_lsa` member; set_input LSA branch; DSA `build_attn`
   nullptr-top_k dense fast path; `build_attn_inp_k_dsa` LongCat
   overrides incl. the load-bearing `flash_attn=false` F32 LID mask).
   Dead code until activation; tripwire frozen-512 reproduced
   `b8067779…` byte-exact.
2. **`54f06950b` — activation + semantic model graft (Commit B):**
   `llama-model.cpp` verbatim (sparse arch into the DSA `create_memory`
   case, owner-filtered `filter_lid`; MTP context unchanged — dense MLA,
   no indexer); `longcat-flash-ngram.cpp` semantic graft M1–M6 (includes;
   LSA prologue locals; `build_attn_inp_k_dsa` branch + `prev_top_k`;
   `q_lora` ALIAS taps at the post-q_a_layernorm surfaces — hardened
   chains byte-untouched; the indexer sub-graph inserted 201/201 lines
   byte-identical to the historical block; `build_attn` switch to the
   DSA overload with `top_k` + `sparse_expected` assert). Deleted lines
   = exactly the two replacement anchors.
3. **`36fe02796` — the two instrumentation-only `cb()` de-clobbers**
   (see the deviations section below).
4. **`3ac389d99` — injectffn prior-reference epoch correction**
   (harness-only, one executable line; see below).
5. **`71b14ea5a` — docs checkpoint** (combined lsaC/lsaD Stage-3 record).

## The two approved instrumentation deviations from historical M5

The transplanted indexer block is otherwise verbatim `98f5dd1cc`. Exactly
two historical `cb()` structural markers were REMOVED (review-authorized;
recorded in-source as `LONGCAT_LSA_DEVIATION` comments; mechanical diff
proof = the two `cb()` lines deleted, comments added, nothing else):

1. **`cb(indexer_k, "lsa_full_owner", il)`** — in the sparse-inactive
   regime it renamed the canonical `lsa_indexer_k-<il>` surface that the
   planned ≤512 below-threshold indexer-K dump proof must key on.
2. **`cb(cur, "lsa_full_reuse", il)`** — it renamed the standing
   `attn_norm-<il>` tensor of every odd block, silently breaking the
   name-keyed `attn_norm-3` full-sequence dump (the original Stage-3
   inventory failure).

A `cb()` call only assigns a tensor name: **zero graph nodes, values,
owner/reuse semantics (`prev_top_k` + fail-closed asserts), masks, top-K,
K-norm, RoPE/layout, or YaRN behavior changed. The `LONGCAT_LSA_AUDIT`
`LLAMA_LOG_DEBUG` owner/reuse/mask logs remain intact and are the >2048
structural-evidence mechanism.**

## The harness prior-epoch correction (why it strengthens, not weakens)

The injectffn prior-manifest gate compared against the pre-N2
`cpp_resid_walk_injectffn_ffnNorm_512` while the N2 promotion had
graduated `logical0_mlp0_resid.bin`/`logical0_attn1_resid.bin` into the
standing invariant set at their N2 values — a latent self-contradiction
created at graduation (the promoN2 run predated the graduated harness;
the first transplant run threw earlier at inventory), first exercised by
the lsaC run: **FAIL 2/17 with both surfaces byte-exact to the STANDING
values** (`32134b64…`/`398de74c…` vs the retired pre-N2 `de18420a…`/
`8a7ab8c6…`). Correction (`3ac389d99`, one executable line): injectffn
`$priorDir` → `cpp_resid_walk_injectffn_promoN2_512`, the standing
Stage-A+N2 reference epoch. Mechanical pre-edit record: the promoN2
manifest carries 17/17 allowlisted entries; the two epochs agree
byte-for-byte on the other 15 names and differ on exactly the two
graduated surfaces. **The correct standing comparison epoch is restored
with no comparison dropped or weakened — stronger than the rejected
alternative of excluding the two names.**

## The complete pre-registered ≤2048 inertness checkpoint: PASS

**Scope statement (mandatory phrasing): this is a PASS of the complete
pre-registered ≤2048 inertness checkpoint — NOT exhaustive testing of
every sequence length ≤2048.** All runs under the full runtime contract
(child-PATH-pinned live-verified cuBLAS **6.14.11.1330**; audited env
sweep clean; placement `(29, 15, ATTN)`, offloaded 29/30;
`graphs reused = 0`).

| Gate | Result |
|---|---|
| Frozen-512 (`cpp_logits_512_lsaC/`; also lsaA/lsaB) | **PASS — logits `b8067779…` BYTE-EXACT; 0/131,072; top-1 483** |
| Project Gate-3 (`cpp_logits_gate3_lsaC/`; also lsaB) | **PASS — logits `8852bd5b…` BYTE-EXACT; 0 violations; top-1 444** |
| DSA two-cache structural confirmation | **PASS — both creation lines in the load log; placement/offload unchanged** |
| 15/15 standing invariants (`cpp_resid_walk_injectffn_lsaD_512/`) | **PASS byte-exact** |
| Landing | **`block1_attn0_resid_full.bin` == `4718460b…` byte-exact** |
| Restored surface | **`block1_attn1_norm_full.bin` == `1ce81e69…`** (the N2-promotion reference, pre-registered) |
| Operator surface | **`block1_ffn0_norm_full.bin` == `9815422f…` byte-exact** |
| Full dump inventory | **PASS — 32/32 present, sizes OK** |
| Prior-manifest reproduction | **PASS — 17/17 byte-identical vs the promoN2 reference** |
| Binary provenance (every build of the round) | **only `llama.dll` moved**; exe/`llama-common.dll`/`ggml-cuda.dll` invariant |

## Standing constraints (carried forward, unchanged)

- **Project Gate 4 remains NOT RUN. No 2050-token execution has occurred
  on the transplanted standing branch.**
- **The four indexer questions remain deliberately untouched** (the
  transplant preserved historical LSA semantics so later measurements
  retain causal attribution): (1) indexer K-norm BF16 cast ordering
  (F32-kept vs HF's twice-cast class at eps 1e-6); (2) rope/nope
  split/layout equivalence (rope-at-offset-0, NORM-interleaved);
  (3) YaRN `attn_factor` behavior on the indexer RoPE; (4) top-K
  determinism — ordering, membership set, resulting mask, and final
  logits compared SEPARATELY.
- **No arithmetic correction is authorized.** No Gate-4 acceptance
  criterion is authorized or established (a future review decision).
  Never widen any frozen criterion; endpoint review baselines run
  from 0. Eventual long-context work is **FA-off only** (FA-on remains
  unfixed/prohibited).
- Runtime contract unchanged: pinned live-verified cuBLAS 6.14.11.1330;
  audited env sweeps; `.venv`/toolchain/reference trees immutable; no
  `.bin`/`.log`/GGUF commits; never overwrite canonical artifacts.

## Next phase (a plan question, NOT begun): measurement-only >2048 preparation

The next round is the **pre-freeze surface audit**, then the Type-S/
Type-P 2050 measurement protocol design — **not a 2050 execution**:

1. **Below-threshold owner-K surfaces** (`lsa_indexer_k_proj`,
   `lsa_indexer_k_norm`, `lsa_indexer_k` — owners compute/store indexer
   K at every length): source proof **plus a ≤512 runtime dumpability
   proof** (dtype, shape at the chosen `-ub`, byte size, dump-helper
   behavior incl. BF16 widening and the ne[2]≠1 constraint).
2. **Sparse-only scoring/selection surfaces** (`lsa_indexer_q`,
   `lsa_indexer_weights`, `lsa_indexer_kq`, `lsa_indexer_score`,
   `lsa_top_k_owner`/`_reuse` — cannot exist at ≤512): **static
   dtype/shape/size/serializer audit first**, explicitly including the
   **I32 top-K serialization question** (the float-oriented dump helper
   has never been proven to serialize I32; do not assume `cont_2d`
   solves dtype support). Runtime proof only in the eventually approved
   2050 Type-S run.
3. Then the final proposed **Type-S/Type-P split protocol** (Type S:
   instrumented surface capture, authorized dump env var only, no
   `--save-logits`; Type P: production logits, zero `LONGCAT_*`;
   `--save-logits` disables the eval callback so the two can never be
   one run), the two-family N≥3 determinism probe (ordering / membership
   / masks from Type S; logits from Type P; separated verdicts), the
   2050 placement policy (per-type fit/placement tuples; `-fitt 4096`
   fixed; the standing `(29,15,ATTN)` gate belongs to `-ub 512` runs),
   frozen 2050 assets (prompt `" a"×2050` `e2791fac…` needs a committed
   repo copy; tokens 2050×483 `eb04e101…`), and the HF-side 2050 wrapper
   with fail-closed sparse-engagement asserts. Any new dump plumbing is
   instrumentation-only and re-passes ≤512 inertness before any 2050
   execution.

## Fresh-session opening prompt (self-contained; copy-paste)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. Read FIRST
> `WIN11_HANDOFF_2026-08-18_DELTA_LSA_TRANSPLANT.md` (the bootstrap
> delta), then reconcile in its stated order: the four prior WIN11
> handoffs, `WIN11_HANDOFF_2026-08-18_DELTA_GATE4_READINESS.md`,
> `AUDIT_GATE4_READINESS_2026-08-18.md`, the 2026-08-18 addenda of
> `STATUS_2026-08-17.md`, `NEXT_ACTION.md`, `CLAUDE.md`. Verify
> read-only: branch; clean tracked tree; **the final session-close HEAD
> = the docs-only commit containing the bootstrap delta** (recover via
> `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_LSA_TRANSPLANT.md`,
> confirm it equals `git rev-parse HEAD`) **with parent `71b14ea5a`**;
> transplant lineage `7d2289b97` → `54f06950b` → `36fe02796` →
> `3ac389d99` → `71b14ea5a` in history; standing arithmetic Stage A + N2;
> LSA structurally present with the DSA two-cache live; the complete
> pre-registered ≤2048 inertness checkpoint PASS (frozen-512 `b8067779…`
> byte-exact, Gate-3 `8852bd5b…` byte-exact, 15/15 invariants, landing
> `4718460b…`, `block1_attn1_norm_full.bin` = `1ce81e69…`, operator
> `9815422f…`, inventory 32/32, prior reproduction 17/17, only
> `llama.dll` = `b58eae1d…` moved); Gate 4 NOT RUN; no 2050 execution has
> occurred on the transplanted branch. Report any discrepancy before
> planning. The next round is **measurement-only >2048 preparation, NOT a
> 2050 execution**, in exactly this order: (1) recover/verify the final
> HEAD as above; (2) audit the proposed LSA capture surfaces for
> dtype/exact shape/byte size/dump-helper support; (3) perform (or, where
> instrumentation would be needed, design) the ≤512 below-threshold
> owner-K dump proof (`lsa_indexer_k_proj` / `lsa_indexer_k_norm` /
> `lsa_indexer_k`); (4) statically audit the sparse-only scoring/top-K
> surfaces including the I32 top-K serialization question (never assume
> the float-oriented dump helper or `cont_2d` handles I32); (5) produce
> the final proposed Type-S/Type-P 2050 measurement protocol (split
> runs, two-family N≥3 determinism probe with separated
> ordering/membership/mask/logits verdicts, per-type placement tuples,
> frozen 2050 assets, HF wrapper with sparse-engagement asserts);
> (6) STOP FOR REVIEW before any 2050 execution. Do not implement
> measurement instrumentation without review, do not run 2050 tokens, do
> not change any arithmetic, do not run Gate 4, never widen any frozen
> criterion, and honor the full runtime contract on every run. Start in
> Plan mode; stop for review before executing anything.
