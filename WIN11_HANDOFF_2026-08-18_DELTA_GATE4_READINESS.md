# LongCat Sparse Parity — Delta Checkpoint: Gate-4 Readiness Audit (774c5dc98 → this commit)

Self-sufficient **delta** handoff for the 2026-08-18 Gate-4 readiness
audit round (docs-only). Prior handoffs remain authoritative for
machine/runtime/oracle context and are not duplicated:
`WIN11_HANDOFF_2026-08-17_FROZEN512.md`,
`WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`,
`WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`,
`WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`.

## Recommended fresh-session document read order

1. `WIN11_HANDOFF_2026-08-17_FROZEN512.md` (machine/runtime contract)
2. `WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
3. `WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`
4. `WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`
5. **This delta**
6. `AUDIT_GATE4_READINESS_2026-08-18.md` (full audit findings)
7. The 2026-08-18 addenda of `STATUS_2026-08-17.md`
8. `NEXT_ACTION.md`
9. `AUDIT_MLA_PRODSCOPE_2026-08-18.md`
10. `CLAUDE.md` (base guardrails; superseded where later handoffs say so)

## Git / standing state

- **Branch:** `claude/longcat-win11`. **Checkpoint/HEAD: `96e39c609`**
  (the docs-only Gate-4 readiness-audit commit; recover with
  `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_GATE4_READINESS.md`
  for THIS delta's own commit on top). **Tracked tree clean** (untracked
  files are the intentional gitignored capture/log classes only).
- **Standing production arithmetic = Stage A + N2** (five il≥1 MLA BF16
  output boundaries + trunk `ffn_norm` HF cast semantics at il=0..27,
  eps 1e-5; promotion commit `bec291558`, diff-identity `fed0370e…`).
- **Strict frozen-512 criterion: PASS — 0/131,072 violations, top-1 483,
  standing logits `b8067779…`** (NOT full-model HF parity; the
  uncorrected-mechanism register stands: N1 trunk attn_norm il≥1, LoRA
  cast+eps pair, production RoPE angles, il≥1 attention core, MLP/MoE,
  MTP twins — no identified interaction partner).
- **Project Gate 3: PASS — 0 violations, top-1 444, standing logits
  `8852bd5b…`.**
- **Project Gate 4: NOT RUN.**

## Gate-4 readiness verdict: **BLOCKED**

- **Decisive blocker: the LongCat Sparse Attention (LSA) implementation
  does not exist on the standing branch.** Verified from git ancestry and
  source wiring: the DSA two-cache is never constructed for the sparse
  arch, the LSA mask builder is absent, `indexer_k_norm_eps` is consumed
  nowhere. **A >2048-token run today would silently remain DENSE and is
  therefore not a valid Gate-4 run** (finite, plausible, semantically
  wrong logits).
- **Historical LSA source: commit `98f5dd1cc`** ("WIP: LongCat Sparse
  Attention, Gate 4 in progress") on the old `longcat-sparse` lineage
  (merge-base with the standing branch: `cb7729cf1`). It predates the
  entire parity-hardening program and **must NOT be cherry-picked or
  merged wholesale** — it would overwrite hardened arithmetic.
- **No arithmetic changes and no LSA transplant were begun in the
  readiness-audit round** (docs-only: `96e39c609`).

## Next round (a plan question, NOT begun): the reviewed surgical LSA semantic transplant

Isolate **only the LSA-specific feature delta** from `98f5dd1cc` (DSA
two-cache construction/wiring, graph input + `set_input_longcat_lsa_mask`
builder, indexer graph + owner/reuse/top-K logic, `filter_lid`/
memory-params pieces) and transplant it onto the hardened Stage-A+N2
source, preserving current arithmetic byte-untouched. **Mandatory ≤2048
byte-exact inertness gates (proving preservation of the standing state):**

1. frozen-512 logits reproduce **`b8067779…` byte-exact**;
2. Gate-3 logits reproduce **`8852bd5b…` byte-exact**;
3. the `injectffn` operator gate **`9815422f…` byte-exact**;
4. **all 15 standing invariants + landing gates** hold (harness
   `run_longcat_resid_walk_512.ps1`; binary provenance discipline
   throughout).

**Only after those inertness gates pass may a >2048 measurement-only
indexer round occur** — preferably ONE coherent first-owner indexer
capture banking the K-norm/layout/score/selection surfaces, resolving the
four open indexer items:

1. **Indexer K-norm BF16 cast ordering** (transplant source is F32-kept;
   HF's indexer `k_norm` is the twice-byte-closed cast class at eps 1e-6
   — upstream of every top-K decision);
2. **rope/nope split & layout equivalence** (transplant source inverts
   the split order vs main attention);
3. **YaRN `attn_factor` behavior** on the indexer RoPE;
4. **top-K determinism characterization** — a repeat-run probe comparing
   **separately**: top-K membership sets, ordering, resulting masks, and
   final logits (CUDA/CUB ordering is not guaranteed; selected-set
   nondeterminism is NOT established; blocked-status applies only if
   membership/masks/logits actually vary).

## Constraints carried forward

- **The HF >2048 logits oracle does not yet exist** (capture machinery is
  proven and length-agnostic; a 2050 raw-ids wrapper is the missing
  piece; frozen 2050 assets exist: prompt `" a"×2050` `e2791fac…`, tokens
  2050×483 `eb04e101…` — currently mtp-tree-only, needs a committed repo
  copy).
- **Eventual long-context work is FA-off only**; FA-on remains
  unfixed/prohibited (half overflow pre-scale; broad prescale patch
  diagnostic-only). Any eventual Gate-4 PASS certifies the FA-off path
  only.
- **No Gate-4 acceptance criterion has historically been established** —
  any future protocol proposes it explicitly as a review decision (the
  universal frozen formula is the natural candidate, never assumed).
- Runtime contract unchanged: child-PATH-pinned live-verified cuBLAS
  6.14.11.1330; 43-name env sweep; placement `(29, 15, ATTN)`;
  `graphs reused = 0`; `.venv`/toolchain/reference trees immutable; the
  frozen-512 criterion is never widened; future endpoint review
  baselines run from 0. Trap recorded: `-c 2048` would make sparsity
  unreachable (silent dense); the standing `-c 4608` is correct.

## Fresh-session opening prompt (self-contained; copy-paste)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. Read, in order:
> the four prior WIN11 handoffs, then
> `WIN11_HANDOFF_2026-08-18_DELTA_GATE4_READINESS.md`,
> `AUDIT_GATE4_READINESS_2026-08-18.md`, the 2026-08-18 addenda of
> `STATUS_2026-08-17.md`, `NEXT_ACTION.md`,
> `AUDIT_MLA_PRODSCOPE_2026-08-18.md`, `CLAUDE.md`. Verify read-only:
> branch, clean tracked tree, HEAD = the docs-only commit containing the
> Gate-4-readiness delta, with `96e39c609` and the promotion `bec291558`
> in its history; standing production arithmetic = Stage A + N2 with
> frozen-512 PASS (0/131,072, top-1 483, logits `b8067779…`) and Gate-3
> PASS (0, top-1 444, `8852bd5b…`); Gate 4 NOT RUN; Gate-4 readiness
> verdict BLOCKED (LSA absent from the standing branch — a >2048 run
> today would silently remain dense and is not a valid Gate-4 run).
> Report any discrepancy before planning. The next task is the reviewed
> **surgical LSA semantic transplant** round: begin with a read-only
> source audit isolating the LSA-specific feature delta of the historical
> `98f5dd1cc` (old `longcat-sparse` lineage — never cherry-pick/merge it
> wholesale) against the hardened Stage-A+N2 source, then plan the
> transplant with the mandatory ≤2048 byte-exact inertness gates
> (`b8067779…`, `8852bd5b…`, `9815422f…`, all 15 standing invariants +
> landings) before any change. Only after those gates pass may the >2048
> measurement-only indexer round (four recorded blockers: K-norm cast
> ordering, rope/nope layout, YaRN attn_factor, the separated
> membership/ordering/mask/logits top-K determinism probe) be planned.
> The HF >2048 oracle does not yet exist; long-context work is FA-off
> only; no Gate-4 acceptance criterion exists (a future review
> decision); never widen any frozen criterion; full runtime contract on
> every run. Start in Plan mode; stop for review before executing
> anything.
