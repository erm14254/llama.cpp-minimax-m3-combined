# LongCat Sparse Parity — Delta Checkpoint: LSA Measurement Apparatus (7fe956a67-parented session close)

Self-sufficient **delta** handoff for the 2026-08-18 LSA
measurement-apparatus instrumentation round. **This is the BOOTSTRAP
document for a fresh session: read it FIRST**, then reconcile through the
prior documents in the order below. Prior handoffs remain authoritative
for machine/runtime/oracle context and are not duplicated.

## Fresh-session document read order (this file first)

1. **This delta** (bootstrap).
2. `WIN11_HANDOFF_2026-08-18_DELTA_LSA_TRANSPLANT.md` (LSA transplant +
   ≤2048 inertness checkpoint)
3. `WIN11_HANDOFF_2026-08-17_FROZEN512.md` (machine/runtime contract)
4. `WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
5. `WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`
6. `WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`
7. `WIN11_HANDOFF_2026-08-18_DELTA_GATE4_READINESS.md` +
   `AUDIT_GATE4_READINESS_2026-08-18.md`
8. The 2026-08-18 addenda of `STATUS_2026-08-17.md` (the final addendum
   is this round's evidence record)
9. `NEXT_ACTION.md`
10. `CLAUDE.md` (base guardrails; superseded where later handoffs say so)

## Git / standing state

- **Branch:** `claude/longcat-win11`.
- **`09e42fc14` = the committed instrumentation checkpoint** (full SHA
  `09e42fc14bfc99d852a10a22164e1935074d03d9`; parent = the transplant
  bootstrap `46f412728`). Every run of the round executed at this commit
  with a clean tracked tree; each run's `run_provenance.json` records
  `git_head`.
- **`7fe956a67` = the operational/docs checkpoint for the completed
  round** (full SHA `7fe956a67734b4a9d4798fd52fdfdfd1fe9981cb`; the
  STATUS addendum + NEXT_ACTION round record).
- **The final session-close HEAD is this handoff's own docs-only
  commit**: recover it with
  `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_LSA_MEASUREMENT_APPARATUS.md`,
  verify it equals `git rev-parse HEAD`, and verify `7fe956a67` is its
  parent with `09e42fc14` in history. **Tracked tree clean** (untracked
  files are the intentional gitignored capture/log classes only).
- **Standing production arithmetic remains Stage A + N2** (promotion
  `bec291558`). **The LSA/DSA transplant is unchanged** (lineage
  `7d2289b97 → 54f06950b → 36fe02796 → 3ac389d99 → 71b14ea5a`; DSA
  two-cache live; sparse scoring graph-unreachable at ≤2048; the two
  recorded `LONGCAT_LSA_DEVIATION` de-clobbers stand). This round
  changed **no arithmetic**.

## The instrumentation-only additions (commit `09e42fc14`)

1. `lsa_indexer_k_2d` — il==0-gated dump-only `ggml_cont_2d` copy of the
   canonical 3D `lsa_indexer_k` (evaluation forced, no consumer).
2. `lsa_indexer_q_proj-0` — name-only first-owner surface: an
   il==0-gated `cb()` on the existing pre-RoPE BF16 Q-projection cast
   node (owners 2..26 deliberately unnamed; zero graph nodes).
3. `lsa_indexer_q_2d` — il==0-gated dump-only 2D copy (head-major,
   `h*128+d`) of the post-RoPE 3D `lsa_indexer_q`; exists only when
   sparse-active.
4. **Separate `LONGCAT_LSA_DUMP_DIR` dump family** in `common/debug.cpp`
   (exact-name, full-sequence): attribution anchors
   `attn_norm-0`/`q_a_norm-0` → `lsa_anchor_*_full.bin`; the owner-K
   trio; sparse-only first-owner `q_proj`/`q_2d`/`weights`; and 14
   `lsa_top_k_reuse-<odd>` entries → `lsa_top_k_owner<NN>_full.bin`
   (reuse-name keying is mandatory: the owner cb name is renamed in
   place and never survives to eval time). Standing spec tables and the
   shared writer untouched; the writer's I32 → f32 conversion is exact
   for the top-K indices (< 4608 < 2^24).
5. **44-name sweep contracts** in all four runners (added
   `LONGCAT_LSA_DUMP_DIR` everywhere) **including the F7 Type-P
   hardening** (`LONGCAT_FFN_INP2_INJECT_DIR` added to both Type-P
   scripts); `git_head` recorded in every run provenance.
6. **Committed `prompt_2050_a.txt`** = the frozen 2050 prompt, SHA
   `e2791fac7561166c1e4865db64db8726d2ccd499ccfd891efd78d5fd2c42b310`
   (4,100 B; the frozen token stream `eb04e101…` remains mtp-tree
   reference data).
7. New `run_longcat_lsa_dump_proof_512.ps1` +
   `analyze_longcat_lsa_dump_proof_512.py` (py_compile-clean, SHA
   `fd279f8c…`).

Two-phase name-collision audit CLEAN (pre-edit: all four stems unused;
post-edit: exact occurrence counts — one il==0-gated model `cb()` per new
graph name, all other references intentional spec/harness strings).

## Standing binary set (build dir `D:\llama.cpp-longcat-claude-build-cuda132`)

- `llama.dll` = **`37431a1916e5118af619defe864db63e96d2b5dd290580fa205c36737d4e2d5b`**
- `llama-common.dll` = **`39bffc906c03a59af82931cb2505735e3c8ad4e99fc24c121b6113cf77e62bd2`**
- `llama-debug.exe` = `df2a57f6…` **unchanged**
- `ggml-cuda.dll` = `502e50e8…` **unchanged**

(Movement exactly as pre-registered: only the two DLLs carrying the model
and debug edits moved.)

## Full standing inertness re-pass at `09e42fc14`: CLEAN

All runs: 44/44 sweep, child-PATH-pinned live-verified cuBLAS
6.14.11.1330, `graphs reused = 0`, placement `(29, 15, ATTN)`.

| Gate | Run | Result |
|---|---|---|
| Frozen-512 | `cpp_logits_512_lsaE/` | **PASS — logits `b8067779…` BYTE-EXACT** (0/131,072, worst ratio 0.974568646, top-1 483, stream `4893d787…`) |
| Project Gate-3 | `cpp_logits_gate3_lsaE/` | **PASS — logits `8852bd5b…` BYTE-EXACT** (0 violations, top-1 444 both sides, stream `ad9883df…`) |
| injectffn | `cpp_resid_walk_injectffn_lsaE_512/` | **ALL GATES PASS** — 15/15 standing invariants, inventory 32/32, prior reproduction 17/17 (promoN2 epoch), landing `4718460b…` byte-exact, operator `block1_ffn0_norm_full.bin` = `9815422f…` and restored `block1_attn1_norm_full.bin` = `1ce81e69…` byte-exact in the manifest; **zero `lsa_*` files** (new family dormant when unset) |

## ≤512 LSA dumpability proof: CLEAN (`cpp_lsa_dump_proof_lsaE_512/`)

- **Exactly 5 bins + 5 sidecars** (exact sizes), Type S: child env =
  pinned PATH + `LONGCAT_LSA_DUMP_DIR` only; filter/spec/inventory all
  describe the same five-surface experiment.
- **Sparse-only negative control clean** — no q_proj/q_2d/weights/top-K
  entry materialized at ≤512 (real-decode-only callback semantics
  confirmed; reserve graphs are never callback-visited).
- **Nope-half identity byte-exact** (`lsa_indexer_k[:,64:128]` ==
  `lsa_indexer_k_norm[:,64:128]`; no expectation registered for the
  roped columns — blocker territory).
- **Both attribution anchors captured successfully** with 100%
  BF16-lattice membership (incl. the real semantic check on
  `attn_norm-0`, 1,572,864/1,572,864).
- **Bonus reproduction: `lsa_anchor_q_a_norm0_full.bin` =
  `956bd3e87b02a89ad1e3dd71801decffd10103d37bade7c490836aedd384dd37`**,
  byte-identical to the standing frozen Q-trio `q_a_layernorm` surface
  through the completely independent new family — **an observation
  only, explicitly NOT promoted into a new frozen criterion**.
- Other proof hashes: k_proj `9151f585…`, k_norm `923a2379…`, k
  `91876870…`, anchor attn_norm0 `35a1939a…` (manifest committed
  evidence in the run dir).

## Standing constraints (carried forward, unchanged)

- **No 2050-token execution has occurred. Project Gate 4 remains NOT
  RUN.** No Gate-4 acceptance criterion exists (a future review
  decision); never widen any frozen criterion; FA-off only for eventual
  long-context work; runtime contract (pinned live-verified cuBLAS
  6.14.11.1330, audited 44-name sweeps, `.venv`/toolchain/reference
  trees immutable, no `.bin`/`.log`/GGUF commits) unchanged.
- **The four indexer blockers remain deliberately unresolved:**
  (1) indexer K-norm BF16 cast ordering; (2) rope/nope split/layout
  equivalence; (3) YaRN `attn_factor` behavior on the indexer RoPE;
  (4) top-K determinism (ordering / membership / mask / logits,
  separated). No arithmetic correction is authorized.

## Pre-registered determinism semantics (the 2050 comparator implements these, never re-decides them)

Per owner block, per row, across repeats:

1. **Structural top-K validity FIRST**: exactly 2048 recovered indices;
   all integral under exact `lrint` recovery; all in `[0, n_kv_lid)`
   (2,304 for the `-ub 2304` geometry, bound taken from the run); all
   2048 unique. A validity failure is a round-stopping anomaly, not a
   determinism verdict.
2. **V-ord** — raw index ordering: **characterization only** (this
   build's `ggml_top_k` is CUB `DeviceTopK::MaxPairs`, CCCL 3.2.0,
   `determinism::not_guaranteed` + `output_ordering::unsorted`).
3. **V-mem-raw** — complete raw selected set: **characterization**;
   reported split rows p ≤ 2047 vs p ∈ {2048, 2049}.
4. **V-mem-effective** — selected ∩ causally-visible: **BLOCKING**.
5. **V-mask** — reconstructed effective attention mask: **BLOCKING**.
6. **V-logit** — final logits (Type-P family): **BLOCKING**.
7. **Invisible-only raw membership variation is NONBLOCKING** (rows with
   ≤2048 visible entries necessarily fill slots from the −inf tie pool;
   ties there are attention-inert). Positive expectations: for p ≤ 2047
   the effective set equals the full causal set in every run; for rows
   2048/2049 (the only truly sparse-selective rows, reported
   separately) raw ≡ effective, |effective| = 2048, and the 1,040
   forced entries are contained.

Real-decode structural evidence at 2050 keys on `LONGCAT_LSA_AUDIT`
owner/reuse lines with **`n_kv=2304`** plus the single `mask` line
(`query_pos=2049 visible=2050 forced=1040 init_pos=[0,15]
local_pos=[1026,2049]`); reserve/fit builds print `n_kv=4608` lines in
every `-c 4608` run and are ignored.

## Next round (a plan question, NOT begun): C++ 2050 determinism measurement ONLY

**Not an HF comparison and not Gate 4.** In exactly this order:

1. **Author the Type-S runner** (`-c 4608 -b 4608 -ub 2304 -fa off
   -ctk f32 -ctv f32 --no-warmup -fitt 4096 -v`; child env = pinned PATH
   + `LONGCAT_LSA_DUMP_DIR` only; fresh dir per run; fresh `-ub 2304`
   placement tuple recorded on S1 and gated identical on every later
   run — the standing `(29, 15, ATTN)` tuple belongs to `-ub 512` only).
   **The Type-S tensor filter MUST include both attribution anchors
   (`attn_norm-0`, `q_a_norm-0`) as well as the approved LSA surfaces**
   (owner-K trio, `lsa_indexer_q_proj-0`, `lsa_indexer_q_2d-0`,
   `lsa_indexer_weights-0`, `lsa_top_k_reuse-<odd>`).
2. **Author the Type-P runner** (`--save-logits`, zero `LONGCAT_*`, same
   geometry; tokens gate `eb04e101…`, logits 524,288 B).
3. **Author the offline determinism comparator** implementing the
   pre-registered semantics above (py_compile + SHA discipline).
4. **Review the protocol before execution.**
5. **Execute S1/S2/S3 and P1/P2/P3 only if approved.**
6. **STOP FOR REVIEW.**

The **HF 2050 wrapper / first-owner semantic comparison remains a
subsequent reviewed round** (blockers 1–3 are judged there, not in the
determinism round). This handoff round contains **no arithmetic changes,
no HF wrapper implementation, no 2050 execution, and no Gate-4
criterion**.

## Fresh-session opening prompt (self-contained; copy-paste)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. Read FIRST
> `WIN11_HANDOFF_2026-08-18_DELTA_LSA_MEASUREMENT_APPARATUS.md` (the
> bootstrap delta), then reconcile in its stated order. Verify
> read-only: branch; clean tracked tree; **the session-close HEAD = the
> docs-only commit containing that bootstrap delta** (recover via
> `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_LSA_MEASUREMENT_APPARATUS.md`,
> confirm it equals `git rev-parse HEAD`) **with parent `7fe956a67` and
> `09e42fc14` (the instrumentation checkpoint) in history**; standing
> arithmetic Stage A + N2; LSA/DSA transplant unchanged; binary set
> `llama.dll 37431a19…` / `llama-common.dll 39bffc90…` / exe
> `df2a57f6…` / `ggml-cuda.dll 502e50e8…`; inertness re-pass CLEAN
> (frozen-512 `b8067779…` byte-exact, Gate-3 `8852bd5b…` byte-exact,
> injectffn 15/15 + 32/32 + 17/17 with landing `4718460b…` and operator
> `9815422f…`); ≤512 LSA dumpability proof CLEAN
> (`cpp_lsa_dump_proof_lsaE_512/`: 5 bins + 5 sidecars, negative
> control clean, nope-half identity byte-exact, anchors captured;
> `q_a_norm-0` bonus `956bd3e8…` is an observation, not a criterion);
> the four indexer blockers unresolved; Gate 4 NOT RUN; no 2050
> execution has occurred. Report any discrepancy before planning. The
> next round is **C++ 2050 determinism measurement ONLY — not an HF
> comparison and not Gate 4**: (1) author the Type-S 2050 runner
> (`-ub 2304`, `LONGCAT_LSA_DUMP_DIR` only, filter including BOTH
> attribution anchors `attn_norm-0`/`q_a_norm-0` plus the approved LSA
> surfaces, fresh placement tuple recorded then gated); (2) author the
> Type-P 2050 runner (`--save-logits`, zero `LONGCAT_*`, tokens
> `eb04e101…`); (3) author the offline determinism comparator
> implementing the PRE-REGISTERED semantics from the handoff (structural
> top-K validity first; V-ord and V-mem-raw characterization only;
> V-mem-effective/V-mask/V-logit blocking; invisible-only raw variation
> nonblocking; rows 2048/2049 reported separately); (4) STOP FOR REVIEW
> of the protocol; (5) execute S1/S2/S3 + P1/P2/P3 only if approved;
> (6) STOP FOR REVIEW. The HF 2050 wrapper / first-owner semantic
> comparison is a subsequent reviewed round. Do not change any
> arithmetic, do not fix the indexer blockers, do not run Gate 4, never
> widen any frozen criterion, and honor the full runtime contract
> (pinned live-verified cuBLAS 6.14.11.1330; audited 44-name sweeps) on
> every run. Start in Plan mode; stop for review before executing
> anything.
