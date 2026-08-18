# LongCat Sparse Parity — Delta Checkpoint: C++ 2050 Determinism Round (64f3199e6-parented session close)

Self-sufficient **delta** handoff for the completed 2026-08-18 C++ 2050
Type-S/Type-P determinism round — **the first sparse-active execution on
the transplanted branch**. **This is the BOOTSTRAP document for a fresh
session: read it FIRST**, then reconcile through the prior documents in
the order below. Prior handoffs remain authoritative for
machine/runtime/oracle context and are not duplicated.

## Fresh-session document read order (this file first)

1. **This delta** (bootstrap).
2. `WIN11_HANDOFF_2026-08-18_DELTA_LSA_MEASUREMENT_APPARATUS.md`
   (measurement apparatus + the pre-registered determinism semantics)
3. `WIN11_HANDOFF_2026-08-18_DELTA_LSA_TRANSPLANT.md` (LSA transplant +
   ≤2048 inertness checkpoint)
4. `WIN11_HANDOFF_2026-08-17_FROZEN512.md` (machine/runtime contract)
5. `WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`
6. `WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`
7. `WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`
8. `WIN11_HANDOFF_2026-08-18_DELTA_GATE4_READINESS.md` +
   `AUDIT_GATE4_READINESS_2026-08-18.md` (incl. the HF capture-machinery
   pointers the next round needs)
9. The 2026-08-18 addenda of `STATUS_2026-08-17.md` (the final addendum
   is this round's evidence record)
10. `NEXT_ACTION.md`
11. `CLAUDE.md` (base guardrails; superseded where later handoffs say so)

## Git / standing state

- **Branch:** `claude/longcat-win11`.
- **`2dd49d39c11a4378ebd3abed2a51aea3f575accb` = the committed 2050
  determinism protocol** (three files, authored + committed BEFORE any
  run): `run_longcat_lsa_2050_typeS.ps1` (`9f9ab9c0…`),
  `run_longcat_lsa_2050_typeP.ps1` (`c77645a5…`),
  `compare_longcat_lsa_determinism_2050.py` (`5c60d642…`,
  py_compile-clean). Every run of the round executed at this commit with
  a clean tracked tree (per-run `run_provenance.json` records `git_head`
  plus cryptographic log binding: absolute path + SHA256 + byte size of
  both saved process logs).
- **`64f3199e636909f3e8f1598d657d6b9b6cd3aefd` = the C++ determinism
  docs checkpoint** (STATUS final addendum + NEXT_ACTION update;
  parent = `2dd49d39c`). **Treat it as the parent operational/docs
  checkpoint of this handoff.**
- **The final session-close HEAD is this handoff's own docs-only
  commit**: recover it with
  `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_LSA_CPP2050_DETERMINISM.md`,
  verify it equals `git rev-parse HEAD`, and verify `64f3199e6…` is its
  parent with `2dd49d39c` in history. **Tracked tree clean** (untracked
  files are the intentional gitignored capture/log classes, now
  including the six 2050 run dirs and `lsa_determinism_2050/`).
- **Standing production arithmetic remains Stage A + N2** (promotion
  `bec291558`). **The LSA/DSA transplant and the measurement
  instrumentation are unchanged** (lineage `7d2289b97 → 54f06950b →
  36fe02796 → 3ac389d99 → 71b14ea5a`; instrumentation `09e42fc14`;
  binary set `llama.dll 37431a19…` / `llama-common.dll 39bffc90…` /
  `llama-debug.exe df2a57f6…` / `ggml-cuda.dll 502e50e8…`). This round
  changed **no arithmetic and no model/protocol code** after the
  protocol commit.

## The completed round (evidence record: the final 2026-08-18 addendum of `STATUS_2026-08-17.md`; verdict JSON: `lsa_determinism_2050/verdict.json`)

- **All six runs passed every runner gate**: `cpp_lsa_2050_S1/S2/S3`
  (Type S: eval callback; child env = pinned CUDA 13.2 PATH +
  `LONGCAT_LSA_DUMP_DIR` only; 22-surface capture ≈295.4 MiB/run) and
  `cpp_logits_2050_P1/P2/P3` (Type P: `--save-logits`, zero `LONGCAT_*`,
  diagnostic-silence gate). Every run: binary set 4/4, 44/44 sweep,
  live-verified cuBLAS **6.14.11.1330**, `graphs reused = 0`, DSA
  two-cache lines, fresh dir, manifest (45/5 lines) + provenance.
- **`-ub 2304` placement** = `set ngl_per_device[0].(n_layer, n_part,
  overflow_type)=(29, 17, UP), id_dense_start=0`, `offloaded 29/30
  layers` — **established on S1 and reproduced EXACTLY by all six runs**
  (the standing `(29, 15, ATTN)` tuple remains the `-ub 512` contract).
- **All six real sparse executions** show exactly **14 owner + 14 reuse
  `LONGCAT_LSA_AUDIT` lines at `n_kv=2304`** (blocks 0..27 in order,
  `top_k=2048`, **pointer pairing 14/14**; reserve/fit `n_kv=4608` lines
  ignored as non-evidence) and **exactly one mask line**
  `query_pos=2049 visible=2050 forced=1040 init_pos=[0,15]
  local_pos=[1026,2049]`.
- **Input stream identity proven on both families:**
  `eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed`
  (Type-S stdout-parse reconstruction of 2050 × id 483; Type-P
  `*-tokens.bin`).
- **V-input STABLE**: all eight non-top-K Type-S surfaces byte-identical
  across S1/S2/S3 (anchors `28f15cb7…`/`49d3d02d…`; owner-K trio
  `5ddd67d8…`/`57deb53c…`/`2f57bc0f…`; Q trio `a75dfb80…`/`cc8ecd1a…`;
  weights `321a15e6…`) — indexer inputs bitwise stable, so all top-K
  churn is attributable to selection mechanics.
- **Structural top-K validity PASS** for all 3×14×2050 rows
  (176,332,800 values: finite, exact-`lrint`-integral, in `[0, 2304)`,
  per-row unique). Positive expectations PASS everywhere (rows p ≤ 2047
  select their full causal set; rows 2048/2049 raw ≡ effective,
  |effective| = 2048, all 1,040 forced positions contained).
- **V-ord**: raw ordering differs in 86,100/86,100 row-comparisons —
  **characterization-only** (CUB `DeviceTopK::MaxPairs`,
  `output_ordering::unsorted` + `determinism::not_guaranteed`).
- **V-mem-raw**: **69,504/86,100** row-pair comparisons differ as raw
  sets — **all differences confined to causally invisible −inf filler
  membership in rows p ≤ 2047; ZERO visible-affecting differences**.
- **Sparse-selective rows 2048 and 2049: ZERO raw-set differences for
  every owner/pair** — the truly selective membership is bitwise
  repeatable.
- **V-mem-effective: zero differences. V-mask: all 42 owner×pair
  reconstructed-mask comparisons equal.**
- **V-logit: P1 = P2 = P3 logits SHA
  `52a95141e0b8f10135eb9e632692b22fb5d66804f5857f3da0f1b2f58990ea16`**,
  finite 131,072/131,072 each, **top-1 = 483** in all three (cross-run
  C++ determinism only; compared to nothing else — no HF oracle exists).
- **Determinism comparator: exit 0, reasons empty, no anomaly.**

## Scoped conclusion (mandatory phrasing, carried into all future docs)

**Indexer blocker 4 (top-K determinism) is DISCHARGED only for this
frozen C++ machine/runtime/`-ub 2304` protocol** (Win11 RTX PRO 6000,
pinned live-verified cuBLAS 6.14.11.1330, instrumentation binary set,
`-c 4608 -b 4608 -ub 2304 -fa off -ctk f32 -ctv f32 --no-warmup
-fitt 4096`, N=3 per family): effective membership, reconstructed masks,
and final logits are bitwise repeatable, while **raw top-K ordering
remains nondeterministic and invisible −inf-filler membership may vary**
(attention-inert by construction). **Explicitly: NOT a cross-runtime
guarantee and NOT a Gate-4 criterion.**

**Blockers 1–3 remain unresolved** (HF-semantic questions, out of this
round's scope):

1. indexer K-norm BF16 cast ordering;
2. rope/nope split/layout equivalence;
3. YaRN `attn_factor`/mscale behavior on the indexer RoPE.

## Standing constraints (carried forward, unchanged)

- **No more C++ 2050 execution is currently required.** No arithmetic
  correction is authorized. Never widen any frozen criterion.
- **FA-off remains mandatory** for all long-context work (FA-on remains
  unfixed/prohibited).
- **Gate 4 remains NOT RUN; no Gate-4 acceptance criterion exists** (a
  future review decision). HF 2050 final-row logits, once captured, may
  be **banked as a future oracle but must NOT be treated as a Gate-4
  acceptance test yet**.
- Runtime contract unchanged: child-PATH-pinned live-verified cuBLAS
  6.14.11.1330; audited 44-name sweeps; `.venv`/toolchain/reference
  trees immutable; no `.bin`/`.log`/GGUF commits; never overwrite
  canonical artifacts; frozen 2050 assets: prompt `prompt_2050_a.txt`
  `e2791fac…` (committed), token stream `eb04e101…66bed`.

## Next round (a plan question, NOT begun): HF 2050 first-owner semantic capture, then offline C++↔HF attribution

**Separately reviewed; design first, no HF execution without review.**
In exactly this order:

1. **Design the HF 2050 wrapper/capture** (frozen runtime `a3bc3161…`;
   raw-ids injection pinned to `eb04e101…`/2050 per the proven 512
   wrapper pattern `d267bf29…` and the length-agnostic Gate-3 core
   `bb82bcb6…` — see `AUDIT_GATE4_READINESS_2026-08-18.md`;
   `use_cache=False`; **fail-closed sparse-engagement asserts** — HF
   auto-engages sparse at `total_kv_len > 2048`; the `use_cache=False`
   sparse-owner path of that script family is unexercised, a recorded
   caveat): capture the first-owner indexer surfaces matching the
   S-family capture set (K path, Q path, weights, selection), plus
   optionally the final-row logits for banking.
2. **Design the offline C++↔HF analysis for blockers 1–3** against the
   byte-identical S-family captures (`cpp_lsa_2050_S1/S2/S3`): K-norm
   cast-ordering comparison on the owner-K trio; rope/nope split/layout
   equivalence on the roped columns (the ≤512 nope-half identity is
   already byte-proven); YaRN `attn_factor`/mscale behavior on the
   indexer RoPE angles. Judgement semantics are a review question —
   pre-register comparison surfaces, tolerances/expectations, and
   verdict separation BEFORE execution.
3. **STOP FOR REVIEW before executing anything HF-side.**

## Fresh-session opening prompt (self-contained; copy-paste)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. Read FIRST
> `WIN11_HANDOFF_2026-08-18_DELTA_LSA_CPP2050_DETERMINISM.md` (the
> bootstrap delta), then reconcile in its stated order. Verify
> read-only: branch; clean tracked tree; **the session-close HEAD = the
> docs-only commit containing that bootstrap delta** (recover via
> `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_LSA_CPP2050_DETERMINISM.md`,
> confirm it equals `git rev-parse HEAD`) **with parent `64f3199e6…` and
> the protocol commit `2dd49d39c` in history**; standing arithmetic
> Stage A + N2; LSA/DSA transplant + measurement instrumentation
> unchanged; binary set `llama.dll 37431a19…` / `llama-common.dll
> 39bffc90…` / exe `df2a57f6…` / `ggml-cuda.dll 502e50e8…`; the six 2050
> run dirs (`cpp_lsa_2050_S1/S2/S3`, `cpp_logits_2050_P1/P2/P3`) with
> manifests/provenance intact and `lsa_determinism_2050/verdict.json`
> present (comparator exit 0, reasons empty, no anomaly); `-ub 2304`
> placement `(29, 17, UP), id_dense_start=0`, offloaded 29/30 in all six
> provenances; V-input stable 8/8; P1=P2=P3 logits `52a95141…`, top-1
> 483; blocker 4 discharged ONLY for the frozen C++
> machine/runtime/-ub2304 protocol (raw order nondeterministic,
> invisible fillers may vary — not a cross-runtime guarantee, not a
> Gate-4 criterion); blockers 1–3 unresolved; Gate 4 NOT RUN; no HF 2050
> execution has occurred. Report any discrepancy before planning. The
> next round is **DESIGN ONLY: the HF 2050 first-owner semantic-capture
> wrapper (frozen runtime `a3bc3161…`, raw-ids injection pinned to
> `eb04e101…`/2050, `use_cache=False`, fail-closed sparse-engagement
> asserts; capture the first-owner indexer surfaces matching the
> S-family set, optionally bank final-row logits as a future oracle —
> NOT a Gate-4 acceptance test) plus the offline C++↔HF analysis design
> for blockers 1/2/3 (indexer K-norm BF16 cast ordering; rope/nope
> split/layout equivalence; YaRN attn_factor/mscale on the indexer RoPE)
> against the byte-identical S-family captures, with pre-registered
> comparison surfaces and verdict separation — then STOP FOR REVIEW
> before executing anything HF-side.** Do not execute the HF wrapper, do
> not run any new C++ 2050 execution (none is currently required), do
> not change any arithmetic, do not run Gate 4, never widen any frozen
> criterion, and honor the full runtime contract on every eventual run.
> Start in Plan mode; stop for review before executing anything.
