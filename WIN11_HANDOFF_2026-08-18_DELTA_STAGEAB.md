# LongCat Sparse Parity — Delta Checkpoint: Stage A/B Production Round (5e06842c3 → this commit)

Self-sufficient **delta** handoff covering the first production-arithmetic
round, executed 2026-08-18 under the user-approved reviewed plan. The base
handoff `WIN11_HANDOFF_2026-08-17_FROZEN512.md` and the pre-Gate-4 delta
`WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md` remain authoritative
for machine/runtime/oracle context and are not duplicated.

A fresh session reads, in order: the two prior handoffs → this delta →
`STATUS_2026-08-17.md` (four 2026-08-18 addenda appended) →
`NEXT_ACTION.md` → `AUDIT_MLA_PRODSCOPE_2026-08-18.md` → `CLAUDE.md`.

## Git state

- Branch `claude/longcat-win11`. This document's commit is docs-only on
  top of the session sequence; recover it with
  `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`.
- **Session commit sequence (from `5e06842c3`):** `f5802407f` scope audit ·
  `458a03685` stage-A arithmetic (five il≥1 BF16 output boundaries) ·
  `923fad90d` q_scaled dump spec · `2f4366d28` stage-A tooling
  (known-answer-gated targets, gate comparator, harness re-scope) ·
  `0d1b6c97b` stage-A results (all gates byte-exact; endpoint 40→1) ·
  `39abf9d49` stage-B arithmetic (LoRA-norm cast+eps) · `2e8824525`
  stage-B tables · `63b165b13` stage-B dual-reset gates (near-tie stop) ·
  `750fc9eb5` stage-B endpoint 96 (stop) · `11d93b56a` **revert of
  39abf9d49** (review decision) · `308c72e99` post-revert tables · this
  docs commit.
- **Standing production arithmetic = stage A**: source byte-identical to
  `458a03685`+`923fad90d` (verified 0-line diff after the revert).
  Binaries: exe `df2a57f6…`, `llama-common.dll` `9367c541…`,
  `ggml-cuda.dll` `502e50e8…`, `llama.dll` `99ad8993…` (post-revert
  recompile of byte-identical source; the original stage-A build was
  `c890671e…` — MSVC timestamp embedding; **functional identity proven by
  byte-exact endpoint reproduction**, below).

## What the session established (all committed with artifacts)

1. **Scope audit** (`AUDIT_MLA_PRODSCOPE_2026-08-18.md`): line-cited
   C++/HF maps proving the three measured MLA mechanisms are structurally
   shared by trunk il 1..27 (HF eps split follows tensor width; layers
   constructed identically; MTP a separate un-corrected C++ copy —
   excluded by user decision). Graph mechanics closed with measured
   numbers (sparse-arch node capacity 4312 vs 2192 current).
2. **Stage A (STANDING): five il≥1 BF16 output-boundary sites** — A1
   `wq_a`, A2 `wq_b`, A3 post-`mla_scale_q` (source-audit-derived +
   block-0-known-answer-supported), A4 full-576 `kv_cmpr_pe`, A5
   post-`mla_scale_kv`; il==0 branches preserved literally; F32 GEMM
   operands (the measured configuration). **All five local semantic gates
   byte-exact** under the dual reset (`32173b18…`, `28ea5b52…`, T3
   `c0e3536a…`, T4 `15231699…`, T5 `efef3bc0…`) plus both quad-norm
   premise gates (`2b600082…`, `93d7442a…`). **Frozen-512 endpoint:
   40 → 1 violation** (id 14720, ratio 1.0586; top-1 483; logits
   `9d8583e3…`).
3. **Stage B (EXECUTED, MEASURED, REVERTED)**: LoRA-norm HF cast + eps
   1e-6 at il≥1. Per-operator: **byte-provably more HF-faithful than
   stage A** — `q_a_norm` byte-closed in-graph vs HF `4c979243…`;
   `kv_a_norm` **byte-identical to the offline D6 model (262,144/262,144)**
   with the documented 7-element one-ulp near-tie vs HF (token 177,
   positions identical to the quad-reset model residue; irreducible
   without bit-matching HF's f32 reduction); `q_b`/`q_scaled` byte-exact
   through the exact chain. **Endpoint: 1 → 96 violations** (top-1 held)
   → mandatory review → **user decision: revert**. Committed
   interpretation: direct evidence of error cancellation between the
   F32-kept il≥1 LoRA norms and the still-uncorrected mechanisms
   (trunk-norm cast, RoPE angles, attention core, MLP/MoE); stage-B
   semantics are expected to be REQUIRED again once those are corrected —
   re-measure, never assume.
4. **Revert proven byte-exact**: the post-revert endpoint reproduces the
   stage-A logits `9d8583e3…` **byte-identically**
   (`cpp_logits_512_stageArevert/`).
5. **Project Gate 3 re-verified: PASS** (0/131,072, top-1 444 both sides;
   4-token stream `ad9883df…` reproduced; oracle located and hashed
   `2c178ea5…`; reconstruction recorded in `run_longcat_gate3_4tok.ps1`).
   **Gate 4 remains NOT RUN.**
6. All block-0 byte-exact invariants preserved throughout (14/14 +
   landings in every harness run).

## Gates standing (project terminology preserved)

- **Project Gate 3 (≤2048 exact/full-attention regression): PASS** —
  re-verified this session against the standing arithmetic.
- **Strict frozen-512 production-style criterion: FAIL at 1/131,072**
  (top-1 agrees), down from 40. The criterion is unchanged and never
  widened; PASS is claimed only at 0. Neither 40 nor 1 is an acceptance
  threshold.
- **Project Gate 4 (>2048 true LSA): NOT RUN.**

## Tooling added (committed; py_compile + SHAs recorded in the manifests)

`make_longcat_stageA_targets.py` (`7f2622d8…`) ·
`analyze_longcat_stage_gates.py` (`7c304f63…`) ·
`run_longcat_frozen512_production.ps1` (reconstructed recorded invocation;
comparator FAIL-verdict exit handled; case-sensitive diagnostic-activity
check) · `run_longcat_gate3_4tok.ps1` · harness
`run_longcat_resid_walk_512.ps1` re-scoped for the il≥1 era
(expected-moved `ffn_inp-1`, invariant-subset reproduction gates, 31-dump
inventory). Two runner post-processing bugs were found and fixed mid-run
with untouched outputs completed manually in scripted order — recorded
transparently in the provenance sidecars and STATUS.

## Guardrails delta

- Everything from the prior handoffs carries over unchanged (cuBLAS 13.2
  child-runtime contract, 42-name sweep, `.venv`/toolchain/reference-tree
  immutability, no Gate-4/2050 run, never widen frozen criteria, no
  production FA, no RoPE/core/MLP-MoE arithmetic).
- The harness `$expectedBins` table now tracks the standing build; the
  historical 15-file upstream set is split 14 invariant + `ffn_inp-1`
  expected-moved (recorded, not gated).
- New standing baselines: stage-A logits `9d8583e3…` (byte-reproduced
  twice), stage-A dual-reset manifest (`cpp_resid_walk_inject2_stageA_512/`),
  Gate-3 standing manifest (`cpp_logits_gate3_standing/`).

## Next objective (plan question, NOT begun)

Close the final violation (id 14720) / the wider mechanism set. Candidate
rounds, each requiring its own reviewed plan: (a) cast-vs-eps bisect of
the stage-B interaction; (b) trunk RMSNorm cast semantics at il≥1
(prohibition stands; `ffn_norm` unmeasured); (c) production RoPE angle
generation; (d) il≥1 attention-core measurement; (e) MLP/MoE per-operator
measurement (measurement-only already authorized); (f) MTP twins; (g)
stage-B re-application contingent on (b)–(e).

## Fresh-session opening prompt (self-contained)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. Read, in order:
> `WIN11_HANDOFF_2026-08-17_FROZEN512.md`,
> `WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`,
> `WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`, the 2026-08-18 addenda of
> `STATUS_2026-08-17.md`, `NEXT_ACTION.md`,
> `AUDIT_MLA_PRODSCOPE_2026-08-18.md`, `CLAUDE.md`. Verify the branch, a
> clean tracked tree, and that HEAD is the docs-only commit containing
> the 2026-08-18 delta (recover via
> `git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`),
> with `11d93b56a` (the stage-B revert) in its parent history. Standing:
> production arithmetic = stage A (five il≥1 MLA BF16 output boundaries);
> strict frozen-512 criterion FAIL 1/131,072 (id 14720, top-1 483
> agrees, logits `9d8583e3…`); project Gate 3 PASS (re-verified); Gate 4
> NOT RUN; stage B reverted after per-operator byte-exact validation but
> endpoint regression 1→96 (cancellation evidence, committed). Runtime
> contract: pinned child-PATH cuBLAS 6.14.11.1330 live-verified; 42-name
> sweep; all frozen gates and prohibitions unchanged. Next: prepare the
> reviewed plan for the next mechanism round (see NEXT_ACTION options
> (a)–(g)); no arithmetic until that plan is reviewed and approved. Start
> in Plan mode; stop for review before executing.
