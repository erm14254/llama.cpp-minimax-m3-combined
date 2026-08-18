# LongCat Sparse Parity — Delta Checkpoint: Stage A + N2 Promotion (5147ce878 → this commit)

Self-sufficient **delta** handoff for the 2026-08-18 promotion. Prior
handoffs (`WIN11_HANDOFF_2026-08-17_FROZEN512.md`,
`WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`,
`WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`) remain authoritative for
machine/runtime/oracle context. Read order for a fresh session: the three
prior handoffs → this delta → the 2026-08-18 addenda of
`STATUS_2026-08-17.md` → `NEXT_ACTION.md` → `AUDIT_MLA_PRODSCOPE_2026-08-18.md`
→ `CLAUDE.md`.

## Standing state (the headline)

**Standing production arithmetic: Stage A + N2** — the five il≥1 MLA BF16
output boundaries (`458a03685`) plus trunk `ffn_norm` HF cast semantics at
il=0..27, eps 1e-5 (promotion commit `bec291558`, diff-identity
`fed0370e…` to the measured transient).

- **Strict frozen-512 criterion: PASS for the standing state** —
  0/131,072 violations, worst ratio 0.9746, top-1 483, logits
  `b8067779…` (byte-exact reproduction of the measured factorial cell).
  **First PASS in the project. NOT full-model HF parity** (see scope
  statement below).
- **Project Gate 3: PASS** — 0/131,072, top-1 = 444, standing Gate-3
  logits `8852bd5b…`.
- **Project Gate 4 (>2048 true LSA): NOT RUN.**

## How it was established (2026-08-18 session sequence, all committed)

Stage A/B round → cast-vs-eps bisect (stage-B 1→96 is a strong/dominant
non-additive interaction) → ffn_norm causal closure (`9815422f…` capture;
D byte-closure at eps 1e-5) → trunk-norm N1/N2 round (both roles
byte-exact at the operator level; N1 18, N1+N2 17; both reverted) →
N2-only factorial completion (**the missing cell: 0/131,072 PASS**;
continuous interaction residual ‖I‖₂ = 56.86 > ‖L11−L00‖₂ = 47.44,
dominating both averaged main effects) → promotion with five hard gates
(diff-identity `fed0370e…`; binary provenance; operator `9815422f…`
byte-exact; frozen logits `b8067779…` byte-exact; Gate-3 PASS).
Completed 2×2 (violations): A=1 / **N2-only=0** / N1=18 / N1+N2=17.

## Scope statement (mandatory phrasing)

The frozen-512 PASS is a statement about the frozen 512-token criterion
only. **Known uncorrected mechanisms remain:** trunk `attn_norm` HF cast
semantics at il≥1 (N1 — byte-exact-proven, endpoint-interacting,
preserved in history at `720e134a3`), the LoRA-norm cast+eps pair
(stage B, `39abf9d49`), production RoPE angle generation, the il≥1
attention core, MLP/MoE per-operator semantics, the MTP twins.
**No particular remaining mechanism has been identified as the
cancellation/interaction partner** of the proven-faithful corrections.

## Standing gates & baselines

- Frozen-512 comparator `6976fbc0…` vs oracle `8825d92d…`; standing
  logits `b8067779…`; future endpoint review baselines run from 0; the
  criterion is never widened.
- Gate-3 runner `run_longcat_gate3_4tok.ps1` (oracle `2c178ea5…`, prompt
  "Hello, world!" → `ad9883df…`); standing Gate-3 logits `8852bd5b…`.
- Harness `run_longcat_resid_walk_512.ps1`: **15 hard upstream
  invariants** — the 13 block-0 surfaces plus the graduated standing
  Stage-A+N2 values `logical0_mlp0_resid.bin` = `32134b64…` and
  `logical0_attn1_resid.bin` = `398de74c…` — all landing/oracle gates
  unchanged; `$expectedMovedSurfaces` empty (future experiments
  reclassify causally as needed).
- Binary set (promotion build): exe `df2a57f6…`, `llama-common.dll`
  `261f08a5…`, `llama.dll` `15543e91…`, `ggml-cuda.dll` `502e50e8…`.
- Runtime contract unchanged: child-PATH-pinned live-verified cuBLAS
  6.14.11.1330; 43-name env sweep; placement `(29, 15, ATTN)`;
  `graphs reused = 0`; `.venv`/toolchain/reference trees immutable.

## Next objective (a plan question, NOT begun)

Candidate rounds, each requiring its own reviewed plan: (a) Gate-4
readiness review (the standing 512-parity PASS satisfies the historical
"no 2050 run until 512 parity is resolved" precondition **as a question
for review, not an automatic green light** — LSA indexer semantics,
`indexer_k_norm_eps` wiring, and FA-off long-context behavior need their
own audit first); (b) the remaining-mechanism measurement rounds (RoPE
angle generation, il≥1 attention core, MLP/MoE per-operator —
measurement-only already authorized for MoE); (c) the eventual full-stack
faithfulness round (re-applying N1 + stage-B semantics together with
whatever mechanisms the interaction analysis identifies, re-measured
never assumed).

## Fresh-session opening prompt (self-contained)

> Continue the LongCat-Flash-Lite-Sparse llama.cpp parity investigation on
> this Windows 11 / RTX PRO 6000 workstation. Writable: repo
> `D:\llama.cpp-longcat-claude` (branch `claude/longcat-win11`) and build
> dir `D:\llama.cpp-longcat-claude-build-cuda132`; everything else under
> `D:\` related to this project is read-only reference. Read, in order:
> the three prior handoffs, `WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`,
> the 2026-08-18 addenda of `STATUS_2026-08-17.md`, `NEXT_ACTION.md`,
> `AUDIT_MLA_PRODSCOPE_2026-08-18.md`, `CLAUDE.md`. Verify the branch, a
> clean tracked tree, and that HEAD is the docs-only commit containing
> this delta (`git log -1 --format=%H -- WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`),
> with the promotion commit `bec291558` in its parent history. Standing:
> production arithmetic = Stage A + N2; strict frozen-512 **PASS**
> (0/131,072, top-1 483, logits `b8067779…` — NOT full-model HF parity;
> known uncorrected mechanisms remain and no interaction partner has been
> identified); Project Gate 3 PASS (`8852bd5b…`); Gate 4 NOT RUN. Runtime
> contract: pinned child-PATH cuBLAS 6.14.11.1330 live-verified; 43-name
> sweep; all frozen gates and prohibitions unchanged; the frozen-512
> criterion is never widened and future endpoint baselines run from 0.
> Next: prepare the reviewed plan for the next round (Gate-4 readiness
> review, remaining-mechanism measurements, or the full-stack
> faithfulness round — see NEXT_ACTION); no arithmetic until that plan is
> reviewed and approved. Start in Plan mode; stop for review before
> executing.
