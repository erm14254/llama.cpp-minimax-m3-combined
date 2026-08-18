# Next Action — Staged KV Precision Experiments

Updated 2026-08-17. The previous next action here — capture the HF block-0 MLA
intermediates and locate the first true MLA divergence — is **complete and
authoritative**. Full result: `STATUS_2026-08-17.md`. In short: the Q path is
byte-exact to the HF Blackwell oracles; `kv_a_proj_with_mqa` is a pure BF16
output boundary (294,912/294,912 elements after RNE rounding); the first
genuine divergence is `kv_a_layernorm`, explained byte-exactly by HF RMSNorm
cast semantics (`bf16( bf16(x*rsqrt(var+eps)) * w )`, eps=1e-6 **from source**
— the sweep excludes 1e-5 but cannot distinguish 1e-6 from 1e-8).

## cuBLAS runtime contract — read before any run

Authoritative C++ parity runs must resolve `cublas64_13.dll` to CUDA **v13.2**
(cuBLAS **6.14.11.1330**). `ggml-cuda.dll` imports it by bare name and this
machine's PATH lists `CUDA\v13.0\bin\x64` first. Pin session-locally by
prepending `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64`
to the **child process** PATH (never the machine-wide PATH) and verify the
loaded module path + version from the live process. Wrong-runtime signature:
anchors `d0e9edc8…` / `a1c4c20c…` pass while the residual is `49d729e1…`.

## The staged experiments

**Experiment A** — one arithmetic commit, `il == 0` KV branch of
`src/models/longcat-flash-ngram.cpp` only, mirroring the accepted Q-side
pattern:

1. Full-576 BF16 roundtrip on `kv_cmpr_pe` after the GEMM, before the
   split views (HF Linear output is BF16; split happens after).
2. KV RMSNorm cast semantics: `rms_norm(1e-6)` → BF16 → F32 → `ggml_mul(w)` →
   BF16, callbacks on the post-round tensors.

Hard byte-exact gates: anchors `d0e9edc8…` / `a1c4c20c…` and the Q trio
`ddf69fe4…` / `956bd3e8…` / `4f3b647b…` unchanged, and the two KV surfaces
must equal the HF oracles:

- `kv_a_proj_with_mqa.bin` = `513390418c9877fa46286d397db7c9c9fb6408852836fb7827106acd183ceecc`
- `kv_a_layernorm.bin` = `b44cc101b03b11d96c0d9c52613f7469141dd7786b8128f93e3b7e912c550373`

**Experiment B** — only if A passes: post-scale BF16 round after
`mla_scale_kv` (HF scales in bf16; source-supported, ungated by any capture
surface). All A gates must still pass; the `o_proj`/residual delta(A→B)
isolates the scale-boundary contribution.

## Baselines

- HF attn0 residual (the target): `2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177`
- `2c804a35a0397e380d77f08d2a7ffd11fc6e4672722c31a98f7f96620c2a4a4e` is the
  **immutable old-arithmetic baseline** (provenance: `pre-gate4` frozen dir,
  committed `SHA256SUMS.txt`, `STATUS_2026-08-17.md`). It is retired as a
  pass/fail gate once KV arithmetic changes — the residual is causally
  downstream of K/V and must move.

**Update 2026-08-17 (localization complete):** Experiments A and B passed all
gates, and the measurement-only S1→S4 localization identified the surviving
`o_proj` remainder's first genuine divergence as **RoPE (H1)** — HF computes
rotary with BF16 cos/sin and BF16 elementwise arithmetic while ggml uses F32
throughout. The 512-wide compressed-KV cache input is byte-exact (S2b); the
`wo` boundary is analytically byte-exact plain bf16-linear (S4a); H3/H4
(softmax/ordering) remain bracketed inside S3, unreachable until RoPE parity
exists. See the final addendum of `STATUS_2026-08-17.md`.

**Update 2026-08-17 (frozen-512 checkpoint):** all previous next actions are
complete — R0/R1 done, attention-core mechanism closed, and the frozen
512-token criterion measured: **FAIL at 40/131,072 violations** (down from
the 2,122 of the clean pre-diagnostic F32-KV baseline `sparse_512_fa_off_f32`
= `1a8e37e2…`; the similarly named `sparse_512_fa_off` = `f39f77b6…` is the
bf16-cache variant and NOT the memo baseline). Delta owed to the aggregate
block-0 corrective stack, not A+B alone. See
`WIN11_HANDOFF_2026-08-17_FROZEN512.md` (authoritative handoff) and the final
addenda of `STATUS_2026-08-17.md`.

**Update 2026-08-17 (downstream boundary walk complete):** the zero-new-run
walk described below was executed per the approved plan
(`analyze_longcat_downstream_boundary_walk.py` = `6c600599…`,
`downstream_walk_512/downstream_boundary_walk.json` = `2e4854c7…`, both
post-review-wording-correction with numeric output verified identical; full
addendum in `STATUS_2026-08-17.md`). Harness validity: all 38 inputs
SHA-gated, both known-answer anchors reproduced digit-for-digit
(attn0_resid rel-RMSE 0.00390108; endpoint 40 violations / RMSE 0.164743901
/ cosine 0.999799076 / top-1 483). Semantic order proven from source on both
sides (recorded in the JSON), including that `logical_12 → result_norm`
spans the undumped logical layer 13 + final norm, and MTP is outside the
trunk on both sides. **Result: no single downstream defect boundary at
logical-block granularity; growth is smooth and the error direction changes
at nearly every block** — error L2 grows ×743 across 15 boundaries at
ratios 1.17-2.64 (no step > 3× median), the new-direction fraction exceeds
0.5 at 13/15 steps, and the final trunk error retains almost none of the
block-0 seed direction (cos = +0.011; a direction-persistence bound only,
NOT a causal bound). **The walk is observational and cannot distinguish
nonlinear propagation/rotation of inherited block-0 error from new local
per-block discrepancies** — that discrimination requires the causal reset
experiment below.

**Update 2026-08-17 (causal reset COMPLETE — see the final addendum of
`STATUS_2026-08-17.md`):** follow-up A was executed end-to-end with every
gate passing (causal-cut proof; instrumentation `2f827a91e`; only
`llama-common.dll` changed — other three binaries byte-identical to the
checkpoint; HF full-seq capture with 14/14 row-511 + same-pass norm gates;
control run 29/29 final-row regression; injection landing byte-exact with
15/15 upstream inertness; known-answer row-511 slice ≤ 1e-12).
**Causal result: the downstream trunk locally regenerates the dominant share
of the residual divergence under this intervention** — with a byte-exact
`logical_00` input, logical block 1 alone regenerates rel 9.05e-3
(‖e_reset‖ = 0.679 of the observed error L2 at that boundary), and with all
upstream divergence zeroed the endpoint error L2 remains at 0.773 of the
observed value (`result_norm`). **These ratios quantify the exact-upstream
counterfactual and are NOT additive downstream/upstream partitions — the
nonlinear downstream means contributions do not superpose.** First-raw and
the 0.01/0.10/0.50 crossings all coincide at `logical_01`.

**Update 2026-08-17 (block-1 sub-boundary localization COMPLETE — see the
final addendum of `STATUS_2026-08-17.md`):** executed with every gate
passing (semantic-equivalence proofs; instrumentation `d95939f49` with only
`llama-common.dll` changed; wrapper-aware 39-name env audit with recorded
19→21→23→39 reconciliation; HF layer-1 stages with input ≡ the reset oracle
byte-exactly; both re-runs reproducing their prior manifests 69/69).
**Result: first-raw = first-bf16-irreducible = `attn0_norm`
(`input_layernorm[0]` of logical layer 1), operator input byte-exact →
attribution PERMITTED, and the mechanism is closed**: HF =
`bf16(bf16(x·rsqrt(var+1e-5))·w)` reproduced 1,572,864/1,572,864 byte-exact
(eps byte-identified as 1e-5; 1e-6 excluded), C++ = plain F32-kept norm (to
≤4-ulp reduction noise), and BF16-rounding the C++ output recovers HF for
only 74.3% — the block-0 `kv_a_layernorm` cast-ordering class, now
operator-isolated at the trunk input norm. Reset-family divergence
compounds 2.36e-3 → 9.05e-3 across the block; per the
predecessor-exactness rule no attribution is made beyond `attn0_norm`.

**Update 2026-08-17 (dual reset COMPLETE — see the final addendum of
`STATUS_2026-08-17.md`):** the `attn_norm-2` predecessor reset was executed
on top of the `logical_00` reset with every gate passing (both landings
byte-exact; upstream inertness incl. 18/18 single-reset-manifest subset;
HF re-capture determinism 7/7; 40-name sweep). **Result — grid case 3: with
both block-2 attention operands exact in value, the pre-add `attn_out-2` is
BF16-IRREDUCIBLE (26.7% bf16-match, rel 6.13e-3, dense from token 0) →
block-2 attention is causally implicated, including its dtype/kernel
semantics (the C++ F32 carrier of BF16-on-lattice values is part of the
implementation under test).** Bonus closure: the HF residual-add mechanism is closed **for this frozen
full-sequence capture** by the exact-input S3 reconstruction
(`bf16(f32(HF_attn_out)+f32(resid))` byte-exact 1,572,864/1,572,864);
supporting evidence, not closure basis: S1 count == `ffn_inp-2` bf16-match
count, showing the C++ add introduces nothing beyond its operands.

**Update 2026-08-17 (block-2 MLA-internal walk COMPLETE — see the final
addendum of `STATUS_2026-08-17.md`):** executed under the dual reset with
every gate passing (root-GEMM weights raw-BF16 bit-identical; block-0
known-answer target-generation gates byte-exact; 81/81 dual-manifest
reproduction incl. the `attn_out-2` endpoint). **Causal frontier verdict:
both root projection GEMMs are BF16-REDUCIBLE from all-exact inputs
(786,432/786,432 and 294,912/294,912 after output rounding) — they are
HF-equivalent at the BF16 output boundary, but the raw C++ F32 outputs are
NOT pipeline-equivalent (later C++ norms consume the off-lattice values):
a missing/different representation boundary, not a GEMM arithmetic
failure — so no operator in the walk is both attribution-eligible and
irreducible.** The per-branch first-irreducible surfaces (`q_a_norm-2`,
`kv_a_norm-2`) sit behind those off-lattice predecessors AND a **direct
semantic parameter mismatch: C++ il ≥ 1 LoRA norms run eps = 1e-5
(`f_norm_rms_eps`) while HF uses eps = 1e-6** — recorded for review, not
attributed; its quantitative contribution relative to predecessor
representation and cast ordering is the unmeasured decomposition. RoPE surfaces
remain under the production-composite rule; the attention core under the
multi-input rule.

**Update 2026-08-17 (quad reset COMPLETE — see the final addendum of
`STATUS_2026-08-17.md`):** the exact projection-output resets were executed
on top of the dual reset with every gate passing (landings 1–4 byte-exact;
runtime HF eps gate 1e-6/1e-6 from the instantiated modules; norm weights
exact widenings; 41-name sweep). **Result (scoped: under the exact-predecessor block-2 norm intervention in
this frozen 512-token capture): both block-2 LoRA norms remain
bf16-irreducible under byte-exact activation inputs and verified weights
(rel ≈ 2.33e-3 / 2.36e-3, ~73.7% bf16-match) — attribution PERMITTED to
the norm operator composite, and the pre-registered non-additive 2×2
decomposes it: HF BF16 cast ordering is the DOMINANT MEASURED FACTOR
(~2.3e-3 under either eps) while the 1e-5-vs-1e-6 eps mismatch and the
missing projection BF16 boundary are smaller but independently real
(eps ~1e-4 F32-regime to ~8.7e-4 BF16-cast-regime; projection boundary
65.0→73.8% / 60.0→73.5% bf16-match).** HF mechanism byte-CLOSED for
`q_a_norm` (D6 786,432/786,432); for `kv_a_norm` D6 reaches
262,137/262,144 with a 7-element single-bf16-ulp residue —
**near-closure/model residue, not exact closure**. C++ side matches the
F32/eps-1e-5 model to ≤4-ulp reduction noise on both. The exact-predecessor
intervention improved the norm surfaces (65.0→73.8% and 60.0→73.5%
bf16-match) — a real but minority share; the cast+eps composite dominates.

**Update 2026-08-17 (hex reset COMPLETE — see the final addendum of
`STATUS_2026-08-17.md`):** the exact norm-output resets were executed on
top of the quad reset with every gate passing (landings 1–6 byte-exact;
q_b weight raw-BF16 bit-identical 9,437,184/9,437,184 with bias absent
both sides; `mla_scale_kv` f32-bit-identical `0x401cc471`; 42-name sweep).
**Result: both frontier operators are BF16-EQUIVALENT from all-exact
inputs and verified parameters — `q_b_proj-2` 3,145,728/3,145,728 and
`kv_cmpr_scaled-2` 262,144/262,144 after output rounding — HF-equivalent
at the BF16 output boundary / representation-boundary differences, not
operator arithmetic failures. Being raw-different, NEITHER causally
advances the frontier**; future advancement requires the narrowest
exact-output predecessor resets first (designs recorded, not executed):
exact HF `q_b_proj-2` reset before judging downstream Q/RoPE/absorption,
exact target `kv_cmpr_scaled-2` reset before judging downstream
KV/value/core. RoPE surfaces remain under the composite rule
(observational: q 2.875e-3, k 1.279e-3); `kqv_out-2` remains behind the
all-input rule (observational rel 4.540e-3, improved from 5.625e-3 under
cleaner upstream). Emerging picture (descriptive): every operator made
all-exact so far is HF-equivalent at the BF16 output boundary; the
causally established block-2 divergence machinery is missing BF16
representation boundaries + the norm composite (cast dominant, eps
secondary) — mirroring block-0 A+B without assumption transfer.

**Decision 2026-08-17 (recorded in
`WIN11_HANDOFF_2026-08-17_DELTA_PREGATE4_CAUSAL.md`): option (a) selected.**
The next action is to **prepare the reviewed narrow production arithmetic
plan** from the independently measured il≥1 MLA findings (measured BF16
representation boundaries at projection/scale outputs; HF norm cast
semantics; the C++ 1e-5 vs HF 1e-6 LoRA-norm eps mismatch) — **beginning
with a source audit to determine exactly which standard-path
operations/layers share these mechanisms before proposing any code change;
global il≥1 scope must NOT be assumed solely from the block-2
measurements. NOT begun; no arithmetic until that plan is reviewed and
approved.** The recorded predecessor-reset measurement designs (option (b))
remain available if review requests more evidence first. Still forbidden:
any arithmetic change (incl. MLP/MoE, any generalization of block-0 A+B or
the trunk RMSNorm semantics), production FA, Gate-4/2050-token runs,
widening any frozen criterion, production RoPE changes.

The executed design (for the record):

- **Causal reset at `logical_00`:** full-sequence `[512, 3072]` HF oracle
  reset at the proven `l_out-1` boundary via env-gated callback overwrite
  (`LONGCAT_RESID_INJECT_DIR`, R0 pattern, `common/debug.cpp` only), after
  a source-referenced **causal-cut proof** that no mutable upstream state
  bypasses `logical_00` (ScMoE shortcut, LSA indexer state, caches,
  skip/embedding, MTP, aux tensors).
- **Whole-sequence downstream comparison** of `logical_01…logical_13` (new
  `logical_13` = `l_out-27` boundary added) plus the full pre-filter
  result-norm surface (`h_nextn`), against a new same-pass-gated HF
  full-sequence capture; control run then injection run, identical
  recorded binary set (rebuild provenance incl. `llama-common.dll`),
  production RoPE, cuBLAS 6.14.11.1330 live-verified, graphs-reused = 0.
- **Dual stop rule:** first-raw whole-tensor divergence reported
  separately from the `||e_reset||/||e_observed||` materiality trajectory
  (0.01/0.10/0.50 first crossings; 0.10 is a conventional marker only).
  If raw onset and consequential growth are separated → stop for review,
  no automatic block selection; otherwise propose sub-boundary
  instrumentation only inside the indicated block. Stop for review.

Measurement-only MLP/MoE boundary work stays authorized; still forbidden:
MLP/MoE arithmetic (pending reviewed plan), production FA patch, 2050-token
run, widening any frozen criterion, production RoPE changes, any other new
arithmetic.

The superseded walk directive (for the record): production-angle A+B
captures (`cpp_attn0_mla_expB_512/`, `cpp_attn0_mla_attnpath_512/`)
preferred over R1 (clean-RoPE diagnostic only, different block-0 angle
state), against `pre-gate4` HF oracles (`mlp0_resid` `cf48a0ad…`,
`attn1_resid` `b4c1e5f6…`, `logical0_out` `5292e88a…`, `hf_hidden_512_v4`
logical_01…12/result_norm), with mandatory per-pair verification
(existence, SHA, shape/dtype, row/token representation,
logical-vs-physical mapping, semantic equivalence — never filename
inference).

**Update 2026-08-18 (Stage A COMPLETE — see the 2026-08-18 addendum of
`STATUS_2026-08-17.md`):** the reviewed production plan was approved (MTP
excluded; two validated stages) and Stage A (BF16 output boundaries at the
five il≥1 MLA projection/scale sites) landed with every gate passing: all
five local semantic gates byte-exact under the dual reset (A1/A4 vs HF
oracles, A2/A3/A5 vs known-answer-gated targets; quad-norm premise gates
byte-exact), block-0 invariants 14/14, and the frozen-512 endpoint moved
**40 → 1 violations** (top-1 483 agrees; worst ratio 1.059; still FAIL —
PASS requires 0; not an acceptance threshold). Current action: **Stage B**
— LoRA-norm HF cast semantics + eps 1e-6 at il≥1 (replace `build_norm` at
the two LoRA-norm sites with the byte-proven il==0 chain), then the
stage-B dual-reset chain gates (`4c979243…`, `c91991eb…` near-tie
stop-rule, `ecb70ef6…`, TB4 `d9234fa9…`, `22f1d6be…`), the frozen-512
endpoint, and after both stages the Gate-3 4-token regression (oracle
located: `D:\llama.cpp-longcat-mtp\longcat_sparse_gate3_hf_v4_logits.bin`
= `2c178ea5…`; historical run dir `longcat_sparse_gate3_default_cache_after_fix`
prompt "Hello, world!").

**Update 2026-08-18 (session complete — stage A landed and standing; stage
B executed, measured, and reverted by review; Gate 3 re-verified):** see
the four 2026-08-18 addenda of `STATUS_2026-08-17.md` and
`WIN11_HANDOFF_2026-08-18_DELTA_STAGEAB.md`. Standing production
arithmetic = **stage A** (five il≥1 MLA BF16 output boundaries; source
`458a03685`+`923fad90d`, restored by revert `11d93b56a`). Strict
frozen-512 criterion: **FAIL 1/131,072** (id 14720, ratio 1.0586, top-1
483 agrees) — from 40. **Project Gate 3 PASS re-verified (0 violations,
top-1 444); Gate 4 NOT RUN.** Stage-B evidence committed and preserved:
per-operator byte-exactness proven (q_a_norm byte-closed in-graph;
kv_a_norm == D6 model byte-exact with the documented 7-element near-tie,
token 177), endpoint regressed 1 → 96 (cancellation with the
still-uncorrected mechanisms), reverted per review.

**Next objective (a plan question, NOT begun):** close the final
violation (id 14720) and/or the wider mechanism set — candidate rounds,
each needing its own reviewed plan: (a) the cast-vs-eps endpoint bisect
of the stage-B interaction (first designed measurement); (b) trunk
RMSNorm cast semantics at il≥1 (standing prohibition; `attn_norm-2`
measured, `ffn_norm` unmeasured); (c) production RoPE angle generation;
(d) il≥1 attention-core measurement; (e) MLP/MoE per-operator
measurement (measurement-only already authorized); (f) MTP twins;
(g) stage-B re-application contingent on (b)-(e). Standing prohibitions
otherwise unchanged: no Gate-4/2050 run until 512 parity is resolved,
never widen any frozen criterion, no production FA, `.venv`/toolchain/
reference trees immutable, cuBLAS 13.2 child-runtime contract mandatory.

**Update 2026-08-18 (cast-vs-eps bisect COMPLETE — see the final
2026-08-18 addendum of `STATUS_2026-08-17.md`): the stage-B endpoint
regression is an INTERACTION effect.** Four-way frozen-512 comparison
(violations / logits): Stage A **1** (`9d8583e3…`, standing) · V-eps
**3** (`8c1be7de…`) · V-cast **1** (`43fdae39…`) · combined Stage B
**96** (`fd5cefbf…`). Both variants passed their pre-registered operator
gates (A6 max-ulp-4; D5 **byte-exact both norms** — zero near-ties at
eps 1e-5, proving the token-177 knife-edge is eps-regime-dependent).
Both variants fully reverted; standing Stage-A state restored and proven
by the third independent byte-exact `9d8583e3…` endpoint reproduction.
**Stopped for review.** Next-round candidates unchanged (see the
2026-08-18 session-close update above), now with the bisect baselines as
pre-registered expectations for any future D6 re-application.

**Update 2026-08-18 (ffn_norm causal measurement COMPLETE — see the final
2026-08-18 addendum of `STATUS_2026-08-17.md`): the second trunk
hidden-size norm role is independently validated.** Under an exact
`ffn_inp-2` predecessor (new `injectffn` harness mode, landing
`4718460b…` byte-exact, 14/14 invariants, 43-name sweep) with the
all-input rule satisfied (weight widening 3072/3072; eps verified BOTH
sides at 1e-5 — runtime-instantiated-module gate + GGUF metadata), the HF
`post_attention_layernorm[0]` mechanism is **BYTE-CLOSED:
`bf16(bf16(x·rsqrt(var+1e-5))·w)` 1,572,864/1,572,864**; C++ is the plain
F32-kept norm (A-model max-ulp 4); cross C++-vs-HF bf16-recovery 74.0%,
rel 2.312e-3 — the same class and magnitude as the closed `attn_norm-2`
(74.3%, 2.357e-3). Both trunk-norm roles are now individually measured.
Bisect wording corrected per review (strong/dominant non-additive
interaction). **Next decision (NOT begun): the reviewed trunk-norm source
audit + staged production design** (norm sites, layer coverage, MoE/dense
identity; endpoint expectations informed by the bisect interaction
finding) — production trunk-norm arithmetic remains forbidden until that
plan is reviewed and approved. All other standing prohibitions unchanged.

**Update 2026-08-18 (trunk-norm round COMPLETE — see the three final
2026-08-18 addenda of `STATUS_2026-08-17.md`):** both role stages were
executed with **byte-exact operator gates** (attn_norm-2 == `afa16c6c…`
under the l_out-1 reset; ffn_norm-2 == `9815422f…` under the ffn_inp-2
reset — the trunk-norm class is now production-proven at the operator
level for both roles), endpoints measured (Stage A 1 / N1 18 / N1+N2 17,
top-1 483 held throughout), and **both stages reverted per the final
review** — standing = Stage A, restoration proven by the fourth
byte-exact `9d8583e3…` reproduction, **Project Gate-3 PASS re-verified**
(0 violations, top-1 444, logits `2dfd730e…` byte-identical to the prior
Gate-3 run). **Next decision (NOT begun):** the remaining-mechanism
rounds — production RoPE angle generation, il≥1 attention-core
measurement, MLP/MoE per-operator measurement (measurement-only already
authorized) — each needing its own reviewed plan; the eventual full-stack
round re-applies the three proven norm-class corrections together with
those mechanisms and re-measures. All standing prohibitions unchanged
(no Gate-4/2050 until 512 parity; never widen frozen criteria; no
production FA; runtime/toolchain contracts).

**Update 2026-08-18 (N2-only factorial COMPLETE — see the final
2026-08-18 addendum of `STATUS_2026-08-17.md`): the missing cell is a
PASS.** With the proven ffn_norm HF chain applied ALONE (N1 absent,
operator gate `9815422f…` byte-exact), the strict frozen-512 endpoint
measured **0/131,072 violations — the first PASS-level result** (worst
ratio 0.9746, top-1 483, top-100 98/100, logits `b8067779…`). Completed
2×2 (violations): A=1 / N2-only=**0** / N1=18 / N1+N2=17. Continuous
factorial (float64): the interaction residual ‖I‖₂=56.86 **exceeds** the
combined effect ‖L11−L00‖₂=47.44 and is 1.31× the larger averaged main
effect — interaction dominates; counts are descriptive only.
Interpretation corrections applied per review (no attribution of the
cancellation to any particular remaining mechanism; ffn_norm is NOT
independently endpoint-regressing — its solo effect is an improvement to
PASS). The transient was reverted per the pre-registered
no-auto-promotion rule; **standing = Stage A** (fifth byte-exact
`9d8583e3…` restoration proof). **Next decision (NOT begun): the review
of whether to promote Stage A + N2 (ffn_norm) as the new standing
production state — requiring its own reviewed promotion protocol
(re-application, operator gate, endpoint reproduction of `b8067779…`,
Gate-3 regression) — versus holding Stage A.** All other standing
prohibitions unchanged (no RoPE/core/MoE/MTP arithmetic, no
Gate-4/2050 run until 512 parity is resolved and reviewed, never widen
frozen criteria).

**Update 2026-08-18 (PROMOTION COMPLETE — see the final 2026-08-18
addendum of `STATUS_2026-08-17.md` and
`WIN11_HANDOFF_2026-08-18_DELTA_N2PROMOTION.md`): Stage A + N2 is the
new standing production arithmetic.** All five promotion hard gates
passed: diff-identity `fed0370e…`, binary provenance, operator
`9815422f…` byte-exact, frozen-512 logits `b8067779…` byte-exact
(**PASS: 0/131,072, worst ratio 0.9746, top-1 483 — first PASS in the
project; NOT full-model HF parity**), Project Gate-3 PASS (0 violations,
top-1 444, standing logits `8852bd5b…`). Harness baselines graduated
(15 hard invariants incl. the standing `32134b64…`/`398de74c…` values);
future endpoint review baselines run from 0. Known uncorrected
mechanisms remain (N1 trunk attn_norm il≥1, LoRA cast+eps pair, RoPE
angles, il≥1 attention core, MLP/MoE, MTP) with no identified
interaction partner; N1/stage-B proofs preserved in history. **Gate 4
remains NOT RUN. Next decision (NOT begun): Gate-4 readiness review
(the 512-parity precondition is now satisfied as a REVIEW QUESTION, not
an automatic green light — LSA indexer semantics and long-context
behavior need their own audit), the remaining-mechanism measurement
rounds, or the full-stack faithfulness round — each with its own
reviewed plan.**

**Update 2026-08-18 (Gate-4 readiness audit COMPLETE — VERDICT: BLOCKED;
see `AUDIT_GATE4_READINESS_2026-08-18.md` and the final addendum of
`STATUS_2026-08-17.md`):** the LSA implementation is absent from the
standing branch (`98f5dd1cc` is not an ancestor; a >2048 run today would
silently run dense attention). **Gate 4 cannot be executed or
protocol-frozen against the standing branch and remains NOT RUN.**
Next decision (NOT begun; each its own reviewed plan): (a) the
**surgical LSA semantic transplant** round — feature-delta-only onto the
Stage-A+N2 standing source, ≤2048 byte-exact inertness gates
(`b8067779…`, `8852bd5b…`, `9815422f…`, 15 invariants); (b) the
**first-owner >2048 HF indexer capture** resolving the four recorded
indexer blockers (K-norm cast ordering, rope/nope layout, YaRN
attn_factor, top-K determinism probe with separated
membership/ordering/mask/logits comparison); (c) thereafter the
conditional Gate-4 protocol (review-visible future path only — 2050
assets exist, HF >2048 oracle must be created, acceptance criterion is a
review decision, FA-off-only certification). All standing prohibitions
unchanged.
