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

**Immediate next action (designed, NOT begun — awaiting review):** the
narrowest **exact projection-output predecessor resets** — inject
`bf16(q_a_proj-2)` ≡ HF `q_a_proj` (and/or the KV analogue) at the
respective nodes so the LoRA norms become the next operators with
byte-exact activation inputs — together with a **reviewed decision on the
LoRA-norm eps constant** (measure 1e-6 vs 1e-5 under exact inputs before
any arithmetic change is proposed). Measurement-only; still forbidden: any
arithmetic change (incl. MLP/MoE, any generalization of block-0 A+B or the
trunk RMSNorm semantics), production FA, 2050-token runs, widening any
frozen criterion, production RoPE changes.

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
