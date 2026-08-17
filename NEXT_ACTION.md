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

**Immediate next action:** design the block-0 RoPE precision experiment
(HF BF16 cos/sin + BF16 elementwise semantics for `q_pe`/`k_pe` at `il == 0`),
hard-gated byte-exact on the recorded S1/S2 references, then re-measure S3
with clean inputs. Still forbidden: MLP/MoE work, production FA patch,
repaired 2050 run, widening any frozen criterion.
