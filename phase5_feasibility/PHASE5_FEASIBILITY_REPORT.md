# Phase 5 — Full-GGUF Feasibility on RTX 3090 / 32 GB

**Date:** 2026-08-16
**Status:** complete, measured

> **Isolation notice.** Phase 5 is a hardware/IO feasibility experiment and has
> **no bearing on the LongCat parity investigation.** No LongCat arithmetic, C++
> parity instrumentation, frozen artifact, model file, or the authoritative
> Blackwell capture procedure was modified for it. Nothing here may be used to
> relax, reinterpret, or alter any parity gate. In particular the Phase 3a gate
> (`2c804a35…`) remains FAILED on this machine and the Blackwell C++ capture
> remains the priority dependency.

---

## Question

Can the full 138.26 GB (128.76 GiB) BF16 GGUF run on 24 GB VRAM + 32 GB RAM
using the normal hybrid hierarchy — as much as practical in VRAM, hot CPU-side
weights in system RAM, remainder demand-paged from NVMe via mmap, no mlock?

## Answer

**Yes. Every configuration completed a full 512-token forward pass with exit
code 0, no OOM, and no pathological thrashing.**

VRAM was never the binding constraint — peak GPU use was 8.0 GiB of 24 GB, and
in the proven configuration only ~2.9 GiB was attributable to the process.
The pressure point is **system RAM under realistic MoE routing**, where
physical memory saturates and the run becomes demand-paging bound. It degrades
gracefully rather than collapsing.

---

## Measurement method

Harness: `measure_phase5_run.ps1`
(SHA256 `8bc0e02decaf0232f14cdf260d8c32179d56a198bfb62a52b12cea421ae3373a`
before the two fixes noted below; re-hash before reuse).

- `Start-Process` with real stdout/stderr redirection — PowerShell's `2>&1` on
  native commands wraps stderr in `NativeCommandError` records and loses lines.
- Counters sampled every 250 ms from `Win32_PerfRawData_*` CIM classes and
  `Win32_OperatingSystem`, **not** `Get-Counter` paths, because this machine is
  not on an English locale and counter path names are localized.
- GPU via `nvidia-smi`, sampled system-wide with a measured idle baseline.

### Measurement caveats — read these before quoting numbers

1. **Per-process VRAM is unavailable.** `nvidia-smi --query-compute-apps` returns
   nothing for this process under WDDM, so `peak_gpu_proc_mib` is 0 in every
   report. GPU figures below are **system-wide peak minus measured idle
   baseline** (idle was 648–805 MiB across runs).
2. **Load/eval split is derived, not reported by llama.cpp.** `llama-debug` at
   `-n 0` does not emit `llama_perf` timings, and the verbose `load_tensors:` /
   buffer-size lines did not reach either captured stream. The split below uses
   llama.cpp's own log timestamp for `system_info`, which immediately precedes
   evaluation. Treat load/eval as **derived wall-clock**, not as llama.cpp's
   internal counters.
3. **OS file-cache warmth varies between runs** and is not controlled. Run A was
   effectively cold; A2 ran immediately after and was warm. This is why A2 reads
   0.164 GiB where A read 5.801 GiB for identical work.
4. **`peak_phys_used_mb` is system-wide**, including the OS file cache. Under
   mmap streaming the cache correctly expands to fill available RAM; hitting
   ~32.4/32.67 GB is expected behaviour, not a failure signal.

---

## Configurations

| Tag | Placement | Prompt |
|---|---|---|
| `narrow` | `-ngl 0` + explicit CUDA0 allow-list: `blk.0` attention + indexer, `token_embd`, `ngram_proj.*` (~855 MiB) | — |
| `hybrid` | `-ngl 99` with `ngram_embd.*`, `ffn_*_exps`, `ffn_*_shexp`, `ffn_gate_inp` overridden to `CPU` | — |
| `uniform` | frozen parity prompt, `(" a" * 512)` → 512 × token 483 | |
| `diverse` | mixed-topic English prose → 390 tokens | |

Common flags: `-c 1024 -n 0 -fa off -ctk f32 -ctv f32 --no-warmup`, mmap
enabled (default `--load-mode auto`), **no `--mlock`**.

Exact command line for the proven baseline (`B2`):

```
llama-debug.exe -m <...>-00001-of-00008.gguf -f prompt_512_a.txt
  -c 1024 -n 0 -ngl 0
  -ot "^blk\.0\.attn_(norm|q_a|q_a_norm|q_b|kv_a_mqa|kv_a_norm|k_b|v_b|output)\.weight$=CUDA0,^blk\.0\.indexer\.(attn_k|attn_q_b|k_norm|proj)\.weight$=CUDA0,^token_embd\.weight$=CUDA0,^ngram_proj\.[0-9]+\.weight$=CUDA0"
  -fa off -ctk f32 -ctv f32 --no-warmup --tensor-filter zzzz_no_such_tensor_zzzz
```

`--no-warmup` is required — warmup is on by default and runs a dummy graph.
The non-matching `--tensor-filter` is required to *disable* debug printing: an
**empty** filter list means every tensor matches, which is why the first `B`/`C`/`D`
attempts were 4–10× slower. Those runs are retained as `B`, `C`, `D` but
superseded by `B2`, `C2`, `D2`.

---

## Results

| Run | Placement / prompt | Tokens | Wall s | Load s | Eval s | tok/s | Peak GPU MiB | Peak WS MB | Peak phys MB | Peak commit MB | Hard page reads | Disk read GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | narrow / uniform, **cold**, callback on | 512 | 10.47 | — | — | — | 3615 | 15581 | 25396 | 17883 | 57 069 | 5.801 |
| A2 | narrow / uniform, warm, callback on | 512 | 5.88 | — | — | — | 3579 | 15025 | 24518 | 15670 | 288 | 0.164 |
| **B2** | **narrow / uniform, clean** | 512 | 13.00 | 1.03 | 11.97 | **42.8** | 3714 | 15841 | 21753 | 15074 | 56 440 | 12.745 |
| **D2** | **narrow / diverse, clean** | 390 | 78.92 | 0.86 | 78.06 | **5.00** | 3694 | 27305 | 32404 | 15045 | 199 494 | 56.13 |
| **C2** | **hybrid / diverse, clean** | 390 | 70.69 | 3.27 | 67.42 | **5.78** | 8041 | 27231 | 32393 | 19291 | 129 407 | 58.50 |

Idle GPU baseline 648–805 MiB. Physical RAM total 32 665 MB; commit limit
54 169 MB.

---

## Findings

### 1. It completes. VRAM is not the constraint.

Peak GPU was 8.0 GiB of 24 GB even in the VRAM-maximising hybrid layout —
about 7.2 GiB attributable. The narrow layout used ~2.9 GiB. There is
substantial unused VRAM headroom in every configuration.

### 2. The uniform parity prompt reads far less than diverse text.

The best available comparison is **B2 vs D2** — same placement, same
instrumentation, both clean runs:

| Run | Prompt | Tokens | Disk read | Hard page reads | tok/s |
|---|---|---|---|---|---|
| B2 | uniform `(" a" * 512)` | 512 | 12.745 GiB | 56 440 | 42.8 |
| D2 | diverse prose | 390 | 56.13 GiB | 199 494 | 5.00 |

That is a **~4.4× difference in disk read and ~3.5× in hard page reads**, with
D2 doing *fewer* tokens than B2 — so the direction is not an artefact of
sequence length.

**This ratio is still not a controlled measurement.** OS file-cache warmth was
not controlled between runs (caveat 3), so some unknown part of the difference
is cache state rather than prompt content. The figure is an *observed
difference between these two runs*, not an isolated causal multiplier.

The wider spread across all runs — 0.164 GiB (A2) to 58.5 GiB (C2) — must
**not** be quoted as a prompt-content effect. A2 was an explicitly warm rerun
immediately following A, and that comparison conflates cache state with prompt
content.

**Mechanism (hypothesis, consistent with the data but not isolated by it):**
all 512 tokens of the parity prompt are token ID 483, so the MoE router sees
identical inputs at every position and should select the same top-12 of 256
experts throughout, touching only a small fraction of expert weight. Diverse
text should spread routing across many more experts per layer. Confirming this
would require pinning cache state and instrumenting actual expert selection,
which was not done.

One independent corroboration that D2 genuinely streamed a large working set:
`pages_input` of 14.9 M pages × 4 KiB ≈ 58 GiB, matching its measured disk read.

**Consequence:** the earlier accidental "the full model ran in ~10 s" result was
real but is not a general feasibility figure, and should not be quoted as one.

### 3. Under realistic routing the run is paging-bound, not bandwidth-bound.

Sustained ~505–736 MiB/s against a drive rated 6.6–7.1 GB/s. The limiter is
demand-paging latency on scattered expert reads plus CPU-side compute, not
sequential throughput. Physical RAM saturates (32.4 of 32.67 GB) and the OS
file cache thrashes over a ~57 GiB working set — but degradation is graceful
and monotonic, with no stall, no OOM, and no runaway.

Commit charge peaked at 19.3 GB against a 54.2 GB limit, so there was **no
pagefile pressure**. This is file-backed mmap paging, not swap.

### 4. The hybrid layout gives a modest, real win.

`C2` vs `D2`, same diverse prompt: 70.69 s vs 78.92 s (~10% faster), 5.78 vs
5.00 tok/s, 129 k vs 199 k hard page reads — for 7.2 GiB VRAM instead of
2.9 GiB. Moving attention/norms/embeddings onto the GPU removes them from the
RAM contention, leaving more cache for expert streaming. The gain is bounded
because the bottleneck is CPU-side expert traffic that the GPU cannot absorb.

Load time rises 0.86 s → 3.27 s, which is the cost of copying ~7 GiB to VRAM.
With mmap, "model load" is address-space mapping only; the real I/O cost is
deferred into evaluation and shows up in the eval column.

---

## Practical recommendation (feasibility only)

For non-parity work on this machine, the hybrid layout is the better default:
attention, norms, embeddings and `ngram_proj` on CUDA0; `ngram_embd.*`,
`ffn_*_exps`, `ffn_*_shexp`, `ffn_gate_inp` on CPU; mmap on; no mlock. Expect
roughly 5–6 tok/s prompt processing on realistic text at 512-token scale, and
~40 tok/s only on degenerate low-entropy prompts.

There is unused VRAM headroom (~16 GB). Promoting a bounded subset of expert
tensors — e.g. the lowest-numbered blocks' `ffn_*_exps` at 4.5 GiB per even
block — is the obvious next lever. **Not pursued:** that is open-ended
performance tuning, and the parity work has priority.

---

## Artifacts

All under `phase5_feasibility/`:

- `measure_phase5_run.ps1` — harness
- `<label>.report.json` — per-run aggregate
- `<label>.samples.csv` — 250 ms time series
- `<label>.stdout.log`, `<label>.stderr.log`
- `prompt_512_diverse.txt` — diverse-routing control prompt
- `scratch_dumps_A/` — throwaway dumps from runs A/A2, written here **only** to
  keep `LONGCAT_HIDDEN_DUMP_DIR` away from the parity capture directories

No file outside `phase5_feasibility/` was created or modified by Phase 5.
