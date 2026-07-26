# Reference generator schema v2

Schema v2 separates localization, a single fresh-process worker, and acceptance.
`core-diagnose` never creates an accepted candidate and skips greedy generation;
`core-worker` writes one staged run; `core` launches two or more workers and only
renames a fully validated staging tree after source, finite, inventory, and
cross-run gates pass. The accepted array contract remains exactly 433 arrays.

Every one of the 28 physical sub-block boundaries and every named module below
the 14-layer text trunk is checked on-device. A scoped SDPA wrapper checks Q, K,
V, floating masks, and output, and restores the original function on every exit.
The first failing tensor is atomically recorded in `first-nonfinite.json`.
`sdpa-f32` is diagnostic-only and is prohibited for acceptance.

The pinned shard manifest is `scripts/longcat-next/checkpoint-shards-v2.json`.
Its sizes and SHA-256 OIDs come from the official Hugging Face Git-LFS pointer
files at revision `0cf0631862402ff36366e513e4023d22e7e5c84c`; obtaining the
pointers does not download checkpoint payloads. Accepted workers hash and scan
all shards before loading the model.

## Windows workstation commands (`cmd.exe`)

Run from `D:\llama.cpp-longcat-next-core-spike`. Each command is offline-only and
writes a unique log. The rejected `bf16-candidate-15b7fe8c` is never targeted.

```bat
for /f %i in ('git rev-parse --short HEAD') do set GEN=%i

set HF_HUB_OFFLINE=1 && set TRANSFORMERS_OFFLINE=1 && python scripts\longcat-next\make-reference-fixtures.py --mode preflight --model-dir D:\LongCat-Next --placement auto --runtime-profile blackwell-compatible --offload-dir D:\LongCat-Next-offload > D:\LongCat-Next-reference\v2-preflight.log 2>&1

python scripts\longcat-next\make-reference-fixtures.py --mode core-diagnose --model-dir D:\LongCat-Next --output-dir D:\LongCat-Next-reference\bf16-diagnose-default-v2 --precision bf16 --placement auto --offload-dir D:\LongCat-Next-offload --attention-backend default --case eos_window_position_0 --case prompt_at_once_vs_token_at_a_time > D:\LongCat-Next-reference\bf16-diagnose-default-v2.log 2>&1

python scripts\longcat-next\make-reference-fixtures.py --mode core-diagnose --model-dir D:\LongCat-Next --output-dir D:\LongCat-Next-reference\bf16-diagnose-eager-v2 --precision bf16 --placement auto --offload-dir D:\LongCat-Next-offload --attention-backend eager --case eos_window_position_0 --case prompt_at_once_vs_token_at_a_time > D:\LongCat-Next-reference\bf16-diagnose-eager-v2.log 2>&1

python scripts\longcat-next\make-reference-fixtures.py --mode core-diagnose --model-dir D:\LongCat-Next --output-dir D:\LongCat-Next-reference\bf16-diagnose-sdpa-math-v2 --precision bf16 --placement auto --offload-dir D:\LongCat-Next-offload --attention-backend sdpa-math --case eos_window_position_0 --case prompt_at_once_vs_token_at_a_time > D:\LongCat-Next-reference\bf16-diagnose-sdpa-math-v2.log 2>&1

python scripts\longcat-next\make-reference-fixtures.py --mode core-diagnose --model-dir D:\LongCat-Next --output-dir D:\LongCat-Next-reference\bf16-diagnose-sdpa-f32-v2 --precision bf16 --placement auto --offload-dir D:\LongCat-Next-offload --attention-backend sdpa-f32 --case eos_window_position_0 --case prompt_at_once_vs_token_at_a_time > D:\LongCat-Next-reference\bf16-diagnose-sdpa-f32-v2.log 2>&1

python scripts\longcat-next\make-reference-fixtures.py --mode core --model-dir D:\LongCat-Next --output-dir D:\LongCat-Next-reference\bf16-candidate-%GEN%-default-v2 --precision bf16 --placement auto --offload-dir D:\LongCat-Next-offload --attention-backend default --repeat-count 2 > D:\LongCat-Next-reference\bf16-candidate-%GEN%-default-v2.log 2>&1

python -c "import json,pathlib,numpy as np; p=pathlib.Path(r'D:\LongCat-Next-reference\bf16-candidate-%GEN%-default-v2'); v=json.loads((p/'candidate-validation.json').read_text()); assert v['whole_candidate_finite'] and v['exact_inventory_count']==433; runs=list((p/'runs').glob('run-*')); assert len(runs)>=2; [(__import__('builtins').exec('with np.load(r/\"arrays.npz\",allow_pickle=False) as z: assert len(z.files)==433; assert all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind==\"f\")')) for r in runs]; print(json.dumps(v,indent=2))" > D:\LongCat-Next-reference\bf16-candidate-%GEN%-default-v2-validation.log 2>&1
```

The eager command is needed only if the default diagnostic fails. Acceptance
must use a finite, justified non-diagnostic backend; it must never use
`sdpa-f32`. Numerical parity is not implied by generator acceptance.
