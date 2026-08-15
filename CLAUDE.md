# CLAUDE.md — LongCat Sparse llama.cpp Diagnostic Handoff

Read `HANDOFF_MEMORANDUM_2026-08-15.md` and `NEXT_ACTION.md` before changing code.

## Mission

Continue the LongCat-Flash-Lite-Sparse llama.cpp parity work directly on the user's Windows filesystem. Minimize manual user copy/paste: inspect files, run bounded commands, edit, build, execute diagnostics, and summarize results yourself.

## Working checkout

`D:\llama.cpp-longcat-pre-gate4`

Build:

`D:\llama.cpp-longcat-pre-gate4-build-cuda132`

Expected handoff SHAs:

- `src/models/longcat-flash-ngram.cpp` = `aaff66b65e5fc4ca245cfe6b379a60b6bfae268b94cf5b69f0dfd7ca10486cf1`
- `common/debug.cpp` = `ee673463453c3c7f39de4d43a778551c7db97f8ee42bd0e936ddffd3994c3fc4`

If local SHA differs, inspect local diff/history first. Do **not** blindly restore.

## Current diagnostic state

The source intentionally contains:

1. N-gram BF16 intermediate rounding + restore-F32 widening; proven exact at `inp_embd_ngram`.
2. Physical block-0 main attention RMSNorm HF precision semantics; proven exact at `attn_norm-0`.
3. Physical block-0 Q-A/KV-A epsilon `1e-6` diagnostic.
4. Physical block-0 Q-side BF16 semantic diagnostic.

The current Q-BF16 state is diagnostic and has worse final residual parity than the best exact-main-norm baseline. Do not promote it to production.

## Frozen vectors

HF exact input: `d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f`

HF exact block0 main RMSNorm: `a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af`

HF attn0 residual: `2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177`

Best exact-main-norm C++ residual: `8ea9b911d4810982af4186e66562cb5f316e7a0a9c2439101f6654eb10887dfd`

HF-epsilon-only residual: `c2b8473b9d044ba50a978e7249a694b81f111cd5bc434b585ecd776a922c2199`

Current Q-BF16 residual: `2c804a35a0397e380d77f08d2a7ffd11fc6e4672722c31a98f7f96620c2a4a4e`

## Immediate next action

Do **not** guess another BF16 patch.

Capture actual HF full-sequence 512-token block-0 MLA intermediate outputs:

- q_a_proj
- q_a_layernorm
- q_b_proj
- kv_a_proj_with_mqa
- kv_a_layernorm
- o_proj

Use frozen runtime SHA `a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428`, exact 512 tokens of ID 483, token-stream SHA `4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c`, BF16 CUDA, TF32 disabled, `use_cache=False`.

Then add **callback-only** C++ dumps for matching boundaries and locate the first true MLA divergence. Do not change arithmetic during stage capture.

## Guardrails

- Never widen frozen logit criterion: every logit must satisfy `abs(cpp-hf) <= 0.5 + 0.05*abs(hf)` and top1 must agree.
- No repaired 2050 run until 512 numerical parity is resolved.
- No production FA patch yet.
- Preserve real LSA, CLI, and one-physical-block/three-conceptual-step MTP semantics.
- Never set `NEXTN_PREDICT_LAYERS=3`.
- No tokenizer regex fix unless proven necessary.
- Never overwrite canonical/hard-linked Safetensors.
- Do not rewrite/reset `handoff/longcat-sparse-gate4-wip-20260814`.
- Do not commit `.bin`, `.log`, GGUF, Safetensors, or build outputs.
- Generated Python: record SHA256 and `python -m py_compile`.
- Prefer fail-fast Windows PowerShell 5.1.
- Ask the user only for facts that cannot be established directly from the local machine/repo.

## Git branch for this diagnostic handoff

`handoff/longcat-parity-diagnostics-20260815`

## User workflow preference

The user is switching to Claude Code Desktop specifically to avoid manually copying long command blocks. If filesystem/shell access is available, execute the work directly rather than asking the user to relay routine outputs.
