# Next Action — Locate First MLA Divergence

## Goal

Two upstream boundaries are already exact to HF:

1. `inp_embd_ngram`
2. physical block-0 `attn_norm-0`

The next error is inside MLA. Stop guessing precision patches and measure the real intermediate tensors.

## HF full-sequence 512-token surfaces

Capture final-token outputs from `model.model.layers[0].self_attn[0]` for:

- `q_a_proj`
- `q_a_layernorm`
- `q_b_proj`
- `kv_a_proj_with_mqa`
- `kv_a_layernorm`
- `o_proj`

Requirements:

- model: `D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved`
- frozen runtime SHA: `a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428`
- sequence: 512 copies of raw token ID 483
- token-stream SHA: `4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c`
- model dtype BF16 on CUDA
- TF32 disabled
- `use_cache=False`
- capture full-sequence module outputs, then save only final position as little-endian F32

A draft `capture_longcat_hf_attn0_mla_stages.py` may exist locally. It had not been executed at handoff. Inspect/hash/`py_compile` it before use; do not assume a SHA.

## After HF capture

Add callback-only C++ surfaces for corresponding Q/KV stages. Do not alter arithmetic in the same step.

Interpret first divergence:

- `q_a_proj` diverges: Q-A weight/input/GEMM rounding.
- q_a projection matches, Q-A norm diverges: epsilon/precision boundary.
- Q path matches through q_b: compressed KV path next.
- Q/KV pre-attention stages match: RoPE/scaling/absorption/attention/o_proj.

## Frozen residual baselines

HF attn0 residual:

`2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177`

Best exact-main-norm C++:

`8ea9b911d4810982af4186e66562cb5f316e7a0a9c2439101f6654eb10887dfd`

HF-epsilon-only C++:

`c2b8473b9d044ba50a978e7249a694b81f111cd5bc434b585ecd776a922c2199`

Current Q-BF16 diagnostic C++:

`2c804a35a0397e380d77f08d2a7ffd11fc6e4672722c31a98f7f96620c2a4a4e`
