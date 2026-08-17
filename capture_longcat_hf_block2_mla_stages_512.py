#!/usr/bin/env python3
"""HF block-2 MLA internals capture (layers[1].self_attn[0], full-sequence).

Pure module hooks on one trunk forward — no forward re-implementation.
Captured surfaces (all full-sequence, f32-widened, token-major <f4):

  q_a_proj_input (3072)   pre-hook on q_a_proj — MUST equal the attn0_norm
                          oracle byte-exactly (the dual-reset input gate)
  q_a_proj (1536), q_a_layernorm (1536), q_b_proj (6144)
  kv_a_proj_with_mqa (576), kv_a_layernorm (512)
  attn_o_input (4096)     pre-hook on o_proj — the pre-wo attention context
  o_proj_out (3072)       MUST equal the committed attn0_out (c90c8e06...)
  rope_cos / rope_sin (64) from trunk.rotary_emb — MUST equal the committed
                          block-0 capture files byte-exactly (positions and
                          config are layer-independent; verified, not assumed)

Fail-closed; measurement-only; no model or runtime modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

EXPECTED_RUNTIME_SHA256 = "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
EXPECTED_TOKEN_SHA256 = "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
EXPECTED_ORACLE2_SHA256 = "afa16c6c3324387e9261c708cae044b8fcb08acda8c8f6315d2ba8d39a8f0fd7"
EXPECTED_ATTN0_OUT_SHA256 = "c90c8e0669b9261f3bfa21abc1cc7f4f7fae48ee4393755ad99ed9e7c1a5e2e9"
EXPECTED_ROPE_COS_SHA256 = "8771da1ea77d102e07bbc08064e6da6226ab4c2cb2a195c25d197c35487d9bb2"
EXPECTED_ROPE_SIN_SHA256 = "5c5dede92d05f23dfab1a27285685f61e5596df888dd429fae6d8b6591b7ff0a"
EXPECTED_TOKEN_COUNT = 512
VOCAB_SIZE = 131072

WIDTHS = {
    "q_a_proj_input": 3072,
    "q_a_proj": 1536,
    "q_a_layernorm": 1536,
    "q_b_proj": 6144,
    "kv_a_proj_with_mqa": 576,
    "kv_a_layernorm": 512,
    "attn_o_input": 4096,
    "o_proj_out": 3072,
    "rope_cos": 64,
    "rope_sin": 64,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-bin", required=True)
    ap.add_argument("--oracle2-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    tokens_bin = Path(ns.tokens_bin).resolve()
    oracle2_bin = Path(ns.oracle2_bin).resolve()
    out_dir = Path(ns.out_dir).resolve()

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set - refusing to run")

    runtime = model_dir / "modeling_longcat_flash_sparse.py"
    for p, what in ((model_dir, "model dir"), (tokens_bin, "tokens"), (oracle2_bin, "oracle2"), (runtime, "runtime")):
        if not p.exists():
            stop(f"{what} missing: {p}")

    runtime_sha = sha256_file(runtime)
    token_sha = sha256_file(tokens_bin)
    oracle2_sha = sha256_file(oracle2_bin)
    print(f"runtime_sha256={runtime_sha}")
    print(f"tokens_bin_sha256={token_sha}")
    print(f"oracle2_sha256={oracle2_sha}")
    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop("frozen runtime SHA mismatch")
    if token_sha != EXPECTED_TOKEN_SHA256:
        stop("authoritative token SHA mismatch")
    if oracle2_sha != EXPECTED_ORACLE2_SHA256:
        stop("attn0_norm oracle SHA mismatch")

    raw = tokens_bin.read_bytes()
    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop("token file size")
    ids = list(struct.unpack(f"<{EXPECTED_TOKEN_COUNT}i", raw))
    for i, tid in enumerate(ids):
        if not 0 <= tid < VOCAB_SIZE:
            stop(f"token {i} out of range")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    print(f"torch={torch.__version__} transformers={transformers.__version__}")

    oracle2 = np.frombuffer(oracle2_bin.read_bytes(), dtype="<f4").reshape(
        EXPECTED_TOKEN_COUNT, 3072
    )

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"load_seconds={time.perf_counter() - t0:.3f}")

    trunk = model.model
    if len(trunk.layers) != 14:
        stop("unexpected layer count")
    attn = trunk.layers[1].self_attn[0]
    for mod_name in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "o_proj"):
        if not hasattr(attn, mod_name):
            stop(f"attention module missing: {mod_name}")
    if not hasattr(trunk, "rotary_emb"):
        stop("trunk has no rotary_emb")

    caps: dict[str, torch.Tensor] = {}

    def once(name, t):
        if name in caps:
            stop(f"hook fired twice: {name}")
        caps[name] = t.detach().clone()

    def out_hook(name):
        def h(_m, _i, output):
            once(name, output)
        return h

    def pre_hook(name):
        def h(_m, inputs):
            once(name, inputs[0])
        return h

    def rope_hook(_m, _i, output):
        once("rope_cos", output[0])
        once("rope_sin", output[1])

    handles = [
        attn.q_a_proj.register_forward_pre_hook(pre_hook("q_a_proj_input")),
        attn.q_a_proj.register_forward_hook(out_hook("q_a_proj")),
        attn.q_a_layernorm.register_forward_hook(out_hook("q_a_layernorm")),
        attn.q_b_proj.register_forward_hook(out_hook("q_b_proj")),
        attn.kv_a_proj_with_mqa.register_forward_hook(out_hook("kv_a_proj_with_mqa")),
        attn.kv_a_layernorm.register_forward_hook(out_hook("kv_a_layernorm")),
        attn.o_proj.register_forward_pre_hook(pre_hook("attn_o_input")),
        attn.o_proj.register_forward_hook(out_hook("o_proj_out")),
        trunk.rotary_emb.register_forward_hook(rope_hook),
    ]

    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
    try:
        t1 = time.perf_counter()
        with torch.inference_mode():
            trunk(input_ids=input_ids, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        print(f"forward_seconds={time.perf_counter() - t1:.3f}")
    finally:
        for h in handles:
            h.remove()

    missing = [n for n in WIDTHS if n not in caps]
    if missing:
        stop(f"hooks did not fire: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": "HF block-2 MLA internals (layers[1].self_attn[0], full-sequence, dual-reset comparanda)",
        "runtime_sha256": runtime_sha,
        "tokens_bin_sha256": token_sha,
        "oracle2_sha256": oracle2_sha,
        "sequence_length": EXPECTED_TOKEN_COUNT,
        "layout": "token-major [512, width] float32-le",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tf32_disabled": True,
        "surfaces": {},
    }
    sums_lines = []
    arrays: dict[str, np.ndarray] = {}
    for name, width in WIDTHS.items():
        t = caps[name]
        if t.dim() == 3:
            if tuple(t.shape) != (1, EXPECTED_TOKEN_COUNT, width):
                stop(f"{name}: unexpected shape {tuple(t.shape)}")
            v = t[0]
        else:
            stop(f"{name}: unexpected rank {t.dim()} shape {tuple(t.shape)}")
        v = v.float().detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        if not np.isfinite(v).all():
            stop(f"{name}: non-finite")
        arrays[name] = v
        path = out_dir / f"{name}.bin"
        path.write_bytes(v.tobytes())
        sha = sha256_file(path)
        summary["surfaces"][name] = {
            "sha256": sha,
            "min": float(v.min()),
            "max": float(v.max()),
            "rms": float(np.sqrt(np.mean(v.astype(np.float64) ** 2))),
        }
        sums_lines.append(f"{sha}  {name}.bin")
        print(f"{name}: sha256={sha}")

    # Fail-closed identity gates.
    if not np.array_equal(arrays["q_a_proj_input"], oracle2):
        stop("input gate FAIL: q_a_proj input != attn0_norm oracle")
    print("input gate: q_a_proj input == attn0_norm oracle (byte-exact) PASS")
    if summary["surfaces"]["o_proj_out"]["sha256"] != EXPECTED_ATTN0_OUT_SHA256:
        stop("endpoint gate FAIL: o_proj_out != committed attn0_out")
    print("endpoint gate: o_proj_out == committed attn0_out PASS")
    if summary["surfaces"]["rope_cos"]["sha256"] != EXPECTED_ROPE_COS_SHA256:
        stop("rope_cos identity gate FAIL")
    if summary["surfaces"]["rope_sin"]["sha256"] != EXPECTED_ROPE_SIN_SHA256:
        stop("rope_sin identity gate FAIL")
    print("rope cos/sin identity gates: == committed block-0 capture PASS")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sums_lines.append(f"{sha256_file(out_dir / 'summary.json')}  summary.json")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    print("HF BLOCK2 MLA CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
