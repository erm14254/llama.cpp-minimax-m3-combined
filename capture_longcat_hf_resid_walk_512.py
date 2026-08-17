#!/usr/bin/env python3
"""HF full-sequence residual-walk capture for the causal-reset experiment.

Captures, from ONE trunk forward of the frozen LongCat-Flash-Lite-Sparse
runtime (SHA-gated) on the authoritative 512-token stream (SHA-gated):

  logical_00_oracle.bin   hidden_states[1]  = output of logical layer 0
                          (the [512, 3072] causal-reset oracle)
  logical_01..12.bin      hidden_states[2..13] = outputs of logical layers 1..12
  logical_13.bin          forward-hook output of trunk.layers[13]
                          (pre-final-norm; NOT present in output_hidden_states)
  result_norm_full.bin    hidden_states[14] = post-final-norm

All full-sequence [512, 3072] float32-LE, token-major.

Fail-closed identity gates (any failure aborts, nothing partial is trusted):
  - runtime + token-stream SHA256 gates (frozen values)
  - row 511 of every hidden-state surface must byte-equal the committed
    final-row oracles of hf_hidden_512_v4 (SHA table below); logical_13 has
    no prior and is recorded fresh
  - same-pass final-norm hooks: norm INPUT must byte-equal the layers[13]
    hook output, and norm OUTPUT must byte-equal hidden_states[14]
    (no norm re-run)

Measurement-only; no model or runtime modification.
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
EXPECTED_TOKEN_COUNT = 512
EXPECTED_HIDDEN = 3072
EXPECTED_HIDDEN_STATES = 15
VOCAB_SIZE = 131072

# Committed final-row oracles (pre-gate4\hf_hidden_512_v4\summary.json): row
# 511 of each full-sequence surface must hash to these exactly.
FINAL_ROW_SHA = {
    "logical_00": "5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff",
    "logical_01": "48d23a758c0a7e806c2521560bf0b6245f3e6031039f53c7b0a4169c5fc426a2",
    "logical_02": "e8bf6f3a9019fff470fe57bb0e0fde1377d366a14e9a6855b46f3ef2534ae895",
    "logical_03": "fd40a04bb0b2f8cd17193cb04f75737459435915203962b1f8d764906d05a785",
    "logical_04": "f61901209f3884d5db7dda5204b41d17de7760148ab6318a4f3af282085a70b9",
    "logical_05": "5fec605d82963e8f7859a92c377036b7e9ff8747927e0f1080c3dada0cec0400",
    "logical_06": "2bdde1a5e89c25ec87d926d8ef96e2b5e5ba13b3e5a8bcfccef497479e18924a",
    "logical_07": "3c7c09108205b28a55aeded4cae721f0448803d4c2be5033af026998e9ce4e43",
    "logical_08": "b1f934886eb95ffd342064f5f10008e878d7abce918b9a0b62f2c31ff7d37350",
    "logical_09": "8a4d375eae758d9fd786f122c7d90371f0836d052a37ed1e4bdc05a95a2577cb",
    "logical_10": "b2f160a859371974a694fd6ead8971ce605fe2ae753868e798bb279bffa3ec39",
    "logical_11": "17bd4cbc8ecba464fe06822d0f71986b25373f4cfae5f76c93f69e0619b96ed4",
    "logical_12": "45a7656f15350f2f85b611f4768828db1d77ecbaf7aa42ff839600ab99e843cc",
    "result_norm": "caac1af10e7d445729e84379a39bd7ee47e10a69525575712ab78050fa9a533a",
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
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    tokens_bin = Path(ns.tokens_bin).resolve()
    out_dir = Path(ns.out_dir).resolve()

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set - refusing to run")

    if not model_dir.is_dir():
        stop(f"model directory missing: {model_dir}")
    if not tokens_bin.is_file():
        stop(f"token file missing: {tokens_bin}")

    runtime = model_dir / "modeling_longcat_flash_sparse.py"
    if not runtime.is_file():
        stop(f"runtime missing: {runtime}")

    runtime_sha = sha256_file(runtime)
    token_sha = sha256_file(tokens_bin)

    print(f"runtime_sha256={runtime_sha}")
    print(f"tokens_bin_sha256={token_sha}")

    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop("frozen runtime SHA mismatch")
    if token_sha != EXPECTED_TOKEN_SHA256:
        stop("authoritative token SHA mismatch")

    raw = tokens_bin.read_bytes()
    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop(f"unexpected token file size: {len(raw)}")

    ids = list(struct.unpack(f"<{EXPECTED_TOKEN_COUNT}i", raw))
    for i, token_id in enumerate(ids):
        if not 0 <= token_id < VOCAB_SIZE:
            stop(f"token {i} out of range: {token_id}")

    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        stop(f"import failure: {exc}")

    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")

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

    if not hasattr(model, "model"):
        stop("CausalLM has no .model trunk")
    trunk = model.model
    if not hasattr(trunk, "layers"):
        stop("model trunk has no layers")
    if len(trunk.layers) != 14:
        stop(f"unexpected logical layer count: {len(trunk.layers)}")
    if not hasattr(trunk, "norm"):
        stop("model trunk has no final norm")

    hooked: dict[str, torch.Tensor] = {}

    def layer13_hook(_module, _inputs, output):
        if "logical_13" in hooked:
            stop("layers[13] fired twice")
        if not isinstance(output, torch.Tensor):
            stop(f"layers[13] output is {type(output)}, expected Tensor")
        hooked["logical_13"] = output.detach().clone()

    def norm_hook(_module, inputs, output):
        if "norm_in" in hooked:
            stop("trunk.norm fired twice")
        hooked["norm_in"] = inputs[0].detach().clone()
        hooked["norm_out"] = output.detach().clone()

    h1 = trunk.layers[13].register_forward_hook(layer13_hook)
    h2 = trunk.norm.register_forward_hook(norm_hook)

    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")

    try:
        t1 = time.perf_counter()
        with torch.inference_mode():
            out = trunk(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        torch.cuda.synchronize()
        print(f"forward_seconds={time.perf_counter() - t1:.3f}")
    finally:
        h1.remove()
        h2.remove()

    hidden_states = out.hidden_states
    if hidden_states is None:
        stop("HF did not return hidden_states")
    if len(hidden_states) != EXPECTED_HIDDEN_STATES:
        stop(
            f"unexpected hidden-state count: {len(hidden_states)} "
            f"!= {EXPECTED_HIDDEN_STATES}"
        )
    for k in ("logical_13", "norm_in", "norm_out"):
        if k not in hooked:
            stop(f"hook did not fire: {k}")

    # Same-pass final-norm identity gates (no norm re-run).
    if not torch.equal(hooked["norm_in"], hooked["logical_13"]):
        stop("final-norm INPUT != layers[13] output (bytewise)")
    if not torch.equal(hooked["norm_out"], hidden_states[14]):
        stop("final-norm OUTPUT != hidden_states[14] (bytewise)")
    print("same-pass final-norm identity gates: PASS")

    # Surfaces to write: name -> (tensor, expected final-row SHA or None)
    surfaces: list[tuple[str, torch.Tensor, str | None]] = [
        ("logical_00_oracle", hidden_states[1], FINAL_ROW_SHA["logical_00"]),
    ]
    for n in range(1, 13):
        surfaces.append(
            (f"logical_{n:02d}", hidden_states[n + 1], FINAL_ROW_SHA[f"logical_{n:02d}"])
        )
    surfaces.append(("logical_13", hooked["logical_13"], None))
    surfaces.append(("result_norm_full", hidden_states[14], FINAL_ROW_SHA["result_norm"]))

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "description": "HF full-sequence residual-walk capture (causal-reset experiment)",
        "runtime_sha256": runtime_sha,
        "tokens_bin_sha256": token_sha,
        "sequence_length": EXPECTED_TOKEN_COUNT,
        "hidden_size": EXPECTED_HIDDEN,
        "layout": "token-major [512, 3072] float32-le",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tf32_disabled": True,
        "same_pass_norm_gates": "norm_in == layers[13].out and norm_out == hidden_states[14], bytewise",
        "surfaces": {},
    }

    sums_lines = []
    for name, t, expect_row_sha in surfaces:
        if tuple(t.shape) != (1, EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN):
            stop(f"{name}: unexpected shape {tuple(t.shape)}")

        v = (
            t[0]
            .float()
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )
        if not np.isfinite(v).all():
            stop(f"{name}: non-finite values")

        row_bytes = v[-1].tobytes()
        row_sha = hashlib.sha256(row_bytes).hexdigest()
        if expect_row_sha is not None and row_sha != expect_row_sha:
            stop(
                f"{name}: row-511 identity gate FAIL: {row_sha} != {expect_row_sha}"
            )

        path = out_dir / f"{name}.bin"
        path.write_bytes(v.tobytes())
        full_sha = sha256_file(path)

        summary["surfaces"][name] = {
            "sha256": full_sha,
            "row511_sha256": row_sha,
            "row511_gate": "PASS" if expect_row_sha is not None else "recorded-fresh",
            "min": float(v.min()),
            "max": float(v.max()),
            "rms": float(np.sqrt(np.mean(v.astype(np.float64) ** 2))),
        }
        sums_lines.append(f"{full_sha}  {name}.bin")
        print(
            f"{name}: sha256={full_sha} row511={row_sha} "
            f"gate={'PASS' if expect_row_sha is not None else 'fresh'}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    sums_lines.append(f"{sha256_file(out_dir / 'summary.json')}  summary.json")
    (out_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sums_lines) + "\n", encoding="utf-8"
    )

    print("HF RESID WALK CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
