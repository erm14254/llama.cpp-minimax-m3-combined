#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

EXPECTED_RUNTIME_SHA256 = "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
EXPECTED_TOKEN_SHA256 = "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
EXPECTED_TOKEN_COUNT = 512
EXPECTED_HIDDEN = 3072
EXPECTED_HIDDEN_STATES = 15
VOCAB_SIZE = 131072


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
    print(f"sequence_length={len(ids)}")

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

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device="cuda:0",
    )

    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    torch.cuda.synchronize()
    print(f"forward_seconds={time.perf_counter() - t1:.3f}")

    hidden_states = out.hidden_states

    if hidden_states is None:
        stop("HF did not return hidden_states")
    if len(hidden_states) != EXPECTED_HIDDEN_STATES:
        stop(
            f"unexpected hidden-state count: {len(hidden_states)} "
            f"!= {EXPECTED_HIDDEN_STATES}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    names = (
        ["inp_embd_ngram"]
        + [f"logical_{i:02d}" for i in range(13)]
        + ["result_norm"]
    )

    summary = {
        "runtime_sha256": runtime_sha,
        "tokens_bin_sha256": token_sha,
        "sequence_length": EXPECTED_TOKEN_COUNT,
        "hidden_size": EXPECTED_HIDDEN,
        "surfaces": {},
    }

    for idx, name in enumerate(names):
        t = hidden_states[idx]

        if tuple(t.shape) != (
            1,
            EXPECTED_TOKEN_COUNT,
            EXPECTED_HIDDEN,
        ):
            stop(f"{name}: unexpected shape {tuple(t.shape)}")

        # Compare the final prompt token only.
        v = (
            t[0, -1]
            .float()
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )

        if not np.isfinite(v).all():
            stop(f"{name}: non-finite values")

        path = out_dir / f"{name}.bin"
        path.write_bytes(v.tobytes())

        summary["surfaces"][name] = {
            "sha256": sha256_file(path),
            "min": float(v.min()),
            "max": float(v.max()),
            "rms": float(np.sqrt(np.mean(v.astype(np.float64) ** 2))),
        }

        print(
            f"{name}: "
            f"min={v.min():.9g} "
            f"max={v.max():.9g} "
            f"sha256={summary['surfaces'][name]['sha256']}"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("HF 512 HIDDEN CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())