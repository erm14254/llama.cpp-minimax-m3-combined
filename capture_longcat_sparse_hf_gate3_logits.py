#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
import time
from pathlib import Path

EXPECTED_RUNTIME_SHA256 = "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
INPUT_IDS = [20769, 235, 3121, 224]
VOCAB_SIZE = 131072

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stop(msg: str):
    raise SystemExit(f"STOP: {msg}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-json", required=True)
    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    out_bin = Path(ns.out_bin).resolve()
    out_json = Path(ns.out_json).resolve()

    if not model_dir.is_dir():
        stop(f"model directory does not exist: {model_dir}")

    runtime = model_dir / "modeling_longcat_flash_sparse.py"
    if not runtime.is_file():
        stop(f"missing frozen Sparse runtime: {runtime}")

    runtime_sha = sha256_file(runtime)
    print(f"runtime_sha256={runtime_sha}")
    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop(
            "frozen runtime SHA mismatch; expected "
            f"{EXPECTED_RUNTIME_SHA256}, got {runtime_sha}"
        )

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        stop(f"failed to import torch/transformers: {exc}")

    if not torch.cuda.is_available():
        stop("CUDA is not available in this Python environment")

    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"input_ids={INPUT_IDS}")

    # This is a numerical oracle run, not generation.
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
        )
    except Exception as exc:
        stop(f"model load failed: {type(exc).__name__}: {exc}")

    model.eval()
    torch.cuda.synchronize()
    print(f"load_seconds={time.perf_counter() - t0:.3f}")

    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device="cuda:0")

    t1 = time.perf_counter()
    with torch.inference_mode():
        try:
            output = model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
        except Exception as exc:
            stop(f"HF forward failed: {type(exc).__name__}: {exc}")

    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - t1

    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.ndim != 3:
        stop(f"unexpected logits rank/shape: {tuple(logits.shape)}")
    if tuple(logits.shape[:2]) != (1, len(INPUT_IDS)):
        stop(f"unexpected batch/sequence logits shape: {tuple(logits.shape)}")
    if logits.shape[-1] != VOCAB_SIZE:
        stop(f"unexpected vocab size: {logits.shape[-1]} != {VOCAB_SIZE}")

    last = logits[0, -1].float().detach().cpu().contiguous()

    finite = torch.isfinite(last)
    nonfinite = int((~finite).sum().item())
    if nonfinite:
        stop(f"HF last-position logits contain {nonfinite} non-finite values")

    vals = last.tolist()
    mn = min(vals)
    mx = max(vals)
    top = torch.topk(last, k=20)
    top_ids = [int(x) for x in top.indices.tolist()]
    top_vals = [float(x) for x in top.values.tolist()]

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as f:
        for v in vals:
            f.write(struct.pack("<f", float(v)))

    summary = {
        "runtime_sha256": runtime_sha,
        "model_dir": str(model_dir),
        "input_ids": INPUT_IDS,
        "sequence_length": len(INPUT_IDS),
        "vocab_size": VOCAB_SIZE,
        "dtype_saved": "float32 little-endian",
        "nonfinite": nonfinite,
        "finite_min": mn,
        "finite_max": mx,
        "top20_ids": top_ids,
        "top20_values": top_vals,
        "forward_seconds": forward_seconds,
        "logits_bin_sha256": sha256_file(out_bin),
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"forward_seconds={forward_seconds:.3f}")
    print(f"logits={len(vals)}")
    print(f"nonfinite={nonfinite}")
    print(f"finite_min={mn}")
    print(f"finite_max={mx}")
    print(f"top1_id={top_ids[0]}")
    print(f"out_bin={out_bin}")
    print(f"logits_bin_sha256={summary['logits_bin_sha256']}")
    print("GATE-3 HF V4 CAPTURE: PASS")

    del output, logits, last, input_ids, model
    gc.collect()
    torch.cuda.empty_cache()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
