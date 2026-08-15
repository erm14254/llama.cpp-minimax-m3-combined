#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open

from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashRMSNorm,
)


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

MODEL_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)

INPUT_PATH = ROOT / "hf_hidden_512_v4" / "inp_embd_ngram.bin"

OUT_PATH = ROOT / "hf_attn0_norm_512.bin"

EXPECTED_INPUT_SHA = (
    "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f"
)

HIDDEN = 3072

WEIGHT_NAME = "model.layers.0.input_layernorm.0.weight"


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main() -> int:
    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    if not INPUT_PATH.is_file():
        stop("HF input oracle missing")

    input_sha = sha256_file(INPUT_PATH)

    print("transformers =", transformers.__version__)
    print("torch        =", torch.__version__)
    print("input_sha256 =", input_sha)

    if input_sha != EXPECTED_INPUT_SHA:
        stop("HF input oracle SHA changed")

    config_path = MODEL_DIR / "config.json"
    index_path = MODEL_DIR / "model.safetensors.index.json"

    if not config_path.is_file():
        stop("config.json missing")

    if not index_path.is_file():
        stop("Safetensors index missing")

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    eps = float(config["rms_norm_eps"])

    index = json.loads(
        index_path.read_text(encoding="utf-8")
    )

    weight_map = index.get("weight_map", {})

    shard_name = weight_map.get(WEIGHT_NAME)

    if shard_name is None:
        stop(
            "missing checkpoint tensor: "
            + WEIGHT_NAME
        )

    shard_path = MODEL_DIR / shard_name

    if not shard_path.is_file():
        stop("norm shard missing: %s" % shard_path)

    with safe_open(
        shard_path,
        framework="pt",
        device="cpu",
    ) as handle:
        if WEIGHT_NAME not in handle.keys():
            stop("indexed norm tensor absent from shard")

        weight = (
            handle.get_tensor(WEIGHT_NAME)
            .contiguous()
        )

    print("weight_name  =", WEIGHT_NAME)
    print("weight_shard =", shard_name)
    print("weight_shape =", tuple(weight.shape))
    print("weight_dtype =", weight.dtype)
    print("rms_norm_eps =", eps)

    if tuple(weight.shape) != (HIDDEN,):
        stop(
            "unexpected norm weight shape: %s"
            % (tuple(weight.shape),)
        )

    if weight.dtype != torch.bfloat16:
        stop(
            "norm weight is not BF16: %s"
            % weight.dtype
        )

    raw_input = np.fromfile(
        INPUT_PATH,
        dtype="<f4",
    )

    if raw_input.shape != (HIDDEN,):
        stop(
            "input vector shape is %s"
            % (raw_input.shape,)
        )

    if not np.isfinite(raw_input).all():
        stop("input oracle contains nonfinite values")

    # The stored F32 values are exact expansions of BF16 values.
    x = torch.from_numpy(
        raw_input.copy()
    ).view(1, 1, HIDDEN)

    device = torch.device("cuda:0")

    x = x.to(
        device=device,
        dtype=torch.bfloat16,
    )

    norm = LongcatFlashRMSNorm(
        HIDDEN,
        eps=eps,
    ).to(
        device=device,
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        norm.weight.copy_(
            weight.to(
                device=device,
                dtype=torch.bfloat16,
            )
        )

    norm.eval()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with torch.inference_mode():
        y = norm(x)

    torch.cuda.synchronize()

    print("output_dtype =", y.dtype)
    print("output_shape =", tuple(y.shape))

    if y.dtype != torch.bfloat16:
        stop(
            "RMSNorm output unexpectedly has dtype %s"
            % y.dtype
        )

    out = (
        y[0, 0]
        .float()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )

    if out.shape != (HIDDEN,):
        stop("wrong output vector shape")

    if not np.isfinite(out).all():
        stop("RMSNorm output contains nonfinite values")

    OUT_PATH.write_bytes(out.tobytes())

    digest = sha256_file(OUT_PATH)

    print()
    print("output_file   =", OUT_PATH)
    print("output_bytes  =", OUT_PATH.stat().st_size)
    print("output_sha256 =", digest)
    print("output_min    =", float(out.min()))
    print("output_max    =", float(out.max()))
    print()
    print("HF ATTN0 RMSNORM STANDALONE: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())