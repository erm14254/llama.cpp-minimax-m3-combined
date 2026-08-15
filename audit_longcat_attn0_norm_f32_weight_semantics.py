#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)

GGUF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16"
)

INPUT_PATH = (
    ROOT
    / "hf_hidden_512_v4"
    / "inp_embd_ngram.bin"
)

HF_NORM_PATH = ROOT / "hf_attn0_norm_512.bin"

CPP_NORM_PATH = (
    ROOT
    / "cpp_attn0_norm_512"
    / "logical0_attn0_norm.bin"
)

HF_WEIGHT_NAME = (
    "model.layers.0.input_layernorm.0.weight"
)

GGUF_WEIGHT_NAME = "blk.0.attn_norm.weight"

EXPECTED_INPUT_SHA = (
    "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f"
)

EXPECTED_HF_NORM_SHA = (
    "a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af"
)

HIDDEN = 3072
VECTOR_BYTES = HIDDEN * 4


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_f32(path: Path) -> np.ndarray:
    if not path.is_file():
        stop("missing vector: %s" % path)

    if path.stat().st_size != VECTOR_BYTES:
        stop(
            "%s has %d bytes, expected %d"
            % (
                path,
                path.stat().st_size,
                VECTOR_BYTES,
            )
        )

    x = np.fromfile(path, dtype="<f4")

    if x.shape != (HIDDEN,):
        stop(
            "%s shape %s != (%d,)"
            % (
                path,
                x.shape,
                HIDDEN,
            )
        )

    if not np.isfinite(x).all():
        stop("nonfinite vector: %s" % path)

    return x


def metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int]:
    c = candidate.astype(np.float64)
    r = reference.astype(np.float64)

    delta = c - r
    abs_delta = np.abs(delta)

    rmse = float(
        np.sqrt(np.mean(delta * delta))
    )

    ref_rms = float(
        np.sqrt(np.mean(r * r))
    )

    rel = (
        rmse / ref_rms
        if ref_rms != 0.0
        else float("nan")
    )

    denom = float(
        np.linalg.norm(c)
        * np.linalg.norm(r)
    )

    cosine = (
        float(np.dot(c, r) / denom)
        if denom != 0.0
        else float("nan")
    )

    return {
        "max_abs": float(abs_delta.max()),
        "mean_abs": float(abs_delta.mean()),
        "rmse": rmse,
        "rel": rel,
        "cosine": cosine,
        "exact": int(
            np.count_nonzero(
                candidate == reference
            )
        ),
    }


def print_metrics(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> None:
    m = metrics(candidate, reference)

    raw = (
        np.asarray(
            candidate,
            dtype="<f4",
        )
        .tobytes()
    )

    print(
        "{:<28} {}  "
        "max={:.9g} mean={:.9g} rmse={:.9g} "
        "rel={:.9g} cos={:.10f} exact={}/3072".format(
            name,
            sha256_bytes(raw),
            m["max_abs"],
            m["mean_abs"],
            m["rmse"],
            m["rel"],
            m["cosine"],
            m["exact"],
        )
    )


def to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return (
        t
        .float()
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .reshape(HIDDEN)
        .astype("<f4", copy=False)
    )


def main() -> int:
    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    input_sha = sha256_file(INPUT_PATH)
    hf_norm_sha = sha256_file(HF_NORM_PATH)

    print("input SHA256   =", input_sha)
    print("HF norm SHA256 =", hf_norm_sha)

    if input_sha != EXPECTED_INPUT_SHA:
        stop("HF input oracle changed")

    if hf_norm_sha != EXPECTED_HF_NORM_SHA:
        stop("HF RMSNorm oracle changed")

    input_f32 = load_f32(INPUT_PATH)
    hf_norm = load_f32(HF_NORM_PATH)
    cpp_norm = load_f32(CPP_NORM_PATH)

    # ---------------------------------------------------------
    # Load HF BF16 norm weight.
    # ---------------------------------------------------------

    index_path = HF_DIR / "model.safetensors.index.json"

    if not index_path.is_file():
        stop("Safetensors index missing")

    index = json.loads(
        index_path.read_text(encoding="utf-8")
    )

    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict):
        stop("invalid Safetensors weight_map")

    shard_name = weight_map.get(HF_WEIGHT_NAME)

    if shard_name is None:
        stop(
            "missing HF tensor: "
            + HF_WEIGHT_NAME
        )

    shard_path = HF_DIR / shard_name

    with safe_open(
        shard_path,
        framework="pt",
        device="cpu",
    ) as handle:
        hf_weight = (
            handle.get_tensor(HF_WEIGHT_NAME)
            .contiguous()
        )

    if tuple(hf_weight.shape) != (HIDDEN,):
        stop(
            "HF norm weight shape is %s"
            % (tuple(hf_weight.shape),)
        )

    if hf_weight.dtype != torch.bfloat16:
        stop(
            "HF norm weight is %s, expected BF16"
            % hf_weight.dtype
        )

    hf_weight_f32 = (
        hf_weight
        .float()
        .numpy()
        .astype("<f4", copy=False)
    )

    # ---------------------------------------------------------
    # Load actual GGUF F32 norm weight.
    # ---------------------------------------------------------

    gguf_py = ROOT / "gguf-py"

    if not gguf_py.is_dir():
        stop("gguf-py missing")

    sys.path.insert(0, str(gguf_py))

    try:
        from gguf import GGUFReader
    except Exception as exc:
        stop(
            "GGUFReader import failed: %s"
            % exc
        )

    matches = []

    for gguf_path in sorted(
        GGUF_DIR.glob("*.gguf")
    ):
        reader = GGUFReader(
            str(gguf_path),
            mode="r",
        )

        for tensor in reader.tensors:
            if tensor.name == GGUF_WEIGHT_NAME:
                matches.append(
                    (
                        gguf_path,
                        tensor,
                    )
                )

    if len(matches) != 1:
        stop(
            "%s match count is %d, expected 1"
            % (
                GGUF_WEIGHT_NAME,
                len(matches),
            )
        )

    gguf_path, gguf_tensor = matches[0]

    print()
    print("===== WEIGHT TYPES =====")
    print("HF tensor     =", HF_WEIGHT_NAME)
    print("HF shard      =", shard_name)
    print("HF dtype      =", hf_weight.dtype)
    print("GGUF tensor   =", GGUF_WEIGHT_NAME)
    print("GGUF shard    =", gguf_path.name)
    print("GGUF qtype    =", gguf_tensor.tensor_type.name)
    print(
        "GGUF shape    =",
        tuple(
            int(v)
            for v in gguf_tensor.shape.tolist()
        ),
    )

    if gguf_tensor.tensor_type.name != "F32":
        stop(
            "this v2 audit expected observed GGUF F32, got %s"
            % gguf_tensor.tensor_type.name
        )

    gguf_weight_f32 = (
        np.asarray(
            gguf_tensor.data,
            dtype=np.float32,
        )
        .reshape(-1)
        .astype("<f4", copy=False)
    )

    if gguf_weight_f32.shape != (HIDDEN,):
        stop(
            "GGUF weight shape after decode is %s"
            % (gguf_weight_f32.shape,)
        )

    weight_exact = np.array_equal(
        gguf_weight_f32,
        hf_weight_f32,
    )

    hf_expanded_sha = sha256_bytes(
        hf_weight_f32.tobytes()
    )

    gguf_f32_sha = sha256_bytes(
        gguf_weight_f32.tobytes()
    )

    print()
    print("===== DECODED WEIGHT AUDIT =====")
    print(
        "HF BF16 -> F32 SHA =",
        hf_expanded_sha,
    )
    print(
        "GGUF F32 SHA        =",
        gguf_f32_sha,
    )
    print(
        "decoded_exact       =",
        weight_exact,
    )

    if not weight_exact:
        delta = (
            gguf_weight_f32.astype(np.float64)
            - hf_weight_f32.astype(np.float64)
        )

        stop(
            "decoded norm weight differs: "
            "max_abs=%g mismatches=%d"
            % (
                float(np.abs(delta).max()),
                int(
                    np.count_nonzero(
                        gguf_weight_f32
                        != hf_weight_f32
                    )
                ),
            )
        )

    # ---------------------------------------------------------
    # Exact Transformers 5.14.1 LongCat RMSNorm ordering:
    #
    # input BF16
    # -> F32 normalize
    # -> cast normalized activation to BF16
    # -> multiply BF16 weight
    # -> BF16 output
    #
    # Compare that to F32 variants.
    # ---------------------------------------------------------

    config = json.loads(
        (HF_DIR / "config.json").read_text(
            encoding="utf-8"
        )
    )

    eps = float(config["rms_norm_eps"])

    device = torch.device("cuda:0")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    x_bf = (
        torch.from_numpy(
            input_f32.copy()
        )
        .to(
            device=device,
            dtype=torch.bfloat16,
        )
        .view(1, 1, HIDDEN)
    )

    w_bf = hf_weight.to(
        device=device,
        dtype=torch.bfloat16,
    )

    w_f32 = w_bf.float()

    with torch.inference_mode():
        x32 = x_bf.float()

        variance = (
            x32
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        norm32 = (
            x32
            * torch.rsqrt(
                variance + eps
            )
        )

        # Exact HF ordering.
        norm_bf = norm32.to(
            torch.bfloat16
        )

        hf_manual = (
            w_bf
            * norm_bf
        )

        # Same BF16 normalized activation, but keep weight
        # multiply/output in F32.
        bf16_norm_f32_weight = (
            w_f32
            * norm_bf.float()
        )

        # Fully F32 after the BF16 input has been expanded.
        f32_all = (
            w_f32
            * norm32
        )

        # Final-rounding-only variant.
        f32_all_then_bf16 = (
            f32_all
            .to(torch.bfloat16)
        )

        torch.cuda.synchronize()

        variants = {
            "hf_manual":
                to_numpy_f32(hf_manual),
            "bf16_norm_f32_weight":
                to_numpy_f32(
                    bf16_norm_f32_weight
                ),
            "f32_all":
                to_numpy_f32(f32_all),
            "f32_all_then_bf16":
                to_numpy_f32(
                    f32_all_then_bf16
                ),
        }

    print()
    print("rms_norm_eps =", eps)

    print()
    print("===== AGAINST HF ORACLE =====")

    for name, values in variants.items():
        print_metrics(
            name,
            values,
            hf_norm,
        )

    manual_sha = sha256_bytes(
        variants["hf_manual"].tobytes()
    )

    manual_exact = (
        manual_sha == EXPECTED_HF_NORM_SHA
    )

    print()
    print(
        "hf_manual byte-exact to HF oracle =",
        manual_exact,
    )

    if not manual_exact:
        stop(
            "manual HF semantics failed oracle self-check"
        )

    print()
    print("===== AGAINST ACTUAL C++ =====")

    for name, values in variants.items():
        print_metrics(
            name,
            values,
            cpp_norm,
        )

    # ---------------------------------------------------------
    # Also quantify whether C++ is merely a global RMS-scale
    # difference from our F32-all reconstruction.
    # ---------------------------------------------------------

    f = (
        variants["f32_all"]
        .astype(np.float64)
    )

    c = cpp_norm.astype(np.float64)

    denom = float(np.dot(f, f))

    if denom == 0.0:
        stop("F32 reconstruction has zero norm")

    alpha = float(
        np.dot(c, f) / denom
    )

    residual = c - alpha * f

    residual_rmse = float(
        np.sqrt(
            np.mean(
                residual * residual
            )
        )
    )

    raw_diff = c - f

    raw_rmse = float(
        np.sqrt(
            np.mean(
                raw_diff * raw_diff
            )
        )
    )

    print()
    print("===== C++ VS F32-ALL SCALE CHECK =====")
    print("alpha =", alpha)
    print("alpha_minus_1 =", alpha - 1.0)
    print("raw_rmse =", raw_rmse)
    print(
        "residual_after_alpha_rmse =",
        residual_rmse,
    )

    if raw_rmse != 0.0:
        print(
            "residual/raw ratio =",
            residual_rmse / raw_rmse,
        )

    print()
    print(
        "ATTN0 RMSNORM F32-WEIGHT SEMANTICS AUDIT: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())