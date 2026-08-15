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

LLAMA_REPO = ROOT

HF_VEC = ROOT / "hf_attn0_norm_512.bin"

CPP_VEC = (
    ROOT
    / "cpp_attn0_norm_512"
    / "logical0_attn0_norm.bin"
)

HF_WEIGHT_NAME = (
    "model.layers.0.input_layernorm.0.weight"
)

GGUF_WEIGHT_NAME = "blk.0.attn_norm.weight"

EXPECTED_HF_VEC_SHA = (
    "a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af"
)

HIDDEN = 3072
EXPECTED_WEIGHT_BYTES = HIDDEN * 2
EXPECTED_VECTOR_BYTES = HIDDEN * 4


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


def load_vec(path: Path) -> np.ndarray:
    if not path.is_file():
        stop("missing vector: %s" % path)

    if path.stat().st_size != EXPECTED_VECTOR_BYTES:
        stop(
            "%s has %d bytes, expected %d"
            % (
                path,
                path.stat().st_size,
                EXPECTED_VECTOR_BYTES,
            )
        )

    x = np.fromfile(path, dtype="<f4")

    if x.shape != (HIDDEN,):
        stop(
            "unexpected vector shape: %s"
            % (x.shape,)
        )

    if not np.isfinite(x).all():
        stop("nonfinite vector: %s" % path)

    return x


def torch_bf16_raw(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()

    if t.dtype != torch.bfloat16:
        stop(
            "expected BF16 tensor, got %s"
            % t.dtype
        )

    raw = (
        t.view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
    )

    return raw.tobytes(order="C")


def metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float, float]:
    c = candidate.astype(np.float64)
    r = reference.astype(np.float64)

    d = c - r

    rmse = float(
        np.sqrt(np.mean(d * d))
    )

    ref_rms = float(
        np.sqrt(np.mean(r * r))
    )

    rel = (
        rmse / ref_rms
        if ref_rms != 0.0
        else float("nan")
    )

    max_abs = float(
        np.abs(d).max()
    )

    mean_abs = float(
        np.abs(d).mean()
    )

    return max_abs, mean_abs, rmse, rel


def main() -> int:
    # ---------------------------------------------------------
    # Frozen RMSNorm vectors.
    # ---------------------------------------------------------

    hf_vec_sha = sha256_file(HF_VEC)

    if hf_vec_sha != EXPECTED_HF_VEC_SHA:
        stop(
            "HF RMSNorm oracle changed: %s"
            % hf_vec_sha
        )

    hf = load_vec(HF_VEC)
    cpp = load_vec(CPP_VEC)

    print("HF vector SHA256  =", hf_vec_sha)
    print("C++ vector SHA256 =", sha256_file(CPP_VEC))

    # ---------------------------------------------------------
    # HF checkpoint weight.
    # ---------------------------------------------------------

    index_path = HF_DIR / "model.safetensors.index.json"

    if not index_path.is_file():
        stop("missing Safetensors index")

    index = json.loads(
        index_path.read_text(encoding="utf-8")
    )

    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict):
        stop("invalid Safetensors weight_map")

    shard_name = weight_map.get(HF_WEIGHT_NAME)

    if shard_name is None:
        stop(
            "missing HF norm weight: "
            + HF_WEIGHT_NAME
        )

    shard_path = HF_DIR / shard_name

    if not shard_path.is_file():
        stop(
            "missing HF norm shard: %s"
            % shard_path
        )

    with safe_open(
        shard_path,
        framework="pt",
        device="cpu",
    ) as handle:
        weight = (
            handle.get_tensor(HF_WEIGHT_NAME)
            .contiguous()
        )

    if tuple(weight.shape) != (HIDDEN,):
        stop(
            "HF norm weight shape %s != (%d,)"
            % (
                tuple(weight.shape),
                HIDDEN,
            )
        )

    hf_weight_raw = torch_bf16_raw(weight)

    if len(hf_weight_raw) != EXPECTED_WEIGHT_BYTES:
        stop(
            "HF raw norm weight has %d bytes"
            % len(hf_weight_raw)
        )

    # ---------------------------------------------------------
    # GGUF weight.
    # ---------------------------------------------------------

    gguf_py = LLAMA_REPO / "gguf-py"

    if not gguf_py.is_dir():
        stop("missing gguf-py")

    sys.path.insert(0, str(gguf_py))

    try:
        from gguf import GGUFReader
    except Exception as exc:
        stop(
            "GGUFReader import failed: %s"
            % exc
        )

    matches = []
    block0_norm_candidates = []

    for gguf_path in sorted(GGUF_DIR.glob("*.gguf")):
        reader = GGUFReader(
            str(gguf_path),
            mode="r",
        )

        for tensor in reader.tensors:
            if (
                tensor.name.startswith("blk.0.")
                and "norm" in tensor.name.lower()
            ):
                block0_norm_candidates.append(
                    (
                        tensor.name,
                        gguf_path.name,
                        tensor.tensor_type.name,
                        tuple(
                            int(x)
                            for x in tensor.shape.tolist()
                        ),
                    )
                )

            if tensor.name == GGUF_WEIGHT_NAME:
                matches.append(
                    (
                        gguf_path,
                        tensor,
                    )
                )

    if len(matches) != 1:
        print()
        print("===== BLOCK-0 NORM CANDIDATES =====")

        for row in block0_norm_candidates:
            print(row)

        stop(
            "%s match count is %d, expected exactly 1"
            % (
                GGUF_WEIGHT_NAME,
                len(matches),
            )
        )

    gguf_path, tensor = matches[0]

    if tensor.tensor_type.name != "BF16":
        stop(
            "GGUF norm weight is %s, expected BF16"
            % tensor.tensor_type.name
        )

    gguf_shape = tuple(
        int(x)
        for x in tensor.shape.tolist()
    )

    if gguf_shape != (HIDDEN,):
        stop(
            "GGUF norm weight shape %s != (%d,)"
            % (
                gguf_shape,
                HIDDEN,
            )
        )

    gguf_weight_raw = (
        np.ascontiguousarray(
            tensor.data
        )
        .tobytes(order="C")
    )

    if len(gguf_weight_raw) != EXPECTED_WEIGHT_BYTES:
        stop(
            "GGUF raw norm weight has %d bytes, expected %d"
            % (
                len(gguf_weight_raw),
                EXPECTED_WEIGHT_BYTES,
            )
        )

    hf_weight_sha = sha256_bytes(
        hf_weight_raw
    )

    gguf_weight_sha = sha256_bytes(
        gguf_weight_raw
    )

    weight_exact = (
        hf_weight_raw == gguf_weight_raw
    )

    print()
    print("===== NORM WEIGHT AUDIT =====")
    print("HF tensor    =", HF_WEIGHT_NAME)
    print("HF shard     =", shard_name)
    print("GGUF tensor  =", GGUF_WEIGHT_NAME)
    print("GGUF shard   =", gguf_path.name)
    print("HF raw SHA   =", hf_weight_sha)
    print("GGUF raw SHA =", gguf_weight_sha)
    print("raw_exact    =", weight_exact)

    if not weight_exact:
        stop(
            "layer-0 attention RMSNorm weight differs between HF and GGUF"
        )

    # ---------------------------------------------------------
    # Is the output discrepancy almost entirely a single
    # per-token RMS scaling factor?
    #
    # Fit cpp ~= alpha * hf.
    # ---------------------------------------------------------

    h = hf.astype(np.float64)
    c = cpp.astype(np.float64)

    hh = float(np.dot(h, h))

    if hh == 0.0:
        stop("HF norm vector unexpectedly has zero norm")

    alpha = float(
        np.dot(c, h) / hh
    )

    scaled_hf = (
        alpha * h
    ).astype(np.float64)

    raw_max, raw_mean, raw_rmse, raw_rel = metrics(
        cpp,
        hf,
    )

    scaled_max, scaled_mean, scaled_rmse, scaled_rel = metrics(
        c,
        scaled_hf,
    )

    # Also measure cpp corrected back to HF scale.
    corrected_cpp = (
        c / alpha
    )

    corr_max, corr_mean, corr_rmse, corr_rel = metrics(
        corrected_cpp,
        h,
    )

    # Ratio statistics, excluding values near zero where ratios
    # are numerically meaningless.
    ratio_mask = np.abs(h) >= 0.01

    if not bool(ratio_mask.any()):
        stop("ratio mask unexpectedly empty")

    ratios = c[ratio_mask] / h[ratio_mask]

    q01, q50, q99 = np.quantile(
        ratios,
        [0.01, 0.50, 0.99],
    )

    print()
    print("===== SINGLE-SCALE DIAGNOSTIC =====")
    print("fit: C++ ~= alpha * HF")
    print("alpha             =", alpha)
    print("alpha_minus_1     =", alpha - 1.0)
    print("ratio_count       =", int(ratios.size))
    print("ratio_p01         =", float(q01))
    print("ratio_median      =", float(q50))
    print("ratio_p99         =", float(q99))
    print("ratio_std         =", float(np.std(ratios)))

    print()
    print("raw_rmse          =", raw_rmse)
    print("raw_rel_rmse      =", raw_rel)
    print("raw_max_abs       =", raw_max)

    print()
    print("residual_after_alpha_rmse     =", scaled_rmse)
    print("residual_after_alpha_rel_rmse =", scaled_rel)
    print("residual_after_alpha_max_abs  =", scaled_max)

    print()
    print("corrected_cpp_rmse      =", corr_rmse)
    print("corrected_cpp_rel_rmse  =", corr_rel)
    print("corrected_cpp_max_abs   =", corr_max)

    if raw_rmse != 0.0:
        print(
            "residual/raw RMSE ratio =",
            scaled_rmse / raw_rmse,
        )

    print()
    print("ATTN0 RMSNORM WEIGHT+SCALE AUDIT: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())