#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF_PATH = ROOT / "hf_attn0_norm_512.bin"

CPP_PATH = (
    ROOT
    / "cpp_attn0_norm_512"
    / "logical0_attn0_norm.bin"
)

EXPECTED_HF_SHA = (
    "a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af"
)

EXPECTED_BYTES = 3072 * 4


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


def load(path: Path) -> np.ndarray:
    if not path.is_file():
        stop("missing vector: %s" % path)

    if path.stat().st_size != EXPECTED_BYTES:
        stop(
            "%s has %d bytes, expected %d"
            % (
                path,
                path.stat().st_size,
                EXPECTED_BYTES,
            )
        )

    x = np.fromfile(path, dtype="<f4")

    if x.shape != (3072,):
        stop("wrong vector shape: %s" % (x.shape,))

    if not np.isfinite(x).all():
        stop("nonfinite vector: %s" % path)

    return x


def metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int]:
    c = candidate.astype(np.float64)
    r = reference.astype(np.float64)

    d = c - r
    ad = np.abs(d)

    rmse = float(np.sqrt(np.mean(d * d)))
    ref_rms = float(np.sqrt(np.mean(r * r)))

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
        "max_abs": float(ad.max()),
        "mean_abs": float(ad.mean()),
        "rmse": rmse,
        "rel": rel,
        "cosine": cosine,
        "exact": int(
            np.count_nonzero(
                candidate == reference
            )
        ),
    }


def bf16_round_f32(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(
        np.asarray(x, dtype=np.float32).copy()
    )

    y = (
        t.to(torch.bfloat16)
        .to(torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )

    return y


def print_metrics(
    name: str,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> None:
    m = metrics(candidate, reference)

    raw = np.asarray(
        candidate,
        dtype="<f4",
    ).tobytes()

    print(
        "%-18s sha=%s" %
        (
            name,
            sha256_bytes(raw),
        )
    )

    print(
        "  max_abs=%g mean_abs=%g rmse=%g "
        "rel_rmse=%g cosine=%.10f exact=%d/3072"
        % (
            m["max_abs"],
            m["mean_abs"],
            m["rmse"],
            m["rel"],
            m["cosine"],
            m["exact"],
        )
    )


def main() -> int:
    hf_sha = sha256_file(HF_PATH)

    if hf_sha != EXPECTED_HF_SHA:
        stop(
            "HF RMSNorm oracle changed: %s"
            % hf_sha
        )

    hf = load(HF_PATH)
    cpp = load(CPP_PATH)

    hf_rounded = bf16_round_f32(hf)
    cpp_rounded = bf16_round_f32(cpp)

    print("HF SHA256        =", hf_sha)
    print("C++ SHA256       =", sha256_file(CPP_PATH))

    print()
    print("===== HF LATTICE SANITY =====")

    hf_lattice_exact = np.array_equal(
        hf_rounded,
        hf,
    )

    print(
        "HF survives BF16 round exactly =",
        hf_lattice_exact,
    )

    if not hf_lattice_exact:
        stop(
            "HF RMSNorm oracle is unexpectedly not on BF16 lattice"
        )

    print()
    print("===== AGAINST HF =====")

    print_metrics(
        "C++ raw",
        cpp,
        hf,
    )

    print_metrics(
        "C++ -> BF16",
        cpp_rounded,
        hf,
    )

    rounded_sha = sha256_bytes(
        cpp_rounded.tobytes()
    )

    byte_exact = rounded_sha == hf_sha

    print()
    print(
        "C++ -> BF16 SHA =",
        rounded_sha,
    )

    print(
        "BF16-rounded C++ byte-exact to HF =",
        byte_exact,
    )

    print()
    print("ATTN0 RMSNORM BF16-ROUNDING DIAGNOSTIC: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())