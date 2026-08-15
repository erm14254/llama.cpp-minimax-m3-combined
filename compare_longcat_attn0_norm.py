#!/usr/bin/env python3

from pathlib import Path
import hashlib

import numpy as np


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF = ROOT / "hf_attn0_norm_512.bin"

CPP = (
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


def main() -> int:
    hf_sha = sha256_file(HF)
    cpp_sha = sha256_file(CPP)

    print("HF SHA256  =", hf_sha)
    print("C++ SHA256 =", cpp_sha)

    if hf_sha != EXPECTED_HF_SHA:
        stop("HF RMSNorm oracle SHA changed")

    hf = load(HF)
    cpp = load(CPP)

    d = cpp.astype(np.float64) - hf.astype(np.float64)
    ad = np.abs(d)

    rmse = float(np.sqrt(np.mean(d * d)))
    hf_rms = float(
        np.sqrt(
            np.mean(
                hf.astype(np.float64) ** 2
            )
        )
    )

    rel = (
        rmse / hf_rms
        if hf_rms != 0.0
        else float("nan")
    )

    denom = float(
        np.linalg.norm(cpp.astype(np.float64))
        * np.linalg.norm(hf.astype(np.float64))
    )

    cosine = (
        float(
            np.dot(
                cpp.astype(np.float64),
                hf.astype(np.float64),
            )
            / denom
        )
        if denom != 0.0
        else float("nan")
    )

    exact = int(
        np.count_nonzero(cpp == hf)
    )

    print()
    print("max_abs   =", float(ad.max()))
    print("mean_abs  =", float(ad.mean()))
    print("rmse      =", rmse)
    print("rel_rmse  =", rel)
    print("cosine    =", cosine)
    print("exact     = %d/3072" % exact)
    print("byte_exact =", cpp_sha == hf_sha)

    print()
    print("ATTN0 RMSNORM COMPARISON: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())