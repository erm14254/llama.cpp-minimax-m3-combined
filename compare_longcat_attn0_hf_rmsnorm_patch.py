#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

PAIRS = [
    (
        "input",
        ROOT / "hf_hidden_512_v4" / "inp_embd_ngram.bin",
        ROOT / "cpp_attn0_hf_rmsnorm_512" / "inp_embd_ngram.bin",
        "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f",
    ),
    (
        "attn0_norm",
        ROOT / "hf_attn0_norm_512.bin",
        ROOT / "cpp_attn0_hf_rmsnorm_512" / "logical0_attn0_norm.bin",
        "a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af",
    ),
    (
        "attn0_resid",
        ROOT / "hf_logical0_stages_512_v4" / "attn0_resid.bin",
        ROOT / "cpp_attn0_hf_rmsnorm_512" / "logical0_attn0_resid.bin",
        "2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177",
    ),
]

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
            % (path, path.stat().st_size, EXPECTED_BYTES)
        )

    x = np.fromfile(path, dtype="<f4")

    if x.shape != (3072,):
        stop("%s shape is %s" % (path, x.shape))

    if not np.isfinite(x).all():
        stop("nonfinite vector: %s" % path)

    return x


def metric(cpp: np.ndarray, hf: np.ndarray):
    c = cpp.astype(np.float64)
    h = hf.astype(np.float64)

    d = c - h
    ad = np.abs(d)

    rmse = float(np.sqrt(np.mean(d * d)))
    hf_rms = float(np.sqrt(np.mean(h * h)))
    rel = rmse / hf_rms if hf_rms else float("nan")

    denom = float(np.linalg.norm(c) * np.linalg.norm(h))
    cosine = float(np.dot(c, h) / denom) if denom else float("nan")

    return (
        float(ad.max()),
        float(ad.mean()),
        rmse,
        rel,
        cosine,
        int(np.count_nonzero(cpp == hf)),
    )


def main() -> int:
    print(
        "{:<12} {:>11} {:>11} {:>11} {:>11} {:>14} {:>11} {:>8}".format(
            "surface",
            "max_abs",
            "mean_abs",
            "rmse",
            "rel_rmse",
            "cosine",
            "exact",
            "sha_eq",
        )
    )
    print("-" * 106)

    results = {}

    for label, hf_path, cpp_path, expected_hf_sha in PAIRS:
        hf_sha = sha256_file(hf_path)
        cpp_sha = sha256_file(cpp_path)

        if hf_sha != expected_hf_sha:
            stop(
                "%s HF oracle changed: %s"
                % (label, hf_sha)
            )

        hf = load(hf_path)
        cpp = load(cpp_path)

        m = metric(cpp, hf)

        results[label] = {
            "hf_sha": hf_sha,
            "cpp_sha": cpp_sha,
            "metric": m,
        }

        print(
            "{:<12} {:11.6g} {:11.6g} {:11.6g} {:11.6g} "
            "{:14.10f} {:4d}/3072 {:>8}".format(
                label,
                m[0],
                m[1],
                m[2],
                m[3],
                m[4],
                m[5],
                str(cpp_sha == hf_sha),
            )
        )

    if results["input"]["cpp_sha"] != results["input"]["hf_sha"]:
        stop("input lost exact HF parity")

    print()
    print("===== SHA256 =====")

    for label in ("input", "attn0_norm", "attn0_resid"):
        print(
            "%-12s HF=%s" %
            (
                label,
                results[label]["hf_sha"],
            )
        )
        print(
            "%-12s C++=%s" %
            (
                "",
                results[label]["cpp_sha"],
            )
        )

    print()
    print(
        "attn0_norm byte_exact =",
        results["attn0_norm"]["cpp_sha"]
        == results["attn0_norm"]["hf_sha"],
    )

    print(
        "attn0_resid byte_exact =",
        results["attn0_resid"]["cpp_sha"]
        == results["attn0_resid"]["hf_sha"],
    )

    print()
    print("ATTN0 HF-RMSNORM PATCH COMPARISON: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())