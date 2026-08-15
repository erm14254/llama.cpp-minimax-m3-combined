#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF_DIR = ROOT / "hf_logical0_stages_512_v4"
CPP_DIR = ROOT / "cpp_logical0_stages_512"

EXPECTED_BYTES = 3072 * 4

PAIRS = [
    (
        "input",
        "input.bin",
        "inp_embd_ngram.bin",
        "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f",
    ),
    (
        "attn0_resid",
        "attn0_resid.bin",
        "logical0_attn0_resid.bin",
        "2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177",
    ),
    (
        "mlp0_resid",
        "mlp0_resid.bin",
        "logical0_mlp0_resid.bin",
        "cf48a0ad3001e82ae41020675458df66219ea929caa1168ffddf64196d70404f",
    ),
    (
        "attn1_resid",
        "attn1_resid.bin",
        "logical0_attn1_resid.bin",
        "b4c1e5f684afefcec4129e3e6ec095a38d9b7f880115f819f78f8a698fe14431",
    ),
    (
        "logical0_out",
        "logical0_out.bin",
        "logical_00.bin",
        "5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff",
    ),
]


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
        stop(
            "%s has shape %s"
            % (path, x.shape)
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


def main() -> int:
    if not HF_DIR.is_dir():
        stop("HF stage directory missing")

    if not CPP_DIR.is_dir():
        stop("C++ stage directory missing")

    rows = []

    for (
        label,
        hf_name,
        cpp_name,
        expected_hf_sha,
    ) in PAIRS:
        hf_path = HF_DIR / hf_name
        cpp_path = CPP_DIR / cpp_name

        hf_sha = sha256_file(hf_path)

        if hf_sha != expected_hf_sha:
            stop(
                "%s HF oracle SHA changed: %s"
                % (label, hf_sha)
            )

        hf = load(hf_path)
        cpp = load(cpp_path)

        rows.append(
            (
                label,
                sha256_file(cpp_path),
                metrics(cpp, hf),
            )
        )

    if rows[0][2]["rmse"] != 0.0:
        stop(
            "input is not byte/numerically exact to HF"
        )

    print(
        "{:<14} {:>11} {:>11} {:>11} {:>11} "
        "{:>14} {:>11}".format(
            "surface",
            "max_abs",
            "mean_abs",
            "rmse",
            "rel_rmse",
            "cosine",
            "exact",
        )
    )

    print("-" * 100)

    for label, _sha, m in rows:
        print(
            "{:<14} {:11.6g} {:11.6g} {:11.6g} "
            "{:11.6g} {:14.10f} {:4d}/3072".format(
                label,
                m["max_abs"],
                m["mean_abs"],
                m["rmse"],
                m["rel"],
                m["cosine"],
                m["exact"],
            )
        )

    print()
    print("===== C++ STAGE SHA256 =====")

    for label, digest, _m in rows:
        print(
            "%-14s %s"
            % (label, digest)
        )

    print()
    print("input RMSE = 0 exactly")
    print("LOGICAL-0 STAGE COMPARISON: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())