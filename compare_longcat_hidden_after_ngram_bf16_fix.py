#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF_DIR = ROOT / "hf_hidden_512_v4"

OLD_DIR = ROOT / "cpp_hidden_512_fa_off_f32"

NEW_DIR = (
    ROOT
    / "cpp_hidden_512_fa_off_f32_ngram_bf16_restore_f32"
)

SURFACES = (
    ["inp_embd_ngram"]
    + [f"logical_{i:02d}" for i in range(13)]
    + ["result_norm"]
)

EXPECTED_BYTES = 3072 * 4

EXPECTED_HF_INPUT_SHA = (
    "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f"
)


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_vector(directory: Path, surface: str) -> np.ndarray:
    path = directory / f"{surface}.bin"

    if not path.is_file():
        stop(f"missing {path}")

    size = path.stat().st_size

    if size != EXPECTED_BYTES:
        stop(
            f"{path} has {size} bytes, "
            f"expected {EXPECTED_BYTES}"
        )

    x = np.fromfile(path, dtype="<f4")

    if x.size != 3072:
        stop(
            f"{path} contains {x.size} floats, expected 3072"
        )

    if not np.isfinite(x).all():
        stop(f"{path} contains nonfinite values")

    return x


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    c = candidate.astype(np.float64)
    r = reference.astype(np.float64)

    delta = c - r
    abs_delta = np.abs(delta)

    rmse = float(np.sqrt(np.mean(delta * delta)))
    ref_rms = float(np.sqrt(np.mean(r * r)))

    rel_rmse = (
        rmse / ref_rms
        if ref_rms != 0.0
        else float("nan")
    )

    denom = float(np.linalg.norm(c) * np.linalg.norm(r))

    cosine = (
        float(np.dot(c, r) / denom)
        if denom != 0.0
        else float("nan")
    )

    return {
        "max_abs": float(abs_delta.max()),
        "mean_abs": float(abs_delta.mean()),
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "cosine": cosine,
        "exact": int(np.count_nonzero(candidate == reference)),
    }


def main() -> int:
    for directory in (HF_DIR, OLD_DIR, NEW_DIR):
        if not directory.is_dir():
            stop(f"missing directory: {directory}")

    hf_input = HF_DIR / "inp_embd_ngram.bin"
    new_input = NEW_DIR / "inp_embd_ngram.bin"

    hf_sha = sha256_file(hf_input)
    new_sha = sha256_file(new_input)

    print("HF input SHA  =", hf_sha)
    print("NEW input SHA =", new_sha)

    if hf_sha != EXPECTED_HF_INPUT_SHA:
        stop("frozen HF input oracle changed")

    if new_sha != hf_sha:
        stop("new C++ input boundary is not byte-exact to HF")

    print()
    print("===== OLD VS NEW, BOTH AGAINST HF =====")
    print(
        "{:<16} {:>11} {:>11} {:>11} {:>11} {:>11} {:>14}".format(
            "surface",
            "old_rel",
            "new_rel",
            "new/old",
            "old_rmse",
            "new_rmse",
            "new_cosine",
        )
    )
    print("-" * 94)

    rows = []

    for surface in SURFACES:
        hf = load_vector(HF_DIR, surface)
        old = load_vector(OLD_DIR, surface)
        new = load_vector(NEW_DIR, surface)

        om = metrics(old, hf)
        nm = metrics(new, hf)

        ratio = (
            nm["rel_rmse"] / om["rel_rmse"]
            if om["rel_rmse"] != 0.0
            else 0.0 if nm["rel_rmse"] == 0.0 else float("inf")
        )

        rows.append(
            (
                surface,
                om,
                nm,
                ratio,
            )
        )

        print(
            "{:<16} {:11.6g} {:11.6g} {:11.6g} "
            "{:11.6g} {:11.6g} {:14.10f}".format(
                surface,
                om["rel_rmse"],
                nm["rel_rmse"],
                ratio,
                om["rmse"],
                nm["rmse"],
                nm["cosine"],
            )
        )

    print()
    print("===== NEW RUN DETAIL =====")
    print(
        "{:<16} {:>11} {:>11} {:>11} {:>11} {:>14} {:>11}".format(
            "surface",
            "max_abs",
            "mean_abs",
            "rmse",
            "rel_rmse",
            "cosine",
            "exact",
        )
    )
    print("-" * 101)

    for surface, _old, new, _ratio in rows:
        print(
            "{:<16} {:11.6g} {:11.6g} {:11.6g} {:11.6g} "
            "{:14.10f} {:4d}/3072".format(
                surface,
                new["max_abs"],
                new["mean_abs"],
                new["rmse"],
                new["rel_rmse"],
                new["cosine"],
                new["exact"],
            )
        )

    input_new = dict(
        (surface, new)
        for surface, _old, new, _ratio in rows
    )["inp_embd_ngram"]

    if input_new["rmse"] != 0.0:
        stop(
            "input metrics are nonzero despite byte-exact SHA"
        )

    print()
    print("inp_embd_ngram new RMSE = 0 exactly")
    print("HIDDEN PROGRESSION AFTER N-GRAM BF16 FIX: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())