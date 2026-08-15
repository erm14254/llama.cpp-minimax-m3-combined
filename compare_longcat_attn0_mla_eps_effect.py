#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF = (
    ROOT
    / "hf_logical0_stages_512_v4"
    / "attn0_resid.bin"
)

OLD = (
    ROOT
    / "cpp_attn0_hf_rmsnorm_512"
    / "logical0_attn0_resid.bin"
)

NEW = (
    ROOT
    / "cpp_attn0_hf_rmsnorm_mlaeps1e6_512"
    / "logical0_attn0_resid.bin"
)

EXPECTED_HF_SHA = (
    "2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177"
)

EXPECTED_OLD_SHA = (
    "8ea9b911d4810982af4186e66562cb5f316e7a0a9c2439101f6654eb10887dfd"
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
        stop(
            "%s shape is %s"
            % (
                path,
                x.shape,
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

    d = c - r
    ad = np.abs(d)

    rmse = float(
        np.sqrt(
            np.mean(d * d)
        )
    )

    ref_rms = float(
        np.sqrt(
            np.mean(r * r)
        )
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
        float(
            np.dot(c, r)
            / denom
        )
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


def print_row(
    label: str,
    m: dict[str, float | int],
) -> None:
    print(
        "{:<12} {:11.6g} {:11.6g} {:11.6g} "
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


def main() -> int:
    hf_sha = sha256_file(HF)
    old_sha = sha256_file(OLD)
    new_sha = sha256_file(NEW)

    if hf_sha != EXPECTED_HF_SHA:
        stop(
            "HF residual oracle changed: %s"
            % hf_sha
        )

    if old_sha != EXPECTED_OLD_SHA:
        stop(
            "old main-RMSNorm baseline changed: %s"
            % old_sha
        )

    hf = load(HF)
    old = load(OLD)
    new = load(NEW)

    old_m = metrics(old, hf)
    new_m = metrics(new, hf)
    delta_m = metrics(new, old)

    print(
        "{:<12} {:>11} {:>11} {:>11} {:>11} "
        "{:>14} {:>11}".format(
            "variant",
            "max_abs",
            "mean_abs",
            "rmse",
            "rel_rmse",
            "cosine",
            "exact",
        )
    )

    print("-" * 94)

    print_row("old_eps1e-5", old_m)
    print_row("new_eps1e-6", new_m)

    print()
    print("===== SHA256 =====")
    print("HF          =", hf_sha)
    print("old eps1e-5 =", old_sha)
    print("new eps1e-6 =", new_sha)

    print()
    print("===== EPSILON EFFECT =====")

    print(
        "old rel_rmse =",
        old_m["rel"],
    )

    print(
        "new rel_rmse =",
        new_m["rel"],
    )

    print(
        "new/old RMSE ratio =",
        new_m["rmse"] / old_m["rmse"],
    )

    print(
        "RMSE reduction fraction =",
        1.0 - (
            new_m["rmse"]
            / old_m["rmse"]
        ),
    )

    print()
    print("new versus old:")
    print(
        "  max_abs =",
        delta_m["max_abs"],
    )
    print(
        "  rmse =",
        delta_m["rmse"],
    )
    print(
        "  rel_rmse =",
        delta_m["rel"],
    )

    print()
    print("ATTN0 MLA EPS EFFECT COMPARISON: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())