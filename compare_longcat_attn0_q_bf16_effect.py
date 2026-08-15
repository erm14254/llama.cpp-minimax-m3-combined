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

BASE_EPS1E5 = (
    ROOT
    / "cpp_attn0_hf_rmsnorm_512"
    / "logical0_attn0_resid.bin"
)

BASE_EPS1E6 = (
    ROOT
    / "cpp_attn0_hf_rmsnorm_mlaeps1e6_512"
    / "logical0_attn0_resid.bin"
)

QBF16 = (
    ROOT
    / "cpp_attn0_hf_rmsnorm_mlaeps1e6_qbf16_512"
    / "logical0_attn0_resid.bin"
)

EXPECTED_HF_SHA = (
    "2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177"
)

EXPECTED_EPS1E5_SHA = (
    "8ea9b911d4810982af4186e66562cb5f316e7a0a9c2439101f6654eb10887dfd"
)

EXPECTED_EPS1E6_SHA = (
    "c2b8473b9d044ba50a978e7249a694b81f111cd5bc434b585ecd776a922c2199"
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


def main() -> int:
    hf_sha = sha256_file(HF)
    eps1e5_sha = sha256_file(BASE_EPS1E5)
    eps1e6_sha = sha256_file(BASE_EPS1E6)
    qbf16_sha = sha256_file(QBF16)

    if hf_sha != EXPECTED_HF_SHA:
        stop("HF oracle changed: %s" % hf_sha)

    if eps1e5_sha != EXPECTED_EPS1E5_SHA:
        stop(
            "eps1e-5 baseline changed: %s"
            % eps1e5_sha
        )

    if eps1e6_sha != EXPECTED_EPS1E6_SHA:
        stop(
            "eps1e-6 baseline changed: %s"
            % eps1e6_sha
        )

    hf = load(HF)
    eps1e5 = load(BASE_EPS1E5)
    eps1e6 = load(BASE_EPS1E6)
    qbf16 = load(QBF16)

    m_1e5 = metrics(eps1e5, hf)
    m_1e6 = metrics(eps1e6, hf)
    m_q = metrics(qbf16, hf)

    q_vs_1e6 = metrics(qbf16, eps1e6)

    print(
        "{:<14} {:>11} {:>11} {:>11} {:>11} "
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
    print("-" * 96)

    print_row("eps1e-5", m_1e5)
    print_row("eps1e-6", m_1e6)
    print_row("q_bf16", m_q)

    print()
    print("===== SHA256 =====")
    print("HF       =", hf_sha)
    print("eps1e-5  =", eps1e5_sha)
    print("eps1e-6  =", eps1e6_sha)
    print("q_bf16   =", qbf16_sha)

    print()
    print("===== Q-BF16 EFFECT =====")
    print("eps1e-5 rel_rmse =", m_1e5["rel"])
    print("eps1e-6 rel_rmse =", m_1e6["rel"])
    print("q_bf16  rel_rmse =", m_q["rel"])

    print()
    print(
        "q_bf16 / eps1e-6 RMSE ratio =",
        m_q["rmse"] / m_1e6["rmse"],
    )

    print(
        "q_bf16 vs eps1e-6 reduction fraction =",
        1.0 - (
            m_q["rmse"]
            / m_1e6["rmse"]
        ),
    )

    print(
        "q_bf16 / eps1e-5 RMSE ratio =",
        m_q["rmse"] / m_1e5["rmse"],
    )

    print(
        "q_bf16 vs eps1e-5 reduction fraction =",
        1.0 - (
            m_q["rmse"]
            / m_1e5["rmse"]
        ),
    )

    print()
    print("q_bf16 versus eps1e-6 baseline:")
    print("  max_abs =", q_vs_1e6["max_abs"])
    print("  rmse    =", q_vs_1e6["rmse"])
    print("  rel     =", q_vs_1e6["rel"])

    print()
    print("ATTN0 Q-BF16 EFFECT COMPARISON: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())