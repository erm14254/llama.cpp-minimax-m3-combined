#!/usr/bin/env python3
"""Continuous 2x2 factorial analysis of the trunk-norm frozen-512 logits.

Cells (SHA-gated complete final-position logits vectors, 131,072 f32):
  L00 = Stage A            (attn_norm current, ffn_norm current)
  L10 = N1-only            (attn_norm HF,      ffn_norm current)
  L01 = N2-only            (attn_norm current, ffn_norm HF)
  L11 = N1+N2              (attn_norm HF,      ffn_norm HF)

All logits are cast to float64 BEFORE any differencing or norm calculation.

Reported vectors (RMSE / L2 / max-abs each):
  CONDITIONAL/SIMPLE effects (so labeled - not main effects):
    L10-L00 (N1 | ffn current), L11-L01 (N1 | ffn HF),
    L01-L00 (N2 | attn current), L11-L10 (N2 | attn HF)
  AVERAGED MAIN-EFFECT vectors (conventional):
    N1_main = 0.5*((L10-L00) + (L11-L01))
    N2_main = 0.5*((L01-L00) + (L11-L10))
  INTERACTION residual:
    I = L11 - L10 - L01 + L00
Normalized interaction measures: ||I||2 / ||L11-L00||2, and ||I||2 relative
to the larger of ||N1_main||2 / ||N2_main||2.

Factorial interaction magnitude is inferred from these continuous vectors,
never solely from the thresholded violation counts (the four-cell
count/metric table is descriptive endpoint data only).

Measurement-only analysis; no arithmetic on any production path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
LOGITS_NAME = "llamacpp-LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.bin"

CELLS = {
    "L00_stageA": ("cpp_logits_512_stageA",
                   "9d8583e33bc177c0e458a4568da29556206ee1496baabb842c892850f494498a"),
    "L10_N1only": ("cpp_logits_512_stageN1",
                   "69dece2db09fb1512cc609de89c3486f638dcb279c3006b6cd9860ecbc7ab9a6"),
    "L01_N2only": ("cpp_logits_512_stageN2only",
                   "b8067779da58fda3c8ea8472b3f5121562f1599488a0503589feb50293b4c514"),
    "L11_N1N2": ("cpp_logits_512_stageN2",
                 "9a19fce83bbbbfc9ac49fe99e0fed361b176fa64b21d04084e209afec262c963"),
}
VOCAB = 131072


def stop(msg: str) -> None:
    raise SystemExit("STOP: " + msg)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cell(key: str) -> np.ndarray:
    d, expected = CELLS[key]
    p = REPO / d / LOGITS_NAME
    if not p.is_file():
        stop("missing logits: %s" % p)
    got = sha256_file(p)
    if got != expected:
        stop("SHA mismatch for %s: %s != %s" % (key, got, expected))
    v = np.frombuffer(p.read_bytes(), dtype="<f4")
    if v.size != VOCAB:
        stop("size mismatch for %s" % key)
    return v.astype(np.float64)


def stats(v: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt((v ** 2).mean())),
        "l2": float(np.sqrt((v ** 2).sum())),
        "max_abs": float(np.abs(v).max()),
    }


def main() -> int:
    l00 = load_cell("L00_stageA")
    l10 = load_cell("L10_N1only")
    l01 = load_cell("L01_N2only")
    l11 = load_cell("L11_N1N2")

    cond = {
        "N1_given_ffn_current (L10-L00)": l10 - l00,
        "N1_given_ffn_HF (L11-L01)": l11 - l01,
        "N2_given_attn_current (L01-L00)": l01 - l00,
        "N2_given_attn_HF (L11-L10)": l11 - l10,
    }
    n1_main = 0.5 * ((l10 - l00) + (l11 - l01))
    n2_main = 0.5 * ((l01 - l00) + (l11 - l10))
    inter = l11 - l10 - l01 + l00
    both = l11 - l00

    out = {
        "description": "continuous 2x2 factorial over the trunk-norm frozen-512 logits (float64)",
        "inputs": {k: {"dir": v[0], "sha256": v[1]} for k, v in CELLS.items()},
        "conditional_simple_effects": {k: stats(v) for k, v in cond.items()},
        "averaged_main_effects": {
            "N1_main = 0.5*((L10-L00)+(L11-L01))": stats(n1_main),
            "N2_main = 0.5*((L01-L00)+(L11-L10))": stats(n2_main),
        },
        "interaction_residual_I = L11-L10-L01+L00": stats(inter),
        "combined_effect_L11-L00": stats(both),
        "normalized_interaction": {
            "l2_I_over_l2_L11_minus_L00": float(
                np.sqrt((inter ** 2).sum()) / np.sqrt((both ** 2).sum())),
            "l2_I_over_larger_main_l2": float(
                np.sqrt((inter ** 2).sum())
                / max(np.sqrt((n1_main ** 2).sum()), np.sqrt((n2_main ** 2).sum()))),
        },
        "reading_rule": (
            "factorial interaction magnitude is inferred from these continuous "
            "vectors, never solely from the thresholded violation counts; the "
            "four-cell count/metric table is descriptive endpoint data only"
        ),
    }
    out_dir = REPO / "norm_factorial_512"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "norm_factorial.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for k, v in cond.items():
        s = stats(v)
        print("%-36s rmse=%.6e l2=%.4f max=%.4f" % (k, s["rmse"], s["l2"], s["max_abs"]))
    for name, v in (("N1_main", n1_main), ("N2_main", n2_main), ("I(interaction)", inter), ("L11-L00", both)):
        s = stats(v)
        print("%-36s rmse=%.6e l2=%.4f max=%.4f" % (name, s["rmse"], s["l2"], s["max_abs"]))
    ni = out["normalized_interaction"]
    print("||I||/||L11-L00|| = %.6f   ||I||/max(||N1m||,||N2m||) = %.6f"
          % (ni["l2_I_over_l2_L11_minus_L00"], ni["l2_I_over_larger_main_l2"]))
    print("written: %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
