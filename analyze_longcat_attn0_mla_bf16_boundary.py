#!/usr/bin/env python3

"""Classify each differing block-0 MLA surface as a BF16 boundary or a real divergence.

The C++ side computes these surfaces in F32 where HF stores them in BF16. A
surface whose C++ output becomes (near-)identical to HF once rounded to the
BF16 lattice is a missing precision boundary, not an arithmetic difference. A
surface that does not improve under rounding is a genuine divergence.

Rounding is round-to-nearest-even on the F32 bit pattern, matching what a
`ggml_fp32_to_bf16`-style cast does. HF-side lattice occupancy is reported too,
so "already BF16" is verified rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SURFACES = [
    ("q_a_proj", 1536),
    ("q_a_layernorm", 1536),
    ("q_b_proj", 6144),
    ("kv_a_proj_with_mqa", 576),
    ("kv_a_layernorm", 512),
    ("o_proj", 3072),
]

EXPECTED_TOKENS = 512


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def load(root: Path, name: str, width: int) -> np.ndarray:
    binary = root / (name + ".bin")
    sidecar = root / (name + ".json")

    if not binary.is_file():
        stop("missing surface %s in %s" % (name, root))
    if not sidecar.is_file():
        stop("missing sidecar for %s in %s" % (name, root))

    meta = json.loads(sidecar.read_text(encoding="utf-8"))

    if meta.get("shape") != [EXPECTED_TOKENS, width]:
        stop("%s in %s has shape %s" % (name, root, meta.get("shape")))
    if meta.get("order") != "token-major":
        stop("%s in %s is not token-major" % (name, root))
    if meta.get("dtype") != "float32-le":
        stop("%s in %s has dtype %s" % (name, root, meta.get("dtype")))

    raw = binary.read_bytes()

    if len(raw) != EXPECTED_TOKENS * width * 4:
        stop("%s in %s is %d bytes" % (name, root, len(raw)))

    values = np.frombuffer(raw, dtype="<f4").reshape(EXPECTED_TOKENS, width)

    if not np.isfinite(values).all():
        stop("%s in %s contains nonfinite values" % (name, root))

    return values


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round F32 to the BF16 lattice, round-to-nearest-even, staying in F32."""
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def on_bf16_lattice(values: np.ndarray) -> float:
    return float((to_bf16(values) == values).mean() * 100.0)


def metrics(cpp: np.ndarray, hf: np.ndarray) -> dict:
    a = cpp.astype(np.float64)
    b = hf.astype(np.float64)
    diff = a - b

    ref_rms = float(np.sqrt((b ** 2).mean()))
    rmse = float(np.sqrt((diff ** 2).mean()))

    return {
        "max_abs": float(np.abs(diff).max()),
        "rmse": rmse,
        "rel_rmse": rmse / ref_rms if ref_rms > 0 else float("nan"),
        "exact_pct": float((cpp == hf).mean() * 100.0),
        "divergent_rows": int(np.any(cpp != hf, axis=1).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--cpp-dir", required=True)
    ap.add_argument("--json-out", default=None)
    ns = ap.parse_args()

    hf_dir = Path(ns.hf_dir).resolve()
    cpp_dir = Path(ns.cpp_dir).resolve()

    print("hf_dir  =", hf_dir)
    print("cpp_dir =", cpp_dir)
    print()

    header = "%-20s %10s %12s %12s %10s %12s %10s" % (
        "surface", "hf_on_bf16", "exact% raw", "exact% bf16",
        "rel raw", "rel bf16", "verdict",
    )
    print("===== BF16 BOUNDARY ANALYSIS (full sequence, 512 tokens) =====")
    print(header)
    print("-" * len(header))

    report = {}

    for name, width in SURFACES:
        hf = load(hf_dir, name, width)
        cpp = load(cpp_dir, name, width)

        raw = metrics(cpp, hf)
        rounded = metrics(to_bf16(cpp), hf)
        lattice = on_bf16_lattice(hf)

        if raw["exact_pct"] == 100.0:
            verdict = "byte-exact"
        elif rounded["exact_pct"] >= 99.0:
            verdict = "BF16 boundary"
        elif rounded["rel_rmse"] < raw["rel_rmse"] * 0.5:
            verdict = "mostly BF16"
        else:
            verdict = "REAL"

        report[name] = {
            "hf_on_bf16_lattice_pct": lattice,
            "raw": raw,
            "bf16_rounded": rounded,
            "verdict": verdict,
        }

        print("%-20s %9.2f%% %11.2f%% %11.2f%% %10.4g %12.4g %10s" % (
            name, lattice, raw["exact_pct"], rounded["exact_pct"],
            raw["rel_rmse"], rounded["rel_rmse"], verdict,
        ))

    print()
    print("===== READING =====")
    for name, _ in SURFACES:
        e = report[name]
        if e["verdict"] == "byte-exact":
            print("%-20s identical to HF across all 512 tokens" % name)
        elif e["verdict"] in ("BF16 boundary", "mostly BF16"):
            print("%-20s %s: rel-RMSE %.4g -> %.4g under BF16 rounding, exact %.2f%% -> %.2f%%" % (
                name, e["verdict"], e["raw"]["rel_rmse"], e["bf16_rounded"]["rel_rmse"],
                e["raw"]["exact_pct"], e["bf16_rounded"]["exact_pct"],
            ))
        else:
            print("%-20s REAL divergence: rel-RMSE %.4g -> %.4g under BF16 rounding (not explained)" % (
                name, e["raw"]["rel_rmse"], e["bf16_rounded"]["rel_rmse"],
            ))

    first_real = next(
        (n for n, _ in SURFACES if report[n]["verdict"] == "REAL"), None
    )
    print()
    print("FIRST SURFACE WITH A REAL (non-BF16) DIVERGENCE:", first_real or "none")

    if ns.json_out:
        Path(ns.json_out).write_text(
            json.dumps(
                {"surfaces": report, "first_real_divergence": first_real},
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print()
        print("report written to", ns.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
