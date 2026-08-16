#!/usr/bin/env python3

"""Compare HF and C++ physical block-0 MLA stage surfaces, full sequence.

Shapes are read from the JSON sidecars written by both sides, never inferred
from file length, so a layout difference cannot be mistaken for an arithmetic
one. Metrics are computed over all 512 x width values, and on any mismatch the
first divergent token index is reported so an early-position K/V divergence
cannot hide behind a matching final row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# Pipeline order. Names match on both sides by construction.
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_side(root: Path, name: str, width: int) -> np.ndarray:
    binary = root / (name + ".bin")
    sidecar = root / (name + ".json")

    if not binary.is_file():
        stop("missing surface %s in %s" % (name, root))

    if not sidecar.is_file():
        stop(
            "missing sidecar for %s in %s -- shape must come from metadata, "
            "not from file length" % (name, root)
        )

    meta = json.loads(sidecar.read_text(encoding="utf-8"))

    shape = meta.get("shape")

    if shape != [EXPECTED_TOKENS, width]:
        stop(
            "%s in %s has shape %s, expected [%d, %d]"
            % (name, root, shape, EXPECTED_TOKENS, width)
        )

    if meta.get("order") != "token-major":
        stop(
            "%s in %s is not token-major (order=%s)"
            % (name, root, meta.get("order"))
        )

    if meta.get("dtype") != "float32-le":
        stop("%s in %s has dtype %s" % (name, root, meta.get("dtype")))

    raw = binary.read_bytes()

    if len(raw) != EXPECTED_TOKENS * width * 4:
        stop(
            "%s in %s is %d bytes, expected %d"
            % (name, root, len(raw), EXPECTED_TOKENS * width * 4)
        )

    if meta.get("bytes") not in (None, len(raw)):
        stop("%s in %s disagrees with its sidecar byte count" % (name, root))

    values = np.frombuffer(raw, dtype="<f4").reshape(EXPECTED_TOKENS, width)

    if not np.isfinite(values).all():
        stop("%s in %s contains nonfinite values" % (name, root))

    return values


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--cpp-dir", required=True)
    ap.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="measured same-surface reproduction noise (rel-RMSE) for context",
    )
    ap.add_argument("--json-out", default=None)

    ns = ap.parse_args()

    hf_dir = Path(ns.hf_dir).resolve()
    cpp_dir = Path(ns.cpp_dir).resolve()

    if not hf_dir.is_dir():
        stop("HF directory missing: %s" % hf_dir)

    if not cpp_dir.is_dir():
        stop("C++ directory missing: %s" % cpp_dir)

    print("hf_dir  =", hf_dir)
    print("cpp_dir =", cpp_dir)

    if ns.noise_floor is not None:
        print("noise_floor (rel-RMSE) = %.6g" % ns.noise_floor)

    print()

    report = {"surfaces": {}}
    first_divergent = None

    header = (
        "%-20s %-8s %12s %12s %12s %14s %10s"
        % ("surface", "sha", "max_abs", "RMSE", "rel_RMSE", "cosine", "exact%")
    )

    print("===== BLOCK-0 MLA STAGE COMPARISON (full sequence) =====")
    print(header)
    print("-" * len(header))

    for name, width in SURFACES:
        hf = load_side(hf_dir, name, width)
        cpp = load_side(cpp_dir, name, width)

        hf_sha = sha256_bytes(hf.tobytes())
        cpp_sha = sha256_bytes(cpp.tobytes())

        equal = hf_sha == cpp_sha

        a = cpp.astype(np.float64)
        b = hf.astype(np.float64)
        diff = a - b

        max_abs = float(np.abs(diff).max())
        rmse = float(np.sqrt((diff ** 2).mean()))
        ref_rms = float(np.sqrt((b ** 2).mean()))
        rel_rmse = rmse / ref_rms if ref_rms > 0 else float("nan")

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float((a * b).sum() / denom) if denom > 0 else float("nan")

        exact = float((cpp == hf).mean() * 100.0)

        row_bad = np.any(cpp != hf, axis=1)
        n_bad_rows = int(row_bad.sum())
        first_row = int(np.argmax(row_bad)) if n_bad_rows else -1

        entry = {
            "shape": [EXPECTED_TOKENS, width],
            "byte_equal": equal,
            "hf_sha256": hf_sha,
            "cpp_sha256": cpp_sha,
            "max_abs": max_abs,
            "rmse": rmse,
            "rel_rmse": rel_rmse,
            "cosine": cosine,
            "exact_element_pct": exact,
            "divergent_rows": n_bad_rows,
            "first_divergent_token": first_row,
            "hf_final_row_sha256": sha256_bytes(hf[-1].tobytes()),
            "cpp_final_row_sha256": sha256_bytes(cpp[-1].tobytes()),
        }

        if n_bad_rows:
            row_diff = diff[first_row]
            entry["first_divergent_row_max_abs"] = float(
                np.abs(row_diff).max()
            )
            entry["first_divergent_row_rmse"] = float(
                np.sqrt((row_diff ** 2).mean())
            )

        report["surfaces"][name] = entry

        if not equal and first_divergent is None:
            first_divergent = name

        print(
            "%-20s %-8s %12.6g %12.6g %12.6g %14.10f %9.2f%%"
            % (
                name,
                "MATCH" if equal else "DIFFER",
                max_abs,
                rmse,
                rel_rmse,
                cosine,
                exact,
            )
        )

    print()
    print("===== DIVERGENCE DETAIL =====")

    for name, _ in SURFACES:
        e = report["surfaces"][name]

        if e["byte_equal"]:
            print("%-20s byte-exact across all 512 tokens" % name)
            continue

        detail = (
            "%-20s first divergent token=%d  divergent rows=%d/%d"
            % (
                name,
                e["first_divergent_token"],
                e["divergent_rows"],
                EXPECTED_TOKENS,
            )
        )

        if "first_divergent_row_max_abs" in e:
            detail += "  row_max_abs=%.6g row_rmse=%.6g" % (
                e["first_divergent_row_max_abs"],
                e["first_divergent_row_rmse"],
            )

        print(detail)

    print()

    if first_divergent is None:
        print("ALL SIX SURFACES BYTE-EXACT")
    else:
        print("FIRST DIVERGENT SURFACE:", first_divergent)

        if ns.noise_floor is not None:
            rel = report["surfaces"][first_divergent]["rel_rmse"]
            ratio = rel / ns.noise_floor if ns.noise_floor else float("nan")
            print(
                "  rel-RMSE %.6g = %.2fx the supplied noise floor %.6g"
                % (rel, ratio, ns.noise_floor)
            )

    report["first_divergent_surface"] = first_divergent

    if ns.json_out:
        Path(ns.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print()
        print("report written to", ns.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
