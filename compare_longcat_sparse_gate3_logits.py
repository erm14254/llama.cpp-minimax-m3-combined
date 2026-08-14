#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

VOCAB_SIZE = 131072

# Frozen before observing the Sparse parity result. These are the BF16
# complete-final-position-logit tolerances already used in the LongCat-Next
# parity harness.
ATOL = 0.5
RTOL = 0.05

def stop(msg: str):
    raise SystemExit(f"STOP: {msg}")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_f32(path: Path):
    data = path.read_bytes()
    expected = VOCAB_SIZE * 4
    if len(data) != expected:
        stop(f"{path}: expected {expected} bytes ({VOCAB_SIZE} f32), got {len(data)}")
    return [v[0] for v in struct.iter_unpack("<f", data)]

def topk_ids(values, k):
    return sorted(range(len(values)), key=values.__getitem__, reverse=True)[:k]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-bin", required=True)
    ap.add_argument("--cpp-bin", required=True)
    ap.add_argument("--out-json", required=True)
    ns = ap.parse_args()

    hf_path = Path(ns.hf_bin).resolve()
    cpp_path = Path(ns.cpp_bin).resolve()
    out_json = Path(ns.out_json).resolve()

    for path in (hf_path, cpp_path):
        if not path.is_file():
            stop(f"missing logits file: {path}")

    hf = load_f32(hf_path)
    cpp = load_f32(cpp_path)

    for label, vals in (("HF", hf), ("C++", cpp)):
        bad = sum(not math.isfinite(x) for x in vals)
        if bad:
            stop(f"{label} vector contains {bad} non-finite values")

    abs_err = [abs(c - h) for c, h in zip(cpp, hf)]
    max_abs = max(abs_err)
    max_i = abs_err.index(max_abs)
    mean_abs = math.fsum(abs_err) / VOCAB_SIZE
    rmse = math.sqrt(math.fsum(e * e for e in abs_err) / VOCAB_SIZE)

    dot = math.fsum(c * h for c, h in zip(cpp, hf))
    cpp_norm = math.sqrt(math.fsum(c * c for c in cpp))
    hf_norm = math.sqrt(math.fsum(h * h for h in hf))
    cosine = dot / (cpp_norm * hf_norm)

    violations = []
    worst_ratio = -1.0
    worst_ratio_i = -1
    for i, (c, h, e) in enumerate(zip(cpp, hf, abs_err)):
        limit = ATOL + RTOL * abs(h)
        ratio = e / limit if limit > 0 else (0.0 if e == 0 else math.inf)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_ratio_i = i
        if e > limit:
            violations.append(i)

    hf_top1 = max(range(VOCAB_SIZE), key=hf.__getitem__)
    cpp_top1 = max(range(VOCAB_SIZE), key=cpp.__getitem__)

    hf_top20 = topk_ids(hf, 20)
    cpp_top20 = topk_ids(cpp, 20)
    hf_top100 = topk_ids(hf, 100)
    cpp_top100 = topk_ids(cpp, 100)

    top20_overlap = len(set(hf_top20) & set(cpp_top20))
    top100_overlap = len(set(hf_top100) & set(cpp_top100))

    result = {
        "criterion": "abs(cpp-hf) <= atol + rtol * abs(hf)",
        "atol": ATOL,
        "rtol": RTOL,
        "vocab_size": VOCAB_SIZE,
        "hf_sha256": sha256_file(hf_path),
        "cpp_sha256": sha256_file(cpp_path),
        "max_abs_error": max_abs,
        "max_abs_error_index": max_i,
        "hf_at_max_abs_error": hf[max_i],
        "cpp_at_max_abs_error": cpp[max_i],
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "violations": len(violations),
        "violation_fraction": len(violations) / VOCAB_SIZE,
        "first_20_violation_ids": violations[:20],
        "worst_tolerance_ratio": worst_ratio,
        "worst_tolerance_ratio_index": worst_ratio_i,
        "hf_top1": hf_top1,
        "cpp_top1": cpp_top1,
        "top1_agree": hf_top1 == cpp_top1,
        "top20_overlap": top20_overlap,
        "top100_overlap": top100_overlap,
        "hf_top20": hf_top20,
        "cpp_top20": cpp_top20,
    }

    # Gate criterion: every final-position logit satisfies the frozen BF16
    # elementwise tolerance, and the greedy top-1 token agrees.
    passed = len(violations) == 0 and hf_top1 == cpp_top1
    result["passed"] = passed

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"hf_sha256={result['hf_sha256']}")
    print(f"cpp_sha256={result['cpp_sha256']}")
    print(f"max_abs_error={max_abs:.9g} @ token {max_i}")
    print(f"mean_abs_error={mean_abs:.9g}")
    print(f"rmse={rmse:.9g}")
    print(f"cosine_similarity={cosine:.12f}")
    print(f"violations={len(violations)}/{VOCAB_SIZE}")
    print(f"worst_tolerance_ratio={worst_ratio:.9g} @ token {worst_ratio_i}")
    print(f"hf_top1={hf_top1}")
    print(f"cpp_top1={cpp_top1}")
    print(f"top1_agree={hf_top1 == cpp_top1}")
    print(f"top20_overlap={top20_overlap}/20")
    print(f"top100_overlap={top100_overlap}/100")
    print(f"out_json={out_json}")

    if not passed:
        print("GATE-3 HF-v4 vs llama.cpp: FAIL")
        return 1

    print("GATE-3 HF-v4 vs llama.cpp: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
