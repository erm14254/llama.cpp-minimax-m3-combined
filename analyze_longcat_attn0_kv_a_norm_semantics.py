#!/usr/bin/env python3

"""Locate and explain the first genuine block-0 MLA divergence.

`compare_longcat_attn0_mla_stages.py` reports WHERE the C++ and HF surfaces
part company; `analyze_longcat_attn0_mla_bf16_boundary.py` separates missing
precision boundaries from real arithmetic differences. This script explains the
first surface that survives both, `kv_a_layernorm`, by reconstructing the norm
offline and sweeping cast orderings against the authoritative HF surface.

Three stages, each a falsifiable check:

  1. Sanity   -- the offline F32 model must reproduce the C++ surface exactly,
                 proving the reconstruction and the GGUF weight are correct.
  2. Control  -- bf16(C++ kv_a_proj) must equal HF kv_a_proj, proving the
                 upstream surface differs only by an output rounding boundary.
  3. Sweep    -- feed HF's own BF16 input through candidate cast orderings and
                 find which one reproduces HF's kv_a_layernorm.

The norm weight is read from the GGUF the C++ run actually loaded, so no
HF/GGUF weight equivalence is assumed. Nothing here changes arithmetic; it is
measurement only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KV_LORA_RANK = 512
WIDTH_PROJ = 576
N_TOK = 512
WEIGHT_NAME = "blk.0.attn_kv_a_norm.weight"


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round F32 to the BF16 lattice, round-to-nearest-even, staying in F32."""
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def load(root: Path, name: str, width: int) -> np.ndarray:
    path = root / (name + ".bin")
    if not path.is_file():
        stop("missing %s" % path)
    raw = path.read_bytes()
    if len(raw) != N_TOK * width * 4:
        stop("%s is %d bytes, expected %d" % (path, len(raw), N_TOK * width * 4))
    return np.frombuffer(raw, dtype="<f4").reshape(N_TOK, width)


def score(tag: str, got: np.ndarray, ref: np.ndarray) -> dict:
    diff = got.astype(np.float64) - ref.astype(np.float64)
    rmse = float(np.sqrt((diff ** 2).mean()))
    ref_rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
    entry = {
        "rel_rmse": rmse / ref_rms if ref_rms > 0 else float("nan"),
        "exact_pct": float((got == ref).mean() * 100.0),
        "max_abs": float(np.abs(diff).max()),
    }
    print("%-56s rel_RMSE=%-12.6g exact=%6.2f%%  max_abs=%.6g"
          % (tag, entry["rel_rmse"], entry["exact_pct"], entry["max_abs"]))
    return entry


def read_norm_weight(gguf_path: str, gguf_py: str) -> np.ndarray:
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader  # noqa: PLC0415

    for tensor in GGUFReader(gguf_path, "r").tensors:
        if tensor.name == WEIGHT_NAME:
            weight = np.array(tensor.data, dtype=np.float32).reshape(-1)
            if weight.size != KV_LORA_RANK:
                stop("%s has %d elements, expected %d"
                     % (WEIGHT_NAME, weight.size, KV_LORA_RANK))
            return weight

    stop("%s not found in %s" % (WEIGHT_NAME, gguf_path))


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    """ggml_rms_norm: F32 activation, F64 variance accumulation."""
    x32 = x.astype(np.float32)
    var = (x32.astype(np.float64) ** 2).mean(axis=1)
    return x32 * (1.0 / np.sqrt(var + eps)).astype(np.float32)[:, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--cpp-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gguf-py", default="gguf-py")
    ap.add_argument("--json-out", default=None)
    ns = ap.parse_args()

    hf_dir = Path(ns.hf_dir).resolve()
    cpp_dir = Path(ns.cpp_dir).resolve()

    weight = read_norm_weight(ns.gguf, str(Path(ns.gguf_py).resolve()))

    hf_proj = load(hf_dir, "kv_a_proj_with_mqa", WIDTH_PROJ)[:, :KV_LORA_RANK]
    cpp_proj = load(cpp_dir, "kv_a_proj_with_mqa", WIDTH_PROJ)[:, :KV_LORA_RANK]
    hf_norm = load(hf_dir, "kv_a_layernorm", KV_LORA_RANK)
    cpp_norm = load(cpp_dir, "kv_a_layernorm", KV_LORA_RANK)

    report = {"lattice": {
        "weight_on_bf16_pct": float((to_bf16(weight) == weight).mean() * 100.0),
        "hf_kv_a_layernorm_on_bf16_pct": float((to_bf16(hf_norm) == hf_norm).mean() * 100.0),
        "hf_kv_a_proj_on_bf16_pct": float((to_bf16(hf_proj) == hf_proj).mean() * 100.0),
    }}

    print("=== BF16 lattice occupancy ===")
    for k, v in report["lattice"].items():
        print("%-40s %6.2f%%" % (k, v))
    print()

    print("=== 1. sanity: offline F32 model reproduces the C++ surface ===")
    report["sanity_offline_vs_cpp"] = score(
        "offline_f32(cpp kv_a_proj) vs C++ kv_a_layernorm",
        normalize(cpp_proj, 1e-6) * weight[None, :], cpp_norm)
    print()

    print("=== 2. control: the upstream surface is a pure output boundary ===")
    identical = bool((to_bf16(cpp_proj) == hf_proj).all())
    report["control_bf16_cpp_proj_equals_hf_proj"] = identical
    print("bf16(C++ kv_a_proj[:, :512]) == HF kv_a_proj[:, :512] : %s" % identical)
    print()

    print("=== 3. sweep: which cast ordering reproduces HF kv_a_layernorm? ===")
    report["sweep"] = {}
    best = None

    for eps in (1e-6, 1e-5, 1e-8):
        hn = normalize(hf_proj, eps)
        variants = {
            "A f32 norm, f32 weight, f32 out (llama.cpp today)":
                hn * weight[None, :],
            "B f32 norm, f32 weight, bf16 out":
                to_bf16(hn * weight[None, :]),
            "C bf16(norm), f32 weight, bf16 out":
                to_bf16(to_bf16(hn) * weight[None, :]),
            "D bf16(norm), bf16 weight, bf16 out (HF RMSNorm pattern)":
                to_bf16(to_bf16(hn) * to_bf16(weight)[None, :]),
            "E bf16(norm), bf16 weight, f32 out":
                to_bf16(hn) * to_bf16(weight)[None, :],
        }
        for tag, got in variants.items():
            entry = score("eps=%-6g %s" % (eps, tag), got, hf_norm)
            report["sweep"]["eps=%g %s" % (eps, tag)] = entry
            if best is None or (entry["exact_pct"], -entry["rel_rmse"]) > best[0]:
                best = ((entry["exact_pct"], -entry["rel_rmse"]), eps, tag, entry)
        print()

    report["best"] = {"eps": best[1], "variant": best[2], **best[3]}
    print("BEST: eps=%g  %s  ->  rel_RMSE=%.6g exact=%.2f%%"
          % (best[1], best[2], best[3]["rel_rmse"], best[3]["exact_pct"]))

    if ns.json_out:
        Path(ns.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print()
        print("report written to", ns.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
