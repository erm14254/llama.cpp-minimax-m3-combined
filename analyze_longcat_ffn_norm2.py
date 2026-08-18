#!/usr/bin/env python3
"""ffn_norm-2 / HF post_attention_layernorm[0] causal comparison (block 2,
exact predecessor via LONGCAT_FFN_INP2_INJECT).

All-input rule BEFORE any attribution (verified, never assumed):
  - weight: GGUF blk.2.ffn_norm.weight must be the exact numerical widening
    of HF model.layers.1.post_attention_layernorm.0.weight (3072/3072);
  - eps, C++ side: GGUF attention.layer_norm_rms_epsilon metadata (expected
    1e-5; build_norm at llama-graph.cpp:1790 consumes hparams.f_norm_rms_eps;
    call site longcat-flash-ngram.cpp:1053);
  - eps, HF side: the runtime-instantiated post_attention_layernorm[0]
    variance_epsilon recorded by the capture (summary.json runtime_eps_gate);
  - input identity: C++ landing dump == HF capture attn0_resid == the
    committed 4718460b... oracle;
  - dtype/layout: [512, 3072] float32-le token-major on both sides.

Pre-registered reconstructions (committed model recipe: F32 activation,
f64-accumulated variance):
  A  = f32_norm(x, eps_cpp) * w        vs the C++ ffn_norm-2 dump
       - established F32 reduction-noise protocol: max f32-ulp <= 4;
         exact count recorded; BF16 agreement reported for this comparison
         is explicitly A-MODEL-VS-C++ agreement (NOT C++-vs-HF recovery).
  D(eps_hf) = bf16(bf16(f32_norm(x, eps_hf)) * w)  vs the HF ffn0_norm dump
       - the SOLE mechanism-closure candidate; CLOSED only on whole-tensor
         byte-exact 1,572,864/1,572,864; any residue ulp-quantified and
         labeled near-closure/model residue.
  D(1e-6) vs the HF dump - EPSILON-DISCRIMINATION DIAGNOSTIC, not a
       validity gate: the actual HF epsilon is established by the runtime
       gate + source/config. If D(1e-6) differs from HF, that is numerical
       discrimination supporting the runtime-verified value; if it also
       reproduces HF (BF16 quantization can absorb the eps difference),
       epsilon is recorded as non-identifiable from this tensor alone.
       Neither outcome invalidates the experiment or capture.
  Cross-classification: C++ dump vs HF dump - raw-eq, C++-vs-HF bf16
       recovery count, rel-RMSE (irreducibility class measured, not assumed
       from attn_norm-2).

Measurement-only; no arithmetic is performed on any production path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
HF_DIR = Path(r"D:\lc_block1_ffn_norm_512")
ORACLE = Path(r"D:\lc_block1_stages_512\attn0_resid.bin")
GGUF = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf")
CKPT = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved")

ORACLE_SHA = "4718460be4d2bb0243c4b9bcf76e20ca4b8d5a0f35ec3717ca6b8dd5cb5f73c3"
N_TOK, HID = 512, 3072
TOTAL = N_TOK * HID


def stop(msg: str) -> None:
    raise SystemExit("STOP: " + msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mat(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) != TOTAL * 4:
        stop("size mismatch: %s (%d)" % (path, len(raw)))
    return np.frombuffer(raw, dtype="<f4").reshape(N_TOK, HID)


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    var = (x32.astype(np.float64) ** 2).mean(axis=1)
    return x32 * (1.0 / np.sqrt(var + eps)).astype(np.float32)[:, None]


def f32_ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua = a.view(np.uint32).astype(np.int64)
    ub = b.view(np.uint32).astype(np.int64)
    ka = np.where(ua >= 0x80000000, 0xFFFFFFFF - ua, ua + 0x80000000)
    kb = np.where(ub >= 0x80000000, 0xFFFFFFFF - ub, ub + 0x80000000)
    return np.abs(ka - kb)


def bf16_ulp_stats(a: np.ndarray, b: np.ndarray) -> dict:
    ab = (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bb = (b.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    mism = ab != bb
    n = int(mism.sum())
    if n == 0:
        return {"mismatches": 0}
    a64 = ab.astype(np.int64)
    b64 = bb.astype(np.int64)
    ka = np.where((a64 & 0x8000) != 0, 0xFFFF - a64, a64 | 0x8000)
    kb = np.where((b64 & 0x8000) != 0, 0xFFFF - b64, b64 | 0x8000)
    d = np.abs(ka - kb)[mism]
    pos = np.argwhere(mism.reshape(N_TOK, HID))
    return {
        "mismatches": n,
        "total": TOTAL,
        "max_bf16_ulp": int(d.max()),
        "one_ulp_count": int((d == 1).sum()),
        "positions_token_ch_first32": [tuple(int(x) for x in p) for p in pos[:32]],
    }


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    denom = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
    return float(np.sqrt((d ** 2).mean())) / denom if denom > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(REPO / "cpp_resid_walk_injectffn_ffnNorm_512"))
    ap.add_argument("--out-dir", default=str(REPO / "ffn_norm2_512"))
    ns = ap.parse_args()
    run_dir = Path(ns.run_dir)
    out_dir = Path(ns.out_dir)
    if not run_dir.is_dir():
        stop("run dir missing: %s" % run_dir)

    # ---- all-input rule: input identity ------------------------------------
    oracle_sha = sha256_file(ORACLE)
    if oracle_sha != ORACLE_SHA:
        stop("attn0_resid oracle SHA mismatch: %s" % oracle_sha)
    landing = run_dir / "block1_attn0_resid_full.bin"
    landing_sha = sha256_file(landing)
    if landing_sha != ORACLE_SHA:
        stop("C++ landing dump != oracle: %s" % landing_sha)
    hf_resid = HF_DIR / "attn0_resid.bin"
    if sha256_file(hf_resid) != ORACLE_SHA:
        stop("HF capture attn0_resid != oracle")
    x = load_mat(ORACLE)

    # ---- all-input rule: HF runtime eps from the capture summary -----------
    summary = json.loads((HF_DIR / "summary.json").read_text(encoding="utf-8"))
    eps_gate = summary["runtime_eps_gate"]
    eps_hf = float(eps_gate["post_attention_layernorm_0_variance_epsilon"])
    cfg_eps = float(eps_gate["config_rms_norm_eps"])

    # HF dump SHA cross-checked against the capture's own manifest.
    hf_dump_path = HF_DIR / "ffn0_norm.bin"
    hf_dump_sha = sha256_file(hf_dump_path)
    man = {}
    for line in (HF_DIR / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            man[name.strip()] = sha.lower()
    if man.get("ffn0_norm.bin") != hf_dump_sha:
        stop("HF ffn0_norm.bin not consistent with capture manifest")
    hf = load_mat(hf_dump_path)

    # ---- all-input rule: GGUF eps + weight widening-equivalence ------------
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    r = GGUFReader(str(GGUF), "r")
    eps_cpp = None
    for k, f in r.fields.items():
        if "layer_norm_rms_epsilon" in k:
            eps_cpp = float(f.parts[f.data[0]][0])
    if eps_cpp is None:
        stop("GGUF eps metadata missing")
    gw = None
    for t in r.tensors:
        if t.name == "blk.2.ffn_norm.weight":
            gw = np.asarray(t.data, dtype=np.float32).reshape(-1)
    if gw is None or gw.size != HID:
        stop("GGUF blk.2.ffn_norm.weight missing/odd")
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    hkey = "model.layers.1.post_attention_layernorm.0.weight"
    with safe_open(str(CKPT / idx["weight_map"][hkey]), framework="pt") as f:
        hw = f.get_tensor(hkey)
    hw32 = hw.float().numpy().reshape(-1)
    widen_eq = int((gw == hw32).sum())
    if widen_eq != HID:
        stop("ffn_norm weight widening-equivalence FAIL: %d/%d" % (widen_eq, HID))
    w = gw

    # ---- C++ dump ----------------------------------------------------------
    cpp_path = run_dir / "block1_ffn0_norm_full.bin"
    cpp = load_mat(cpp_path)
    cpp_sha = sha256_file(cpp_path)

    # ---- reconstructions ---------------------------------------------------
    a_model = (normalize(x, eps_cpp) * w).astype("<f4")
    a_ulp = f32_ulp_diff(a_model, cpp)
    a_stats = {
        "protocol": "established F32 reduction-noise: max f32-ulp <= 4; exact count recorded",
        "exact_count": int((a_model.view(np.uint32) == cpp.view(np.uint32)).sum()),
        "total": TOTAL,
        "max_f32_ulp": int(a_ulp.max()),
        "a_model_vs_cpp_bf16_agreement": int(
            ((to_bf16(a_model).view(np.uint32)) == (to_bf16(cpp).view(np.uint32))).sum()),
        "a_model_vs_cpp_bf16_agreement_note": (
            "A-MODEL-VS-C++ agreement only - NOT C++-vs-HF BF16 recovery"),
        "verdict": None,
    }
    a_stats["verdict"] = "PASS" if a_stats["max_f32_ulp"] <= 4 else "FAIL"

    d_hf = to_bf16(to_bf16(normalize(x, eps_hf)) * w).astype("<f4")
    d_eq = int((d_hf.view(np.uint32) == hf.view(np.uint32)).sum())
    d_stats = {
        "eps_used": eps_hf,
        "byte_exact_count": d_eq,
        "total": TOTAL,
        "closure": "CLOSED (whole-tensor byte-exact)" if d_eq == TOTAL else "NEAR-CLOSURE/model residue - NOT closed",
    }
    if d_eq != TOTAL:
        d_stats["ulp_analysis"] = bf16_ulp_stats(d_hf, hf)

    d_alt = to_bf16(to_bf16(normalize(x, 1e-6)) * w).astype("<f4")
    alt_eq = int((d_alt.view(np.uint32) == hf.view(np.uint32)).sum())
    if alt_eq == TOTAL:
        eps_diag = ("D(1e-6) ALSO reproduces the HF dump byte-exactly: epsilon is "
                    "NON-IDENTIFIABLE from this tensor alone (BF16 quantization "
                    "absorbs the difference); the HF epsilon rests on the runtime "
                    "gate + source/config, which is sufficient")
    else:
        eps_diag = ("D(1e-6) differs from the HF dump (%d/%d equal): numerical "
                    "discrimination SUPPORTING the runtime-verified eps %r"
                    % (alt_eq, TOTAL, eps_hf))

    cross = {
        "raw_eq": int((cpp.view(np.uint32) == hf.view(np.uint32)).sum()),
        "cpp_vs_hf_bf16_recovery": int(
            ((to_bf16(cpp).view(np.uint32)) == (hf.view(np.uint32))).sum()),
        "cpp_vs_hf_bf16_recovery_note": "count of bf16(cpp) == hf (HF dump is bf16-on-lattice)",
        "rel_rmse": rel_rmse(cpp, hf),
        "total": TOTAL,
    }

    out = {
        "experiment": "ffn_norm-2 / HF post_attention_layernorm[0] under exact ffn_inp-2 predecessor",
        "all_input_rule": {
            "input_identity": "C++ landing == HF capture attn0_resid == oracle %s" % ORACLE_SHA,
            "weight_widening_equivalence": "%d/%d (GGUF blk.2.ffn_norm.weight == widen(HF %s))" % (widen_eq, HID, hkey),
            "eps_cpp_gguf_metadata": eps_cpp,
            "eps_cpp_source_cites": [
                "llama-graph.cpp:1790 build_norm LLM_NORM_RMS -> hparams.f_norm_rms_eps",
                "longcat-flash-ngram.cpp:1053 build_norm(ffn_inp, ffn_norm, ...)",
            ],
            "eps_hf_runtime_gate": eps_gate,
            "attribution": "PERMITTED at this surface (exact input + verified parameters)",
        },
        "cpp_dump_sha256": cpp_sha,
        "hf_dump_sha256": hf_dump_sha,
        "A_model_vs_cpp": a_stats,
        "D_model_vs_hf": d_stats,
        "epsilon_discrimination_diagnostic": eps_diag,
        "cross_classification_cpp_vs_hf": cross,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ffn_norm2.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("all-input rule: input identity OK; weight widening %d/%d; eps cpp=%r hf(runtime)=%r cfg=%r"
          % (widen_eq, HID, eps_cpp, eps_hf, cfg_eps))
    print("A-model vs C++: %s (max_f32_ulp=%d, exact=%d/%d, A-model-vs-C++ bf16-agree=%d)"
          % (a_stats["verdict"], a_stats["max_f32_ulp"], a_stats["exact_count"], TOTAL,
             a_stats["a_model_vs_cpp_bf16_agreement"]))
    print("D(eps_hf) vs HF: %s (%d/%d)" % (d_stats["closure"], d_eq, TOTAL))
    print("eps diagnostic: %s" % eps_diag)
    print("cross C++ vs HF: raw_eq=%d, bf16-recovery=%d/%d, rel_rmse=%.6e"
          % (cross["raw_eq"], cross["cpp_vs_hf_bf16_recovery"], TOTAL, cross["rel_rmse"]))
    print("written: %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
