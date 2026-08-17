#!/usr/bin/env python3

"""Corrected C++ attention-core mechanism model (V7'') -- offline, static.

The earlier V7 attribution stopped correctly: it missed the measured kqv_out
by 6.6x the reference discrepancy, and its prediction that kqv_out lies on
the BF16 lattice was falsified (108/2097152). Re-deriving the ggml dispatch
resolved the falsification: ggml-cuda.cu:1510-1519 sets prefer_f32_output for
BF16 compute on NVIDIA, so the BF16-weight GEMMs (absorption, wv_b) write F32
output DIRECTLY (C = CUDA_R_32F, no temporary, no rounding). The corrected
model is:

  1. absorb   : BF16 x BF16, f32-acc, F32 OUT (no round). Operands are
                bf16-lattice, so an IEEE f32 matmul of the widened operands is
                the exact semantics up to accumulation order.
  2. scores   : F32 x F32 GemmBatchedEx under the handle-global
                CUBLAS_TF32_TENSOR_OP_MATH -> TF32. The q_abs half of Qcur is
                an off-lattice F32 GEMM output, so TF32 operand quantization
                is genuinely lossy there; the K side is bf16-lattice
                (lossless).
  3. softmax  : f32, C++ f32 kq_scale constant, additive -inf causal mask.
  4. context  : F32 x F32 TF32 -> quantizes the off-lattice probs; latent V
                is lattice (lossless).
  5. wv_b     : src1 (latent context, off-lattice) CONVERTED TO BF16 (lossy),
                BF16 weights, f32-acc, F32 OUT (no round).

Result (recorded in the JSON): V7'' reproduces the measured kqv_out to
0.98x the measured reference reduction discrepancy -- a collapse -- while a
no-TF32 control is 3x worse, and V7'' sits at the same distance from the HF
oracle as the real C++ does.

TF32 RNE quantizer: 13 fraction bits dropped, halfway bit 0x1000, bias
0x0FFF + tie-lsb; self-tested; non-finite rejected. Reference matmuls run
with torch.backends.cuda.matmul.fp32_precision = "ieee" (PyTorch 2.13 API,
legacy toggles untouched), asserted before every GEMM; fail-closed on
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1. The float64-accumulation cross-check
MEASURES the reference reduction discrepancy; it is not a formal bound on
cuBLAS TF32 kernel behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

EXPECTED = {
    "hf": {
        "q_b_proj.bin": "4f3b647b62c60475fc03f023ce46a5c01951c45847ced2557b5692b2ed3e79b1",
        "attn_o_input.bin": "965760c8d6eca03f1a385bbe76bc23987a414dd7d1f7077b0ee4cf1ae306d24a",
    },
    "targets": {
        "q_pe_rope_target.bin": "c8b9b6bfd8759f839c333e2b74f3775fe0b89bf82dc296497ee17990669dfc95",
        "k_pe_rope_target.bin": "3ed6f4e731227d49952fc687aefb2ede9067eceec7eed39096d861634158bc1d",
    },
    "r0": {
        "kv_cmpr_scaled.bin": "909b7ee75366b0ee1d5a912c103762563236cd07c6fd8385ceb1e549f2a86ce8",
        "kqv_out.bin": "d2137b526391bb2e75d4fe92f5ee9a546d818e4a0136cda9d5b0fc7c5c5bbbac",
    },
}

MSQ = 1.4142135623730951
KQ_SCALE_CPP = float(np.float32(0.15781141572701204))


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def tf32_rne(values: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(v).all():
        stop("tf32_rne: non-finite input")
    bits = v.view(np.uint32)
    bias = np.uint32(0x0FFF) + ((bits >> np.uint32(13)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFFE000)).view(np.float32)


def tf32_selftest() -> None:
    f = lambda x: tf32_rne(np.array([x], dtype=np.float32))[0]
    step, half, eps = 2.0 ** -10, 2.0 ** -11, 2.0 ** -23
    cases = [
        (1.0, 1.0), (1.5, 1.5), (-2.25, -2.25), (1.0 + step, 1.0 + step),
        (1.0 + half - eps, 1.0), (1.0 + half, 1.0), (1.0 + half + eps, 1.0 + step),
        (1.0 + step + half, 1.0 + 2 * step), (1.0 + step + half - eps, 1.0 + step),
        (-(1.0 + half), -1.0), (-(1.0 + step + half), -(1.0 + 2 * step)),
    ]
    for x, want in cases:
        if f(np.float32(x)) != np.float32(want):
            stop("tf32_rne self-test FAILED at %r" % x)
    for bad in (np.inf, -np.inf, np.nan):
        try:
            f(bad)
        except SystemExit:
            pass
        else:
            stop("tf32_rne self-test FAILED: non-finite accepted")
    print("tf32_rne self-test: PASS")


def load(root: Path, name: str, group: str, width: int) -> np.ndarray:
    raw = (root / name).read_bytes()
    if sha256_bytes(raw) != EXPECTED[group][name]:
        stop("input SHA mismatch: %s" % name)
    values = np.frombuffer(raw, dtype="<f4").reshape(-1, width)
    if not np.isfinite(values).all():
        stop("%s nonfinite" % name)
    return values


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.float64) - b.astype(np.float64)
    rmse = float(np.sqrt((d ** 2).mean()))
    ref = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
    return {
        "equal": int((a == b).sum()), "elements": int(a.size),
        "max_abs": float(np.abs(d).max()), "rmse": rmse,
        "rel_rmse": rmse / ref,
        "cosine": float((a.astype(np.float64) * b.astype(np.float64)).sum()
                        / (np.linalg.norm(a) * np.linalg.norm(b))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--targets-dir", required=True)
    ap.add_argument("--r0-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--json-out", default=None)
    ns = ap.parse_args()

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 present")

    import torch

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"

    def assert_ieee() -> None:
        if torch.backends.cuda.matmul.fp32_precision != "ieee":
            stop("matmul fp32_precision is not ieee")

    assert_ieee()
    tf32_selftest()
    dev = torch.device("cuda:0")

    precision_state = {
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    hf_dir, tgt_dir, r0_dir = Path(ns.hf_dir), Path(ns.targets_dir), Path(ns.r0_dir)
    qb = load(hf_dir, "q_b_proj.bin", "hf", 6144).reshape(512, 32, 192)
    oracle = load(hf_dir, "attn_o_input.bin", "hf", 4096)
    lat = load(r0_dir, "kv_cmpr_scaled.bin", "r0", 512)
    meas = load(r0_dir, "kqv_out.bin", "r0", 4096)
    qt = load(tgt_dir, "q_pe_rope_target.bin", "targets", 2048).reshape(512, 32, 64)
    kt = load(tgt_dir, "k_pe_rope_target.bin", "targets", 64)

    idx = json.loads((Path(ns.model_dir) / "model.safetensors.index.json").read_text())
    from safetensors import safe_open
    wname = "model.layers.0.self_attn.0.kv_b_proj.weight"
    with safe_open(str(Path(ns.model_dir) / idx["weight_map"][wname]),
                   framework="pt", device="cpu") as h:
        W = h.get_tensor(wname).float().numpy()
    wk = W.reshape(32, 256, 512)[:, :128, :]
    wv = W.reshape(32, 256, 512)[:, 128:, :]
    # P0 equivalence to the GGUF tensors was proven element-exact in
    # attn_core_attribution.json (attn_k_b under per-head transpose, attn_v_b
    # direct; 2097152/2097152 each), licensing HF-weight reuse here.

    T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    Q = lambda t: T(tf32_rne(t.float().cpu().numpy()))
    qn = to_bf16(qb[:, :, :128] * np.float32(MSQ))
    mask = torch.full((512, 512), float("-inf"), device=dev).triu(1)

    def pipeline(quantize: bool, f64_gemms: bool) -> np.ndarray:
        maybe_q = (lambda t: Q(t)) if quantize else (lambda t: t.float())
        acc = (lambda t: t.double()) if f64_gemms else (lambda t: t)
        assert_ieee()
        # 1. absorb: bf16-lattice operands, f32 accumulate, F32 out
        qa = torch.einsum("htd,hdl->htl", acc(T(qn).permute(1, 0, 2)),
                          acc(T(to_bf16(wk)))).float()
        # 2. scores: TF32 on off-lattice q_abs (K lattice: quantization lossless)
        Qcur = torch.cat([qa, T(qt).permute(1, 0, 2)], -1)
        Kcur = torch.cat([T(lat), T(kt)], -1)
        s = torch.einsum("htd,sd->hts", acc(maybe_q(Qcur)), acc(maybe_q(Kcur))).float()
        # 3. softmax f32 with the C++ f32 kq_scale
        p = torch.softmax(s * KQ_SCALE_CPP + mask, -1)
        # 4. context: TF32 on off-lattice probs
        cl = torch.einsum("hts,sl->htl", acc(maybe_q(p)), acc(maybe_q(T(lat)))).float()
        # 5. wv_b: lossy src1 BF16 conversion, bf16 weights, F32 out (no round)
        o = torch.einsum("htl,hdl->htd",
                         acc(T(to_bf16(cl.float().cpu().numpy()))),
                         acc(T(to_bf16(wv)))).float()
        return o.permute(1, 0, 2).reshape(512, 4096).float().cpu().numpy().astype("<f4")

    v7pp = pipeline(quantize=True, f64_gemms=False)
    v7pp_f64 = pipeline(quantize=True, f64_gemms=True)
    v_no_tf32 = pipeline(quantize=False, f64_gemms=False)

    report = {
        "precision_state": precision_state,
        "meas_kqv_bf16_lattice": {"equal": int((to_bf16(meas) == meas).sum()),
                                  "elements": int(meas.size)},
        "v7pp_vs_meas": metrics(v7pp, meas),
        "v7pp_vs_oracle": metrics(v7pp, oracle),
        "meas_vs_oracle": metrics(meas, oracle),
        "no_tf32_control_vs_meas": metrics(v_no_tf32, meas),
        "reference_reduction_discrepancy": metrics(v7pp, v7pp_f64),
    }
    rel = report["v7pp_vs_meas"]["rel_rmse"]
    disc = report["reference_reduction_discrepancy"]["rel_rmse"]
    report["collapse"] = {
        "v7pp_vs_meas_rel": rel,
        "reference_discrepancy_rel": disc,
        "ratio": rel / disc,
        "collapses": bool(rel <= 2.0 * disc),
    }

    print(json.dumps({k: v for k, v in report.items() if k != "precision_state"},
                     indent=2, sort_keys=True))
    print("V7'' vs MEAS rel=%.4g  reference discrepancy=%.4g  ratio=%.2fx  collapse=%s"
          % (rel, disc, rel / disc, report["collapse"]["collapses"]))

    if ns.json_out:
        Path(ns.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
        print("report written to", ns.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
