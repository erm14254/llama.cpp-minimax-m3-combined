#!/usr/bin/env python3

"""Attention-core attribution grid (H3 / H4 / H-TF32) -- offline, measurement-only.

Anchors:
  ORACLE = attn_o_input (HF context, byte-frozen)
  MEAS   = measured C++ kqv_out from the exact-RoPE R0 run

Variant grid (orthogonal pairs only; see plan):
  V1  expand-first, fused efficient SDPA (byte-exact C0)  -- oracle anchor only
  V8  expand-first, decomposed bf16 discipline            -- V1<->V8 = decomposition offset
  V2  V8 but probs stay f32                               -- H3 softmax-cast axis (V8<->V2)
  V3  expand-first, decomposed true f32                   -- precision bundle (V8<->V3)
  V4  V3 + TF32-RNE-quantized GEMM operands               -- H-TF32 at HF ordering (V3<->V4)
  V5  latent-absorbed, true f32                           -- H4 ordering axis (V3<->V5)
  V6  latent-absorbed, C++-faithful dtypes, no TF32       -- dtype discipline (V5<->V6)
  V7  V6 + TF32 on the F32xF32 GEMMs (whole endpoint)     -- C++ mechanism model (V7<->MEAS)
  V9  contingent (only if V2/V8 leave ambiguity)

C++ dispatch facts modeled in V6/V7 (verified in ggml source):
  * compute_type = src0->type (ggml-cuda.cu:1620); GGML_PREC_F32 forces F32
    for the score GEMM (:1626).
  * BF16-weight GEMMs (absorption, wv_b): src1 converted to BF16
    (batched_mul_mat_traits<BF16>::convert, :1387), GemmEx BF16xBF16 with
    CUBLAS_COMPUTE_32F, dst in CUDA_R_16BF -- i.e. the OUTPUT IS ROUNDED TO
    BF16 -- then widened to F32 (:1612-1615). Prediction checked empirically
    below: MEAS kqv_out must lie 100% on the BF16 lattice.
  * F32xF32 GEMMs (scores, latent context): GemmBatchedEx under the handle
    with CUBLAS_TF32_TENSOR_OP_MATH (common.cuh:1502) -> TF32.

Precision contract (PyTorch 2.13): the newer backend API only --
torch.backends.cuda.matmul.fp32_precision = "ieee"; asserted before every
reference GEMM; fail closed on TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1; legacy
allow_tf32 toggles are never touched. TF32 emulation quantizes operands with
round-to-nearest-even (halfway bit 0x1000, bias 0x0FFF + tie-lsb) and runs
the reference matmul in IEEE f32; a float64-accumulation cross-check MEASURES
the reference reduction discrepancy (not a formal bound on cuBLAS TF32).
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
        "kv_a_proj_with_mqa.bin": "513390418c9877fa46286d397db7c9c9fb6408852836fb7827106acd183ceecc",
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

SCALE_HF = 0.15781141572701202   # HF attn_scaling (double, applied in sdpa)
MSQ = 1.4142135623730951         # mla_scale_q_lora


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def tf32_rne(values: np.ndarray) -> np.ndarray:
    """FP32 -> TF32 lattice, round-to-nearest-even. 13 fraction bits dropped;
    halfway bit 0x1000, bias 0x0FFF + tie-lsb. Non-finite inputs rejected."""
    v = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(v).all():
        stop("tf32_rne: non-finite input")
    bits = v.view(np.uint32)
    bias = np.uint32(0x0FFF) + ((bits >> np.uint32(13)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFFE000)).view(np.float32)


def tf32_selftest() -> None:
    f = lambda x: tf32_rne(np.array([x], dtype=np.float32))[0]
    step = 2.0 ** -10           # retained-mantissa step at [1, 2)
    half = 2.0 ** -11           # halfway offset
    eps = 2.0 ** -23
    cases = [
        # lattice values unchanged
        (1.0, 1.0), (1.5, 1.5), (-2.25, -2.25), (1.0 + step, 1.0 + step),
        # just below / at / just above halfway, retained lsb = 0 (base 1.0)
        (1.0 + half - eps, 1.0),
        (1.0 + half, 1.0),                       # tie -> even (lsb 0 stays)
        (1.0 + half + eps, 1.0 + step),
        # tie with retained lsb = 1 (base 1.0 + step) -> rounds UP to even
        (1.0 + step + half, 1.0 + 2 * step),
        (1.0 + step + half - eps, 1.0 + step),
        # negatives mirror
        (-(1.0 + half), -1.0),
        (-(1.0 + step + half), -(1.0 + 2 * step)),
    ]
    for x, want in cases:
        got = f(np.float32(x))
        if got != np.float32(want):
            stop("tf32_rne self-test FAILED: %r -> %r, want %r" % (x, got, want))
    for bad in (np.inf, -np.inf, np.nan):
        try:
            f(bad)
        except SystemExit:
            pass
        else:
            stop("tf32_rne self-test FAILED: non-finite accepted")
    print("tf32_rne self-test: PASS (%d finite cases + non-finite rejection)" % len(cases))


def load(root: Path, name: str, group: str, width: int) -> np.ndarray:
    raw = (root / name).read_bytes()
    got = sha256_bytes(raw)
    if got != EXPECTED[group][name]:
        stop("input SHA mismatch %s: %s" % (name, got))
    values = np.frombuffer(raw, dtype="<f4").reshape(-1, width)
    if not np.isfinite(values).all():
        stop("%s contains nonfinite values" % name)
    return values


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.float64) - b.astype(np.float64)
    rmse = float(np.sqrt((d ** 2).mean()))
    ref = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
    return {
        "equal": int((a == b).sum()),
        "elements": int(a.size),
        "max_abs": float(np.abs(d).max()),
        "rmse": rmse,
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
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--gguf-py", default="gguf-py")
    ap.add_argument("--json-out", default=None)
    ns = ap.parse_args()

    # ---------------------------------------------------------------- contract
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 present -- reference path invalid")

    import torch

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"   # 2.13 API; legacy toggles untouched

    def assert_ieee() -> None:
        if torch.backends.cuda.matmul.fp32_precision != "ieee":
            stop("matmul fp32_precision is not ieee")

    assert_ieee()
    dev = torch.device("cuda:0")

    precision_state = {
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print("precision state:", json.dumps(precision_state))
    tf32_selftest()

    def T(x: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

    def mm_ieee(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        assert_ieee()
        assert a.dtype == torch.float32 and b.dtype == torch.float32
        return a @ b

    # ------------------------------------------------------------------ inputs
    hf_dir = Path(ns.hf_dir); tgt_dir = Path(ns.targets_dir); r0_dir = Path(ns.r0_dir)

    qb = load(hf_dir, "q_b_proj.bin", "hf", 6144).reshape(512, 32, 192)
    oracle = load(hf_dir, "attn_o_input.bin", "hf", 4096)
    meas = load(r0_dir, "kqv_out.bin", "r0", 4096)
    lat = load(r0_dir, "kv_cmpr_scaled.bin", "r0", 512)
    qt = load(tgt_dir, "q_pe_rope_target.bin", "targets", 2048).reshape(512, 32, 64)
    kt = load(tgt_dir, "k_pe_rope_target.bin", "targets", 64)

    def P_fwd(cpp: np.ndarray) -> np.ndarray:
        out = np.empty_like(cpp)
        out[..., :32] = cpp[..., 0::2]
        out[..., 32:] = cpp[..., 1::2]
        return out

    q_nope = to_bf16(qb[:, :, :128] * np.float32(MSQ))       # [512, 32, 128] lattice

    meas_lattice = int((to_bf16(meas) == meas).sum())
    print("MEAS kqv_out on BF16 lattice: %d/%d (dispatch prediction: all)"
          % (meas_lattice, meas.size))

    # ------------------------------------------------- P0: weight provenance
    idx = json.loads((Path(ns.model_dir) / "model.safetensors.index.json").read_text())
    wname = "model.layers.0.self_attn.0.kv_b_proj.weight"
    from safetensors import safe_open
    with safe_open(str(Path(ns.model_dir) / idx["weight_map"][wname]),
                   framework="pt", device="cpu") as h:
        W_hf = h.get_tensor(wname).float().numpy()           # [8192, 512]
    wk_hf = W_hf.reshape(32, 256, 512)[:, :128, :]           # [32, 128, 512]
    wv_hf = W_hf.reshape(32, 256, 512)[:, 128:, :]

    sys.path.insert(0, str(Path(ns.gguf_py).resolve()))
    from gguf import GGUFReader
    gg = {}
    for tsr in GGUFReader(ns.gguf, "r").tensors:
        if tsr.name in ("blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight"):
            arr = np.array(tsr.data)
            if arr.dtype == np.uint8:
                arr = arr.view(np.uint16)
            if arr.dtype == np.uint16:
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            gg[tsr.name] = np.array(arr, dtype=np.float32)
    if len(gg) != 2:
        stop("GGUF attn_k_b/attn_v_b not found")

    def match_orient(garr: np.ndarray, ref: np.ndarray, tag: str) -> dict:
        # candidate orientations: [32,r,c] direct and per-head transpose
        n = ref.size
        cands = {}
        if garr.size == n:
            for shape, tr in ((ref.shape, False), ((ref.shape[0], ref.shape[2], ref.shape[1]), True)):
                try:
                    c = garr.reshape(shape)
                    c = c.transpose(0, 2, 1) if tr else c
                    cands["shape=%s%s" % (shape, " ^T" if tr else "")] = int((c == ref).sum())
                except ValueError:
                    pass
        best = max(cands.items(), key=lambda kv: kv[1]) if cands else ("none", -1)
        entry = {"candidates": cands, "best": best[0], "equal": best[1], "elements": n}
        print("P0 %s: best=%s equal=%d/%d" % (tag, best[0], best[1], n))
        return entry

    p0 = {
        "attn_k_b": match_orient(gg["blk.0.attn_k_b.weight"], wk_hf, "attn_k_b vs HF k rows"),
        "attn_v_b": match_orient(gg["blk.0.attn_v_b.weight"], wv_hf, "attn_v_b vs HF v rows"),
    }
    identical = all(p0[k]["equal"] == p0[k]["elements"] for k in p0)
    p0["identical_under_reshape"] = bool(identical)
    if not identical:
        print("P0: GGUF differs from HF -- C++-faithful variants use GGUF weights")
    wk_cpp = (wk_hf if identical else
              gg["blk.0.attn_k_b.weight"].reshape(-1)[: wk_hf.size].reshape(wk_hf.shape))
    wv_cpp = (wv_hf if identical else
              gg["blk.0.attn_v_b.weight"].reshape(-1)[: wv_hf.size].reshape(wv_hf.shape))
    if not identical:
        stop("P0: orientation-verified GGUF mapping required before V5/V6/V7 -- "
             "resolve manually (mismatch recorded above)")

    # ---------------------------------------------------------------- variants
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend

    report = {"precision_state": precision_state, "p0_weight_provenance": p0,
              "meas_kqv_bf16_lattice": {"equal": meas_lattice, "elements": int(meas.size)},
              "variants": {}}

    W_bf = T(W_hf).to(torch.bfloat16)
    q_pass_hf = T(q_nope).to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
    q_rot_hf = T(P_fwd(qt)).to(torch.bfloat16).permute(1, 0, 2).unsqueeze(0)
    k_rot_hf = T(P_fwd(kt)).to(torch.bfloat16).reshape(1, 1, 512, 64)
    lat_bf = T(lat).to(torch.bfloat16)
    mask64 = torch.full((512, 512), float("-inf"), device=dev).triu(1)

    def out4096(x_h_t_d: "torch.Tensor") -> np.ndarray:
        return (x_h_t_d.permute(1, 0, 2).reshape(512, 4096)
                .float().cpu().numpy().astype("<f4"))

    def rec(tag: str, arr: np.ndarray, also_meas: bool = False) -> None:
        entry = {"vs_oracle": metrics(arr, oracle)}
        if also_meas:
            entry["vs_meas"] = metrics(arr, meas)
        report["variants"][tag] = entry
        m = entry["vs_oracle"]
        line = (f"{tag:6s} vs ORACLE eq={m['equal']}/{m['elements']} max={m['max_abs']:.3e} "
                f"rel={m['rel_rmse']:.6g} cos={m['cosine']:.8f}")
        if also_meas:
            mm = entry["vs_meas"]
            line += f" | vs MEAS eq={mm['equal']} max={mm['max_abs']:.3e} rel={mm['rel_rmse']:.6g}"
        print(line)

    # V1 -- fused efficient SDPA (C0)
    kvn = (lat_bf @ W_bf.T).view(1, 512, 32, 256).transpose(1, 2)
    k_nope, val = torch.split(kvn, [128, 128], dim=-1)
    key = kvn.new_empty(1, 32, 512, 192)
    key[..., :128].copy_(k_nope)
    key[..., 128:].copy_(k_rot_hf.expand(-1, 32, -1, -1))
    query = torch.cat([q_pass_hf, q_rot_hf], dim=-1)
    with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        o = F.scaled_dot_product_attention(query, key, val, attn_mask=None,
                                           is_causal=True, scale=SCALE_HF)
    rec("V1", out4096(o[0]))

    # decomposed expand-first machinery
    q_hf32 = query.float()[0]                    # [32, 512, 192]
    k_bf = key[0]; v_bf = val[0]                 # bf16 [32, 512, ...]
    k32 = k_bf.float(); v32 = v_bf.float()

    def softmax_f32(scores32: "torch.Tensor") -> "torch.Tensor":
        return torch.softmax(scores32 * SCALE_HF + mask64, dim=-1)

    # V8 -- decomposed bf16: bf16 GEMMs (f32-acc, bf16 out), f32 softmax,
    #       probs->bf16, bf16 PV
    s_bf = torch.einsum("htd,hsd->hts", query[0], k_bf)          # bf16 out
    probs8 = softmax_f32(s_bf.float()).to(torch.bfloat16)
    ctx8 = torch.einsum("hts,hsd->htd", probs8, v_bf)            # bf16 out
    rec("V8", out4096(ctx8))

    # V2 -- V8 but probs stay f32 (PV in ieee f32 on widened V, bf16 out kept)
    probs2 = softmax_f32(s_bf.float())
    assert_ieee()
    ctx2 = torch.einsum("hts,hsd->htd", probs2, v32).to(torch.bfloat16)
    rec("V2", out4096(ctx2))

    # V3 -- decomposed true f32 (expand GEMM f32 too)
    assert_ieee()
    kvn32 = (T(lat) @ T(W_hf).T).view(512, 32, 256).permute(1, 0, 2)
    k3 = torch.cat([kvn32[..., :128], T(P_fwd(kt)).reshape(1, 512, 64).expand(32, -1, -1)], dim=-1)
    v3 = kvn32[..., 128:]
    q3 = q_hf32
    s3 = torch.einsum("htd,hsd->hts", q3, k3)
    ctx3 = torch.einsum("hts,hsd->htd", softmax_f32(s3), v3)
    rec("V3", out4096(ctx3))

    # V4 -- V3 with TF32-RNE quantized operands on all three GEMM stages
    def q32(x: "torch.Tensor") -> "torch.Tensor":
        return T(tf32_rne(x.float().cpu().numpy()))
    assert_ieee()
    kvn4 = (q32(T(lat)) @ q32(T(W_hf)).T).view(512, 32, 256).permute(1, 0, 2)
    k4 = torch.cat([kvn4[..., :128], T(P_fwd(kt)).reshape(1, 512, 64).expand(32, -1, -1)], dim=-1)
    v4 = kvn4[..., 128:]
    s4 = torch.einsum("htd,hsd->hts", q32(q3), q32(k4))
    ctx4 = torch.einsum("hts,hsd->htd", q32(softmax_f32(s4)), q32(v4))
    rec("V4", out4096(ctx4))

    # latent-absorbed machinery (CPP-layout rope tensors, as C++ consumed)
    wk_t = T(wk_cpp)                              # [32, 128, 512]
    wv_t = T(wv_cpp)
    qn32 = T(q_nope).permute(1, 0, 2)             # [32, 512, 128]
    qpe_c = T(qt).permute(1, 0, 2)                # [32, 512, 64] CPP layout
    kpe_c = T(kt)                                 # [512, 64]
    lat32 = T(lat)
    KQ_SCALE_CPP = np.float32(np.float32(MSQ**0) * 0.15781141572701204)  # C++ f32 constant

    def latent_scores(q_abs, qpe):
        return (torch.einsum("htl,sl->hts", q_abs, lat32)
                + torch.einsum("htd,sd->hts", qpe, kpe_c))

    def latent_softmax(s):
        return torch.softmax(s * float(KQ_SCALE_CPP) + mask64, dim=-1)

    # V5 -- latent-absorbed true f32
    assert_ieee()
    qa5 = torch.einsum("htd,hdl->htl", qn32, wk_t)
    p5 = latent_softmax(latent_scores(qa5, qpe_c))
    cl5 = torch.einsum("hts,sl->htl", p5, lat32)
    o5 = torch.einsum("htl,hdl->htd", cl5, wv_t)
    rec("V5", out4096(o5), also_meas=True)

    # V6 -- C++-faithful dtypes, no TF32:
    #   absorb: bf16 x bf16, f32-acc, BF16 OUT (dispatch fact) -> widen
    #   scores/ctx: ieee f32
    #   wv_b: src1 bf16 round (lossy), bf16 weight, f32-acc, BF16 OUT -> widen
    qa6 = torch.einsum("htd,hdl->htl", qn32.to(torch.bfloat16),
                       wk_t.to(torch.bfloat16))                  # bf16 out
    assert_ieee()
    p6 = latent_softmax(latent_scores(qa6.float(), qpe_c))
    cl6 = torch.einsum("hts,sl->htl", p6, lat32)
    o6 = torch.einsum("htl,hdl->htd", cl6.to(torch.bfloat16),
                      wv_t.to(torch.bfloat16))                   # bf16 out
    rec("V6", out4096(o6), also_meas=True)

    # V7 -- V6 + TF32-RNE on the two F32xF32 GEMMs (scores, latent context);
    #       whole-endpoint model per plan enumeration
    assert_ieee()
    qa7 = qa6.float()
    s7 = (torch.einsum("htl,sl->hts", q32(qa7), q32(lat32))
          + torch.einsum("htd,sd->hts", q32(qpe_c), q32(kpe_c)))
    p7 = latent_softmax(s7)
    cl7 = torch.einsum("hts,sl->htl", q32(p7), q32(lat32))
    o7 = torch.einsum("htl,hdl->htd", cl7.to(torch.bfloat16),
                      wv_t.to(torch.bfloat16))
    v7 = out4096(o7)
    rec("V7", v7, also_meas=True)

    # reference reduction discrepancy: float64 accumulation of the SAME
    # TF32-quantized operands (measured discrepancy, not a formal bound)
    s7d = (torch.einsum("htl,sl->hts", q32(qa7).double(), q32(lat32).double())
           + torch.einsum("htd,sd->hts", q32(qpe_c).double(), q32(kpe_c).double())).float()
    p7d = latent_softmax(s7d)
    cl7d = torch.einsum("hts,sl->htl", q32(p7d).double(), q32(lat32).double()).float()
    o7d = torch.einsum("htl,hdl->htd", cl7d.to(torch.bfloat16),
                       wv_t.to(torch.bfloat16))
    disc = metrics(v7, out4096(o7d))
    report["v7_reference_reduction_discrepancy"] = disc
    print("V7 f64-accum cross-check (measured reference reduction discrepancy): "
          f"rel={disc['rel_rmse']:.3e} max={disc['max_abs']:.3e}")

    v7_meas_rel = report["variants"]["V7"]["vs_meas"]["rel_rmse"]
    closure = v7_meas_rel <= max(10 * disc["rel_rmse"], 1e-5)
    report["v7_meas_closure"] = {
        "v7_vs_meas_rel_rmse": v7_meas_rel,
        "discrepancy_rel_rmse": disc["rel_rmse"],
        "collapses_near_discrepancy": bool(closure),
    }
    print("V7<->MEAS closure:", "YES" if closure else
          "NO -- simplified offline model does not fully reproduce the measured "
          "C++ execution; missing detail may include cuBLAS TF32 "
          "algorithm/tiling/reduction behavior or another unmodeled operation")

    if ns.json_out:
        Path(ns.json_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
        print("report written to", ns.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
