#!/usr/bin/env python3

"""Localize the first attention-path divergence (segments S1..S4).

All pre-attention block-0 MLA surfaces are byte-exact between llama.cpp and
the HF Blackwell oracles. This script walks the surviving path in execution
order and reports, per segment, whether the C++ tensor matches an
HF-semantics reference built ONLY from captured oracles and elementwise ops
(no reduction freedom), with integer mismatch counts:

  S1   post-RoPE rotary Q       (q_pe_rope  vs interleave(bf16-scale(q_b_proj), cos/sin))
  S2   post-RoPE rotary K       (k_pe_rope  vs interleave(k_rot, cos/sin))
  S2b  post-scale compressed KV (kv_cmpr_scaled vs bf16(kv_a_layernorm * mla_scale_kv))
  S3   pre-wo attention context (kqv_out    vs attn_o_input oracle)
  S4   wo boundary              (conditional; see plan -- the offline GEMM here
                                 is ANALYTIC ONLY, it does not execute ggml)

Layout facts used (recorded in the JSON):
  * RoPE permutation P: HF apply_rotary_pos_emb_interleave returns
    cat([even', odd']) while ggml NORM RoPE rotates pairs in place, so
    CPP[2j] == HF[j] and CPP[2j+1] == HF[32+j]. Applied to S1/S2.
  * Context layout: C++ permute(0,2,1,3)+cont_2d (llama-graph.cpp:2845-2848)
    and HF transpose(1,2).contiguous().reshape (modeling:274) both flatten a
    token row as index = head*128 + v_dim. Identical -- no permutation in S3.

HF-side references are computed with torch BF16 on CUDA -- the same kernels
the capture itself executed -- and the trivial S2b case is cross-checked with
a numpy round-to-nearest-even model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_TOK = 512
N_HEAD = 32
QK_NOPE = 128
QK_ROPE = 64
V_DIM = 128
KV_LORA = 512


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def load(root: Path, name: str, rows: int, width: int) -> np.ndarray:
    path = root / (name + ".bin")
    if not path.is_file():
        stop("missing %s" % path)
    raw = path.read_bytes()
    if len(raw) != rows * width * 4:
        stop("%s is %d bytes, expected %d" % (path, len(raw), rows * width * 4))
    values = np.frombuffer(raw, dtype="<f4").reshape(rows, width)
    if not np.isfinite(values).all():
        stop("%s contains nonfinite values" % path)
    return values


def seg(tag: str, cpp: np.ndarray, ref: np.ndarray) -> dict:
    """Standard segment report: raw + bf16-rounded integer counts + metrics."""
    diff = cpp.astype(np.float64) - ref.astype(np.float64)
    rmse = float(np.sqrt((diff ** 2).mean()))
    ref_rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
    rounded = to_bf16(cpp)
    entry = {
        "elements": int(cpp.size),
        "raw_equal": int((cpp == ref).sum()),
        "bf16_equal": int((rounded == ref).sum()),
        "bf16_mismatch": int((rounded != ref).sum()),
        "max_abs": float(np.abs(diff).max()),
        "rel_rmse": rmse / ref_rms if ref_rms > 0 else float("nan"),
        "ref_on_bf16_lattice": int((to_bf16(ref) == ref).sum()),
    }
    if entry["bf16_mismatch"] == 0:
        entry["verdict"] = "boundary-only (bf16(cpp) == ref exactly)"
    elif entry["raw_equal"] == entry["elements"]:
        entry["verdict"] = "byte-exact"
    else:
        entry["verdict"] = "REAL"
    print(
        "%-42s raw=%d/%d  bf16=%d/%d  max_abs=%.6g  rel_RMSE=%.6g  -> %s"
        % (
            tag, entry["raw_equal"], entry["elements"],
            entry["bf16_equal"], entry["elements"],
            entry["max_abs"], entry["rel_rmse"], entry["verdict"],
        )
    )
    return entry


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

    import torch

    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda:0")

    summary = json.loads((hf_dir / "summary.json").read_text(encoding="utf-8"))
    scale_q = float(summary["mla_scale_q_lora"])
    scale_kv = float(summary["mla_scale_kv_lora"])
    attn_scaling = float(summary["attn_scaling"])
    print("mla_scale_q_lora  =", scale_q)
    print("mla_scale_kv_lora =", scale_kv)
    print("attn_scaling      =", attn_scaling)
    print()

    report = {
        "scales": {
            "mla_scale_q_lora": scale_q,
            "mla_scale_kv_lora": scale_kv,
            "attn_scaling": attn_scaling,
        },
        "layout_proofs": {
            "rope_permutation_P": (
                "HF apply_rotary_pos_emb_interleave (modeling:322-329) returns "
                "cat([q1*cos-q2*sin, q2*cos+q1*sin]) over even/odd slices; ggml "
                "NORM RoPE rotates pairs in place. CPP[2j]==HF[j], "
                "CPP[2j+1]==HF[32+j]; applied to S1/S2."
            ),
            "context_flatten": (
                "C++ permute(0,2,1,3)+cont_2d, llama-graph.cpp:2845-2848, "
                "flattens [v=128, head=32, tok] with v fastest -> token row "
                "index head*128+v. HF transpose(1,2).contiguous().reshape "
                "(modeling:274) gives the same head*128+v. Identical; no "
                "permutation applied in S3."
            ),
        },
        "segments": {},
    }

    def t(x: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

    cos = load(hf_dir, "rope_cos", N_TOK, QK_ROPE)
    sin = load(hf_dir, "rope_sin", N_TOK, QK_ROPE)
    if not ((to_bf16(cos) == cos).all() and (to_bf16(sin) == sin).all()):
        stop("rope oracles are not on the BF16 lattice")

    cos_h = t(cos[:, : QK_ROPE // 2]).to(torch.bfloat16)  # [512, 32]
    sin_h = t(sin[:, : QK_ROPE // 2]).to(torch.bfloat16)

    def hf_rope(x_bf16: "torch.Tensor") -> "torch.Tensor":
        """modeling:326-331 verbatim: x [512, H, 64] bf16, cos/sin [512,1,32]."""
        c = cos_h.unsqueeze(1)
        s = sin_h.unsqueeze(1)
        x1 = x_bf16[..., 0::2]
        x2 = x_bf16[..., 1::2]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    def unpermute_P(hf_out: "torch.Tensor") -> np.ndarray:
        """Map HF cat([even', odd']) layout back to ggml in-place pairs."""
        out = torch.empty_like(hf_out)
        half = QK_ROPE // 2
        out[..., 0::2] = hf_out[..., :half]
        out[..., 1::2] = hf_out[..., half:]
        return out.float().cpu().numpy().astype("<f4")

    # ------------------------------------------------------------------ S1
    q_b = load(hf_dir, "q_b_proj", N_TOK, N_HEAD * (QK_NOPE + QK_ROPE))
    q_rot = q_b.reshape(N_TOK, N_HEAD, QK_NOPE + QK_ROPE)[:, :, QK_NOPE:]
    q_rot_scaled = (t(q_rot).to(torch.bfloat16) * scale_q)  # bf16 * float -> bf16
    q_ref = unpermute_P(hf_rope(q_rot_scaled)).reshape(N_TOK, N_HEAD * QK_ROPE)

    cpp_q = load(cpp_dir, "q_pe_rope", N_TOK, N_HEAD * QK_ROPE)
    report["segments"]["S1_q_pe_rope"] = seg("S1 q_pe_rope vs HF-semantics rope", cpp_q, q_ref)

    # ------------------------------------------------------------------ S2
    kv_proj = load(hf_dir, "kv_a_proj_with_mqa", N_TOK, KV_LORA + QK_ROPE)
    k_rot = kv_proj[:, KV_LORA:].reshape(N_TOK, 1, QK_ROPE)
    k_ref = unpermute_P(hf_rope(t(k_rot).to(torch.bfloat16))).reshape(N_TOK, QK_ROPE)

    cpp_k = load(cpp_dir, "k_pe_rope", N_TOK, QK_ROPE)
    report["segments"]["S2_k_pe_rope"] = seg("S2 k_pe_rope vs HF-semantics rope", cpp_k, k_ref)

    # ------------------------------------------------------------------ S2b
    kv_norm = load(hf_dir, "kv_a_layernorm", N_TOK, KV_LORA)
    ref_torch = (t(kv_norm).to(torch.bfloat16) * scale_kv).float().cpu().numpy().astype("<f4")
    ref_numpy = to_bf16(kv_norm * np.float32(scale_kv))
    cross = int((ref_torch == ref_numpy).sum())
    print("S2b cross-check torch-vs-numpy reference: %d/%d equal" % (cross, ref_numpy.size))

    cpp_kv = load(cpp_dir, "kv_cmpr_scaled", N_TOK, KV_LORA)
    entry = seg("S2b kv_cmpr_scaled vs bf16(norm*scale)", cpp_kv, ref_torch)
    entry["torch_numpy_reference_equal"] = cross
    report["segments"]["S2b_kv_cmpr_scaled"] = entry

    # ------------------------------------------------------------------ S3
    ctx_oracle = load(hf_dir, "attn_o_input", N_TOK, N_HEAD * V_DIM)
    cpp_ctx = load(cpp_dir, "kqv_out", N_TOK, N_HEAD * V_DIM)
    entry = seg("S3 kqv_out vs attn_o_input oracle", cpp_ctx, ctx_oracle)

    d_tok = np.abs(cpp_ctx.astype(np.float64) - ctx_oracle.astype(np.float64))
    entry["first_divergent_token"] = int(
        np.argmax(np.any(cpp_ctx != ctx_oracle, axis=1))
    ) if (cpp_ctx != ctx_oracle).any() else -1
    per_head = d_tok.reshape(N_TOK, N_HEAD, V_DIM)
    head_rms = np.sqrt((per_head ** 2).mean(axis=(0, 2)))
    entry["per_head_rmse_min"] = float(head_rms.min())
    entry["per_head_rmse_max"] = float(head_rms.max())
    report["segments"]["S3_context"] = entry

    # ------------------------------------------------------------------ S4
    # Conditional per plan: S3 decides which interpretation is licensed.
    s3_exact = report["segments"]["S3_context"]["bf16_mismatch"] == 0

    sys.path.insert(0, str(Path(ns.gguf_py).resolve()))
    from gguf import GGUFReader  # noqa: PLC0415

    wo = None
    for tensor in GGUFReader(ns.gguf, "r").tensors:
        if tensor.name == "blk.0.attn_output.weight":
            wo = np.array(tensor.data)
            break
    if wo is None:
        stop("blk.0.attn_output.weight not found in GGUF")

    # GGUF stores BF16 as uint8 pairs or raw bf16; normalize to f32 [3072, 4096]
    if wo.dtype == np.uint8:
        wo = wo.view(np.uint16)
    if wo.dtype == np.uint16:
        wo = (wo.astype(np.uint32) << 16).view(np.float32)
    wo = wo.reshape(3072, 4096).astype(np.float32)
    print("wo weight on bf16 lattice: %d/%d" % (int((to_bf16(wo) == wo).sum()), wo.size))

    wo_t = t(wo).to(torch.bfloat16)

    # Analytic wo semantics test on the ORACLE context (perfect input). This is
    # torch/cuBLAS-in-torch, NOT ggml execution -- analytic only.
    ctx_bf16 = t(ctx_oracle).to(torch.bfloat16)
    o_ref = (ctx_bf16 @ wo_t.T).float().cpu().numpy().astype("<f4")
    o_oracle = load(hf_dir, "o_proj", N_TOK, 3072)
    a = seg("S4a ANALYTIC torch-bf16 wo(oracle ctx) vs o_proj oracle", o_ref, o_oracle)
    a["note"] = (
        "analytic only: torch bf16 matmul, not the actual C++/ggml/cuBLAS wo "
        "execution"
    )
    report["segments"]["S4a_analytic_wo_semantics"] = a

    b = seg("S4b bf16(cpp kqv_out) vs attn_o_input oracle", to_bf16(cpp_ctx), ctx_oracle)
    b["note"] = "quantifies the wo INPUT error inherited from S3"
    report["segments"]["S4b_wo_input_error"] = b

    cpp_o = load(cpp_dir, "o_proj", N_TOK, 3072)
    c = seg("S4c cpp o_proj (real ggml wo) vs o_proj oracle", cpp_o, o_oracle)
    c["note"] = (
        "real ggml wo execution on the DIVERGENT C++ context; isolates the wo "
        "boundary only if S3 were byte-exact (it is%s)" % ("" if s3_exact else " NOT")
    )
    report["segments"]["S4c_real_wo_on_cpp_context"] = c

    report["s3_byte_exact_after_rounding"] = bool(s3_exact)

    # ------------------------------------------------------------- verdict
    order = ["S1_q_pe_rope", "S2_k_pe_rope", "S2b_kv_cmpr_scaled", "S3_context"]
    first_real = next(
        (k for k in order if report["segments"][k]["verdict"] == "REAL"), None
    )
    report["first_real_segment"] = first_real
    print()
    print("FIRST REAL SEGMENT:", first_real or "none before S3/S4")

    if ns.json_out:
        Path(ns.json_out).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("report written to", ns.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
