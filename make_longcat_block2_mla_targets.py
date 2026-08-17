#!/usr/bin/env python3
"""Generate the block-2 MLA offline comparanda, gated on block-0 known answers.

Two generator paths, each validated against a committed block-0 known-answer
artifact BEFORE the block-2 output is generated or trusted:

1. RoPE targets (the make_longcat_rope_targets.py transformation, verbatim:
   the ACTUAL installed apply_rotary_pos_emb_interleave on captured oracles,
   BF16 on CUDA, canonicalized through the proven permutation P).
   KNOWN-ANSWER GATE: the same code must regenerate the committed block-0
   canonical targets byte-exact (c8b9b6bf... / 3ed6f4e7...) from the block-0
   capture inputs. Only then are block-2 targets produced, with the t=0
   identity check retained.

2. kv_cmpr_scaled comparandum (to_bf16(hf_kv_a_layernorm * sqrt(6)), the S2b
   pattern with the torch/numpy RNE cross-check).
   KNOWN-ANSWER GATE: the path must reproduce the committed byte-exact
   block-0 kv_cmpr_scaled artifact (909b7ee7...) from the block-0 HF
   kv_a_layernorm oracle. Only then is the block-2 comparandum produced.

Explicit RoPE interpretation rule (pre-registered, recorded here): the
production C++ run uses ggml-generated angles while these targets use
captured HF cos/sin, which are known NOT byte-exact (sin 3,377/16,384
certain mismatches). Divergence at q_pe_rope-2 / k_pe_rope-2 therefore
localizes at most to the production-RoPE composite / angle-generation
state - never to rotation arithmetic (R1 is the standing proof that the
rotation is exact given exact angles).

Measurement-only; no arithmetic changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

B0_DIR = Path(r"D:\lc_mla_blackwell_attn")
B0_KV_DIR = Path(r"D:\lc_mla_blackwell")
B2_DIR = Path(r"D:\lc_block2_mla_512")
REPO = Path(__file__).resolve().parent
B0_SCALED = REPO / "cpp_attn0_mla_attnpath_512" / "kv_cmpr_scaled.bin"

B0_INPUTS = {
    "rope_cos.bin": "8771da1ea77d102e07bbc08064e6da6226ab4c2cb2a195c25d197c35487d9bb2",
    "rope_sin.bin": "5c5dede92d05f23dfab1a27285685f61e5596df888dd429fae6d8b6591b7ff0a",
    "q_b_proj.bin": "4f3b647b62c60475fc03f023ce46a5c01951c45847ced2557b5692b2ed3e79b1",
    "kv_a_proj_with_mqa.bin": "513390418c9877fa46286d397db7c9c9fb6408852836fb7827106acd183ceecc",
}
B0_KV_NORM_SHA = "b44cc101b03b11d96c0d9c52613f7469141dd7786b8128f93e3b7e912c550373"
B0_SCALED_SHA = "909b7ee75366b0ee1d5a912c103762563236cd07c6fd8385ceb1e549f2a86ce8"
B0_TARGET_Q = "c8b9b6bfd8759f839c333e2b74f3775fe0b89bf82dc296497ee17990669dfc95"
B0_TARGET_K = "3ed6f4e731227d49952fc687aefb2ede9067eceec7eed39096d861634158bc1d"

B2_INPUTS = {
    "rope_cos.bin": "8771da1ea77d102e07bbc08064e6da6226ab4c2cb2a195c25d197c35487d9bb2",
    "rope_sin.bin": "5c5dede92d05f23dfab1a27285685f61e5596df888dd429fae6d8b6591b7ff0a",
    "q_b_proj.bin": "ecb70ef6c9bd4d6f28a67467f3b5ec3fc575d4f37cd2e81bce7cc7554323f308",
    "kv_a_proj_with_mqa.bin": "28ea5b52221a94ddf780f04507f11aee7b6fc8617974f53d558424d41c470f3f",
}
B2_KV_NORM_SHA = "c91991eb459352ec407aebcee5ee2b12e7b25db0bafd3e0462955a8f8144df6b"

MLA_SCALE_Q = 1.4142135623730951   # sqrt(3072/1536)
MLA_SCALE_KV = 2.449489742783178   # sqrt(3072/512)


def stop(msg: str) -> None:
    raise SystemExit("STOP: " + msg)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(root: Path, name: str, rows: int, width: int, expected: dict) -> np.ndarray:
    raw = (root / name).read_bytes()
    got = sha256_bytes(raw)
    if got != expected[name]:
        stop("input SHA mismatch for %s/%s: %s" % (root, name, got))
    return np.frombuffer(raw, dtype="<f4").reshape(rows, width)


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def unpermute_P(hf_arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(hf_arr)
    out[..., 0::2] = hf_arr[..., :32]
    out[..., 1::2] = hf_arr[..., 32:]
    return out


def gen_rope_targets(src: Path, expected: dict):
    """The make_longcat_rope_targets.py transformation, verbatim."""
    import torch
    from transformers.models.longcat_flash.modeling_longcat_flash import (
        apply_rotary_pos_emb_interleave,
    )

    dev = torch.device("cuda:0")
    cos = load(src, "rope_cos.bin", 512, 64, expected)
    sin = load(src, "rope_sin.bin", 512, 64, expected)
    q_b = load(src, "q_b_proj.bin", 512, 6144, expected).reshape(512, 32, 192)
    kv = load(src, "kv_a_proj_with_mqa.bin", 512, 576, expected)

    def t(x: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

    q_rot = (t(q_b[:, :, 128:]).to(torch.bfloat16) * MLA_SCALE_Q)
    q_hf = q_rot.permute(1, 0, 2).unsqueeze(0).contiguous()
    k_hf = t(kv[:, 512:]).to(torch.bfloat16).reshape(1, 1, 512, 64).contiguous()
    cos_hf = t(cos).to(torch.bfloat16).unsqueeze(0)
    sin_hf = t(sin).to(torch.bfloat16).unsqueeze(0)

    q_gt, k_gt = apply_rotary_pos_emb_interleave(q_hf, k_hf, cos_hf, sin_hf)

    gt_q = q_gt[0].permute(1, 0, 2).float().cpu().numpy()
    gt_k = k_gt[0, 0].float().cpu().numpy()
    tq = unpermute_P(gt_q).reshape(512, 2048).astype("<f4")
    tk = unpermute_P(gt_k).astype("<f4")

    # t=0 identity: at position 0 (cos=1, sin=0) the rotation is exact
    # identity in BF16 in the ORIGINAL interleaved element order -- the HF
    # interleave function deinterleaves internally and permutation P
    # re-interleaves, so the canonical target row 0 equals the RAW (scaled,
    # bf16) input row 0 with no further mapping. This is the block-0
    # methodology's t=0 gate (target vs mapped C++ input, raw equality).
    in_q0 = q_rot.float().cpu().numpy()[0].reshape(2048)
    in_k0 = k_hf[0, 0].float().cpu().numpy()[0]
    t0_q = int((tq[0] == in_q0.astype("<f4")).sum())
    t0_k = int((tk[0] == in_k0.astype("<f4")).sum())
    return tq, tk, t0_q, t0_k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "block2_mla_targets"))
    ns = ap.parse_args()
    out_dir = Path(ns.out_dir).resolve()

    import torch

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    # ---------------- known-answer gate 1: block-0 rope regeneration -------
    tq0, tk0, _, _ = gen_rope_targets(B0_DIR, B0_INPUTS)
    got_q = sha256_bytes(tq0.tobytes())
    got_k = sha256_bytes(tk0.tobytes())
    if got_q != B0_TARGET_Q:
        stop("BLOCK-0 KNOWN-ANSWER FAIL (q): %s != %s" % (got_q, B0_TARGET_Q))
    if got_k != B0_TARGET_K:
        stop("BLOCK-0 KNOWN-ANSWER FAIL (k): %s != %s" % (got_k, B0_TARGET_K))
    print("known-answer gate 1: block-0 rope targets regenerated BYTE-EXACT (q %s..., k %s...)"
          % (got_q[:8], got_k[:8]))

    # ---------------- known-answer gate 2: block-0 S2b scale path ----------
    raw = (B0_KV_DIR / "kv_a_layernorm.bin").read_bytes()
    if sha256_bytes(raw) != B0_KV_NORM_SHA:
        stop("block-0 kv_a_layernorm oracle SHA mismatch")
    b0_norm = np.frombuffer(raw, dtype="<f4").reshape(512, 512)
    ref_numpy = to_bf16(b0_norm * np.float32(MLA_SCALE_KV))
    dev = torch.device("cuda:0")
    ref_torch = (
        (torch.from_numpy(np.ascontiguousarray(b0_norm)).to(dev).to(torch.bfloat16) * MLA_SCALE_KV)
        .float().cpu().numpy().astype("<f4")
    )
    cross = int((ref_torch == ref_numpy).sum())
    if cross != ref_numpy.size:
        stop("S2b torch/numpy cross-check FAIL: %d/%d" % (cross, ref_numpy.size))
    raw_scaled = B0_SCALED.read_bytes()
    if sha256_bytes(raw_scaled) != B0_SCALED_SHA:
        stop("committed block-0 kv_cmpr_scaled artifact SHA mismatch")
    if sha256_bytes(ref_numpy.astype("<f4").tobytes()) != B0_SCALED_SHA:
        stop("BLOCK-0 S2b KNOWN-ANSWER FAIL: reconstruction != committed 909b7ee7")
    print("known-answer gate 2: block-0 kv_cmpr_scaled reproduced BYTE-EXACT (909b7ee7...; torch/numpy cross %d/%d)"
          % (cross, ref_numpy.size))

    # ---------------- block-2 generation (now trusted) ---------------------
    tq2, tk2, t0_q, t0_k = gen_rope_targets(B2_DIR, B2_INPUTS)
    if t0_q != 2048:
        stop("block-2 t=0 identity FAIL (q): %d/2048" % t0_q)
    if t0_k != 64:
        stop("block-2 t=0 identity FAIL (k): %d/64" % t0_k)
    print("block-2 t=0 identity: q 2048/2048, k 64/64 PASS")

    raw2 = (B2_DIR / "kv_a_layernorm.bin").read_bytes()
    if sha256_bytes(raw2) != B2_KV_NORM_SHA:
        stop("block-2 kv_a_layernorm SHA mismatch")
    b2_norm = np.frombuffer(raw2, dtype="<f4").reshape(512, 512)
    scaled_numpy = to_bf16(b2_norm * np.float32(MLA_SCALE_KV)).astype("<f4")
    scaled_torch = (
        (torch.from_numpy(np.ascontiguousarray(b2_norm)).to(dev).to(torch.bfloat16) * MLA_SCALE_KV)
        .float().cpu().numpy().astype("<f4")
    )
    cross2 = int((scaled_torch == scaled_numpy).sum())
    if cross2 != scaled_numpy.size:
        stop("block-2 torch/numpy cross-check FAIL: %d/%d" % (cross2, scaled_numpy.size))
    print("block-2 kv_cmpr_scaled comparandum: torch/numpy cross %d/%d PASS" % (cross2, scaled_numpy.size))

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "block2_q_pe_rope_target.bin": tq2,
        "block2_k_pe_rope_target.bin": tk2,
        "block2_kv_cmpr_scaled_target.bin": scaled_numpy,
    }
    manifest = {
        "known_answer_gates": {
            "block0_rope_regeneration": "BYTE-EXACT (%s / %s)" % (B0_TARGET_Q, B0_TARGET_K),
            "block0_s2b_kv_cmpr_scaled": "BYTE-EXACT (%s)" % B0_SCALED_SHA,
        },
        "rope_interpretation_rule": (
            "targets use captured HF cos/sin; production C++ uses ggml angles "
            "(known not byte-exact: sin 3,377/16,384) - divergence at "
            "q_pe_rope-2/k_pe_rope-2 localizes at most to the production-RoPE "
            "composite / angle-generation state, never rotation arithmetic (R1)"
        ),
        "targets": {},
    }
    sums = []
    for name, arr in outputs.items():
        data = arr.astype("<f4").tobytes()
        sha = sha256_bytes(data)
        (out_dir / name).write_bytes(data)
        meta = {
            "name": name,
            "shape": list(arr.shape),
            "order": "token-major",
            "dtype": "float32-le",
            "sha256": sha,
        }
        (out_dir / name.replace(".bin", ".json")).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["targets"][name] = meta
        sums.append("%s  %s" % (sha, name))
        print("%-34s %s" % (name, sha))
    (out_dir / "targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print("BLOCK2 MLA TARGETS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
