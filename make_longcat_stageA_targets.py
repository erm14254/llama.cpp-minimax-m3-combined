#!/usr/bin/env python3
"""Generate the stage-A / stage-B offline gate targets for the il>=1 MLA
BF16 output-boundary change, gated on block-0 known answers.

Targets (all stored f32-widened, token-major, little-endian):

  T3  block2_q_b_proj_stageA_target.bin      = bf16(quad-run q_b raw)
  T4  block2_q_scaled_stageA_target.bin      = bf16(f32(T3) * f32(sqrt(2)))
  T5  block2_kv_cmpr_scaled_stageA_target.bin= bf16(quad-run kv_cmpr_scaled raw)
  TB4 block2_q_scaled_stageB_target.bin      = bf16(f32(HF q_b oracle) * f32(sqrt(2)))

Premise (verified in-run, not assumed): under the dual reset with stage-A
binaries, the old-norm outputs must byte-reproduce the committed quad-run
norm dumps (2b600082... / 93d7442a...), so the raw q_b GEMM /
kv-scale outputs equal the quad-run raws and stage A adds exactly one RNE
BF16 output rounding on top - which is what T3/T4/T5 encode. TB4 encodes
the stage-B (byte-exact-chain) expectation for the q post-scale surface.

KNOWN-ANSWER GATES (all must pass BEFORE any target is written):
  1. block-0 S2b scale path: bf16(hf_kv_a_layernorm * sqrt(6)) must
     reproduce the committed block-0 kv_cmpr_scaled artifact byte-exact
     (909b7ee7...), with the torch/numpy RNE cross-check - the exact gate-2
     pattern of make_longcat_block2_mla_targets.py.
  2. scale-constant bit identity: f32(3072/1536) == 2.0 exactly and
     f32(sqrt(2)) == 0x3fb504f3, mirroring the hex-reset 0x401cc471 proof
     for sqrt(6) (C++ computes sqrtf((float)3072/(float)1536); IEEE sqrtf
     is correctly rounded, so the f64-derived f32 value is bit-identical).
  3. torch/numpy RNE cross-check on every scale-path target (T4, TB4) and
     every pure-round target (T3, T5).

Measurement-only tooling; no production arithmetic is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
QUAD_DIR = REPO / "cpp_resid_walk_inject3_b2_512"
B2_DIR = Path(r"D:\lc_block2_mla_512")
B0_KV_DIR = Path(r"D:\lc_mla_blackwell")
B0_SCALED = REPO / "cpp_attn0_mla_attnpath_512" / "kv_cmpr_scaled.bin"

INPUTS = {
    "quad_q_b_raw": (
        QUAD_DIR / "block2_q_b_proj_full.bin",
        "23257a1e6891f5daaee74e9cdab16e314c37d86da3605c25c6f6018a03d0b60b",
        (512, 6144),
    ),
    "quad_kv_scaled_raw": (
        QUAD_DIR / "block2_kv_cmpr_scaled_full.bin",
        "7450dd4a91683330d2c213370702223abd13e2758995d85fe8fe24dd215dc3a8",
        (512, 512),
    ),
    "hf_q_b": (
        B2_DIR / "q_b_proj.bin",
        "ecb70ef6c9bd4d6f28a67467f3b5ec3fc575d4f37cd2e81bce7cc7554323f308",
        (512, 6144),
    ),
    "b0_kv_a_norm": (
        B0_KV_DIR / "kv_a_layernorm.bin",
        "b44cc101b03b11d96c0d9c52613f7469141dd7786b8128f93e3b7e912c550373",
        (512, 512),
    ),
}
B0_SCALED_SHA = "909b7ee75366b0ee1d5a912c103762563236cd07c6fd8385ceb1e549f2a86ce8"

# Premise hashes recorded for the manifest (gated in-run by the harness /
# stage comparator, not consumed here):
QUAD_NORM_PREMISE = {
    "block2_q_a_norm_full.bin": "2b60008293032656185fa55ca5f0bb579855c67998ad7082feee8b3991ec8bb4",
    "block2_kv_a_norm_full.bin": "93d7442a30cd7d742f21b777398783ea00faf0e9012658d37dae7d13a07698a9",
}

MLA_SCALE_KV = 2.449489742783178   # sqrt(3072/512), f32 bits 0x401cc471 (hex-reset proof)
SQRT2_BITS = 0x3FB504F3


def stop(msg: str) -> None:
    raise SystemExit("STOP: " + msg)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(key: str) -> np.ndarray:
    path, expected, shape = INPUTS[key]
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != expected:
        stop("input SHA mismatch for %s (%s): %s != %s" % (key, path, got, expected))
    return np.frombuffer(raw, dtype="<f4").reshape(*shape)


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "block2_stageA_targets"))
    ns = ap.parse_args()
    out_dir = Path(ns.out_dir).resolve()

    import torch

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")

    # ------- known-answer gate 1: block-0 S2b scale path (verbatim pattern) --
    b0_norm = load("b0_kv_a_norm")
    ref_numpy = to_bf16(b0_norm * np.float32(MLA_SCALE_KV))
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
    print("known-answer gate 1: block-0 kv_cmpr_scaled reproduced BYTE-EXACT "
          "(909b7ee7...; torch/numpy cross %d/%d)" % (cross, ref_numpy.size))

    # ------- known-answer gate 2: scale-constant bit identity ----------------
    ratio = np.float32(3072.0) / np.float32(1536.0)
    if ratio != np.float32(2.0):
        stop("scale ratio f32(3072/1536) != 2.0")
    sqrt2_f32 = np.float32(np.sqrt(np.float64(2.0)))
    sqrt2_f32_from_f32 = np.float32(np.sqrt(np.float32(2.0), dtype=np.float32))
    bits = int(sqrt2_f32.view(np.uint32))
    bits2 = int(sqrt2_f32_from_f32.view(np.uint32))
    if bits != SQRT2_BITS or bits2 != SQRT2_BITS:
        stop("sqrt(2) f32 bit identity FAIL: 0x%08x / 0x%08x != 0x%08x"
             % (bits, bits2, SQRT2_BITS))
    print("known-answer gate 2: mla_scale_q constant bit-identical 0x%08x "
          "(f64-derived == f32-sqrtf-derived == C++ sqrtf(3072/1536))" % bits)

    # ------- target generation (now trusted) ---------------------------------
    quad_q_b = load("quad_q_b_raw")
    quad_kv_scaled = load("quad_kv_scaled_raw")
    hf_q_b = load("hf_q_b")

    # T3: pure RNE round of the quad-run raw q_b GEMM output.
    t3_numpy = to_bf16(quad_q_b).astype("<f4")
    t3_torch = (
        torch.from_numpy(np.ascontiguousarray(quad_q_b)).to(dev).to(torch.bfloat16)
        .float().cpu().numpy().astype("<f4")
    )
    if int((t3_torch == t3_numpy).sum()) != t3_numpy.size:
        stop("T3 torch/numpy cross-check FAIL")

    # T4: stage-A q post-scale semantics: bf16(f32(T3) * f32(sqrt(2))).
    t4_numpy = to_bf16(t3_numpy * sqrt2_f32).astype("<f4")
    t4_torch = (
        (torch.from_numpy(np.ascontiguousarray(t3_numpy)).to(dev).to(torch.bfloat16) * float(np.float64(2.0) ** 0.5))
        .float().cpu().numpy().astype("<f4")
    )
    if int((t4_torch == t4_numpy).sum()) != t4_numpy.size:
        stop("T4 torch/numpy cross-check FAIL")

    # T5: pure RNE round of the quad-run raw post-scale kv output.
    t5_numpy = to_bf16(quad_kv_scaled).astype("<f4")
    t5_torch = (
        torch.from_numpy(np.ascontiguousarray(quad_kv_scaled)).to(dev).to(torch.bfloat16)
        .float().cpu().numpy().astype("<f4")
    )
    if int((t5_torch == t5_numpy).sum()) != t5_numpy.size:
        stop("T5 torch/numpy cross-check FAIL")

    # TB4: stage-B q post-scale expectation from the byte-exact HF chain:
    # bf16(f32(HF q_b oracle) * f32(sqrt(2))). The HF oracle is verified
    # 100% BF16-on-lattice by construction (f32-widened bf16 capture).
    on_lattice = int((to_bf16(hf_q_b) == hf_q_b).sum())
    if on_lattice != hf_q_b.size:
        stop("HF q_b oracle not BF16-on-lattice: %d/%d" % (on_lattice, hf_q_b.size))
    tb4_numpy = to_bf16(hf_q_b * sqrt2_f32).astype("<f4")
    tb4_torch = (
        (torch.from_numpy(np.ascontiguousarray(hf_q_b)).to(dev).to(torch.bfloat16) * float(np.float64(2.0) ** 0.5))
        .float().cpu().numpy().astype("<f4")
    )
    if int((tb4_torch == tb4_numpy).sum()) != tb4_numpy.size:
        stop("TB4 torch/numpy cross-check FAIL")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "block2_q_b_proj_stageA_target.bin": t3_numpy,
        "block2_q_scaled_stageA_target.bin": t4_numpy,
        "block2_kv_cmpr_scaled_stageA_target.bin": t5_numpy,
        "block2_q_scaled_stageB_target.bin": tb4_numpy,
    }
    manifest = {
        "known_answer_gates": {
            "block0_s2b_kv_cmpr_scaled": "BYTE-EXACT (%s)" % B0_SCALED_SHA,
            "mla_scale_q_bit_identity": "0x%08x" % SQRT2_BITS,
        },
        "premise": {
            "description": (
                "T3/T4/T5 are valid stage-A gates only if the stage-A run's "
                "old-norm outputs byte-reproduce the committed quad-run norm "
                "dumps (checked in-run by the stage comparator)"
            ),
            "quad_norm_hashes": QUAD_NORM_PREMISE,
        },
        "inputs": {k: {"path": str(v[0]), "sha256": v[1]} for k, v in INPUTS.items()},
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
        print("%-42s %s" % (name, sha))
    (out_dir / "targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print("STAGE-A TARGETS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
