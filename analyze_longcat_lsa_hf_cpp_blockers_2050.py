#!/usr/bin/env python3
"""Offline C++<->HF blocker-1/2/3 comparator for the 2050 first-owner bank.

CPU + numpy + stdlib only. No GPU, no model, no torch. Consumes the
byte-identical C++ S-family capture (cpp_lsa_2050_S1) and the HF Run A
capture (hf_lsa_2050_capture); emits <out-dir>/verdict.json.

Exit codes: 0 = analysis completed with every integrity/known-answer/
validity gate passed (dependency flags and CONDITIONAL labels are
verdict content); 1 = analysis aborted (reasons in verdict.json);
3 = REFUSED TO START (non-scientific: --out-dir already exists, an
earlier verdict is never overwritten). `--self-test` runs the helper
unit tests only (0 pass / 1 fail) and touches no directories.

Pre-registered semantics (authored and committed BEFORE any HF
execution; the comparator implements them and never re-decides them):

  Integrity phase (before any science; the analogue of the C++
  determinism comparator's Phase-0 discipline): parse and reverify both
  capture manifests (rehash every consumed C++ artifact against the
  S-dir SHA256SUMS.txt AND the eight frozen byte-stable hashes; rehash
  the complete expected HF SHA256SUMS.txt inventory), bind the C++
  run_provenance.json (git_head = the 2050 protocol commit, token
  stream eb04e101...) and the HF summary.json gates (runA==runB logits,
  frozen runtime/token SHAs). Any integrity failure aborts.

  Phase 0 (cross-side upstream attribution guard; guards all blocker
  analyses): compare, in causal order,
    (1) attn_norm-0        C++ vs HF   (bitwise on widened-F32 carriers;
        C++ BF16-lattice membership is part of the expectation),
    (2) q_a_norm-0         C++ vs HF   (bitwise),
    (3) rne_bf16(C++ k_proj raw F32) vs HF wk output (bitwise),
    (4) C++ q_proj (BF16)  vs HF wq_b output (bitwise),
    (5) indexer scoring weights: C++ lsa_indexer_weights (post-scale
        F32) vs HF hf_indexer_weights_prescale * f32(1/sqrtf(2048))
        (replicating the ggml_scale constant). Surface 5 is a raw-F32
        GEMM pair from different backends: the pre-registered agreement
        class is reduction-noise (max rel <= 1e-4 with the denominator
        floored at 1e-3*max|w|, and max abs <= 1e-4*max|w|); exact-equal
        counts and difference stats are always reported explicitly.
  "bitwise" means uint32 comparison of the widened-F32 carriers --
  signed zero is NOT normalized. Disagreements are reported as explicit
  numerical classes (counts + bf16-ulp histograms), never silently
  tolerated; no frozen project criterion is widened. Dependency flags:
  UPSTREAM_K_PROJ_MISMATCH, UPSTREAM_Q_PROJ_MISMATCH,
  UPSTREAM_INDEXER_WEIGHTS_MISMATCH, EARLIEST_DIVERGENCE. Under a flag
  the affected conclusions are CONDITIONAL and cannot discharge their
  blocker; membership/top-K causal attribution additionally requires
  surface-5 agreement.

  Blocker 1 (K-norm cast ordering): C_raw / C_bf16 / H_bf16 models with
  dual known-answer gates (<=1 bf16 ulp classes), the three reported
  deltas, the post-RoPE known-answer gate on the C++-class RoPE
  reference (outside <=1 ulp => membership reconstruction INVALID and
  stops), the C_raw membership baseline gate, then the C_bf16/H_bf16
  swaps. Verdicts MEMBERSHIP-INVARIANT / MEMBERSHIP-AFFECTING(n) /
  BORDERLINE (margin_floor = 1e-4 * max|finite row score|).

  Blocker 2 (rope/nope layout): per-side K nope identities (hard),
  per-side Q no-RoPE-half invariants, pre-RoPE K-norm cross-side
  agreement recorded explicitly, per-side pair-decode hypothesis
  acceptance, and the candidate permutation pi: HF[i] = C++[2i],
  HF[32+i] = C++[2i+1]. The cross-side K mapping under pi receives an
  unconditional verdict ONLY when the pre-RoPE K-norm surfaces agree
  bitwise (otherwise it is characterization CONDITIONAL on blocker 1 --
  K-side layout is then adjudicated from per-side decoding + identities
  + source proof). ROPE-SLICE MISMATCH is an implemented verdict class:
  a per-side Q no-RoPE-half invariant failure, or cross-side Q-nope
  inequality under Phase-0 Q-projection agreement, yields it and
  forbids any layout-equivalent verdict. Vocabulary: EQUAL AS-IS /
  EQUAL UNDER PI / ROUNDING-CLASS RESIDUAL under PI (max ulp <= 4) /
  ROPE-SLICE MISMATCH / NUMERIC DISAGREEMENT.

  Blocker 3 (YaRN attn_factor/mscale): explicit SCALE, SCHEDULE and
  ROUNDING-CLASS sub-verdicts. Effective-mscale extraction runs on K
  AND Q, per side. Static verification measured the ggml-class vs
  HF-class f32-chain reference separation at ~3e-5, below BF16
  resolution: when both references fit a side's decoded angles within
  the pre-registered BF16-class residual (0.01) and the reference
  separation is < 2e-3, the SCHEDULE outcome is "NOT EMPIRICALLY
  DISTINGUISHABLE at captured BF16 precision" -- formula discrimination
  is never overclaimed. Membership impact of the angle-set swap is
  reported separately.

  Section 11: exact top-K membership sets, rows 2048/2049, all 14
  owners. Causal blocker-1/2/3 attribution is authorized for owner00
  ONLY (its upstream surfaces are captured and Phase-0 gated, incl.
  surface 5); owners 02..26 are cross-owner semantic observations only.

This comparator makes NO Gate-4 statement; Gate 4 remains NOT RUN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

N_TOKENS = 2050
TOPK = 2048
HEAD_DIM = 128
ROPE_DIM = 64
N_HEADS = 16
INIT_TOKENS = 16
LOCAL_TOKENS = 1024
EPS_INDEXER = 1e-6

FREQ_BASE = 1.0e6
YARN_FACTOR = 120.0
N_CTX_ORIG = 8192
BETA_FAST = 32.0
BETA_SLOW = 1.0
EXT_FACTOR = 1.0

ROUNDING_CLASS_MAX_ULP = 4
MARGIN_FLOOR_REL = 1e-4
WEIGHTS_CLASS_REL = 1e-4
SCHEDULE_FIT_MAX_RESID = 0.01
SCHEDULE_REF_SEPARATION = 2e-3

EXPECTED_CPP_GIT_HEAD = "2dd49d39c11a4378ebd3abed2a51aea3f575accb"
EXPECTED_TOKEN_SHA256 = (
    "eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed"
)
EXPECTED_RUNTIME_SHA256 = (
    "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
)

# Frozen byte-stable C++ S-family surfaces (V-input table of the
# determinism-round verdict; byte-identical across S1/S2/S3).
CPP_STABLE_SHA256 = {
    "lsa_anchor_attn_norm0_full.bin": "28f15cb7fb59b64d6ee565155e6b29cbbb66e297f362f9a66b4c8d1245d3c046",
    "lsa_anchor_q_a_norm0_full.bin": "49d3d02d70705cad4ca371e535933cd2cea5651d8b02230ae2e303f68ac01d2e",
    "lsa_indexer_k_proj_full.bin": "5ddd67d8c0085aa63443b6f75b837004914097464a3815192d1961e11d4b0e96",
    "lsa_indexer_k_norm_full.bin": "57deb53cbb34ef3a14f36d4672ef3c269f9ae93138fe7cca55d12bf8186880e6",
    "lsa_indexer_k_full.bin": "2f57bc0f5d39534e3e55c17a7c59581476fd394dcb3a26a03e0723b99be1cf75",
    "lsa_indexer_q_proj_full.bin": "a75dfb8070dba5682cb31031dde3e6d29e02fac516946daff633117dce9dd7f3",
    "lsa_indexer_q_full.bin": "cc8ecd1a56a53f57c365d3cde552712d9b5c2b88cd013593e6f470458c6ccfd6",
    "lsa_indexer_weights_full.bin": "321a15e612641287a1ad5456ea5d5b9af2ee5134e2155ffba17a4593270d741e",
}

CPP_WIDTHS = {
    "lsa_anchor_attn_norm0_full.bin": 3072,
    "lsa_anchor_q_a_norm0_full.bin": 1536,
    "lsa_indexer_k_proj_full.bin": 128,
    "lsa_indexer_k_norm_full.bin": 128,
    "lsa_indexer_k_full.bin": 128,
    "lsa_indexer_q_proj_full.bin": 2048,
    "lsa_indexer_q_full.bin": 2048,
    "lsa_indexer_weights_full.bin": 16,
}

HF_WIDTHS = {
    "hf_attn_norm0.bin": 3072,
    "hf_q_a_norm0.bin": 1536,
    "hf_indexer_k_proj.bin": 128,
    "hf_indexer_k_norm.bin": 128,
    "hf_indexer_k.bin": 128,
    "hf_indexer_q_proj.bin": 2048,
    "hf_indexer_q.bin": 2048,
    "hf_indexer_weights_prescale.bin": 16,
    "hf_rope_cos.bin": 64,
    "hf_rope_sin.bin": 64,
}

HF_WEIGHT_BINS = {
    "hf_weight_k_norm.bin": 128,
    "hf_weight_wk.bin": 128 * 3072,
    "hf_weight_wq_b.bin": 2048 * 1536,
    "hf_weight_weights_proj.bin": 16 * 3072,
}

REASONS: list[str] = []


def stop(msg: str) -> None:
    REASONS.append(msg)
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- bf16 / bitwise helpers ----------------

def _u32(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="<f4")
    return x.view(np.uint32)


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise equality of the widened-F32 carriers (signed zero is NOT
    normalized: -0.0 != +0.0 here)."""
    ua, ub = _u32(a), _u32(b)
    return ua.shape == ub.shape and bool(np.array_equal(ua, ub))


def bits_diff_count(a: np.ndarray, b: np.ndarray) -> int:
    return int((_u32(a) != _u32(b)).sum())


def rne_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16 bit pattern (finite inputs)."""
    u = _u32(x).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def widen_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view("<f4")


def rne_bf16(x: np.ndarray) -> np.ndarray:
    return widen_bits(rne_bf16_bits(x))


def bf16_order_key(b: np.ndarray) -> np.ndarray:
    """Monotone total-order key over BF16 bit patterns.

    -0 is intentionally identified with +0 (both map to 0x8000), and
    adjacent finite values remain exactly one unit apart across zero:
    -min_subnormal (0x8001) -> 0x7FFF, zero -> 0x8000,
    +min_subnormal (0x0001) -> 0x8001.
    """
    b = b.astype(np.int64)
    b = np.where(b == 0x8000, 0, b)  # -0 -> +0
    return np.where(b >= 0x8000, 0x10000 - b, b + 0x8000)


def bf16_ulp_dist(a_f32: np.ndarray, b_f32: np.ndarray) -> np.ndarray:
    ka = bf16_order_key(rne_bf16_bits(a_f32))
    kb = bf16_order_key(rne_bf16_bits(b_f32))
    return np.abs(ka - kb)


def ulp_hist(dist: np.ndarray) -> dict:
    dist = dist.ravel()
    return {
        "n": int(dist.size),
        "eq": int((dist == 0).sum()),
        "ulp1": int((dist == 1).sum()),
        "ulp2": int((dist == 2).sum()),
        "ulp3_4": int(((dist >= 3) & (dist <= 4)).sum()),
        "gt4": int((dist > 4).sum()),
        "max": int(dist.max()) if dist.size else 0,
    }


def load_f32(path: Path, width: int) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype="<f4")
    if data.size != N_TOKENS * width:
        stop(f"{path.name}: size {data.size} != {N_TOKENS}x{width}")
    return data.reshape(N_TOKENS, width)


def parse_sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        stop(f"manifest missing: {path}")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = re.match(r"^([0-9a-f]{64})\s+(.+)$", line.strip())
        if not m:
            stop(f"malformed manifest line in {path.name}: {line!r}")
        entries[m.group(2)] = m.group(1)
    return entries


# ---------------- angle references ----------------

def f32(x) -> np.float32:
    return np.float32(x)


def ggml_corr_dims() -> tuple[np.float32, np.float32]:
    def corr_dim(n_rot: float) -> np.float32:
        return f32(
            f32(ROPE_DIM)
            * np.log(f32(N_CTX_ORIG / (n_rot * 2.0 * math.pi)))
            / f32(2.0 * np.log(f32(FREQ_BASE)))
        )

    low = f32(max(0.0, math.floor(float(corr_dim(BETA_FAST)))))
    high = f32(min(ROPE_DIM - 1, math.ceil(float(corr_dim(BETA_SLOW)))))
    return low, high


def cpp_attn_factor() -> np.float32:
    freq_scale = f32(1.0) / f32(YARN_FACTOR)
    factor = f32(1.0) / freq_scale
    # llama-context: get_mscale(factor,1)/get_mscale(factor,1) == 1.0,
    # then the ext-factor cancellation 1/(1+0.1*ln(factor)).
    return f32(1.0) / (f32(1.0) + f32(0.1) * np.log(factor))


def ggml_angles() -> tuple[np.ndarray, np.ndarray, np.float32]:
    """f32-chain replication of the CUDA rope_norm/rope_yarn path.

    Returns cos[2050, 32], sin[2050, 32] (per pair index i = i0/2) and
    the effective mscale actually multiplied into cos/sin.
    """
    freq_scale = f32(1.0) / f32(YARN_FACTOR)
    theta_scale = np.power(f32(FREQ_BASE), f32(-2.0 / ROPE_DIM), dtype=np.float32)
    low, high = ggml_corr_dims()
    attn_factor = cpp_attn_factor()
    mscale = f32(attn_factor * (f32(1.0) + f32(0.1) * np.log(f32(1.0) / freq_scale)))

    pos = np.arange(N_TOKENS, dtype=np.float32)[:, None]
    i0 = np.arange(0, ROPE_DIM, 2, dtype=np.float32)[None, :]
    theta_extrap = (pos * np.power(theta_scale, (i0 / f32(2.0)), dtype=np.float32)).astype(np.float32)
    theta_interp = (freq_scale * theta_extrap).astype(np.float32)
    y = ((i0 / f32(2.0)) - low) / np.maximum(f32(0.001), high - low)
    ramp = (f32(1.0) - np.minimum(f32(1.0), np.maximum(f32(0.0), y))).astype(np.float32)
    mix = (ramp * f32(EXT_FACTOR)).astype(np.float32)
    theta = (theta_interp * (f32(1.0) - mix) + theta_extrap * mix).astype(np.float32)
    cos = (np.cos(theta, dtype=np.float32) * mscale).astype(np.float32)
    sin = (np.sin(theta, dtype=np.float32) * mscale).astype(np.float32)
    return cos, sin, mscale


def hf_angles() -> tuple[np.ndarray, np.ndarray, float]:
    """f32-chain replication of modeling_rope_utils yarn + rotary forward.

    Returns cos[2050, 32], sin[2050, 32] in f32 (pre-bf16-cast; the
    at-use HF values are rne_bf16 of these), and attention_scaling.
    """
    k = np.arange(0, ROPE_DIM, 2, dtype=np.float32) / f32(ROPE_DIM)
    pos_freqs = np.power(f32(FREQ_BASE), k, dtype=np.float32)
    extrap = (f32(1.0) / pos_freqs).astype(np.float32)
    interp = (f32(1.0) / (f32(YARN_FACTOR) * pos_freqs)).astype(np.float32)

    def corr(n_rot: float) -> float:
        return (ROPE_DIM * math.log(N_CTX_ORIG / (n_rot * 2 * math.pi))) / (
            2 * math.log(FREQ_BASE)
        )

    low = max(math.floor(corr(BETA_FAST)), 0)
    high = min(math.ceil(corr(BETA_SLOW)), ROPE_DIM - 1)
    lin = (np.arange(ROPE_DIM // 2, dtype=np.float32) - f32(low)) / f32(high - low)
    ramp = np.clip(lin, 0.0, 1.0).astype(np.float32)
    ext = (f32(1.0) - ramp).astype(np.float32)
    inv_freq = (interp * (f32(1.0) - ext) + extrap * ext).astype(np.float32)

    pos = np.arange(N_TOKENS, dtype=np.float32)[:, None]
    theta = (pos * inv_freq[None, :]).astype(np.float32)
    attention_scaling = 1.0  # mscale == mscale_all_dim == 1 -> exactly 1.0
    cos = (np.cos(theta, dtype=np.float32) * f32(attention_scaling)).astype(np.float32)
    sin = (np.sin(theta, dtype=np.float32) * f32(attention_scaling)).astype(np.float32)
    return cos, sin, attention_scaling


def rotate_cpp_class(norm_out_f32: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """C++-class rope propagation: widen -> rotate pairs (2i, 2i+1) in
    float64 with f32 cos/sin -> concat nope -> single rne_bf16."""
    x = norm_out_f32.astype(np.float64)
    x0 = x[:, 0:ROPE_DIM:2]
    x1 = x[:, 1:ROPE_DIM:2]
    c = cos.astype(np.float64)
    s = sin.astype(np.float64)
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    out = np.empty_like(x)
    out[:, 0:ROPE_DIM:2] = r0
    out[:, 1:ROPE_DIM:2] = r1
    out[:, ROPE_DIM:] = x[:, ROPE_DIM:]
    return rne_bf16(out.astype(np.float32))


def pi_map(cpp_roped: np.ndarray) -> np.ndarray:
    """Candidate permutation pi: HF[i] = C++[2i], HF[32+i] = C++[2i+1]."""
    out = np.empty_like(cpp_roped)
    out[:, : ROPE_DIM // 2] = cpp_roped[:, 0:ROPE_DIM:2]
    out[:, ROPE_DIM // 2:] = cpp_roped[:, 1:ROPE_DIM:2]
    return out


# ---------------- blocker-1 norm models ----------------

def model_C_class(x_f32: np.ndarray, w_f64: np.ndarray) -> np.ndarray:
    x = x_f32.astype(np.float64)
    var = np.mean(x * x, axis=1, keepdims=True)
    y = x / np.sqrt(var + EPS_INDEXER) * w_f64[None, :]
    return rne_bf16(y.astype(np.float32))


def model_H_class(x_f32: np.ndarray, w_f64: np.ndarray) -> np.ndarray:
    xb = rne_bf16(x_f32).astype(np.float64)
    var = np.mean(xb * xb, axis=1, keepdims=True)
    z = xb / np.sqrt(var + EPS_INDEXER)
    inner = rne_bf16(z.astype(np.float32)).astype(np.float64)
    return rne_bf16((inner * w_f64[None, :]).astype(np.float32))


# ---------------- membership re-scoring ----------------

def forced_positions(p: int) -> np.ndarray:
    return np.concatenate(
        (
            np.arange(0, INIT_TOKENS, dtype=np.int64),
            np.arange(p - LOCAL_TOKENS + 1, p + 1, dtype=np.int64),
        )
    )


def rescore_row(
    p: int,
    q_row_f64: np.ndarray,        # [16, 128]
    weights_row_f64: np.ndarray,  # [16] (C++ post-scale weights)
    k_all_f32: np.ndarray,        # [2050, 128] variant post-rope K
) -> tuple[set[int], float, float]:
    """Returns (membership set, boundary margin, max abs finite score)."""
    k = k_all_f32[: p + 1].astype(np.float64)          # causal candidates
    dots = q_row_f64 @ k.T                             # [16, p+1]
    scores = weights_row_f64 @ np.maximum(dots, 0.0)   # [p+1]
    finite_max = float(np.max(np.abs(scores)))
    forced = forced_positions(p)
    masked = scores.copy()
    masked[forced] = np.inf
    order = np.argsort(-masked, kind="stable")
    selected = order[:TOPK]
    boundary_in = masked[order[TOPK - 1]]
    boundary_out = masked[order[TOPK]] if order.size > TOPK else -np.inf
    if np.isinf(boundary_in):
        margin = float("inf")
    else:
        margin = float(boundary_in - boundary_out)
    return set(int(x) for x in selected), margin, finite_max


def membership_verdict(
    base: set[int], variant: set[int], margin: float, finite_max: float
) -> dict:
    floor = MARGIN_FLOOR_REL * finite_max
    diff = base.symmetric_difference(variant)
    if not diff:
        v = "MEMBERSHIP-INVARIANT"
        if margin < floor:
            v = "BORDERLINE"
    else:
        v = f"MEMBERSHIP-AFFECTING ({len(diff)} positions)"
        if margin < floor:
            v += " [boundary margin below noise floor]"
    return {
        "verdict": v,
        "sym_diff_size": len(diff),
        "sym_diff": sorted(diff)[:64],
        "boundary_margin": margin,
        "margin_floor": floor,
    }


def pair_norm_ratio(pre: np.ndarray, post: np.ndarray, layout: str) -> dict:
    """Effective-mscale extraction: pre/post are [rows, 64] roped blocks;
    pre pairs are interleaved (2i, 2i+1); post pairing per `layout`."""
    p0 = pre[:, 0:ROPE_DIM:2].astype(np.float64)
    p1 = pre[:, 1:ROPE_DIM:2].astype(np.float64)
    if layout == "interleaved":
        q0 = post[:, 0:ROPE_DIM:2].astype(np.float64)
        q1 = post[:, 1:ROPE_DIM:2].astype(np.float64)
    else:  # half-split output
        q0 = post[:, : ROPE_DIM // 2].astype(np.float64)
        q1 = post[:, ROPE_DIM // 2:].astype(np.float64)
    pre_n = np.sqrt(p0 * p0 + p1 * p1)
    post_n = np.sqrt(q0 * q0 + q1 * q1)
    good = pre_n > np.quantile(pre_n, 0.05)
    r = post_n[good] / pre_n[good]
    return {
        "median": float(np.median(r)),
        "mean": float(np.mean(r)),
        "std": float(np.std(r)),
    }


# ---------------- self-tests (correction 6) ----------------

def run_self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"self-test {'PASS' if cond else 'FAIL'}: {name}")
        if not cond:
            failures.append(name)

    # RNE ties-to-even, both directions.
    t = np.array([1.00390625, 1.01171875], dtype=np.float32)
    rt = rne_bf16(t)
    check("rne tie-to-even down", rt[0] == np.float32(1.0))
    check("rne tie-to-even up", rt[1] == np.float32(1.015625))

    # Bitwise equality does NOT normalize signed zero.
    check(
        "bits_equal distinguishes -0/+0",
        not bits_equal(np.array([-0.0], dtype=np.float32),
                       np.array([0.0], dtype=np.float32)),
    )

    # ULP adjacency: positive, negative, signed zero, cross-zero.
    pos_min = widen_bits(np.array([0x0001], dtype=np.uint16))
    neg_min = widen_bits(np.array([0x8001], dtype=np.uint16))
    zero = np.array([0.0], dtype=np.float32)
    nzero = np.array([-0.0], dtype=np.float32)
    one = np.array([1.0], dtype=np.float32)
    one_next = widen_bits(np.array([0x3F81], dtype=np.uint16))   # nextafter(1)
    mone = np.array([-1.0], dtype=np.float32)
    mone_toz = widen_bits(np.array([0xBF7F], dtype=np.uint16))   # toward zero
    check("positive adjacency = 1", int(bf16_ulp_dist(one, one_next)[0]) == 1)
    check("negative adjacency = 1", int(bf16_ulp_dist(mone, mone_toz)[0]) == 1)
    check("signed zero identified", int(bf16_ulp_dist(nzero, zero)[0]) == 0)
    check("cross-zero -min_sub<->0 = 1", int(bf16_ulp_dist(neg_min, zero)[0]) == 1)
    check("cross-zero 0<->+min_sub = 1", int(bf16_ulp_dist(zero, pos_min)[0]) == 1)
    check("cross-zero -min_sub<->+min_sub = 2",
          int(bf16_ulp_dist(neg_min, pos_min)[0]) == 2)

    # Monotone key over a negative-to-positive sample.
    sample = np.array([-2.0, -1.0, -0.0078125, 0.0, 0.0078125, 1.0, 2.0],
                      dtype=np.float32)
    keys = bf16_order_key(rne_bf16_bits(sample))
    check("order key strictly monotone", bool(np.all(np.diff(keys) > 0)))

    # Rotation identity (zero angle) and energy preservation.
    x = rne_bf16(np.random.RandomState(0).randn(4, HEAD_DIM).astype(np.float32))
    cos1 = np.ones((4, 32), dtype=np.float32)
    sin0 = np.zeros((4, 32), dtype=np.float32)
    check("rotation identity at zero angle", bits_equal(rotate_cpp_class(x, cos1, sin0), x))
    th = np.full((4, 32), 0.7, dtype=np.float32)
    out2 = rotate_cpp_class(x, np.cos(th).astype(np.float32), np.sin(th).astype(np.float32))
    p0, p1 = x[:, 0:ROPE_DIM:2].astype(np.float64), x[:, 1:ROPE_DIM:2].astype(np.float64)
    q0, q1 = out2[:, 0:ROPE_DIM:2].astype(np.float64), out2[:, 1:ROPE_DIM:2].astype(np.float64)
    ratio = np.sqrt((q0 ** 2 + q1 ** 2) / (p0 ** 2 + p1 ** 2 + 1e-30))
    check("rotation energy ratio ~ 1", abs(float(np.median(ratio)) - 1.0) < 0.01)

    # pi mapping definition.
    blk = np.arange(4 * ROPE_DIM, dtype=np.float32).reshape(4, ROPE_DIM)
    mapped = pi_map(blk)
    check("pi maps even lanes to first half",
          bool(np.array_equal(mapped[:, :32], blk[:, 0:ROPE_DIM:2])))
    check("pi maps odd lanes to second half",
          bool(np.array_equal(mapped[:, 32:], blk[:, 1:ROPE_DIM:2])))

    # Angle references.
    lo, hi = ggml_corr_dims()
    check("corr dims == (8, 17)", (float(lo), float(hi)) == (8.0, 17.0))
    gc_, gs_, gm = ggml_angles()
    hc_, hs_, hsc = hf_angles()
    check("ggml angle grid shape", gc_.shape == (N_TOKENS, 32))
    check("hf angle grid shape", hc_.shape == (N_TOKENS, 32))
    check("ggml effective mscale ~ 1.0", abs(float(gm) - 1.0) < 1e-6)
    check("hf attention_scaling == 1.0", hsc == 1.0)
    sep = max(float(np.max(np.abs(gc_ - hc_))), float(np.max(np.abs(gs_ - hs_))))
    check("reference separation below BF16 resolution",
          sep < SCHEDULE_REF_SEPARATION)

    # Re-scoring structure (forced containment, sizes).
    q = np.random.RandomState(1).randn(N_HEADS, HEAD_DIM)
    w = np.abs(np.random.RandomState(2).randn(N_HEADS))
    kk = rne_bf16(np.random.RandomState(3).randn(N_TOKENS, HEAD_DIM).astype(np.float32))
    s, margin, _ = rescore_row(2049, q, w, kk)
    forced = set(range(INIT_TOKENS)) | set(range(2049 - LOCAL_TOKENS + 1, 2050))
    check("rescore selects 2048 causal", len(s) == TOPK and max(s) <= 2049)
    check("rescore forced containment", forced.issubset(s))
    check("rescore positive margin", margin > 0)

    if failures:
        print(f"SELF-TEST FAILURES: {failures}")
        return 1
    print("comparator self-test: ALL PASS")
    return 0


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-s-dir")
    ap.add_argument("--hf-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--self-test", action="store_true")
    ns = ap.parse_args()

    if ns.self_test:
        return run_self_test()
    if not (ns.cpp_s_dir and ns.hf_dir and ns.out_dir):
        print("STOP: --cpp-s-dir, --hf-dir and --out-dir are required")
        return 1

    cpp_dir = Path(ns.cpp_s_dir).resolve()
    hf_dir = Path(ns.hf_dir).resolve()
    out_dir = Path(ns.out_dir).resolve()

    # Fresh-dir contract: never overwrite an earlier verdict. This is a
    # non-scientific refusal, distinct from analysis failure.
    if out_dir.exists():
        print(f"REFUSED-TO-START: out dir already exists: {out_dir}")
        print("(an earlier verdict is never overwritten; exit 3)")
        return 3
    out_dir.mkdir(parents=True, exist_ok=False)

    verdict: dict = {
        "protocol": (
            "HF 2050 first-owner blockers 1-3 offline comparator "
            "(pre-registered; integrity phase then Phase-0 attribution "
            "guard; NOT a Gate-4 criterion; Gate 4 remains NOT RUN)"
        ),
        "cpp_s_dir": str(cpp_dir),
        "hf_dir": str(hf_dir),
        "reasons": REASONS,
    }

    def finish(code: int) -> int:
        (out_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"verdict_json={out_dir / 'verdict.json'}")
        print(f"exit={code} reasons={REASONS}")
        return code

    try:
        # ============ integrity phase (before any science) ============
        integrity: dict = {}

        # C++ manifest + provenance binding.
        cpp_sums_path = cpp_dir / "SHA256SUMS.txt"
        cpp_sums = parse_sums(cpp_sums_path)
        integrity["cpp_manifest_sha256"] = sha256_file(cpp_sums_path)
        cpp_needed = list(CPP_STABLE_SHA256) + [
            f"lsa_top_k_owner{2 * li:02d}_full.bin" for li in range(14)
        ]
        for name in cpp_needed:
            p = cpp_dir / name
            if not p.is_file():
                stop(f"C++ artifact missing: {p}")
            if name not in cpp_sums:
                stop(f"C++ manifest lacks entry for {name}")
            got = sha256_file(p)
            if got != cpp_sums[name]:
                stop(f"C++ manifest rehash mismatch: {name}")
            if name in CPP_STABLE_SHA256 and got != CPP_STABLE_SHA256[name]:
                stop(
                    f"C++ surface {name} SHA {got} != frozen byte-stable "
                    f"{CPP_STABLE_SHA256[name]}"
                )
        cpp_prov_path = cpp_dir / "run_provenance.json"
        if not cpp_prov_path.is_file():
            stop(f"C++ run_provenance.json missing: {cpp_prov_path}")
        cpp_prov = json.loads(cpp_prov_path.read_text(encoding="utf-8"))
        if cpp_prov.get("git_head") != EXPECTED_CPP_GIT_HEAD:
            stop(
                "C++ provenance git_head "
                f"{cpp_prov.get('git_head')} != {EXPECTED_CPP_GIT_HEAD}"
            )
        if cpp_prov.get("token_stream_sha256_reconstructed") != EXPECTED_TOKEN_SHA256:
            stop("C++ provenance token stream SHA mismatch")
        integrity["cpp_provenance_git_head"] = cpp_prov.get("git_head")

        # HF manifest: exact expected inventory, every entry rehashed.
        hf_sums_path = hf_dir / "SHA256SUMS.txt"
        hf_sums = parse_sums(hf_sums_path)
        integrity["hf_manifest_sha256"] = sha256_file(hf_sums_path)
        hf_expected_names = (
            list(HF_WIDTHS)
            + [f"hf_top_k_owner{2 * li:02d}.bin" for li in range(14)]
            + list(HF_WEIGHT_BINS)
            + ["hf_logits_2050_runA.bin", "summary.json"]
        )
        if set(hf_sums) != set(hf_expected_names):
            missing = sorted(set(hf_expected_names) - set(hf_sums))
            extra = sorted(set(hf_sums) - set(hf_expected_names))
            stop(f"HF manifest inventory mismatch: missing={missing} extra={extra}")
        for name, sha in hf_sums.items():
            p = hf_dir / name
            if not p.is_file():
                stop(f"HF artifact missing: {p}")
            if sha256_file(p) != sha:
                stop(f"HF manifest rehash mismatch: {name}")
        hf_summary = json.loads((hf_dir / "summary.json").read_text(encoding="utf-8"))
        integrity["hf_summary_sha256"] = hf_sums["summary.json"]
        gates = hf_summary.get("gates", {})
        if gates.get("runA_logits_byte_equal_runB") is not True:
            stop("HF summary gate runA_logits_byte_equal_runB is not True")
        if hf_summary.get("tokens_bin_sha256") != EXPECTED_TOKEN_SHA256:
            stop("HF summary token stream SHA mismatch")
        if hf_summary.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256:
            stop("HF summary frozen runtime SHA mismatch")
        verdict["integrity"] = integrity

        # ---- load surfaces ----
        cpp: dict[str, np.ndarray] = {
            name: load_f32(cpp_dir / name, CPP_WIDTHS[name])
            for name in CPP_STABLE_SHA256
        }
        cpp_topk = {}
        for li in range(14):
            name = f"lsa_top_k_owner{2 * li:02d}_full.bin"
            arr = load_f32(cpp_dir / name, TOPK)
            ints = arr.astype(np.int64)
            if not np.array_equal(ints.astype("<f4"), arr):
                stop(f"{name}: values not exactly integral")
            cpp_topk[2 * li] = ints
        hf: dict[str, np.ndarray] = {
            name: load_f32(hf_dir / name, HF_WIDTHS[name]) for name in HF_WIDTHS
        }
        hf_topk = {}
        for li in range(14):
            name = f"hf_top_k_owner{2 * li:02d}.bin"
            arr = load_f32(hf_dir / name, TOPK)
            ints = arr.astype(np.int64)
            if not np.array_equal(ints.astype("<f4"), arr):
                stop(f"{name}: values not exactly integral")
            hf_topk[2 * li] = ints
        w_knorm = np.frombuffer((hf_dir / "hf_weight_k_norm.bin").read_bytes(), dtype="<f4")
        if w_knorm.size != HEAD_DIM:
            stop(f"hf_weight_k_norm size {w_knorm.size} != {HEAD_DIM}")
        if not bits_equal(rne_bf16(w_knorm), w_knorm):
            stop("hf_weight_k_norm is not on the BF16 lattice")

        # BF16-lattice membership where BF16 sources are claimed.
        for name in (
            "hf_attn_norm0.bin",
            "hf_q_a_norm0.bin",
            "hf_indexer_k_proj.bin",
            "hf_indexer_k_norm.bin",
            "hf_indexer_k.bin",
            "hf_indexer_q_proj.bin",
            "hf_indexer_q.bin",
            "hf_rope_cos.bin",
            "hf_rope_sin.bin",
        ):
            v = hf[name]
            bad = bits_diff_count(rne_bf16(v), v)
            if bad:
                stop(f"{name}: {bad} values off the BF16 lattice")

        # ================= Phase 0 =================
        phase0: dict = {"surfaces": {}, "flags": []}

        def compare_bitwise(tag, a_f32, b_f32, lattice_violations=0):
            beq = bits_equal(a_f32, b_f32)
            eq = beq and lattice_violations == 0
            entry = {
                "bitwise_equal": bool(beq),
                "cpp_lattice_violations": int(lattice_violations),
                "agreement": bool(eq),
            }
            if not beq:
                entry["n_diff_bits"] = bits_diff_count(a_f32, b_f32)
                entry["ulp_hist"] = ulp_hist(bf16_ulp_dist(a_f32, b_f32))
            phase0["surfaces"][tag] = entry
            return eq

        cpp_an0 = cpp["lsa_anchor_attn_norm0_full.bin"]
        lat_an0 = bits_diff_count(rne_bf16(cpp_an0), cpp_an0)
        ok1 = compare_bitwise("1_attn_norm0", cpp_an0, hf["hf_attn_norm0.bin"], lat_an0)
        ok2 = compare_bitwise(
            "2_q_a_norm0",
            cpp["lsa_anchor_q_a_norm0_full.bin"],
            hf["hf_q_a_norm0.bin"],
        )
        ok3 = compare_bitwise(
            "3_k_proj_bf16_boundary",
            rne_bf16(cpp["lsa_indexer_k_proj_full.bin"]),
            hf["hf_indexer_k_proj.bin"],
        )
        ok4 = compare_bitwise(
            "4_q_proj",
            cpp["lsa_indexer_q_proj_full.bin"],
            hf["hf_indexer_q_proj.bin"],
        )

        # Surface 5: indexer scoring weights (raw-F32 GEMM pair; the
        # pre-registered class is fp32 reduction noise, never bf16).
        cpp_w = cpp["lsa_indexer_weights_full.bin"]
        scale_c = f32(1.0) / np.sqrt(f32(2048.0), dtype=np.float32)
        hf_w_scaled = (hf["hf_indexer_weights_prescale.bin"] * scale_c).astype(np.float32)
        wmax = float(np.max(np.abs(cpp_w)))
        abs_d = np.abs(cpp_w.astype(np.float64) - hf_w_scaled.astype(np.float64))
        denom = np.maximum(np.abs(cpp_w.astype(np.float64)), 1e-3 * wmax)
        rel_d = abs_d / denom
        ok5 = bool(
            float(rel_d.max()) <= WEIGHTS_CLASS_REL
            and float(abs_d.max()) <= WEIGHTS_CLASS_REL * wmax
        )
        phase0["surfaces"]["5_indexer_weights"] = {
            "comparison": "cpp lsa_indexer_weights vs hf_indexer_weights_prescale * f32(1/sqrtf(2048))",
            "bitwise_equal": bool(bits_equal(cpp_w, hf_w_scaled)),
            "exact_equal_count": int((_u32(cpp_w) == _u32(hf_w_scaled)).sum()),
            "n": int(cpp_w.size),
            "max_abs_diff": float(abs_d.max()),
            "max_rel_diff": float(rel_d.max()),
            "class": f"fp32 reduction-noise, rel <= {WEIGHTS_CLASS_REL}",
            "agreement": ok5,
        }

        order = [
            ("1_attn_norm0", ok1),
            ("2_q_a_norm0", ok2),
            ("3_k_proj_bf16_boundary", ok3),
            ("4_q_proj", ok4),
            ("5_indexer_weights", ok5),
        ]
        earliest = next((t for t, ok in order if not ok), None)
        for t, ok in order:
            if ok:
                phase0["flags"].append(f"UPSTREAM_AGREEMENT:{t}")
        if not ok3:
            phase0["flags"].append("UPSTREAM_K_PROJ_MISMATCH")
        if not ok4:
            phase0["flags"].append("UPSTREAM_Q_PROJ_MISMATCH")
        if not ok5:
            phase0["flags"].append("UPSTREAM_INDEXER_WEIGHTS_MISMATCH")
        if earliest is not None:
            phase0["flags"].append(f"EARLIEST_DIVERGENCE={earliest}")
        verdict["phase0"] = phase0

        k_branch_ok = ok1 and ok3
        q_branch_ok = ok1 and ok2 and ok4
        weights_ok = ok5
        membership_ok = k_branch_ok and weights_ok
        all_upstream_ok = ok1 and ok2 and ok3 and ok4 and ok5

        def cond(flag_ok: bool, v: str) -> str:
            return v if flag_ok else f"CONDITIONAL: {v}"

        # ================= Blocker 1 =================
        b1: dict = {}
        w_f64 = w_knorm.astype(np.float64)
        x_raw = cpp["lsa_indexer_k_proj_full.bin"]
        cpp_knorm = cpp["lsa_indexer_k_norm_full.bin"]
        hf_knorm = hf["hf_indexer_k_norm.bin"]

        c_raw = model_C_class(x_raw, w_f64)
        d = bf16_ulp_dist(c_raw, cpp_knorm)
        b1["ka_C_raw_vs_cpp_k_norm"] = ulp_hist(d)
        if int(d.max()) > 1:
            stop(
                "blocker-1 known-answer gate: C_raw vs banked C++ k_norm "
                f"max ulp {int(d.max())} > 1 (reduction-noise class exceeded)"
            )

        h_on_hf = model_H_class(hf["hf_indexer_k_proj.bin"], w_f64)
        d = bf16_ulp_dist(h_on_hf, hf_knorm)
        b1["ka_H_bf16_vs_hf_k_norm"] = ulp_hist(d)
        if int(d.max()) > 1:
            stop(
                "blocker-1 known-answer gate: H_bf16 vs HF k_norm max ulp "
                f"{int(d.max())} > 1"
            )

        c_bf16 = model_C_class(rne_bf16(x_raw), w_f64)
        h_bf16 = model_H_class(x_raw, w_f64)
        b1["delta_input_boundary_C_bf16_minus_C_raw"] = ulp_hist(
            bf16_ulp_dist(c_bf16, c_raw)
        )
        b1["delta_cast_order_H_bf16_minus_C_bf16"] = ulp_hist(
            bf16_ulp_dist(h_bf16, c_bf16)
        )
        b1["delta_combined_H_bf16_minus_C_raw"] = {
            "label": cond(k_branch_ok, "combined production delta"),
            "hist": ulp_hist(bf16_ulp_dist(h_bf16, c_raw)),
        }

        # Membership re-scoring rows 2048/2049.
        ggml_cos, ggml_sin, ggml_mscale = ggml_angles()
        b1["cpp_rope_reference_mscale"] = float(ggml_mscale)

        k_C_raw = rotate_cpp_class(c_raw, ggml_cos, ggml_sin)
        d = bf16_ulp_dist(k_C_raw, cpp["lsa_indexer_k_full.bin"])
        b1["ka_post_rope_C_raw_vs_cpp_k"] = ulp_hist(d)
        if int(d.max()) > 1:
            stop(
                "blocker-1 post-RoPE known-answer: reconstructed C_raw K vs "
                f"banked lsa_indexer_k max ulp {int(d.max())} > 1 - "
                "membership reconstruction INVALID"
            )

        k_C_bf16 = rotate_cpp_class(c_bf16, ggml_cos, ggml_sin)
        k_H_bf16 = rotate_cpp_class(h_bf16, ggml_cos, ggml_sin)

        q_all = cpp["lsa_indexer_q_full.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        wts = cpp["lsa_indexer_weights_full.bin"]
        b1["membership"] = {}
        base_sets: dict[int, set[int]] = {}
        base_margins: dict[int, tuple[float, float]] = {}
        for p in (2048, 2049):
            qr = q_all[p].astype(np.float64)
            wr = wts[p].astype(np.float64)
            base_set, margin, fmax = rescore_row(p, qr, wr, k_C_raw)
            banked = set(int(x) for x in cpp_topk[0][p])
            if len(banked) != TOPK or min(banked) < 0:
                stop(f"banked C++ owner00 row {p} malformed")
            if base_set != banked:
                stop(
                    f"blocker-1 baseline gate: C_raw membership row {p} != "
                    f"banked C++ membership (sym diff "
                    f"{len(base_set.symmetric_difference(banked))})"
                )
            base_sets[p] = base_set
            base_margins[p] = (margin, fmax)
            row = {
                "baseline_margin": margin,
                "baseline_max_abs_score": fmax,
                "baseline_reproduces_banked": True,
            }
            v_set, _, _ = rescore_row(p, qr, wr, k_C_bf16)
            row["input_boundary_swap"] = membership_verdict(base_set, v_set, margin, fmax)
            h_set, _, _ = rescore_row(p, qr, wr, k_H_bf16)
            row["combined_swap"] = membership_verdict(base_set, h_set, margin, fmax)
            row["cast_order_isolated"] = membership_verdict(v_set, h_set, margin, fmax)
            for key in ("input_boundary_swap", "combined_swap", "cast_order_isolated"):
                row[key]["verdict"] = cond(membership_ok, row[key]["verdict"])
            b1["membership"][str(p)] = row

        b1["closure_authority"] = (
            "unconditional (K branch + scoring weights Phase-0 gated)"
            if membership_ok
            else "CONDITIONAL (Phase-0 flag on the K branch and/or the "
            "scoring weights): no production K-norm closure is claimed"
        )
        verdict["blocker1"] = b1

        # ================= Blocker 2 =================
        b2: dict = {}
        hf_k = hf["hf_indexer_k.bin"]
        cpp_k = cpp["lsa_indexer_k_full.bin"]

        # Per-side K nope identities (hard: layout model wrong on failure).
        b2["hf_k_nope_identity"] = bool(
            bits_equal(hf_k[:, ROPE_DIM:], hf_knorm[:, ROPE_DIM:])
        )
        if not b2["hf_k_nope_identity"]:
            stop("blocker-2: HF K nope-placement identity failed "
                 "(hf_k[:,64:] != hf_k_norm[:,64:]) - layout model wrong")
        b2["cpp_k_nope_identity_2050"] = bool(
            bits_equal(cpp_k[:, ROPE_DIM:], cpp_knorm[:, ROPE_DIM:])
        )
        if not b2["cpp_k_nope_identity_2050"]:
            stop("blocker-2: C++ 2050 K nope identity failed "
                 "(cpp_k[:,64:] != cpp_k_norm[:,64:])")

        # Blocker-1 decontamination: the cross-side K mapping is only
        # unconditional when the pre-RoPE K-norm surfaces agree bitwise.
        pre_k_equal = bits_equal(cpp_knorm, hf_knorm)
        b2["pre_rope_k_norm_bitwise_equal"] = bool(pre_k_equal)
        if not pre_k_equal:
            b2["pre_rope_k_norm_ulp_hist"] = ulp_hist(
                bf16_ulp_dist(cpp_knorm, hf_knorm)
            )
        b2["k_nope_cross_side_equal"] = bool(
            bits_equal(cpp_k[:, ROPE_DIM:], hf_k[:, ROPE_DIM:])
        )

        def pair_decode(pre, post, hypothesis):
            a = pre[:, 0:ROPE_DIM:2].astype(np.float64)
            b_ = pre[:, 1:ROPE_DIM:2].astype(np.float64)
            if hypothesis == "interleaved":
                u = post[:, 0:ROPE_DIM:2].astype(np.float64)
                v = post[:, 1:ROPE_DIM:2].astype(np.float64)
            else:  # half-split output
                u = post[:, : ROPE_DIM // 2].astype(np.float64)
                v = post[:, ROPE_DIM // 2:].astype(np.float64)
            det = a * a + b_ * b_
            good = det > np.quantile(det, 0.05)
            c = np.where(good, (a * u + b_ * v) / np.where(det == 0, 1, det), np.nan)
            s = np.where(good, (a * v - b_ * u) / np.where(det == 0, 1, det), np.nan)
            r = c * c + s * s
            rv = r[np.isfinite(r)]
            return c, s, {
                "median_r": float(np.median(rv)),
                "frac_r_in_0p9_1p1": float(np.mean((rv > 0.81) & (rv < 1.21))),
                "iqr_r": float(np.quantile(rv, 0.75) - np.quantile(rv, 0.25)),
            }

        decode: dict = {}
        for side, pre, post in (
            ("cpp", cpp_knorm[:, :ROPE_DIM], cpp_k[:, :ROPE_DIM]),
            ("hf", hf_knorm[:, :ROPE_DIM], hf_k[:, :ROPE_DIM]),
        ):
            decode[side] = {}
            for hyp in ("interleaved", "half-split"):
                c, s, stats = pair_decode(pre, post, hyp)
                decode[side][hyp] = stats
                decode[side][hyp + "_cs"] = (c, s)
        b2["pair_decode"] = {
            side: {h: decode[side][h] for h in ("interleaved", "half-split")}
            for side in ("cpp", "hf")
        }
        for side, expect in (("cpp", "interleaved"), ("hf", "half-split")):
            stats = decode[side][expect]
            alt = decode[side]["half-split" if expect == "interleaved" else "interleaved"]
            ok = stats["frac_r_in_0p9_1p1"] > 0.99 and stats["iqr_r"] < 0.05
            b2.setdefault("hypothesis_accept", {})[side] = {
                "expected": expect,
                "accepted": bool(ok),
                "expected_stats": stats,
                "alternative_stats": alt,
            }
            if not ok:
                stop(f"blocker-2: pair-decode did not accept the expected "
                     f"{expect!r} hypothesis on the {side} side")

        def mapping_entry(cpp_roped, hf_roped):
            as_is = bits_equal(cpp_roped, hf_roped)
            mapped = pi_map_blocks(cpp_roped)
            under_pi = bits_equal(mapped, hf_roped)
            d = bf16_ulp_dist(mapped, hf_roped)
            return as_is, under_pi, d

        def pi_map_blocks(block: np.ndarray) -> np.ndarray:
            if block.shape[1] == ROPE_DIM:
                return pi_map(block)
            b3d = block.reshape(N_TOKENS, N_HEADS, ROPE_DIM)
            out = np.empty_like(b3d)
            out[:, :, : ROPE_DIM // 2] = b3d[:, :, 0:ROPE_DIM:2]
            out[:, :, ROPE_DIM // 2:] = b3d[:, :, 1:ROPE_DIM:2]
            return out.reshape(N_TOKENS, N_HEADS * ROPE_DIM)

        def base_layout_verdict(as_is, under_pi, d):
            if as_is:
                return "EQUAL AS-IS"
            if under_pi:
                return "EQUAL UNDER PI (pure layout)"
            if int(d.max()) <= ROUNDING_CLASS_MAX_ULP:
                return (f"ROUNDING-CLASS RESIDUAL under PI (max ulp "
                        f"{int(d.max())} <= {ROUNDING_CLASS_MAX_ULP})")
            return "NUMERIC DISAGREEMENT (beyond all hypotheses)"

        # K mapping (blocker-1-decontaminated authority).
        as_is, under_pi, d = mapping_entry(cpp_k[:, :ROPE_DIM], hf_k[:, :ROPE_DIM])
        kv = base_layout_verdict(as_is, under_pi, d)
        k_entry = {
            "equal_as_is": bool(as_is),
            "equal_under_pi": bool(under_pi),
            "ulp_hist_under_pi": ulp_hist(d),
        }
        if k_branch_ok and pre_k_equal:
            k_entry["verdict"] = kv
        elif k_branch_ok:
            k_entry["verdict"] = (
                "CONDITIONAL on blocker-1 (pre-RoPE K-norm surfaces differ; "
                f"residual includes norm-class deltas): {kv}"
            )
            k_entry["note"] = (
                "K-side layout adjudication under differing pre-RoPE inputs "
                "rests on per-side pair-decode hypotheses, the per/cross "
                "nope identities, and the source proof - not on this "
                "cross-side residual"
            )
        else:
            k_entry["verdict"] = f"CONDITIONAL (Phase-0 K-branch mismatch): {kv}"
        b2["k_mapping"] = k_entry

        # Q per-side no-RoPE-half invariants + cross-side slice check.
        cpp_q3 = cpp["lsa_indexer_q_full.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        hf_q3 = hf["hf_indexer_q.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        cpp_qp3 = cpp["lsa_indexer_q_proj_full.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        hf_qp3 = hf["hf_indexer_q_proj.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        cpp_q_nope_inv = bits_equal(cpp_q3[:, :, ROPE_DIM:], cpp_qp3[:, :, ROPE_DIM:])
        hf_q_nope_inv = bits_equal(hf_q3[:, :, ROPE_DIM:], hf_qp3[:, :, ROPE_DIM:])
        b2["cpp_q_no_rope_half_invariant"] = bool(cpp_q_nope_inv)
        b2["hf_q_no_rope_half_invariant"] = bool(hf_q_nope_inv)
        q_nope_cross = bits_equal(cpp_q3[:, :, ROPE_DIM:], hf_q3[:, :, ROPE_DIM:])
        b2["q_nope_cross_side_equal"] = bool(q_nope_cross)

        slice_anomalies: list[str] = []
        if not cpp_q_nope_inv:
            slice_anomalies.append(
                "C++ post-RoPE Q dims 64..127 != its own pre-RoPE dims 64..127"
            )
        if not hf_q_nope_inv:
            slice_anomalies.append(
                "HF post-RoPE Q dims 64..127 != its own pre-RoPE dims 64..127"
            )
        if ok4 and not q_nope_cross:
            slice_anomalies.append(
                "cross-side Q-nope inequality under Phase-0 Q-projection "
                "agreement"
            )

        as_is, under_pi, d = mapping_entry(
            cpp_q3[:, :, :ROPE_DIM].reshape(N_TOKENS, N_HEADS * ROPE_DIM),
            hf_q3[:, :, :ROPE_DIM].reshape(N_TOKENS, N_HEADS * ROPE_DIM),
        )
        q_entry = {
            "equal_as_is": bool(as_is),
            "equal_under_pi": bool(under_pi),
            "ulp_hist_under_pi": ulp_hist(d),
        }
        if slice_anomalies:
            q_entry["verdict"] = (
                "ROPE-SLICE MISMATCH (different columns rotated - "
                "score-relevant): " + "; ".join(slice_anomalies)
            )
            q_entry["note"] = (
                "a layout-equivalent verdict is forbidden while a slice "
                "anomaly stands; the pi comparison below is "
                "characterization only"
            )
        else:
            q_entry["verdict"] = cond(q_branch_ok, base_layout_verdict(as_is, under_pi, d))
        b2["q_mapping"] = q_entry
        b2["interpretation_rule"] = (
            "a common permutation applied consistently to q and k is "
            "dot-product-invariant; only ROPE-SLICE MISMATCH or NUMERIC "
            "DISAGREEMENT can affect scores"
        )
        verdict["blocker2"] = b2

        # ================= Blocker 3 =================
        b3: dict = {}
        hf_cos_ref, hf_sin_ref, hf_scaling = hf_angles()
        b3["hf_attention_scaling"] = hf_scaling
        b3["cpp_effective_mscale_f32chain"] = float(ggml_mscale)

        # Effective-mscale extraction: K AND Q, per side.
        b3["effective_mscale_measured"] = {
            "cpp_k": pair_norm_ratio(cpp_knorm[:, :ROPE_DIM], cpp_k[:, :ROPE_DIM], "interleaved"),
            "hf_k": pair_norm_ratio(hf_knorm[:, :ROPE_DIM], hf_k[:, :ROPE_DIM], "half-split"),
            "cpp_q": pair_norm_ratio(
                cpp_qp3[:, :, :ROPE_DIM].reshape(-1, ROPE_DIM),
                cpp_q3[:, :, :ROPE_DIM].reshape(-1, ROPE_DIM),
                "interleaved",
            ),
            "hf_q": pair_norm_ratio(
                hf_qp3[:, :, :ROPE_DIM].reshape(-1, ROPE_DIM),
                hf_q3[:, :, :ROPE_DIM].reshape(-1, ROPE_DIM),
                "half-split",
            ),
        }
        yarn_big = 1.0 + 0.1 * math.log(YARN_FACTOR)

        def scale_class(m: float) -> str:
            if abs(m - 1.0) < 0.02:
                return "1.0"
            if abs(m - yarn_big) < 0.02:
                return f"{yarn_big:.6f} (1+0.1*ln(120))"
            return f"OTHER ({m:.6f})"

        sc = {
            key: scale_class(b3["effective_mscale_measured"][key]["median"])
            for key in ("cpp_k", "hf_k", "cpp_q", "hf_q")
        }
        scale_match = len(set(sc.values())) == 1
        b3["scale_verdict"] = {
            "per_surface": sc,
            "verdict": cond(
                k_branch_ok and q_branch_ok,
                "SCALE MATCH (all four surfaces in class "
                f"{next(iter(set(sc.values())))})" if scale_match
                else f"SCALE MISMATCH ({sc})",
            ),
        }

        # SCHEDULE sub-verdict (explicit; never overclaims formula
        # discrimination below BF16 resolution).
        def schedule_residual(side, cs_key, cos_ref, sin_ref):
            c, s = decode[side][cs_key]
            return max(
                float(np.nanmax(np.abs(c - cos_ref.astype(np.float64)))),
                float(np.nanmax(np.abs(s - sin_ref.astype(np.float64)))),
            )

        ref_sep = max(
            float(np.max(np.abs(ggml_cos - hf_cos_ref))),
            float(np.max(np.abs(ggml_sin - hf_sin_ref))),
        )
        b3["reference_separation_max_abs"] = ref_sep
        sched = {}
        for side, cs_key in (("cpp", "interleaved_cs"), ("hf", "half-split_cs")):
            res_g = schedule_residual(side, cs_key, ggml_cos, ggml_sin)
            res_h = schedule_residual(side, cs_key, hf_cos_ref, hf_sin_ref)
            fits_g = res_g <= SCHEDULE_FIT_MAX_RESID
            fits_h = res_h <= SCHEDULE_FIT_MAX_RESID
            if fits_g and fits_h and ref_sep < SCHEDULE_REF_SEPARATION:
                v = ("SCHEDULE MATCH - ggml-class and HF-class references "
                     "NOT EMPIRICALLY DISTINGUISHABLE at captured BF16 "
                     f"precision (reference separation {ref_sep:.2e})")
            elif fits_g and fits_h:
                v = "SCHEDULE MATCH (both references fit)"
            elif fits_g:
                v = "SCHEDULE MATCH ggml-class reference only"
            elif fits_h:
                v = "SCHEDULE MATCH HF-class reference only"
            else:
                v = (f"SCHEDULE MISMATCH (neither reference fits; residuals "
                     f"ggml {res_g:.3e} / hf {res_h:.3e})")
            sched[side] = {
                "max_resid_vs_ggml_ref": res_g,
                "max_resid_vs_hf_ref": res_h,
                "verdict": v,
            }
        b3["schedule_verdict"] = sched

        # HF captured cos/sin (bf16 lattice, 64-wide duplicated halves).
        hf_cos_cap = hf["hf_rope_cos.bin"]
        hf_sin_cap = hf["hf_rope_sin.bin"]
        dup_ok = bool(
            bits_equal(hf_cos_cap[:, :32], hf_cos_cap[:, 32:])
            and bits_equal(hf_sin_cap[:, :32], hf_sin_cap[:, 32:])
        )
        b3["hf_cos_sin_halves_duplicated"] = dup_ok
        if not dup_ok:
            stop("blocker-3: HF cos/sin halves are not duplicated "
                 "(emb = cat(freqs, freqs) expectation failed)")
        cchk = {
            "cos_vs_hf_ref_bf16": ulp_hist(
                bf16_ulp_dist(hf_cos_cap[:, :32], rne_bf16(hf_cos_ref))
            ),
            "sin_vs_hf_ref_bf16": ulp_hist(
                bf16_ulp_dist(hf_sin_cap[:, :32], rne_bf16(hf_sin_ref))
            ),
            "cos_vs_ggml_ref_bf16": ulp_hist(
                bf16_ulp_dist(hf_cos_cap[:, :32], rne_bf16(ggml_cos))
            ),
            "sin_vs_ggml_ref_bf16": ulp_hist(
                bf16_ulp_dist(hf_sin_cap[:, :32], rne_bf16(ggml_sin))
            ),
        }
        b3["hf_captured_cos_sin"] = cchk

        # ROUNDING-CLASS sub-verdict (elementwise-chain magnitude summary).
        b3["rounding_class_verdict"] = {
            "k_under_pi_max_ulp": b2["k_mapping"]["ulp_hist_under_pi"]["max"],
            "q_under_pi_max_ulp": b2["q_mapping"]["ulp_hist_under_pi"]["max"],
            "hf_cos_vs_best_ref_max_ulp": min(
                cchk["cos_vs_hf_ref_bf16"]["max"],
                cchk["cos_vs_ggml_ref_bf16"]["max"],
            ),
            "hf_sin_vs_best_ref_max_ulp": min(
                cchk["sin_vs_hf_ref_bf16"]["max"],
                cchk["sin_vs_ggml_ref_bf16"]["max"],
            ),
            "verdict": (
                "ROUNDING-CLASS: BF16-elementwise-vs-F32-chain effects; "
                "max observed "
                f"{max(b2['k_mapping']['ulp_hist_under_pi']['max'], b2['q_mapping']['ulp_hist_under_pi']['max'])} "
                "bf16 ulp on mapped roped surfaces "
                f"(rounding-class bound {ROUNDING_CLASS_MAX_ULP})"
            ),
        }

        # Membership impact of the angle set (rows 2048/2049).
        k_angle_swap = rotate_cpp_class(c_raw, rne_bf16(hf_cos_ref), rne_bf16(hf_sin_ref))
        b3["membership_angle_swap"] = {}
        for p in (2048, 2049):
            qr = q_all[p].astype(np.float64)
            wr = wts[p].astype(np.float64)
            margin, fmax = base_margins[p]
            a_set, _, _ = rescore_row(p, qr, wr, k_angle_swap)
            mv = membership_verdict(base_sets[p], a_set, margin, fmax)
            mv["verdict"] = cond(membership_ok, mv["verdict"])
            b3["membership_angle_swap"][str(p)] = mv
        b3["closure_authority"] = (
            "per-side extractions unconditional; cross-side conclusions "
            + ("unconditional" if (k_branch_ok and q_branch_ok) else
               "CONDITIONAL (Phase-0 upstream mismatch)")
            + "; membership impact "
            + ("unconditional" if membership_ok else
               "CONDITIONAL (K branch and/or scoring weights)")
        )
        verdict["blocker3"] = b3

        # ============ Section 11: membership sets (scoped) ============
        s11: dict = {"owners": {}}
        for li in range(14):
            owner = 2 * li
            entry: dict = {}
            for p in (2048, 2049):
                crow = cpp_topk[owner][p]
                if crow.min() < 0:
                    stop(f"C++ owner{owner:02d} row {p} contains negatives")
                cset = set(int(x) for x in crow)
                hrow = hf_topk[owner][p]
                if (hrow == -1).any():
                    stop(f"HF owner{owner:02d} row {p} contains -1 fillers")
                hset = set(int(x) for x in hrow)
                if len(cset) != TOPK or len(hset) != TOPK:
                    stop(f"owner{owner:02d} row {p}: set sizes "
                         f"{len(cset)}/{len(hset)} != {TOPK}")
                diff = cset.symmetric_difference(hset)
                entry[str(p)] = {
                    "equal": not diff,
                    "sym_diff_size": len(diff),
                    "sym_diff": sorted(diff)[:64],
                }
            if owner == 0:
                entry["scope"] = (
                    "attribution-eligible (owner00 upstream surfaces "
                    "captured and Phase-0 gated)"
                )
                entry["attribution"] = (
                    "attributable to blockers 1-3"
                    if all_upstream_ok
                    else "CONDITIONAL - NOT uniquely attributable "
                    "(upstream dependency flag set; see phase0)"
                )
            else:
                entry["scope"] = (
                    "cross-owner semantic observation ONLY (no upstream "
                    "captures exist for this owner; differences are not "
                    "claimed to be uniquely attributable to first-owner "
                    "mechanisms)"
                )
            s11["owners"][f"owner{owner:02d}"] = entry
        verdict["section11_membership"] = s11

        verdict["anomaly"] = None
        return finish(0)

    except SystemExit as exc:
        verdict["anomaly"] = str(exc)
        return finish(1)


if __name__ == "__main__":
    raise SystemExit(main())
