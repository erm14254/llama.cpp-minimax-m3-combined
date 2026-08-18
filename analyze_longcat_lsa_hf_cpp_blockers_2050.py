#!/usr/bin/env python3
"""Offline C++<->HF blocker-1/2/3 comparator for the 2050 first-owner bank.

CPU + numpy + stdlib only. No GPU, no model, no torch. Consumes the
byte-identical C++ S-family capture (cpp_lsa_2050_S1) and the HF Run A
capture (hf_lsa_2050_capture); emits lsa_hf_blockers_2050/verdict.json.

Pre-registered semantics (authored and committed BEFORE any HF
execution; the comparator implements them and never re-decides them):

  Phase 0 (comparator-wide upstream attribution guard; guards all
  blocker analyses; runs first): compare, in causal order,
    (1) attn_norm-0        C++ vs HF        (BF16-lattice byte equality;
        the C++ carrier is F32 and its lattice membership is itself
        part of the expectation),
    (2) q_a_norm-0         C++ vs HF        (byte equality),
    (3) rne_bf16(C++ k_proj raw F32) vs HF wk output (byte equality at
        the BF16 output boundary),
    (4) C++ q_proj (BF16)  vs HF wq_b output (byte equality).
  Disagreements are reported as explicit numerical classes (counts +
  bf16-ulp histograms), never silently tolerated; no frozen project
  criterion is widened. Dependency flags: UPSTREAM_K_PROJ_MISMATCH,
  UPSTREAM_Q_PROJ_MISMATCH, EARLIEST_DIVERGENCE. Under a flag the
  affected blocker emits CONDITIONAL characterization only and cannot
  be discharged.

  Blocker 1 (K-norm cast ordering): three models on the banked C++ raw
  F32 GEMM output x_raw --
    C_raw  = bf16( f32rms(x_raw; eps 1e-6) * w )          (C++ class)
    C_bf16 = bf16( f32rms(rne_bf16(x_raw); eps 1e-6) * w )
    H_bf16 = bf16mul( w, bf16( f32(rne(x)) * rsqrt(mean(f32(rne(x))^2)+eps) ) )
  Known-answer gates: C_raw reproduces the banked C++ k_norm bytes
  (<=1 bf16 ulp reduction-noise class; >1 ulp aborts); H_bf16 applied
  to HF's own k_proj reproduces the HF k_norm bytes (byte-exact
  expected; 1-ulp near-ties tolerated and reported; >1 ulp aborts).
  Reported deltas: C_bf16-C_raw (input boundary), H_bf16-C_bf16 (pure
  cast order), H_bf16-C_raw (combined). Membership re-scoring rows
  2048/2049 gates IN ORDER: (i) the C_raw-reconstructed K propagated
  through the C++-class RoPE reference must match the banked
  lsa_indexer_k within <=1 bf16 ulp everywhere (else the membership
  reconstruction is INVALID and stops); (ii) the C_raw baseline must
  reproduce the banked C++ top-K membership. Then C_bf16 / H_bf16
  swaps. Verdicts: MEMBERSHIP-INVARIANT / MEMBERSHIP-AFFECTING(n) /
  BORDERLINE (boundary margin below the pre-registered noise floor
  margin_floor = 1e-4 * max|finite row score|).

  Blocker 2 (rope/nope layout): HF nope-placement identity
  (hf_k[:,64:] == hf_k_norm[:,64:]), 2050-scale C++ nope identity
  re-proof, pair-decode consistency under interleaved vs half-split
  hypotheses, and the cross-side candidate permutation
  pi: HF[i] = C++[2i], HF[32+i] = C++[2i+1] on each roped 64-block.
  Verdict vocabulary: EQUAL AS-IS / EQUAL UNDER PI (pure layout) /
  ROUNDING-CLASS RESIDUAL under pi (all |ulp| <= 4; the known
  BF16-elementwise-vs-F32-chain class) / ROPE-SLICE MISMATCH /
  NUMERIC DISAGREEMENT. A common permutation applied consistently to
  q and k is dot-product-invariant.

  Blocker 3 (YaRN attn_factor/mscale): per-side effective-mscale
  extraction from pre/post pair norms; angle-schedule fit of the
  decoded rotations against BOTH float-chain references (ggml-class:
  theta-space blend, f32 chain replicating rope.cu + llama-context;
  HF-class: inv-freq-space blend, f32 chain replicating
  modeling_rope_utils + bf16-at-use cast); HF captured cos/sin checked
  against both references; membership impact of an angle-set swap.
  Verdicts SCALE / SCHEDULE / ROUNDING-CLASS / membership impact are
  reported separately and never collapsed.

  Section 11: exact top-K membership set comparison, rows 2048/2049,
  all 14 owners, position space [0, 2050). Attribution to blockers 1-3
  only under the relevant UPSTREAM_AGREEMENT.

Exit 0 = analysis completed with every known-answer/validity gate
passed (dependency flags and CONDITIONAL labels are verdict content).
Nonzero = analysis aborted (reasons listed in verdict.json). This
comparator makes NO Gate-4 statement; Gate 4 remains NOT RUN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

# Frozen byte-stable C++ S-family surfaces (V-input table, verdict.json of
# the determinism round; byte-identical across S1/S2/S3).
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


# ---------------- bf16 helpers ----------------

def _u32(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="<f4")
    return x.view(np.uint32)


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
    """Monotone total-order key: negatives below positives, -0 == +0."""
    b = b.astype(np.int64)
    b = np.where(b == 0x8000, 0, b)  # -0 -> +0
    return np.where(b >= 0x8000, 0xFFFF - b, b + 0x8000)


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
    q_row_f64: np.ndarray,      # [16, 128]
    weights_row_f64: np.ndarray,  # [16] (C++ post-scale weights)
    k_all_f32: np.ndarray,      # [2050, 128] variant post-rope K (bf16 lattice)
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
        verdict = "MEMBERSHIP-INVARIANT"
        if margin < floor:
            verdict = "BORDERLINE"
    else:
        verdict = f"MEMBERSHIP-AFFECTING ({len(diff)} positions)"
        if margin < floor:
            verdict += " [boundary margin below noise floor]"
    return {
        "verdict": verdict,
        "sym_diff_size": len(diff),
        "sym_diff": sorted(diff)[:64],
        "boundary_margin": margin,
        "margin_floor": floor,
    }


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-s-dir", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args()

    cpp_dir = Path(ns.cpp_s_dir).resolve()
    hf_dir = Path(ns.hf_dir).resolve()
    out_dir = Path(ns.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    verdict: dict = {
        "protocol": (
            "HF 2050 first-owner blockers 1-3 offline comparator "
            "(pre-registered; Phase 0 attribution guard first; "
            "NOT a Gate-4 criterion; Gate 4 remains NOT RUN)"
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
        # ---- integrity: C++ stable-surface SHA gates ----
        cpp: dict[str, np.ndarray] = {}
        cpp_shas = {}
        for name, expected in CPP_STABLE_SHA256.items():
            p = cpp_dir / name
            if not p.is_file():
                stop(f"C++ surface missing: {p}")
            got = sha256_file(p)
            cpp_shas[name] = got
            if got != expected:
                stop(f"C++ surface {name} SHA {got} != frozen {expected}")
            cpp[name] = load_f32(p, CPP_WIDTHS[name])
        cpp_topk = {}
        for li in range(14):
            name = f"lsa_top_k_owner{2 * li:02d}_full.bin"
            p = cpp_dir / name
            if not p.is_file():
                stop(f"C++ top-K missing: {p}")
            arr = load_f32(p, TOPK)
            ints = arr.astype(np.int64)
            if not np.array_equal(ints.astype("<f4"), arr):
                stop(f"{name}: values not exactly integral")
            cpp_topk[2 * li] = ints
        verdict["cpp_stable_sha256"] = cpp_shas

        # ---- integrity: HF surfaces ----
        hf: dict[str, np.ndarray] = {}
        hf_shas = {}
        for name, width in HF_WIDTHS.items():
            p = hf_dir / name
            if not p.is_file():
                stop(f"HF surface missing: {p}")
            hf_shas[name] = sha256_file(p)
            hf[name] = load_f32(p, width)
        hf_topk = {}
        for li in range(14):
            name = f"hf_top_k_owner{2 * li:02d}.bin"
            p = hf_dir / name
            if not p.is_file():
                stop(f"HF top-K missing: {p}")
            arr = load_f32(p, TOPK)
            ints = arr.astype(np.int64)
            if not np.array_equal(ints.astype("<f4"), arr):
                stop(f"{name}: values not exactly integral")
            hf_topk[2 * li] = ints
        wpath = hf_dir / "hf_weight_k_norm.bin"
        if not wpath.is_file():
            stop(f"HF k_norm weight missing: {wpath}")
        w_knorm = np.frombuffer(wpath.read_bytes(), dtype="<f4")
        if w_knorm.size != HEAD_DIM:
            stop(f"hf_weight_k_norm size {w_knorm.size} != {HEAD_DIM}")
        if not np.array_equal(rne_bf16(w_knorm), w_knorm):
            stop("hf_weight_k_norm is not on the BF16 lattice")
        verdict["hf_sha256"] = hf_shas

        # For every HF activation surface the source is BF16 except the
        # fp32 gates; verify lattice membership where BF16 is claimed.
        for name in (
            "hf_q_a_norm0.bin",
            "hf_indexer_k_proj.bin",
            "hf_indexer_k_norm.bin",
            "hf_indexer_k.bin",
            "hf_indexer_q_proj.bin",
            "hf_indexer_q.bin",
            "hf_rope_cos.bin",
            "hf_rope_sin.bin",
            "hf_attn_norm0.bin",
        ):
            v = hf[name]
            bad = int((~(rne_bf16(v) == v)).sum())
            if bad:
                stop(f"{name}: {bad} values off the BF16 lattice")

        # ================= Phase 0 =================
        phase0: dict = {"surfaces": {}, "flags": []}

        def compare_surface(tag, a_f32, b_f32, lattice_violations=0):
            eq = np.array_equal(a_f32, b_f32) and lattice_violations == 0
            entry = {
                "byte_equal": bool(np.array_equal(a_f32, b_f32)),
                "cpp_lattice_violations": int(lattice_violations),
                "agreement": bool(eq),
            }
            if not entry["byte_equal"]:
                d = bf16_ulp_dist(a_f32, b_f32)
                entry["n_diff"] = int((d > 0).sum())
                entry["ulp_hist"] = ulp_hist(d)
            phase0["surfaces"][tag] = entry
            return eq

        cpp_an0 = cpp["lsa_anchor_attn_norm0_full.bin"]
        lat_an0 = int((~(rne_bf16(cpp_an0) == cpp_an0)).sum())
        ok1 = compare_surface("1_attn_norm0", cpp_an0, hf["hf_attn_norm0.bin"], lat_an0)
        ok2 = compare_surface(
            "2_q_a_norm0",
            cpp["lsa_anchor_q_a_norm0_full.bin"],
            hf["hf_q_a_norm0.bin"],
        )
        ok3 = compare_surface(
            "3_k_proj_bf16_boundary",
            rne_bf16(cpp["lsa_indexer_k_proj_full.bin"]),
            hf["hf_indexer_k_proj.bin"],
        )
        ok4 = compare_surface(
            "4_q_proj",
            cpp["lsa_indexer_q_proj_full.bin"],
            hf["hf_indexer_q_proj.bin"],
        )

        order = [("1_attn_norm0", ok1), ("2_q_a_norm0", ok2),
                 ("3_k_proj_bf16_boundary", ok3), ("4_q_proj", ok4)]
        earliest = next((t for t, ok in order if not ok), None)
        for t, ok in order:
            if ok:
                phase0["flags"].append(f"UPSTREAM_AGREEMENT:{t}")
        if not ok3:
            phase0["flags"].append("UPSTREAM_K_PROJ_MISMATCH")
        if not ok4:
            phase0["flags"].append("UPSTREAM_Q_PROJ_MISMATCH")
        if earliest is not None:
            phase0["flags"].append(f"EARLIEST_DIVERGENCE={earliest}")
        verdict["phase0"] = phase0

        k_branch_ok = ok1 and ok3
        q_branch_ok = ok1 and ok2 and ok4
        all_upstream_ok = ok1 and ok2 and ok3 and ok4

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
        combined = ulp_hist(bf16_ulp_dist(h_bf16, c_raw))
        b1["delta_combined_H_bf16_minus_C_raw"] = {
            "label": cond(k_branch_ok, "combined production delta"),
            "hist": combined,
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
            row = {
                "baseline_margin": margin,
                "baseline_max_abs_score": fmax,
                "baseline_reproduces_banked": True,
            }
            v_set, v_margin, _ = rescore_row(p, qr, wr, k_C_bf16)
            row["input_boundary_swap"] = membership_verdict(
                base_set, v_set, margin, fmax
            )
            h_set, h_margin, _ = rescore_row(p, qr, wr, k_H_bf16)
            row["combined_swap"] = membership_verdict(base_set, h_set, margin, fmax)
            row["cast_order_isolated"] = membership_verdict(v_set, h_set, margin, fmax)
            for key in ("input_boundary_swap", "combined_swap", "cast_order_isolated"):
                row[key]["verdict"] = cond(k_branch_ok, row[key]["verdict"])
            b1["membership"][str(p)] = row

        b1["closure_authority"] = (
            "unconditional" if k_branch_ok else
            "CONDITIONAL (UPSTREAM_K_PROJ_MISMATCH or anchor divergence): "
            "no production K-norm closure is claimed"
        )
        verdict["blocker1"] = b1

        # ================= Blocker 2 =================
        b2: dict = {}
        hf_k = hf["hf_indexer_k.bin"]
        cpp_k = cpp["lsa_indexer_k_full.bin"]
        b2["hf_nope_identity"] = bool(
            np.array_equal(hf_k[:, ROPE_DIM:], hf_knorm[:, ROPE_DIM:])
        )
        if not b2["hf_nope_identity"]:
            stop("blocker-2: HF nope-placement identity failed "
                 "(hf_k[:,64:] != hf_k_norm[:,64:]) - layout model wrong")
        b2["cpp_nope_identity_2050"] = bool(
            np.array_equal(cpp_k[:, ROPE_DIM:], cpp_knorm[:, ROPE_DIM:])
        )
        if not b2["cpp_nope_identity_2050"]:
            stop("blocker-2: C++ 2050 nope identity failed "
                 "(cpp_k[:,64:] != cpp_k_norm[:,64:])")

        def pair_decode(pre, post, hypothesis):
            if hypothesis == "interleaved":
                a, b_ = pre[:, 0:ROPE_DIM:2], pre[:, 1:ROPE_DIM:2]
                u, v = post[:, 0:ROPE_DIM:2], post[:, 1:ROPE_DIM:2]
            else:  # half-split output
                a, b_ = pre[:, 0:ROPE_DIM:2], pre[:, 1:ROPE_DIM:2]
                u, v = post[:, : ROPE_DIM // 2], post[:, ROPE_DIM // 2:]
            a = a.astype(np.float64)
            b_ = b_.astype(np.float64)
            u = u.astype(np.float64)
            v = v.astype(np.float64)
            det = a * a + b_ * b_
            good = det > np.quantile(det, 0.05)
            c = np.where(good, (a * u + b_ * v) / np.where(det == 0, 1, det), np.nan)
            s = np.where(good, (a * v - b_ * u) / np.where(det == 0, 1, det), np.nan)
            r = c * c + s * s
            rv = r[np.isfinite(r)]
            return c, s, {
                "median_r": float(np.median(rv)),
                "frac_r_in_0p9_1p1": float(np.mean((rv > 0.81) & (rv < 1.21))),
                "iqr_r": float(
                    np.quantile(rv, 0.75) - np.quantile(rv, 0.25)
                ),
            }

        decode = {}
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
        accepted = {}
        for side, expect in (("cpp", "interleaved"), ("hf", "half-split")):
            stats = decode[side][expect]
            other = decode[side]["half-split" if expect == "interleaved" else "interleaved"]
            ok = stats["frac_r_in_0p9_1p1"] > 0.99 and stats["iqr_r"] < 0.05
            accepted[side] = expect if ok else None
            b2.setdefault("hypothesis_accept", {})[side] = {
                "expected": expect,
                "accepted": bool(ok),
                "expected_stats": stats,
                "alternative_stats": other,
            }
        if accepted["cpp"] is None or accepted["hf"] is None:
            stop("blocker-2: pair-decode did not accept the expected pairing "
                 "hypothesis on at least one side")

        def pi_map(cpp_block: np.ndarray) -> np.ndarray:
            out = np.empty_like(cpp_block)
            out[:, : ROPE_DIM // 2] = cpp_block[:, 0:ROPE_DIM:2]
            out[:, ROPE_DIM // 2:] = cpp_block[:, 1:ROPE_DIM:2]
            return out

        def layout_verdict(tag, cpp_roped, hf_roped, branch_ok):
            as_is = np.array_equal(cpp_roped, hf_roped)
            mapped = pi_map(cpp_roped)
            under_pi = np.array_equal(mapped, hf_roped)
            d = bf16_ulp_dist(mapped, hf_roped)
            entry = {"equal_as_is": bool(as_is), "equal_under_pi": bool(under_pi),
                     "ulp_hist_under_pi": ulp_hist(d)}
            if as_is:
                v = "EQUAL AS-IS"
            elif under_pi:
                v = "EQUAL UNDER PI (pure layout)"
            elif int(d.max()) <= ROUNDING_CLASS_MAX_ULP:
                v = (f"ROUNDING-CLASS RESIDUAL under PI (max ulp "
                     f"{int(d.max())} <= {ROUNDING_CLASS_MAX_ULP})")
            else:
                v = "NUMERIC DISAGREEMENT (beyond all hypotheses)"
            entry["verdict"] = cond(branch_ok, v)
            b2[tag] = entry

        layout_verdict("k_mapping", cpp_k[:, :ROPE_DIM], hf_k[:, :ROPE_DIM], k_branch_ok)

        cpp_q = cpp["lsa_indexer_q_full.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        hf_q = hf["hf_indexer_q.bin"].reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        q_nope_equal = bool(
            np.array_equal(cpp_q[:, :, ROPE_DIM:], hf_q[:, :, ROPE_DIM:])
        )
        b2["q_nope_cross_side_equal"] = q_nope_equal
        cq = cpp_q[:, :, :ROPE_DIM].reshape(N_TOKENS, N_HEADS * ROPE_DIM)
        hq = hf_q[:, :, :ROPE_DIM].reshape(N_TOKENS, N_HEADS * ROPE_DIM)

        def pi_map_heads(block):
            b3 = block.reshape(N_TOKENS, N_HEADS, ROPE_DIM)
            out = np.empty_like(b3)
            out[:, :, : ROPE_DIM // 2] = b3[:, :, 0:ROPE_DIM:2]
            out[:, :, ROPE_DIM // 2:] = b3[:, :, 1:ROPE_DIM:2]
            return out.reshape(N_TOKENS, N_HEADS * ROPE_DIM)

        as_is = np.array_equal(cq, hq)
        mapped = pi_map_heads(cq)
        under_pi = np.array_equal(mapped, hq)
        d = bf16_ulp_dist(mapped, hq)
        entry = {"equal_as_is": bool(as_is), "equal_under_pi": bool(under_pi),
                 "ulp_hist_under_pi": ulp_hist(d)}
        if as_is:
            v = "EQUAL AS-IS"
        elif under_pi:
            v = "EQUAL UNDER PI (pure layout)"
        elif int(d.max()) <= ROUNDING_CLASS_MAX_ULP:
            v = (f"ROUNDING-CLASS RESIDUAL under PI (max ulp "
                 f"{int(d.max())} <= {ROUNDING_CLASS_MAX_ULP})")
        else:
            v = "NUMERIC DISAGREEMENT (beyond all hypotheses)"
        entry["verdict"] = cond(q_branch_ok, v)
        b2["q_mapping"] = entry
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

        def pair_norm_ratio(pre, post, layout):
            p0 = pre[:, 0:ROPE_DIM:2].astype(np.float64)
            p1 = pre[:, 1:ROPE_DIM:2].astype(np.float64)
            if layout == "interleaved":
                q0 = post[:, 0:ROPE_DIM:2].astype(np.float64)
                q1 = post[:, 1:ROPE_DIM:2].astype(np.float64)
            else:
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

        b3["effective_mscale_measured"] = {
            "cpp_k": pair_norm_ratio(cpp_knorm[:, :ROPE_DIM], cpp_k[:, :ROPE_DIM], "interleaved"),
            "hf_k": pair_norm_ratio(hf_knorm[:, :ROPE_DIM], hf_k[:, :ROPE_DIM], "half-split"),
        }
        med_cpp = b3["effective_mscale_measured"]["cpp_k"]["median"]
        med_hf = b3["effective_mscale_measured"]["hf_k"]["median"]
        yarn_big = 1.0 + 0.1 * math.log(YARN_FACTOR)
        def scale_class(m):
            if abs(m - 1.0) < 0.02:
                return "1.0"
            if abs(m - yarn_big) < 0.02:
                return f"{yarn_big:.6f} (1+0.1*ln(120))"
            return f"OTHER ({m:.6f})"
        b3["scale_verdict"] = {
            "cpp": scale_class(med_cpp),
            "hf": scale_class(med_hf),
            "match": cond(
                k_branch_ok,
                "SCALE MATCH" if scale_class(med_cpp) == scale_class(med_hf)
                else "SCALE MISMATCH",
            ),
        }

        # Schedule fit: decoded (c,s) vs both f32-chain references.
        def schedule_residual(side, cs_key, cos_ref, sin_ref):
            c, s = decode[side][cs_key]
            rc = np.nanmean(np.abs(c - cos_ref.astype(np.float64)))
            rs = np.nanmean(np.abs(s - sin_ref.astype(np.float64)))
            mc = np.nanmax(np.abs(c - cos_ref.astype(np.float64)))
            ms = np.nanmax(np.abs(s - sin_ref.astype(np.float64)))
            return {"mean_abs_cos": float(rc), "mean_abs_sin": float(rs),
                    "max_abs_cos": float(mc), "max_abs_sin": float(ms)}

        b3["schedule_fit"] = {
            "cpp_vs_ggml_ref": schedule_residual("cpp", "interleaved_cs", ggml_cos, ggml_sin),
            "cpp_vs_hf_ref": schedule_residual("cpp", "interleaved_cs", hf_cos_ref, hf_sin_ref),
            "hf_vs_hf_ref": schedule_residual("hf", "half-split_cs", hf_cos_ref, hf_sin_ref),
            "hf_vs_ggml_ref": schedule_residual("hf", "half-split_cs", ggml_cos, ggml_sin),
        }

        # HF captured cos/sin (bf16 lattice, 64-wide duplicated halves).
        hf_cos_cap = hf["hf_rope_cos.bin"]
        hf_sin_cap = hf["hf_rope_sin.bin"]
        dup_ok = bool(
            np.array_equal(hf_cos_cap[:, :32], hf_cos_cap[:, 32:])
            and np.array_equal(hf_sin_cap[:, :32], hf_sin_cap[:, 32:])
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

        # Membership impact of the angle set (rows 2048/2049): rotate the
        # C_raw norm output with the HF at-use angles instead.
        hf_cos_use = rne_bf16(hf_cos_ref)
        hf_sin_use = rne_bf16(hf_sin_ref)
        k_angle_swap = rotate_cpp_class(c_raw, hf_cos_use, hf_sin_use)
        b3["membership_angle_swap"] = {}
        for p in (2048, 2049):
            qr = q_all[p].astype(np.float64)
            wr = wts[p].astype(np.float64)
            base_set, margin, fmax = rescore_row(p, qr, wr, k_C_raw)
            a_set, _, _ = rescore_row(p, qr, wr, k_angle_swap)
            mv = membership_verdict(base_set, a_set, margin, fmax)
            mv["verdict"] = cond(k_branch_ok, mv["verdict"])
            b3["membership_angle_swap"][str(p)] = mv
        b3["closure_authority"] = (
            "per-side extractions unconditional; cross-side conclusions "
            + ("unconditional" if k_branch_ok else
               "CONDITIONAL (upstream mismatch on the K branch)")
        )
        verdict["blocker3"] = b3

        # ================= Section 11: membership sets =================
        s11: dict = {"owners": {}}
        for li in range(14):
            owner = 2 * li
            entry = {}
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
            s11["owners"][f"owner{owner:02d}"] = entry
        s11["attribution"] = (
            "attributable to blockers 1-3" if all_upstream_ok else
            "NOT uniquely attributable to blockers 1-3 "
            "(upstream dependency flag set; see phase0)"
        )
        verdict["section11_membership"] = s11

        verdict["anomaly"] = None
        return finish(0)

    except SystemExit as exc:
        verdict["anomaly"] = str(exc)
        return finish(1)


if __name__ == "__main__":
    raise SystemExit(main())
