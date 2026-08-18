#!/usr/bin/env python3
"""Generate the cast-vs-epsilon bisect operator targets (A6 / D5), gated on
reproducing ALL FOUR committed 2x2 closure results first.

Targets (block-2 LoRA norms, from byte-exact HF-projection inputs):

  A6_q  block2_q_a_norm_A6_target.bin   = f32_norm(x, 1e-6) * w      (F32, off-lattice)
  A6_kv block2_kv_a_norm_A6_target.bin  = f32_norm(x, 1e-6) * w      (F32, off-lattice)
  D5_q  block2_q_a_norm_D5_target.bin   = bf16(bf16(f32_norm(x, 1e-5)) * w)  (f32-widened bf16)
  D5_kv block2_kv_a_norm_D5_target.bin  = bf16(bf16(f32_norm(x, 1e-5)) * w)  (f32-widened bf16)

Model recipe is the committed analyze_longcat_block2_norms.py recipe
verbatim: F32 activation, f64-accumulated variance, per-candidate cast
semantics, RNE to_bf16.

KNOWN-ANSWER GATES (all four committed closure results must reproduce
EXACTLY before any target is written):
  1. A5(q)  vs quad-run C++ dump: exact == 581,661/786,432, max f32-ulp <= 4
  2. A5(kv) vs quad-run C++ dump: exact == 190,428/262,144, max f32-ulp <= 4
  3. D6(q)  vs HF oracle: byte-exact 786,432/786,432
  4. D6(kv) vs HF oracle: 262,137/262,144 with exactly 7 one-bf16-ulp misses

Gate protocols the emitted targets serve (pre-registered in the reviewed
bisect plan):
  V-eps  in-graph dump vs A6: established F32 reduction-noise protocol
         (max f32-ulp <= 4; exact counts recorded).
  V-cast in-graph dump vs D5: BYTE-EXACT, both norms, no ULP allowance -
         the stage-B empirical standard is C++ == offline model byte-exact
         (262,144/262,144); the 7-element near-tie was D6-model-vs-HF
         residue and is NOT a valid C++-vs-D5 tolerance. Any mismatch =
         STOP for review.

Measurement-only tooling; no production arithmetic is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
B2_DIR = Path(r"D:\lc_block2_mla_512")
QUAD_DIR = REPO / "cpp_resid_walk_inject3_b2_512"
GGUF = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf")

INPUTS = {
    "hf_q_a_proj": (B2_DIR / "q_a_proj.bin",
                    "32173b18459358494f943288b974ef7df70eb540ff9e366c720c14f250407a96", (512, 1536)),
    "hf_kv_a_proj": (B2_DIR / "kv_a_proj_with_mqa.bin",
                     "28ea5b52221a94ddf780f04507f11aee7b6fc8617974f53d558424d41c470f3f", (512, 576)),
    "hf_q_a_norm": (B2_DIR / "q_a_layernorm.bin",
                    "4c9792430fee2716b573ccf365617e537adf8305571e2a5a0b1a881c0c4de340", (512, 1536)),
    "hf_kv_a_norm": (B2_DIR / "kv_a_layernorm.bin",
                     "c91991eb459352ec407aebcee5ee2b12e7b25db0bafd3e0462955a8f8144df6b", (512, 512)),
    "quad_q_a_norm": (QUAD_DIR / "block2_q_a_norm_full.bin",
                      "2b60008293032656185fa55ca5f0bb579855c67998ad7082feee8b3991ec8bb4", (512, 1536)),
    "quad_kv_a_norm": (QUAD_DIR / "block2_kv_a_norm_full.bin",
                       "93d7442a30cd7d742f21b777398783ea00faf0e9012658d37dae7d13a07698a9", (512, 512)),
}

# Committed 2x2 closure numbers (block2_norms_512/block2_norms.json).
KA_A5_Q_EXACT = 581661
KA_A5_KV_EXACT = 190428
KA_A5_MAX_ULP = 4
KA_D6_Q_EXACT = 786432
KA_D6_KV_EXACT = 262137
KA_D6_KV_MISS = 7


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


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    """The committed analyze_longcat_block2_norms.py recipe, verbatim."""
    x32 = x.astype(np.float32)
    var = (x32.astype(np.float64) ** 2).mean(axis=1)
    return x32 * (1.0 / np.sqrt(var + eps)).astype(np.float32)[:, None]


def f32_ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign-aware monotonic-key f32 ulp distance (radix-sortable transform)."""
    ua = a.view(np.uint32).astype(np.int64)
    ub = b.view(np.uint32).astype(np.int64)
    ka = np.where(ua >= 0x80000000, 0xFFFFFFFF - ua, ua + 0x80000000)
    kb = np.where(ub >= 0x80000000, 0xFFFFFFFF - ub, ub + 0x80000000)
    return np.abs(ka - kb)


def read_gguf_norm_weights() -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415

    r = GGUFReader(str(GGUF), "r")
    eps = None
    for k, f in r.fields.items():
        if "layer_norm_rms_epsilon" in k:
            eps = float(f.parts[f.data[0]][0])
    if eps is None or abs(eps - 1e-5) > 1e-12:
        stop("GGUF eps metadata unexpected: %r" % eps)
    out = {}
    for t in r.tensors:
        if t.name == "blk.2.attn_q_a_norm.weight":
            out["q"] = np.asarray(t.data, dtype=np.float32).reshape(1536)
        if t.name == "blk.2.attn_kv_a_norm.weight":
            out["kv"] = np.asarray(t.data, dtype=np.float32).reshape(512)
    if "q" not in out or "kv" not in out:
        stop("GGUF norm weights not found")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "block2_bisect_targets"))
    ns = ap.parse_args()
    out_dir = Path(ns.out_dir).resolve()

    w = read_gguf_norm_weights()
    x_q = load("hf_q_a_proj")
    x_kv = load("hf_kv_a_proj")[:, :512]
    hf_q = load("hf_q_a_norm")
    hf_kv = load("hf_kv_a_norm")
    quad_q = load("quad_q_a_norm")
    quad_kv = load("quad_kv_a_norm")

    # ---- known-answer gates 1+2: A5 vs quad C++ dumps ----------------------
    a5_q = (normalize(x_q, 1e-5) * w["q"]).astype("<f4")
    a5_kv = (normalize(x_kv, 1e-5) * w["kv"]).astype("<f4")
    ex_q = int((a5_q.view(np.uint32) == quad_q.view(np.uint32)).sum())
    ex_kv = int((a5_kv.view(np.uint32) == quad_kv.view(np.uint32)).sum())
    ulp_q = int(f32_ulp_diff(a5_q, quad_q).max())
    ulp_kv = int(f32_ulp_diff(a5_kv, quad_kv).max())
    if ex_q != KA_A5_Q_EXACT or ulp_q > KA_A5_MAX_ULP:
        stop("KA gate 1 FAIL: A5(q) exact %d (want %d), max_ulp %d" % (ex_q, KA_A5_Q_EXACT, ulp_q))
    if ex_kv != KA_A5_KV_EXACT or ulp_kv > KA_A5_MAX_ULP:
        stop("KA gate 2 FAIL: A5(kv) exact %d (want %d), max_ulp %d" % (ex_kv, KA_A5_KV_EXACT, ulp_kv))
    print("KA gates 1+2: A5 closure reproduced (q %d/786432 ulp<=%d; kv %d/262144 ulp<=%d)"
          % (ex_q, ulp_q, ex_kv, ulp_kv))

    # ---- known-answer gates 3+4: D6 vs HF oracles --------------------------
    d6_q = to_bf16(to_bf16(normalize(x_q, 1e-6)) * w["q"]).astype("<f4")
    d6_kv = to_bf16(to_bf16(normalize(x_kv, 1e-6)) * w["kv"]).astype("<f4")
    d6q_eq = int((d6_q.view(np.uint32) == hf_q.view(np.uint32)).sum())
    d6kv_eq = int((d6_kv.view(np.uint32) == hf_kv.view(np.uint32)).sum())
    if d6q_eq != KA_D6_Q_EXACT:
        stop("KA gate 3 FAIL: D6(q) %d/786432" % d6q_eq)
    if d6kv_eq != KA_D6_KV_EXACT:
        stop("KA gate 4 FAIL: D6(kv) %d/262144" % d6kv_eq)
    mism = (d6_kv.view(np.uint32) >> 16).astype(np.uint16) != (hf_kv.view(np.uint32) >> 16).astype(np.uint16)
    dm = np.abs(
        ((d6_kv.view(np.uint32) >> 16).astype(np.int64)[mism]) -
        ((hf_kv.view(np.uint32) >> 16).astype(np.int64)[mism]))
    if int(mism.sum()) != KA_D6_KV_MISS or (dm.size and int(dm.max()) != 1):
        stop("KA gate 4 FAIL: D6(kv) residue class %d misses, max bf16 delta %s"
             % (int(mism.sum()), int(dm.max()) if dm.size else 0))
    print("KA gates 3+4: D6 closure reproduced (q byte-exact 786432; kv 262137 with 7 one-ulp)")

    # ---- targets (now trusted) ----------------------------------------------
    a6_q = (normalize(x_q, 1e-6) * w["q"]).astype("<f4")
    a6_kv = (normalize(x_kv, 1e-6) * w["kv"]).astype("<f4")
    d5_q = to_bf16(to_bf16(normalize(x_q, 1e-5)) * w["q"]).astype("<f4")
    d5_kv = to_bf16(to_bf16(normalize(x_kv, 1e-5)) * w["kv"]).astype("<f4")

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "block2_q_a_norm_A6_target.bin": (a6_q, "f32 off-lattice; V-eps gate: max f32-ulp <= 4"),
        "block2_kv_a_norm_A6_target.bin": (a6_kv, "f32 off-lattice; V-eps gate: max f32-ulp <= 4"),
        "block2_q_a_norm_D5_target.bin": (d5_q, "f32-widened bf16; V-cast gate: BYTE-EXACT only"),
        "block2_kv_a_norm_D5_target.bin": (d5_kv, "f32-widened bf16; V-cast gate: BYTE-EXACT only"),
    }
    manifest = {
        "known_answer_gates": {
            "A5_q_exact": ex_q, "A5_q_max_ulp": ulp_q,
            "A5_kv_exact": ex_kv, "A5_kv_max_ulp": ulp_kv,
            "D6_q_byte_exact": d6q_eq, "D6_kv_exact": d6kv_eq,
            "D6_kv_residue": "7 one-bf16-ulp (token 177 class)",
        },
        "inputs": {k: {"path": str(v[0]), "sha256": v[1]} for k, v in INPUTS.items()},
        "gguf_eps_metadata": 1e-5,
        "targets": {},
    }
    sums = []
    for name, (arr, note) in outputs.items():
        data = arr.tobytes()
        sha = sha256_bytes(data)
        (out_dir / name).write_bytes(data)
        meta = {"name": name, "shape": list(arr.shape), "order": "token-major",
                "dtype": "float32-le", "sha256": sha, "gate_note": note}
        (out_dir / name.replace(".bin", ".json")).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["targets"][name] = meta
        sums.append("%s  %s" % (sha, name))
        print("%-36s %s" % (name, sha))
    (out_dir / "targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print("BISECT TARGETS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
