#!/usr/bin/env python3
"""Block-2 LoRA norms under the quad reset — exact-input judgment + 2x2.

With all four landings byte-exact (logical_00, attn_norm-2, q_a_proj-2,
kv_cmpr_pe-2), the two LoRA norms are the next operators with byte-exact
activation inputs. Their weights were proven the exact numerical widening of
the HF BF16 weights (1536/1536, 512/512) and the epsilon constants proven
DIFFERENT (C++ f_norm_rms_eps = 1e-5 from GGUF metadata + build_norm source;
HF variance_epsilon = 1e-6 from the instantiated modules, runtime-gated this
session). Attribution therefore targets the NORM OPERATOR COMPOSITE
(differing eps constant + cast semantics + any kernel detail), decomposed by
the pre-registered 2x2 below.

2x2 candidates per norm (input x = the exact injected HF oracle; w = the
verified GGUF weight; normalize() = F32 activation, F64 variance — the
established model):
    A5 = f32_norm(x, 1e-5) * w                       (C++ style, C++ eps)
    A6 = f32_norm(x, 1e-6) * w                       (C++ style, HF eps)
    D5 = bf16( bf16(f32_norm(x, 1e-5)) * w )         (HF cast, C++ eps)
    D6 = bf16( bf16(f32_norm(x, 1e-6)) * w )         (HF cast, HF eps)
A mechanism is called CLOSED only on whole-tensor byte-exact reproduction
(A5 <-> C++ dump; D6 <-> HF dump), scoped to this frozen capture; anything
short is reported as model residue (ulp-quantified).

EFFECTS ARE NON-ADDITIVE: eps is reported under both cast regimes, cast
under both eps regimes, plus the combined A5<->D6 gap; the factors are not
forced to sum to a causal partition.

Predecessor-representation intervention (zero extra runs): the dual-reset
norm dumps (off-lattice C++ F32 projection predecessor) vs the quad-reset
norm dumps (exact HF projection-output predecessor), against the same HF
oracles — phrased as the effect of the exact-predecessor intervention on
this frozen capture, not an additive causal percentage.

Measurement-only; no arithmetic changes.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
QUAD_DIR = REPO / "cpp_resid_walk_inject3_b2_512"
DUAL_DIR = REPO / "cpp_resid_walk_inject2_b2_512"
HF_DIR = Path(r"D:\lc_block2_mla_512")
CKPT = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved")
GGUF = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf")

N_TOK = 512
ORACLE3_SHA = "32173b18459358494f943288b974ef7df70eb540ff9e366c720c14f250407a96"
ORACLE4_SHA = "28ea5b52221a94ddf780f04507f11aee7b6fc8617974f53d558424d41c470f3f"

RUNTIME_EPS_GATE = {
    "method": (
        "from_pretrained instantiation identical to the frozen captures; "
        "attribute read from the actual modules this session"
    ),
    "q_a_layernorm.variance_epsilon": 1e-6,
    "kv_a_layernorm.variance_epsilon": 1e-6,
    "result": "PASS (both 1e-6)",
}


def stop(msg: str) -> None:
    raise SystemExit("STOP: " + msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    var = (x32.astype(np.float64) ** 2).mean(axis=1)
    return x32 * (1.0 / np.sqrt(var + eps)).astype(np.float32)[:, None]


def load_manifest(dir_: Path) -> dict[str, str]:
    p = dir_ / "SHA256SUMS.txt"
    if not p.is_file():
        stop("manifest missing: %s" % p)
    out = {}
    for line in p.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line:
            continue
        sha, name = line.split(None, 1)
        out[name.strip()] = sha.lower()
    return out


def load_mat(dir_: Path, name: str, manifest: dict[str, str], width: int) -> np.ndarray:
    p = dir_ / name
    if not p.is_file():
        stop("missing input: %s" % p)
    if name not in manifest:
        stop("%s not in manifest of %s" % (name, dir_))
    got = sha256_file(p)
    if got != manifest[name]:
        stop("SHA mismatch: %s got %s manifest %s" % (p, got, manifest[name]))
    raw = p.read_bytes()
    if len(raw) != N_TOK * width * 4:
        stop("size mismatch: %s" % p)
    v = np.frombuffer(raw, dtype="<f4").reshape(N_TOK, width).copy()
    if not np.isfinite(v).all():
        stop("non-finite: %s" % p)
    return v


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.float64) - b.astype(np.float64)
    b_rms = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
    eq = int((a == b).sum())
    return {
        "elements": int(a.size),
        "exact": eq,
        "rel_rmse": float(np.sqrt((d ** 2).mean())) / b_rms if b_rms > 0 else float("nan"),
        "max_abs": float(np.abs(d).max()),
    }


def classify(cpp: np.ndarray, hf: np.ndarray) -> dict:
    m = metrics(cpp, hf)
    bf16_eq = int((to_bf16(cpp) == hf).sum())
    m["bf16_cpp_equal_hf"] = bf16_eq
    m["hf_on_bf16_lattice"] = int((to_bf16(hf) == hf).sum())
    if m["exact"] == m["elements"]:
        m["classification"] = "raw-exact"
    elif bf16_eq == m["elements"]:
        m["classification"] = "bf16-reducible"
    else:
        m["classification"] = "bf16-irreducible"
    mism_rows = np.nonzero((cpp != hf).any(axis=1))[0]
    m["first_divergent_token"] = int(mism_rows[0]) if mism_rows.size else -1
    return m


def read_gguf_norm_weights_and_eps():
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    r = GGUFReader(str(GGUF), "r")
    eps = None
    for k, f in r.fields.items():
        if "layer_norm_rms_epsilon" in k:
            eps = float(f.parts[f.data[0]][0])
    if eps is None or abs(eps - 1e-5) > 1e-12:
        stop("GGUF eps metadata unexpected: %r" % eps)

    targets = {
        "blk.2.attn_q_a_norm.weight": ("model.layers.1.self_attn.0.q_a_layernorm.weight", 1536),
        "blk.2.attn_kv_a_norm.weight": ("model.layers.1.self_attn.0.kv_a_layernorm.weight", 512),
    }
    gg = {}
    for t in r.tensors:
        if t.name in targets:
            gg[t.name] = np.array(t.data).reshape(-1).astype(np.float32)
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = idx["weight_map"]
    weights = {}
    import torch  # noqa: PLC0415
    for gname, (hkey, width) in targets.items():
        if gname not in gg or gg[gname].size != width:
            stop("GGUF norm weight missing/odd: %s" % gname)
        with safe_open(str(CKPT / wm[hkey]), framework="pt") as f:
            hw = f.get_tensor(hkey)
        hw32 = hw.float().numpy().reshape(-1)
        if int((gg[gname] == hw32).sum()) != width:
            stop("norm weight widening-equivalence FAIL: %s" % gname)
        weights[gname] = gg[gname]
    return eps, weights


def main() -> int:
    print("block-2 LoRA norms under the quad reset (measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    quad_man = load_manifest(QUAD_DIR)
    dual_man = load_manifest(DUAL_DIR)
    hf_man = load_manifest(HF_DIR)

    # Landings re-verified from disk (3 and 4 are the exact-input licenses).
    if quad_man.get("block2_q_a_proj_full.bin") != ORACLE3_SHA:
        stop("landing3 mismatch in quad manifest")
    if quad_man.get("block2_kv_a_proj_full.bin") != ORACLE4_SHA:
        stop("landing4 mismatch in quad manifest")
    for f, sha in (("block2_q_a_proj_full.bin", ORACLE3_SHA), ("block2_kv_a_proj_full.bin", ORACLE4_SHA)):
        if sha256_file(QUAD_DIR / f) != sha:
            stop("landing disk mismatch: %s" % f)
    print("landing gates 3/4 re-verified byte-exact")

    gguf_eps, weights = read_gguf_norm_weights_and_eps()
    print("parameter provenance: GGUF eps = %.12g (f32(1e-5)); both norm weights exact widening of HF bf16" % gguf_eps)

    hf_q_proj = load_mat(HF_DIR, "q_a_proj.bin", hf_man, 1536)
    hf_kv_proj = load_mat(HF_DIR, "kv_a_proj_with_mqa.bin", hf_man, 576)

    report: dict = {
        "description": "Block-2 LoRA norms under the quad reset: exact-input judgment + non-additive 2x2",
        "quad_dir": str(QUAD_DIR),
        "dual_dir": str(DUAL_DIR),
        "hf_dir": str(HF_DIR),
        "parameter_provenance": {
            "gguf_eps_metadata": gguf_eps,
            "cpp_eps_source": "build_norm uses hparams.f_norm_rms_eps at il>=1 (longcat-flash-ngram.cpp:710-715, :923-928)",
            "hf_runtime_eps_gate": RUNTIME_EPS_GATE,
            "norm_weights": "GGUF F32 == exact widening of HF BF16 (1536/1536, 512/512), verified in-script",
        },
        "norms": {},
        "two_by_two": {},
        "predecessor_intervention": {},
        "caveats": [
            "Attribution targets the NORM OPERATOR COMPOSITE (differing eps "
            "constant + cast semantics + kernel detail); the 2x2 decomposes it.",
            "2x2 factors are NON-ADDITIVE: eps reported under both cast "
            "regimes, cast under both eps regimes, plus the combined gap; no "
            "forced causal partition.",
            "Mechanism closure claims are scoped to this frozen full-sequence "
            "capture and require whole-tensor byte-exact reproduction.",
            "Predecessor-intervention numbers are the effect of the exact "
            "HF projection-output reset on this capture, not additive causal "
            "percentages (the nonlinear norm does not superpose).",
        ],
    }
    p = QUAD_DIR / "run_provenance.json"
    if p.is_file():
        report["provenance"] = json.loads(p.read_text(encoding="ascii"))

    for tag, cpp_name, hf_name, width, x_input, w_name in (
        ("q_a_norm", "block2_q_a_norm_full.bin", "q_a_layernorm.bin", 1536,
         hf_q_proj, "blk.2.attn_q_a_norm.weight"),
        ("kv_a_norm", "block2_kv_a_norm_full.bin", "kv_a_layernorm.bin", 512,
         hf_kv_proj[:, :512], "blk.2.attn_kv_a_norm.weight"),
    ):
        cpp_quad = load_mat(QUAD_DIR, cpp_name, quad_man, width)
        cpp_dual = load_mat(DUAL_DIR, cpp_name, dual_man, width)
        hf = load_mat(HF_DIR, hf_name, hf_man, width)
        w = weights[w_name]

        m = classify(cpp_quad, hf)
        m["attribution"] = (
            "PERMITTED to the norm operator composite: activation input "
            "byte-exact (landing), weight exact widening, eps constants "
            "proven different (C++ 1e-5 vs HF 1e-6)"
        )
        report["norms"][tag] = m
        print("%-10s quad vs HF: %s rel %.4e raw %d/%d bf16_eq %d/%d" % (
            tag, m["classification"], m["rel_rmse"], m["exact"], m["elements"],
            m["bf16_cpp_equal_hf"], m["elements"]))

        # 2x2 candidates from the exact input.
        x = x_input
        a5 = np.ascontiguousarray(normalize(x, 1e-5) * w[None, :], dtype="<f4")
        a6 = np.ascontiguousarray(normalize(x, 1e-6) * w[None, :], dtype="<f4")
        d5 = to_bf16(to_bf16(normalize(x, 1e-5)) * w[None, :]).astype("<f4")
        d6 = to_bf16(to_bf16(normalize(x, 1e-6)) * w[None, :]).astype("<f4")

        def closure(cand: np.ndarray, dump: np.ndarray, name: str) -> dict:
            eq = int((cand == dump).sum())
            out = {"exact": eq, "elements": int(cand.size), "byte_exact": bool(eq == cand.size)}
            if not out["byte_exact"]:
                ulp = np.abs(cand.view(np.int32).astype(np.int64)
                             - np.ascontiguousarray(dump, dtype="<f4").view(np.int32).astype(np.int64))
                out["max_ulp"] = int(ulp.max())
                out["ulp_le_1_pct"] = float((ulp <= 1).mean() * 100.0)
            out["closure"] = (
                ("CLOSED for this frozen capture (byte-exact)" if out["byte_exact"]
                 else "NOT closed - model residue reported")
            )
            print("  closure %-22s %d/%d %s" % (name, eq, cand.size,
                  "BYTE-EXACT" if out["byte_exact"] else "residue max_ulp=%d" % out["max_ulp"]))
            return out

        tt = {
            "closure_A5_vs_cpp_dump": closure(a5, cpp_quad, "A5<->cpp(%s)" % tag),
            "closure_D6_vs_hf_dump": closure(d6, hf, "D6<->hf(%s)" % tag),
            "eps_effect_under_A": metrics(a5, a6),
            "eps_effect_under_D": metrics(d5, d6),
            "cast_effect_under_1e5": metrics(a5, d5),
            "cast_effect_under_1e6": metrics(a6, d6),
            "combined_A5_vs_D6": metrics(a5, d6),
            "non_additivity_note": (
                "factors are non-additive; each effect is conditional on the "
                "other's regime; no causal partition is implied"
            ),
        }
        report["two_by_two"][tag] = tt
        print("  eps|A %.3e  eps|D %.3e  cast|1e-5 %.3e  cast|1e-6 %.3e  combined %.3e" % (
            tt["eps_effect_under_A"]["rel_rmse"], tt["eps_effect_under_D"]["rel_rmse"],
            tt["cast_effect_under_1e5"]["rel_rmse"], tt["cast_effect_under_1e6"]["rel_rmse"],
            tt["combined_A5_vs_D6"]["rel_rmse"]))

        m_old = classify(cpp_dual, hf)
        report["predecessor_intervention"][tag] = {
            "old_offlattice_predecessor": {k: m_old[k] for k in ("rel_rmse", "bf16_cpp_equal_hf", "classification", "max_abs")},
            "new_exact_predecessor": {k: m[k] for k in ("rel_rmse", "bf16_cpp_equal_hf", "classification", "max_abs")},
            "phrasing": (
                "effect of the exact HF projection-output reset on this frozen "
                "capture; not an additive causal percentage"
            ),
        }
        print("  predecessor intervention: old rel %.4e (bf16_eq %d) -> new rel %.4e (bf16_eq %d)" % (
            m_old["rel_rmse"], m_old["bf16_cpp_equal_hf"], m["rel_rmse"], m["bf16_cpp_equal_hf"]))

    OUT = REPO / "block2_norms_512"
    OUT.mkdir(exist_ok=True)
    out_json = OUT / "block2_norms.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT / "SHA256SUMS.txt").write_text("%s  block2_norms.json\n" % json_sha, encoding="utf-8")
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("BLOCK2 NORMS JUDGMENT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
