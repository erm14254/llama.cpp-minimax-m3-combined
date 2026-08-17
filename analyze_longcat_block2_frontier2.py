#!/usr/bin/env python3
"""Block-2 frontier under the hex reset: q_b_proj-2 and kv_cmpr_scaled-2.

With landings 1-6 byte-exact (residual stream, attention norm input, both
projection outputs, both LoRA norm outputs), the two frontier operators
consume byte-exact-in-value inputs:

  Q  branch: q_b_proj-2      = wq_b GEMM on the injected HF q_a_layernorm
  KV branch: kv_cmpr_scaled-2 = ggml_scale(mla_scale_kv) on the injected HF
                                kv_a_layernorm

Parameter provenance (verified this session, re-run fail-closed here):
  - blk.2.attn_q_b.weight raw-BF16 bit-identical to HF q_b_proj.weight
    (9,437,184/9,437,184); bias absent on both sides.
  - mla_scale_kv: C++ sqrtf(3072/512) and HF f32(float64 sqrt(6)) reduce to
    the identical f32 bit pattern 0x401cc471; multiplication semantics
    known-answer-validated (block-0 S2b, 909b7ee7 byte-exact).

Classification per surface: raw-exact / BF16-equivalent / BF16-irreducible.

VERDICT RULES (pre-registered):
  - BF16-equivalent but raw-different does NOT causally advance the
    frontier: the real C++ downstream graph still consumes the off-lattice
    F32 output. Recorded as "HF-equivalent at the BF16 output boundary /
    representation-boundary difference"; future advancement past it
    requires the narrowest exact-output predecessor reset first (designs
    recorded, not executed).
  - raw byte-exact -> no reset required for that branch.
  - BF16-irreducible with all inputs/parameters exact -> the operator is
    causally implicated incl. dtype/kernel semantics.

Observational context (explicit non-attribution): q_pe_rope-2 / k_pe_rope-2
under production angles (composite rule: divergence localizes at most to
the production-RoPE composite / angle-generation state, never rotation
arithmetic - R1 stands); kqv_out-2 (all-input rule NOT met: its Q/K/V
inputs are not all exact).

Measurement-only; no arithmetic changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
RUN_DIR = REPO / "cpp_resid_walk_inject4_b2_512"
HF_DIR = Path(r"D:\lc_block2_mla_512")
TGT_DIR = REPO / "block2_mla_targets"
CKPT = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved")
GGUF = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.gguf")

N_TOK = 512
LANDINGS = {
    "logical_00_full.bin": "d810f93c50ea42c5909ab289ebf62a0c5629f40530d2e5fc706dde67f0eaf763",
    "block1_attn0_norm_full.bin": "afa16c6c3324387e9261c708cae044b8fcb08acda8c8f6315d2ba8d39a8f0fd7",
    "block2_q_a_proj_full.bin": "32173b18459358494f943288b974ef7df70eb540ff9e366c720c14f250407a96",
    "block2_kv_a_proj_full.bin": "28ea5b52221a94ddf780f04507f11aee7b6fc8617974f53d558424d41c470f3f",
    "block2_q_a_norm_full.bin": "4c9792430fee2716b573ccf365617e537adf8305571e2a5a0b1a881c0c4de340",
    "block2_kv_a_norm_full.bin": "c91991eb459352ec407aebcee5ee2b12e7b25db0bafd3e0462955a8f8144df6b",
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


def classify(cpp: np.ndarray, ref: np.ndarray) -> dict:
    d = cpp.astype(np.float64) - ref.astype(np.float64)
    ref_rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
    raw_eq = int((cpp == ref).sum())
    bf16_eq = int((to_bf16(cpp) == ref).sum())
    if raw_eq == cpp.size:
        cls = "raw-exact"
    elif bf16_eq == cpp.size:
        cls = "BF16-equivalent"
    else:
        cls = "BF16-irreducible"
    mism_rows = np.nonzero((cpp != ref).any(axis=1))[0]
    return {
        "elements": int(cpp.size),
        "raw_equal": raw_eq,
        "bf16_cpp_equal_ref": bf16_eq,
        "ref_on_bf16_lattice": int((to_bf16(ref) == ref).sum()),
        "classification": cls,
        "rel_rmse": float(np.sqrt((d ** 2).mean())) / ref_rms if ref_rms > 0 else float("nan"),
        "max_abs": float(np.abs(d).max()),
        "first_divergent_token": int(mism_rows[0]) if mism_rows.size else -1,
        "divergent_token_rows": int(mism_rows.size),
    }


def verify_provenance() -> dict:
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    import torch  # noqa: PLC0415

    idx = json.loads((CKPT / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wm = idx["weight_map"]
    bias_keys = [k for k in wm if "layers.1.self_attn.0" in k and "bias" in k]
    if bias_keys:
        stop("unexpected HF attention bias keys: %r" % bias_keys)

    qb = None
    gguf_attn_bias = []
    for t in GGUFReader(str(GGUF), "r").tensors:
        if t.name == "blk.2.attn_q_b.weight":
            qb = t
        if t.name.startswith("blk.2.attn") and "bias" in t.name:
            gguf_attn_bias.append(t.name)
    if gguf_attn_bias:
        stop("unexpected GGUF attention bias tensors: %r" % gguf_attn_bias)
    if qb is None:
        stop("blk.2.attn_q_b.weight not found")
    raw16 = np.array(qb.data).view(np.uint16).reshape(-1)
    hkey = "model.layers.1.self_attn.0.q_b_proj.weight"
    with safe_open(str(CKPT / wm[hkey]), framework="pt") as f:
        hw = f.get_tensor(hkey)
    hf16 = hw.view(torch.uint16).numpy().reshape(-1)
    if raw16.size != hf16.size or int((raw16 == hf16).sum()) != hf16.size:
        stop("q_b weight raw-BF16 bit equality FAIL")

    cpp_scale = np.sqrt(np.float32(3072.0) / np.float32(512.0)).astype(np.float32)
    hf_scale = np.float32(math.sqrt(6.0))
    if int(cpp_scale.view(np.uint32)) != int(hf_scale.view(np.uint32)):
        stop("mla_scale_kv f32 bit identity FAIL")

    return {
        "q_b_weight": "raw-BF16 bit-identical 9437184/9437184 (GGUF ne [1536,6144] == HF [6144,1536])",
        "bias": "absent on both sides (no HF bias keys under layers.1.self_attn.0; no GGUF blk.2.attn* bias)",
        "mla_scale_kv": "identical f32 bits 0x%08x (cpp sqrtf(3072/512) == f32(float64 sqrt 6)); multiplication semantics known-answer-validated (block-0 S2b 909b7ee7)" % int(cpp_scale.view(np.uint32)),
    }


def main() -> int:
    print("block-2 frontier under the hex reset (measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    run_man = load_manifest(RUN_DIR)
    hf_man = load_manifest(HF_DIR)
    tgt_man = load_manifest(TGT_DIR)

    for name, sha in LANDINGS.items():
        if run_man.get(name) != sha:
            stop("landing manifest mismatch: %s" % name)
    for name in ("block2_q_a_norm_full.bin", "block2_kv_a_norm_full.bin"):
        if sha256_file(RUN_DIR / name) != LANDINGS[name]:
            stop("landing disk mismatch: %s" % name)
    print("landings 1-6 verified (manifest; 5/6 re-hashed from disk)")

    prov = verify_provenance()
    print("provenance: q_b weight bits OK, bias absent, scale constant f32-identical")

    report: dict = {
        "description": "Block-2 frontier under the hex reset: q_b_proj-2 and kv_cmpr_scaled-2",
        "run_dir": str(RUN_DIR),
        "landings": LANDINGS,
        "parameter_provenance": prov,
        "frontier": {},
        "observational_context": {},
        "verdict": {},
        "caveats": [
            "F32-carrier note: the C++ frontier inputs are F32 carriers of HF "
            "BF16-on-lattice values; any surviving difference includes "
            "dtype/kernel semantics.",
            "BF16-equivalent but raw-different does NOT causally advance the "
            "frontier: the real C++ downstream graph consumes the off-lattice "
            "F32 output (representation-boundary difference, not operator "
            "arithmetic failure). Future advancement requires the narrowest "
            "exact-output predecessor reset first.",
            "RoPE composite rule: production ggml angles differ from the "
            "captured HF cos/sin (known non-byte-exact), so rope surfaces "
            "localize at most to the production-RoPE composite / "
            "angle-generation state; rotation arithmetic is NOT attributed "
            "(R1 stands).",
            "kqv_out-2: the all-input rule is NOT met (its Q/K/V inputs are "
            "not all exact) - observational only, no attribution either way.",
        ],
    }
    p = RUN_DIR / "run_provenance.json"
    if p.is_file():
        report["provenance"] = json.loads(p.read_text(encoding="ascii"))

    # ------------------------- frontier surfaces (attribution-eligible)
    hf_qb = load_mat(HF_DIR, "q_b_proj.bin", hf_man, 6144)
    cpp_qb = load_mat(RUN_DIR, "block2_q_b_proj_full.bin", run_man, 6144)
    m_qb = classify(cpp_qb, hf_qb)
    m_qb["eligibility"] = "ELIGIBLE: activation = injected HF q_a_layernorm (landing 5); weight bit-identical; no bias"
    report["frontier"]["q_b_proj"] = m_qb
    print("q_b_proj-2       %-16s rel %.4e raw %8d/%8d bf16_eq %8d/%8d" % (
        m_qb["classification"], m_qb["rel_rmse"], m_qb["raw_equal"], m_qb["elements"],
        m_qb["bf16_cpp_equal_ref"], m_qb["elements"]))

    tgt_scaled = load_mat(TGT_DIR, "block2_kv_cmpr_scaled_target.bin", tgt_man, 512)
    cpp_scaled = load_mat(RUN_DIR, "block2_kv_cmpr_scaled_full.bin", run_man, 512)
    m_sc = classify(cpp_scaled, tgt_scaled)
    m_sc["eligibility"] = "ELIGIBLE: input = injected HF kv_a_layernorm (landing 6); scale constant f32-identical"
    report["frontier"]["kv_cmpr_scaled"] = m_sc
    print("kv_cmpr_scaled-2 %-16s rel %.4e raw %8d/%8d bf16_eq %8d/%8d" % (
        m_sc["classification"], m_sc["rel_rmse"], m_sc["raw_equal"], m_sc["elements"],
        m_sc["bf16_cpp_equal_ref"], m_sc["elements"]))

    # ------------------------- observational context (non-attribution)
    for label, cpp_name, ref_dir, ref_man, ref_name, width, note in (
        ("q_pe_rope", "block2_q_pe_rope_full.bin", TGT_DIR, tgt_man, "block2_q_pe_rope_target.bin", 2048,
         "production-RoPE composite rule; no rotation attribution"),
        ("k_pe_rope", "block2_k_pe_rope_full.bin", TGT_DIR, tgt_man, "block2_k_pe_rope_target.bin", 64,
         "production-RoPE composite rule; no rotation attribution"),
        ("kqv_out", "block2_kqv_out_full.bin", HF_DIR, hf_man, "attn_o_input.bin", 4096,
         "all-input rule NOT met (Q/K/V inputs not all exact); observational only"),
    ):
        m = classify(load_mat(RUN_DIR, cpp_name, run_man, width),
                     load_mat(ref_dir, ref_name, ref_man, width))
        m["non_attribution_note"] = note
        report["observational_context"][label] = m
        print("[obs] %-12s %-16s rel %.4e raw %d bf16_eq %d/%d" % (
            label, m["classification"], m["rel_rmse"], m["raw_equal"],
            m["bf16_cpp_equal_ref"], m["elements"]))

    # ------------------------- verdict per pre-registered rules
    def branch_verdict(label: str, m: dict, reset_design: str) -> str:
        if m["classification"] == "raw-exact":
            return "%s raw byte-exact: no reset required for this branch" % label
        if m["classification"] == "BF16-equivalent":
            return (
                "%s is HF-equivalent at the BF16 output boundary / "
                "representation-boundary difference (raw-different). This does "
                "NOT causally advance the frontier; future advancement requires "
                "the narrowest exact-output predecessor reset first: %s "
                "(design recorded, not executed)" % (label, reset_design)
            )
        return (
            "%s is BF16-IRREDUCIBLE with all inputs/parameters exact -> the "
            "operator is causally implicated including dtype/kernel semantics" % label
        )

    report["verdict"] = {
        "Q_branch": branch_verdict(
            "q_b_proj-2", m_qb,
            "exact HF q_b_proj-2 output reset before judging downstream Q/RoPE/absorption"),
        "KV_branch": branch_verdict(
            "kv_cmpr_scaled-2", m_sc,
            "exact target kv_cmpr_scaled-2 reset before judging downstream KV/value/core behavior"),
        "kqv_gate": "kqv_out-2 not judged: all-input condition not met",
    }
    print("Q  branch: %s" % report["verdict"]["Q_branch"])
    print("KV branch: %s" % report["verdict"]["KV_branch"])

    OUT = REPO / "block2_frontier2_512"
    OUT.mkdir(exist_ok=True)
    out_json = OUT / "block2_frontier2.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT / "SHA256SUMS.txt").write_text("%s  block2_frontier2.json\n" % json_sha, encoding="utf-8")
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("BLOCK2 FRONTIER2: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
