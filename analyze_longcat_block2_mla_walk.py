#!/usr/bin/env python3
"""Block-2 MLA-internal walk under the dual reset — causal frontier.

Walks the block-2 attention internals from the injected byte-exact
attn_norm-2 input to the known-discrepant attn_out-2 endpoint, comparing the
extended dual-reset run (cpp_resid_walk_inject2_b2_512) against the HF
block-2 internals capture (D:\\lc_block2_mla_512) and the known-answer-gated
offline targets (block2_mla_targets).

ALL-INPUT EXACTNESS RULE (incl. operator parameters): causal attribution to
an operator requires every semantically load-bearing numerical input -
incoming activations AND the operator's weights / learned scales / relevant
constants - to be byte-exact C++<->HF or a deterministic common input proven
equivalent. In this walk only the two root GEMMs qualify by construction:
their activation input is the injected attn_norm-2 oracle (landing-gated)
and their weights were verified raw-BF16 bit-identical this session
(blk.2.attn_q_a.weight 4,718,592/4,718,592; blk.2.attn_kv_a_mqa.weight
1,769,472/1,769,472). Every later operator consumes a differing activation
and/or differing constants and is classified observationally only.

Recorded parameter differences (source-cited, not assumptions):
- C++ il>=1 LoRA norms run build_norm with hparams.f_norm_rms_eps = 1e-5
  (GGUF attention.layer_norm_rms_epsilon) while the HF q_a_layernorm /
  kv_a_layernorm use the LongcatFlashRMSNorm default eps = 1e-6 (proven at
  block 0). The il==0 diagnostics pinned 1e-6 explicitly; il>=1 does not.
- RoPE rule: the production run uses ggml-generated angles; the targets use
  captured HF cos/sin, known NOT byte-exact (sin 3,377/16,384). Divergence
  at q_pe_rope-2 / k_pe_rope-2 localizes at most to the production-RoPE
  composite / angle-generation state - never rotation arithmetic (R1).

CAUSAL FRONTIER: Q and KV are parallel branches from the exact input; the
earliest causally defensible divergence is reported independently per
branch. If both root GEMMs are irreducible from exact inputs they are
co-earliest and the verdict is stop-for-review (no order-based choice).

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
RUN_DIR = REPO / "cpp_resid_walk_inject2_b2_512"
PRIOR_DUAL = REPO / "cpp_resid_walk_inject2_b1_512"
HF_DIR = Path(r"D:\lc_block2_mla_512")
TGT_DIR = REPO / "block2_mla_targets"

N_TOK = 512
ORACLE2_SHA = "afa16c6c3324387e9261c708cae044b8fcb08acda8c8f6315d2ba8d39a8f0fd7"

WEIGHT_PROVENANCE = {
    "blk.2.attn_q_a.weight": {
        "hf": "model.layers.1.self_attn.0.q_a_proj.weight",
        "raw_bf16_bit_equal": "4718592/4718592",
        "layout": "GGUF ne [3072,1536] row-major == HF [1536,3072] row-major",
    },
    "blk.2.attn_kv_a_mqa.weight": {
        "hf": "model.layers.1.self_attn.0.kv_a_proj_with_mqa.weight",
        "raw_bf16_bit_equal": "1769472/1769472",
        "layout": "GGUF ne [3072,576] row-major == HF [576,3072] row-major",
    },
}

# (label, branch, cpp_file, ref_dir_tag, ref_file, width, eligible, eligibility_note)
SURFACES = [
    ("q_a_proj", "Q", "block2_q_a_proj_full.bin", "hf", "q_a_proj.bin", 1536, True,
     "ROOT GEMM: activation = injected attn_norm-2 oracle (landing-gated); "
     "weight raw-BF16 bit-identical -> ALL inputs exact, attribution permitted"),
    ("q_a_norm", "Q", "block2_q_a_norm_full.bin", "hf", "q_a_layernorm.bin", 1536, False,
     "input = C++ q_a_proj-2 (differs from HF); constants differ by source: "
     "C++ eps 1e-5 (f_norm_rms_eps) vs HF LoRA eps 1e-6 -> observational only"),
    ("q_b_proj", "Q", "block2_q_b_proj_full.bin", "hf", "q_b_proj.bin", 6144, False,
     "input = C++ q_a_norm-2 (differs) -> observational only"),
    ("q_pe_rope", "Q", "block2_q_pe_rope_full.bin", "tgt", "block2_q_pe_rope_target.bin", 2048, False,
     "production ggml angles vs captured HF cos/sin (known non-byte-exact) + "
     "differing q_b input -> localizes at most to the production-RoPE "
     "composite / angle-generation state (R1 covers rotation arithmetic)"),
    ("kv_a_proj", "KV", "block2_kv_a_proj_full.bin", "hf", "kv_a_proj_with_mqa.bin", 576, True,
     "ROOT GEMM: activation = injected attn_norm-2 oracle (landing-gated); "
     "weight raw-BF16 bit-identical -> ALL inputs exact, attribution permitted"),
    ("kv_a_norm", "KV", "block2_kv_a_norm_full.bin", "hf", "kv_a_layernorm.bin", 512, False,
     "input = C++ kv_cmpr_pe-2 slice (differs); constants differ by source: "
     "C++ eps 1e-5 vs HF LoRA eps 1e-6 -> observational only"),
    ("kv_cmpr_scaled", "KV", "block2_kv_cmpr_scaled_full.bin", "tgt", "block2_kv_cmpr_scaled_target.bin", 512, False,
     "input = C++ kv_a_norm-2 (differs); C++ has no post-scale round at il>=1 "
     "-> observational only"),
    ("k_pe_rope", "KV", "block2_k_pe_rope_full.bin", "tgt", "block2_k_pe_rope_target.bin", 64, False,
     "production-RoPE composite rule (as q_pe_rope) + differing kv input -> "
     "observational only"),
    ("kqv_out", "MERGE", "block2_kqv_out_full.bin", "hf", "attn_o_input.bin", 4096, False,
     "multi-input merge: consumes Q-branch state, KV cache state, absorbed "
     "attn_k_b/attn_v_b weights, mask - incoming branch states differ -> "
     "observational only (all-input rule)"),
    ("attn_out", "MERGE", "block1_attn0_out_full.bin", "hf", "o_proj_out.bin", 3072, False,
     "endpoint context (committed dual-reset result, reproduced 81/81); "
     "inputs differ -> observational"),
]


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
    c = cpp.astype(np.float64)
    h = ref.astype(np.float64)
    d = c - h
    rmse = float(np.sqrt((d ** 2).mean()))
    ref_rms = float(np.sqrt((h ** 2).mean()))
    mism_rows = np.nonzero((cpp != ref).any(axis=1))[0]
    raw_eq = int((cpp == ref).sum())
    bf16_eq = int((to_bf16(cpp) == ref).sum())
    if raw_eq == cpp.size:
        cls = "raw-exact"
    elif bf16_eq == cpp.size:
        cls = "bf16-reducible"
    else:
        cls = "bf16-irreducible"
    return {
        "elements": int(cpp.size),
        "raw_equal": raw_eq,
        "ref_on_bf16_lattice": int((to_bf16(ref) == ref).sum()),
        "bf16_cpp_equal_ref": bf16_eq,
        "classification": cls,
        "rel_rmse": rmse / ref_rms if ref_rms > 0 else float("nan"),
        "max_abs": float(np.abs(d).max()),
        "first_divergent_token": int(mism_rows[0]) if mism_rows.size else -1,
        "divergent_token_rows": int(mism_rows.size),
    }


def main() -> int:
    print("block-2 MLA-internal walk under the dual reset (measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    run_man = load_manifest(RUN_DIR)
    prior_man = load_manifest(PRIOR_DUAL)
    hf_man = load_manifest(HF_DIR)
    tgt_man = load_manifest(TGT_DIR)

    # Identity gates.
    if run_man.get("block1_attn0_norm_full.bin") != ORACLE2_SHA:
        stop("landing 2 mismatch in b2 manifest")
    if run_man.get("block1_attn0_out_full.bin") != prior_man.get("block1_attn0_out_full.bin"):
        stop("attn_out-2 endpoint reproduction mismatch (b2 vs committed dual-reset)")
    if hf_man.get("q_a_proj_input.bin") != ORACLE2_SHA:
        stop("HF internals input != attn0_norm oracle")
    print("identity gates: landing2, attn_out-2 endpoint reproduction, HF input gate PASS")

    report: dict = {
        "description": "Block-2 MLA-internal walk under the dual reset - causal frontier",
        "run_dir": str(RUN_DIR),
        "hf_dir": str(HF_DIR),
        "targets_dir": str(TGT_DIR),
        "weight_provenance": WEIGHT_PROVENANCE,
        "surfaces": {},
        "frontier": {},
        "caveats": [
            "All-input exactness (incl. parameters): only the two root GEMMs are "
            "eligible for causal attribution in this walk; every later operator "
            "consumes a differing activation and/or differing constants and is "
            "classified observationally.",
            "Strict classification: a single differing element makes a surface "
            "bf16-irreducible; magnitudes and counts are recorded so review can "
            "weigh reduction-order-scale effects.",
            "The C++ surfaces are F32 carriers (uncorrected il>=1 path); HF "
            "boundaries are BF16-on-lattice.",
            "Representation-boundary distinction: BF16-reducible root GEMMs are "
            "HF-equivalent AT THE BF16 OUTPUT BOUNDARY, but the raw C++ F32 "
            "outputs are NOT pipeline-equivalent - later C++ norms consume the "
            "off-lattice values. This is a missing/different representation "
            "boundary, not a GEMM arithmetic failure.",
            "Eps distinction: the il>=1 LoRA-norm eps difference (C++ 1e-5 vs "
            "HF 1e-6) is a direct semantic parameter mismatch proven from "
            "source/config; its quantitative contribution relative to "
            "predecessor representation and cast-ordering differences is "
            "unmeasured in this walk.",
        ],
    }
    p = RUN_DIR / "run_provenance.json"
    if p.is_file():
        report["provenance"] = json.loads(p.read_text(encoding="ascii"))

    branch_first_raw: dict[str, str] = {}
    branch_first_irred: dict[str, str] = {}
    root_class: dict[str, str] = {}

    for label, branch, cpp_name, ref_tag, ref_name, width, eligible, note in SURFACES:
        cpp = load_mat(RUN_DIR, cpp_name, run_man, width)
        ref_dir, ref_man = (HF_DIR, hf_man) if ref_tag == "hf" else (TGT_DIR, tgt_man)
        ref = load_mat(ref_dir, ref_name, ref_man, width)
        m = classify(cpp, ref)
        m["branch"] = branch
        m["attribution_eligible"] = eligible
        m["eligibility_note"] = note
        report["surfaces"][label] = m

        if branch in ("Q", "KV"):
            if m["classification"] != "raw-exact" and branch not in branch_first_raw:
                branch_first_raw[branch] = label
            if m["classification"] == "bf16-irreducible" and branch not in branch_first_irred:
                branch_first_irred[branch] = label
        if label in ("q_a_proj", "kv_a_proj"):
            root_class[label] = m["classification"]

        print(
            "%-15s %-5s %-16s rel %.4e raw %8d/%8d bf16_eq %8d/%8d tok0 %d %s"
            % (label, branch, m["classification"], m["rel_rmse"], m["raw_equal"],
               m["elements"], m["bf16_cpp_equal_ref"], m["elements"],
               m["first_divergent_token"], "ELIGIBLE" if eligible else "")
        )

    # Causal frontier.
    frontier = {
        "per_branch_first_raw": branch_first_raw,
        "per_branch_first_bf16_irreducible": branch_first_irred,
        "root_classifications": root_class,
    }
    q_root_irred = root_class.get("q_a_proj") == "bf16-irreducible"
    kv_root_irred = root_class.get("kv_a_proj") == "bf16-irreducible"
    if q_root_irred and kv_root_irred:
        frontier["verdict"] = (
            "CO-EARLIEST: both root GEMMs are bf16-irreducible from all-exact "
            "inputs (activations + bit-identical weights) -> both block-2 "
            "projection GEMMs are causally implicated, including dtype/kernel "
            "semantics; stop for review (no order-based choice)"
        )
    elif q_root_irred or kv_root_irred:
        which = "q_a_proj" if q_root_irred else "kv_a_proj"
        frontier["verdict"] = (
            "%s is bf16-irreducible from all-exact inputs -> that projection "
            "GEMM is causally implicated (incl. dtype/kernel semantics); the "
            "other branch's first irreducible surface sits behind differing "
            "inputs -> observational; stop for review" % which
        )
    else:
        frontier["verdict"] = (
            "Neither root GEMM is bf16-irreducible (both reproduce HF under "
            "output rounding from all-exact inputs) -> no operator in this "
            "walk is both attribution-eligible and irreducible; every later "
            "irreducible surface sits behind differing inputs/constants -> "
            "stop for review; the designed next steps are the narrowest exact "
            "predecessor resets (e.g. exact bf16 projection-output injection) "
            "and/or the recorded parameter-difference review (LoRA-norm eps)"
        )
    report["frontier"] = frontier
    print("frontier verdict: %s" % frontier["verdict"])

    OUT = REPO / "block2_mla_walk_512"
    OUT.mkdir(exist_ok=True)
    out_json = OUT / "block2_mla_walk.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT / "SHA256SUMS.txt").write_text("%s  block2_mla_walk.json\n" % json_sha, encoding="utf-8")
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("BLOCK2 MLA WALK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
