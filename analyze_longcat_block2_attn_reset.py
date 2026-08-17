#!/usr/bin/env python3
"""Block-2 attention judgment under the dual reset (logical_00 + attn_norm-2).

Both block-2 attention operands are exact in value in the dual-reset run:
the residual-stream operand (injected l_out-1 = HF logical_00 oracle) and
the attention-normalized operand (injected attn_norm-2 = HF attn0_norm
oracle). F32-CARRIER NOTE (mandated, verbatim): the C++ tensors are F32
carrying HF BF16-on-lattice values, so any surviving difference localizes to
the block-2 attention implementation INCLUDING ITS DTYPE/KERNEL SEMANTICS —
the F32-vs-BF16 carrier is itself part of the implementation under test.

Ordered classification — attn_out-2 (pre-residual-add, post-o_proj) FIRST,
then ffn_inp-2 (post-add). ffn_inp-2 alone cannot prove the attention output
value-exact (BF16 rounding can mask a small attention difference) nor
uniquely attribute an irreducibility to attention (post-add irreducibility
initially localizes to the attention+residual-add composite).

Pre-registered interpretation grid:
  1. attn_out-2 raw-exact       -> attention output exact; later ffn_inp-2
                                   mismatch belongs to residual-add/storage
                                   semantics.
  2. attn_out-2 BF16-reducible  -> attention BF16-equivalent, NOT value-
                                   exact; inspect whether the add preserves
                                   or amplifies the off-lattice difference.
  3. attn_out-2 BF16-irreducible-> block-2 attention causally implicated
                                   under value-exact inputs; only then design
                                   deeper MLA-internal capture.
  4. attn_out-2 reducible but ffn_inp-2 irreducible -> first irreducibility
                                   localizes to the residual-add/carrier
                                   boundary, not attention internals.

Next-branch predecessor-exactness: causal attribution of the post-attention-
norm/MLP path is licensed only if ffn_inp-2 is RAW byte-exact; BF16-reducible
is not sufficient (the next operator would still consume a different F32
predecessor).

Add-semantics candidates S1/S2/S3 are CANDIDATE reconstructions; the add
mechanism is claimed closed only if a candidate reproduces the HF boundary
byte-exactly (1,572,864/1,572,864).

Measurement-only. No arithmetic changes anywhere.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
HF_DIR = Path(r"D:\lc_block1_stages_512")
DUAL_DIR = REPO / "cpp_resid_walk_inject2_b1_512"
SINGLE_DIR = REPO / "cpp_resid_walk_inject_b1_512"
ORACLE_DIR = Path(r"D:\lc_resid_walk_512")
OUT_DIR = REPO / "block2_attn_reset_512"

N_TOK = 512
HIDDEN = 3072
VEC_BYTES = N_TOK * HIDDEN * 4

ORACLE_SHA = "d810f93c50ea42c5909ab289ebf62a0c5629f40530d2e5fc706dde67f0eaf763"
ORACLE2_SHA = "afa16c6c3324387e9261c708cae044b8fcb08acda8c8f6315d2ba8d39a8f0fd7"

SEMANTIC_EQUIVALENCE = {
    "attn_out_pair": (
        "C++ attn_out-2: post-wo/o_proj, pre-residual-add (cb at "
        "longcat-flash-ngram.cpp:984, before the ffn_inp add at :996-997), "
        "F32 [3072,512]. HF attn0_out: the self_attn[0] return value snapped "
        "BEFORE the residual add (modeling :994-1004; at 512 tokens the "
        "dense fast-path o_proj output), BF16-on-lattice widened to f32. "
        "Same semantic boundary; dtype carrier differs by design."
    ),
    "ffn_inp_pair": (
        "C++ ffn_inp-2 = attn_out-2 + inpSA (F32 add of the injected "
        "residual oracle). HF attn0_resid = residual + attn0_out computed in "
        "BF16 (output on-lattice). Same semantic boundary; the add's "
        "operand/rounding semantics differ by design and are separated by "
        "the S1/S2/S3 candidates."
    ),
    "f32_carrier_note": (
        "The C++ tensors are F32 carrying HF BF16-on-lattice input values, "
        "so any surviving difference localizes to the block-2 attention "
        "implementation including its dtype/kernel semantics - the "
        "F32-vs-BF16 carrier is itself part of the implementation under "
        "test."
    ),
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


def load_mat(dir_: Path, name: str, manifest: dict[str, str]) -> np.ndarray:
    p = dir_ / name
    if not p.is_file():
        stop("missing input: %s" % p)
    if name not in manifest:
        stop("%s not in manifest of %s" % (name, dir_))
    got = sha256_file(p)
    if got != manifest[name]:
        stop("SHA mismatch: %s got %s manifest %s" % (p, got, manifest[name]))
    raw = p.read_bytes()
    if len(raw) != VEC_BYTES:
        stop("size mismatch: %s" % p)
    v = np.frombuffer(raw, dtype="<f4").reshape(N_TOK, HIDDEN).copy()
    if not np.isfinite(v).all():
        stop("non-finite values in %s" % p)
    return v


def classify(cpp: np.ndarray, hf: np.ndarray) -> dict:
    c = cpp.astype(np.float64)
    h = hf.astype(np.float64)
    d = c - h
    rmse = float(np.sqrt((d ** 2).mean()))
    hf_rms = float(np.sqrt((h ** 2).mean()))
    row_l2 = np.sqrt((d ** 2).sum(axis=1))
    mism_rows = np.nonzero((cpp != hf).any(axis=1))[0]
    raw_eq = int((cpp == hf).sum())
    bf16_eq = int((to_bf16(cpp) == hf).sum())
    if raw_eq == cpp.size:
        cls = "raw-exact"
    elif bf16_eq == cpp.size:
        cls = "bf16-reducible"
    else:
        cls = "bf16-irreducible"
    return {
        "elements": int(cpp.size),
        "raw_equal": raw_eq,
        "hf_on_bf16_lattice": int((to_bf16(hf) == hf).sum()),
        "bf16_cpp_equal_hf": bf16_eq,
        "classification": cls,
        "rel_rmse": rmse / hf_rms if hf_rms > 0 else float("nan"),
        "abs_rmse": rmse,
        "err_l2": float(np.sqrt((d ** 2).sum())),
        "max_abs": float(np.abs(d).max()),
        "first_divergent_token": int(mism_rows[0]) if mism_rows.size else -1,
        "divergent_token_rows": int(mism_rows.size),
        "row_l2_max": float(row_l2.max()),
        "row_l2_final": float(row_l2[-1]),
    }


def main() -> int:
    print("block-2 attention judgment under dual reset (measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    hf_man = load_manifest(HF_DIR)
    dual_man = load_manifest(DUAL_DIR)
    single_man = load_manifest(SINGLE_DIR)
    oracle_man = load_manifest(ORACLE_DIR)

    # Identity gates (fail-closed).
    if oracle_man.get("logical_00_oracle.bin") != ORACLE_SHA:
        stop("oracle manifest mismatch")
    if hf_man.get("attn0_norm.bin") != ORACLE2_SHA:
        stop("oracle2 manifest mismatch")
    if dual_man.get("logical_00_full.bin") != ORACLE_SHA:
        stop("dual-run landing 1 mismatch")
    if dual_man.get("block1_attn0_norm_full.bin") != ORACLE2_SHA:
        stop("dual-run landing 2 mismatch")
    for name in ("attn0_out.bin", "attn0_resid.bin"):
        if name not in hf_man:
            stop("HF surface missing from manifest: %s" % name)
    print("identity gates: both landings + HF surfaces manifest-gated PASS")

    report: dict = {
        "description": "Block-2 attention judgment under the dual reset (logical_00 + attn_norm-2)",
        "oracle_sha256": ORACLE_SHA,
        "oracle2_sha256": ORACLE2_SHA,
        "hf_dir": str(HF_DIR),
        "dual_dir": str(DUAL_DIR),
        "semantic_equivalence": SEMANTIC_EQUIVALENCE,
        "provenance": {},
        "surfaces": {},
        "add_semantics_candidates": {},
        "downstream_descriptive": {},
        "verdict": {},
    }
    p = DUAL_DIR / "run_provenance.json"
    if p.is_file():
        report["provenance"]["inject2_b1"] = json.loads(p.read_text(encoding="ascii"))

    # ---------------------------------------------------------- ordered pair 1
    hf_attn_out = load_mat(HF_DIR, "attn0_out.bin", hf_man)
    cpp_attn_out = load_mat(DUAL_DIR, "block1_attn0_out_full.bin", dual_man)
    m_out = classify(cpp_attn_out, hf_attn_out)
    report["surfaces"]["attn_out_2"] = m_out
    print(
        "attn_out-2 : %s | rel %.4e | raw %d | bf16_eq %d/%d | first_tok %d"
        % (m_out["classification"], m_out["rel_rmse"], m_out["raw_equal"],
           m_out["bf16_cpp_equal_hf"], m_out["elements"], m_out["first_divergent_token"])
    )

    # ---------------------------------------------------------- ordered pair 2
    hf_resid = load_mat(HF_DIR, "attn0_resid.bin", hf_man)
    cpp_resid = load_mat(DUAL_DIR, "block1_attn0_resid_full.bin", dual_man)
    m_ffn = classify(cpp_resid, hf_resid)
    report["surfaces"]["ffn_inp_2"] = m_ffn
    print(
        "ffn_inp-2  : %s | rel %.4e | raw %d | bf16_eq %d/%d | first_tok %d"
        % (m_ffn["classification"], m_ffn["rel_rmse"], m_ffn["raw_equal"],
           m_ffn["bf16_cpp_equal_hf"], m_ffn["elements"], m_ffn["first_divergent_token"])
    )

    # ------------------------------------------- add-semantics candidates
    resid_oracle = load_mat(DUAL_DIR, "logical_00_full.bin", dual_man)  # == HF oracle
    cands = {
        "S1_bf16_of_f32cppout_plus_f32resid": to_bf16(
            cpp_attn_out.astype(np.float32) + resid_oracle.astype(np.float32)
        ),
        "S2_bf16_of_bf16cppout_plus_bf16resid": to_bf16(
            to_bf16(cpp_attn_out) + to_bf16(resid_oracle)
        ),
        "S3_bf16_of_f32hfout_plus_f32resid": to_bf16(
            hf_attn_out.astype(np.float32) + resid_oracle.astype(np.float32)
        ),
    }
    closed = None
    for name, cand in cands.items():
        eq = int((cand == hf_resid).sum())
        report["add_semantics_candidates"][name] = {
            "exact_vs_hf_attn0_resid": eq,
            "elements": int(cand.size),
            "byte_exact": bool(eq == cand.size),
        }
        if eq == cand.size and closed is None:
            closed = name
        print("candidate %s: %d/%d%s" % (name, eq, cand.size, "  BYTE-EXACT" if eq == cand.size else ""))
    report["add_semantics_candidates"]["closure"] = (
        ("CLOSED by %s" % closed) if closed else
        "NOT closed - no candidate reproduces the HF boundary byte-exactly; results are non-closing evidence"
    )

    # ------------------------------------------- downstream descriptive
    for label, dual_name, hf_name in (
        ("mlp0_resid", "block1_mlp0_resid_full.bin", "mlp0_resid.bin"),
        ("attn1_norm", "block1_attn1_norm_full.bin", "attn1_norm.bin"),
        ("attn1_resid", "block1_attn1_resid_full.bin", "attn1_resid.bin"),
        ("logical_01", "logical_01_full.bin", "layer_out.bin"),
    ):
        hf_m = load_mat(HF_DIR, hf_name, hf_man)
        dual_m = load_mat(DUAL_DIR, dual_name, dual_man)
        single_m = load_mat(SINGLE_DIR, dual_name, single_man)
        e_dual = float(np.sqrt(((dual_m.astype(np.float64) - hf_m.astype(np.float64)) ** 2).sum()))
        e_single = float(np.sqrt(((single_m.astype(np.float64) - hf_m.astype(np.float64)) ** 2).sum()))
        report["downstream_descriptive"][label] = {
            "dual_reset_err_l2": e_dual,
            "single_reset_err_l2": e_single,
            "ratio_dual_over_single": e_dual / e_single if e_single > 0 else float("nan"),
            "note": "descriptive only; no attribution (predecessors differ)",
        }
        print("downstream %-11s dual L2 %.5f single L2 %.5f ratio %.4f" % (
            label, e_dual, e_single,
            e_dual / e_single if e_single > 0 else float("nan")))

    # ------------------------------------------- grid verdict
    out_cls = m_out["classification"]
    ffn_cls = m_ffn["classification"]
    verdict = {
        "attn_out_2_classification": out_cls,
        "ffn_inp_2_classification": ffn_cls,
        "f32_carrier_note": SEMANTIC_EQUIVALENCE["f32_carrier_note"],
    }
    if out_cls == "raw-exact":
        verdict["grid_case"] = 1
        verdict["reading"] = (
            "attention output exact at this boundary; any ffn_inp-2 mismatch "
            "belongs to residual-add/storage semantics"
        )
    elif out_cls == "bf16-reducible":
        verdict["grid_case"] = 4 if ffn_cls == "bf16-irreducible" else 2
        verdict["reading"] = (
            "attention output is BF16-equivalent, NOT value-exact"
            + (
                "; ffn_inp-2 irreducible -> first irreducibility localizes to "
                "the residual-add/carrier boundary, not attention internals"
                if ffn_cls == "bf16-irreducible" else
                "; inspect add preserve/amplify via the candidates"
            )
        )
    else:
        verdict["grid_case"] = 3
        verdict["reading"] = (
            "block-2 attention is causally implicated under value-exact "
            "inputs (including dtype/kernel semantics per the F32-carrier "
            "note); design the deeper block-2 MLA-internal capture"
        )
    # Next-branch predecessor-exactness (cases 1-2 only).
    if verdict["grid_case"] in (1, 2):
        if ffn_cls == "raw-exact":
            verdict["next_branch"] = (
                "post-attention-norm/MLP path design licensed: ffn_inp-2 is "
                "raw byte-exact, so the next operator's input is byte-exact"
            )
        else:
            verdict["next_branch"] = (
                "STOP FOR REVIEW: ffn_inp-2 is not raw byte-exact (BF16-"
                "equivalence is not byte-exact input), so the next operator "
                "consumes a different F32 predecessor; designed next step = "
                "narrowest exact ffn_inp-2 predecessor reset before judging "
                "post-attention-norm/MLP"
            )
    elif verdict["grid_case"] == 3:
        verdict["next_branch"] = (
            "design (only) the narrowest block-2 MLA-internal capture "
            "mirroring the proven block-0 methodology: full-seq il=2 stage "
            "dumps (q_a_proj-2, q_a_norm-2, q_b_proj-2, kv_cmpr_pe-2, "
            "kv_a_norm-2, kv_cmpr_scaled-2, q_pe_rope-2, k_pe_rope-2, "
            "kqv_out-2) + an HF layer-1 self_attn[0] internals capture"
        )
    else:
        verdict["next_branch"] = (
            "design (only) the residual-add/carrier boundary experiment"
        )
    report["verdict"] = verdict
    print("grid case %s: %s" % (verdict["grid_case"], verdict["reading"]))
    print("next branch: %s" % verdict["next_branch"])

    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / "block2_attn_reset.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT_DIR / "SHA256SUMS.txt").write_text(
        "%s  block2_attn_reset.json\n" % json_sha, encoding="utf-8"
    )
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("BLOCK2 ATTENTION RESET JUDGMENT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
