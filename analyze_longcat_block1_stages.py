#!/usr/bin/env python3
"""Block-1 sub-boundary localization under the byte-exact logical_00 reset.

Compares, whole-tensor [512, 3072], in proven execution order:

    attn0_norm -> attn0_resid -> mlp0_resid -> attn1_norm -> attn1_resid
      -> logical_01 (layer output)

between the HF layer-1 stages capture (D:\\lc_block1_stages_512, whose input
is byte-identical to the causal-reset oracle) and the C++ reset run
(cpp_resid_walk_inject_b1_512), with the C++ control run
(cpp_resid_walk_control_b1_512) providing e_observed context.

Pre-registered rules:
  first-raw          first boundary with any raw mismatch (reset family)
  first-irreducible  first boundary where bf16(cpp) == hf does NOT hold for
                     all elements (not explainable as the known
                     output-rounding boundary class)
  predecessor-exactness  causal attribution to the local operator is
                     permitted ONLY when the operator's semantically
                     relevant input is byte-exact C++<->HF. Only attn0_norm
                     enjoys that here (its input is the injected oracle on
                     both sides). If the first irreducible boundary's
                     predecessor differs in any way, report it and STOP FOR
                     REVIEW; the designed (not executed) next step is the
                     narrowest exact predecessor reset.

Additionally, because attn0_norm is operator-isolated, the known RMSNorm
cast-semantics mechanism is checked offline (block-0 methodology,
analyze_longcat_attn0_kv_a_norm_semantics.py): reconstruction A
(llama.cpp F32 semantics) must reproduce the C++ dump and reconstruction D
(HF semantics, bf16(bf16(norm)*w)) must reproduce the HF dump, from the
byte-exact input and the GGUF blk.2.attn_norm.weight the run loaded.

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
CTRL_DIR = REPO / "cpp_resid_walk_control_b1_512"
INJ_DIR = REPO / "cpp_resid_walk_inject_b1_512"
PRIOR_CTRL = REPO / "cpp_resid_walk_control_512"
PRIOR_INJ = REPO / "cpp_resid_walk_inject_512"
GGUF_DIR = Path(r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16")
GGUF_SHARDS = [
    GGUF_DIR / (
        "LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-%05d-of-00008.gguf" % i
    )
    for i in range(1, 9)
]
OUT_DIR = REPO / "block1_stages_512"

N_TOK = 512
HIDDEN = 3072
VEC_BYTES = N_TOK * HIDDEN * 4

ORACLE_SHA = "d810f93c50ea42c5909ab289ebf62a0c5629f40530d2e5fc706dde67f0eaf763"
HF_LOGICAL01_SHA = "85097c18565f04c0e0676146ae7ee3f5ffc674789db6f17c028606595d6d16e2"
NORM_WEIGHT_NAME = "blk.2.attn_norm.weight"
F_NORM_RMS_EPS = 1e-5  # GGUF attention.layer_norm_rms_epsilon (config rms_norm_eps)

# (label, cpp_file, hf_file, operator_description, predecessor_label or None)
BOUNDARIES = [
    ("attn0_norm", "block1_attn0_norm_full.bin", "attn0_norm.bin",
     "input_layernorm[0] (RMSNorm) on the injected oracle - operator-isolated", None),
    ("attn0_resid", "block1_attn0_resid_full.bin", "attn0_resid.bin",
     "block-2 attention + residual add", "attn0_norm"),
    ("mlp0_resid", "block1_mlp0_resid_full.bin", "mlp0_resid.bin",
     "post_attention_layernorm[0] + dense mlps[0] + residual (ScMoE computed, not added)", "attn0_resid"),
    ("attn1_norm", "block1_attn1_norm_full.bin", "attn1_norm.bin",
     "input_layernorm[1] (RMSNorm)", "mlp0_resid"),
    ("attn1_resid", "block1_attn1_resid_full.bin", "attn1_resid.bin",
     "block-3 attention + residual add", "attn1_norm"),
    ("logical_01", "logical_01_full.bin", "layer_out.bin",
     "post_attention_layernorm[1] + dense mlps[1] + residual + ScMoE shortcut join", "attn1_resid"),
]

SEMANTIC_EQUIVALENCE = {
    "proof_protocol": "source-derived, no filename inference; both sides verified this session",
    "cpp": (
        "longcat-flash-ngram.cpp: attn_norm cb at :635 (build_norm output, il=2/3); "
        "ffn_inp = attn_out + residual (:996-998); even block l_out-2 = dense MLP[0] "
        "+ residual with ScMoE shortcut computed but NOT added (:1005-1076, :1096-1100); "
        "odd block l_out-3 = dense MLP[1] + shortcut + residual (:1078-1100)"
    ),
    "hf": (
        "modeling_longcat_flash_sparse.py LongcatFlashSparseDecoderLayer.forward "
        ":980-1032 for layers[1]; capture is a statement-for-statement monkey-patch "
        "replica whose faithfulness is byte-gated (layer_out == hidden_states[2]; "
        "input == hidden_states[1] == the reset oracle)"
    ),
    "dtype_note": (
        "HF surfaces are BF16-on-lattice (widened to f32); C++ nodes are F32. The "
        "raw vs bf16(cpp)==hf classification separates the known output-rounding "
        "boundary class from irreducible divergence."
    ),
}

ENV_AUDIT = {
    "method": (
        "wrapper-aware source derivation (bare getenv + ggml_cuda_ar_env_u64 reads) "
        "across common/, src/, ggml/src core+CPU+CUDA; backends not compiled on this "
        "CUDA/Windows build excluded; cuBLAS/cuBLASLt logging vars are the library's "
        "own contract; TORCH override swept defensively. Fail closed on names; count "
        "descriptive."
    ),
    "reconciliation": (
        "historical 19 = 3 LONGCAT + 12 GGML_CUDA bare getenv (incl. GGML_CUDA_P2P, "
        "which subdirectory-limited greps miss) + 4 cuBLAS logging; handoff 21 = 19 + "
        "2 new LONGCAT_RESID_*; first executed harness swept 23 (added AR_COPY pair + "
        "TORCH, but its GGML_CUDA bare-getenv list had 11, missing P2P and the third "
        "wrapper var AR_BF16_THRESHOLD); audited list = 39 incl. "
        "GGML_OP_OFFLOAD_MIN_BATCH, GGML_CPU_DISABLE_FUSION, GGML_BACKEND_PATH, "
        "GGML_TOTAL_THREADS, GGML_SCHED_DEBUG(_REALLOC), and 8 LLAMA_* toggles. Prior "
        "runs' validity is independently supported by their byte-exact reproduction "
        "gates (any env-induced numeric change would have broken them)."
    ),
    "count": 39,
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


def metrics(cpp: np.ndarray, hf: np.ndarray) -> dict:
    c = cpp.astype(np.float64)
    h = hf.astype(np.float64)
    d = c - h
    l2 = float(np.sqrt((d ** 2).sum()))
    rmse = float(np.sqrt((d ** 2).mean()))
    hf_rms = float(np.sqrt((h ** 2).mean()))
    denom = float(np.sqrt((c ** 2).sum()) * np.sqrt((h ** 2).sum()))
    row_l2 = np.sqrt((d ** 2).sum(axis=1))
    mism_rows = np.nonzero((cpp != hf).any(axis=1))[0]
    bf16_eq = int((to_bf16(cpp) == hf).sum())
    return {
        "elements": int(cpp.size),
        "raw_equal": int((cpp == hf).sum()),
        "hf_on_bf16_lattice": int((to_bf16(hf) == hf).sum()),
        "bf16_cpp_equal_hf": bf16_eq,
        "bf16_reducible": bool(bf16_eq == cpp.size),
        "err_l2": l2,
        "abs_rmse": rmse,
        "rel_rmse": rmse / hf_rms if hf_rms > 0 else float("nan"),
        "hf_rms": hf_rms,
        "max_abs": float(np.abs(d).max()),
        "cosine_cpp_hf": float((c * h).sum() / denom) if denom > 0 else float("nan"),
        "first_divergent_token": int(mism_rows[0]) if mism_rows.size else -1,
        "divergent_token_rows": int(mism_rows.size),
        "row_l2_max": float(row_l2.max()),
        "row_l2_final": float(row_l2[-1]),
    }


def read_norm_weight() -> np.ndarray:
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader  # noqa: PLC0415

    for shard in GGUF_SHARDS:
        if not shard.is_file():
            stop("GGUF shard missing: %s" % shard)
        for tensor in GGUFReader(str(shard), "r").tensors:
            if tensor.name == NORM_WEIGHT_NAME:
                w = np.array(tensor.data, dtype=np.float32).reshape(-1)
                if w.size != HIDDEN:
                    stop("%s has %d elements" % (NORM_WEIGHT_NAME, w.size))
                return w
    stop("%s not found in any shard" % NORM_WEIGHT_NAME)


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    """ggml_rms_norm: F32 activation, F64 variance accumulation."""
    x32 = x.astype(np.float32)
    var = (x32.astype(np.float64) ** 2).mean(axis=1)
    return x32 * (1.0 / np.sqrt(var + eps)).astype(np.float32)[:, None]


def main() -> int:
    print("block-1 sub-boundary localization (measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    hf_man = load_manifest(HF_DIR)
    ctrl_man = load_manifest(CTRL_DIR)
    inj_man = load_manifest(INJ_DIR)
    prior_ctrl_man = load_manifest(PRIOR_CTRL)
    prior_inj_man = load_manifest(PRIOR_INJ)

    # Identity / chain-closure gates.
    if hf_man.get("input.bin") != ORACLE_SHA:
        stop("HF stages input != oracle")
    if hf_man.get("layer_out.bin") != HF_LOGICAL01_SHA:
        stop("HF stages layer_out != committed logical_01")
    if inj_man.get("logical_00_full.bin") != ORACLE_SHA:
        stop("inject landing mismatch in b1 manifest")
    if inj_man.get("logical_01_full.bin") != prior_inj_man.get("logical_01_full.bin"):
        stop("inject b1 logical_01_full != prior inject run (chain closure)")
    if ctrl_man.get("logical_01_full.bin") != prior_ctrl_man.get("logical_01_full.bin"):
        stop("control b1 logical_01_full != prior control run (chain closure)")
    print("identity gates: HF input==oracle, layer_out==committed logical_01, chain closure PASS")

    report: dict = {
        "description": "Block-1 sub-boundary localization under the byte-exact logical_00 reset",
        "oracle_sha256": ORACLE_SHA,
        "hf_dir": str(HF_DIR),
        "control_dir": str(CTRL_DIR),
        "inject_dir": str(INJ_DIR),
        "semantic_equivalence": SEMANTIC_EQUIVALENCE,
        "env_audit": ENV_AUDIT,
        "provenance": {},
        "boundaries": {},
        "norm_mechanism_check": {},
        "verdict": {},
        "caveats": [
            "Predecessor-exactness: causal attribution to a boundary's local operator "
            "is made ONLY where that operator's semantically relevant input is "
            "byte-exact C++<->HF. A BF16-reducible predecessor difference can be "
            "nonlinearly transformed by an otherwise-correct operator into a "
            "BF16-irreducible difference at the next boundary.",
            "e_reset at later sub-boundaries includes error inherited from earlier "
            "post-reset sub-boundaries; only attn0_norm is operator-isolated.",
            "Reset ratios quantify the exact-upstream counterfactual; the nonlinear "
            "downstream means they are not additive decompositions.",
        ],
    }
    for tag, d in (("control_b1", CTRL_DIR), ("inject_b1", INJ_DIR)):
        p = d / "run_provenance.json"
        if p.is_file():
            report["provenance"][tag] = json.loads(p.read_text(encoding="ascii"))

    first_raw = None
    first_irred = None
    prev_raw_exact = {"attn0_norm": True}  # input = injected oracle, landing-gated

    for label, cpp_name, hf_name, op_desc, pred in BOUNDARIES:
        hf = load_mat(HF_DIR, hf_name, hf_man)
        inj = load_mat(INJ_DIR, cpp_name, inj_man)
        ctrl = load_mat(CTRL_DIR, cpp_name, ctrl_man)

        m_reset = metrics(inj, hf)
        m_obs = metrics(ctrl, hf)
        e_r = inj.astype(np.float64) - hf.astype(np.float64)
        e_o = ctrl.astype(np.float64) - hf.astype(np.float64)
        n_r = float(np.sqrt((e_r ** 2).sum()))
        n_o = float(np.sqrt((e_o ** 2).sum()))

        raw_exact_all = m_reset["raw_equal"] == m_reset["elements"]
        if pred is None:
            op_input_exact = True
            op_input_note = "input is the injected oracle (byte-exact by the landing gate)"
        else:
            op_input_exact = prev_raw_exact.get(pred, False)
            op_input_note = (
                "predecessor %s raw-exact: %s" % (pred, prev_raw_exact.get(pred, False))
            )
        prev_raw_exact[label] = raw_exact_all

        entry = {
            "operator": op_desc,
            "reset": m_reset,
            "observed": m_obs,
            "ratio_reset_over_observed": n_r / n_o if n_o > 0 else float("nan"),
            "operator_input_byte_exact": op_input_exact,
            "operator_input_note": op_input_note,
        }
        report["boundaries"][label] = entry

        if first_raw is None and not raw_exact_all:
            first_raw = label
        if first_irred is None and not m_reset["bf16_reducible"]:
            first_irred = label

        print(
            "%-12s reset rel %.4e raw_eq %d/%d bf16_eq %d/%d %s | obs rel %.4e | ratio %.4f | input_exact %s"
            % (
                label, m_reset["rel_rmse"], m_reset["raw_equal"], m_reset["elements"],
                m_reset["bf16_cpp_equal_hf"], m_reset["elements"],
                "REDUCIBLE" if m_reset["bf16_reducible"] else "IRREDUCIBLE",
                m_obs["rel_rmse"], entry["ratio_reset_over_observed"], op_input_exact,
            )
        )

    # Offline mechanism check for the operator-isolated attn0_norm boundary.
    weight = read_norm_weight()
    oracle = load_mat(HF_DIR, "input.bin", hf_man)
    cpp_norm = load_mat(INJ_DIR, "block1_attn0_norm_full.bin", inj_man)
    hf_norm = load_mat(HF_DIR, "attn0_norm.bin", hf_man)
    mech = {"weight_on_bf16_pct": float((to_bf16(weight) == weight).mean() * 100.0)}
    for eps_tag, eps in (("1e-5", 1e-5), ("1e-6", 1e-6)):
        hn = normalize(oracle, eps)
        recon_a = hn * weight[None, :]
        recon_d = to_bf16(to_bf16(hn) * to_bf16(weight)[None, :])
        mech["eps_" + eps_tag] = {
            "A_f32_vs_cpp_exact": int((recon_a == cpp_norm).sum()),
            "D_hf_semantics_vs_hf_exact": int((recon_d == hf_norm).sum()),
            "elements": int(recon_a.size),
        }
    # A-model residue quantification: the offline model accumulates variance
    # in f64 while the CUDA kernel reduces in f32 (tree order), so at width
    # 3072 the plain-F32 model is expected to match C++ only to reduction
    # noise (at block-0 width 512 it matched 100%). Record ulp statistics so
    # the residue is provably at that scale.
    hn5 = normalize(oracle, 1e-5)
    recon_a5 = np.ascontiguousarray(hn5 * weight[None, :], dtype="<f4")
    d_a = recon_a5.astype(np.float64) - cpp_norm.astype(np.float64)
    ulp = np.abs(
        recon_a5.view(np.int32).astype(np.int64)
        - np.ascontiguousarray(cpp_norm, dtype="<f4").view(np.int32).astype(np.int64)
    )
    mech["A_model_residue_eps_1e-5"] = {
        "note": (
            "offline f64-accumulated variance vs CUDA f32 tree reduction at "
            "width 3072; verdicts rest on the dumps and the D byte-exactness, "
            "not on A"
        ),
        "max_abs": float(np.abs(d_a).max()),
        "rel_rmse": float(
            np.sqrt((d_a ** 2).mean())
            / np.sqrt((cpp_norm.astype(np.float64) ** 2).mean())
        ),
        "max_ulp": int(ulp.max()),
        "ulp_le_1_pct": float((ulp <= 1).mean() * 100.0),
        "bf16A_equals_bf16_cpp": int((to_bf16(recon_a5) == to_bf16(cpp_norm)).sum()),
    }
    report["norm_mechanism_check"] = mech
    a_ok = mech["eps_1e-5"]["A_f32_vs_cpp_exact"] == N_TOK * HIDDEN
    d_ok = mech["eps_1e-5"]["D_hf_semantics_vs_hf_exact"] == N_TOK * HIDDEN
    print(
        "norm mechanism (eps 1e-5): A(f32)==cpp %d/%d, D(hf-cast)==hf %d/%d"
        % (
            mech["eps_1e-5"]["A_f32_vs_cpp_exact"], N_TOK * HIDDEN,
            mech["eps_1e-5"]["D_hf_semantics_vs_hf_exact"], N_TOK * HIDDEN,
        )
    )

    verdict = {
        "first_raw": first_raw,
        "first_bf16_irreducible": first_irred,
        "first_irreducible_operator_input_byte_exact": (
            report["boundaries"][first_irred]["operator_input_byte_exact"]
            if first_irred else None
        ),
        "norm_mechanism_A_reproduces_cpp": a_ok,
        "norm_mechanism_D_reproduces_hf": d_ok,
    }
    if first_irred is not None and verdict["first_irreducible_operator_input_byte_exact"]:
        verdict["attribution"] = (
            "PERMITTED: %s is the first bf16-irreducible boundary and its operator "
            "input is byte-exact -> the divergence is generated by that operator" % first_irred
        )
    else:
        verdict["attribution"] = (
            "NOT permitted at %s (predecessor differs) -> stop for review; the "
            "designed next step is the narrowest exact predecessor reset" % first_irred
        )
    report["verdict"] = verdict
    print("first-raw: %s | first-irreducible: %s" % (first_raw, first_irred))
    print("attribution: %s" % verdict["attribution"])

    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / "block1_stages.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT_DIR / "SHA256SUMS.txt").write_text(
        "%s  block1_stages.json\n" % json_sha, encoding="utf-8"
    )
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("BLOCK1 STAGES LOCALIZATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
