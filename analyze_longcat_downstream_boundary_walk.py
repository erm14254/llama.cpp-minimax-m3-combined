#!/usr/bin/env python3
"""Zero-new-run downstream boundary walk for the frozen-512 residual gap.

Measurement-only. Compares the 18 final-row [3072] boundary dumps of the
production-angle A+B C++ capture (cpp_attn0_mla_attnpath_512/, byte-identical
to cpp_attn0_mla_expB_512/ on every surface used here) against the pre-gate4
HF oracles (hf_logical0_stages_512_v4/, hf_hidden_512_v4/), in the semantic
execution order proven from source (recorded in the output JSON), and
decomposes the error vector at each consecutive residual-stream boundary into
magnitude growth, direction persistence, and orthogonal (new-direction)
components.

Fail-closed: every input file is SHA256-gated against the committed capture
manifest / sidecar values before any comparison; known-answer anchors
(attn0_resid rel-RMSE, frozen-logits endpoint metrics) must reproduce the
committed record or the run aborts.

No C++ runs, no GPU, no arithmetic changes anywhere.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
CPP_DIR = REPO / "cpp_attn0_mla_attnpath_512"
HF_STAGES = Path(r"D:\llama.cpp-longcat-pre-gate4\hf_logical0_stages_512_v4")
HF_HIDDEN = Path(r"D:\llama.cpp-longcat-pre-gate4\hf_hidden_512_v4")
CPP_LOGITS = (
    REPO
    / "cpp_logits_512_postAB"
    / "llamacpp-LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-BF16-00001-of-00008.bin"
)
HF_LOGITS = Path(r"D:\llama.cpp-longcat-pre-gate4\hf_sparse_512_v4.bin")
OUT_DIR = REPO / "downstream_walk_512"

HIDDEN = 3072
VOCAB = 131072
VEC_BYTES = HIDDEN * 4

# --------------------------------------------------------------------------
# Frozen expectations. C++ hashes: committed cpp_attn0_mla_attnpath_512/
# SHA256SUMS.txt (byte-identical to cpp_attn0_mla_expB_512/ per the Class-2
# gate). HF hashes: the two capture sidecar summary.json files, whose
# per-surface values match WIN11_HANDOFF_2026-08-17_FROZEN512.md.
# --------------------------------------------------------------------------

CPP_SHA = {
    "inp_embd_ngram": "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f",
    "logical0_attn0_resid": "7e05940b5c1b6b8f3bcaf210ac8938a44ea3006052c1f16e20c855df6452f109",
    "logical0_mlp0_resid": "de18420a5b2e0d4e2575d1f597cc67c01ccd6e1b17c03dccfd5a20d687b31cb7",
    "logical0_attn1_resid": "af49add8343451d0f9379dd874a5cd9f59cbe43f95179a7a4fca69cce3da0e12",
    "logical_00": "fa813b529fba809778497da7b43a5c5dc653dcb5906f9078fa08e8c9b35f1e3b",
    "logical_01": "f6e9e0685a7c7b45f1f85521b641f3dee3b6febeb87963a428fb78a07f5411c0",
    "logical_02": "55e28ae734163814c28a2c592725a86ea6a357d71cfde2cb98aad5144bd74887",
    "logical_03": "a45e1965854f2120228f0712957db0aa047cf70a3b022afba24d713a75681844",
    "logical_04": "f6ea6919e819f67f9f4cd7ff66d3d49d001b8897cb2d0584d98835a6b8d4195d",
    "logical_05": "850956c621cdf86901b087b4f078640bdcdce043b3026608467f6d15922f33d2",
    "logical_06": "f25ed6547a455bdbb62eee0cc265269107c47cc2aa40345ca687bcaad42154f2",
    "logical_07": "43c9475a5d5bedfe7675c2d36d3d583b38ff5dee6ff188bb1bf1742437c75f3b",
    "logical_08": "02bb900289a8d58572b14aa2c4daf75d7703045bb9111b3aa496d050e42d05b2",
    "logical_09": "e1dff304465998848ae5528699f6ad6d7c362e1ac49fca3709c001744f756482",
    "logical_10": "6c9bc832266c587ceb7868cafcd7f02b57201327615034407177234a0afb1f44",
    "logical_11": "570ce9efdf243695440b739c1a6b3902395c760c375753ce6455d571521e19eb",
    "logical_12": "6cab13fd58a75683fd86b783440d21d7b4adfb55ffa4fe61ca203a9c205b79e0",
    "result_norm": "cdca61ccf103d19ec064759970f2e2f84b725a1eb52e713ae73144c505cfead3",
}

HF_STAGES_SHA = {
    "input": "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f",
    "attn0_resid": "2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177",
    "mlp0_resid": "cf48a0ad3001e82ae41020675458df66219ea929caa1168ffddf64196d70404f",
    "attn1_resid": "b4c1e5f684afefcec4129e3e6ec095a38d9b7f880115f819f78f8a698fe14431",
    "logical0_out": "5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff",
}

HF_HIDDEN_SHA = {
    "inp_embd_ngram": "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f",
    "logical_00": "5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff",
    "logical_01": "48d23a758c0a7e806c2521560bf0b6245f3e6031039f53c7b0a4169c5fc426a2",
    "logical_02": "e8bf6f3a9019fff470fe57bb0e0fde1377d366a14e9a6855b46f3ef2534ae895",
    "logical_03": "fd40a04bb0b2f8cd17193cb04f75737459435915203962b1f8d764906d05a785",
    "logical_04": "f61901209f3884d5db7dda5204b41d17de7760148ab6318a4f3af282085a70b9",
    "logical_05": "5fec605d82963e8f7859a92c377036b7e9ff8747927e0f1080c3dada0cec0400",
    "logical_06": "2bdde1a5e89c25ec87d926d8ef96e2b5e5ba13b3e5a8bcfccef497479e18924a",
    "logical_07": "3c7c09108205b28a55aeded4cae721f0448803d4c2be5033af026998e9ce4e43",
    "logical_08": "b1f934886eb95ffd342064f5f10008e878d7abce918b9a0b62f2c31ff7d37350",
    "logical_09": "8a4d375eae758d9fd786f122c7d90371f0836d052a37ed1e4bdc05a95a2577cb",
    "logical_10": "b2f160a859371974a694fd6ead8971ce605fe2ae753868e798bb279bffa3ec39",
    "logical_11": "17bd4cbc8ecba464fe06822d0f71986b25373f4cfae5f76c93f69e0619b96ed4",
    "logical_12": "45a7656f15350f2f85b611f4768828db1d77ecbaf7aa42ff839600ab99e843cc",
    "result_norm": "caac1af10e7d445729e84379a39bd7ee47e10a69525575712ab78050fa9a533a",
}

CPP_LOGITS_SHA = "e14d95bfaaa0fea2977ed4ac852b7a631427e27f79eeb17c04a5a70c824660df"
HF_LOGITS_SHA = "8825d92d7d9cdea42a4ea3aa2e3df5766bdf880323b1f48ea8c17ff63f3c5ecf"

# Known-answer anchors (committed record; the walk aborts if not reproduced).
KNOWN_ATTN0_REL_RMSE = 0.00390108      # STATUS_2026-08-17.md, Exp-B addendum
KNOWN_ATTN0_TOL = 2e-8
KNOWN_LOGITS = {                       # STATUS_2026-08-17.md, frozen-512 addendum
    "violations": 40,
    "rmse": 0.164743901,
    "cosine": 0.999799076,
    "max_abs": 0.851543427,
    "top1": 483,
}
KNOWN_LOGITS_TOL = 1e-8

# --------------------------------------------------------------------------
# Semantic execution order, proven from source (not from filenames).
# All references verified in this session at repo HEAD 98429981e; the HF
# modeling file is the frozen runtime (SHA a3bc3161..., re-hashed).
# --------------------------------------------------------------------------

SEMANTIC_ORDER_EVIDENCE = {
    "hf_runtime": {
        "file": r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved\modeling_longcat_flash_sparse.py",
        "sha256": "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428",
        "facts": [
            "config.json: num_layers=14 logical layers, num_hidden_layers=2*14=28 (modeling line 1400)",
            "trunk forward loops layers[:num_layers] appending hidden_states BEFORE each layer (lines 1472-1474): hidden_states[k] = input to logical layer k = output of layer k-1; index 0 = ngram_embeddings output (lines 1437-1440)",
            "after the loop: hidden_states = self.norm(hidden_states); appended last (lines 1487-1489) -> index 14 = post-final-norm",
            "MTP is a separate module (LongcatFlashSparseMTP, line 1226+), not executed in the trunk forward",
        ],
    },
    "hf_capture_scripts": {
        "stages": r"D:\llama.cpp-longcat-pre-gate4\capture_longcat_sparse_hf_512_logical0_stages.py",
        "stages_facts": [
            "monkey-patched layer0 forward snaps final-row states in explicit order: input (line 207) -> attn0_resid after residual+self_attn[0] (lines 209-228) -> mlp0_resid after residual+mlps[0], ScMoE shortcut computed from the same post-attn norm but NOT added (lines 230-246) -> attn1_resid after residual+self_attn[1] with topk_indices reused from attn[0] (LSA; lines 248-268) -> logical0_out = residual+mlps[1]+shortcut_mlp_output (lines 270-286)",
            "snap keeps tensor[0, -1] only: final prompt token (lines 189-194)",
        ],
        "hidden": r"D:\llama.cpp-longcat-pre-gate4\capture_longcat_sparse_hf_512_hidden.py",
        "hidden_facts": [
            "asserts 15 hidden states; names indices 0..14 = inp_embd_ngram, logical_00..logical_12, result_norm (lines 136-140)",
            "therefore logical_NN = output of logical layer N (N=0..12) and result_norm = norm(output of logical layer 13)",
            "final row only, F32-widened (lines 160-169); runtime and token stream SHA-gated (lines 11-12, 57-60); TF32 disabled (line 88); bf16 weights on cuda:0; use_cache=False",
        ],
    },
    "cpp_sources": {
        "debug_cpp": "common/debug.cpp lines 240-286 (repo HEAD): final-row [3072] dump specs: inp_embd_ngram; ffn_inp-0 -> logical0_attn0_resid.bin; l_out-0 -> logical0_mlp0_resid.bin; ffn_inp-1 -> logical0_attn1_resid.bin; l_out-(2N+1) -> logical_NN.bin for N=0..12; result_norm",
        "model_cpp": "src/models/longcat-flash-ngram.cpp: main loop il=0..n_layer-1 with is_even_block = il%2==0 (lines 594-597); ffn_inp = attn_out + residual (996-998); even block computes MoE shortcut but does NOT add it (1005-1066); odd block adds dense MLP[1] then the paired shortcut (1078-1092) then residual (1096-1100) -> l_out; result_norm = build_norm(output_norm) on the last main block output (1105-1117); the nextn/MTP block is a separate graph_mtp at layer index n_layer with n_layer_nextn==1 asserted (1128-1138), outside the trunk",
        "mapping": "one logical layer = two physical blocks: logical N = physical {2N, 2N+1}; l_out-(2N+1) = logical layer N output, matching HF hidden_states[N+1]",
    },
    "gap_note": (
        "Between logical_12 (output of logical layer 12 = l_out-25) and "
        "result_norm lies the entire UNDUMPED logical layer 13 (physical "
        "blocks 26-27: attn+denseMLP+attn+denseMLP+ScMoE) plus the final "
        "RMSNorm, on BOTH sides. The logical_12 -> result_norm step therefore "
        "spans one full logical layer and the norm; no dumped boundary exists "
        "inside it in any existing capture."
    ),
}

# Walk chain: (label, cpp_name, hf_dir_tag, hf_name, residual_stream_member)
CHAIN = (
    [
        ("inp_embd_ngram", "inp_embd_ngram", "hidden", "inp_embd_ngram", True),
        ("attn0_resid", "logical0_attn0_resid", "stages", "attn0_resid", True),
        ("mlp0_resid", "logical0_mlp0_resid", "stages", "mlp0_resid", True),
        ("attn1_resid", "logical0_attn1_resid", "stages", "attn1_resid", True),
        ("logical_00", "logical_00", "stages", "logical0_out", True),
    ]
    + [(f"logical_{i:02d}", f"logical_{i:02d}", "hidden", f"logical_{i:02d}", True) for i in range(1, 13)]
    + [("result_norm", "result_norm", "hidden", "result_norm", False)]
)


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


def load_vec(path: Path, expect_sha: str, n: int) -> np.ndarray:
    if not path.is_file():
        stop("missing input: %s" % path)
    got = sha256_file(path)
    if got != expect_sha:
        stop("SHA mismatch for %s: got %s expected %s" % (path, got, expect_sha))
    raw = path.read_bytes()
    if len(raw) != n * 4:
        stop("size mismatch for %s: %d != %d" % (path, len(raw), n * 4))
    v = np.frombuffer(raw, dtype="<f4").copy()
    if v.size != n:
        stop("element-count mismatch for %s" % path)
    if not np.isfinite(v).all():
        stop("non-finite values in %s" % path)
    return v


def pair_metrics(cpp: np.ndarray, hf: np.ndarray) -> dict:
    d = cpp.astype(np.float64) - hf.astype(np.float64)
    hf64 = hf.astype(np.float64)
    cpp64 = cpp.astype(np.float64)
    rmse = float(np.sqrt((d ** 2).mean()))
    hf_rms = float(np.sqrt((hf64 ** 2).mean()))
    err_l2 = float(np.sqrt((d ** 2).sum()))
    denom = float(np.sqrt((cpp64 ** 2).sum()) * np.sqrt((hf64 ** 2).sum()))
    cosine = float((cpp64 * hf64).sum() / denom) if denom > 0 else float("nan")
    amax = int(np.argmax(np.abs(d)))
    return {
        "elements": int(cpp.size),
        "raw_equal": int((cpp == hf).sum()),
        "hf_on_bf16_lattice": int((to_bf16(hf) == hf).sum()),
        "bf16_cpp_equal_hf": int((to_bf16(cpp) == hf).sum()),
        "rel_rmse": rmse / hf_rms if hf_rms > 0 else float("nan"),
        "abs_rmse": rmse,
        "err_l2": err_l2,
        "hf_rms": hf_rms,
        "max_abs": float(np.abs(d).max()),
        "argmax": amax,
        "cosine_cpp_hf": cosine,
    }


def decomp_step(e_prev: np.ndarray, e_cur: np.ndarray) -> dict:
    p = e_prev.astype(np.float64)
    c = e_cur.astype(np.float64)
    np_l2 = float(np.sqrt((p ** 2).sum()))
    nc_l2 = float(np.sqrt((c ** 2).sum()))
    if np_l2 == 0.0 or nc_l2 == 0.0:
        return {
            "prev_l2": np_l2,
            "cur_l2": nc_l2,
            "ratio": float("inf") if np_l2 == 0.0 and nc_l2 > 0.0 else float("nan"),
            "cos_prev_cur": float("nan"),
            "proj_coeff": float("nan"),
            "orth_l2": nc_l2,
            "orth_frac": float("nan"),
        }
    dot = float((p * c).sum())
    proj_coeff = dot / np_l2                     # signed length of e_cur along e_prev-hat
    orth = c - (dot / (np_l2 ** 2)) * p
    orth_l2 = float(np.sqrt((orth ** 2).sum()))
    return {
        "prev_l2": np_l2,
        "cur_l2": nc_l2,
        "ratio": nc_l2 / np_l2,
        "cos_prev_cur": dot / (np_l2 * nc_l2),
        "proj_coeff": proj_coeff,
        "orth_l2": orth_l2,
        "orth_frac": orth_l2 / nc_l2,
    }


def main() -> int:
    print("downstream boundary walk (zero-new-run, measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    report: dict = {
        "description": "Zero-new-run downstream boundary walk, frozen-512 residual gap localization (measurement-only)",
        "cpp_capture": str(CPP_DIR),
        "cpp_capture_note": (
            "production-angle A+B state; byte-identical to cpp_attn0_mla_expB_512 on all "
            "surfaces used here (committed Class-2 gate); representative of the frozen-512 "
            "measurement binaries at b009d6f68 because git diff 5b408206c..b009d6f68 touches "
            "only env-gated R0/R1 + callback plumbing in common/debug.cpp and "
            "src/models/longcat-flash-ngram.cpp, proven inert by the committed Class-2 18/18 "
            "and probe 39/39 gates. NOT the R1 clean-RoPE state: block-0 RoPE here is "
            "production ggml_rope_ext angles, matching the 40-violation run."
        ),
        "hf_oracles": {"stages": str(HF_STAGES), "hidden": str(HF_HIDDEN)},
        "representation": "final prompt-token row only, F32 [3072], both sides",
        "semantic_order_evidence": SEMANTIC_ORDER_EVIDENCE,
        "verification": {},
        "pairs": {},
        "error_decomposition_steps": {},
        "flags": [],
        "endpoint": {},
        "caveats": [
            "Inherited-error caveat: a flagged boundary marks where divergence grows or "
            "changes character; it is NOT proof that the local operation generated the "
            "error. Inherited input error passing through nonlinear MLP/MoE (including "
            "router top-k flips) can create new error components.",
            "The 40-violation frozen-512 result is attributed to the aggregate block-0 "
            "corrective stack, not to any single boundary; this walk does not assume the "
            "violations originate in block-0 attention.",
            SEMANTIC_ORDER_EVIDENCE["gap_note"],
        ],
    }

    # ------------------------------------------------------------- Step 1a
    vecs_cpp: dict[str, np.ndarray] = {}
    vecs_hf: dict[str, np.ndarray] = {}
    for label, cpp_name, hf_tag, hf_name, _resid in CHAIN:
        vecs_cpp[label] = load_vec(CPP_DIR / (cpp_name + ".bin"), CPP_SHA[cpp_name], HIDDEN)
        hf_dir, hf_sha = (
            (HF_STAGES, HF_STAGES_SHA) if hf_tag == "stages" else (HF_HIDDEN, HF_HIDDEN_SHA)
        )
        vecs_hf[label] = load_vec(hf_dir / (hf_name + ".bin"), hf_sha[hf_name], HIDDEN)

    # Zero-cost identity gates.
    stages_input = load_vec(HF_STAGES / "input.bin", HF_STAGES_SHA["input"], HIDDEN)
    if not np.array_equal(stages_input, vecs_hf["inp_embd_ngram"]):
        stop("identity gate failed: stages input != hidden inp_embd_ngram")
    if not np.array_equal(vecs_cpp["inp_embd_ngram"], vecs_hf["inp_embd_ngram"]):
        stop("identity gate failed: C++ inp_embd_ngram != HF inp_embd_ngram (origin must be byte-exact)")
    hidden_l00 = load_vec(HF_HIDDEN / "logical_00.bin", HF_HIDDEN_SHA["logical_00"], HIDDEN)
    if not np.array_equal(hidden_l00, vecs_hf["logical_00"]):
        stop("identity gate failed: hidden logical_00 != stages logical0_out")
    report["verification"]["identity_gates"] = {
        "stages_input_eq_hidden_inp_embd_ngram": True,
        "cpp_origin_byte_exact": True,
        "hidden_logical_00_eq_stages_logical0_out": True,
        "all_input_sha256_gated": True,
    }
    print("step 1a: all %d inputs SHA-gated; identity gates PASS" % (len(CHAIN) * 2 + 2))

    # ------------------------------------------------------------- Step 2
    order_labels = [c[0] for c in CHAIN]
    errors: dict[str, np.ndarray] = {}
    for label, _cpp_name, _hf_tag, _hf_name, _resid in CHAIN:
        m = pair_metrics(vecs_cpp[label], vecs_hf[label])
        report["pairs"][label] = m
        errors[label] = vecs_cpp[label].astype(np.float64) - vecs_hf[label].astype(np.float64)

    # Known-answer anchor: attn0_resid must reproduce the committed rel-RMSE.
    got = report["pairs"]["attn0_resid"]["rel_rmse"]
    if abs(got - KNOWN_ATTN0_REL_RMSE) > KNOWN_ATTN0_TOL:
        stop(
            "known-answer FAIL: attn0_resid rel-RMSE %.12g != committed %.12g"
            % (got, KNOWN_ATTN0_REL_RMSE)
        )
    report["verification"]["known_answer_attn0_rel_rmse"] = {
        "computed": got,
        "committed": KNOWN_ATTN0_REL_RMSE,
        "pass": True,
    }
    print("known-answer: attn0_resid rel-RMSE %.9f reproduces committed 0.00390108 PASS" % got)

    # Error decomposition along consecutive residual-stream boundaries.
    resid_labels = [c[0] for c in CHAIN if c[4]]
    # The origin is byte-exact (zero error): decomposition starts at attn0_resid.
    if resid_labels[0] != "inp_embd_ngram":
        stop("chain invariant broken")
    walk = resid_labels[1:]
    for prev, cur in zip(walk[:-1], walk[1:]):
        report["error_decomposition_steps"]["%s->%s" % (prev, cur)] = decomp_step(
            errors[prev], errors[cur]
        )

    # Cross-projection onto the final trunk error (descriptive): how much of
    # e(logical_12) lies along each earlier boundary's error direction. All
    # residual-stream boundaries share the same 3072-dim residual basis, so
    # the comparison is well-defined. Quantifies seed persistence -- in
    # particular how much of the final trunk error is explainable as the
    # block-0 attention seed direction surviving downstream.
    e_final = errors["logical_12"]
    nf = float(np.sqrt((e_final ** 2).sum()))
    report["cross_projection_to_logical_12"] = {}
    for label in walk[:-1]:
        e_i = errors[label]
        ni = float(np.sqrt((e_i ** 2).sum()))
        dot = float((e_i * e_final).sum())
        report["cross_projection_to_logical_12"][label] = {
            "cos_e_i_vs_e_final": dot / (ni * nf) if ni > 0 and nf > 0 else float("nan"),
            "final_error_fraction_along_e_i": abs(dot) / (ni * nf) if ni > 0 and nf > 0 else float("nan"),
        }

    # Descriptive flags (heuristic, non-binding; interpretation deferred).
    steps = report["error_decomposition_steps"]
    ratios = [s["ratio"] for s in steps.values() if np.isfinite(s["ratio"])]
    med_ratio = float(np.median(ratios))
    for name, s in steps.items():
        reasons = []
        if np.isfinite(s["ratio"]) and med_ratio > 0 and s["ratio"] > 3.0 * med_ratio:
            reasons.append("L2 growth ratio %.3f > 3x median %.3f" % (s["ratio"], med_ratio))
        if np.isfinite(s["orth_frac"]) and s["orth_frac"] > 0.5:
            reasons.append("orthogonal (new-direction) fraction %.3f > 0.5" % s["orth_frac"])
        if reasons:
            report["flags"].append(
                {
                    "step": name,
                    "reasons": reasons,
                    "reading": "divergence grows or changes character here (NOT proof of local origin; see caveats)",
                }
            )
    report["flag_heuristic"] = {
        "median_l2_growth_ratio": med_ratio,
        "rules": ["ratio > 3x median", "orth_frac > 0.5"],
    }

    # ------------------------------------------------------------- Step 3
    cpp_logits = load_vec(CPP_LOGITS, CPP_LOGITS_SHA, VOCAB)
    hf_logits = load_vec(HF_LOGITS, HF_LOGITS_SHA, VOCAB)
    d = cpp_logits.astype(np.float64) - hf_logits.astype(np.float64)
    tol = 0.5 + 0.05 * np.abs(hf_logits.astype(np.float64))
    violations = int((np.abs(d) > tol).sum())
    rmse = float(np.sqrt((d ** 2).mean()))
    denom = float(
        np.sqrt((cpp_logits.astype(np.float64) ** 2).sum())
        * np.sqrt((hf_logits.astype(np.float64) ** 2).sum())
    )
    cosine = float((cpp_logits.astype(np.float64) * hf_logits.astype(np.float64)).sum() / denom)
    max_abs = float(np.abs(d).max())
    top1_cpp = int(np.argmax(cpp_logits))
    top1_hf = int(np.argmax(hf_logits))
    hf_logits_rms = float(np.sqrt((hf_logits.astype(np.float64) ** 2).mean()))

    ep = {
        "violations": violations,
        "rmse": rmse,
        "cosine": cosine,
        "max_abs": max_abs,
        "top1_cpp": top1_cpp,
        "top1_hf": top1_hf,
        "hf_logits_rms": hf_logits_rms,
        "logits_rel_rmse": rmse / hf_logits_rms,
        "result_norm_rel_rmse": report["pairs"]["result_norm"]["rel_rmse"],
        "logical_12_rel_rmse": report["pairs"]["logical_12"]["rel_rmse"],
        "note": (
            "Descriptive only. The result_norm -> logits relationship does NOT isolate "
            "lm_head: no offline projection of the final head is performed here, so any "
            "difference between result_norm-level and logits-level divergence remains "
            "unattributed between the final norm inputs, lm_head execution, and metric "
            "geometry. lm_head attribution: UNRESOLVED."
        ),
    }
    # Known-answer endpoint gates vs the committed frozen-512 record.
    if violations != KNOWN_LOGITS["violations"]:
        stop("known-answer FAIL: logits violations %d != committed 40" % violations)
    if top1_cpp != KNOWN_LOGITS["top1"] or top1_hf != KNOWN_LOGITS["top1"]:
        stop("known-answer FAIL: top1 %d/%d != committed 483" % (top1_cpp, top1_hf))
    for key in ("rmse", "cosine", "max_abs"):
        if abs(ep[key] - KNOWN_LOGITS[key]) > KNOWN_LOGITS_TOL:
            stop(
                "known-answer FAIL: logits %s %.12g != committed %.12g"
                % (key, ep[key], KNOWN_LOGITS[key])
            )
    ep["known_answer_pass"] = True
    report["endpoint"] = ep
    print(
        "endpoint known-answer: violations=%d rmse=%.9f cosine=%.9f max_abs=%.9f top1=%d PASS"
        % (violations, rmse, cosine, max_abs, top1_cpp)
    )

    # ------------------------------------------------------------- output
    print()
    print(
        "%-14s %10s %12s %12s %8s %8s | %8s %8s %9s"
        % ("boundary", "rel_rmse", "abs_rmse", "err_l2", "raw_eq", "bf16_eq", "ratio", "cos_e", "orth_frac")
    )
    prev = None
    for label in order_labels:
        m = report["pairs"][label]
        step = report["error_decomposition_steps"].get("%s->%s" % (prev, label)) if prev else None
        extra = (
            "%8.3f %8.4f %9.4f" % (step["ratio"], step["cos_prev_cur"], step["orth_frac"])
            if step
            else "%8s %8s %9s" % ("-", "-", "-")
        )
        print(
            "%-14s %10.3e %12.5e %12.5e %8d %8d | %s"
            % (label, m["rel_rmse"], m["abs_rmse"], m["err_l2"], m["raw_equal"], m["bf16_cpp_equal_hf"], extra)
        )
        if label in walk:
            prev = label
        elif label == "result_norm":
            prev = None
    print()
    print("seed persistence: cos(e_boundary, e_logical_12) per boundary")
    for label in walk[:-1]:
        c = report["cross_projection_to_logical_12"][label]["cos_e_i_vs_e_final"]
        print("  %-14s %+.4f" % (label, c))
    print()
    for f in report["flags"]:
        print("FLAG %s: %s" % (f["step"], "; ".join(f["reasons"])))
    print(
        "endpoint: logits rel-RMSE %.6f vs result_norm rel-RMSE %.6f (descriptive; lm_head unresolved)"
        % (ep["logits_rel_rmse"], ep["result_norm_rel_rmse"])
    )

    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / "downstream_boundary_walk.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT_DIR / "SHA256SUMS.txt").write_text(
        "%s  downstream_boundary_walk.json\n" % json_sha, encoding="utf-8"
    )
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("DOWNSTREAM BOUNDARY WALK: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
