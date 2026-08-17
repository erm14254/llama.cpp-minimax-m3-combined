#!/usr/bin/env python3
"""Causal-reset comparison: post-reset regenerated divergence vs observational.

Inputs (all fail-closed SHA-gated against their manifests before any math):
  HF full-sequence oracles   D:\\lc_resid_walk_512\\            (Stage 4)
  C++ control run (no reset) cpp_resid_walk_control_512\\       (Stage 5)
  C++ injection run (reset)  cpp_resid_walk_inject_512\\        (Stage 6)

For every downstream boundary logical_01..logical_13 and the whole-sequence
post-final-norm surface (result_norm_full = h_nextn), whole-tensor [512,3072]:

  e_observed = cpp_control - hf   (total divergence, inherited + local)
  e_reset    = cpp_inject  - hf   (divergence REGENERATED downstream of the
                                   byte-exact logical_00 reset -- causally
                                   free of inherited block-0..logical-0 error)

Dual stop rule (pre-registered):
  first-raw:   first boundary with any whole-tensor raw mismatch after the
               exact reset (reported regardless of size)
  materiality: full ||e_reset||/||e_observed|| trajectory with first
               crossings at 0.01 / 0.10 / 0.50 (0.10 is an explicitly
               conventional reporting marker, not a decision criterion)
If raw onset and consequential growth are separated, the verdict is
"stop-for-review", never an automatic block selection.

Known-answer gate: row 511 of e_observed must reproduce the committed
observational-walk rel-RMSE at every shared boundary exactly (the inputs are
byte-identical by the proven row-511 identity gates).

Measurement-only. No arithmetic changes anywhere.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
HF_DIR = Path(r"D:\lc_resid_walk_512")
CTRL_DIR = REPO / "cpp_resid_walk_control_512"
INJ_DIR = REPO / "cpp_resid_walk_inject_512"
WALK_JSON = REPO / "downstream_walk_512" / "downstream_boundary_walk.json"
OUT_DIR = REPO / "resid_inject_walk_512"

N_TOK = 512
HIDDEN = 3072
VEC_BYTES = N_TOK * HIDDEN * 4

ORACLE_SHA = "d810f93c50ea42c5909ab289ebf62a0c5629f40530d2e5fc706dde67f0eaf763"

# Walk order: (label, hf_file, cpp_file). result_norm_full is the post-norm
# space change (h_nextn on the C++ side, hidden_states[14] on the HF side).
BOUNDARIES = [
    (f"logical_{n:02d}", f"logical_{n:02d}.bin", f"logical_{n:02d}_full.bin")
    for n in range(1, 14)
] + [("result_norm", "result_norm_full.bin", "result_norm_full.bin")]

# Stage-1 causal-cut proof (source-referenced), embedded verbatim in the
# output JSON as the license for interpreting e_reset causally.
CAUSAL_CUT_PROOF = {
    "claim": (
        "l_out-1 / logical_00 is a complete causal cut: the residual stream "
        "is the only mutable inter-layer carrier on both sides; a byte-exact "
        "reset there makes every downstream trunk state a function of "
        "(injected state, weights, deterministic inputs) only."
    ),
    "cpp": {
        "loop_carried_state": (
            "exactly two tensor variables cross main-loop iterations: inpL "
            "(the residual stream; the injected l_out-1 node IS the inpL "
            "consumed by physical block 2) and moe_shortcut "
            "(longcat-flash-ngram.cpp:513-514), which is computed on the even "
            "block from that block's own ffn_norm output (1005-1066) and "
            "consumed + set nullptr on the paired odd block (1088-1093) -- at "
            "the injection point it is already consumed upstream of l_out-1"
        ),
        "attention_inputs": (
            "build_attn_inp_k builds only the deterministic causal KQ mask "
            "(llama-graph.cpp:3046-3070); the LSA indexer weights are loaded "
            "(longcat-flash-ngram.cpp:430-470) but appear nowhere in the "
            "trunk graph -- no activation-derived attention state exists"
        ),
        "kv_caches": (
            "per-layer cache slots written by each block from its own inputs "
            "in this same forward; later blocks never read earlier layers' "
            "cache lines"
        ),
        "mtp": (
            "nextn/MTP is a separate graph (graph_mtp at layer index n_layer, "
            "n_layer_nextn == 1 asserted, longcat-flash-ngram.cpp:1128-1138), "
            "not executed in this run type; il==n_layer-1 row filter inert "
            "(embeddings_nextn_masked defaults false, llama-context.cpp:119)"
        ),
        "cvec": (
            "build_cvec is identity (no control vector loaded) and l_out is "
            "the post-cvec node (1099-1100): the injection replaces exactly "
            "what downstream consumes"
        ),
    },
    "hf": {
        "trunk": (
            "the trunk loop passes only hidden_states between layers "
            "(modeling_longcat_flash_sparse.py:1472-1485); causal mask and "
            "position embeddings are computed once and deterministic; "
            "past_key_values is None under use_cache=False (NgramCache never "
            "created, :1442-1443)"
        ),
        "layer_locals": (
            "topk_indices and shortcut_mlp_output are locals of one layer "
            "forward (:980-1032), consumed within the same layer, never "
            "stored or returned"
        ),
        "lsa_at_512": (
            "total_kv_len=512 <= index_topk=2048 puts both attentions on the "
            "dense fast path ('full-owner' :877-889, 'full-reuse' :917-937); "
            "topk_indices is None everywhere; the indexer only projects keys "
            "from the layer's own input and persists nothing with cache=None "
            "(:869-874); last_lsa_* writes are telemetry never read in the "
            "trunk forward"
        ),
        "mtp": (
            "LongcatFlashSparseMTP is a separate module (:1226+), not called "
            "by the trunk forward"
        ),
    },
    "verdict": "COMPLETE CAUSAL CUT -- no mutable bypass found",
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
    return {
        "elements": int(cpp.size),
        "raw_equal": int((cpp == hf).sum()),
        "hf_on_bf16_lattice": int((to_bf16(hf) == hf).sum()),
        "bf16_cpp_equal_hf": int((to_bf16(cpp) == hf).sum()),
        "err_l2": l2,
        "abs_rmse": rmse,
        "rel_rmse": rmse / hf_rms if hf_rms > 0 else float("nan"),
        "hf_rms": hf_rms,
        "max_abs": float(np.abs(d).max()),
        "cosine_cpp_hf": float((c * h).sum() / denom) if denom > 0 else float("nan"),
        "first_divergent_token": int(mism_rows[0]) if mism_rows.size else -1,
        "divergent_token_rows": int(mism_rows.size),
        "row_l2_max": float(row_l2.max()),
        "row_l2_argmax": int(np.argmax(row_l2)),
        "row_l2_final": float(row_l2[-1]),
        "row_l2": [float("%.6g" % x) for x in row_l2],
    }


def main() -> int:
    print("causal-reset comparison (measurement-only)")
    print("numpy=%s python=%s" % (np.__version__, platform.python_version()))

    hf_man = load_manifest(HF_DIR)
    ctrl_man = load_manifest(CTRL_DIR)
    inj_man = load_manifest(INJ_DIR)

    # Landing re-verification from disk.
    if hf_man.get("logical_00_oracle.bin") != ORACLE_SHA:
        stop("HF manifest oracle SHA mismatch")
    if inj_man.get("logical_00_full.bin") != ORACLE_SHA:
        stop("injection landing mismatch in manifest")
    got = sha256_file(INJ_DIR / "logical_00_full.bin")
    if got != ORACLE_SHA:
        stop("injection landing mismatch on disk: %s" % got)
    print("landing re-verified: logical_00_full.bin == oracle %s..." % ORACLE_SHA[:8])

    # Control full-seq must differ from oracle at logical_00 (it is the real
    # C++ value, not the injected one) -- guards against dir mixups.
    if ctrl_man.get("logical_00_full.bin") == ORACLE_SHA:
        stop("control run logical_00_full equals the oracle - dir mixup?")

    walk_ref = json.loads(WALK_JSON.read_text(encoding="utf-8"))

    report: dict = {
        "description": "Causal-reset comparison at logical_00 (full-sequence walk)",
        "oracle_sha256": ORACLE_SHA,
        "hf_dir": str(HF_DIR),
        "control_dir": str(CTRL_DIR),
        "inject_dir": str(INJ_DIR),
        "causal_cut_proof": CAUSAL_CUT_PROOF,
        "provenance": {},
        "boundaries": {},
        "known_answer": {},
        "verdict": {},
        "caveats": [
            "e_reset measures divergence REGENERATED downstream of a byte-exact "
            "logical_00 reset. It is causal for 'the downstream trunk alone "
            "generates this much divergence from a clean input', but a "
            "boundary's e_reset still contains error inherited from the "
            "post-reset boundaries BEFORE it; only logical_01's e_reset is "
            "purely single-logical-layer-generated.",
            "result_norm is a post-RMSNorm space change; its rel-RMSE is not "
            "directly comparable to residual-stream boundaries.",
        ],
    }

    for tag, d in (("control", CTRL_DIR), ("inject", INJ_DIR)):
        p = d / "run_provenance.json"
        if p.is_file():
            report["provenance"][tag] = json.loads(p.read_text(encoding="ascii"))

    ratios = []
    first_raw = None
    crossings = {"0.01": None, "0.10": None, "0.50": None}
    prev_reset_err = None

    for label, hf_name, cpp_name in BOUNDARIES:
        hf = load_mat(HF_DIR, hf_name, hf_man)
        ctrl = load_mat(CTRL_DIR, cpp_name, ctrl_man)
        inj = load_mat(INJ_DIR, cpp_name, inj_man)

        m_obs = metrics(ctrl, hf)
        m_reset = metrics(inj, hf)

        e_obs = ctrl.astype(np.float64) - hf.astype(np.float64)
        e_reset = inj.astype(np.float64) - hf.astype(np.float64)
        n_obs = float(np.sqrt((e_obs ** 2).sum()))
        n_reset = float(np.sqrt((e_reset ** 2).sum()))
        dot = float((e_obs * e_reset).sum())
        entry = {
            "observed": m_obs,
            "reset": m_reset,
            "ratio_reset_over_observed": n_reset / n_obs if n_obs > 0 else float("nan"),
            "cos_e_reset_vs_e_observed": dot / (n_obs * n_reset)
            if n_obs > 0 and n_reset > 0
            else float("nan"),
            "inject_equals_control": bool(np.array_equal(ctrl, inj)),
        }
        if prev_reset_err is not None and label != "result_norm":
            np_prev = float(np.sqrt((prev_reset_err ** 2).sum()))
            if np_prev > 0 and n_reset > 0:
                dpc = float((prev_reset_err * e_reset).sum())
                entry["reset_step_growth"] = n_reset / np_prev
                entry["reset_step_cos"] = dpc / (np_prev * n_reset)
        if label != "result_norm":
            prev_reset_err = e_reset

        report["boundaries"][label] = entry
        ratios.append((label, entry["ratio_reset_over_observed"]))

        if first_raw is None and m_reset["raw_equal"] < m_reset["elements"]:
            first_raw = label
        for k, thr in (("0.01", 0.01), ("0.10", 0.10), ("0.50", 0.50)):
            if crossings[k] is None and entry["ratio_reset_over_observed"] >= thr:
                crossings[k] = label

        print(
            "%-12s obs rel %.4e | reset rel %.4e | ratio %.4f | raw_eq %d/%d | first_tok %d"
            % (
                label,
                m_obs["rel_rmse"],
                m_reset["rel_rmse"],
                entry["ratio_reset_over_observed"],
                m_reset["raw_equal"],
                m_reset["elements"],
                m_reset["first_divergent_token"],
            )
        )

    # Consumption anomaly check (corrected gate: data, but all-identical would
    # indicate non-consumption despite the landed reset).
    any_moved = any(
        not report["boundaries"][lbl]["inject_equals_control"]
        for lbl, _, _ in BOUNDARIES
    )
    if not any_moved:
        stop("ANOMALY: every downstream dump byte-identical to control despite landed reset")

    # Known-answer: row-511 slice of e_observed must reproduce the committed
    # observational walk rel-RMSE exactly at shared boundaries.
    ka = {}
    for label in [f"logical_{n:02d}" for n in range(1, 13)] + ["result_norm"]:
        hf_name, cpp_name = next(
            (h, c) for (l, h, c) in BOUNDARIES if l == label
        )
        hf_row = load_mat(HF_DIR, hf_name, hf_man)[-1]
        ctrl_row = load_mat(CTRL_DIR, cpp_name, ctrl_man)[-1]
        d = ctrl_row.astype(np.float64) - hf_row.astype(np.float64)
        rel = float(
            np.sqrt((d ** 2).mean())
            / np.sqrt((hf_row.astype(np.float64) ** 2).mean())
        )
        committed = walk_ref["pairs"][label]["rel_rmse"]
        if abs(rel - committed) > 1e-12:
            stop(
                "known-answer FAIL at %s: row-511 rel %.15g != committed %.15g"
                % (label, rel, committed)
            )
        ka[label] = {"row511_rel_rmse": rel, "committed": committed, "pass": True}
    report["known_answer"] = ka
    print("known-answer: row-511 observational slice reproduces the committed walk at %d boundaries" % len(ka))

    # Dual stop rule.
    resid_ratios = [(l, r) for (l, r) in ratios if l != "result_norm"]
    verdict = {
        "first_raw_divergence": first_raw,
        "ratio_trajectory": {l: r for (l, r) in ratios},
        "first_crossing_0.01": crossings["0.01"],
        "first_crossing_0.10": crossings["0.10"],
        "first_crossing_0.50": crossings["0.50"],
        "note_0.10": "0.10 is an explicitly conventional reporting marker, not a decision criterion",
    }
    if first_raw is not None and crossings["0.10"] is not None and first_raw == crossings["0.10"]:
        verdict["reading"] = (
            "raw onset and consequential growth coincide at %s" % first_raw
        )
    else:
        verdict["reading"] = (
            "raw onset (%s) and consequential growth (0.10-crossing: %s) are "
            "separated -> stop for review; no automatic block selection"
            % (first_raw, crossings["0.10"])
        )
    report["verdict"] = verdict
    print("first-raw: %s | crossings 0.01=%s 0.10=%s 0.50=%s" % (
        first_raw, crossings["0.01"], crossings["0.10"], crossings["0.50"]))
    print("reading: %s" % verdict["reading"])

    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / "resid_inject_walk.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    json_sha = sha256_file(out_json)
    (OUT_DIR / "SHA256SUMS.txt").write_text(
        "%s  resid_inject_walk.json\n" % json_sha, encoding="utf-8"
    )
    print("wrote %s (sha256 %s)" % (out_json, json_sha))
    print("CAUSAL-RESET COMPARISON: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
