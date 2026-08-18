#!/usr/bin/env python3
"""Stage-A / stage-B local semantic gates for the il>=1 MLA precision change.

Evaluates the pre-registered per-site byte-exact gates of the reviewed plan
against a dual-reset (inject2) run directory, and writes one JSON verdict.

Stage A (five changed sites -> five gates + two known-answer premise gates):
  A1 block2_q_a_proj_full.bin       == HF q_a_proj oracle        (direct)
  A4 block2_kv_a_proj_full.bin      == HF kv_a_proj oracle       (direct)
  KAq block2_q_a_norm_full.bin      == committed quad-run dump   (premise)
  KAkv block2_kv_a_norm_full.bin    == committed quad-run dump   (premise)
  A2 block2_q_b_proj_full.bin       == T3 offline target
  A3 block2_q_scaled_full.bin       == T4 offline target
  A5 block2_kv_cmpr_scaled_full.bin == T5 offline target
  (A2/A3/A5 are valid only while KAq/KAkv hold - evaluated in that order.)

Stage B (five boundary sites re-gated on the byte-exact HF chain + 2 norms):
  B1 q_a_proj == HF oracle; B4 kv_a_proj == HF oracle
  Bqn  block2_q_a_norm_full.bin  == HF q_a_layernorm oracle (D6 closure)
  Bkvn block2_kv_a_norm_full.bin == HF kv_a_layernorm oracle; on mismatch a
       bf16-ulp near-tie analysis runs: verdict NEAR_TIE_STOP_FOR_REVIEW iff
       every mismatch is exactly one bf16 ulp (the documented D6 model
       residue class), else FAIL. Near-tie is NEVER auto-accepted.
  B2 q_b == HF q_b oracle; B3 q_scaled == TB4; B5 kv_cmpr_scaled == the
       committed block2_mla_targets comparandum.

RoPE / kqv_out / attn_out surfaces are recorded observationally (hashes
only; expected divergent - production angles + untouched attention core).

Measurement/verification tooling only; no arithmetic is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
HF_DIR = Path(r"D:\lc_block2_mla_512")
STAGEA_TGT = REPO / "block2_stageA_targets"
B2_TGT = REPO / "block2_mla_targets"

HF = {
    "q_a_proj.bin": "32173b18459358494f943288b974ef7df70eb540ff9e366c720c14f250407a96",
    "kv_a_proj_with_mqa.bin": "28ea5b52221a94ddf780f04507f11aee7b6fc8617974f53d558424d41c470f3f",
    "q_a_layernorm.bin": "4c9792430fee2716b573ccf365617e537adf8305571e2a5a0b1a881c0c4de340",
    "kv_a_layernorm.bin": "c91991eb459352ec407aebcee5ee2b12e7b25db0bafd3e0462955a8f8144df6b",
    "q_b_proj.bin": "ecb70ef6c9bd4d6f28a67467f3b5ec3fc575d4f37cd2e81bce7cc7554323f308",
}
STAGEA_TARGETS = {
    "block2_q_b_proj_stageA_target.bin": "c0e3536a7289bad6540315eefa20e2fc77b710ce9b2742ce529f23d467aad301",
    "block2_q_scaled_stageA_target.bin": "152316998b149ea4ed3554ce10fa83f220eb0e2800090ef3e4fc62cb2c34aafa",
    "block2_kv_cmpr_scaled_stageA_target.bin": "efef3bc0754e5b463496fd8af381a421794a9c8b6f88bcda3b18ebf451a83dcd",
    "block2_q_scaled_stageB_target.bin": "d9234fa93a69490a5932c4ba10a5cfef58c96752b094362093ef92336f29da21",
}
B2_KV_SCALED_TARGET_SHA = "22f1d6be4cee2b8704f973110876d62761e1891a95e0b71681f606f40a93d0c9"
QUAD_NORM = {
    "block2_q_a_norm_full.bin": "2b60008293032656185fa55ca5f0bb579855c67998ad7082feee8b3991ec8bb4",
    "block2_kv_a_norm_full.bin": "93d7442a30cd7d742f21b777398783ea00faf0e9012658d37dae7d13a07698a9",
}
OBSERVATIONAL = [
    "block2_q_pe_rope_full.bin", "block2_k_pe_rope_full.bin",
    "block2_kqv_out_full.bin", "block1_attn0_out_full.bin",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_f32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4")


def bf16_ulp_analysis(run_arr: np.ndarray, ref_arr: np.ndarray) -> dict:
    """Both arrays are f32-widened bf16 values. Returns mismatch stats in
    bf16-ulp steps (monotonic keyed ordering, sign-aware)."""
    rb = (run_arr.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    fb = (ref_arr.view(np.uint32) >> np.uint32(16)).astype(np.uint16)

    def key(b: np.ndarray) -> np.ndarray:
        b32 = b.astype(np.int64)
        neg = (b32 & 0x8000) != 0
        out = np.where(neg, 0xFFFF - b32, b32 | 0x8000)
        return out

    mism = rb != fb
    n = int(mism.sum())
    if n == 0:
        return {"mismatches": 0}
    d = np.abs(key(rb[mism]) - key(fb[mism]))
    return {
        "mismatches": n,
        "total": int(run_arr.size),
        "max_ulp": int(d.max()),
        "one_ulp_count": int((d == 1).sum()),
        "all_one_ulp": bool((d == 1).all()),
        "first_mismatch_flat_index": int(np.argmax(mism)),
    }


def gate_hash(results: list, name: str, run_path: Path, expected_sha: str,
              note: str = "") -> bool:
    entry = {"gate": name, "file": str(run_path.name), "expected_sha256": expected_sha}
    if not run_path.is_file():
        entry.update(verdict="FAIL", reason="missing")
        results.append(entry)
        return False
    got = sha256_file(run_path)
    entry["actual_sha256"] = got
    entry["verdict"] = "PASS" if got == expected_sha else "FAIL"
    if note:
        entry["note"] = note
    results.append(entry)
    return entry["verdict"] == "PASS"


def verify_reference(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit("STOP: missing reference %s (%s)" % (label, path))
    got = sha256_file(path)
    if got != expected:
        raise SystemExit("STOP: reference SHA mismatch %s: %s != %s" % (label, got, expected))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ns = ap.parse_args()

    run_dir = Path(ns.run_dir) if ns.run_dir else REPO / ("cpp_resid_walk_inject2_stage%s_512" % ns.stage)
    out_dir = Path(ns.out_dir) if ns.out_dir else REPO / ("stage%s_gates_512" % ns.stage)
    if not run_dir.is_dir():
        raise SystemExit("STOP: run dir missing: %s" % run_dir)

    # Fail-closed reference verification before any gate is evaluated.
    for name, sha in HF.items():
        verify_reference(HF_DIR / name, sha, "HF %s" % name)
    for name, sha in STAGEA_TARGETS.items():
        verify_reference(STAGEA_TGT / name, sha, "stage target %s" % name)
    verify_reference(B2_TGT / "block2_kv_cmpr_scaled_target.bin",
                     B2_KV_SCALED_TARGET_SHA, "block2 kv_cmpr_scaled target")

    results: list = []
    extra: dict = {}

    if ns.stage == "A":
        ok_a1 = gate_hash(results, "A1_q_a_proj", run_dir / "block2_q_a_proj_full.bin", HF["q_a_proj.bin"])
        ok_a4 = gate_hash(results, "A4_kv_a_proj", run_dir / "block2_kv_a_proj_full.bin", HF["kv_a_proj_with_mqa.bin"])
        ok_kaq = gate_hash(results, "KAq_quad_norm_premise", run_dir / "block2_q_a_norm_full.bin",
                           QUAD_NORM["block2_q_a_norm_full.bin"],
                           "old-norm output must byte-reproduce the committed quad-run dump")
        ok_kakv = gate_hash(results, "KAkv_quad_norm_premise", run_dir / "block2_kv_a_norm_full.bin",
                            QUAD_NORM["block2_kv_a_norm_full.bin"],
                            "old-norm output must byte-reproduce the committed quad-run dump")
        note_chain = "" if (ok_kaq and ok_kakv) else "PREMISE BROKEN: T3/T4/T5 not interpretable"
        gate_hash(results, "A2_q_b_proj", run_dir / "block2_q_b_proj_full.bin",
                  STAGEA_TARGETS["block2_q_b_proj_stageA_target.bin"], note_chain)
        gate_hash(results, "A3_q_scaled", run_dir / "block2_q_scaled_full.bin",
                  STAGEA_TARGETS["block2_q_scaled_stageA_target.bin"], note_chain)
        gate_hash(results, "A5_kv_cmpr_scaled", run_dir / "block2_kv_cmpr_scaled_full.bin",
                  STAGEA_TARGETS["block2_kv_cmpr_scaled_stageA_target.bin"], note_chain)
        _ = ok_a1, ok_a4
    else:
        gate_hash(results, "B1_q_a_proj", run_dir / "block2_q_a_proj_full.bin", HF["q_a_proj.bin"])
        gate_hash(results, "B4_kv_a_proj", run_dir / "block2_kv_a_proj_full.bin", HF["kv_a_proj_with_mqa.bin"])
        ok_qn = gate_hash(results, "Bqn_q_a_norm", run_dir / "block2_q_a_norm_full.bin",
                          HF["q_a_layernorm.bin"], "D6 byte-closure prediction")
        run_kvn = run_dir / "block2_kv_a_norm_full.bin"
        ok_kvn = gate_hash(results, "Bkvn_kv_a_norm", run_kvn, HF["kv_a_layernorm.bin"],
                           "predicted byte-exact; near-tie class -> STOP FOR REVIEW")
        if not ok_kvn and run_kvn.is_file():
            ana = bf16_ulp_analysis(load_f32(run_kvn), load_f32(HF_DIR / "kv_a_layernorm.bin"))
            extra["kv_a_norm_near_tie_analysis"] = ana
            if ana.get("all_one_ulp") and ana["mismatches"] > 0:
                results[-1]["verdict"] = "NEAR_TIE_STOP_FOR_REVIEW"
                results[-1]["note"] = (
                    "%d/%d one-bf16-ulp mismatches (documented D6 residue class was "
                    "7/262144); NOT auto-accepted - review required"
                    % (ana["mismatches"], ana["total"]))
        chain_note = "" if ok_qn else "upstream norm gate failed: chain gate not independently interpretable"
        gate_hash(results, "B2_q_b_proj", run_dir / "block2_q_b_proj_full.bin", HF["q_b_proj.bin"], chain_note)
        gate_hash(results, "B3_q_scaled", run_dir / "block2_q_scaled_full.bin",
                  STAGEA_TARGETS["block2_q_scaled_stageB_target.bin"], chain_note)
        kv_chain_note = "" if ok_kvn else "upstream kv norm gate not byte-exact: chain gate not independently interpretable"
        gate_hash(results, "B5_kv_cmpr_scaled", run_dir / "block2_kv_cmpr_scaled_full.bin",
                  B2_KV_SCALED_TARGET_SHA, kv_chain_note)

    observational = {}
    for name in OBSERVATIONAL:
        p = run_dir / name
        observational[name] = sha256_file(p) if p.is_file() else "MISSING"

    prov_path = run_dir / "run_provenance.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8-sig")) if prov_path.is_file() else None

    verdicts = [r["verdict"] for r in results]
    overall = "PASS" if all(v == "PASS" for v in verdicts) else (
        "NEAR_TIE_STOP_FOR_REVIEW" if all(v in ("PASS", "NEAR_TIE_STOP_FOR_REVIEW") for v in verdicts)
        else "FAIL")

    out = {
        "stage": ns.stage,
        "run_dir": str(run_dir),
        "gates": results,
        "observational_expected_divergent": observational,
        "run_provenance": provenance,
        "overall": overall,
    }
    out.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("stage%s_gates.json" % ns.stage)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for r in results:
        print("%-24s %s%s" % (r["gate"], r["verdict"],
                              ("  (" + r.get("note", "") + ")") if r.get("note") else ""))
    print("STAGE %s LOCAL GATES: %s" % (ns.stage, overall))
    print("written: %s" % out_path)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
