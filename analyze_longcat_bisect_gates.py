#!/usr/bin/env python3
"""Cast-vs-epsilon bisect: per-variant operator-level gates (reviewed plan).

--variant eps  (run dir cpp_resid_walk_inject2_epsOnly_512):
    block2_q_a_norm_full  vs A6_q, block2_kv_a_norm_full vs A6_kv
    PASS rule: the established F32 reduction-noise protocol - max f32-ulp
    <= 4, exact counts recorded (A5-magnitude class expected). ulp > 4 = FAIL.

--variant cast (run dir cpp_resid_walk_inject2_castOnly_512):
    block2_q_a_norm_full  vs D5_q, block2_kv_a_norm_full vs D5_kv
    PASS rule: BYTE-EXACT, both norms, no ULP allowance. The stage-B
    empirical standard is C++ == offline model byte-exact; the 7-element
    near-tie was D6-model-vs-HF residue, not a C++-vs-model tolerance.
    ANY mismatch = STOP_FOR_REVIEW (ulp-quantified here for that review).

Downstream chain surfaces are recorded observationally (hashes only).
Measurement/verification tooling only; no arithmetic is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
TGT = REPO / "block2_bisect_targets"

TARGETS = {
    "block2_q_a_norm_A6_target.bin": "a4d2dd1c6bbf20cadb44e82a896fedb4574472f1769baa97dc893acccc01c65d",
    "block2_kv_a_norm_A6_target.bin": "e68065566f20dbfdec371f389a5f27af500b85c0dbd0eb5e03d44d335a072245",
    "block2_q_a_norm_D5_target.bin": "0fa0234c3f232bfd116d5c34b1f7970d286cf20a1081888589dc675215442d3c",
    "block2_kv_a_norm_D5_target.bin": "6b25aeadf87d32d53fe99e54e5c597e61d2587742a3466f2122ccc070498c606",
}
OBSERVATIONAL = [
    "block2_q_a_proj_full.bin", "block2_kv_a_proj_full.bin",
    "block2_q_b_proj_full.bin", "block2_q_scaled_full.bin",
    "block2_kv_cmpr_scaled_full.bin", "block2_q_pe_rope_full.bin",
    "block2_k_pe_rope_full.bin", "block2_kqv_out_full.bin",
    "block1_attn0_out_full.bin",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_f32(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4")


def f32_ulp_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua = a.view(np.uint32).astype(np.int64)
    ub = b.view(np.uint32).astype(np.int64)
    ka = np.where(ua >= 0x80000000, 0xFFFFFFFF - ua, ua + 0x80000000)
    kb = np.where(ub >= 0x80000000, 0xFFFFFFFF - ub, ub + 0x80000000)
    return np.abs(ka - kb)


def bf16_ulp_stats(a: np.ndarray, b: np.ndarray) -> dict:
    ab = (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bb = (b.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    mism = ab != bb
    n = int(mism.sum())
    if n == 0:
        return {"mismatches": 0}
    a64 = ab.astype(np.int64)
    b64 = bb.astype(np.int64)
    ka = np.where((a64 & 0x8000) != 0, 0xFFFF - a64, a64 | 0x8000)
    kb = np.where((b64 & 0x8000) != 0, 0xFFFF - b64, b64 | 0x8000)
    d = np.abs(ka - kb)[mism]
    pos = np.argwhere(mism.reshape(512, -1))
    return {
        "mismatches": n,
        "total": int(a.size),
        "max_bf16_ulp": int(d.max()),
        "one_ulp_count": int((d == 1).sum()),
        "positions_token_ch": [tuple(int(x) for x in p) for p in pos[:32]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["eps", "cast"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ns = ap.parse_args()

    tag = "epsOnly" if ns.variant == "eps" else "castOnly"
    run_dir = Path(ns.run_dir) if ns.run_dir else REPO / ("cpp_resid_walk_inject2_%s_512" % tag)
    out_dir = Path(ns.out_dir) if ns.out_dir else REPO / ("bisect_%s_gates_512" % tag)
    if not run_dir.is_dir():
        raise SystemExit("STOP: run dir missing: %s" % run_dir)
    for name, sha in TARGETS.items():
        p = TGT / name
        if not p.is_file() or sha256_file(p) != sha:
            raise SystemExit("STOP: target missing/SHA mismatch: %s" % name)

    suffix = "A6" if ns.variant == "eps" else "D5"
    gates = []
    overall = "PASS"
    for norm, width in (("q_a_norm", 1536), ("kv_a_norm", 512)):
        run_p = run_dir / ("block2_%s_full.bin" % norm)
        tgt_p = TGT / ("block2_%s_%s_target.bin" % (norm, suffix))
        entry = {"gate": "%s_%s" % (norm, suffix), "file": run_p.name,
                 "target": tgt_p.name, "target_sha256": TARGETS[tgt_p.name]}
        if not run_p.is_file():
            entry.update(verdict="FAIL", reason="missing dump")
            gates.append(entry)
            overall = "FAIL"
            continue
        run = load_f32(run_p)
        tgt = load_f32(tgt_p)
        if run.size != tgt.size:
            entry.update(verdict="FAIL", reason="size mismatch")
            gates.append(entry)
            overall = "FAIL"
            continue
        exact = int((run.view(np.uint32) == tgt.view(np.uint32)).sum())
        entry["exact_count"] = exact
        entry["total"] = int(run.size)
        if ns.variant == "eps":
            ulp = f32_ulp_diff(run, tgt)
            entry["max_f32_ulp"] = int(ulp.max())
            entry["verdict"] = "PASS" if int(ulp.max()) <= 4 else "FAIL"
            entry["protocol"] = "established F32 reduction-noise: max f32-ulp <= 4; exact count recorded"
        else:
            if exact == run.size:
                entry["verdict"] = "PASS"
            else:
                entry["verdict"] = "STOP_FOR_REVIEW"
                entry["ulp_analysis"] = bf16_ulp_stats(run, tgt)
            entry["protocol"] = ("BYTE-EXACT only (stage-B empirical standard: C++ == offline "
                                 "model byte-exact); any mismatch = STOP for review")
        gates.append(entry)
        if entry["verdict"] == "FAIL":
            overall = "FAIL"
        elif entry["verdict"] == "STOP_FOR_REVIEW" and overall == "PASS":
            overall = "STOP_FOR_REVIEW"

    observational = {}
    for name in OBSERVATIONAL:
        p = run_dir / name
        observational[name] = sha256_file(p) if p.is_file() else "MISSING"
    prov_path = run_dir / "run_provenance.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8-sig")) if prov_path.is_file() else None

    out = {
        "variant": ns.variant,
        "run_dir": str(run_dir),
        "gates": gates,
        "observational_expected_divergent": observational,
        "run_provenance": provenance,
        "overall": overall,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("bisect_%s_gates.json" % tag)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for g in gates:
        extra = ""
        if "max_f32_ulp" in g:
            extra = "  max_f32_ulp=%d exact=%d/%d" % (g["max_f32_ulp"], g["exact_count"], g["total"])
        elif "exact_count" in g:
            extra = "  exact=%d/%d" % (g["exact_count"], g["total"])
        print("%-22s %s%s" % (g["gate"], g["verdict"], extra))
    print("BISECT %s GATES: %s" % (ns.variant.upper(), overall))
    print("written: %s" % out_path)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
