#!/usr/bin/env python
"""LSA <=512 dumpability-proof offline analyzer (measurement round 2026-08-18).

Fail-closed structural validation of the five-surface Type-S proof run:

  1. exact file inventory and byte sizes;
  2. sidecar metadata (shape / dtype / source_type / source_ne);
  3. finiteness of every dumped value;
  4. nope-half identity: columns 64..127 of lsa_indexer_k_full.bin must be
     byte-identical to columns 64..127 of lsa_indexer_k_norm_full.bin for
     all 512 rows (the BF16->F32->BF16 roundtrip is the identity on the
     un-roped half; RoPE touches only columns 0..63). No expectation is
     registered for the roped columns - that is blocker territory.
  5. BF16-lattice membership (low 16 bits of the f32 encoding zero) for the
     three BF16-source surfaces (dump-path validation) and for the
     attn_norm-0 anchor (a real semantic check: its producer ends in a
     BF16->F32 roundtrip). lsa_indexer_k_proj is a raw F32 GEMM output and
     is deliberately NOT lattice-checked.

Serialization/surface validation only: no HF comparison, no arithmetic
claims, no verdict about the four indexer blockers.
"""

import argparse
import json
import os
import sys

import numpy as np

N_TOKENS = 512

SURFACES = {
    # file -> (tensor_name, width, source_type, lattice_check)
    "lsa_anchor_attn_norm0_full.bin": ("attn_norm-0",          3072, "f32",  True),
    "lsa_anchor_q_a_norm0_full.bin":  ("q_a_norm-0",           1536, "bf16", True),
    "lsa_indexer_k_proj_full.bin":    ("lsa_indexer_k_proj-0",  128, "f32",  False),
    "lsa_indexer_k_norm_full.bin":    ("lsa_indexer_k_norm-0",  128, "bf16", True),
    "lsa_indexer_k_full.bin":         ("lsa_indexer_k_2d-0",    128, "bf16", True),
}


def fail(msg: str) -> None:
    print(f"ANALYZER FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run_dir = args.run_dir
    arrays = {}

    for fname, (tname, width, source_type, _) in SURFACES.items():
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            fail(f"missing {fname}")
        expect_bytes = width * N_TOKENS * 4
        size = os.path.getsize(path)
        if size != expect_bytes:
            fail(f"{fname}: size {size} != {expect_bytes}")

        sidecar = os.path.splitext(path)[0] + ".json"
        if not os.path.isfile(sidecar):
            fail(f"missing sidecar for {fname}")
        with open(sidecar, "r", encoding="ascii") as fh:
            meta = json.load(fh)
        if meta.get("tensor") != tname:
            fail(f"{fname}: sidecar tensor {meta.get('tensor')!r} != {tname!r}")
        if meta.get("shape") != [N_TOKENS, width]:
            fail(f"{fname}: sidecar shape {meta.get('shape')} != [{N_TOKENS}, {width}]")
        if meta.get("order") != "token-major":
            fail(f"{fname}: sidecar order {meta.get('order')!r}")
        if meta.get("dtype") != "float32-le":
            fail(f"{fname}: sidecar dtype {meta.get('dtype')!r}")
        if meta.get("bytes") != expect_bytes:
            fail(f"{fname}: sidecar bytes {meta.get('bytes')} != {expect_bytes}")
        if meta.get("source_type") != source_type:
            fail(f"{fname}: sidecar source_type {meta.get('source_type')!r} != {source_type!r}")
        src_ne = meta.get("source_ne")
        if src_ne != [width, N_TOKENS, 1, 1]:
            fail(f"{fname}: sidecar source_ne {src_ne} != [{width}, {N_TOKENS}, 1, 1]")

        a = np.fromfile(path, dtype="<f4").reshape(N_TOKENS, width)
        if not np.isfinite(a).all():
            fail(f"{fname}: non-finite values present")
        arrays[fname] = a
        print(f"ok: {fname} [{N_TOKENS}, {width}] source_type={source_type} finite")

    # Nope-half identity (columns 64..127, all rows, byte-exact).
    k_full = arrays["lsa_indexer_k_full.bin"][:, 64:128]
    k_norm = arrays["lsa_indexer_k_norm_full.bin"][:, 64:128]
    if k_full.tobytes() != k_norm.tobytes():
        diff = int(np.count_nonzero(k_full.view("<u4") != k_norm.view("<u4")))
        fail(f"nope-half identity FAIL: {diff}/{k_full.size} elements differ")
    print("ok: nope-half identity (lsa_indexer_k[:,64:128] == lsa_indexer_k_norm[:,64:128] byte-exact)")

    # BF16-lattice membership.
    for fname, (_, _, _, lattice) in SURFACES.items():
        if not lattice:
            continue
        bits = arrays[fname].view("<u4")
        off = int(np.count_nonzero(bits & np.uint32(0xFFFF)))
        if off != 0:
            fail(f"{fname}: {off}/{bits.size} values off the BF16 lattice")
        print(f"ok: {fname} on the BF16 lattice ({bits.size}/{bits.size})")

    print("LSA DUMP PROOF ANALYZER: ALL CHECKS PASS")


if __name__ == "__main__":
    main()
