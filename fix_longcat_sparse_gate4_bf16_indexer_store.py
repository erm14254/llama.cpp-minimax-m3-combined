#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"

EXPECTED_CURRENT_SHA256 = {
    "src/llama-kv-cache.h": "71dc0b8f595a01689bce3a6b1113b849d170c7b50136b6a2ab60b5aa15874440",
    "src/llama-kv-cache.cpp": "c26be6e7c3cea58100f1f73dd9f70e97a836bdb9064bc9edd6aa01ea8b339cab",
    "src/llama-kv-cache-dsa.cpp": "60735775d1a120c5b1d310000dd7f2515b0b975de0b61e86d1ec5120ff377e32",
    "src/llama-graph.h": "ca9108bcfea780beaa035c7b127428ef2d49a0b83253c14848876e2c20cd0b25",
    "src/llama-graph.cpp": "a3e3c12514ba857724e244d3e74cea4f835ae4198fa1f077d062b6e46ccb9cbb",
    "src/llama-model.cpp": "e2e3f2e5fbfab3606ac68e04bd83b1ab7d82c67e11e3cc5b60fc64bc82a7eb28",
    "src/models/longcat-flash-ngram.cpp": "ae2ff5fded789c3af903f843e75fa23d1e4958b20c1bda6649c7b03c12956626",
}

TARGET = "src/models/longcat-flash-ngram.cpp"

def fail(msg: str):
    raise SystemExit(f"STOP: {msg}")

def run(root: Path, *args: str) -> str:
    p = subprocess.run(
        list(args),
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        fail(f"command failed ({' '.join(args)}):\n{p.stdout}{p.stderr}")
    return p.stdout.strip()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        fail(f"{label}: expected anchor exactly once, found {n}")
    return text.replace(old, new, 1)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    if not (root / ".git").exists():
        fail(f"not a git checkout: {root}")

    branch = run(root, "git", "branch", "--show-current")
    if branch != "longcat-sparse":
        fail(f"expected branch longcat-sparse, got {branch!r}")

    head = run(root, "git", "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        fail(f"expected HEAD {EXPECTED_HEAD}, got {head}")

    # Guard the entire current Gate-4 working tree by exact hashes.
    for rel, expected in EXPECTED_CURRENT_SHA256.items():
        path = root / rel
        if not path.is_file():
            fail(f"missing expected Gate-4 file: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"{rel}: expected SHA256 {expected}, got {actual}")

    p = root / TARGET
    s = p.read_text(encoding="utf-8")

    old = '''                    const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                    ggml_build_forward_expand(
                        gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                    if (sparse_active) {
'''
    new = '''                    // ggml_set_rows() currently accepts F32/F16 source rows,
                    // not BF16. Preserve the BF16-projected/RoPE'd values above,
                    // then widen exactly for the cache write. The destination
                    // indexer cache remains BF16.
                    ggml_tensor * indexer_k_store =
                        ggml_cast(ctx0, indexer_k, GGML_TYPE_F32);

                    const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                    ggml_build_forward_expand(
                        gf, mctx_lid->cpy_k(ctx0, indexer_k_store, k_idxs_lid, il));

                    if (sparse_active) {
'''
    s = replace_once(s, old, new, "LongCat BF16 indexer cache store")
    p.write_text(s, encoding="utf-8", newline="\n")

    pcheck = subprocess.run(
        ["git", "diff", "--check", "--", TARGET],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if pcheck.returncode != 0:
        fail(f"git diff --check failed:\n{pcheck.stdout}{pcheck.stderr}")

    print("GATE-4 BF16 INDEXER STORE FIX: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    print(f"{TARGET} SHA256 {sha256_file(p)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
