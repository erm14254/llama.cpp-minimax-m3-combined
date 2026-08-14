#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"
TARGET = "src/models/longcat-flash-ngram.cpp"
EXPECTED_SHA256 = "b5e80d70e6ece5027a1684a8ba8cbbbb4ad2041fea23502a61ec8f5d4dba6fb6"

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

    path = root / TARGET
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256:
        fail(f"{TARGET}: expected SHA256 {EXPECTED_SHA256}, got {actual}")

    text = path.read_text(encoding="utf-8")

    old_k = '''                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_BF16);
                    cb(indexer_k, "lsa_indexer_k_norm", il);

                    ggml_tensor * indexer_k_pe =
                        ggml_view_3d(ctx0, indexer_k,
'''
    new_k = '''                    // Match HF's BF16 projection/norm rounding point, but
                    // CPU ggml_rope supports only F16/F32. Widen the already-
                    // rounded BF16 values solely for RoPE, then round back.
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_BF16);
                    cb(indexer_k, "lsa_indexer_k_norm", il);
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_F32);

                    ggml_tensor * indexer_k_pe =
                        ggml_view_3d(ctx0, indexer_k,
'''
    text = replace_once(text, old_k, new_k, "LongCat indexer K RoPE widening")

    old_q = '''                        ggml_tensor * indexer_q =
                            ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, indexer_q_in);
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_BF16);

                        ggml_tensor * indexer_q_pe =
                            ggml_view_3d(ctx0, indexer_q,
'''
    new_q = '''                        ggml_tensor * indexer_q =
                            ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, indexer_q_in);
                        // Preserve BF16 Q projection rounding, then widen only
                        // for the CPU RoPE kernel exactly as for indexer K.
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_BF16);
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_F32);

                        ggml_tensor * indexer_q_pe =
                            ggml_view_3d(ctx0, indexer_q,
'''
    text = replace_once(text, old_q, new_q, "LongCat indexer Q RoPE widening")

    path.write_text(text, encoding="utf-8", newline="\n")

    p = subprocess.run(
        ["git", "diff", "--check", "--", TARGET],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        fail(f"git diff --check failed:\n{p.stdout}{p.stderr}")

    print("GATE-4 INDEXER ROPE TYPE FIX: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    print(f"{TARGET} SHA256 {sha256_file(path)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
