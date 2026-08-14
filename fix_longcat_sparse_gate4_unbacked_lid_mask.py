#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"
TARGET = "src/llama-graph.cpp"
EXPECTED_SHA256 = "a3e3c12514ba857724e244d3e74cea4f835ae4198fa1f077d062b6e46ccb9cbb"

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

    old = '''    if (longcat_lsa) {
        GGML_ASSERT(cparams.causal_attn);
        mctx->get_lid()->set_input_longcat_lsa_mask(
            self_kq_mask_lid,
            ubatch,
            hparams.indexer_init_tokens,
            hparams.indexer_local_tokens);
    } else {
        mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);
    }
'''
    new = '''    if (longcat_lsa) {
        GGML_ASSERT(cparams.causal_attn);

        // <= index_topk, the LongCat graph keeps indexer K history but bypasses
        // indexer scoring entirely. In that graph the LID score mask is not
        // reachable and therefore has no backend buffer. Only populate it when
        // the sparse score path actually made it into the allocated graph.
        if (self_kq_mask_lid && self_kq_mask_lid->buffer) {
            mctx->get_lid()->set_input_longcat_lsa_mask(
                self_kq_mask_lid,
                ubatch,
                hparams.indexer_init_tokens,
                hparams.indexer_local_tokens);
        }
    } else {
        mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);
    }
'''
    text = replace_once(text, old, new, "LongCat unbacked LID mask guard")
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

    print("GATE-4 UNBACKED LID MASK FIX: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    print(f"{TARGET} SHA256 {sha256_file(path)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
