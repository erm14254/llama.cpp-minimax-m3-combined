#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"

EXPECTED_V2_SHA256 = {
    "src/llama-kv-cache.h": "ca105174c295f2c526cff1be0da298d2ea8120b0da5d40a4f72995ab0edd5a08",
    "src/llama-kv-cache.cpp": "a8c80745251ae5c48fcc74a0022c2c86bb875ac6f500023177dcdb8fbe1b7e05",
    "src/llama-kv-cache-dsa.cpp": "60735775d1a120c5b1d310000dd7f2515b0b975de0b61e86d1ec5120ff377e32",
    "src/llama-graph.h": "ca9108bcfea780beaa035c7b127428ef2d49a0b83253c14848876e2c20cd0b25",
    "src/llama-graph.cpp": "a3e3c12514ba857724e244d3e74cea4f835ae4198fa1f077d062b6e46ccb9cbb",
    "src/llama-model.cpp": "e2e3f2e5fbfab3606ac68e04bd83b1ab7d82c67e11e3cc5b60fc64bc82a7eb28",
    "src/models/longcat-flash-ngram.cpp": "ae2ff5fded789c3af903f843e75fa23d1e4958b20c1bda6649c7b03c12956626",
}

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

    for rel, expected in EXPECTED_V2_SHA256.items():
        path = root / rel
        if not path.is_file():
            fail(f"missing expected v2 file: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"{rel}: expected v2 SHA256 {expected}, got {actual}")

    p = root / "src/llama-kv-cache.h"
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '''    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
''',
        '''    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_longcat_lsa_mask(
            ggml_tensor * dst,
            const llama_ubatch * ubatch,
            uint32_t num_init_tokens,
            uint32_t num_local_tokens) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
''',
        "llama-kv-cache_context LongCat LSA declaration",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    p = root / "src/llama-kv-cache.cpp"
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '''void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
''',
        '''void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_longcat_lsa_mask(
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        uint32_t num_init_tokens,
        uint32_t num_local_tokens) const {
    kv->set_input_longcat_lsa_mask(
        dst, ubatch, num_init_tokens, num_local_tokens);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
''',
        "llama-kv-cache_context LongCat LSA forwarding implementation",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    pcheck = subprocess.run(
        ["git", "diff", "--check", "--",
         "src/llama-kv-cache.h", "src/llama-kv-cache.cpp"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if pcheck.returncode != 0:
        fail(f"git diff --check failed:\n{pcheck.stdout}{pcheck.stderr}")

    print("GATE-4 V2 CONTEXT FORWARDER FIX: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    for rel in ("src/llama-kv-cache.h", "src/llama-kv-cache.cpp"):
        print(f"{rel} SHA256 {sha256_file(root / rel)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
