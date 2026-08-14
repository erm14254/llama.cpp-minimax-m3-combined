#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "484db978356bcff6e2c53f7bca6fa09f5aa8087d"
TARGETS = (
    "src/llama-memory.h",
    "src/llama-memory.cpp",
    "src/llama-context.cpp",
)

def fail(msg: str):
    raise SystemExit(f"STOP: {msg}")

def run(root: Path, *args: str) -> str:
    p = subprocess.run(list(args), cwd=root, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        fail(f"command failed ({' '.join(args)}):\n{p.stdout}{p.stderr}")
    return p.stdout.strip()

def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        fail(f"{label}: expected anchor exactly once, found {n}")
    return text.replace(old, new, 1)

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ns = ap.parse_args()
    root = Path(ns.root).resolve()

    if not (root / ".git").exists():
        fail(f"not a git checkout: {root}")

    branch = run(root, "git", "branch", "--show-current")
    if not branch:
        fail("detached HEAD is not allowed")

    head = run(root, "git", "rev-parse", "HEAD")
    if head != EXPECTED_HEAD:
        fail(f"expected HEAD {EXPECTED_HEAD}, got {head}")

    dirty = run(root, "git", "status", "--porcelain", "--", *TARGETS)
    if dirty:
        fail(f"target files are dirty before patch:\n{dirty}")

    paths = {rel: root / rel for rel in TARGETS}
    for rel, path in paths.items():
        if not path.is_file():
            fail(f"missing target file: {rel}")

    # src/llama-memory.h
    p = paths["src/llama-memory.h"]
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '#include "llama.h"\n#include "llama-graph.h"\n',
        '#include "llama.h"\n#include "llama-arch.h"\n#include "llama-graph.h"\n',
        "llama-memory.h include",
    )
    s = replace_once(
        s,
        "#include <map>\n#include <memory>\n#include <functional>\n",
        "#include <map>\n#include <memory>\n#include <functional>\n#include <string>\n",
        "llama-memory.h string include",
    )
    s = replace_once(
        s,
        '''struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;
};

enum llama_memory_status {
''',
        '''struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;
};

// Resolve architecture-specific cache constraints before constructing memory.
// Returns false and sets error when the requested cache is unsupported.
bool llama_memory_params_resolve(
        llm_arch arch, llama_memory_params & params, bool & promoted, std::string & error);

enum llama_memory_status {
''',
        "llama-memory.h resolver declaration",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    # src/llama-memory.cpp
    p = paths["src/llama-memory.cpp"]
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '#include "llama-memory.h"\n\nllama_memory_status llama_memory_status_combine',
        '''#include "llama-memory.h"

bool llama_memory_params_resolve(
        llm_arch arch, llama_memory_params & params, bool & promoted, std::string & error) {
    promoted = false;
    error.clear();

    if (arch != LLM_ARCH_LONGCAT_FLASH_SPARSE) {
        return true;
    }

    switch (params.type_k) {
        case GGML_TYPE_F16:
            params.type_k = GGML_TYPE_BF16;
            promoted = true;
            break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            break;
        default:
            error = std::string("unsupported LongCat-Flash-Sparse K cache type ") +
                    ggml_type_name(params.type_k) +
                    "; supported types are F16 (promoted to BF16), BF16, and F32";
            return false;
    }

    // Absorbed MLA stores the compressed KV state in the K cache and does not
    // use an independent V cache. Keep both requested cache types identical.
    params.type_v = params.type_k;
    return true;
}

llama_memory_status llama_memory_status_combine''',
        "llama-memory.cpp resolver",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    # src/llama-context.cpp
    p = paths["src/llama-context.cpp"]
    s = p.read_text(encoding="utf-8")
    old = '''        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
'''
    new = '''        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        bool cache_promoted = false;
        std::string cache_error;
        if (!llama_memory_params_resolve(model.arch, params_mem, cache_promoted, cache_error)) {
            throw std::runtime_error(cache_error);
        }
        if (cache_promoted) {
            LLAMA_LOG_WARN("%s: LongCat-Flash-Sparse absorbed MLA requires BF16 or F32 KV cache; "
                    "promoting the K/V cache from F16 to BF16\\n", __func__);
        }

        memory.reset(model.create_memory(params_mem, cparams));
'''
    s = replace_once(s, old, new, "llama-context.cpp resolver hook")
    p.write_text(s, encoding="utf-8", newline="\n")

    p = subprocess.run(["git", "diff", "--check", "--", *TARGETS], cwd=root,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        fail(f"git diff --check failed:\n{p.stdout}{p.stderr}")

    print("GATE-3 BF16 CACHE FIX: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    for rel in TARGETS:
        print(f"{rel} SHA256 {file_sha256(paths[rel])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
