#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"

TARGETS = (
    "src/llama-kv-cache.cpp",
    "src/models/longcat-flash-ngram.cpp",
)

AUDIT_MARKER = "LONGCAT_LSA_AUDIT"


def fail(msg: str) -> None:
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
    ap = argparse.ArgumentParser(
        description="Add temporary fail-closed Gate-4 LongCat LSA audit logging/assertions."
    )
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

    staged = run(root, "git", "diff", "--cached", "--name-only")
    if staged:
        fail(f"staged changes present; refusing temporary audit instrumentation:\n{staged}")

    for rel in TARGETS:
        path = root / rel
        if not path.is_file():
            fail(f"missing target file: {rel}")

    kv_path = root / "src/llama-kv-cache.cpp"
    model_path = root / "src/models/longcat-flash-ngram.cpp"

    kv = kv_path.read_text(encoding="utf-8")
    model = model_path.read_text(encoding="utf-8")

    if AUDIT_MARKER in kv or AUDIT_MARKER in model:
        fail("Gate-4 LSA audit instrumentation is already present")

    # Guard the expected current Gate-4 implementation state. These are semantic
    # anchors from the applied trunk-LSA + follow-up fixes, not provenance guesses.
    for needle in (
        "void llama_kv_cache::set_input_longcat_lsa_mask(",
        "data[idst + visible[r].second] = INFINITY;",
    ):
        if needle not in kv:
            fail(f"missing current Gate-4 KV-cache anchor: {needle!r}")

    for needle in (
        'cb(top_k, "lsa_top_k_owner", il);',
        'cb(top_k, "lsa_top_k_reuse", il);',
        "ggml_tensor * indexer_k_store =",
        "indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_F32);",
    ):
        if needle not in model:
            fail(f"missing current Gate-4 model anchor: {needle!r}")

    # llama-impl.h supplies LLAMA_LOG_DEBUG. Keep this explicit rather than
    # depending on a transitive include.
    model = replace_once(
        model,
        '''#include "../llama-graph.h"
#include "../llama-kv-cache-dsa.h"
#include "../llama-model.h"
''',
        '''#include "../llama-graph.h"
#include "../llama-impl.h"
#include "../llama-kv-cache-dsa.h"
#include "../llama-model.h"
''',
        "explicit llama-impl include",
    )

    # Validate the host-built LongCat mask itself. For each sparse query:
    #   - exactly the union of first init-token ranks and last local-token ranks
    #     is +inf;
    #   - the forced set fits inside the fixed 2048 budget.
    # Log only the final query in each ubatch so a long run remains concise.
    old_mask_tail = '''            const size_t local_begin =
                n_visible > num_local_tokens ? n_visible - num_local_tokens : 0;
            for (size_t r = local_begin; r < n_visible; ++r) {
                data[idst + visible[r].second] = INFINITY;
            }
'''
    new_mask_tail = '''            const size_t local_begin =
                n_visible > num_local_tokens ? n_visible - num_local_tokens : 0;
            for (size_t r = local_begin; r < n_visible; ++r) {
                data[idst + visible[r].second] = INFINITY;
            }

            if (n_visible > 2048) {
                size_t forced_count = 0;
                for (const auto & [_, j] : visible) {
                    if (std::isinf(data[idst + j]) && data[idst + j] > 0.0f) {
                        ++forced_count;
                    }
                }

                const size_t local_count = n_visible - local_begin;
                const size_t overlap =
                    local_begin < n_init ? n_init - local_begin : 0;
                const size_t expected_forced =
                    n_init + local_count - overlap;

                GGML_ASSERT(forced_count == expected_forced);
                GGML_ASSERT(expected_forced <= 2048);

                if (ii + 1 == n_tps) {
                    const llama_pos first_pos =
                        visible.empty() ? -1 : visible.front().first;
                    const llama_pos init_last_pos =
                        n_init ? visible[n_init - 1].first : -1;
                    const llama_pos local_first_pos =
                        local_count ? visible[local_begin].first : -1;
                    const llama_pos last_pos =
                        visible.empty() ? -1 : visible.back().first;

                    LLAMA_LOG_DEBUG(
                        "LONGCAT_LSA_AUDIT mask seq=%d query_pos=%d "
                        "visible=%zu forced=%zu init_pos=[%d,%d] "
                        "local_pos=[%d,%d]\\n",
                        (int) seq_id,
                        (int) p1,
                        n_visible,
                        forced_count,
                        (int) first_pos,
                        (int) init_last_pos,
                        (int) local_first_pos,
                        (int) last_pos);
                }
            }
'''
    kv = replace_once(
        kv, old_mask_tail, new_mask_tail, "LongCat mask audit"
    )

    old_owner = '''                        top_k = ggml_cont(
                            ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                        prev_top_k = top_k;
                        cb(top_k, "lsa_top_k_owner", il);
'''
    new_owner = '''                        top_k = ggml_cont(
                            ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                        GGML_ASSERT(top_k != nullptr);
                        GGML_ASSERT(top_k->ne[0] == n_indexer_top_k);

                        prev_top_k = top_k;
                        LLAMA_LOG_DEBUG(
                            "LONGCAT_LSA_AUDIT owner block=%d n_kv=%u "
                            "top_k=%lld tensor=%p\\n",
                            il,
                            n_kv_lid,
                            (long long) top_k->ne[0],
                            (void *) top_k);
                        cb(top_k, "lsa_top_k_owner", il);
'''
    model = replace_once(
        model, old_owner, new_owner, "LongCat sparse owner audit"
    )

    old_reuse = '''                        GGML_ASSERT(prev_top_k != nullptr &&
                            "LongCat CLI reuse block must follow an owner top-K");
                        top_k = prev_top_k;
                        cb(top_k, "lsa_top_k_reuse", il);
'''
    new_reuse = '''                        GGML_ASSERT(prev_top_k != nullptr &&
                            "LongCat CLI reuse block must follow an owner top-K");
                        top_k = prev_top_k;
                        GGML_ASSERT(top_k->ne[0] == n_indexer_top_k);

                        LLAMA_LOG_DEBUG(
                            "LONGCAT_LSA_AUDIT reuse block=%d owner_block=%d "
                            "n_kv=%u top_k=%lld tensor=%p\\n",
                            il,
                            il - 1,
                            n_kv_lid,
                            (long long) top_k->ne[0],
                            (void *) top_k);
                        cb(top_k, "lsa_top_k_reuse", il);
'''
    model = replace_once(
        model, old_reuse, new_reuse, "LongCat sparse reuse audit"
    )

    # Fail closed at the attention dispatch boundary: above 2048 Sparse must have
    # a non-null selection; at/below 2048 it must remain the exact full path.
    old_dispatch = '''            if (longcat_lsa) {
                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr,
                        model.layers[il].wv_b, top_k, kq_scale, il);
'''
    new_dispatch = '''            if (longcat_lsa) {
                GGML_ASSERT(inp_attn_dsa != nullptr);
                const uint32_t n_kv_lid =
                    inp_attn_dsa->mctx->get_lid()->get_n_kv();
                const bool sparse_expected =
                    n_kv_lid > n_indexer_top_k;
                GGML_ASSERT(sparse_expected == (top_k != nullptr));

                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr,
                        model.layers[il].wv_b, top_k, kq_scale, il);
'''
    model = replace_once(
        model, old_dispatch, new_dispatch, "LongCat fail-closed dispatch audit"
    )

    kv_path.write_text(kv, encoding="utf-8", newline="\n")
    model_path.write_text(model, encoding="utf-8", newline="\n")

    pcheck = subprocess.run(
        ["git", "diff", "--check", "--", *TARGETS],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if pcheck.returncode != 0:
        fail(f"git diff --check failed:\n{pcheck.stdout}{pcheck.stderr}")

    for rel in TARGETS:
        text = (root / rel).read_text(encoding="utf-8")
        if AUDIT_MARKER not in text:
            fail(f"audit marker missing after patch: {rel}")

    print("GATE-4 LONGCAT LSA AUDIT INSTRUMENTATION: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    for rel in TARGETS:
        print(f"{rel} SHA256 {sha256_file(root / rel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
