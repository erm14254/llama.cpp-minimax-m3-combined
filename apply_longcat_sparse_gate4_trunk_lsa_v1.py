#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

EXPECTED_HEAD = "cb7729cf18088ae6cd6d9cac52e3ee536be02dc4"

TARGETS = (
    "src/llama-kv-cache.h",
    "src/llama-kv-cache.cpp",
    "src/llama-kv-cache-dsa.cpp",
    "src/llama-graph.h",
    "src/llama-graph.cpp",
    "src/llama-model.cpp",
    "src/models/longcat-flash-ngram.cpp",
)

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

def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        fail(f"{label}: expected anchor exactly once, found {n}")
    return text.replace(old, new, 1)

def sha256_file(path: Path) -> str:
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
    if branch != "longcat-sparse":
        fail(f"expected branch longcat-sparse, got {branch!r}")

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

    p = paths["src/llama-kv-cache.h"]
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '''    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
''',
        '''    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_longcat_lsa_mask(
            ggml_tensor * dst,
            const llama_ubatch * ubatch,
            uint32_t num_init_tokens,
            uint32_t num_local_tokens) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;
''',
        "llama-kv-cache.h LongCat LSA mask declaration",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    p = paths["src/llama-kv-cache.cpp"]
    s = p.read_text(encoding="utf-8")
    anchor = '''void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
'''
    impl = '''void llama_kv_cache::set_input_longcat_lsa_mask(
        ggml_tensor * dst,
        const llama_ubatch * ubatch,
        uint32_t num_init_tokens,
        uint32_t num_local_tokens) const {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->is_pos_2d());

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3];

    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(n_tokens % n_stream == 0);
    GGML_ASSERT(dst->ne[1] == n_tokens / n_stream);
    GGML_ASSERT(dst->ne[2] == 1);

    float * data = (float *) dst->data;
    std::fill(data, data + ggml_nelements(dst), -INFINITY);

    const int64_t n_tps = n_tokens / n_stream;

    // HF LongCat defines sink/local membership by valid-token rank, not by
    // physical ring-buffer cell index. Reconstruct that rank per query from
    // the cache metadata. Empty cells, other sequences, and future positions
    // remain -inf.
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;
            GGML_ASSERT(ubatch->n_seq_id[i] > 0);

            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

            const llama_pos p1 = ubatch->pos[i];
            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

            std::vector<std::pair<llama_pos, uint32_t>> visible;
            visible.reserve(std::min<int64_t>(n_kv, cells.size()));

            for (uint32_t j = 0; j < (uint32_t) n_kv; ++j) {
                if (j >= cells.size() || cells.is_empty(j)) {
                    continue;
                }
                if (!cells.seq_has(j, seq_id)) {
                    continue;
                }

                const llama_pos p0 = cells.pos_get(j);
                if (p0 > p1) {
                    continue;
                }

                visible.emplace_back(p0, j);
            }

            std::sort(visible.begin(), visible.end(),
                    [](const auto & a, const auto & b) {
                        return a.first < b.first || (a.first == b.first && a.second < b.second);
                    });

            const int64_t idst = n_kv*i;

            for (const auto & [_, j] : visible) {
                data[idst + j] = 0.0f;
            }

            const size_t n_visible = visible.size();
            const size_t n_init = std::min<size_t>(num_init_tokens, n_visible);
            for (size_t r = 0; r < n_init; ++r) {
                data[idst + visible[r].second] = INFINITY;
            }

            const size_t local_begin =
                n_visible > num_local_tokens ? n_visible - num_local_tokens : 0;
            for (size_t r = local_begin; r < n_visible; ++r) {
                data[idst + visible[r].second] = INFINITY;
            }
        }
    }
}

'''
    if s.count(anchor) != 1:
        fail(f"llama-kv-cache.cpp LongCat LSA insertion anchor count={s.count(anchor)}")
    s = s.replace(anchor, impl + anchor, 1)
    p.write_text(s, encoding="utf-8", newline="\n")

    p = paths["src/llama-kv-cache-dsa.cpp"]
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '''    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;
''',
        '''    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.rope_type =
        model.arch == LLM_ARCH_LONGCAT_FLASH_SPARSE ? LLAMA_ROPE_TYPE_NORM : LLAMA_ROPE_TYPE_NEOX;
''',
        "llama-kv-cache-dsa.cpp LongCat LID rope type",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    p = paths["src/llama-graph.h"]
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '''    ggml_tensor * self_k_rot_lid = nullptr;

    const llama_hparams hparams;
''',
        '''    ggml_tensor * self_k_rot_lid = nullptr;

    bool longcat_lsa = false;

    const llama_hparams hparams;
''',
        "llama-graph.h LongCat DSA input flag",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    p = paths["src/llama-graph.cpp"]
    s = p.read_text(encoding="utf-8")

    s = replace_once(
        s,
        '''void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}
''',
        '''void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    if (longcat_lsa) {
        GGML_ASSERT(cparams.causal_attn);
        mctx->get_lid()->set_input_longcat_lsa_mask(
            self_kq_mask_lid,
            ubatch,
            hparams.indexer_init_tokens,
            hparams.indexer_local_tokens);
    } else {
        mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);
    }

    if (self_k_rot_lid) {
        mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
    }
}
''',
        "llama-graph.cpp DSA set_input",
    )

    old_sparse_mask = '''    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
'''
    new_sparse_mask = '''    const auto & kq_mask = inp->get_kq_mask_mla();

    // nullptr top_k is the exact <=index_topk full-attention fast path used by
    // LongCat. DSA memory still stores indexer K history for a later crossing.
    ggml_tensor * kq_mask_top_k = kq_mask;

    if (top_k) {
        ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

        kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);

        ggml_tensor * sparse_only = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

        sparse_only = ggml_view_4d(ctx0, sparse_only, sparse_only->ne[1], sparse_only->ne[2], 1, sparse_only->ne[3], sparse_only->nb[2], sparse_only->nb[3], sparse_only->nb[3], 0);

        kq_mask_top_k = ggml_add(ctx0, sparse_only, kq_mask);
    }
'''
    s = replace_once(s, old_sparse_mask, new_sparse_mask, "llama-graph.cpp nullable DSA top_k")

    s = replace_once(
        s,
        '''    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);
''',
        '''    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);
    inp->longcat_lsa = arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);
''',
        "llama-graph.cpp DSA LongCat flag",
    )

    s = replace_once(
        s,
        '''        // ensure that mask type matches fused lightning indexer use (requires f16 mask)
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
''',
        '''        // GLM fused LID requires F16. LongCat's mask carries +inf
        // sink/local bias and is consumed by the explicit FP32 scoring path.
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = inp->longcat_lsa ? false : cparams.fused_lid;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid =
            inp->longcat_lsa ? nullptr : mctx_cur->get_lid()->build_input_k_rot(ctx0);
''',
        "llama-graph.cpp LongCat F32 LID mask",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    p = paths["src/llama-model.cpp"]
    s = p.read_text(encoding="utf-8")
    s = replace_once(
        s,
        '''        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
            {
''',
        '''        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_LONGCAT_FLASH_SPARSE:
            {
''',
        "llama-model.cpp Sparse DSA memory case",
    )
    s = replace_once(
        s,
        '''                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) { return il < hparams.n_layer() && (arch != LLM_ARCH_GLM_DSA || hparams.is_indexer_full(il)); };
''',
        '''                    llama_kv_cache::layer_filter_cb filter_lid = [&](uint32_t il) {
                        const bool owner_filtered =
                            arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;
                        return il < hparams.n_layer() &&
                               (!owner_filtered || hparams.is_indexer_full(il));
                    };
''',
        "llama-model.cpp Sparse owner-only LID filter",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    p = paths["src/models/longcat-flash-ngram.cpp"]
    s = p.read_text(encoding="utf-8")

    s = replace_once(
        s,
        '''#include "../llama-graph.h"
#include "../llama-model.h"
''',
        '''#include "../llama-graph.h"
#include "../llama-kv-cache-dsa.h"
#include "../llama-model.h"
''',
        "longcat include DSA cache",
    )

    s = replace_once(
        s,
        '''    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // MLA LoRA scaling factors (LongCat-Flash-specific, not in DeepSeek2)
''',
        '''    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    const bool longcat_lsa = arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;
    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = n_embd_head_qk_rope;
    const int64_t n_embd_indexer_head_nope =
        n_embd_indexer_head - n_embd_indexer_head_rope;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    // MLA LoRA scaling factors (LongCat-Flash-specific, not in DeepSeek2)
''',
        "longcat LSA constants",
    )

    s = replace_once(
        s,
        '''    auto * inp_attn_k = build_attn_inp_k();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
''',
        '''    llm_graph_input_attn_k * inp_attn_k = nullptr;
    llm_graph_input_attn_k_dsa * inp_attn_dsa = nullptr;
    if (longcat_lsa) {
        inp_attn_dsa = build_attn_inp_k_dsa();
    } else {
        inp_attn_k = build_attn_inp_k();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * prev_top_k = nullptr;

    for (int il = 0; il < n_layer; ++il) {
''',
        "longcat attention input selection",
    )

    old_q = '''            ggml_tensor * q = NULL;

            if (model.layers[il].wq_a) {
                // LoRA Q path
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                // MLA LoRA scaling: q *= sqrt(hidden_size / q_lora_rank)
                q = ggml_scale(ctx0, q, mla_scale_q);
                cb(q, "q_scaled", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            // split Q into nope and rope parts
'''
    new_q = '''            ggml_tensor * q = NULL;
            ggml_tensor * q_lora = nullptr;

            if (model.layers[il].wq_a) {
                q_lora = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q_lora, "q", il);

                q_lora = build_norm(q_lora, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q_lora, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q_lora);
                cb(q, "q", il);

                q = ggml_scale(ctx0, q, mla_scale_q);
                cb(q, "q_scaled", il);
            } else {
                q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);
            }

            ggml_tensor * top_k = nullptr;

            if (longcat_lsa) {
                GGML_ASSERT(inp_attn_dsa != nullptr);
                GGML_ASSERT(q_lora != nullptr);
                GGML_ASSERT(hparams.indexer_cli_factor == 2);

                const bool indexer_owner = hparams.is_indexer_full((uint32_t) il);
                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
                const uint32_t n_kv_lid = mctx_lid->get_n_kv();
                const bool sparse_active = n_kv_lid > n_indexer_top_k;

                if (indexer_owner) {
                    GGML_ASSERT(model.layers[il].indexer_k_norm);
                    GGML_ASSERT(model.layers[il].indexer_proj);
                    GGML_ASSERT(model.layers[il].indexer_attn_k);
                    GGML_ASSERT(model.layers[il].indexer_attn_q_b);

                    ggml_tensor * indexer_hidden =
                        ggml_cast(ctx0, cur, GGML_TYPE_BF16);
                    ggml_tensor * indexer_k =
                        ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, indexer_hidden);
                    cb(indexer_k, "lsa_indexer_k_proj", il);

                    indexer_k = ggml_rms_norm(
                        ctx0, indexer_k, hparams.indexer_k_norm_eps);
                    indexer_k = ggml_mul(
                        ctx0, indexer_k, model.layers[il].indexer_k_norm);
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_BF16);
                    cb(indexer_k, "lsa_indexer_k_norm", il);

                    ggml_tensor * indexer_k_pe =
                        ggml_view_3d(ctx0, indexer_k,
                            n_embd_indexer_head_rope, 1, n_tokens,
                            ggml_row_size(indexer_k->type, n_embd_indexer_head),
                            ggml_row_size(indexer_k->type, n_embd_indexer_head), 0);

                    ggml_tensor * indexer_k_nope =
                        ggml_view_3d(ctx0, indexer_k,
                            n_embd_indexer_head_nope, 1, n_tokens,
                            ggml_row_size(indexer_k->type, n_embd_indexer_head),
                            ggml_row_size(indexer_k->type, n_embd_indexer_head),
                            ggml_row_size(indexer_k->type, n_embd_indexer_head_rope));

                    indexer_k_pe = ggml_rope_ext(
                        ctx0, indexer_k_pe, inp_pos, nullptr,
                        n_embd_indexer_head_rope, LLAMA_ROPE_TYPE_NORM,
                        n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                    indexer_k = ggml_concat(
                        ctx0, indexer_k_pe, indexer_k_nope, 0);
                    indexer_k = ggml_cast(ctx0, indexer_k, GGML_TYPE_BF16);
                    cb(indexer_k, "lsa_indexer_k", il);

                    const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                    ggml_build_forward_expand(
                        gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                    if (sparse_active) {
                        ggml_tensor * indexer_q_in =
                            ggml_cast(ctx0, q_lora, GGML_TYPE_BF16);
                        ggml_tensor * indexer_q =
                            ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, indexer_q_in);
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_BF16);

                        ggml_tensor * indexer_q_pe =
                            ggml_view_3d(ctx0, indexer_q,
                                n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                                ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);

                        ggml_tensor * indexer_q_nope =
                            ggml_view_3d(ctx0, indexer_q,
                                n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                                ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                                ggml_row_size(indexer_q->type, n_embd_indexer_head_rope));

                        indexer_q_pe = ggml_rope_ext(
                            ctx0, indexer_q_pe, inp_pos, nullptr,
                            n_embd_indexer_head_rope, LLAMA_ROPE_TYPE_NORM,
                            n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);

                        indexer_q = ggml_concat(
                            ctx0, indexer_q_pe, indexer_q_nope, 0);
                        indexer_q = ggml_cast(ctx0, indexer_q, GGML_TYPE_BF16);
                        cb(indexer_q, "lsa_indexer_q", il);

                        ggml_tensor * indexer_weights =
                            ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                        indexer_weights = ggml_scale(
                            ctx0, indexer_weights,
                            1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                        cb(indexer_weights, "lsa_indexer_weights", il);

                        ggml_tensor * indexer_k_cached = mctx_lid->get_k(ctx0, il);

                        const auto n_stream = indexer_k_cached->ne[3];
                        indexer_q = ggml_view_4d(
                            ctx0, indexer_q,
                            indexer_q->ne[0], indexer_q->ne[1],
                            indexer_q->ne[2] / n_stream, n_stream,
                            indexer_q->nb[1], indexer_q->nb[2],
                            indexer_q->nb[3] / n_stream, 0);
                        indexer_weights = ggml_view_4d(
                            ctx0, indexer_weights,
                            indexer_weights->ne[0],
                            indexer_weights->ne[1] / n_stream,
                            indexer_weights->ne[2], n_stream,
                            indexer_weights->nb[1],
                            indexer_weights->nb[2] / n_stream,
                            indexer_weights->nb[3] / n_stream, 0);

                        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                        indexer_k_cached =
                            ggml_permute(ctx0, indexer_k_cached, 0, 2, 1, 3);

                        ggml_tensor * indexer_kq =
                            ggml_mul_mat(ctx0, indexer_k_cached, indexer_q);
                        ggml_mul_mat_set_prec(indexer_kq, GGML_PREC_F32);
                        cb(indexer_kq, "lsa_indexer_kq", il);

                        indexer_kq = ggml_cont(
                            ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));

                        ggml_tensor * indexer_score =
                            ggml_relu(ctx0, indexer_kq);
                        indexer_score =
                            ggml_mul(ctx0, indexer_score, indexer_weights);
                        indexer_score =
                            ggml_sum_rows(ctx0, indexer_score);
                        indexer_score = ggml_cont(
                            ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));

                        indexer_score = ggml_add(
                            ctx0, indexer_score, inp_attn_dsa->get_kq_mask_lid());
                        cb(indexer_score, "lsa_indexer_score", il);

                        const uint32_t n_top_k =
                            std::min<uint32_t>(
                                (uint32_t) indexer_score->ne[0],
                                n_indexer_top_k);
                        GGML_ASSERT(n_top_k == n_indexer_top_k);

                        top_k = ggml_cont(
                            ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                        prev_top_k = top_k;
                        cb(top_k, "lsa_top_k_owner", il);
                    } else {
                        prev_top_k = nullptr;
                        cb(indexer_k, "lsa_full_owner", il);
                    }
                } else {
                    if (sparse_active) {
                        GGML_ASSERT(prev_top_k != nullptr &&
                            "LongCat CLI reuse block must follow an owner top-K");
                        top_k = prev_top_k;
                        cb(top_k, "lsa_top_k_reuse", il);
                    } else {
                        GGML_ASSERT(prev_top_k == nullptr);
                        cb(cur, "lsa_full_reuse", il);
                    }
                }
            }

            // split Q into nope and rope parts
'''
    s = replace_once(s, old_q, new_q, "longcat Q path + LSA indexer")

    s = replace_once(
        s,
        '''            cur = build_attn(inp_attn_k,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
''',
        '''            if (longcat_lsa) {
                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr,
                        model.layers[il].wv_b, top_k, kq_scale, il);
            } else {
                cur = build_attn(inp_attn_k,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr,
                        model.layers[il].wv_b, kq_scale, il);
            }
''',
        "longcat sparse/dense attention dispatch",
    )
    p.write_text(s, encoding="utf-8", newline="\n")

    must_contain = {
        "src/llama-kv-cache.cpp": (
            "set_input_longcat_lsa_mask",
            "data[idst + visible[r].second] = INFINITY;",
        ),
        "src/llama-graph.cpp": (
            "inp->longcat_lsa = arch == LLM_ARCH_LONGCAT_FLASH_SPARSE;",
            "if (top_k) {",
        ),
        "src/llama-model.cpp": (
            "case LLM_ARCH_LONGCAT_FLASH_SPARSE:",
            "arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_LONGCAT_FLASH_SPARSE",
        ),
        "src/models/longcat-flash-ngram.cpp": (
            'cb(top_k, "lsa_top_k_owner", il);',
            'cb(top_k, "lsa_top_k_reuse", il);',
            "ggml_rms_norm(",
        ),
    }
    for rel, needles in must_contain.items():
        text = paths[rel].read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                fail(f"post-patch sanity missing {needle!r} in {rel}")

    pcheck = subprocess.run(
        ["git", "diff", "--check", "--", *TARGETS],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if pcheck.returncode != 0:
        fail(f"git diff --check failed:\n{pcheck.stdout}{pcheck.stderr}")

    print("GATE-4 LONGCAT TRUNK LSA: APPLIED")
    print(f"branch {branch}")
    print(f"HEAD {head}")
    print("git diff --check PASS")
    for rel in TARGETS:
        print(f"{rel} SHA256 {sha256_file(paths[rel])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
