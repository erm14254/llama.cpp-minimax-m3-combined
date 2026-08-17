#include "debug.h"

#include "common.h"
#include "log.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <regex>
#include <string>
#include <vector>

struct common_debug_cb_user_data::impl {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;
    bool                    abort_on_nan{false};
};

common_debug_cb_user_data::common_debug_cb_user_data() : pimpl(std::make_unique<impl>()) {}
common_debug_cb_user_data::~common_debug_cb_user_data() = default;

common_debug_cb_user_data::common_debug_cb_user_data(common_params & params, const std::vector<std::string> & filter_patterns, bool abort_on_nan)
    : pimpl(std::make_unique<impl>())
{
    for (const auto & pattern : filter_patterns) {
        try {
            std::string anchored_pattern = "^" + pattern;
            pimpl->tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
        } catch (const std::regex_error & e) {
            throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
        }
    }
    pimpl->abort_on_nan = abort_on_nan;

    params.cb_eval           = common_debug_cb_eval;
    params.cb_eval_user_data = this;
}

static std::string common_ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static float common_ggml_get_float_value(const uint8_t * data,
                           ggml_type       type,
                           const size_t *  nb,
                           size_t          i0,
                           size_t          i1,
                           size_t          i2,
                           size_t          i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    float  v;
    if (type == GGML_TYPE_F16) {
        v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
    } else if (type == GGML_TYPE_F32) {
        v = *(const float *) &data[i];
    } else if (type == GGML_TYPE_I64) {
        v = (float) *(const int64_t *) &data[i];
    } else if (type == GGML_TYPE_I32) {
        v = (float) *(const int32_t *) &data[i];
    } else if (type == GGML_TYPE_I16) {
        v = (float) *(const int16_t *) &data[i];
    } else if (type == GGML_TYPE_I8) {
        v = (float) *(const int8_t *) &data[i];
    } else if (type == GGML_TYPE_BF16) {
        v = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
    } else {
        GGML_ABORT("fatal error");
    }
    return v;
}

#define INDENT "    "

// LONGCAT_GATE4_NAN_AUDIT: return whether any actual element is NaN.
// Do not infer NaN from the aggregate sum: LongCat LSA score tensors
// legitimately contain both +inf and -inf, whose sum itself is NaN.
static bool common_debug_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    uint64_t nan_count = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum += v;
                    if (std::isnan(v)) {
                        ++nan_count;
                    }
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG(INDENT "[\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2 * n) {
                LOG(INDENT INDENT "..., \n");
                i2 = ne[2] - n;
            }
            LOG(INDENT INDENT "[\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2 * n) {
                    LOG(INDENT INDENT INDENT "..., \n");
                    i1 = ne[1] - n;
                }
                LOG(INDENT INDENT INDENT "[");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2 * n) {
                        LOG("   ..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = common_ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG(", ");
                    }
                }
                LOG("  ],\n");
            }
            LOG(INDENT INDENT "],\n");
        }
        LOG(INDENT "]\n");
        LOG(INDENT "sum = %f\n", sum);
    }

    LOG(INDENT "nan_count = %llu\n", (unsigned long long) nan_count);

    // LONGCAT_FA_HEAD_AUDIT: dim1 is the attention-head axis for
    // Qcur/FLASH_ATTN_EXT output in this LongCat graph.
    if (ne[1] > 1 && ne[1] <= 64) {
        for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
            uint64_t head_nan    = 0;
            uint64_t head_posinf = 0;
            uint64_t head_neginf = 0;
            float head_max_abs   = 0.0f;

            for (int64_t i3 = 0; i3 < ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < ne[2]; ++i2) {
                    for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                        const float v = common_ggml_get_float_value(
                            data, type, nb, i0, i1, i2, i3);

                        if (std::isnan(v)) {
                            ++head_nan;
                        } else if (std::isinf(v)) {
                            if (v > 0.0f) {
                                ++head_posinf;
                            } else {
                                ++head_neginf;
                            }
                        } else {
                            head_max_abs = std::max(head_max_abs, std::fabs(v));
                        }
                    }
                }
            }

            LOG(INDENT "dim1[%lld]: nan=%llu +inf=%llu -inf=%llu max_abs=%g\n",
                (long long) i1,
                (unsigned long long) head_nan,
                (unsigned long long) head_posinf,
                (unsigned long long) head_neginf,
                (double) head_max_abs);
        }
    }
    return nan_count != 0;
}


// LONGCAT_HIDDEN_VECTOR_DUMP:
// When LONGCAT_HIDDEN_DUMP_DIR is set, dump HF-comparable LongCat 512-token
// diagnostic surfaces. Two modes:
//
//   full_sequence = false -> final-token row only, 3072 F32 = 12288 bytes.
//        This is the historical layout. Every frozen oracle, including the
//        attn0 residual 2c804a35..., is a final-row file, so these must not
//        change or the Phase 3a regression gate is invalidated.
//
//   full_sequence = true  -> all ne[1] rows in canonical token-major
//        [n_tokens, width] F32 order, plus a JSON sidecar. Used for the
//        block-0 MLA stage surfaces, where the final token's attention output
//        depends on K/V at every earlier position.
//
// Rows are always read through the tensor's own nb[] strides, so a view or
// permuted layout can never be mistaken for an arithmetic divergence.
struct common_debug_longcat_dump_spec {
    std::string filename;
    bool        full_sequence = false;
    int64_t     expect_ne0    = 0;
};

static bool common_debug_longcat_dump_spec_for(
        const ggml_tensor * t,
        common_debug_longcat_dump_spec & spec) {
    const std::string tensor_name = t->name;

    // LONGCAT_MLA_STAGE_VECTOR_DUMP: physical block-0 MLA boundaries.
    // Filenames deliberately match the HF capture surface names so the
    // comparator can pair the two sides directly.
    static const struct {
        const char * name;
        const char * file;
        int64_t      ne0;
    } mla_surfaces[] = {
        { "q_a_proj-0",   "q_a_proj.bin",           1536 },
        { "q_a_norm-0",   "q_a_layernorm.bin",      1536 },
        { "q_b_proj-0",   "q_b_proj.bin",           6144 },
        { "kv_cmpr_pe-0", "kv_a_proj_with_mqa.bin",  576 },
        { "kv_a_norm-0",  "kv_a_layernorm.bin",      512 },
        { "attn_out-0",   "o_proj.bin",             3072 },
        // LONGCAT_ATTN_PATH_STAGE_SURFACE (localization, dump-only):
        // post-RoPE Q/K (2D cont copies), the post-scale compressed-KV
        // cache input, and the pre-wo attention context from build_attn.
        { "q_pe_rope-0",      "q_pe_rope.bin",      2048 },
        { "k_pe_rope-0",      "k_pe_rope.bin",        64 },
        { "kv_cmpr_scaled-0", "kv_cmpr_scaled.bin",  512 },
        { "kqv_out-0",        "kqv_out.bin",        4096 },
    };

    for (const auto & surface : mla_surfaces) {
        if (tensor_name == surface.name) {
            spec.filename      = surface.file;
            spec.full_sequence = true;
            spec.expect_ne0    = surface.ne0;
            return true;
        }
    }

    spec.full_sequence = false;
    spec.expect_ne0    = 3072;

    if (tensor_name == "inp_embd_ngram") {
        spec.filename = "inp_embd_ngram.bin";
        return true;
    }

    if (tensor_name == "result_norm") {
        spec.filename = "result_norm.bin";
        return true;
    }

    // LONGCAT_LOGICAL0_STAGE_VECTOR_DUMP:
    // Additional logical-layer-0 parity boundaries.
    // l_out-1 intentionally remains handled by the existing logical_00 mapping.
    if (tensor_name == "ffn_inp-0") {
        spec.filename = "logical0_attn0_resid.bin";
        return true;
    }

    if (tensor_name == "l_out-0") {
        spec.filename = "logical0_mlp0_resid.bin";
        return true;
    }

    if (tensor_name == "ffn_inp-1") {
        spec.filename = "logical0_attn1_resid.bin";
        return true;
    }

    // LONGCAT_LOGICAL0_ATTN0_NORM_VECTOR_DUMP:
    // First normalized activation entering physical attention block 0.
    if (tensor_name == "attn_norm-0") {
        spec.filename = "logical0_attn0_norm.bin";
        return true;
    }

    for (int logical = 0; logical < 13; ++logical) {
        const int physical = 2 * logical + 1;
        if (tensor_name == "l_out-" + std::to_string(physical)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "logical_%02d.bin", logical);
            spec.filename = buf;
            return true;
        }
    }

    return false;
}

// LONGCAT_RESID_WALK_DUMP (causal-reset experiment):
//
// When LONGCAT_RESID_WALK_DUMP_DIR is set, the whole-sequence [512 x 3072]
// residual-stream boundaries are dumped in addition to (and independently of)
// the LONGCAT_HIDDEN_DUMP_DIR final-row set:
//
//   l_out-(2N+1) -> logical_NN_full.bin  for N = 0..13
//                   (N = 13 adds the previously-undumped logical layer 13,
//                    physical block 27)
//   h_nextn      -> result_norm_full.bin (the post-final-norm tensor BEFORE
//                   the inp_out_ids row filter at longcat-flash-ngram.cpp;
//                   the row-filtered node named result_norm is not
//                   whole-sequence, h_nextn is the same values pre-filter)
//
// Everything is dump-only; production arithmetic is untouched.
static bool common_debug_longcat_resid_walk_spec_for(
        const ggml_tensor * t,
        common_debug_longcat_dump_spec & spec) {
    const std::string tensor_name = t->name;

    spec.full_sequence = true;
    spec.expect_ne0    = 3072;

    if (tensor_name == "h_nextn") {
        spec.filename = "result_norm_full.bin";
        return true;
    }

    // Block-2 MLA-internal walk surfaces (physical block 2 attention, under
    // the dual reset). Per-surface widths; all full-sequence.
    static const struct {
        const char * name;
        const char * file;
        int64_t      ne0;
    } block2_mla[] = {
        { "q_a_proj-2",       "block2_q_a_proj_full.bin",       1536 },
        { "q_a_norm-2",       "block2_q_a_norm_full.bin",       1536 },
        { "q_b_proj-2",       "block2_q_b_proj_full.bin",       6144 },
        { "kv_cmpr_pe-2",     "block2_kv_a_proj_full.bin",       576 },
        { "kv_a_norm-2",      "block2_kv_a_norm_full.bin",       512 },
        { "kv_cmpr_scaled-2", "block2_kv_cmpr_scaled_full.bin",  512 },
        { "q_pe_rope-2",      "block2_q_pe_rope_full.bin",      2048 },
        { "k_pe_rope-2",      "block2_k_pe_rope_full.bin",        64 },
        { "kqv_out-2",        "block2_kqv_out_full.bin",        4096 },
    };
    for (const auto & surface : block2_mla) {
        if (tensor_name == surface.name) {
            spec.filename   = surface.file;
            spec.expect_ne0 = surface.ne0;
            return true;
        }
    }

    // Logical-block-1 sub-boundaries (physical blocks 2-3) for the
    // sub-boundary localization under the logical_00 reset. attn_norm-2's
    // input is the injected oracle in the reset run, so that surface is an
    // operator-isolated RMSNorm measurement.
    static const struct {
        const char * name;
        const char * file;
    } block1_surfaces[] = {
        { "attn_norm-2", "block1_attn0_norm_full.bin"  },
        // Pre-residual-add attention output (post-wo/o_proj): the
        // discriminator that separates block-2 attention from the residual
        // add under the dual reset.
        { "attn_out-2",  "block1_attn0_out_full.bin"   },
        { "ffn_inp-2",   "block1_attn0_resid_full.bin" },
        { "l_out-2",     "block1_mlp0_resid_full.bin"  },
        { "attn_norm-3", "block1_attn1_norm_full.bin"  },
        { "ffn_inp-3",   "block1_attn1_resid_full.bin" },
    };
    for (const auto & surface : block1_surfaces) {
        if (tensor_name == surface.name) {
            spec.filename = surface.file;
            return true;
        }
    }

    for (int logical = 0; logical < 14; ++logical) {
        const int physical = 2 * logical + 1;
        if (tensor_name == "l_out-" + std::to_string(physical)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "logical_%02d_full.bin", logical);
            spec.filename = buf;
            return true;
        }
    }

    return false;
}

// NOTE: an early-abort hook (LONGCAT_ABORT_AFTER_TENSOR) was attempted here
// and deliberately removed.
//
// Returning false from a ggml_backend_sched eval callback does NOT stop graph
// evaluation. In ggml_backend_sched_compute_splits the return value only
// breaks the node loop of the *current split*; the enclosing loop over splits
// continues and the rest of the graph still runs. See
// ggml/src/ggml-backend.cpp: "if (need && !sched->callback_eval(t, false, ...))
// { break; }".
//
// Worse, that break also skips the remaining nodes of the split it fires in,
// so anything downstream is computed from incomplete state -- a surface dumped
// after the trigger point would be silently invalid rather than absent.
//
// A full 512-token forward completes on the diagnostic hardware, so no abort
// is needed. Do not reintroduce one without a real ggml-level mechanism.

// True when this tensor has a dump mapping and dumping is enabled. The eval
// callback must request such tensors at ask time even if --tensor-filter does
// not cover them, otherwise the dump would silently never fire.
static bool common_debug_longcat_wants_dump(const ggml_tensor * t) {
    const char * dump_dir = std::getenv("LONGCAT_HIDDEN_DUMP_DIR");

    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return false;
    }

    common_debug_longcat_dump_spec spec;

    return common_debug_longcat_dump_spec_for(t, spec);
}

static bool common_debug_longcat_wants_resid_walk_dump(const ggml_tensor * t) {
    const char * dump_dir = std::getenv("LONGCAT_RESID_WALK_DUMP_DIR");

    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return false;
    }

    common_debug_longcat_dump_spec spec;

    return common_debug_longcat_resid_walk_spec_for(t, spec);
}

// LONGCAT_RESID_INJECT (causal-reset experiment -- oracle reset ONLY):
//
// When LONGCAT_RESID_INJECT_DIR is set, the logical-layer-0 output l_out-1
// (the [512 x 3072] F32 residual-stream boundary proven to be a complete
// causal cut: the residual stream is the only mutable inter-layer carrier on
// both the HF and C++ sides) is OVERWRITTEN with the HF full-sequence oracle
// logical_00_oracle.bin immediately after the node computes and before any
// dependent executes. This is NOT an arithmetic change and NOT evidence about
// C++ block quality; it answers one causal question: with the exact HF
// logical-layer-0 output supplied to the downstream C++ trunk, how much
// divergence does the downstream trunk itself regenerate?
//
// The oracle file's SHA256 is verified by the run-harness preflight; here the
// exact byte size, type, and contiguity are enforced, and any failure aborts
// the process (exit 87) so a partial injection can never produce a
// plausible-looking capture. The dump path runs AFTER injection in the eval
// callback, so the walk dump logical_00_full.bin records the injected bytes
// and serves as the byte-exact landing gate.
static bool common_debug_longcat_wants_resid_inject(const ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_RESID_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return false;
    }
    return strcmp(t->name, "l_out-1") == 0;
}

static void common_debug_maybe_inject_longcat_resid(ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_RESID_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    if (strcmp(t->name, "l_out-1") != 0) {
        return;
    }

    constexpr size_t expect_nbytes = (size_t) 512 * 3072 * 4;

    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
        ggml_nbytes(t) != expect_nbytes ||
        t->ne[0] != 3072 || t->ne[1] != 512) {
        LOG_ERR(
            "LONGCAT_RESID_INJECT ABORT: %s type=%s contig=%d nbytes=%zu "
            "ne={%lld,%lld,%lld,%lld} (expected F32 contiguous %zu, ne 3072x512)\n",
            t->name, ggml_type_name(t->type),
            ggml_is_contiguous(t) ? 1 : 0,
            ggml_nbytes(t),
            (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3],
            expect_nbytes);
        common_log_flush(common_log_main());
        std::exit(87);
    }

    static std::vector<uint8_t> cache;

    if (cache.empty()) {
        const std::string path = std::string(dir) + "/logical_00_oracle.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            LOG_ERR("LONGCAT_RESID_INJECT ABORT: cannot open %s\n", path.c_str());
            common_log_flush(common_log_main());
            std::exit(87);
        }
        cache.assign(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
        if (cache.size() != expect_nbytes) {
            LOG_ERR(
                "LONGCAT_RESID_INJECT ABORT: %s is %zu bytes, expected %zu\n",
                path.c_str(), cache.size(), expect_nbytes);
            common_log_flush(common_log_main());
            std::exit(87);
        }
    }

    ggml_backend_tensor_set(t, cache.data(), 0, expect_nbytes);
    LOG_INF("LONGCAT_RESID_INJECT: %s <- logical_00_oracle.bin (%zu bytes)\n",
            t->name, expect_nbytes);
}

// LONGCAT_ATTN_NORM2_INJECT (dual-reset experiment -- second oracle reset):
//
// When LONGCAT_ATTN_NORM2_INJECT_DIR is set, the block-2 pre-attention norm
// output attn_norm-2 is OVERWRITTEN with the HF layer-1 attn0_norm oracle
// (<dir>/attn0_norm.bin) immediately after the node computes. Combined with
// the l_out-1 reset, both block-2 attention operands (the normalized QKV
// input and the residual-stream operand) are exact in value, licensing a
// causal judgment of the block-2 attention implementation. The C++ tensor
// remains F32 carrying HF BF16-on-lattice values; the F32-vs-BF16 carrier is
// part of the implementation under test. Same fail-closed contract as the
// other injectors; the walk dump of attn_norm-2 records the injected bytes
// (second landing gate).
static bool common_debug_longcat_wants_attn_norm2_inject(const ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_ATTN_NORM2_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return false;
    }
    return strcmp(t->name, "attn_norm-2") == 0;
}

static void common_debug_maybe_inject_longcat_attn_norm2(ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_ATTN_NORM2_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    if (strcmp(t->name, "attn_norm-2") != 0) {
        return;
    }

    constexpr size_t expect_nbytes = (size_t) 512 * 3072 * 4;

    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
        ggml_nbytes(t) != expect_nbytes ||
        t->ne[0] != 3072 || t->ne[1] != 512) {
        LOG_ERR(
            "LONGCAT_ATTN_NORM2_INJECT ABORT: %s type=%s contig=%d nbytes=%zu "
            "ne={%lld,%lld,%lld,%lld} (expected F32 contiguous %zu, ne 3072x512)\n",
            t->name, ggml_type_name(t->type),
            ggml_is_contiguous(t) ? 1 : 0,
            ggml_nbytes(t),
            (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3],
            expect_nbytes);
        common_log_flush(common_log_main());
        std::exit(87);
    }

    static std::vector<uint8_t> cache;

    if (cache.empty()) {
        const std::string path = std::string(dir) + "/attn0_norm.bin";
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            LOG_ERR("LONGCAT_ATTN_NORM2_INJECT ABORT: cannot open %s\n", path.c_str());
            common_log_flush(common_log_main());
            std::exit(87);
        }
        cache.assign(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
        if (cache.size() != expect_nbytes) {
            LOG_ERR(
                "LONGCAT_ATTN_NORM2_INJECT ABORT: %s is %zu bytes, expected %zu\n",
                path.c_str(), cache.size(), expect_nbytes);
            common_log_flush(common_log_main());
            std::exit(87);
        }
    }

    ggml_backend_tensor_set(t, cache.data(), 0, expect_nbytes);
    LOG_INF("LONGCAT_ATTN_NORM2_INJECT: %s <- attn0_norm.bin (%zu bytes)\n",
            t->name, expect_nbytes);
}

// LONGCAT_PROJ_INJECT (quad-reset experiment -- exact projection-output
// resets): when LONGCAT_PROJ_INJECT_DIR is set, the block-2 root projection
// outputs q_a_proj-2 and kv_cmpr_pe-2 are OVERWRITTEN with the captured HF
// full-sequence values immediately after the nodes compute. Combined with
// the l_out-1 and attn_norm-2 resets, the LoRA norms become the next
// operators with byte-exact activation inputs. q_a_proj-2 feeds build_norm
// directly; kv_cmpr_pe-2's buffer is read by the kv_cmpr / k_pe views, so
// the overwrite propagates to all view consumers (R0 pattern). Same
// fail-closed contract; the walk dumps of both nodes are the landing gates.
static const struct {
    const char * name;
    const char * file;
    size_t       nbytes;
    int64_t      ne0;
} longcat_proj_inject_targets[] = {
    { "q_a_proj-2",   "q_a_proj.bin",           (size_t) 512 * 1536 * 4, 1536 },
    { "kv_cmpr_pe-2", "kv_a_proj_with_mqa.bin", (size_t) 512 *  576 * 4,  576 },
};

static bool common_debug_longcat_wants_proj_inject(const ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_PROJ_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return false;
    }
    for (const auto & target : longcat_proj_inject_targets) {
        if (strcmp(t->name, target.name) == 0) {
            return true;
        }
    }
    return false;
}

static void common_debug_maybe_inject_longcat_proj(ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_PROJ_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }

    static std::vector<uint8_t> cache[2];

    for (int i = 0; i < 2; ++i) {
        const auto & target = longcat_proj_inject_targets[i];
        if (strcmp(t->name, target.name) != 0) {
            continue;
        }

        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            ggml_nbytes(t) != target.nbytes ||
            t->ne[0] != target.ne0 || t->ne[1] != 512) {
            LOG_ERR(
                "LONGCAT_PROJ_INJECT ABORT: %s type=%s contig=%d nbytes=%zu "
                "ne={%lld,%lld,%lld,%lld} (expected F32 contiguous %zu, ne %lldx512)\n",
                t->name, ggml_type_name(t->type),
                ggml_is_contiguous(t) ? 1 : 0,
                ggml_nbytes(t),
                (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3],
                target.nbytes, (long long) target.ne0);
            common_log_flush(common_log_main());
            std::exit(87);
        }

        if (cache[i].empty()) {
            const std::string path = std::string(dir) + "/" + target.file;
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                LOG_ERR("LONGCAT_PROJ_INJECT ABORT: cannot open %s\n", path.c_str());
                common_log_flush(common_log_main());
                std::exit(87);
            }
            cache[i].assign(std::istreambuf_iterator<char>(f),
                            std::istreambuf_iterator<char>());
            if (cache[i].size() != target.nbytes) {
                LOG_ERR(
                    "LONGCAT_PROJ_INJECT ABORT: %s is %zu bytes, expected %zu\n",
                    path.c_str(), cache[i].size(), target.nbytes);
                common_log_flush(common_log_main());
                std::exit(87);
            }
        }

        ggml_backend_tensor_set(t, cache[i].data(), 0, target.nbytes);
        LOG_INF("LONGCAT_PROJ_INJECT: %s <- %s (%zu bytes)\n",
                t->name, target.file, target.nbytes);
        return;
    }
}

// LONGCAT_NORM_INJECT (hex-reset experiment -- exact norm-output resets):
// when LONGCAT_NORM_INJECT_DIR is set, the block-2 LoRA norm outputs
// q_a_norm-2 and kv_a_norm-2 are OVERWRITTEN with the captured HF values
// immediately after the nodes compute. Combined with the four upstream
// resets, q_b_proj-2 (wq_b GEMM) and kv_cmpr_scaled-2 (ggml_scale) become
// the next operators with byte-exact-in-value inputs. Same fail-closed
// contract; the walk dumps of both nodes are the landing gates.
static const struct {
    const char * name;
    const char * file;
    size_t       nbytes;
    int64_t      ne0;
} longcat_norm_inject_targets[] = {
    { "q_a_norm-2",  "q_a_layernorm.bin",  (size_t) 512 * 1536 * 4, 1536 },
    { "kv_a_norm-2", "kv_a_layernorm.bin", (size_t) 512 *  512 * 4,  512 },
};

static bool common_debug_longcat_wants_norm_inject(const ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_NORM_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return false;
    }
    for (const auto & target : longcat_norm_inject_targets) {
        if (strcmp(t->name, target.name) == 0) {
            return true;
        }
    }
    return false;
}

static void common_debug_maybe_inject_longcat_norm(ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_NORM_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }

    static std::vector<uint8_t> cache[2];

    for (int i = 0; i < 2; ++i) {
        const auto & target = longcat_norm_inject_targets[i];
        if (strcmp(t->name, target.name) != 0) {
            continue;
        }

        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            ggml_nbytes(t) != target.nbytes ||
            t->ne[0] != target.ne0 || t->ne[1] != 512) {
            LOG_ERR(
                "LONGCAT_NORM_INJECT ABORT: %s type=%s contig=%d nbytes=%zu "
                "ne={%lld,%lld,%lld,%lld} (expected F32 contiguous %zu, ne %lldx512)\n",
                t->name, ggml_type_name(t->type),
                ggml_is_contiguous(t) ? 1 : 0,
                ggml_nbytes(t),
                (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3],
                target.nbytes, (long long) target.ne0);
            common_log_flush(common_log_main());
            std::exit(87);
        }

        if (cache[i].empty()) {
            const std::string path = std::string(dir) + "/" + target.file;
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                LOG_ERR("LONGCAT_NORM_INJECT ABORT: cannot open %s\n", path.c_str());
                common_log_flush(common_log_main());
                std::exit(87);
            }
            cache[i].assign(std::istreambuf_iterator<char>(f),
                            std::istreambuf_iterator<char>());
            if (cache[i].size() != target.nbytes) {
                LOG_ERR(
                    "LONGCAT_NORM_INJECT ABORT: %s is %zu bytes, expected %zu\n",
                    path.c_str(), cache[i].size(), target.nbytes);
                common_log_flush(common_log_main());
                std::exit(87);
            }
        }

        ggml_backend_tensor_set(t, cache[i].data(), 0, target.nbytes);
        LOG_INF("LONGCAT_NORM_INJECT: %s <- %s (%zu bytes)\n",
                t->name, target.file, target.nbytes);
        return;
    }
}

// LONGCAT_ROPE_INJECT (Experiment R0 -- oracle-injection control ONLY):
//
// When LONGCAT_ROPE_INJECT_DIR is set, the post-RoPE physical block-0 tensors
// q_pe-0 / k_pe-0 are OVERWRITTEN with the canonical HF targets produced by
// make_longcat_rope_targets.py, immediately after the node computes and
// before any dependent executes. This is NOT a RoPE implementation, NOT an
// arithmetic fix, and NOT evidence that C++ reproduces HF rotary arithmetic.
// It answers one causal question: with exact HF RoPE outputs supplied to the
// existing downstream graph, what remains at kqv_out / o_proj / the residual?
//
// The op check disambiguates the known name collision: the pre-RoPE views are
// also named q_pe-0 / k_pe-0 but their op is GGML_OP_VIEW, not GGML_OP_ROPE.
// Layout: the rope outputs ([64, 32, 512] and [64, 1, 512], contiguous F32)
// are byte-identical to the token-major target files (d fastest, then head,
// then token) -- proven in the plan -- so the write needs no reordering.
// Target-file SHA256s are verified by the run harness preflight; here the
// exact byte size, type, and contiguity are enforced, and any failure aborts
// the process (exit 87) so a partial injection can never produce a
// plausible-looking capture.
static bool common_debug_longcat_wants_rope_inject(const ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_ROPE_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return false;
    }
    if (t->op != GGML_OP_ROPE) {
        return false;
    }
    return strcmp(t->name, "q_pe-0") == 0 || strcmp(t->name, "k_pe-0") == 0;
}

static void common_debug_maybe_inject_longcat_rope(ggml_tensor * t) {
    const char * dir = std::getenv("LONGCAT_ROPE_INJECT_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return;
    }
    if (t->op != GGML_OP_ROPE) {
        return;
    }

    static const struct {
        const char * name;
        const char * file;
        size_t       nbytes;
    } targets[] = {
        { "q_pe-0", "q_pe_rope_target.bin", (size_t) 512 * 2048 * 4 },
        { "k_pe-0", "k_pe_rope_target.bin", (size_t) 512 *   64 * 4 },
    };

    static std::vector<uint8_t> cache[2];

    for (int i = 0; i < 2; ++i) {
        if (strcmp(t->name, targets[i].name) != 0) {
            continue;
        }

        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            ggml_nbytes(t) != targets[i].nbytes) {
            LOG_ERR(
                "LONGCAT_ROPE_INJECT ABORT: %s type=%s contig=%d nbytes=%zu "
                "(expected F32 contiguous %zu)\n",
                t->name, ggml_type_name(t->type),
                ggml_is_contiguous(t) ? 1 : 0,
                ggml_nbytes(t), targets[i].nbytes);
            common_log_flush(common_log_main());
            std::exit(87);
        }

        if (cache[i].empty()) {
            const std::string path = std::string(dir) + "/" + targets[i].file;
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                LOG_ERR("LONGCAT_ROPE_INJECT ABORT: cannot open %s\n", path.c_str());
                common_log_flush(common_log_main());
                std::exit(87);
            }
            cache[i].assign(std::istreambuf_iterator<char>(f),
                            std::istreambuf_iterator<char>());
            if (cache[i].size() != targets[i].nbytes) {
                LOG_ERR(
                    "LONGCAT_ROPE_INJECT ABORT: %s is %zu bytes, expected %zu\n",
                    path.c_str(), cache[i].size(), targets[i].nbytes);
                common_log_flush(common_log_main());
                std::exit(87);
            }
        }

        ggml_backend_tensor_set(t, cache[i].data(), 0, targets[i].nbytes);
        LOG_INF("LONGCAT_ROPE_INJECT: %s <- %s (%zu bytes)\n",
                t->name, targets[i].file, targets[i].nbytes);
        return;
    }
}

// Shared writer for both dump families. `dump_dir` and `spec` are resolved by
// the thin callers below; the body is the frozen final-row/full-sequence
// writer, unchanged.
static void common_debug_write_longcat_dump(
        uint8_t * data,
        const ggml_tensor * t,
        const char * dump_dir,
        const common_debug_longcat_dump_spec & spec) {
    if (t->ne[0] != spec.expect_ne0 || t->ne[1] < 1 ||
        t->ne[2] != 1 || t->ne[3] != 1) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP bad shape tensor=%s "
            "ne={%lld,%lld,%lld,%lld} expected ne0=%lld\n",
            t->name,
            (long long) t->ne[0],
            (long long) t->ne[1],
            (long long) t->ne[2],
            (long long) t->ne[3],
            (long long) spec.expect_ne0);
        common_log_flush(common_log_main());
        std::exit(87);
    }

    const int64_t width     = t->ne[0];
    const int64_t n_rows    = spec.full_sequence ? t->ne[1] : 1;
    const int64_t first_i1  = spec.full_sequence ? 0 : t->ne[1] - 1;
    const size_t  final_i1  = (size_t) t->ne[1] - 1;

    // Canonical token-major [n_rows, width]. Values are read through nb[] so a
    // non-contiguous view is normalized here rather than dumped verbatim.
    std::vector<float> row((size_t) (width * n_rows));

    for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t i1 = first_i1 + r;

        for (int64_t i0 = 0; i0 < width; ++i0) {
            const float v = common_ggml_get_float_value(
                data, t->type, t->nb, (size_t) i0, (size_t) i1, 0, 0);

            if (!std::isfinite(v)) {
                LOG_ERR(
                    "LONGCAT_HIDDEN_VECTOR_DUMP nonfinite "
                    "tensor=%s i0=%lld i1=%lld value=%f\n",
                    t->name, (long long) i0, (long long) i1, v);
                common_log_flush(common_log_main());
                std::exit(88);
            }

            row[(size_t) (r * width + i0)] = v;
        }
    }

    std::filesystem::path root(dump_dir);
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    if (ec) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP mkdir failed: %s\n",
            ec.message().c_str());
        common_log_flush(common_log_main());
        std::exit(89);
    }

    const auto output_path = root / spec.filename;

    std::ofstream out(
        output_path,
        std::ios::binary | std::ios::trunc);

    if (!out) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP open failed: %s\n",
            output_path.string().c_str());
        common_log_flush(common_log_main());
        std::exit(90);
    }

    out.write(
        reinterpret_cast<const char *>(row.data()),
        (std::streamsize) (row.size() * sizeof(float)));
    out.close();

    if (!out) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP write failed: %s\n",
            output_path.string().c_str());
        common_log_flush(common_log_main());
        std::exit(91);
    }

    // Sidecar carries the source layout so the comparator reads shape from
    // metadata instead of inferring it from file length, and so a storage
    // difference is distinguishable from an arithmetic one.
    if (spec.full_sequence) {
        auto sidecar_path = output_path;
        sidecar_path.replace_extension(".json");

        std::ofstream meta(sidecar_path, std::ios::trunc);

        if (!meta) {
            LOG_ERR(
                "LONGCAT_HIDDEN_VECTOR_DUMP sidecar open failed: %s\n",
                sidecar_path.string().c_str());
            common_log_flush(common_log_main());
            std::exit(92);
        }

        meta << "{\n";
        meta << "  \"tensor\": \"" << t->name << "\",\n";
        meta << "  \"shape\": [" << (long long) n_rows << ", "
             << (long long) width << "],\n";
        meta << "  \"order\": \"token-major\",\n";
        meta << "  \"dtype\": \"float32-le\",\n";
        meta << "  \"bytes\": "
             << (long long) (row.size() * sizeof(float)) << ",\n";
        meta << "  \"source_type\": \"" << ggml_type_name(t->type) << "\",\n";
        meta << "  \"source_contiguous\": "
             << (ggml_is_contiguous(t) ? "true" : "false") << ",\n";
        meta << "  \"source_ne\": [" << (long long) t->ne[0] << ", "
             << (long long) t->ne[1] << ", " << (long long) t->ne[2] << ", "
             << (long long) t->ne[3] << "],\n";
        meta << "  \"source_nb\": [" << (long long) t->nb[0] << ", "
             << (long long) t->nb[1] << ", " << (long long) t->nb[2] << ", "
             << (long long) t->nb[3] << "]\n";
        meta << "}\n";
        meta.close();

        if (!meta) {
            LOG_ERR(
                "LONGCAT_HIDDEN_VECTOR_DUMP sidecar write failed: %s\n",
                sidecar_path.string().c_str());
            common_log_flush(common_log_main());
            std::exit(93);
        }
    }

    LOG(
        "LONGCAT_HIDDEN_VECTOR_DUMP tensor=%s file=%s rows=%lld width=%lld "
        "mode=%s final_i1=%zu type=%s contiguous=%d\n",
        t->name,
        output_path.string().c_str(),
        (long long) n_rows,
        (long long) width,
        spec.full_sequence ? "full-sequence" : "final-row",
        final_i1,
        ggml_type_name(t->type),
        ggml_is_contiguous(t) ? 1 : 0);
}

static void common_debug_maybe_dump_longcat_hidden(
        uint8_t * data,
        const ggml_tensor * t) {
    const char * dump_dir = std::getenv("LONGCAT_HIDDEN_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return;
    }

    common_debug_longcat_dump_spec spec;
    if (!common_debug_longcat_dump_spec_for(t, spec)) {
        return;
    }

    common_debug_write_longcat_dump(data, t, dump_dir, spec);
}

static void common_debug_maybe_dump_longcat_resid_walk(
        uint8_t * data,
        const ggml_tensor * t) {
    const char * dump_dir = std::getenv("LONGCAT_RESID_WALK_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return;
    }

    common_debug_longcat_dump_spec spec;
    if (!common_debug_longcat_resid_walk_spec_for(t, spec)) {
        return;
    }

    common_debug_write_longcat_dump(data, t, dump_dir, spec);
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
bool common_debug_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (common_debug_cb_user_data *) user_data;
    auto * pimpl = cb_data->pimpl.get();

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    bool matches_filter = pimpl->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : pimpl->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    // LONGCAT_GATE4_NAN_AUDIT: at ask time, request only tensors that
    // match --tensor-filter. The stock callback asks for every tensor,
    // forcing needless device-to-host copies even for filtered output.
    // Also request dump targets, so LONGCAT_HIDDEN_DUMP_DIR works whether or
    // not the tensor is covered by --tensor-filter.
    if (ask) {
        return matches_filter || common_debug_longcat_wants_dump(t) ||
               common_debug_longcat_wants_resid_walk_dump(t) ||
               common_debug_longcat_wants_rope_inject(t) ||
               common_debug_longcat_wants_resid_inject(t) ||
               common_debug_longcat_wants_attn_norm2_inject(t) ||
               common_debug_longcat_wants_proj_inject(t) ||
               common_debug_longcat_wants_norm_inject(t);
    }

    // LONGCAT_ROPE_INJECT (R0): overwrite the post-RoPE block-0 tensors with
    // the canonical HF targets before any dependent node executes. Must run
    // before the host copy below, which would otherwise snapshot stale data.
    common_debug_maybe_inject_longcat_rope(t);

    // LONGCAT_RESID_INJECT (causal reset): overwrite l_out-1 with the HF
    // full-sequence logical_00 oracle. Same ordering requirement: before the
    // host copy, so the walk dump below records the injected bytes (landing
    // gate) and every dependent consumes the reset state.
    common_debug_maybe_inject_longcat_resid(t);

    // LONGCAT_ATTN_NORM2_INJECT (dual reset): overwrite attn_norm-2 with the
    // HF layer-1 attn0_norm oracle, same ordering requirement.
    common_debug_maybe_inject_longcat_attn_norm2(t);

    // LONGCAT_PROJ_INJECT (quad reset): overwrite the block-2 root
    // projection outputs with the captured HF values, same ordering.
    common_debug_maybe_inject_longcat_proj(t);

    // LONGCAT_NORM_INJECT (hex reset): overwrite the block-2 LoRA norm
    // outputs with the captured HF values, same ordering.
    common_debug_maybe_inject_longcat_norm(t);

    char src1_str[128] = { 0 };
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, common_ggml_ne_string(src1).c_str());
    }

    if (matches_filter) {
        LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, t->name, ggml_type_name(t->type),
            ggml_op_desc(t), src0->name, common_ggml_ne_string(src0).c_str(), src1 ? src1_str : "",
            common_ggml_ne_string(t).c_str());
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        pimpl->data.resize(n_bytes);
        ggml_backend_tensor_get(t, pimpl->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : pimpl->data.data();

        // The dump is keyed on tensor name, not on --tensor-filter, so an
        // abort target that was requested only for its side effect still gets
        // written before evaluation stops.
        common_debug_maybe_dump_longcat_hidden(data, t);
        common_debug_maybe_dump_longcat_resid_walk(data, t);

        if (matches_filter) {
            const bool saw_nan = common_debug_print_tensor(data, t->type, t->ne, t->nb, 3);
            if (pimpl->abort_on_nan && saw_nan) {
                LOG("LONGCAT_GATE4_NAN_AUDIT FIRST_NAN tensor=%s\n", t->name);
                common_log_flush(common_log_main());
                std::exit(86);
            }
        }
    }

    return true;
}
