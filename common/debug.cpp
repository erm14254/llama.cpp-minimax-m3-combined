#include "debug.h"

#include "common.h"
#include "log.h"

#include <cmath>
#include <cstdlib>
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
// When LONGCAT_HIDDEN_DUMP_DIR is set, dump the final-token hidden vector
// for the 15 HF-comparable LongCat 512-token diagnostic surfaces.
// Files are always little/native-endian F32, 3072 values = 12288 bytes.
static bool common_debug_longcat_hidden_filename(
        const ggml_tensor * t,
        std::string & filename) {
    const std::string tensor_name = t->name;

    if (tensor_name == "inp_embd_ngram") {
        filename = "inp_embd_ngram.bin";
        return true;
    }

    if (tensor_name == "result_norm") {
        filename = "result_norm.bin";
        return true;
    }

    // LONGCAT_LOGICAL0_STAGE_VECTOR_DUMP:
    // Additional logical-layer-0 parity boundaries.
    // l_out-1 intentionally remains handled by the existing logical_00 mapping.
    if (tensor_name == "ffn_inp-0") {
        filename = "logical0_attn0_resid.bin";
        return true;
    }

    if (tensor_name == "l_out-0") {
        filename = "logical0_mlp0_resid.bin";
        return true;
    }

    if (tensor_name == "ffn_inp-1") {
        filename = "logical0_attn1_resid.bin";
        return true;
    }

    // LONGCAT_LOGICAL0_ATTN0_NORM_VECTOR_DUMP:
    // First normalized activation entering physical attention block 0.
    if (tensor_name == "attn_norm-0") {
        filename = "logical0_attn0_norm.bin";
        return true;
    }

    for (int logical = 0; logical < 13; ++logical) {
        const int physical = 2 * logical + 1;
        if (tensor_name == "l_out-" + std::to_string(physical)) {
            char buf[64];
            snprintf(buf, sizeof(buf), "logical_%02d.bin", logical);
            filename = buf;
            return true;
        }
    }

    return false;
}

static void common_debug_maybe_dump_longcat_hidden(
        uint8_t * data,
        const ggml_tensor * t) {
    const char * dump_dir = std::getenv("LONGCAT_HIDDEN_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return;
    }

    std::string filename;
    if (!common_debug_longcat_hidden_filename(t, filename)) {
        return;
    }

    if (t->ne[0] != 3072 || t->ne[1] < 1 ||
        t->ne[2] != 1 || t->ne[3] != 1) {
        LOG_ERR(
            "LONGCAT_HIDDEN_VECTOR_DUMP bad shape tensor=%s "
            "ne={%lld,%lld,%lld,%lld}\n",
            t->name,
            (long long) t->ne[0],
            (long long) t->ne[1],
            (long long) t->ne[2],
            (long long) t->ne[3]);
        common_log_flush(common_log_main());
        std::exit(87);
    }

    const size_t final_i1 = (size_t) t->ne[1] - 1;

    std::vector<float> row(3072);
    for (size_t i0 = 0; i0 < row.size(); ++i0) {
        row[i0] = common_ggml_get_float_value(
            data, t->type, t->nb, i0, final_i1, 0, 0);

        if (!std::isfinite(row[i0])) {
            LOG_ERR(
                "LONGCAT_HIDDEN_VECTOR_DUMP nonfinite "
                "tensor=%s i0=%zu value=%f\n",
                t->name, i0, row[i0]);
            common_log_flush(common_log_main());
            std::exit(88);
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

    const auto output_path = root / filename;

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

    LOG(
        "LONGCAT_HIDDEN_VECTOR_DUMP tensor=%s "
        "file=%s final_i1=%zu type=%s\n",
        t->name,
        output_path.string().c_str(),
        final_i1,
        ggml_type_name(t->type));
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
    if (ask) {
        return matches_filter;
    }

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

    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : pimpl->data.data();
        common_debug_maybe_dump_longcat_hidden(data, t);
        const bool saw_nan = common_debug_print_tensor(data, t->type, t->ne, t->nb, 3);
        if (pimpl->abort_on_nan && saw_nan) {
            LOG("LONGCAT_GATE4_NAN_AUDIT FIRST_NAN tensor=%s\n", t->name);
            common_log_flush(common_log_main());
            std::exit(86);
        }
    }

    return true;
}
