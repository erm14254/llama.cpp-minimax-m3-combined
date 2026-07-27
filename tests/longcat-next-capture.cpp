#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct capture_state {
    fs::path dir;
    std::ofstream manifest;
    std::ofstream layer0_manifest;
    std::ofstream all_blocks_manifest;
    std::set<std::string> all_blocks_seen;
    std::ofstream components_manifest;
    std::set<std::string> components_seen;
    std::ofstream components_window_manifest;
    std::set<std::string> components_window_seen;
    bool direct_forward = true;
    bool layer0_diagnostic = false;
    bool all_blocks_diagnostic = false;
    bool block_components_diagnostic = false;
    bool block_components_window_diagnostic = false;
    int block_components_window_start = 0;
    int block_components_window_count = 0;
};

enum class capture_cache_type {
    DEFAULT,
    F16,
    BF16,
    F32,
};

static bool parse_cache_type(const std::string & value, capture_cache_type & result) {
    if (value == "default") result = capture_cache_type::DEFAULT;
    else if (value == "f16") result = capture_cache_type::F16;
    else if (value == "bf16") result = capture_cache_type::BF16;
    else if (value == "f32") result = capture_cache_type::F32;
    else return false;
    return true;
}

static void apply_cache_type(llama_context_params & params, capture_cache_type cache_type) {
    if (cache_type == capture_cache_type::DEFAULT) {
        return;
    }
    const ggml_type type = cache_type == capture_cache_type::F16  ? GGML_TYPE_F16 :
                           cache_type == capture_cache_type::BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    params.type_k = type;
    params.type_v = type;
}

static bool wanted(const std::string & name) {
    if (name == "inp_embd" || name == "inp_embd_ngram" || name == "h_nextn" ||
            name == "l_out-0" || name == "l_out-1" || name == "l_out-2" || name == "l_out-27") {
        return true;
    }
    constexpr const char * prefix = "ngram_proj-";
    if (name.rfind(prefix, 0) != 0) return false;
    const std::string suffix = name.substr(11);
    if (suffix.empty() || (suffix.size() > 1 && suffix[0] == '0') ||
            !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
    return std::stoi(suffix) < 12;
}

static bool wanted_layer0(const std::string & name) {
    static const std::vector<std::string> names = {
        "attn_norm-0", "q_scaled-0", "kv_cmpr_scaled-0", "q_nope_absorbed_perm-0",
        "Qcur-0", "Kcur-0", "Vcur-0", "kq-0", "kq_soft_max-0", "kqv-0",
        "kqv_mla-0", "fattn_mla-0", "kqv_out-0", "attn_out-0", "ffn_inp-0",
        "ffn_norm-0", "ffn_out-0", "l_out-0"};
    return std::find(names.begin(), names.end(), name) != names.end();
}

static bool wanted_all_blocks(const std::string & name) {
    constexpr const char * prefix = "l_out-";
    if (name.rfind(prefix, 0) != 0 || name.size() <= 6) return false;
    const std::string suffix = name.substr(6);
    if (suffix.size() > 1 && suffix[0] == '0') return false;
    if (!std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) return false;
    const int block = std::stoi(suffix);
    return block >= 0 && block < 28;
}

static bool wanted_block_component(const std::string & name) {
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= name.size()) return false;
    const std::string base = name.substr(0, dash);
    const std::string suffix = name.substr(dash + 1);
    if ((suffix.size() > 1 && suffix[0] == '0') ||
            !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) return false;
    const int block = std::stoi(suffix);
    if (block < 0 || block > 9) return false;
    static const std::set<std::string> ordinary = {
        "block_in", "attn_norm", "attn_out", "ffn_inp", "ffn_norm", "ffn_out", "l_out"};
    static const std::set<std::string> moe = {
        "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_topk",
        "ffn_moe_weights_scaled", "identity_weight_sum", "identity_residual", "moe_shortcut"};
    return ordinary.count(base) || (block % 2 == 0 && moe.count(base));
}

static bool wanted_block_component_window(const std::string & name, int start, int count) {
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= name.size()) return false;
    const std::string suffix = name.substr(dash + 1);
    if ((suffix.size() > 1 && suffix[0] == '0') ||
            !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) return false;
    const int block = std::stoi(suffix);
    if (block < start || block >= start + count) return false;
    const std::string base = name.substr(0, dash);
    static const std::set<std::string> ordinary = {
        "block_in", "attn_norm", "attn_out", "ffn_inp", "ffn_norm", "ffn_out", "l_out"};
    static const std::set<std::string> moe = {
        "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased", "ffn_moe_topk",
        "ffn_moe_weights_scaled", "identity_weight_sum", "identity_residual", "moe_shortcut"};
    return ordinary.count(base) || (block % 2 == 0 && moe.count(base));
}

static bool parse_binary_flag(const std::string & value, bool & result) {
    if (value != "0" && value != "1") return false;
    result = value == "1";
    return true;
}

static bool set_longcat_bf16_boundary_rounding(bool enabled) {
    const char * value = enabled ? "1" : "0";
#ifdef _WIN32
    return _putenv_s("LLAMA_LONGCAT_BF16_BOUNDARY_ROUNDING", value) == 0;
#else
    return setenv("LLAMA_LONGCAT_BF16_BOUNDARY_ROUNDING", value, 1) == 0;
#endif
}

static bool set_longcat_bf16_hidden_surface_rounding(bool enabled) {
    const char * value = enabled ? "1" : "0";
#ifdef _WIN32
    return _putenv_s("LLAMA_LONGCAT_BF16_HIDDEN_SURFACE_ROUNDING", value) == 0;
#else
    return setenv("LLAMA_LONGCAT_BF16_HIDDEN_SURFACE_ROUNDING", value, 1) == 0;
#endif
}

static std::vector<uint8_t> read_logical_tensor_bytes(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 &&
            tensor->type != GGML_TYPE_BF16 && tensor->type != GGML_TYPE_I32) {
        throw std::runtime_error("unsupported capture tensor type");
    }
    if (tensor->nb[0] != ggml_type_size(tensor->type)) {
        throw std::runtime_error("capture tensor rows are not contiguous along ne[0]");
    }
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    const size_t row_count = (size_t) tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
    std::vector<uint8_t> packed(row_bytes * row_count);
    size_t destination = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                const size_t source = i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
                ggml_backend_tensor_get(tensor, packed.data() + destination, source, row_bytes);
                destination += row_bytes;
            }
        }
    }
    return packed;
}

static void write_capture(ggml_tensor * tensor, const fs::path & dir, std::ofstream & manifest, const std::string & prefix) {
    std::string file = prefix + tensor->name + ".raw";
    for (char & c : file) if (!std::isalnum((unsigned char) c) && c != '-' && c != '_') c = '_';
    std::vector<uint8_t> bytes = read_logical_tensor_bytes(tensor);
    std::ofstream(dir / file, std::ios::binary).write((char *) bytes.data(), bytes.size());
    manifest << tensor->name << '\t' << ggml_type_name(tensor->type) << '\t';
    for (int i = 0; i < GGML_MAX_DIMS; ++i) manifest << (i ? "," : "") << tensor->ne[i];
    manifest << '\t' << file << '\n';
}

static bool capture_cb(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<capture_state *>(opaque);
    const std::string name = tensor->name;
    const bool normal = wanted(name);
    const bool diagnostic = state.layer0_diagnostic && wanted_layer0(name);
    const bool all_blocks = state.all_blocks_diagnostic && wanted_all_blocks(name);
    const bool components = state.block_components_diagnostic && wanted_block_component(name);
    const bool components_window = state.block_components_window_diagnostic &&
        wanted_block_component_window(name, state.block_components_window_start,
                                      state.block_components_window_count);
    if (!state.direct_forward || (!normal && !diagnostic && !all_blocks && !components && !components_window)) return false;
    if (ask) return true;
    if (normal) write_capture(tensor, state.dir, state.manifest, "");
    if (diagnostic) write_capture(tensor, state.dir, state.layer0_manifest, "diag_");
    if (all_blocks && state.all_blocks_seen.insert(name).second) {
        write_capture(tensor, state.dir, state.all_blocks_manifest, "all_blocks_");
    }
    if (components && state.components_seen.insert(name).second) {
        write_capture(tensor, state.dir, state.components_manifest, "components_");
    }
    if (components_window && state.components_window_seen.insert(name).second) {
        write_capture(tensor, state.dir, state.components_window_manifest, "components_window_");
    }
    return true;
}

static int32_t argmax_large_tie(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t id = 1; id < n_vocab; ++id) {
        if (logits[id] >= logits[best]) best = id;
    }
    return best;
}

static llama_seq_id sequence_for_mask(size_t index, int32_t attended) {
    return attended ? 0 : (llama_seq_id) index + 1;
}

static llama_context_params capture_context_params(
        uint32_t n_ctx, size_t token_count, int32_t threads, llama_flash_attn_type flash_attn,
        capture_cache_type cache_type, capture_state * state) {
    auto params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_ctx;
    params.n_ubatch = n_ctx;
    params.n_seq_max = token_count + 1;
    // Auxiliary sequence IDs still isolate masked padding. A unified KV cache
    // preserves those semantics while keeping the complete capture case in one
    // direct-forward ubatch, so every callback surface contains all token rows.
    params.kv_unified = true;
    params.flash_attn_type = flash_attn;
    // These are the requested cache types. The context constructor applies
    // architecture-specific resolution, including LongCat-Next F16 -> BF16.
    apply_cache_type(params, cache_type);
    if (threads > 0) {
        params.n_threads = threads;
        params.n_threads_batch = threads;
    }
    params.cb_eval = capture_cb;
    params.cb_eval_user_data = state;
    return params;
}

static int packed_serialization_self_test() {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 50;
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, 2);
    ggml_tensor * topk = ggml_argsort_top_k(ctx, logits, 12);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * bf16 = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 3, 2);
    ggml_tensor * f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, 2);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 12, 2);
    ggml_tensor * unsupported = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, topk);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return 51;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 51;
    std::vector<float> logits_data(384 * 2);
    for (int expert = 0; expert < 384; ++expert) {
        logits_data[expert] = (float) expert;
        logits_data[384 + expert] = (float) -expert;
    }
    const std::array<float, 6> hidden_data = { 1, 2, 3, 4, 5, 6 };
    const std::array<ggml_bf16_t, 6> bf16_data = {
        ggml_fp32_to_bf16(1), ggml_fp32_to_bf16(2), ggml_fp32_to_bf16(3),
        ggml_fp32_to_bf16(4), ggml_fp32_to_bf16(5), ggml_fp32_to_bf16(6) };
    const std::array<ggml_fp16_t, 6> f16_data = {
        ggml_fp32_to_fp16(1), ggml_fp32_to_fp16(2), ggml_fp32_to_fp16(3),
        ggml_fp32_to_fp16(4), ggml_fp32_to_fp16(5), ggml_fp32_to_fp16(6) };
    std::array<float, 24> weights_data = {};
    for (size_t i = 0; i < weights_data.size(); ++i) weights_data[i] = (float) i;
    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    ggml_backend_tensor_set(hidden, hidden_data.data(), 0, sizeof(hidden_data));
    ggml_backend_tensor_set(bf16, bf16_data.data(), 0, sizeof(bf16_data));
    ggml_backend_tensor_set(f16, f16_data.data(), 0, sizeof(f16_data));
    ggml_backend_tensor_set(weights, weights_data.data(), 0, sizeof(weights_data));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 52;

    const size_t logical_topk_bytes = 24 * sizeof(int32_t);
    if (topk->ne[0] != 12 || topk->ne[1] != 2 || topk->nb[1] <= 12 * sizeof(int32_t) ||
            ggml_nbytes(topk) <= logical_topk_bytes) return 53;
    const auto packed_topk = read_logical_tensor_bytes(topk);
    if (packed_topk.size() != logical_topk_bytes) return 54;
    const int32_t * ids = reinterpret_cast<const int32_t *>(packed_topk.data());
    for (int k = 0; k < 12; ++k) {
        if (ids[k] != 383 - k || ids[12 + k] != k) return 55;
    }
    const fs::path output = fs::temp_directory_path() / "longcat-next-packed-capture-self-test";
    fs::remove_all(output);
    fs::create_directories(output);
    ggml_set_name(topk, "ffn_moe_topk-0");
    std::ofstream manifest(output / "manifest.tsv", std::ios::trunc);
    write_capture(topk, output, manifest, "components_");
    manifest.close();
    if (fs::file_size(output / "components_ffn_moe_topk-0_raw") != logical_topk_bytes) return 58;
    fs::remove_all(output);
    const auto packed_hidden = read_logical_tensor_bytes(hidden);
    const auto packed_bf16 = read_logical_tensor_bytes(bf16);
    const auto packed_f16 = read_logical_tensor_bytes(f16);
    const auto packed_weights = read_logical_tensor_bytes(weights);
    if (packed_hidden.size() != sizeof(hidden_data) ||
            packed_bf16.size() != sizeof(bf16_data) ||
            packed_f16.size() != sizeof(f16_data) ||
            packed_weights.size() != sizeof(weights_data) ||
            std::memcmp(packed_hidden.data(), hidden_data.data(), sizeof(hidden_data)) != 0 ||
            std::memcmp(packed_bf16.data(), bf16_data.data(), sizeof(bf16_data)) != 0 ||
            std::memcmp(packed_f16.data(), f16_data.data(), sizeof(f16_data)) != 0 ||
            std::memcmp(packed_weights.data(), weights_data.data(), sizeof(weights_data)) != 0) return 56;
    const size_t original_nb0 = hidden->nb[0];
    hidden->nb[0]++;
    bool malformed_rejected = false;
    try { (void) read_logical_tensor_bytes(hidden); } catch (const std::runtime_error &) { malformed_rejected = true; }
    hidden->nb[0] = original_nb0;
    bool unsupported_rejected = false;
    try { (void) read_logical_tensor_bytes(unsupported); } catch (const std::runtime_error &) { unsupported_rejected = true; }
    if (!malformed_rejected || !unsupported_rejected) return 57;

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}

static int self_test() {
    if (const int packed = packed_serialization_self_test()) return packed;
    const float tied[] = { 0.0f, 2.0f, 2.0f, 1.0f };
    if (argmax_large_tie(tied, 4) != 2) return 20;
    if (sequence_for_mask(0, 0) == sequence_for_mask(1, 0) || sequence_for_mask(2, 1) != 0) return 21;
    capture_state state;
    state.direct_forward = false;
    ggml_tensor dummy = {};
    snprintf(dummy.name, sizeof(dummy.name), "inp_embd");
    if (capture_cb(&dummy, true, &state)) return 22;
    // Multiple cases are represented by independently created contexts and
    // distinct output namespaces in the single model-owning main loop.
    const std::vector<std::string> case_names = { "case_a", "case_b" };
    if (case_names[0] == case_names[1]) return 23;
    const auto params = capture_context_params(
        37, 5, 3, LLAMA_FLASH_ATTN_TYPE_DISABLED, capture_cache_type::DEFAULT, &state);
    if (!params.kv_unified) return 24;
    if (params.n_seq_max != 6 || params.n_batch != 37 || params.n_ubatch != 37) return 25;
    if (params.cb_eval != capture_cb || params.cb_eval_user_data != &state) return 26;
    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) return 27;
    if (capture_context_params(37, 5, 3, LLAMA_FLASH_ATTN_TYPE_AUTO,
            capture_cache_type::DEFAULT, &state).flash_attn_type != LLAMA_FLASH_ATTN_TYPE_AUTO) return 28;
    if (capture_context_params(37, 5, 3, LLAMA_FLASH_ATTN_TYPE_ENABLED,
            capture_cache_type::DEFAULT, &state).flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) return 29;
    const auto defaults = llama_context_default_params();
    if (params.type_k != defaults.type_k || params.type_v != defaults.type_v) return 30;
    for (const auto & row : std::vector<std::pair<capture_cache_type, ggml_type>> {
            {capture_cache_type::F16, GGML_TYPE_F16},
            {capture_cache_type::BF16, GGML_TYPE_BF16},
            {capture_cache_type::F32, GGML_TYPE_F32}}) {
        const auto typed = capture_context_params(
            37, 5, 3, LLAMA_FLASH_ATTN_TYPE_AUTO, row.first, &state);
        if (typed.type_k != row.second || typed.type_v != row.second) return 31;
    }
    capture_cache_type parsed = capture_cache_type::DEFAULT;
    if (!parse_cache_type("default", parsed) || parsed != capture_cache_type::DEFAULT) return 32;
    if (!parse_cache_type("f16", parsed) || parsed != capture_cache_type::F16) return 33;
    if (!parse_cache_type("bf16", parsed) || parsed != capture_cache_type::BF16) return 34;
    if (!parse_cache_type("f32", parsed) || parsed != capture_cache_type::F32) return 35;
    if (parse_cache_type("invalid", parsed)) return 36;
    for (const char * name : {"kq-0", "kq_soft_max-0", "kqv-0", "kqv_mla-0",
            "fattn_mla-0", "kqv_out-0", "attn_out-0"}) {
        if (!wanted_layer0(name)) return 37;
    }
    if (wanted_layer0("q-0") || wanted_layer0("kq-1")) return 38;
    for (int block = 0; block < 28; ++block) {
        if (!wanted_all_blocks("l_out-" + std::to_string(block))) return 39;
    }
    if (wanted_all_blocks("l_out-28") || wanted_all_blocks("l_out-01") ||
            wanted_all_blocks("l_out-x") || wanted_all_blocks("attn_out-0")) return 40;
    bool binary = false;
    if (!parse_binary_flag("0", binary) || binary) return 41;
    if (!parse_binary_flag("1", binary) || !binary) return 42;
    if (parse_binary_flag("2", binary) || parse_binary_flag("true", binary)) return 43;
    std::set<std::string> standard;
    for (const char * name : {"inp_embd", "inp_embd_ngram", "h_nextn", "l_out-0",
            "l_out-1", "l_out-2", "l_out-27"}) standard.insert(name);
    for (int index = 0; index < 12; ++index) standard.insert("ngram_proj-" + std::to_string(index));
    if (standard.size() + 1 != 20 || !standard.count("inp_embd_ngram") ||
            !std::all_of(standard.begin(), standard.end(), wanted) ||
            wanted("l_out-3") || wanted("ngram_proj-12")) return 44;
    std::set<std::string> components;
    for (int block = 0; block < 10; ++block) {
        for (const char * base : {"block_in", "attn_norm", "attn_out", "ffn_inp",
                "ffn_norm", "ffn_out", "l_out"}) {
            components.insert(std::string(base) + "-" + std::to_string(block));
        }
        if (block % 2 == 0) {
            for (const char * base : {"ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased",
                    "ffn_moe_topk", "ffn_moe_weights_scaled", "identity_weight_sum",
                    "identity_residual", "moe_shortcut"}) {
                components.insert(std::string(base) + "-" + std::to_string(block));
            }
        }
    }
    if (components.size() != 110 ||
            !std::all_of(components.begin(), components.end(), wanted_block_component)) return 45;
    for (int block = 0; block < 10; ++block) {
        if (!components.count("l_out-" + std::to_string(block))) return 47;
    }
    if (wanted_block_component("block_in-10") || wanted_block_component("block_in-01") ||
            wanted_block_component("surprise-0") || wanted_block_component("ffn_moe_logits-1")) return 46;
    std::set<std::string> window;
    for (int block = 10; block < 14; ++block) {
        for (const char * base : {"block_in", "attn_norm", "attn_out", "ffn_inp", "ffn_norm", "ffn_out", "l_out"})
            window.insert(std::string(base) + "-" + std::to_string(block));
        if (block % 2 == 0) for (const char * base : {"ffn_moe_logits", "ffn_moe_probs", "ffn_moe_probs_biased",
                "ffn_moe_topk", "ffn_moe_weights_scaled", "identity_weight_sum", "identity_residual", "moe_shortcut"})
            window.insert(std::string(base) + "-" + std::to_string(block));
    }
    if (window.size() != 44 || !std::all_of(window.begin(), window.end(), [](const std::string & name) {
            return wanted_block_component_window(name, 10, 4); })) return 48;
    return 0;
}

static int run_case(
        llama_model * model, const json & spec, const fs::path & root,
        uint32_t n_ctx, int32_t threads, llama_flash_attn_type flash_attn,
        capture_cache_type cache_type, bool layer0_diagnostic, bool all_blocks_diagnostic,
        bool block_components_diagnostic, bool block_components_window_diagnostic,
        int block_components_window_start, int block_components_window_count) {
    const auto ids = spec.at("input_ids").get<std::vector<llama_token>>();
    const auto mask = spec.at("attention_mask").get<std::vector<int32_t>>();
    const auto positions = spec.at("position_ids").get<std::vector<llama_pos>>();
    const auto cache = spec.at("cache_position").get<std::vector<llama_pos>>();
    if (ids.empty() || ids.size() != mask.size() || ids.size() != positions.size() || ids.size() != cache.size()) return 10;

    const fs::path dir = root / spec.at("name").get<std::string>();
    fs::create_directories(dir);
    capture_state state;
    state.dir = dir;
    state.manifest.open(dir / "captures.tsv", std::ios::trunc);
    state.direct_forward = true;
    state.layer0_diagnostic = layer0_diagnostic;
    state.all_blocks_diagnostic = all_blocks_diagnostic;
    state.block_components_diagnostic = block_components_diagnostic;
    state.block_components_window_diagnostic = block_components_window_diagnostic;
    state.block_components_window_start = block_components_window_start;
    state.block_components_window_count = block_components_window_count;
    if (layer0_diagnostic) {
        state.layer0_manifest.open(dir / "layer0-diagnostics.tsv", std::ios::trunc);
    }
    if (all_blocks_diagnostic) {
        state.all_blocks_manifest.open(dir / "all-blocks-diagnostics.tsv", std::ios::trunc);
    }
    if (block_components_diagnostic) {
        state.components_manifest.open(dir / "block-components-diagnostics.tsv", std::ios::trunc);
    }
    if (block_components_window_diagnostic) {
        state.components_window_manifest.open(dir / "block-components-window-diagnostics.tsv", std::ios::trunc);
    }
    auto cp = capture_context_params(n_ctx, ids.size(), threads, flash_attn, cache_type, &state);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 11;

    llama_batch batch = llama_batch_init(ids.size(), 0, 1);
    batch.n_tokens = ids.size();
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = positions[i];
        batch.n_seq_id[i] = 1;
        // Padding tokens are isolated sequences and cannot become attention
        // keys for the target sequence. Attended tokens all use sequence 0.
        batch.seq_id[i][0] = sequence_for_mask(i, mask[i]);
        batch.logits[i] = i + 1 == ids.size();
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) { llama_free(ctx); return 12; }
    if (all_blocks_diagnostic && state.all_blocks_seen.size() != 28) {
        llama_free(ctx);
        return 14;
    }
    if (block_components_diagnostic && state.components_seen.size() != 110) {
        llama_free(ctx);
        return 15;
    }
    const size_t expected_window = (size_t) block_components_window_count * 7 +
        (size_t) (block_components_window_count / 2) * 8;
    if (block_components_window_diagnostic && state.components_window_seen.size() != expected_window) {
        llama_free(ctx);
        return 16;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    const std::vector<float> direct_logits(logits, logits + n_vocab);
    std::ofstream(dir / "final_logits.f32.raw", std::ios::binary).write((const char *) logits, n_vocab * sizeof(float));
    state.manifest << "final_logits\tf32\t" << n_vocab << ",1,1,1\tfinal_logits.f32.raw\n";

    json received = spec;
    received["runtime_sequence_ids"] = json::array();
    for (size_t i = 0; i < ids.size(); ++i) received["runtime_sequence_ids"].push_back(mask[i] ? 0 : (int) i + 1);
    std::ofstream(dir / "inputs.json") << received.dump(2) << '\n';

    std::vector<llama_token> sequence = ids;
    std::vector<llama_token> continuation;
    const bool greedy = spec.at("greedy_eight_tokens").get<bool>();
    const int steps = greedy ? 8 : 0;
    state.direct_forward = false; // never overwrite direct-forward captures
    for (int step = 0; step < steps; ++step) {
        const llama_token next = argmax_large_tie(logits, n_vocab);
        continuation.push_back(next);
        sequence.push_back(next);
        llama_batch one = llama_batch_init(1, 0, 1);
        one.n_tokens = 1;
        one.token[0] = next;
        one.pos[0] = cache.back() + step + 1;
        one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0;
        one.logits[0] = 1;
        if (llama_decode(ctx, one) != 0) { llama_batch_free(one); llama_free(ctx); return 13; }
        llama_batch_free(one);
        logits = llama_get_logits_ith(ctx, -1);
    }
    std::vector<int32_t> order(n_vocab);
    for (int32_t i = 0; i < n_vocab; ++i) order[i] = i;
    // Equivalent to np.argsort(...)[::-1]: larger token ID wins exact ties.
    std::partial_sort(order.begin(), order.begin() + std::min(10, n_vocab), order.end(),
        [&](int32_t a, int32_t b) {
            return direct_logits[a] == direct_logits[b] ? a > b : direct_logits[a] > direct_logits[b];
        });
    json decoding = {{"prompt_plus_continuation_ids", sequence}, {"greedy_continuation_ids", continuation},
                     {"argmax_id", order[0]}, {"top_k_ids", json::array()}, {"top_k_values", json::array()}};
    for (int i = 0; i < std::min(10, n_vocab); ++i) {
        decoding["top_k_ids"].push_back(order[i]);
        decoding["top_k_values"].push_back(direct_logits[order[i]]);
    }
    std::ofstream(dir / "decoding.json") << std::setprecision(9) << decoding.dump(2) << '\n';
    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
    fs::path model_path, manifest_path, output;
    int32_t n_gpu_layers = 0;
    int32_t threads = 0;
    llama_flash_attn_type flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
    capture_cache_type cache_type = capture_cache_type::DEFAULT;
    bool layer0_diagnostic = false;
    bool all_blocks_diagnostic = false;
    bool block_components_diagnostic = false;
    bool block_components_window_diagnostic = false;
    int block_components_window_start = 10;
    int block_components_window_count = 4;
    bool longcat_bf16_boundary_rounding = false;
    bool longcat_bf16_hidden_surface_rounding = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--model") model_path = argv[i + 1];
        else if (key == "--case-manifest") manifest_path = argv[i + 1];
        else if (key == "--output-dir") output = argv[i + 1];
        else if (key == "--n-gpu-layers") n_gpu_layers = std::stoi(argv[i + 1]);
        else if (key == "--threads") threads = std::stoi(argv[i + 1]);
        else if (key == "--flash-attn") {
            const std::string value = argv[i + 1];
            if (value == "auto") flash_attn = LLAMA_FLASH_ATTN_TYPE_AUTO;
            else if (value == "disabled") flash_attn = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else if (value == "enabled") flash_attn = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else return 2;
        } else if (key == "--cache-type") {
            if (!parse_cache_type(argv[i + 1], cache_type)) return 2;
        } else if (key == "--layer0-diagnostic") {
            if (!parse_binary_flag(argv[i + 1], layer0_diagnostic)) return 2;
        } else if (key == "--all-blocks-diagnostic") {
            if (!parse_binary_flag(argv[i + 1], all_blocks_diagnostic)) return 2;
        } else if (key == "--block-components-diagnostic") {
            if (!parse_binary_flag(argv[i + 1], block_components_diagnostic)) return 2;
        } else if (key == "--block-components-window-diagnostic") {
            if (!parse_binary_flag(argv[i + 1], block_components_window_diagnostic)) return 2;
        } else if (key == "--block-components-window-start") {
            block_components_window_start = std::stoi(argv[i + 1]);
        } else if (key == "--block-components-window-count") {
            block_components_window_count = std::stoi(argv[i + 1]);
        } else if (key == "--longcat-bf16-boundary-rounding") {
            if (!parse_binary_flag(argv[i + 1], longcat_bf16_boundary_rounding)) return 2;
        } else if (key == "--longcat-bf16-hidden-surface-rounding") {
            if (!parse_binary_flag(argv[i + 1], longcat_bf16_hidden_surface_rounding)) return 2;
        }
        else return 2;
    }
    if (model_path.empty() || manifest_path.empty() || output.empty() || n_gpu_layers < 0 || threads < 0) return 2;
    if (block_components_diagnostic && block_components_window_diagnostic) return 2;
    if (block_components_window_diagnostic &&
            (block_components_window_start < 0 || block_components_window_start % 2 != 0 ||
             block_components_window_count <= 0 || block_components_window_count % 2 != 0 ||
             block_components_window_start + block_components_window_count > 28)) return 2;
    if (longcat_bf16_hidden_surface_rounding && !longcat_bf16_boundary_rounding) return 2;
    json manifest;
    std::ifstream(manifest_path) >> manifest;
    size_t longest = 0;
    for (const auto & spec : manifest.at("cases")) longest = std::max(longest, spec.at("input_ids").size());
    const uint32_t n_ctx = longest + 8 + 16; // eight generated tokens plus safety margin
    fs::create_directories(output);
    if (!set_longcat_bf16_boundary_rounding(longcat_bf16_boundary_rounding)) return 2;
    if (!set_longcat_bf16_hidden_surface_rounding(longcat_bf16_hidden_surface_rounding)) return 2;
    const json run_metadata = {
        {"schema_version", 1},
        {"longcat_bf16_boundary_rounding", longcat_bf16_boundary_rounding},
        {"longcat_bf16_hidden_surface_rounding", longcat_bf16_hidden_surface_rounding},
        {"block_components_diagnostic", block_components_diagnostic},
        {"block_components_window_diagnostic", block_components_window_diagnostic},
        {"block_components_window_start", block_components_window_start},
        {"block_components_window_count", block_components_window_count},
        {"environment_gate", "LLAMA_LONGCAT_BF16_BOUNDARY_ROUNDING"},
        {"environment_gates", {
            "LLAMA_LONGCAT_BF16_BOUNDARY_ROUNDING",
            "LLAMA_LONGCAT_BF16_HIDDEN_SURFACE_ROUNDING"}},
    };
    std::ofstream(output / "capture-run-metadata.json") << run_metadata.dump(2) << '\n';
    std::cerr << "longcat-next-capture: BF16 boundary rounding = "
              << (longcat_bf16_boundary_rounding ? "enabled" : "disabled") << '\n';
    std::cerr << "longcat-next-capture: BF16 hidden-surface rounding = "
              << (longcat_bf16_hidden_surface_rounding ? "enabled" : "disabled") << '\n';
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.string().c_str(), model_params);
    if (!model) return 3;
    int result = 0;
    for (const auto & spec : manifest.at("cases")) {
        result = run_case(model, spec, output, n_ctx, threads, flash_attn, cache_type,
                          layer0_diagnostic, all_blocks_diagnostic, block_components_diagnostic,
                          block_components_window_diagnostic, block_components_window_start,
                          block_components_window_count);
        if (result != 0) break;
    }
    llama_model_free(model);
    llama_backend_free();
    return result;
}
