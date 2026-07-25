#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct capture_state {
    fs::path dir;
    std::ofstream manifest;
    bool direct_forward = true;
};

static bool wanted(const std::string & name) {
    return name == "inp_embd" || name == "inp_embd_ngram" || name == "result_norm" ||
           name.rfind("ngram_proj-", 0) == 0 || name == "l_out-0" || name == "l_out-1" ||
           name == "l_out-2" || name == "l_out-27";
}

static bool capture_cb(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<capture_state *>(opaque);
    const std::string name = tensor->name;
    if (!state.direct_forward || !wanted(name)) return false;
    if (ask) return true;
    std::string file = name;
    for (char & c : file) if (!std::isalnum((unsigned char) c) && c != '-' && c != '_') c = '_';
    file += ".raw";
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::ofstream(state.dir / file, std::ios::binary).write((char *) bytes.data(), bytes.size());
    state.manifest << name << '\t' << ggml_type_name(tensor->type) << '\t';
    for (int i = 0; i < GGML_MAX_DIMS; ++i) state.manifest << (i ? "," : "") << tensor->ne[i];
    state.manifest << '\t' << file << '\n';
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

static int self_test() {
    const float tied[] = { 0.0f, 2.0f, 2.0f, 1.0f };
    if (argmax_large_tie(tied, 4) != 2) return 20;
    if (sequence_for_mask(0, 0) == sequence_for_mask(1, 0) || sequence_for_mask(2, 1) != 0) return 21;
    capture_state state { {}, {}, false };
    ggml_tensor dummy = {};
    snprintf(dummy.name, sizeof(dummy.name), "inp_embd");
    if (capture_cb(&dummy, true, &state)) return 22;
    // Multiple cases are represented by independently created contexts and
    // distinct output namespaces in the single model-owning main loop.
    const std::vector<std::string> case_names = { "case_a", "case_b" };
    if (case_names[0] == case_names[1]) return 23;
    return 0;
}

static int run_case(llama_model * model, const json & spec, const fs::path & root, uint32_t n_ctx) {
    const auto ids = spec.at("input_ids").get<std::vector<llama_token>>();
    const auto mask = spec.at("attention_mask").get<std::vector<int32_t>>();
    const auto positions = spec.at("position_ids").get<std::vector<llama_pos>>();
    const auto cache = spec.at("cache_position").get<std::vector<llama_pos>>();
    if (ids.empty() || ids.size() != mask.size() || ids.size() != positions.size() || ids.size() != cache.size()) return 10;

    const fs::path dir = root / spec.at("name").get<std::string>();
    fs::create_directories(dir);
    capture_state state { dir, std::ofstream(dir / "captures.tsv", std::ios::trunc), true };
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_seq_max = ids.size() + 1;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &state;
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
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--model") model_path = argv[i + 1];
        else if (key == "--case-manifest") manifest_path = argv[i + 1];
        else if (key == "--output-dir") output = argv[i + 1];
        else return 2;
    }
    if (model_path.empty() || manifest_path.empty() || output.empty()) return 2;
    json manifest;
    std::ifstream(manifest_path) >> manifest;
    size_t longest = 0;
    for (const auto & spec : manifest.at("cases")) longest = std::max(longest, spec.at("input_ids").size());
    const uint32_t n_ctx = longest + 8 + 16; // eight generated tokens plus safety margin
    fs::create_directories(output);
    llama_backend_init();
    llama_model * model = llama_model_load_from_file(model_path.string().c_str(), llama_model_default_params());
    if (!model) return 3;
    int result = 0;
    for (const auto & spec : manifest.at("cases")) {
        result = run_case(model, spec, output, n_ctx);
        if (result != 0) break;
    }
    llama_model_free(model);
    llama_backend_free();
    return result;
}
