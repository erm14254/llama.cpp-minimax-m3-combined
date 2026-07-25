#include "ggml-backend.h"
#include "llama.h"

#include <filesystem>
#include <cctype>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct capture_state {
    fs::path dir;
    std::ofstream manifest;
};

static bool wanted(const std::string & name) {
    return name == "inp_embd" || name == "inp_embd_ngram" || name == "result_norm" ||
           name.rfind("ngram_proj-", 0) == 0 || name == "l_out-0" || name == "l_out-1" ||
           name == "l_out-2" || name == "l_out-27";
}

static bool capture_cb(ggml_tensor * tensor, bool ask, void * opaque) {
    const std::string name = tensor->name;
    if (!wanted(name)) return false;
    if (ask) return true;
    auto & state = *static_cast<capture_state *>(opaque);
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

static std::vector<llama_token> parse_tokens(const std::string & value) {
    std::vector<llama_token> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) result.push_back(std::stoi(item));
    return result;
}

int main(int argc, char ** argv) {
    std::string model_path, token_string;
    fs::path output;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        if (key == "--model") model_path = argv[i + 1];
        else if (key == "--tokens") token_string = argv[i + 1];
        else if (key == "--output-dir") output = argv[i + 1];
        else return 2;
    }
    if (model_path.empty() || token_string.empty() || output.empty()) return 2;
    fs::create_directories(output);
    capture_state state { output, std::ofstream(output / "captures.tsv", std::ios::trunc) };
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 131072;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &state;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    auto tokens = parse_tokens(token_string);
    {
        std::ofstream input(output / "inputs.json");
        input << "{\"input_ids\":[";
        for (size_t i = 0; i < tokens.size(); ++i) input << (i ? "," : "") << tokens[i];
        input << "],\"attention_mask\":[";
        for (size_t i = 0; i < tokens.size(); ++i) input << (i ? ",1" : "1");
        input << "],\"position_ids\":[";
        for (size_t i = 0; i < tokens.size(); ++i) input << (i ? "," : "") << i;
        input << "],\"cache_position\":[";
        for (size_t i = 0; i < tokens.size(); ++i) input << (i ? "," : "") << i;
        input << "]}\n";
    }
    const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    if (rc == 0) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::ofstream(output / "final_logits.f32.raw", std::ios::binary).write((const char *) logits, n_vocab * sizeof(float));
        state.manifest << "final_logits\tf32\t" << n_vocab << ",1,1,1\tfinal_logits.f32.raw\n";
        std::vector<int32_t> order(n_vocab);
        for (int32_t i = 0; i < n_vocab; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + std::min(10, n_vocab), order.end(),
            [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        std::ofstream summary(output / "decoding.json");
        summary << "{\"argmax_id\":" << order[0] << ",\"greedy_continuation_ids\":[" << order[0]
                << "],\"top_k_ids\":[";
        for (int i = 0; i < std::min(10, n_vocab); ++i) summary << (i ? "," : "") << order[i];
        summary << "],\"top_k_values\":[";
        for (int i = 0; i < std::min(10, n_vocab); ++i) summary << (i ? "," : "") << logits[order[i]];
        summary << "]}\n";
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc == 0 ? 0 : 5;
}
