#include "llama-graph.h"
#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

static constexpr int64_t LONGCAT_VOCAB = 131072;
static constexpr int64_t LONGCAT_M = 78 * LONGCAT_VOCAB;
static constexpr int32_t LONGCAT_SPLIT = 4;
static constexpr int32_t LONGCAT_NEIGHBOR = 4;
static constexpr int32_t LONGCAT_EMBEDDERS = 12;
static constexpr int32_t LONGCAT_EOS = 2;

struct ngram_expect {
    std::string name;
    std::vector<int32_t> tokens;
    int pos;
    int32_t ng2;
    int32_t ng3;
    int32_t ng4;
};

static llama_ubatch make_ubatch(
        const std::vector<int32_t> & tokens,
        const std::vector<int32_t> & positions,
        const std::vector<std::vector<int32_t>> & seq_ids) {
    llama_ubatch ubatch = {};
    ubatch.data = std::make_shared<llama_ubatch::data_t>();
    auto & data = *ubatch.data;

    const uint32_t n_tokens = (uint32_t) tokens.size();
    data.token.assign(tokens.begin(), tokens.end());
    data.pos.assign(positions.begin(), positions.end());
    data.n_seq_id.resize(n_tokens);
    data.seq_id.resize(n_tokens);
    size_t n_seq_ids = 0;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        n_seq_ids += seq_ids[i].size();
    }
    data.seq_id_data.reserve(n_seq_ids);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        data.n_seq_id[i] = (int32_t) seq_ids[i].size();
        data.seq_id[i] = data.seq_id_data.data() + data.seq_id_data.size();
        data.seq_id_data.insert(data.seq_id_data.end(), seq_ids[i].begin(), seq_ids[i].end());
    }

    ubatch.n_tokens = n_tokens;
    ubatch.n_seq_tokens = n_tokens;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;
    ubatch.token = data.token.data();
    ubatch.pos = data.pos.data();
    ubatch.n_seq_id = data.n_seq_id.data();
    ubatch.seq_id = data.seq_id.data();
    return ubatch;
}

static llama_ubatch make_ubatch(
        const std::vector<int32_t> & tokens,
        const std::vector<int32_t> & positions,
        const std::vector<int32_t> & seq_ids) {
    std::vector<std::vector<int32_t>> seq_ids_multi(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        seq_ids_multi[i] = { seq_ids[i] };
    }
    return make_ubatch(tokens, positions, seq_ids_multi);
}

class production_ngram_runner {
public:
    production_ngram_runner(int32_t ignored_start = 0, int32_t ignored_count = 0) {
        struct ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 32,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        input = std::make_unique<llm_graph_input_ngram>(
            LONGCAT_EMBEDDERS, LONGCAT_NEIGHBOR, LONGCAT_SPLIT,
            LONGCAT_VOCAB, LONGCAT_M, LONGCAT_EOS, ignored_start, ignored_count, &history);
        for (int i = 0; i < LONGCAT_EMBEDDERS; ++i) {
            input->ngram_ids[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
            input->lookup_masks[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        }
        if (ignored_count > 0) {
            input->preserve_base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
        }
        backend = ggml_backend_cpu_init();
        GGML_ASSERT(backend != nullptr);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer != nullptr);
    }

    ~production_ngram_runner() {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    std::array<int32_t, 3> append_one(int32_t token, int32_t pos, int32_t seq_id = 0) {
        const auto ubatch = make_ubatch(std::vector<int32_t>{ token }, std::vector<int32_t>{ pos }, std::vector<int32_t>{ seq_id });
        input->set_input(&ubatch);
        return read_row(0);
    }

    std::array<int32_t, LONGCAT_EMBEDDERS> append_all(int32_t token, int32_t pos, int32_t seq_id = 0) {
        const auto ubatch = make_ubatch(std::vector<int32_t>{ token }, std::vector<int32_t>{ pos }, std::vector<int32_t>{ seq_id });
        input->set_input(&ubatch);
        std::array<int32_t, LONGCAT_EMBEDDERS> result = {};
        for (int i = 0; i < LONGCAT_EMBEDDERS; ++i) {
            ggml_backend_tensor_get(input->ngram_ids[i], &result[i], 0, sizeof(result[i]));
        }
        return result;
    }

    float preserve_mask() const {
        float result = 0.0f;
        ggml_backend_tensor_get(input->preserve_base, &result, 0, sizeof(result));
        return result;
    }

    float lookup_mask(int table) const {
        float result = -1.0f;
        ggml_backend_tensor_get(input->lookup_masks[table], &result, 0, sizeof(result));
        return result;
    }

    int32_t hash_id(int table) const {
        int32_t result = -1;
        ggml_backend_tensor_get(input->ngram_ids[table], &result, 0, sizeof(result));
        return result;
    }

    std::array<int32_t, 3> append_many(
            const std::vector<int32_t> & tokens,
            const std::vector<int32_t> & positions,
            int row,
            int32_t seq_id = 0) {
        const std::vector<int32_t> seq_ids(tokens.size(), seq_id);
        const auto ubatch = make_ubatch(tokens, positions, seq_ids);
        input->set_input(&ubatch);
        return read_row(row);
    }

    std::array<int32_t, 3> append_shared(
            int32_t token,
            int32_t pos,
            const std::vector<int32_t> & seq_ids) {
        const auto ubatch = make_ubatch(std::vector<int32_t>{ token }, std::vector<int32_t>{ pos }, std::vector<std::vector<int32_t>>{ seq_ids });
        input->set_input(&ubatch);
        return read_row(0);
    }

    std::array<int32_t, 3> run_prefill(const std::vector<int32_t> & tokens, int pos, int32_t seq_id = 0) {
        std::vector<int32_t> positions(tokens.size());
        std::vector<int32_t> seq_ids(tokens.size(), seq_id);
        for (size_t i = 0; i < tokens.size(); ++i) {
            positions[i] = (int32_t) i;
        }
        const auto ubatch = make_ubatch(tokens, positions, seq_ids);
        input->set_input(&ubatch);
        return read_row(pos);
    }

    std::array<int32_t, LONGCAT_EMBEDDERS> run_prefill_all(const std::vector<int32_t> & tokens, int pos, int32_t seq_id = 0) {
        std::vector<int32_t> positions(tokens.size());
        std::vector<int32_t> seq_ids(tokens.size(), seq_id);
        for (size_t i = 0; i < tokens.size(); ++i) positions[i] = (int32_t) i;
        const auto ubatch = make_ubatch(tokens, positions, seq_ids);
        input->set_input(&ubatch);
        std::array<int32_t, LONGCAT_EMBEDDERS> result = {};
        for (int i = 0; i < LONGCAT_EMBEDDERS; ++i) {
            ggml_backend_tensor_get(input->ngram_ids[i], &result[i], pos * sizeof(int32_t), sizeof(int32_t));
        }
        return result;
    }

    void seed_history(int32_t seq_id, const std::vector<std::pair<llama_pos, llama_token>> & entries) {
        auto & hist = history[seq_id];
        hist.clear();
        for (const auto & entry : entries) {
            hist.push_back(entry);
        }
    }

    std::string history_trace(int32_t seq_id = 0) const {
        const auto it = history.find(seq_id);
        if (it == history.end()) {
            return "[]";
        }

        std::ostringstream out;
        out << "[";
        bool first = true;
        for (const auto & entry : it->second) {
            if (!first) {
                out << ",";
            }
            out << entry.first << ":" << entry.second;
            first = false;
        }
        out << "]";
        return out.str();
    }

    size_t history_size(int32_t seq_id = 0) const {
        const auto it = history.find(seq_id);
        return it == history.end() ? 0 : it->second.size();
    }

private:
    std::array<int32_t, 3> read_row(int row) const {
        std::array<int32_t, 3> ids = {};
        ggml_backend_tensor_get(input->ngram_ids[0], &ids[0], row * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_get(input->ngram_ids[4], &ids[1], row * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_get(input->ngram_ids[8], &ids[2], row * sizeof(int32_t), sizeof(int32_t));
        return ids;
    }

    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    llm_ngram_token_history history;
    std::unique_ptr<llm_graph_input_ngram> input;
};

static void test_production_ngram(testing & t) {
    const std::vector<ngram_expect> cases = {
        { "no eos",                 {10, 11, 12, 13, 14},       4, 1703950,  736033, 8151286 },
        { "eos token",              {10, 11,  2, 14, 15, 16},   2, 1441794, 2339134, 2204702 },
        { "eos plus one",           {10, 11,  2, 14, 15, 16},   3,      14,      14,      14 },
        { "eos plus two",           {10, 11,  2, 14, 15, 16},   4, 1835023, 1835023, 1835023 },
        { "eos plus three",         {10, 11,  2, 14, 15, 16},   5, 1966096, 9356547, 9168347 },
        { "multiple eos turns",     { 3,  2,  4,  5,  2, 6, 7, 8}, 7, 917512, 5545366, 5464710 },
        { "eos at zero",            { 2, 10, 11, 12},           3, 1441804, 2339144, 2204712 },
        { "consecutive eos",        {10,  2,  2, 11, 12, 13},   5, 1572877, 6649401, 6501529 },
        { "shorter than shift",     {10, 11},                   1, 1310731, 1310731, 1310731 },
        { "speculative replacement",{10, 11, 99, 100},          3, 2752611, 7829127, 6204250 },
    };

    for (const auto & tc : cases) {
        production_ngram_runner runner;
        const auto ids = runner.run_prefill(tc.tokens, tc.pos);
        t.assert_equal(tc.name + " ng2", tc.ng2, ids[0]);
        t.assert_equal(tc.name + " ng3", tc.ng3, ids[1]);
        t.assert_equal(tc.name + " ng4", tc.ng4, ids[2]);
        t.assert_true(tc.name + " bounded history", runner.history_size() <= (size_t) (LONGCAT_NEIGHBOR - 1 + tc.tokens.size()));
    }

    production_ngram_runner runner;
    runner.append_one(10, 0);
    runner.append_one(11, 1);
    runner.append_one(12, 2);
    const auto replacement = runner.append_one(99, 2);
    t.assert_equal("rollback replacement ng2", 1441891, replacement[0]);
    t.assert_equal("rollback replacement ng3", 2339231, replacement[1]);
    t.assert_equal("rollback replacement ng4", 2204799, replacement[2]);
    t.assert_true("rollback replacement bounded history", runner.history_size() <= (size_t) LONGCAT_NEIGHBOR);

    production_ngram_runner speculative;
    speculative.seed_history(0, { { 97, 97 }, { 98, 98 }, { 99, 99 } });
    t.assert_equal("speculative before first call", std::string("[97:97,98:98,99:99]"), speculative.history_trace());
    speculative.append_many({ 100, 888 }, { 100, 101 }, 1);
    t.assert_equal("speculative after draft ubatch", std::string("[97:97,98:98,99:99,100:100,101:888]"), speculative.history_trace());
    const auto repl = speculative.append_one(101, 101);
    t.assert_equal("speculative after rollback replacement", std::string("[98:98,99:99,100:100,101:101]"), speculative.history_trace());
    t.assert_equal("replacement uses p for ng2", 2883684, repl[0]);
    t.assert_equal("replacement uses p and p-1 for ng3", 7677892, repl[1]);
    t.assert_equal("replacement uses p p-1 p-2 for ng4", 4140793, repl[2]);
    t.assert_true("speculative lifecycle bounded history", speculative.history_size() <= (size_t) (2 * LONGCAT_NEIGHBOR));

    production_ngram_runner multi;
    multi.append_one(10, 0, 0);
    multi.append_one(2, 1, 0);
    const auto seq_a = multi.append_one(20, 2, 0);
    const auto seq_b = multi.append_one(32, 2, 1);
    t.assert_equal("independent sequence a ng2", 20, seq_a[0]);
    t.assert_equal("independent sequence b ng2", 32, seq_b[0]);

    production_ngram_runner shared;
    shared.append_one(10, 0, 0);
    shared.append_one(11, 1, 0);
    shared.append_one(12, 2, 0);
    shared.append_one(10, 0, 1);
    shared.append_one(11, 1, 1);
    shared.append_one(12, 2, 1);
    const auto shared_ids = shared.append_shared(13, 3, { 0, 1 });
    t.assert_equal("shared sequence identical history ng2", 1572877, shared_ids[0]);
    t.assert_equal("shared sequence identical history ng3", 6649401, shared_ids[1]);
    t.assert_equal("shared sequence identical history ng4", 5024532, shared_ids[2]);
    t.assert_equal("shared sequence history updated a", std::string("[0:10,1:11,2:12,3:13]"), shared.history_trace(0));
    t.assert_equal("shared sequence history updated b", std::string("[0:10,1:11,2:12,3:13]"), shared.history_trace(1));
}

static void test_longcat_next_ignored_interval(testing & t) {
    for (int32_t token = 131072; token < 131125; ++token) {
        production_ngram_runner runner(131072, 53);
        const auto hashes = runner.append_all(token, 0);
        for (int i = 0; i < LONGCAT_EMBEDDERS; ++i) {
            t.assert_equal("ignored token hash zero", 0, hashes[i]);
        }
        t.assert_equal("ignored token preserves base", 1.0f, runner.preserve_mask());
    }

    production_ngram_runner literal_zero(131072, 53);
    const auto zero_hashes = literal_zero.append_all(0, 0);
    for (int i = 0; i < LONGCAT_EMBEDDERS; ++i) {
        t.assert_equal("literal zero hash zero", 0, zero_hashes[i]);
        t.assert_equal("literal zero lookup masked", 0.0f, literal_zero.lookup_mask(i));
    }
    t.assert_equal("literal zero still scales", 0.0f, literal_zero.preserve_mask());

    production_ngram_runner max_text(131072, 53);
    max_text.append_all(131071, 0);
    t.assert_equal("max text token still scales", 0.0f, max_text.preserve_mask());

    // Literal zero and normalized ignored controls terminate every visible
    // order, not merely the immediately preceding bigram.
    for (int32_t boundary : { 0, 131072, 131124 }) {
        production_ngram_runner after(131072, 53);
        const auto ids = after.run_prefill_all({ 19, boundary, 29, 31, 37 }, 2);
        for (int split = 0; split < 4; ++split) {
            t.assert_equal("order-3 stops at zero boundary", 29, ids[4 + split]);
            t.assert_equal("order-4 stops at zero boundary", 29, ids[8 + split]);
        }
        production_ngram_runner incremental(131072, 53);
        std::array<int32_t, LONGCAT_EMBEDDERS> incremental_ids = {};
        const std::vector<int32_t> sequence = { 19, boundary, 29, 31, 37 };
        for (size_t pos = 0; pos < sequence.size(); ++pos) {
            incremental_ids = incremental.append_all(sequence[pos], pos);
        }
        production_ngram_runner prompt(131072, 53);
        const auto prompt_ids = prompt.run_prefill_all(sequence, sequence.size() - 1);
        for (int table = 0; table < LONGCAT_EMBEDDERS; ++table) {
            t.assert_equal("prompt and incremental all-table equivalence", prompt_ids[table], incremental_ids[table]);
        }
    }
}

static float execute_masked_projection(int32_t id, float lookup_mask) {
    ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
    ggml_tensor * proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * out = ggml_mul(ctx, ggml_mul_mat(ctx, proj, ggml_get_rows(ctx, table, ids)),
        ggml_reshape_2d(ctx, mask, 1, 1));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const float table_data[] = { 123.0f, 7.0f };
    const float one = 1.0f;
    ggml_backend_tensor_set(table, table_data, 0, sizeof(table_data));
    ggml_backend_tensor_set(proj, &one, 0, sizeof(one));
    ggml_backend_tensor_set(ids, &id, 0, sizeof(id));
    ggml_backend_tensor_set(mask, &lookup_mask, 0, sizeof(lookup_mask));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    float result;
    ggml_backend_tensor_get(out, &result, 0, sizeof(result));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

static void test_zero_hash_projection_mask(testing & t) {
    production_ngram_runner zero(131072, 53);
    zero.append_all(0, 0);
    for (int table = 0; table < LONGCAT_EMBEDDERS; ++table) {
        t.assert_equal("row zero deliberately nonzero but masked", 0.0f,
            execute_masked_projection(zero.hash_id(table), zero.lookup_mask(table)));
        t.assert_equal("nonzero hash keeps correct row for every table", 7.0f,
            execute_masked_projection(1, 1.0f));
    }
}

int main() {
    testing t(std::cout);
    t.test("longcat production ngram", test_production_ngram);
    t.test("longcat-next ignored interval", test_longcat_next_ignored_interval);
    t.test("longcat-next zero hash projection mask", test_zero_hash_projection_mask);
    return t.summary();
}
