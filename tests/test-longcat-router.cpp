#include "llama-graph.h"
#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct route_result {
    std::vector<int> ids;
    std::vector<double> weights;
    double real = 0.0;
    double identity = 0.0;
};

struct route_token_result {
    std::array<int32_t, 2> selected;
    std::array<int32_t, 2> selected_real;
    std::array<float, 2> weights;
    std::array<float, 2> weights_real;
    float identity_sum;
};

static std::vector<double> softmax(const std::vector<double> & logits) {
    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (double & p : probs) {
        p /= sum;
    }
    return probs;
}

static route_result longcat_route(
        const std::vector<double> & logits,
        const std::vector<double> & bias,
        int n_real,
        int top_k,
        double scale) {
    const auto probs = softmax(logits);
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] + bias[a] > probs[b] + bias[b];
    });

    route_result res;
    for (int i = 0; i < top_k; ++i) {
        const int id = order[i];
        const double weight = probs[id] * scale;
        res.ids.push_back(id);
        res.weights.push_back(weight);
        if (id < n_real) {
            res.real += weight;
        } else {
            res.identity += weight;
        }
    }
    return res;
}

static bool near(double a, double b) {
    return std::fabs(a - b) < 1e-6;
}

static void test_router(testing & t) {
    struct router_case {
        std::string name;
        std::vector<double> logits;
        std::vector<double> bias;
        double real;
        double identity;
    };

    const std::vector<double> no_bias(5, 0.0);
    const std::vector<router_case> cases = {
        { "identity highest", {0, 0, 0, 5, 4}, no_bias, 0.0, 5.912626155935248 },
        { "real highest", {5, 4, 3, 0, 0}, no_bias, 5.411305738577755, 0.0 },
        { "mixed topk", {5, 0, 0, 4, 0}, no_bias, 4.3224760735286125, 1.590150082406636 },
        { "bias changes", {0, 4, 3, 0, 0}, {0, 0, 0, 10, 0}, 4.216958708242208, 0.07723629290886723 },
        { "identity outside topk", {5, 4, 3, 2, 1}, no_bias, 5.223181822869405, 0.0 },
        { "extreme", {1000, -1000, 999, 998, -999}, no_bias, 5.459816560977717, 0.0 },
    };

    for (const auto & tc : cases) {
        const auto res = longcat_route(tc.logits, tc.bias, 3, 2, 6.0);
        t.assert_true(tc.name + " real", near(tc.real, res.real));
        t.assert_true(tc.name + " identity", near(tc.identity, res.identity));
        t.assert_true(tc.name + " total bounded", res.real + res.identity <= 6.0 + 1e-9);
    }

    const auto all_equal = longcat_route({0, 0, 0, 0, 0}, no_bias, 3, 2, 6.0);
    t.assert_true("all equal chooses two experts", all_equal.ids.size() == 2);
    t.assert_true("all equal has scaled selected weight", near(all_equal.weights[0], 1.2));
    t.assert_true("all equal has scaled selected weight 2", near(all_equal.weights[1], 1.2));

    const std::vector<std::vector<double>> batch = {
        {5, 0, 0, 4, 0},
        {0, 0, 0, 5, 4},
    };
    double identity = 0.0;
    for (const auto & logits : batch) {
        identity += longcat_route(logits, no_bias, 3, 2, 6.0).identity;
    }
    t.assert_true("multiple token batch identity contribution", identity > 7.0);
}

static std::vector<route_token_result> run_production_route(
        const std::vector<float> & logits_data,
        const std::vector<float> & bias_data,
        int n_tokens) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, n_tokens);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    auto route = llm_graph_build_longcat_moe_route(ctx, logits, bias, n_tokens, 3, 5, 2, 6.0f);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(gf, route.selected_experts);
    ggml_build_forward_expand(gf, route.selected_real);
    ggml_build_forward_expand(gf, route.weights);
    ggml_build_forward_expand(gf, route.weights_real);
    ggml_build_forward_expand(gf, route.identity_weight_sum);

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> logits_storage(5 * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        for (int expert = 0; expert < 5; ++expert) {
            logits_storage[expert + token * 5] = logits_data[token * 5 + expert];
        }
    }
    ggml_backend_tensor_set(logits, logits_storage.data(), 0, ggml_nbytes(logits));
    ggml_backend_tensor_set(bias, bias_data.data(), 0, ggml_nbytes(bias));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);

    std::vector<route_token_result> res(n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        for (int k = 0; k < 2; ++k) {
            ggml_backend_tensor_get(route.selected_experts, &res[token].selected[k],
                k * route.selected_experts->nb[0] + token * route.selected_experts->nb[1], sizeof(int32_t));
            ggml_backend_tensor_get(route.selected_real, &res[token].selected_real[k],
                k * route.selected_real->nb[0] + token * route.selected_real->nb[1], sizeof(int32_t));
            ggml_backend_tensor_get(route.weights, &res[token].weights[k],
                k * route.weights->nb[1] + token * route.weights->nb[2], sizeof(float));
            ggml_backend_tensor_get(route.weights_real, &res[token].weights_real[k],
                k * route.weights_real->nb[1] + token * route.weights_real->nb[2], sizeof(float));
        }
        ggml_backend_tensor_get(route.identity_weight_sum, &res[token].identity_sum,
            token * route.identity_weight_sum->nb[1], sizeof(float));
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return res;
}

static void test_production_route_helper(testing & t) {
    const std::vector<float> no_bias = { 0, 0, 0, 0, 0 };

    const auto identity = run_production_route({ 0, 0, 0, 5, 4 }, no_bias, 1)[0];
    t.assert_true("production route selects identity first", identity.selected[0] >= 3);
    t.assert_true("production route selects identity second", identity.selected[1] >= 3);
    t.assert_true("identity selected mass scaled once", near(5.912626155935248, identity.identity_sum));
    t.assert_true("identity selections have zero real weight 0", near(0.0, identity.weights_real[0]));
    t.assert_true("identity selections have zero real weight 1", near(0.0, identity.weights_real[1]));

    const auto mixed = run_production_route({ 5, 0, 0, 4, 0 }, no_bias, 1)[0];
    t.assert_true("production route mixed real", mixed.selected[0] < 3 || mixed.selected[1] < 3);
    t.assert_true("production route mixed identity", mixed.selected[0] >= 3 || mixed.selected[1] >= 3);
    t.assert_true("mixed real contribution", near(4.3224760735286125, mixed.weights_real[0] + mixed.weights_real[1]));
    t.assert_true("mixed identity contribution excludes unselected", near(1.590150082406636, mixed.identity_sum));

    const auto biased = run_production_route({ 0, 4, 3, 0, 0 }, { 0, 0, 0, 10, 0 }, 1)[0];
    t.assert_true("production route bias selects identity", biased.selected[0] >= 3 || biased.selected[1] >= 3);
    t.assert_true("bias does not alter selected weights", near(0.07723629290886723, biased.identity_sum));

    const std::vector<float> logits = {
        0, 0, 0, 5, 4,
        5, 0, 0, 4, 0,
        5, 4, 3, 0, 0,
    };
    const auto batch = run_production_route(logits, no_bias, 3);

    t.assert_equal("token 0 selected 0", 3, batch[0].selected[0]);
    t.assert_equal("token 0 selected 1", 4, batch[0].selected[1]);
    t.assert_equal("token 0 remap 0", 0, batch[0].selected_real[0]);
    t.assert_equal("token 0 remap 1", 0, batch[0].selected_real[1]);
    t.assert_true("token 0 scaled selected weight 0", near(4.322475910186768, batch[0].weights[0]));
    t.assert_true("token 0 scaled selected weight 1", near(1.5901501178741455, batch[0].weights[1]));
    t.assert_true("token 0 real weight 0", near(0.0, batch[0].weights_real[0]));
    t.assert_true("token 0 real weight 1", near(0.0, batch[0].weights_real[1]));
    t.assert_true("token 0 identity sum", near(5.912626266479492, batch[0].identity_sum));

    t.assert_equal("token 1 selected 0", 0, batch[1].selected[0]);
    t.assert_equal("token 1 selected 1", 3, batch[1].selected[1]);
    t.assert_equal("token 1 remap 0", 0, batch[1].selected_real[0]);
    t.assert_equal("token 1 remap 1", 0, batch[1].selected_real[1]);
    t.assert_true("token 1 scaled selected weight 0", near(4.322475910186768, batch[1].weights[0]));
    t.assert_true("token 1 scaled selected weight 1", near(1.5901501178741455, batch[1].weights[1]));
    t.assert_true("token 1 real weight 0", near(4.322475910186768, batch[1].weights_real[0]));
    t.assert_true("token 1 real weight 1", near(0.0, batch[1].weights_real[1]));
    t.assert_true("token 1 identity sum", near(1.5901501178741455, batch[1].identity_sum));

    t.assert_equal("token 2 selected 0", 0, batch[2].selected[0]);
    t.assert_equal("token 2 selected 1", 1, batch[2].selected[1]);
    t.assert_equal("token 2 remap 0", 0, batch[2].selected_real[0]);
    t.assert_equal("token 2 remap 1", 1, batch[2].selected_real[1]);
    t.assert_true("token 2 scaled selected weight 0", near(3.9559807777404785, batch[2].weights[0]));
    t.assert_true("token 2 scaled selected weight 1", near(1.4553236961364746, batch[2].weights[1]));
    t.assert_true("token 2 real weight 0", near(3.9559807777404785, batch[2].weights_real[0]));
    t.assert_true("token 2 real weight 1", near(1.4553236961364746, batch[2].weights_real[1]));
    t.assert_true("token 2 identity sum", near(0.0, batch[2].identity_sum));
}

static void test_odd_ffn_association(testing & t) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * residual = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 1);
    ggml_tensor * dense    = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 1);
    ggml_tensor * shortcut = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 1);
    ggml_tensor * even = llm_graph_build_longcat_even_ffn_output(ctx, residual, dense);
    t.assert_true("even output is one add", even->op == GGML_OP_ADD);
    t.assert_true("even add starts with residual", even->src[0] == residual);
    t.assert_true("even add uses dense output once", even->src[1] == dense);
    ggml_tensor * saved_shortcut = shortcut;
    ggml_tensor * official = llm_graph_build_longcat_odd_ffn_output(
        ctx, residual, dense, shortcut);

    t.assert_true("odd helper clears shortcut", shortcut == nullptr);
    t.assert_true("odd final node is add", official->op == GGML_OP_ADD);
    t.assert_true("odd final add uses shortcut once", official->src[1] == saved_shortcut);
    ggml_tensor * residual_dense = official->src[0];
    t.assert_true("odd first node is add", residual_dense->op == GGML_OP_ADD);
    t.assert_true("odd first add starts with residual", residual_dense->src[0] == residual);
    t.assert_true("odd first add then uses dense output", residual_dense->src[1] == dense);

    // Retain the previous association only as a numerical counterexample.
    ggml_tensor * old_grouping = ggml_add(ctx, ggml_add(ctx, dense, saved_shortcut), residual);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, official);
    ggml_build_forward_expand(graph, old_grouping);

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    const ggml_bf16_t residual_value = ggml_fp32_to_bf16(-1000.0f);
    const ggml_bf16_t dense_value = ggml_fp32_to_bf16(-1000.0f);
    const ggml_bf16_t shortcut_value = ggml_fp32_to_bf16(-100.0f);
    ggml_backend_tensor_set(residual, &residual_value, 0, sizeof(residual_value));
    ggml_backend_tensor_set(dense, &dense_value, 0, sizeof(dense_value));
    ggml_backend_tensor_set(saved_shortcut, &shortcut_value, 0, sizeof(shortcut_value));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    ggml_bf16_t official_value;
    ggml_bf16_t old_value;
    ggml_backend_tensor_get(official, &official_value, 0, sizeof(official_value));
    ggml_backend_tensor_get(old_grouping, &old_value, 0, sizeof(old_value));
    t.assert_true("official BF16 association expected value",
                  ggml_bf16_to_fp32(official_value) == -2096.0f);
    t.assert_true("old BF16 association differs",
                  ggml_bf16_to_fp32(old_value) == -2112.0f);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void test_capture_alias_preserves_source_name(testing & t) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);

    ggml_tensor * embedding_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * embedding = ggml_scale(ctx, embedding_input, 1.0f);
    ggml_set_name(embedding, "inp_embd_ngram");
    ggml_tensor * block_0 = llm_graph_build_longcat_capture_alias(ctx, graph, embedding);
    ggml_set_name(block_0, "block_in-0");

    ggml_tensor * layer_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * layer_out = ggml_scale(ctx, layer_input, 1.0f);
    ggml_set_name(layer_out, "l_out-0");
    ggml_tensor * block_1 = llm_graph_build_longcat_capture_alias(ctx, graph, layer_out);
    ggml_set_name(block_1, "block_in-1");

    t.assert_true("embedding alias has distinct tensor pointer", block_0 != embedding);
    t.assert_true("layer output alias has distinct tensor pointer", block_1 != layer_out);
    t.assert_true("embedding source name survives alias naming",
                  std::strcmp(ggml_get_name(embedding), "inp_embd_ngram") == 0);
    t.assert_true("layer output source name survives alias naming",
                  std::strcmp(ggml_get_name(layer_out), "l_out-0") == 0);
    t.assert_true("block zero alias has independent name",
                  std::strcmp(ggml_get_name(block_0), "block_in-0") == 0);
    t.assert_true("block one alias has independent name",
                  std::strcmp(ggml_get_name(block_1), "block_in-1") == 0);
    t.assert_true("embedding alias shape matches", ggml_are_same_shape(block_0, embedding));
    t.assert_true("layer output alias shape matches", ggml_are_same_shape(block_1, layer_out));
    t.assert_true("embedding alias dtype matches", block_0->type == embedding->type);
    t.assert_true("layer output alias dtype matches", block_1->type == layer_out->type);

    t.assert_true("embedding source independently addressable",
                  ggml_graph_get_tensor(graph, "inp_embd_ngram") == embedding);
    t.assert_true("block zero alias independently addressable",
                  ggml_graph_get_tensor(graph, "block_in-0") == block_0);
    t.assert_true("layer output independently addressable",
                  ggml_graph_get_tensor(graph, "l_out-0") == layer_out);
    t.assert_true("block one alias independently addressable",
                  ggml_graph_get_tensor(graph, "block_in-1") == block_1);

    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);
    const std::array<float, 6> embedding_values = { 1, 2, 3, 4, 5, 6 };
    const std::array<float, 6> layer_values = { -1, -2, -3, -4, -5, -6 };
    ggml_backend_tensor_set(embedding_input, embedding_values.data(), 0, sizeof(embedding_values));
    ggml_backend_tensor_set(layer_input, layer_values.data(), 0, sizeof(layer_values));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::array<float, 6> block_0_values = {};
    std::array<float, 6> block_1_values = {};
    ggml_backend_tensor_get(block_0, block_0_values.data(), 0, sizeof(block_0_values));
    ggml_backend_tensor_get(block_1, block_1_values.data(), 0, sizeof(block_1_values));
    t.assert_true("embedding alias represents identical values", block_0_values == embedding_values);
    t.assert_true("layer output alias represents identical values", block_1_values == layer_values);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    testing t(std::cout);
    t.test("longcat router", test_router);
    t.test("longcat production route helper", test_production_route_helper);
    t.test("longcat odd FFN association", test_odd_ffn_association);
    t.test("longcat capture alias name preservation", test_capture_alias_preserves_source_name);
    return t.summary();
}
