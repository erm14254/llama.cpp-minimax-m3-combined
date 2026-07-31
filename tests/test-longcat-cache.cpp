#include "llama-memory.h"
#include "testing.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <iostream>
#include <string>

static llama_memory_params memory_params(ggml_type type_k, ggml_type type_v) {
    return { type_k, type_v, false, LLAMA_CONTEXT_TYPE_DEFAULT, nullptr };
}

static void test_cache_policy(testing & t) {
    bool promoted = false;
    std::string error;

    auto other = memory_params(GGML_TYPE_F16, GGML_TYPE_F32);
    t.assert_true("non-LongCat accepted",
        llama_memory_params_resolve(LLM_ARCH_LLAMA, other, promoted, error));
    t.assert_true("non-LongCat K unchanged", other.type_k == GGML_TYPE_F16);
    t.assert_true("non-LongCat V unchanged", other.type_v == GGML_TYPE_F32);
    t.assert_true("non-LongCat not promoted", !promoted);

    auto f16 = memory_params(GGML_TYPE_F16, GGML_TYPE_F16);
    t.assert_true("LongCat F16 accepted",
        llama_memory_params_resolve(LLM_ARCH_LONGCAT_NEXT, f16, promoted, error));
    t.assert_true("LongCat F16 promoted", promoted);
    t.assert_true("LongCat promoted K is BF16", f16.type_k == GGML_TYPE_BF16);
    t.assert_true("LongCat promoted V is BF16", f16.type_v == GGML_TYPE_BF16);
    t.assert_equal("F16 and BF16 cache element widths match",
        ggml_type_size(GGML_TYPE_F16), ggml_type_size(GGML_TYPE_BF16));

    auto bf16 = memory_params(GGML_TYPE_BF16, GGML_TYPE_F16);
    t.assert_true("LongCat BF16 accepted",
        llama_memory_params_resolve(LLM_ARCH_LONGCAT_NEXT, bf16, promoted, error));
    t.assert_true("LongCat BF16 preserved", bf16.type_k == GGML_TYPE_BF16 && !promoted);
    t.assert_true("LongCat BF16 V consistent", bf16.type_v == GGML_TYPE_BF16);

    auto f32 = memory_params(GGML_TYPE_F32, GGML_TYPE_F16);
    t.assert_true("LongCat F32 accepted",
        llama_memory_params_resolve(LLM_ARCH_LONGCAT_NEXT, f32, promoted, error));
    t.assert_true("LongCat F32 preserved", f32.type_k == GGML_TYPE_F32 && !promoted);
    t.assert_true("LongCat F32 V consistent", f32.type_v == GGML_TYPE_F32);

    auto unsupported = memory_params(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
    t.assert_true("LongCat quantized cache rejected",
        !llama_memory_params_resolve(LLM_ARCH_LONGCAT_NEXT, unsupported, promoted, error));
    t.assert_true("LongCat rejection names unsupported type", error.find("q8_0") != std::string::npos);
    t.assert_true("LongCat rejection explains supported types", error.find("supported types") != std::string::npos);

    const auto defaults_before = llama_context_default_params();
    auto resolved_default = memory_params(defaults_before.type_k, defaults_before.type_v);
    t.assert_true("LongCat global default request resolves",
        llama_memory_params_resolve(LLM_ARCH_LONGCAT_NEXT, resolved_default, promoted, error));
    const auto defaults_after = llama_context_default_params();
    t.assert_true("global K default unchanged", defaults_after.type_k == defaults_before.type_k);
    t.assert_true("global V default unchanged", defaults_after.type_v == defaults_before.type_v);
}

static float run_large_query_mul_mat(ggml_type left_type) {
    ggml_init_params init = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(init);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * left = ggml_new_tensor_2d(ctx, left_type, 2, 1);
    ggml_tensor * query = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor * result = ggml_mul_mat(ctx, left, query);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    const float left_f32[2] = { 1.0f, 1.0f };
    const float query_f32[2] = { 80000.0f, 1.0f };
    if (left_type == GGML_TYPE_F16) {
        ggml_fp16_t data[2];
        ggml_cpu_fp32_to_fp16(left_f32, data, 2);
        ggml_backend_tensor_set(left, data, 0, sizeof(data));
    } else if (left_type == GGML_TYPE_BF16) {
        ggml_bf16_t data[2];
        ggml_cpu_fp32_to_bf16(left_f32, data, 2);
        ggml_backend_tensor_set(left, data, 0, sizeof(data));
    } else {
        ggml_backend_tensor_set(left, left_f32, 0, sizeof(left_f32));
    }
    ggml_backend_tensor_set(query, query_f32, 0, sizeof(query_f32));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    float output = 0.0f;
    ggml_backend_tensor_get(result, &output, 0, sizeof(output));

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return output;
}

static void test_operand_conversion(testing & t) {
    // GGML_PREC_F32 retains an F32 result/accumulator, but the left operand's
    // vec_dot_type still controls preparation of the F32 right operand.
    t.assert_true("F16 packing loses large finite query", !std::isfinite(run_large_query_mul_mat(GGML_TYPE_F16)));
    t.assert_true("BF16 packing retains large finite query", std::isfinite(run_large_query_mul_mat(GGML_TYPE_BF16)));
    t.assert_true("F32 operands retain large finite query", std::isfinite(run_large_query_mul_mat(GGML_TYPE_F32)));
}

int main() {
    testing t(std::cout);
    t.test("LongCat cache policy", test_cache_policy);
    t.test("CPU operand conversion range", test_operand_conversion);
    return t.summary();
}
