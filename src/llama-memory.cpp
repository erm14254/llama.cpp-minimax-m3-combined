#include "llama-memory.h"

bool llama_memory_params_resolve(
        llm_arch arch, llama_memory_params & params, bool & promoted, std::string & error) {
    promoted = false;
    error.clear();

    if (arch != LLM_ARCH_LONGCAT_NEXT) {
        return true;
    }

    switch (params.type_k) {
        case GGML_TYPE_F16:
            params.type_k = GGML_TYPE_BF16;
            promoted = true;
            break;
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            break;
        default:
            error = std::string("unsupported LongCat-Next K cache type ") + ggml_type_name(params.type_k) +
                    "; supported types are F16 (promoted to BF16), BF16, and F32";
            return false;
    }

    // Absorbed MLA does not currently store an independent V cache, but keep
    // both requested memory types consistent for future graph changes.
    params.type_v = params.type_k;
    return true;
}

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
