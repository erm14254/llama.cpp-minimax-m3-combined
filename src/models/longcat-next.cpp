#include "../llama-graph.h"
#include "../llama-model.h"
#include "models.h"

void llama_model_longcat_next::load_arch_hparams(llama_model_loader & ml) {
    llama_model_longcat_flash_ngram::load_arch_hparams(ml);

    ml.get_key(LLM_KV_NGRAM_HASH_VOCAB_SIZE, hparams.ngram_hash_vocab_size);
    ml.get_key(LLM_KV_NGRAM_INPUT_OUTPUT_SIZE, hparams.ngram_input_output_size);
    ml.get_key(LLM_KV_NGRAM_SOURCE_VOCAB_SIZE, hparams.ngram_source_vocab_size);
    ml.get_key(LLM_KV_NGRAM_IGNORED_START, hparams.ngram_ignored_start);
    ml.get_key(LLM_KV_NGRAM_IGNORED_COUNT, hparams.ngram_ignored_count);

    GGML_ASSERT(hparams.n_layer_nextn == 0 && "LongCat-Next must not advertise MTP");
    GGML_ASSERT(hparams.ngram_hash_vocab_size == 131072);
    GGML_ASSERT(hparams.ngram_input_output_size == 131125);
    GGML_ASSERT(hparams.ngram_source_vocab_size == 282624);
    GGML_ASSERT(hparams.ngram_ignored_start == 131072);
    GGML_ASSERT(hparams.ngram_ignored_count == 53);
}

void llama_model_longcat_next::load_arch_tensors(llama_model_loader & ml) {
    llama_model_longcat_flash_ngram::load_arch_tensors(ml);
    GGML_ASSERT(tok_embd && tok_embd->ne[1] == 131125);
    GGML_ASSERT(output && output->ne[1] == 131125);
    for (int i = 0; i < 12; ++i) {
        GGML_ASSERT(ngram_embd[i] && ngram_proj[i]);
    }
}

std::unique_ptr<llm_graph_context> llama_model_longcat_next::build_arch_graph(const llm_graph_params & params) const {
    GGML_ASSERT(params.gtype == LLM_GRAPH_TYPE_DEFAULT && "LongCat-Next has no MTP graph");
    return std::make_unique<llama_model_longcat_flash_ngram::graph>(*this, params);
}
