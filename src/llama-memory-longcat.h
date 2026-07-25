#pragma once

#include "llama-longcat-history.h"
#include "llama-memory.h"

class llama_memory_longcat final : public llama_memory_i {
  public:
    explicit llama_memory_longcat(llama_memory_i * base) : base(base) {}

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool                     get_can_shift() const override;
    void                     clear(bool data) override;
    bool                     seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void                     seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) override;
    void                     seq_keep(llama_seq_id seq_id) override;
    void                     seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void                     seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos                seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos                seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) override;

    llama_longcat_token_history history;

  private:
    llama_memory_ptr base;
};
