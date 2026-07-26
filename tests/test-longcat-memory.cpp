#include "llama-io.h"
#include "llama-memory-longcat.h"
#include "llama-batch.h"
#include "testing.h"

#include <cstring>
#include <iostream>
#include <vector>

class fake_context final : public llama_memory_context_i {
public:
    explicit fake_context(bool succeed) : succeed(succeed) {}
    bool next() override { return false; }
    bool apply() override { ++apply_count; return succeed; }
    const llama_ubatch & get_ubatch() const override { return ubatch; }
    llama_memory_status get_status() const override { return LLAMA_MEMORY_STATUS_SUCCESS; }
    bool succeed;
    int apply_count = 0;
    llama_ubatch ubatch = {};
};

class fake_memory final : public llama_memory_i {
  public:
    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return nullptr; }

    llama_memory_context_ptr init_full() override { return nullptr; }

    llama_memory_context_ptr init_update(llama_context *, bool) override { return nullptr; }

    bool get_can_shift() const override { return true; }

    void clear(bool) override {}

    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { return true; }

    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}

    void seq_keep(llama_seq_id) override {}

    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override {}

    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override {}

    llama_pos seq_pos_min(llama_seq_id) const override { return -1; }

    llama_pos seq_pos_max(llama_seq_id) const override { return -1; }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i & io, llama_seq_id, llama_state_seq_flags) const override {
        uint32_t marker = 0x4c434e58;
        io.write(&marker, sizeof(marker));
    }

    void state_read(llama_io_read_i & io, llama_seq_id, llama_state_seq_flags) override {
        uint32_t marker = 0;
        io.read(&marker, sizeof(marker));
        GGML_ASSERT(marker == 0x4c434e58);
    }
};

class buffer_io final : public llama_io_write_i,
                        public llama_io_read_i {
  public:
    void write(const void * src, size_t size) override {
        const auto * ptr = static_cast<const uint8_t *>(src);
        data.insert(data.end(), ptr, ptr + size);
    }

    void read(void * dst, size_t size) override {
        GGML_ASSERT(offset + size <= data.size());
        memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void write_tensor(ggml_tensor *, size_t, size_t) override { GGML_ABORT("unused"); }

    void read_tensor(ggml_tensor *, size_t, size_t) override { GGML_ABORT("unused"); }

    size_t n_bytes() override { return offset; }

    std::vector<uint8_t> data;
    size_t               offset = 0;
};

static void test_lifecycle(testing & t) {
    llama_memory_longcat memory(new fake_memory());
    memory.history[0] = {
        { 0, 10 },
        { 1, 11 },
        { 2, 12 },
        { 3, 13 }
    };
    memory.history[1] = {
        { 0, 20 }
    };

    memory.seq_cp(0, 2, 1, 4);
    t.assert_equal("copy range", (size_t) 3, memory.history[2].size());
    memory.seq_rm(2, 2, 3);
    t.assert_equal("remove range", (size_t) 2, memory.history[2].size());
    memory.seq_add(2, -1, -1, 5);
    t.assert_equal("positive shift", (llama_pos) 6, memory.history[2].front().first);
    memory.seq_add(2, -1, -1, -2);
    t.assert_equal("negative shift", (llama_pos) 4, memory.history[2].front().first);
    memory.seq_div(2, -1, -1, 2);
    t.assert_equal("position division", (llama_pos) 2, memory.history[2].front().first);
    memory.seq_keep(2);
    t.assert_equal("keep one sequence", (size_t) 1, memory.history.size());

    buffer_io io;
    memory.state_write(io, -1, 0);
    llama_memory_longcat restored(new fake_memory());
    restored.state_read(io, -1, 0);
    t.assert_equal("state restore sequence count", (size_t) 1, restored.history.size());
    t.assert_equal("state restore token count", memory.history[2].size(), restored.history[2].size());
    restored.clear(false);
    t.assert_equal("clear history", (size_t) 0, restored.history.size());
}

static void test_atomic_context(testing & t) {
    llama_memory_longcat memory(new fake_memory());
    memory.history[0] = { { 0, 10 } };
    {
        auto raw = std::make_unique<fake_context>(false);
        llama_memory_longcat_context context(memory, std::move(raw));
        context.pending_history()[0].emplace_back(1, 11);
        t.assert_true("failed underlying apply", !context.apply());
        t.assert_equal("failure leaves committed history", (size_t) 1, memory.history[0].size());
    }
    {
        auto raw = std::make_unique<fake_context>(true);
        const auto * raw_ptr = raw.get();
        llama_memory_longcat_context context(memory, std::move(raw));
        const auto & const_context = context;
        t.assert_true("wrapper exposes exact underlying context", const_context.base_context() == raw_ptr);
        t.assert_equal("retry starts from committed history", (size_t) 1, context.pending_history()[0].size());
        context.pending_history()[0].emplace_back(1, 12);
        t.assert_equal("pending history remains independent", (size_t) 1, memory.history[0].size());
        t.assert_true("successful underlying apply", context.apply());
        t.assert_equal("apply alone is not compute commit", (size_t) 1, memory.history[0].size());
        context.commit();
        t.assert_equal("successful compute commits pending", (size_t) 2, memory.history[0].size());
    }
    {
        auto raw = std::make_unique<fake_context>(true);
        llama_memory_longcat_context cancelled(memory, std::move(raw));
        cancelled.pending_history()[0].emplace_back(2, 13);
        t.assert_true("cancelled apply prepared", cancelled.apply());
    }
    t.assert_equal("destruction without compute commit discards", (size_t) 2, memory.history[0].size());

    buffer_io partial;
    memory.state_write(partial, -1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    t.assert_equal("partial-only excludes longcat history", sizeof(uint32_t), partial.data.size());
}

int main() {
    testing t(std::cout);
    t.test("longcat sequence memory lifecycle", test_lifecycle);
    t.test("longcat atomic memory context", test_atomic_context);
    return t.summary();
}
