#include "llama-memory-longcat.h"

#include "llama-io.h"

#include <algorithm>

static bool in_range(llama_pos pos, llama_pos p0, llama_pos p1) {
    return (p0 < 0 || pos >= p0) && (p1 < 0 || pos < p1);
}

llama_memory_context_ptr llama_memory_longcat::init_batch(llama_batch_allocr & b, uint32_t n, bool e) {
    return base->init_batch(b, n, e);
}

llama_memory_context_ptr llama_memory_longcat::init_full() {
    return base->init_full();
}

llama_memory_context_ptr llama_memory_longcat::init_update(llama_context * c, bool o) {
    return base->init_update(c, o);
}

bool llama_memory_longcat::get_can_shift() const {
    return base->get_can_shift();
}

llama_pos llama_memory_longcat::seq_pos_min(llama_seq_id s) const {
    return base->seq_pos_min(s);
}

llama_pos llama_memory_longcat::seq_pos_max(llama_seq_id s) const {
    return base->seq_pos_max(s);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_longcat::memory_breakdown() const {
    return base->memory_breakdown();
}

void llama_memory_longcat::clear(bool data) {
    base->clear(data);
    history.clear();
}

bool llama_memory_longcat::seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1) {
    if (!base->seq_rm(s, p0, p1)) {
        return false;
    }
    for (auto it = history.begin(); it != history.end();) {
        if (s < 0 || it->first == s) {
            auto & h = it->second;
            h.erase(std::remove_if(h.begin(), h.end(), [&](const auto & v) { return in_range(v.first, p0, p1); }),
                    h.end());
            if (h.empty()) {
                it = history.erase(it);
                continue;
            }
        }
        ++it;
    }
    return true;
}

void llama_memory_longcat::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    base->seq_cp(src, dst, p0, p1);
    const auto source = history[src];
    auto &     out    = history[dst];
    out.erase(std::remove_if(out.begin(), out.end(), [&](const auto & value) { return in_range(value.first, p0, p1); }),
              out.end());
    for (const auto & value : source) {
        if (in_range(value.first, p0, p1)) {
            out.push_back(value);
        }
    }
    std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
}

void llama_memory_longcat::seq_keep(llama_seq_id s) {
    base->seq_keep(s);
    auto                        found = history.find(s);
    llama_longcat_token_history kept;
    if (found != history.end()) {
        kept.emplace(s, found->second);
    }
    history.swap(kept);
}

void llama_memory_longcat::seq_add(llama_seq_id s, llama_pos p0, llama_pos p1, llama_pos shift) {
    base->seq_add(s, p0, p1, shift);
    for (auto & [id, values] : history) {
        if (s < 0 || id == s) {
            for (auto & value : values) {
                if (in_range(value.first, p0, p1)) {
                    value.first += shift;
                }
            }
            std::sort(values.begin(), values.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        }
    }
}

void llama_memory_longcat::seq_div(llama_seq_id s, llama_pos p0, llama_pos p1, int d) {
    base->seq_div(s, p0, p1, d);
    GGML_ASSERT(d != 0);
    for (auto & [id, values] : history) {
        if (s < 0 || id == s) {
            for (auto & value : values) {
                if (in_range(value.first, p0, p1)) {
                    value.first /= d;
                }
            }
            std::sort(values.begin(), values.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        }
    }
}

void llama_memory_longcat::state_write(llama_io_write_i & io, llama_seq_id s, llama_state_seq_flags flags) const {
    base->state_write(io, s, flags);
    uint32_t n_seq = s < 0 ? history.size() : history.count(s);
    io.write(&n_seq, sizeof(n_seq));
    for (const auto & [id, values] : history) {
        if (s >= 0 && id != s) {
            continue;
        }
        uint32_t n = values.size();
        io.write(&id, sizeof(id));
        io.write(&n, sizeof(n));
        for (const auto & [pos, tok] : values) {
            io.write(&pos, sizeof(pos));
            io.write(&tok, sizeof(tok));
        }
    }
}

void llama_memory_longcat::state_read(llama_io_read_i & io, llama_seq_id s, llama_state_seq_flags flags) {
    base->state_read(io, s, flags);
    if (s < 0) {
        history.clear();
    } else {
        history.erase(s);
    }
    uint32_t n_seq = 0;
    io.read(&n_seq, sizeof(n_seq));
    for (uint32_t i = 0; i < n_seq; ++i) {
        llama_seq_id id;
        uint32_t     n;
        io.read(&id, sizeof(id));
        io.read(&n, sizeof(n));
        auto & values = history[s < 0 ? id : s];
        for (uint32_t j = 0; j < n; ++j) {
            llama_pos   pos;
            llama_token tok;
            io.read(&pos, sizeof(pos));
            io.read(&tok, sizeof(tok));
            values.emplace_back(pos, tok);
        }
    }
}
