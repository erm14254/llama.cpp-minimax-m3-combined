#include "testing.h"

#include <cstdint>
#include <string>
#include <vector>

static constexpr int64_t LONGCAT_VOCAB = 131072;
static constexpr int64_t LONGCAT_M = 78 * LONGCAT_VOCAB;
static constexpr int32_t LONGCAT_SPLIT = 4;
static constexpr int32_t LONGCAT_EOS = 2;

static int64_t mod_pow_step(int64_t mod, int shift) {
    int64_t v = 1;
    for (int i = 0; i < shift; ++i) {
        v = (v * LONGCAT_VOCAB) % mod;
    }
    return v;
}

static std::vector<int32_t> hf_shift_right_ignore_eos(const std::vector<int32_t> & tokens, int shift) {
    std::vector<int32_t> result(tokens.size(), 0);
    size_t prev = 0;

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] != LONGCAT_EOS) {
            continue;
        }
        const size_t end = i + 1;
        if (end - prev > (size_t) shift) {
            for (size_t p = prev + shift; p < end; ++p) {
                result[p] = tokens[p - shift];
            }
        }
        prev = end;
    }

    if (prev < tokens.size() && tokens.size() - prev > (size_t) shift) {
        for (size_t p = prev + shift; p < tokens.size(); ++p) {
            result[p] = tokens[p - shift];
        }
    }

    return result;
}

static int32_t hf_ngram_id(const std::vector<int32_t> & tokens, int pos, int ngram, int split) {
    const int64_t emb_vocab = LONGCAT_M + ((ngram - 2) * LONGCAT_SPLIT + split) * 2 + 1;
    int64_t hash = tokens[pos];
    for (int shift = 1; shift < ngram; ++shift) {
        const auto shifted = hf_shift_right_ignore_eos(tokens, shift);
        hash += (int64_t) shifted[pos] * mod_pow_step(emb_vocab, shift);
    }
    return (int32_t) (hash % emb_vocab);
}

static void test_ngram(testing & t) {
    struct expected_case {
        std::string name;
        std::vector<int32_t> tokens;
        int pos;
        int32_t ng2;
        int32_t ng3;
        int32_t ng4;
    };

    const std::vector<expected_case> cases = {
        { "no eos",                 {10, 11, 12, 13},       3, 1572877, 6649401, 5024532 },
        { "eos token",              {10, 11,  2, 14},       2, 1441794, 2339134, 2204702 },
        { "first after eos",        {10, 11,  2, 14},       3,      14,      14,      14 },
        { "second after eos",       {10,  2, 12, 13},       3, 1572877, 1572877, 1572877 },
        { "third after eos",        { 2, 10, 11, 12},       3, 1441804, 2339144, 2204712 },
        { "multiple eos turns",     { 3,  2,  4,  5, 2, 6, 7}, 6, 786439, 786439, 786439 },
        { "rewind replacement",     {10, 11, 99},           2, 1441891, 2339231, 2204799 },
        { "independent sequence a", {10,  2, 20},           2,      20,      20,      20 },
        { "independent sequence b", {30, 31, 32},           2, 4063264, 6755284, 6351988 },
    };

    for (const auto & tc : cases) {
        t.assert_equal(tc.name + " ng2", tc.ng2, hf_ngram_id(tc.tokens, tc.pos, 2, 0));
        t.assert_equal(tc.name + " ng3", tc.ng3, hf_ngram_id(tc.tokens, tc.pos, 3, 0));
        t.assert_equal(tc.name + " ng4", tc.ng4, hf_ngram_id(tc.tokens, tc.pos, 4, 0));
    }

    const auto shifted = hf_shift_right_ignore_eos({10, 11, 2, 14}, 1);
    t.assert_equal("eos token sees previous token", 11, shifted[2]);
    t.assert_equal("token after eos sees boundary", 0, shifted[3]);
}

int main() {
    testing t(std::cout);
    t.test("longcat ngram", test_ngram);
    return t.summary();
}
