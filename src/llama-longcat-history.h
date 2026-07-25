#pragma once

#include "llama.h"

#include <deque>
#include <map>
#include <utility>

using llama_longcat_token_history = std::map<llama_seq_id, std::deque<std::pair<llama_pos, llama_token>>>;
