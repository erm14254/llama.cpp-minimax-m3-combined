#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

MODEL_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)

TOKEN_DIR = ROOT / "sparse_512_fa_off"

EXPECTED_TOKEN_SHA = (
    "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
)

EXPECTED_NTOK = 512
EXPECTED_VOCAB = 131072
EXPECTED_NEIGHBOR = 4
EXPECTED_SPLIT = 4
EXPECTED_RATIO = 78
EXPECTED_EOS = 2


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def config_int(config: dict, *names: str) -> int:
    for name in names:
        if name in config and config[name] is not None:
            value = config[name]

            if isinstance(value, list):
                if len(value) != 1:
                    stop(
                        "%s is unexpectedly a multi-value list: %r"
                        % (name, value)
                    )
                value = value[0]

            return int(value)

    stop("missing config key; tried: " + ", ".join(names))


def hf_shift_right_ignore_eos(
    tokens: np.ndarray,
    shift: int,
    eos_token_id: int,
) -> np.ndarray:
    # Exact scalar equivalent of frozen v4
    # NgramEmbedding._shift_right_ignore_eos().
    seq_len = int(tokens.size)
    result = np.zeros(seq_len, dtype=np.int64)

    eos_positions = np.flatnonzero(tokens == eos_token_id)
    prev_idx = 0

    for eos_idx_np in eos_positions:
        eos_idx = int(eos_idx_np)
        end_idx = eos_idx + 1

        if end_idx - prev_idx > shift:
            result[prev_idx + shift:end_idx] = (
                tokens[prev_idx:end_idx - shift]
            )

        prev_idx = end_idx

    if prev_idx < seq_len and seq_len - prev_idx > shift:
        result[prev_idx + shift:seq_len] = (
            tokens[prev_idx:seq_len - shift]
        )

    return result


def cpp_shifted_token_at(
    tokens: np.ndarray,
    pos: int,
    shift: int,
    eos_token_id: int,
) -> int:
    # Exact single-sequence, position-0 prompt equivalent of
    # llm_graph_input_ngram::shifted_token_at().
    prev_pos = pos - shift

    if prev_pos < 0:
        return 0

    for p in range(prev_pos, pos):
        if int(tokens[p]) == eos_token_id:
            return 0

    return int(tokens[prev_pos])


def main() -> int:
    candidates = sorted(TOKEN_DIR.glob("*-tokens.bin"))

    if len(candidates) != 1:
        stop(
            "expected exactly one *-tokens.bin in %s, got %d"
            % (TOKEN_DIR, len(candidates))
        )

    token_path = candidates[0]

    token_sha = sha256_file(token_path)

    print("token file   =", token_path)
    print("token SHA256 =", token_sha)

    if token_sha != EXPECTED_TOKEN_SHA:
        stop(
            "authoritative token SHA mismatch: %s != %s"
            % (token_sha, EXPECTED_TOKEN_SHA)
        )

    if token_path.stat().st_size != EXPECTED_NTOK * 4:
        stop(
            "token file byte count is %d, expected %d"
            % (
                token_path.stat().st_size,
                EXPECTED_NTOK * 4,
            )
        )

    tokens = np.fromfile(token_path, dtype="<i4").astype(
        np.int64,
        copy=False,
    )

    if tokens.size != EXPECTED_NTOK:
        stop(
            "token count is %d, expected %d"
            % (tokens.size, EXPECTED_NTOK)
        )

    config_path = MODEL_DIR / "config.json"

    if not config_path.is_file():
        stop("missing config.json: %s" % config_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))

    vocab_size = config_int(config, "vocab_size")
    n_neighbor = config_int(
        config,
        "emb_neighbor_num",
        "oe_neighbor_num",
    )
    n_split = config_int(
        config,
        "emb_split_num",
        "oe_split_num",
    )
    ratio = config_int(config, "ngram_vocab_size_ratio")
    eos_token_id = config_int(config, "eos_token_id")

    expected_config = (
        (vocab_size, EXPECTED_VOCAB, "vocab_size"),
        (n_neighbor, EXPECTED_NEIGHBOR, "neighbor_num"),
        (n_split, EXPECTED_SPLIT, "split_num"),
        (ratio, EXPECTED_RATIO, "ngram_vocab_size_ratio"),
        (eos_token_id, EXPECTED_EOS, "eos_token_id"),
    )

    for actual, expected, label in expected_config:
        if actual != expected:
            stop(
                "%s = %d, expected %d"
                % (label, actual, expected)
            )

    m = ratio * vocab_size
    n_embedders = (n_neighbor - 1) * n_split

    if n_embedders != 12:
        stop(
            "n_embedders = %d, expected 12"
            % n_embedders
        )

    print()
    print("===== TOKEN STREAM =====")
    print("token_count =", int(tokens.size))
    print("unique_tokens =", int(np.unique(tokens).size))
    print("eos_count =", int(np.count_nonzero(tokens == eos_token_id)))
    print("first_8 =", tokens[:8].tolist())
    print("last_8  =", tokens[-8:].tolist())

    # HF-style shifted vectors.
    hf_shifted = {
        shift: hf_shift_right_ignore_eos(
            tokens,
            shift,
            eos_token_id,
        )
        for shift in range(1, n_neighbor)
    }

    print()
    print("===== HASH-ID PARITY =====")
    print(
        "{:<5} {:<5} {:<5} {:>11} {:>11} {:>11} {:>10}".format(
            "idx",
            "ng",
            "split",
            "vocab_dim",
            "HF_final",
            "CPP_final",
            "mismatch",
        )
    )
    print("-" * 70)

    total_mismatch = 0
    final_ids = []

    for ng in range(2, n_neighbor + 1):
        for split in range(n_split):
            index = (ng - 2) * n_split + split
            emb_vocab_dim = m + index * 2 + 1

            power_mods = []
            power_mod = 1

            for _ in range(ng - 1):
                power_mod = (
                    power_mod * vocab_size
                ) % emb_vocab_dim
                power_mods.append(power_mod)

            # Frozen HF v4 scalar/vector equivalent.
            hf_hash = tokens.copy()

            for p in range(ng - 1):
                hf_hash = (
                    hf_hash
                    + hf_shifted[p + 1] * power_mods[p]
                )

            hf_ids = hf_hash % emb_vocab_dim

            # C++ set_input() scalar equivalent for this single
            # 0..511 prompt.
            cpp_ids = np.empty(
                EXPECTED_NTOK,
                dtype=np.int64,
            )

            for pos in range(EXPECTED_NTOK):
                value = int(tokens[pos])

                for p in range(ng - 1):
                    prev = cpp_shifted_token_at(
                        tokens,
                        pos,
                        p + 1,
                        eos_token_id,
                    )
                    value += prev * power_mods[p]

                cpp_ids[pos] = value % emb_vocab_dim

            mismatch = int(np.count_nonzero(hf_ids != cpp_ids))
            total_mismatch += mismatch

            hf_final = int(hf_ids[-1])
            cpp_final = int(cpp_ids[-1])

            final_ids.append(hf_final)

            print(
                "{:<5d} {:<5d} {:<5d} {:>11d} {:>11d} {:>11d} {:>10d}".format(
                    index,
                    ng,
                    split,
                    emb_vocab_dim,
                    hf_final,
                    cpp_final,
                    mismatch,
                )
            )

            if mismatch:
                where = np.flatnonzero(hf_ids != cpp_ids)
                first = int(where[0])

                stop(
                    "hash mismatch at embedder %d, first position %d: "
                    "HF=%d C++=%d"
                    % (
                        index,
                        first,
                        int(hf_ids[first]),
                        int(cpp_ids[first]),
                    )
                )

    if total_mismatch != 0:
        stop(
            "unexpected total hash mismatches: %d"
            % total_mismatch
        )

    print()
    print("final_ids =", final_ids)
    print("all_positions_compared =", 12 * EXPECTED_NTOK)
    print("total_mismatch =", total_mismatch)
    print()
    print("N-GRAM HASH-ID PARITY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())