#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)

GGUF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16"
)

LLAMA_REPO = Path(
    r"D:\llama.cpp-longcat-pre-gate4"
)

TOKEN_ID = 483

FINAL_IDS = [
    1966557,
    1966545,
    1966533,
    1966521,
    6458256,
    4834970,
    3211684,
    1588398,
    7348131,
    4074420,
    883953,
    8000369,
]

EXPECTED_VOCAB_DIMS = [
    10223617,
    10223619,
    10223621,
    10223623,
    10223625,
    10223627,
    10223629,
    10223631,
    10223633,
    10223635,
    10223637,
    10223639,
]


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def torch_bf16_raw(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype != torch.bfloat16:
        stop("HF tensor is not BF16: %s" % tensor.dtype)

    raw = tensor.view(torch.uint16).numpy()

    # Safetensors and the GGUF being audited are little-endian.
    raw = raw.astype("<u2", copy=False)

    return raw.tobytes(order="C")


def load_hf_piece(
    weight_map: dict[str, str],
    source_name: str,
    row: int | None = None,
) -> tuple[bytes, tuple[int, ...], str]:
    if source_name not in weight_map:
        stop("HF source tensor absent from index: " + source_name)

    shard_name = weight_map[source_name]
    shard_path = HF_DIR / shard_name

    if not shard_path.is_file():
        stop("HF shard missing: %s" % shard_path)

    with safe_open(
        shard_path,
        framework="pt",
        device="cpu",
    ) as handle:
        if source_name not in handle.keys():
            stop(
                "%s is indexed into %s but absent"
                % (source_name, shard_name)
            )

        tensor_slice = handle.get_slice(source_name)
        shape = tuple(int(v) for v in tensor_slice.get_shape())

        if row is None:
            tensor = handle.get_tensor(source_name)
        else:
            if row < 0 or row >= shape[0]:
                stop(
                    "HF row %d outside %s shape %s"
                    % (row, source_name, shape)
                )

            tensor = tensor_slice[row:row + 1]

    return torch_bf16_raw(tensor), shape, shard_name


def main() -> int:
    index_path = HF_DIR / "model.safetensors.index.json"

    if not index_path.is_file():
        stop("missing HF Safetensors index")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict):
        stop("HF Safetensors index has no weight_map")

    gguf_py = LLAMA_REPO / "gguf-py"

    if not gguf_py.is_dir():
        stop("missing gguf-py directory")

    sys.path.insert(0, str(gguf_py))

    try:
        from gguf import GGUFReader
    except Exception as exc:
        stop("GGUFReader import failed: %s" % exc)

    target_names = {"token_embd.weight"}

    target_names.update(
        "ngram_embd.%d.weight" % i
        for i in range(12)
    )

    target_names.update(
        "ngram_proj.%d.weight" % i
        for i in range(12)
    )

    readers = []
    gguf_tensors = {}

    shards = sorted(GGUF_DIR.glob("*.gguf"))

    if not shards:
        stop("no GGUF shards found")

    for shard_path in shards:
        reader = GGUFReader(str(shard_path), mode="r")
        readers.append(reader)

        for tensor in reader.tensors:
            if tensor.name not in target_names:
                continue

            if tensor.name in gguf_tensors:
                stop(
                    "duplicate GGUF tensor: %s"
                    % tensor.name
                )

            gguf_tensors[tensor.name] = (
                shard_path.name,
                tensor,
            )

    missing = sorted(target_names - set(gguf_tensors))

    if missing:
        stop(
            "missing GGUF relevant tensors: "
            + ", ".join(missing)
        )

    if len(gguf_tensors) != 25:
        stop(
            "expected 25 GGUF relevant tensors, got %d"
            % len(gguf_tensors)
        )

    def check_bf16(name: str):
        shard_name, tensor = gguf_tensors[name]

        if tensor.tensor_type.name != "BF16":
            stop(
                "%s is %s, expected BF16"
                % (name, tensor.tensor_type.name)
            )

        if np.asarray(tensor.data).dtype != np.uint8:
            stop(
                "%s GGUF raw view dtype is %s, expected uint8"
                % (name, np.asarray(tensor.data).dtype)
            )

        return shard_name, tensor

    def gguf_row_raw(
        name: str,
        row: int,
        expected_metadata_shape: tuple[int, int],
        expected_row_elements: int,
    ) -> bytes:
        _, tensor = check_bf16(name)

        metadata_shape = tuple(
            int(v)
            for v in tensor.shape.tolist()
        )

        if metadata_shape != expected_metadata_shape:
            stop(
                "%s metadata shape %s != %s"
                % (
                    name,
                    metadata_shape,
                    expected_metadata_shape,
                )
            )

        data = np.asarray(tensor.data)

        expected_numpy_shape = (
            expected_metadata_shape[1],
            expected_row_elements * 2,
        )

        if data.shape != expected_numpy_shape:
            stop(
                "%s raw shape %s != %s"
                % (
                    name,
                    data.shape,
                    expected_numpy_shape,
                )
            )

        if row < 0 or row >= data.shape[0]:
            stop(
                "%s row %d outside raw shape %s"
                % (name, row, data.shape)
            )

        return np.ascontiguousarray(
            data[row]
        ).tobytes(order="C")

    def gguf_full_raw(
        name: str,
        expected_metadata_shape: tuple[int, int],
        expected_hf_shape: tuple[int, int],
    ) -> bytes:
        _, tensor = check_bf16(name)

        metadata_shape = tuple(
            int(v)
            for v in tensor.shape.tolist()
        )

        if metadata_shape != expected_metadata_shape:
            stop(
                "%s metadata shape %s != %s"
                % (
                    name,
                    metadata_shape,
                    expected_metadata_shape,
                )
            )

        data = np.asarray(tensor.data)

        expected_raw_shape = (
            expected_hf_shape[0],
            expected_hf_shape[1] * 2,
        )

        if data.shape != expected_raw_shape:
            stop(
                "%s raw shape %s != %s"
                % (
                    name,
                    data.shape,
                    expected_raw_shape,
                )
            )

        return np.ascontiguousarray(
            data
        ).tobytes(order="C")

    comparisons = []

    # Base token embedding row actually used by the final token.
    hf_raw, hf_shape, hf_shard = load_hf_piece(
        weight_map,
        "model.embed_tokens.weight",
        TOKEN_ID,
    )

    if hf_shape != (131072, 3072):
        stop(
            "unexpected HF token embedding shape: %s"
            % (hf_shape,)
        )

    gg_raw = gguf_row_raw(
        "token_embd.weight",
        TOKEN_ID,
        (3072, 131072),
        3072,
    )

    comparisons.append(
        (
            "token_embd[row=483]",
            hf_raw,
            gg_raw,
            hf_shard,
        )
    )

    # The 12 embedding rows selected for the final prompt token.
    for i, row_id in enumerate(FINAL_IDS):
        source_name = (
            "model.oe_embed_tokens%d.weight" % i
        )

        hf_raw, hf_shape, hf_shard = load_hf_piece(
            weight_map,
            source_name,
            row_id,
        )

        expected_hf_shape = (
            EXPECTED_VOCAB_DIMS[i],
            256,
        )

        if hf_shape != expected_hf_shape:
            stop(
                "%s shape %s != %s"
                % (
                    source_name,
                    hf_shape,
                    expected_hf_shape,
                )
            )

        gg_name = "ngram_embd.%d.weight" % i

        gg_raw = gguf_row_raw(
            gg_name,
            row_id,
            (256, EXPECTED_VOCAB_DIMS[i]),
            256,
        )

        comparisons.append(
            (
                "ngram_embd.%d[row=%d]"
                % (i, row_id),
                hf_raw,
                gg_raw,
                hf_shard,
            )
        )

    # All 12 projection matrices are small enough to compare completely.
    for i in range(12):
        source_name = (
            "model.oe_embed_proj%d.weight" % i
        )

        hf_raw, hf_shape, hf_shard = load_hf_piece(
            weight_map,
            source_name,
            None,
        )

        expected_hf_shape = (3072, 256)

        if hf_shape != expected_hf_shape:
            stop(
                "%s shape %s != %s"
                % (
                    source_name,
                    hf_shape,
                    expected_hf_shape,
                )
            )

        gg_name = "ngram_proj.%d.weight" % i

        gg_raw = gguf_full_raw(
            gg_name,
            (256, 3072),
            expected_hf_shape,
        )

        comparisons.append(
            (
                "ngram_proj.%d[full]" % i,
                hf_raw,
                gg_raw,
                hf_shard,
            )
        )

    if len(comparisons) != 25:
        stop(
            "expected 25 comparisons, got %d"
            % len(comparisons)
        )

    print(
        "{:<34} {:>10}  {}".format(
            "piece",
            "bytes",
            "sha256",
        )
    )
    print("-" * 115)

    all_exact = True

    for label, hf_raw, gg_raw, hf_shard in comparisons:
        if len(hf_raw) != len(gg_raw):
            stop(
                "%s byte-size mismatch: HF=%d GGUF=%d"
                % (
                    label,
                    len(hf_raw),
                    len(gg_raw),
                )
            )

        hf_sha = sha256_bytes(hf_raw)
        gg_sha = sha256_bytes(gg_raw)

        exact = hf_raw == gg_raw
        all_exact = all_exact and exact

        print(
            "{:<34} {:>10d}  {}  {}".format(
                label,
                len(hf_raw),
                hf_sha,
                "EXACT" if exact else "MISMATCH",
            )
        )

        if not exact:
            stop(
                "%s differs: HF sha=%s GGUF sha=%s HF shard=%s"
                % (
                    label,
                    hf_sha,
                    gg_sha,
                    hf_shard,
                )
            )

    print()
    print("comparison_count =", len(comparisons))
    print("all_raw_exact =", all_exact)

    if not all_exact:
        stop("one or more selected weights differ")

    print()
    print("STATIC N-GRAM WEIGHT SLICE AUDIT: EXACT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())