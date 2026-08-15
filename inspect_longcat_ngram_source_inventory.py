#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


HF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def normalized_name(source_name: str) -> str | None:
    if source_name == "model.embed_tokens.weight":
        return "token_embd.weight"

    match = re.fullmatch(
        r"(?:model\.)?oe_embed_tokens(\d+)\.weight",
        source_name,
    )
    if match:
        return f"ngram_embd.{int(match.group(1))}.weight"

    match = re.fullmatch(
        r"(?:model\.)?oe_embed_proj(\d+)\.weight",
        source_name,
    )
    if match:
        return f"ngram_proj.{int(match.group(1))}.weight"

    match = re.fullmatch(
        r"model\.ngram_embeddings\.embedders\.(\d+)\.weight",
        source_name,
    )
    if match:
        return f"ngram_embd.{int(match.group(1))}.weight"

    match = re.fullmatch(
        r"model\.ngram_embeddings\.post_projs\.(\d+)\.weight",
        source_name,
    )
    if match:
        return f"ngram_proj.{int(match.group(1))}.weight"

    return None


def main() -> int:
    index_path = HF_DIR / "model.safetensors.index.json"

    if not index_path.is_file():
        stop(f"missing Safetensors index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict):
        stop("Safetensors index has no weight_map object")

    normalized: dict[str, tuple[str, str]] = {}

    for source_name, shard_name in weight_map.items():
        target = normalized_name(source_name)

        if target is None:
            continue

        if target in normalized:
            previous_source, previous_shard = normalized[target]
            stop(
                "conflicting source representations for "
                f"{target}: "
                f"{previous_source} [{previous_shard}] and "
                f"{source_name} [{shard_name}]"
            )

        normalized[target] = (source_name, shard_name)

    expected = {"token_embd.weight"}

    expected.update(
        f"ngram_embd.{index}.weight"
        for index in range(12)
    )

    expected.update(
        f"ngram_proj.{index}.weight"
        for index in range(12)
    )

    missing = sorted(expected - set(normalized))
    unexpected = sorted(set(normalized) - expected)

    if missing:
        stop("missing normalized source tensors: " + ", ".join(missing))

    if unexpected:
        stop(
            "unexpected normalized source tensors: "
            + ", ".join(unexpected)
        )

    if len(normalized) != 25:
        stop(
            f"normalized source inventory must contain 25 tensors, "
            f"got {len(normalized)}"
        )

    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for target, (source_name, shard_name) in normalized.items():
        by_shard[shard_name].append((target, source_name))

    details = {}

    for shard_name, rows in sorted(by_shard.items()):
        shard_path = HF_DIR / shard_name

        if not shard_path.is_file():
            stop(f"missing source shard: {shard_path}")

        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            available = set(handle.keys())

            for target, source_name in rows:
                if source_name not in available:
                    stop(
                        f"{source_name} is indexed into {shard_name} "
                        "but is absent from that shard"
                    )

                tensor_slice = handle.get_slice(source_name)

                shape = tuple(
                    int(value)
                    for value in tensor_slice.get_shape()
                )

                dtype = str(tensor_slice.get_dtype())

                details[target] = {
                    "source_name": source_name,
                    "shard": shard_name,
                    "dtype": dtype,
                    "shape": shape,
                }

    print("===== NORMALIZED HF N-GRAM INVENTORY =====")

    ordered = ["token_embd.weight"]

    ordered.extend(
        f"ngram_embd.{index}.weight"
        for index in range(12)
    )

    ordered.extend(
        f"ngram_proj.{index}.weight"
        for index in range(12)
    )

    for target in ordered:
        row = details[target]

        print(
            f"{target} <- {row['source_name']} "
            f"| {row['dtype']} "
            f"| {row['shape']} "
            f"| {row['shard']}"
        )

    print()
    print(f"normalized relevant count = {len(details)}")

    embedding_ids = sorted(
        int(name.split(".")[1])
        for name in details
        if name.startswith("ngram_embd.")
    )

    projection_ids = sorted(
        int(name.split(".")[1])
        for name in details
        if name.startswith("ngram_proj.")
    )

    if embedding_ids != list(range(12)):
        stop(
            "n-gram embedding IDs are not exactly 0..11: "
            f"{embedding_ids}"
        )

    if projection_ids != list(range(12)):
        stop(
            "n-gram projection IDs are not exactly 0..11: "
            f"{projection_ids}"
        )

    if any(
        row["dtype"] != "BF16"
        for row in details.values()
    ):
        bad = {
            name: row["dtype"]
            for name, row in details.items()
            if row["dtype"] != "BF16"
        }

        stop(f"relevant source tensors are not all BF16: {bad}")

    print()
    print("HF N-GRAM SOURCE INVENTORY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())