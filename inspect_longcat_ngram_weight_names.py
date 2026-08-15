#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


HF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)

GGUF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-283-Low-KL-GGUF-BF16"
)

LLAMA_REPO = Path(
    r"D:\llama.cpp-longcat-pre-gate4"
)


def stop(msg: str) -> None:
    raise SystemExit("STOP: " + msg)


def main() -> int:
    index_path = HF_DIR / "model.safetensors.index.json"

    if not index_path.is_file():
        stop("HF Safetensors index missing")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})

    print("===== HF RELEVANT TENSORS =====")

    hf_names = sorted(
        name
        for name in weight_map
        if (
            "ngram" in name.lower()
            or "embed_tokens.weight" in name
        )
    )

    for name in hf_names:
        print(name)

    print()
    print("HF relevant count =", len(hf_names))

    gguf_py = LLAMA_REPO / "gguf-py"
    if not gguf_py.is_dir():
        stop("gguf-py directory missing")

    sys.path.insert(0, str(gguf_py))

    try:
        from gguf import GGUFReader
    except Exception as exc:
        stop("GGUFReader import failed: %s" % exc)

    shards = sorted(GGUF_DIR.glob("*.gguf"))

    if not shards:
        stop("no GGUF shards found")

    found = {}

    for shard in shards:
        reader = GGUFReader(str(shard), mode="r")

        for tensor in reader.tensors:
            name = tensor.name

            if (
                "ngram" in name.lower()
                or name == "token_embd.weight"
            ):
                if name in found:
                    stop(
                        "duplicate GGUF tensor %s in %s and %s"
                        % (name, found[name][0], shard.name)
                    )

                found[name] = (
                    shard.name,
                    tensor.tensor_type.name,
                    tuple(int(x) for x in tensor.shape.tolist()),
                )

    print()
    print("===== GGUF RELEVANT TENSORS =====")

    for name in sorted(found):
        shard, qtype, shape = found[name]
        print(
            "%s | %s | %s | %s"
            % (name, qtype, shape, shard)
        )

    print()
    print("GGUF relevant count =", len(found))

    print()
    print("NGRAM WEIGHT-NAME INSPECTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())