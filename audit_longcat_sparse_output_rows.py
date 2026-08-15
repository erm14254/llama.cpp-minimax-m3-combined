#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROW_IDS = [483, 15626, 15777, 25433, 39590, 112084, 122091]
VOCAB_SIZE = 131072


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def choose_hf_tensor(model_dir: Path, weight_map: dict[str, str]):
    config_path = model_dir / "config.json"
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    tied = bool(config.get("tie_word_embeddings", False))
    print(f"tie_word_embeddings={tied}")

    output_candidates = [
        "lm_head.weight",
        "model.lm_head.weight",
        "model.output.weight",
        "output.weight",
    ]

    present = [name for name in output_candidates if name in weight_map]

    if len(present) == 1:
        return present[0], tied

    if len(present) > 1:
        stop(f"ambiguous HF output tensors: {present}")

    if tied:
        tied_candidates = [
            "model.embed_tokens.weight",
            "model.tok_embeddings.weight",
            "embed_tokens.weight",
        ]
        present = [name for name in tied_candidates if name in weight_map]
        if len(present) == 1:
            print("HF output is tied to token embeddings")
            return present[0], tied
        if len(present) > 1:
            stop(f"ambiguous tied HF output tensors: {present}")

    nearby = sorted(
        name
        for name in weight_map
        if (
            "lm_head" in name
            or "embed_tokens" in name
            or "output" in name.lower()
        )
    )

    stop(
        "could not identify HF output tensor; nearby candidates="
        f"{nearby[:40]}"
    )


def read_hf_rows(model_dir: Path):
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:
        stop(f"failed to import torch/safetensors: {exc}")

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        stop(f"missing Safetensors index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict):
        stop("Safetensors index has no weight_map")

    tensor_name, tied = choose_hf_tensor(model_dir, weight_map)

    shard_name = weight_map[tensor_name]
    shard_path = model_dir / shard_name
    if not shard_path.is_file():
        stop(f"HF output shard missing: {shard_path}")

    print(f"hf_tensor={tensor_name}")
    print(f"hf_shard={shard_name}")

    rows: dict[int, np.ndarray] = {}
    raw_rows: dict[int, bytes] = {}
    raw_kind = None

    with safe_open(
        str(shard_path),
        framework="pt",
        device="cpu",
    ) as f:
        if tensor_name not in f.keys():
            stop(f"{tensor_name} absent from declared shard")

        sl = f.get_slice(tensor_name)
        shape = tuple(int(x) for x in sl.get_shape())

        print(f"hf_shape={shape}")

        if len(shape) != 2:
            stop(f"HF output tensor is not rank 2: {shape}")
        if shape[0] != VOCAB_SIZE:
            stop(
                f"HF output first dimension is {shape[0]}, "
                f"expected vocab {VOCAB_SIZE}"
            )

        hf_dtype = None

        for row_id in ROW_IDS:
            row = sl[row_id : row_id + 1].reshape(-1).contiguous()

            if hf_dtype is None:
                hf_dtype = row.dtype

            if row.dtype != hf_dtype:
                stop("HF sliced rows unexpectedly have differing dtypes")

            rows[row_id] = (
                row.to(torch.float32)
                .cpu()
                .numpy()
                .copy()
            )

            if row.dtype == torch.bfloat16:
                raw_kind = "BF16"
                raw_rows[row_id] = (
                    row.view(torch.int16)
                    .cpu()
                    .numpy()
                    .astype("<i2", copy=False)
                    .tobytes()
                )
            elif row.dtype == torch.float16:
                raw_kind = "F16"
                raw_rows[row_id] = (
                    row.view(torch.int16)
                    .cpu()
                    .numpy()
                    .astype("<i2", copy=False)
                    .tobytes()
                )
            elif row.dtype == torch.float32:
                raw_kind = "F32"
                raw_rows[row_id] = (
                    row.cpu()
                    .numpy()
                    .astype("<f4", copy=False)
                    .tobytes()
                )
            else:
                raw_kind = None

    print(f"hf_dtype={hf_dtype}")
    print(f"hf_raw_kind={raw_kind}")

    return {
        "name": tensor_name,
        "shape": shape,
        "rows": rows,
        "raw_rows": raw_rows,
        "raw_kind": raw_kind,
        "tied": tied,
    }


def find_gguf_tensor(gguf_dir: Path, llama_repo: Path, tied: bool):
    gguf_py = llama_repo / "gguf-py"
    if not gguf_py.is_dir():
        stop(f"missing gguf-py: {gguf_py}")

    sys.path.insert(0, str(gguf_py))

    try:
        from gguf import GGUFReader
    except Exception as exc:
        stop(f"failed to import local GGUFReader: {exc}")

    shards = sorted(gguf_dir.glob("*.gguf"))
    if not shards:
        stop(f"no GGUF shards found in {gguf_dir}")

    print(f"gguf_shard_count={len(shards)}")

    output_hits = []
    token_hits = []

    for shard in shards:
        reader = GGUFReader(str(shard), mode="r")

        for tensor in reader.tensors:
            if tensor.name == "output.weight":
                output_hits.append((shard, reader, tensor))
            elif tensor.name == "token_embd.weight":
                token_hits.append((shard, reader, tensor))

    if len(output_hits) == 1:
        return output_hits[0]

    if len(output_hits) > 1:
        stop(
            "output.weight appeared in multiple GGUF shards: "
            f"{[x[0].name for x in output_hits]}"
        )

    if tied and len(token_hits) == 1:
        print("GGUF output falls back to tied token_embd.weight")
        return token_hits[0]

    if tied and len(token_hits) > 1:
        stop(
            "token_embd.weight appeared in multiple GGUF shards: "
            f"{[x[0].name for x in token_hits]}"
        )

    stop(
        "could not locate GGUF output.weight"
        + (
            "; token_embd.weight hits="
            f"{[x[0].name for x in token_hits]}"
        )
    )


def decode_gguf_rows(hit, hf_shape):
    shard, reader, tensor = hit

    if getattr(reader, "byte_order", "I") != "I":
        stop("swapped-endian GGUF is not supported by this audit")

    dims = tuple(int(x) for x in tensor.shape.tolist())
    logical_shape = tuple(reversed(dims))
    qtype = tensor.tensor_type.name

    print(f"gguf_tensor={tensor.name}")
    print(f"gguf_shard={shard.name}")
    print(f"gguf_type={qtype}")
    print(f"gguf_dims_ne={dims}")
    print(f"gguf_logical_shape={logical_shape}")

    if logical_shape != hf_shape:
        stop(
            "HF/GGUF output shapes disagree: "
            f"HF={hf_shape}, GGUF={logical_shape}"
        )

    vocab, width = logical_shape
    if vocab != VOCAB_SIZE:
        stop(f"GGUF vocab dimension {vocab} != {VOCAB_SIZE}")

    if qtype == "BF16":
        item_size = 2
        raw_kind = "BF16"
    elif qtype == "F16":
        item_size = 2
        raw_kind = "F16"
    elif qtype == "F32":
        item_size = 4
        raw_kind = "F32"
    else:
        stop(
            "output row audit only supports BF16/F16/F32, "
            f"got {qtype}"
        )

    row_bytes = width * item_size

    rows: dict[int, np.ndarray] = {}
    raw_rows: dict[int, bytes] = {}

    for row_id in ROW_IDS:
        start = tensor.data_offset + row_id * row_bytes
        end = start + row_bytes

        raw = np.asarray(
            reader.data[start:end],
            dtype=np.uint8,
        ).tobytes()

        if len(raw) != row_bytes:
            stop(
                f"short GGUF row read for {row_id}: "
                f"{len(raw)} != {row_bytes}"
            )

        raw_rows[row_id] = raw

        if qtype == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2")
            u32 = u16.astype(np.uint32) << 16
            row = u32.view(np.float32)
        elif qtype == "F16":
            row = (
                np.frombuffer(raw, dtype="<f2")
                .astype(np.float32)
            )
        else:
            row = np.frombuffer(raw, dtype="<f4").copy()

        if row.size != width:
            stop(
                f"decoded GGUF row width {row.size} != {width}"
            )

        rows[row_id] = row.astype(np.float32, copy=False)

    return {
        "name": tensor.name,
        "shape": logical_shape,
        "rows": rows,
        "raw_rows": raw_rows,
        "raw_kind": raw_kind,
        "qtype": qtype,
    }


def compare_rows(hf, gguf):
    print()
    print("===== ROW COMPARISON =====")

    all_decoded_exact = True
    all_raw_exact = True
    raw_comparable = hf["raw_kind"] == gguf["raw_kind"]

    if not raw_comparable:
        all_raw_exact = False

    for row_id in ROW_IDS:
        h = hf["rows"][row_id]
        g = gguf["rows"][row_id]

        d = np.abs(g - h)

        h64 = h.astype(np.float64)
        g64 = g.astype(np.float64)

        hnorm = float(np.linalg.norm(h64))
        gnorm = float(np.linalg.norm(g64))

        denom = hnorm * gnorm
        cosine = (
            float(np.dot(h64, g64) / denom)
            if denom != 0.0
            else float("nan")
        )

        decoded_exact = bool(np.array_equal(h, g))
        all_decoded_exact &= decoded_exact

        raw_exact = None
        hf_sha = None
        gguf_sha = None

        if raw_comparable:
            hf_raw = hf["raw_rows"][row_id]
            gguf_raw = gguf["raw_rows"][row_id]

            hf_sha = sha256_bytes(hf_raw)
            gguf_sha = sha256_bytes(gguf_raw)
            raw_exact = hf_raw == gguf_raw
            all_raw_exact &= raw_exact

        print(f"row={row_id}")
        print(f"  decoded_exact={decoded_exact}")
        print(f"  max_abs={float(d.max()):.9g}")
        print(f"  mean_abs={float(d.mean()):.9g}")
        print(
            f"  rmse={float(np.sqrt(np.mean(d * d))):.9g}"
        )
        print(f"  hf_norm={hnorm:.9g}")
        print(f"  gguf_norm={gnorm:.9g}")
        print(f"  cosine={cosine:.12g}")

        if raw_comparable:
            print(f"  raw_exact={raw_exact}")
            print(f"  hf_raw_sha256={hf_sha}")
            print(f"  gguf_raw_sha256={gguf_sha}")
        else:
            print(
                "  raw_exact=N/A "
                f"(HF={hf['raw_kind']}, GGUF={gguf['raw_kind']})"
            )

    print()
    print(f"all_decoded_exact={all_decoded_exact}")
    print(f"raw_comparable={raw_comparable}")
    if raw_comparable:
        print(f"all_raw_exact={all_raw_exact}")

    if all_decoded_exact:
        print("OUTPUT ROW AUDIT: EXACT")
        return 0

    print("OUTPUT ROW AUDIT: DIFFERENT")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--llama-repo", required=True)
    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    gguf_dir = Path(ns.gguf_dir).resolve()
    llama_repo = Path(ns.llama_repo).resolve()

    if not model_dir.is_dir():
        stop(f"model directory missing: {model_dir}")
    if not gguf_dir.is_dir():
        stop(f"GGUF directory missing: {gguf_dir}")
    if not llama_repo.is_dir():
        stop(f"llama.cpp directory missing: {llama_repo}")

    print(f"row_ids={ROW_IDS}")

    hf = read_hf_rows(model_dir)

    hit = find_gguf_tensor(
        gguf_dir,
        llama_repo,
        hf["tied"],
    )

    gguf = decode_gguf_rows(hit, hf["shape"])

    return compare_rows(hf, gguf)


if __name__ == "__main__":
    raise SystemExit(main())