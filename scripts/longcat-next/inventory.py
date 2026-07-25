#!/usr/bin/env python3
"""Validate the pinned LongCat-Next checkpoint metadata without loading weights."""

import argparse
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_TOTAL = 13450
EXPECTED_PAYLOAD = 150825367872
EXPECTED_MAIN = {
    "text_trunk": 11143,
    "visual_tokenizer": 425,
    "visual_head": 71,
    "audio_tokenizer": 1740,
    "audio_head": 71,
}
EXPECTED_SUBFAMILIES = {
    "model.visual_tokenizer.visual_model.": 385,
    "model.visual_tokenizer.visual_bridge_model.bridge.": 5,
    "model.visual_tokenizer.visual_bridge_model.quantizer.": 30,
    "model.visual_tokenizer.visual_embedding_layer.": 5,
    "visual_head.": 71,
    "model.audio_tokenizer.audio_model.": 487,
    "model.audio_tokenizer.audio_bridge_model.": 31,
    "model.audio_tokenizer.audio_decoder.": 149,
    "model.audio_tokenizer.audio_flow_matching_decoder.prenet.": 163,
    "model.audio_tokenizer.audio_flow_matching_decoder.conditional_decoder.": 910,
    "audio_head.": 71,
}
EXPECTED_VOCAB = {
    "text_vocab_size": 131072,
    "text_vocab_plus_multimodal_special_token_size": 131125,
    "vocab_size": 282624,
}
EXPECTED_IMAGE = {
    "image_decoder": (558, 433743858, 867487716),
    "image_refiner": (828, 4058323163, 8116646326),
    "visual_model": (385, 631975680, 1263951360),
}

class InventoryError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InventoryError(message)


def load_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read {label} {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def weight_map(index, label):
    require("weight_map" in index, f"{label}: missing required field 'weight_map'")
    require(isinstance(index["weight_map"], dict), f"{label}: weight_map must be an object")
    require("metadata" in index, f"{label}: missing required field 'metadata'")
    require(isinstance(index["metadata"], dict), f"{label}: metadata must be an object")
    return index["weight_map"]


def main_family(name):
    if name.startswith("model.visual_tokenizer."):
        return "visual_tokenizer"
    if name.startswith("visual_head."):
        return "visual_head"
    if name.startswith("model.audio_tokenizer."):
        return "audio_tokenizer"
    if name.startswith("audio_head."):
        return "audio_head"
    return "text_trunk"


def modality_subfamily(name):
    matches = [prefix for prefix in EXPECTED_SUBFAMILIES if name.startswith(prefix)]
    require(len(matches) == 1,
            f"modality tensor is unclassified or ambiguous: {name!r} (matches={matches})")
    return matches[0]


def read_safetensors_header(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(8)
            require(len(raw) == 8, f"image header {path}: missing 8-byte length")
            length = struct.unpack("<Q", raw)[0]
            require(0 < length <= 16 * 1024 * 1024,
                    f"image header {path}: unreasonable JSON length {length}")
            payload = stream.read(length)
            require(len(payload) == length,
                    f"image header {path}: truncated JSON ({len(payload)} of {length} bytes)")
        header = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot parse image safetensors header {path}: {exc}") from exc
    require(isinstance(header, dict), "image safetensors header must be an object")
    header.pop("__metadata__", None)
    return length, header


def validate_image(path):
    length, header = read_safetensors_header(path)
    require(length == 226408, f"image header length: expected 226408, got {length}")
    require(len(header) == 1771, f"image tensor count: expected 1771, got {len(header)}")
    summary = defaultdict(lambda: [0, 0, 0, set()])
    for name, item in header.items():
        require(isinstance(item, dict), f"image tensor {name}: metadata must be an object")
        for field in ("dtype", "shape", "data_offsets"):
            require(field in item, f"image tensor {name}: missing required field {field!r}")
        prefix = name.split(".", 1)[0]
        require(prefix in EXPECTED_IMAGE, f"image tensor has unclassified prefix: {name!r}")
        lo, hi = item["data_offsets"]
        summary[prefix][0] += 1
        summary[prefix][1] += math.prod(item["shape"])
        summary[prefix][2] += hi - lo
        summary[prefix][3].add(item["dtype"])
    for prefix, expected in EXPECTED_IMAGE.items():
        actual = tuple(summary[prefix][:3])
        require(actual == expected, f"image {prefix}: expected {expected}, got {actual}")
        require(summary[prefix][3] == {"BF16"},
                f"image {prefix}: expected only BF16, got {sorted(summary[prefix][3])}")
    return summary


def validate_hift(path):
    data = load_json(path, "HiFT metadata inventory")
    tensors = data.get("tensors")
    require(isinstance(tensors, dict), "HiFT metadata: missing required object field 'tensors'")
    require(len(tensors) == 328, f"HiFT tensor count: expected 328, got {len(tensors)}")
    dtypes = set()
    for name, item in tensors.items():
        require(isinstance(item, dict), f"HiFT tensor {name}: metadata must be an object")
        require("dtype" in item and "shape" in item,
                f"HiFT tensor {name}: missing dtype or shape")
        dtypes.add(str(item["dtype"]).replace("torch.", "").upper())
    require(dtypes == {"FLOAT32"} or dtypes == {"F32"},
            f"HiFT tensors: expected only F32/float32, got {sorted(dtypes)}")
    g = {name[:-9] for name in tensors if name.endswith(".weight_g")}
    v = {name[:-9] for name in tensors if name.endswith(".weight_v")}
    require(g == v and g, "HiFT metadata: weight_g/weight_v pairs are missing or unmatched")
    return len(tensors)


def validate(next_index_path, lite_index_path, config_path, image_header=None, hift_metadata=None):
    nxt = load_json(next_index_path, "LongCat-Next index")
    lite = load_json(lite_index_path, "LongCat-Flash-Lite index")
    config = load_json(config_path, "LongCat-Next config")
    next_names = weight_map(nxt, "LongCat-Next index")
    lite_names = weight_map(lite, "LongCat-Flash-Lite index")

    require(len(next_names) == EXPECTED_TOTAL,
            f"LongCat-Next tensor count: expected {EXPECTED_TOTAL}, got {len(next_names)}")
    require("total_size" in nxt["metadata"],
            "LongCat-Next index: missing required field 'metadata.total_size'")
    require(nxt["metadata"]["total_size"] == EXPECTED_PAYLOAD,
            f"LongCat-Next payload bytes: expected {EXPECTED_PAYLOAD}, got {nxt['metadata']['total_size']}")
    next_mtp = sum(name.startswith("model.mtp.") for name in next_names)
    lite_mtp = sum(name.startswith("model.mtp.") for name in lite_names)
    require(next_mtp == 0, f"LongCat-Next MTP tensor count: expected 0, got {next_mtp}")
    require(lite_mtp == 17, f"LongCat-Flash-Lite MTP tensor count: expected 17, got {lite_mtp}")

    for field, expected in EXPECTED_VOCAB.items():
        require(field in config, f"LongCat-Next config: missing required field {field!r}")
        require(config[field] == expected,
                f"LongCat-Next config {field}: expected {expected}, got {config[field]!r}")

    families = Counter(main_family(name) for name in next_names)
    require(dict(families) == EXPECTED_MAIN,
            f"LongCat-Next main family counts: expected {EXPECTED_MAIN}, got {dict(families)}")

    modal_names = [name for name in next_names if main_family(name) != "text_trunk"]
    subfamilies = Counter(modality_subfamily(name) for name in modal_names)
    require(dict(subfamilies) == EXPECTED_SUBFAMILIES,
            f"LongCat-Next modality subfamily counts: expected {EXPECTED_SUBFAMILIES}, got {dict(subfamilies)}")
    require(sum(families.values()) == len(next_names), "main classification did not consume every tensor")
    require(sum(subfamilies.values()) == len(modal_names), "modality classification did not consume every tensor")

    text_names = {name for name in next_names if main_family(name) == "text_trunk"}
    missing = sorted(text_names - set(lite_names))
    require(len(text_names) == 11143, f"LongCat-Next text/trunk count: expected 11143, got {len(text_names)}")
    require(not missing,
            f"{len(missing)} LongCat-Next text/trunk names are absent from Flash-Lite; first: {missing[:5]}")

    if image_header:
        validate_image(image_header)
    if hift_metadata:
        validate_hift(hift_metadata)

    return {
        "main_tensor_count": len(next_names), "main_payload_bytes": nxt["metadata"]["total_size"],
        "next_mtp_count": next_mtp, "lite_mtp_count": lite_mtp,
        "vocabulary_extents": EXPECTED_VOCAB, "main_families": EXPECTED_MAIN,
        "modality_subfamilies": EXPECTED_SUBFAMILIES, "text_names_in_lite": len(text_names),
        "image_checked": bool(image_header), "hift_checked": bool(hift_metadata),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next-index", required=True, type=Path)
    parser.add_argument("--lite-index", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image-header", type=Path, help="safetensors header prefix or complete file")
    parser.add_argument("--hift-metadata", type=Path, help="JSON inventory; never a pickle checkpoint")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = validate(args.next_index, args.lite_index, args.config,
                          args.image_header, args.hift_metadata)
    except InventoryError as exc:
        print(f"inventory error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
