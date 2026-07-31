#!/usr/bin/env python3
"""Extract two LongCat-Next router matrices without constructing either model."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

LAYERS = ((5, 10), (6, 12))
DEFAULT_PYTHON_NAME_TEMPLATE = "model.layers.{logical}.mlp.router.classifier.weight"
DEFAULT_GGUF_NAME_TEMPLATE = "blk.{physical}.ffn_gate_inp.weight"


def resolve_tensor_name(template, logical, physical):
    """Resolve documented logical/physical coordinates; ``layer`` aliases logical."""
    return template.format(layer=logical, logical=logical, physical=physical)


def canonical_router_weight(value, expert_count=384):
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"router weight must be rank two, got {array.shape}")
    if array.shape[0] == expert_count:
        canonical, transposed = array, False
    elif array.shape[1] == expert_count:
        canonical, transposed = array.T, True
    else:
        raise ValueError(f"router weight has no {expert_count}-expert dimension: {array.shape}")
    if canonical.shape[1] <= expert_count:
        raise ValueError(f"router hidden dimension is implausible: {canonical.shape}")
    return np.ascontiguousarray(canonical), transposed


def bf16_round_to_float32(value):
    bits = np.asarray(value, np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def metrics(left, right):
    a = np.asarray(left, np.float32)
    b = np.asarray(right, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"weight shape mismatch: {a.shape} versus {b.shape}")
    delta = np.asarray(a - b, np.float32)
    denominator = float(np.linalg.norm(a.astype(np.float64)) * np.linalg.norm(b.astype(np.float64)))
    return {"exact_element_count": int(a.size),
            "exact_byte_equality": bool(a.tobytes() == b.tobytes()),
            "exact_match_count": int(np.count_nonzero(a == b)),
            "maximum_absolute_error": float(np.abs(delta).max(initial=0)),
            "rms_error": float(np.sqrt(np.mean(np.square(delta, dtype=np.float64)))),
            "cosine_similarity": float(
                np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64))
                / denominator) if denominator else 1.0}


def weight_equivalence(python_weight, gguf_weight):
    python, python_transposed = canonical_router_weight(python_weight)
    gguf, gguf_transposed = canonical_router_weight(gguf_weight)
    rounded = bf16_round_to_float32(python)
    return {"orientation_validated": True, "python_source_transposed": python_transposed,
            "gguf_source_transposed": gguf_transposed, "canonical_shape": list(python.shape),
            "python_raw_vs_gguf_float32": metrics(python, gguf),
            "python_bf16_rounded_vs_gguf_float32": metrics(rounded, gguf),
            "gguf_equals_bf16_rounded_python_exactly": bool(np.array_equal(rounded, gguf))}


def safetensor_weight(checkpoint_dir, index, tensor_name):
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise RuntimeError("safetensors and torch are required only on the extraction workstation") from exc
    shard = index["weight_map"].get(tensor_name)
    if shard is None:
        raise KeyError(f"checkpoint index does not contain {tensor_name}")
    path = checkpoint_dir / shard
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(tensor_name)
        source_dtype = str(tensor.dtype)
        value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return np.ascontiguousarray(value, dtype=np.float32), source_dtype, path.name


def validate_gguf_router_tensor_type(tensor_type):
    """Return the stable enum name for a supported bounded router tensor type."""
    from gguf.constants import GGMLQuantizationType
    supported_types = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    }
    if tensor_type not in supported_types:
        name = getattr(tensor_type, "name", repr(tensor_type))
        raise ValueError(f"bounded diagnostic extractor supports only F32/F16/BF16, got {name}")
    return tensor_type.name


def gguf_weight(path, tensor_name):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    reader = GGUFReader(path, mode="r")
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None:
        raise KeyError(f"GGUF does not contain {tensor_name}")
    tensor_type = tensor.tensor_type
    kind = validate_gguf_router_tensor_type(tensor_type)
    value = np.asarray(dequantize(tensor.data, tensor_type), dtype=np.float32)
    return value, kind


def deterministic_npz(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload.getvalue())


def make_weight_record(logical, physical, source, name, value, source_dtype, transposed, location):
    return {"logical_layer": logical, "physical_even_block": physical, "source": source,
            "source_tensor_name": name, "source_location": location,
            "name_template_coordinates": {"logical": logical, "physical": physical,
                                          "layer_alias": logical},
            "canonical_tensor_orientation": "experts_by_hidden", "shape": list(value.shape),
            "source_dtype": source_dtype, "serialized_dtype": "float32",
            "source_was_transposed": transposed,
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "finite_audit": {"finite": bool(np.isfinite(value).all()),
                             "element_count": int(value.size)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-index", default="model.safetensors.index.json")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-name-template", default=DEFAULT_PYTHON_NAME_TEMPLATE)
    parser.add_argument("--gguf-name-template", default=DEFAULT_GGUF_NAME_TEMPLATE)
    args = parser.parse_args()
    index_path = args.checkpoint_dir / args.checkpoint_index
    index = json.loads(index_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    records = []
    equivalence_records = []
    for logical, physical in LAYERS:
        python_name = resolve_tensor_name(args.python_name_template, logical, physical)
        gguf_name = resolve_tensor_name(args.gguf_name_template, logical, physical)
        py_raw, py_dtype, shard = safetensor_weight(args.checkpoint_dir, index, python_name)
        gg_raw, gg_type = gguf_weight(args.gguf, gguf_name)
        py, py_t = canonical_router_weight(py_raw)
        gg, gg_t = canonical_router_weight(gg_raw)
        py = np.asarray(py, np.float32)
        gg = np.asarray(gg, np.float32)
        arrays[f"physical_block_{physical:02d}__python_weight"] = py
        arrays[f"physical_block_{physical:02d}__gguf_weight"] = gg
        records += [make_weight_record(logical, physical, "python_checkpoint", python_name,
                                       py, py_dtype, py_t, shard),
                    make_weight_record(logical, physical, "gguf", gguf_name,
                                       gg, gg_type, gg_t, args.gguf.name)]
        equivalence_records.append({"logical_layer": logical, "physical_even_block": physical,
                                    "python_array_key": f"physical_block_{physical:02d}__python_weight",
                                    "gguf_array_key": f"physical_block_{physical:02d}__gguf_weight",
                                    "weight_equivalence_audit": weight_equivalence(py, gg)})
    npz_path = args.output_dir / "router-linear-diagnostic.npz"
    deterministic_npz(npz_path, arrays)
    metadata = {"schema_version": 2, "diagnostic_only": True,
                "model_instantiated": False, "inference_executed": False,
                "bounded_physical_blocks": [10, 12], "weight_records": records,
                "weight_equivalence_records": equivalence_records,
                "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest()}
    (args.output_dir / "router-linear-diagnostic.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
