#!/usr/bin/env python3
"""Bounded extraction of LongCat-Next FFN RMSNorm vectors and epsilon metadata."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("router_extract", HERE / "extract-router-linear-diagnostic.py")
COMMON = importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(COMMON)
LAYERS = COMMON.LAYERS
PYTHON_TEMPLATE = "model.layers.{logical}.post_attention_layernorm.0.weight"
GGUF_TEMPLATE = "blk.{physical}.ffn_norm.weight"


def canonical_norm_weight(value):
    array = np.asarray(value, np.float32)
    if array.size != 3072 or array.ndim not in (1, 2):
        raise ValueError(f"FFN RMSNorm weight must contain exactly 3072 elements, got {array.shape}")
    return np.ascontiguousarray(array.reshape(3072)), bool(array.shape != (3072,))


def python_epsilon(checkpoint_dir):
    config = json.loads((Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
    value = config.get("rms_norm_eps")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("config.json has no finite rms_norm_eps")
    return float(value), "rms_norm_eps"


def gguf_reader_and_epsilon(path):
    repo = HERE.parents[1]; sys.path.insert(0, str(repo / "gguf-py"))
    from gguf import GGUFReader
    from gguf.constants import Keys, MODEL_ARCH, MODEL_ARCH_NAMES
    reader = GGUFReader(path, mode="r")
    key = Keys.Attention.LAYERNORM_RMS_EPS.format(arch=MODEL_ARCH_NAMES[MODEL_ARCH.LONGCAT_NEXT])
    field = reader.get_field(key)
    if field is None:
        raise KeyError(f"GGUF does not contain architecture RMSNorm epsilon key {key}")
    value = float(field.contents())
    if not np.isfinite(value): raise ValueError("GGUF RMSNorm epsilon is non-finite")
    return reader, value, key


def gguf_norm_weight(reader, tensor_name):
    from gguf.quants import dequantize
    tensor = next((item for item in reader.tensors if item.name == tensor_name), None)
    if tensor is None: raise KeyError(f"GGUF does not contain {tensor_name}")
    kind = COMMON.validate_gguf_router_tensor_type(tensor.tensor_type)
    value = np.asarray(dequantize(tensor.data, tensor.tensor_type), np.float32)
    return value, kind


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--checkpoint-index", default="model.safetensors.index.json")
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--python-name-template", default=PYTHON_TEMPLATE)
    p.add_argument("--gguf-name-template", default=GGUF_TEMPLATE)
    args = p.parse_args()
    index = json.loads((args.checkpoint_dir / args.checkpoint_index).read_text(encoding="utf-8"))
    py_eps, py_eps_key = python_epsilon(args.checkpoint_dir)
    reader, gg_eps, gg_eps_key = gguf_reader_and_epsilon(args.gguf)
    arrays = {}; records = []
    for logical, physical in LAYERS:
        py_name = COMMON.resolve_tensor_name(args.python_name_template, logical, physical)
        gg_name = COMMON.resolve_tensor_name(args.gguf_name_template, logical, physical)
        py_raw, py_dtype, shard = COMMON.safetensor_weight(args.checkpoint_dir, index, py_name)
        gg_raw, gg_dtype = gguf_norm_weight(reader, gg_name)
        py, py_t = canonical_norm_weight(py_raw); gg, gg_t = canonical_norm_weight(gg_raw)
        for short, source, name, value, dtype, transposed, location in (
                ("python", "python_checkpoint", py_name, py, py_dtype, py_t, shard),
                ("gguf", "gguf", gg_name, gg, gg_dtype, gg_t, args.gguf.name)):
            key = f"physical_block_{physical:02d}__{short}_ffn_norm_weight"; arrays[key] = value
            records.append(COMMON.make_weight_record(logical, physical, source, name, value,
                                                      dtype, transposed, location))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz = args.output_dir / "ffn-rmsnorm-diagnostic.npz"; COMMON.deterministic_npz(npz, arrays)
    metadata = {"schema_version": 1, "diagnostic_only": True, "model_instantiated": False,
        "inference_executed": False, "bounded_physical_blocks": [10, 12], "weight_records": records,
        "python_epsilon": py_eps, "python_epsilon_key": py_eps_key,
        "gguf_epsilon": gg_eps, "gguf_epsilon_key": gg_eps_key,
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}
    (args.output_dir / "ffn-rmsnorm-diagnostic.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__": main()
