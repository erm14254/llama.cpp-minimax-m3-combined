#!/usr/bin/env python3
"""Local-only LongCat-Next checkpoint inspection and core fixture helpers."""

import hashlib
import io
import json
import math
import os
import random
import zipfile
from pathlib import Path

EXPECTED_TENSORS = 13450
EXPECTED_PAYLOAD = 150825367872
EXPECTED_SHARDS = 15
EXPECTED_SHARD_BYTES = 150827115056
EXPECTED_TRANSFORMERS = "4.57.6"
CORE_SCHEMA_VERSION = 1
REQUIRED_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "model.safetensors.index.json", "configuration_longcat_next.py",
    "configuration_longcat_ngram.py", "modeling_longcat_ngram.py",
    "modeling_longcat_next.py", "modular_longcat_next.py",
    "modular_longcat_next_audio.py", "modular_longcat_next_visual.py",
    "cosy24k_vocoder.py", "image_refiner.py", "refiner_modules.py",
)
EXPECTED_IDENTITIES = {
    "config.json": "9115e9785603b04382a45ebece9092235281f309f56f35eb4e43bcf53150b2a2",
    "tokenizer_config.json": "22dddd0eb59965adf6e4861a7c8a9ed803595cd16bc86ed6e2d4ed915b9718d4",
    "tokenizer.json": "9a378321656d995996c9b7db751b628ca2cf1f8c4c26832c5acea872fec6c835",
    "configuration_longcat_next.py": "bce3c8fc8bc0f4e6f3d0eb39e7a3a0415b4d66a8778f90435fb6849342c41f6c",
    "configuration_longcat_ngram.py": "96a646608a90ae4d42e6b4c8f712b01f1f9033241af97c0b3f7307dc1887d191",
    "modeling_longcat_ngram.py": "f7c6fb4de561e3311a67adea22b8a9467044d3c503d59afe0dde542e55b17e09",
    "modeling_longcat_next.py": "c62cb244285baffbcd1c8fcbf115cc2c978e8bd8811aa51e9c4e22c36c5a4b69",
    "modular_longcat_next.py": "250f63a2f24182f3d96f08818cf166ee4f4aa8ef9a872a91ea416179fe1e3d0e",
    "modular_longcat_next_audio.py": "18288c193803ddab5ee63939a05e489f794b06cc1a328b20837fb4aa60b7eb01",
    "modular_longcat_next_visual.py": "5d9a4d363b302bc598542a0f0fe3f19d6a90e0fc6e2d37046dea0f834f13da7e",
    "cosy24k_vocoder.py": "f66bba41f9afae2105bc2e3577a7d7061e916bcae56d0565cddef9b6e34d1d45",
    "image_refiner.py": "ef1e08c97b7275cb455d3baea84149510605e0cea44ab10142e5fac6d3c0baaf",
    "refiner_modules.py": "0b77854c6c1ad020879a176bc9f923e4759336133e7684b36db411db01454f75",
}
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")

class CoreFixtureError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CoreFixtureError(message)


def read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoreFixtureError(f"cannot read {label} {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(model_dir, hash_shards=False):
    root = Path(model_dir)
    require(root.is_dir(), f"checkpoint directory does not exist: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    require(not missing, "checkpoint is incomplete; missing required files: " + ", ".join(missing))
    identities = {name: file_sha256(root / name) for name in EXPECTED_IDENTITIES}
    wrong_identities = [name for name, expected in EXPECTED_IDENTITIES.items()
                        if identities[name] != expected]
    require(not wrong_identities,
            "checkpoint files do not match the pinned official revision: " + ", ".join(wrong_identities))
    index = read_json(root / "model.safetensors.index.json", "checkpoint index")
    config = read_json(root / "config.json", "checkpoint config")
    require(isinstance(index.get("weight_map"), dict), "checkpoint index is missing weight_map")
    require(isinstance(index.get("metadata"), dict), "checkpoint index is missing metadata")
    names = index["weight_map"]
    require(len(names) == EXPECTED_TENSORS,
            f"indexed tensor count must be {EXPECTED_TENSORS}, got {len(names)}")
    require(index["metadata"].get("total_size") == EXPECTED_PAYLOAD,
            f"index tensor payload bytes must be {EXPECTED_PAYLOAD}, got {index['metadata'].get('total_size')!r}")
    mtp = [name for name in names if name.startswith("model.mtp.")]
    require(not mtp, f"checkpoint unexpectedly contains {len(mtp)} model.mtp.* tensors")
    for field, expected in (("text_vocab_size", 131072),
                            ("text_vocab_plus_multimodal_special_token_size", 131125),
                            ("vocab_size", 282624)):
        require(config.get(field) == expected,
                f"config {field} must be {expected}, got {config.get(field)!r}")
    shards = sorted(set(names.values()))
    require(len(shards) == EXPECTED_SHARDS,
            f"checkpoint must reference exactly {EXPECTED_SHARDS} unique shards, got {len(shards)}")
    require(all(Path(name).name == name and name.endswith(".safetensors") for name in shards),
            "checkpoint index contains an unsafe or non-safetensors shard name")
    missing_shards = [name for name in shards if not (root / name).is_file()]
    require(not missing_shards, "checkpoint is missing referenced shards: " + ", ".join(missing_shards))
    actual_safetensors = sorted(path.name for path in root.glob("model-*.safetensors"))
    additional = sorted(set(actual_safetensors) - set(shards))
    require(not additional, "checkpoint has unreferenced model shards: " + ", ".join(additional))
    shard_sizes = {name: (root / name).stat().st_size for name in shards}
    total_bytes = sum(shard_sizes.values())
    require(total_bytes == EXPECTED_SHARD_BYTES,
            f"total shard-file bytes must be {EXPECTED_SHARD_BYTES}, got {total_bytes}")
    hashes = {name: file_sha256(root / name) for name in shards} if hash_shards else None
    code_hashes = {name: identities[name] for name in REQUIRED_FILES if name.endswith(".py")}
    return {"model_dir": str(root.resolve()), "tensor_count": len(names),
            "tensor_payload_bytes": index["metadata"]["total_size"],
            "shard_count": len(shards), "total_shard_file_bytes": total_bytes,
            "shards": [{"name": name, "bytes": shard_sizes[name],
                        **({"sha256": hashes[name]} if hashes else {})} for name in shards],
            "vocabulary_extents": {field: config[field] for field in
                ("text_vocab_size", "text_vocab_plus_multimodal_special_token_size", "vocab_size")},
            "mtp_tensor_count": 0, "custom_code_sha256": code_hashes,
            "config_sha256": file_sha256(root / "config.json"),
            "tokenizer_config_sha256": file_sha256(root / "tokenizer_config.json"),
            "tokenizer_sha256": file_sha256(root / "tokenizer.json")}


def parse_memory_limit(value, label):
    if value is None:
        return None
    require(isinstance(value, str) and value.strip(), f"{label} memory limit must be a non-empty string")
    require(any(value.upper().endswith(unit) for unit in ("GB", "GIB", "MB", "MIB")),
            f"{label} memory limit must include GB, GiB, MB, or MiB")
    return value.strip()


def validate_core_options(precision, placement, repeats, max_output_bytes):
    require(precision in ("bf16", "f16"), "precision must be bf16 or f16")
    require(placement in ("auto", "cpu", "cuda"), "placement must be auto, cpu, or cuda")
    require(isinstance(repeats, int) and repeats >= 1, "repeat count must be at least 1")
    require(0 < max_output_bytes <= 256 * 1024 * 1024,
            "maximum generated-output size must be between 1 byte and 256 MiB")


def enforce_offline_environment():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def loading_kwargs(precision, placement, offload_dir=None, cpu_memory=None, gpu_memory=None):
    validate_core_options(precision, placement, 1, 1)
    enforce_offline_environment()
    import torch
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    kwargs = {"local_files_only": True, "trust_remote_code": True,
              "use_safetensors": True, "low_cpu_mem_usage": True, "torch_dtype": dtype}
    if placement == "auto":
        kwargs["device_map"] = "auto"
    elif placement == "cpu":
        kwargs["device_map"] = {"": "cpu"}
    else:
        kwargs["device_map"] = {"": 0}
    max_memory = {}
    if cpu_memory:
        max_memory["cpu"] = parse_memory_limit(cpu_memory, "CPU")
    if gpu_memory:
        max_memory[0] = parse_memory_limit(gpu_memory, "GPU")
    if max_memory:
        kwargs["max_memory"] = max_memory
    if offload_dir:
        kwargs["offload_folder"] = str(Path(offload_dir))
        kwargs["offload_state_dict"] = True
    return kwargs


def ensure_transformers_version():
    import transformers
    require(transformers.__version__ == EXPECTED_TRANSFORMERS,
            f"Transformers {EXPECTED_TRANSFORMERS} is required; installed version is {transformers.__version__}")


def resolve_capture_modules(model):
    try:
        trunk = model.model
        layers = trunk.layers
        require(len(layers) == 14, f"official trunk must expose 14 logical layers, got {len(layers)}")
        for index, layer in enumerate(layers):
            require(len(layer.input_layernorm) == 2,
                    f"logical layer {index} does not expose two physical input norms")
        require(len(trunk.ngram_embeddings.post_projs) == 12,
                "official n-gram module does not expose twelve post projections")
        return {"trunk": trunk, "base_embedding": trunk.embed_tokens,
                "ngram_projections": list(trunk.ngram_embeddings.post_projs),
                "physical_block_inputs": {0: layers[0].input_layernorm[1],
                                          2: layers[1].input_layernorm[1]},
                "physical_block_outputs": {1: layers[0], 27: layers[13]},
                "final_norm": trunk.norm, "lm_head": model.lm_head}
    except (AttributeError, TypeError, CoreFixtureError) as exc:
        if isinstance(exc, CoreFixtureError):
            raise
        raise CoreFixtureError(f"cannot resolve required official module hook: {exc}") from exc


def deterministic_npz_bytes(arrays):
    import numpy as np
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            require(not name.endswith(WEIGHT_SUFFIXES), f"model-weight filename is forbidden: {name}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def array_metadata(arrays):
    result = {}
    for name, value in sorted(arrays.items()):
        data = value.tobytes(order="C")
        result[name] = {"shape": list(value.shape), "serialized_dtype": str(value.dtype),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return result


def compare_runs(runs):
    import numpy as np
    require(len(runs) >= 1, "at least one official run is required")
    baseline = runs[0]
    reports = {}
    for name, reference in baseline.items():
        require(all(name in run for run in runs), f"repeated run is missing activation {name}")
        exact_kind = reference.dtype.kind in "biu" or name.startswith(("input_ids", "attention_mask", "position_ids", "greedy_ids"))
        maximum_abs = 0.0
        maximum_rel = 0.0
        byte_identical = True
        for run in runs[1:]:
            candidate = run[name]
            require(candidate.shape == reference.shape, f"repeated shape mismatch for {name}")
            byte_identical &= candidate.tobytes() == reference.tobytes()
            if exact_kind:
                require(np.array_equal(candidate, reference), f"repeated exact output differs for {name}")
            else:
                left = reference.astype(np.float64)
                right = candidate.astype(np.float64)
                diff = np.abs(left - right)
                maximum_abs = max(maximum_abs, float(diff.max(initial=0.0)))
                denom = np.maximum(np.abs(left), np.finfo(np.float64).tiny)
                maximum_rel = max(maximum_rel, float((diff / denom).max(initial=0.0)))
        reports[name] = {"shape": list(reference.shape), "dtype": str(reference.dtype),
                         "byte_identical": byte_identical,
                         "max_absolute_difference": maximum_abs,
                         "max_relative_difference": maximum_rel,
                         "comparison_tolerance": None}
    extra = set().union(*(set(run) for run in runs)) - set(baseline)
    require(not extra, f"repeated run has unexpected activations: {sorted(extra)}")
    return reports


def write_core_outputs(output_dir, stem, arrays, metadata, reproducibility, max_output_bytes):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    require(not any(path.name.lower().endswith(WEIGHT_SUFFIXES) for path in root.iterdir()),
            "output directory contains a model-weight filename")
    npz = deterministic_npz_bytes(arrays)
    metadata = dict(metadata)
    metadata["arrays"] = array_metadata(arrays)
    json_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("ascii")
    repro_bytes = (json.dumps(reproducibility, indent=2, sort_keys=True) + "\n").encode("ascii")
    total = len(npz) + len(json_bytes) + len(repro_bytes)
    require(total <= max_output_bytes,
            f"generated fixture output would be {total} bytes, above limit {max_output_bytes}")
    paths = {"npz": root / f"{stem}.npz", "metadata": root / f"{stem}.json",
             "reproducibility": root / "longcat-next-core-reproducibility.json"}
    payloads = {"npz": npz, "metadata": json_bytes, "reproducibility": repro_bytes}
    for name, path in paths.items():
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payloads[name])
        temporary.replace(path)
    return paths

def tensor_array(tensor):
    import torch
    value = tensor.detach().cpu()
    source_dtype = str(value.dtype).replace("torch.", "")
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.contiguous().numpy(), source_dtype


def load_case_ids(weight_free_fixture, tokenizer):
    corpus = read_json(weight_free_fixture, "weight-free fixture")
    cases = []
    for case in corpus.get("cases", []):
        if "input_ids" in case and case["name"] != "all_ignored_ids":
            cases.append((case["name"], list(case["input_ids"])))
    prompts = ["The capital of France is", "Write one short sentence about a cat."]
    for index, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)
        cases.append((f"tokenizer_prompt_{index}", list(encoded["input_ids"])))
    require(cases, "no explicit token-ID cases were found")
    return cases, prompts


def capture_forward(model, input_ids, selected_logit_ids, case_name):
    import numpy as np
    import torch
    modules = resolve_capture_modules(model)
    captured = {}
    source_dtypes = {}
    handles = []

    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        array, dtype = tensor_array(value)
        captured[name] = array
        source_dtypes[name] = dtype

    handles.append(modules["base_embedding"].register_forward_hook(
        lambda module, args, output: save("base_embedding", output)))
    for index, projection in enumerate(modules["ngram_projections"]):
        handles.append(projection.register_forward_hook(
            lambda module, args, output, index=index: save(f"ngram_projection_raw_{index:02d}", output)))
    for block, module in modules["physical_block_inputs"].items():
        handles.append(module.register_forward_pre_hook(
            lambda module, args, block=block: save(f"physical_block_{block:02d}", args[0])))
    for block, module in modules["physical_block_outputs"].items():
        handles.append(module.register_forward_hook(
            lambda module, args, output, block=block: save(f"physical_block_{block:02d}", output)))
    handles.append(modules["trunk"].layers[0].register_forward_pre_hook(
        lambda module, args: save("fused_pre_trunk_embedding", args[0])))
    handles.append(modules["final_norm"].register_forward_hook(
        lambda module, args, output: save("final_normalized_hidden_state", output)))

    device = modules["base_embedding"].weight.device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids)
    positions = torch.arange(ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    try:
        with torch.inference_mode():
            output = model(input_ids=ids.clone(), attention_mask=attention,
                           position_ids=positions, use_cache=False,
                           logits_to_keep=1, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    require(len([name for name in captured if name.startswith("ngram_projection_raw_")]) == 12,
            f"official module hooks did not capture all twelve n-gram contributions for {case_name}")
    ignored = np.asarray([(131072 <= token < 131125) for token in input_ids], dtype=bool)
    for index in range(12):
        raw_name = f"ngram_projection_raw_{index:02d}"
        effective = captured[raw_name].copy()
        effective[:, ~ignored, :] /= 13.0
        name = f"ngram_contribution_{index:02d}"
        captured[name] = effective
        source_dtypes[name] = source_dtypes[raw_name]
    for block in (0, 1, 2, 27):
        require(f"physical_block_{block:02d}" in captured,
                f"official module hook for physical block {block} did not fire")
    logits, logits_dtype = tensor_array(output.logits[:, -1, :])
    require(logits.shape[-1] == 131125,
            f"official final logits must have 131125 entries, got {logits.shape[-1]}")
    selected = logits[:, selected_logit_ids]
    captured["selected_logits"] = selected
    source_dtypes["selected_logits"] = logits_dtype
    if case_name.startswith("tokenizer_prompt_0"):
        captured["complete_final_position_logits"] = logits
        source_dtypes["complete_final_position_logits"] = logits_dtype
    values, indices = torch.topk(output.logits[:, -1, :].float(), k=10, dim=-1)
    captured["topk_token_ids"] = indices.cpu().numpy()
    captured["topk_values"] = values.cpu().numpy()
    captured["argmax_token_id"] = indices[:, :1].cpu().numpy()
    source_dtypes.update({"topk_token_ids": "int64", "topk_values": "float32",
                          "argmax_token_id": "int64"})
    captured["input_ids"] = ids.cpu().numpy()
    captured["attention_mask"] = attention.cpu().numpy()
    captured["position_ids"] = positions.cpu().numpy()
    source_dtypes.update({"input_ids": "int64", "attention_mask": "int64", "position_ids": "int64"})
    return captured, source_dtypes


def run_core_generation(args, weight_free_fixture):
    validate_core_options(args.precision, args.placement, args.repeat_count, args.max_output_bytes)
    inspection = validate_checkpoint(args.model_dir, args.hash_shards)
    enforce_offline_environment()
    try:
        ensure_transformers_version()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise CoreFixtureError(
            "core generation requires local torch, Transformers 4.57.6, Accelerate, safetensors, and numpy") from exc
    random.seed(20260725)
    torch.manual_seed(20260725)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260725)
    torch.use_deterministic_algorithms(True, warn_only=False)
    load = loading_kwargs(args.precision, args.placement, args.offload_dir,
                          args.cpu_memory, args.gpu_memory)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), **load)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CoreFixtureError(f"local official model loading failed without network fallback: {exc}") from exc
    model.eval()
    resolve_capture_modules(model)
    cases, prompts = load_case_ids(weight_free_fixture, tokenizer)
    selected_logit_ids = sorted({0, 1, 2, 131071, *range(131072, 131125)})
    runs = []
    source_dtypes = {}
    greedy_by_run = []
    for repeat in range(args.repeat_count):
        arrays = {}
        repeat_sources = {}
        for case_name, ids in cases:
            case_arrays, case_sources = capture_forward(model, ids, selected_logit_ids, case_name)
            for name, value in case_arrays.items():
                arrays[f"{case_name}/{name}"] = value
                repeat_sources[f"{case_name}/{name}"] = case_sources[name]
        greedy = {}
        for index, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt")
            device = resolve_capture_modules(model)["base_embedding"].weight.device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False,
                                           max_new_tokens=args.max_new_tokens, use_cache=True)
            result = generated.detach().cpu().numpy()
            arrays[f"greedy_ids/prompt_{index}"] = result
            repeat_sources[f"greedy_ids/prompt_{index}"] = "int64"
            greedy[f"prompt_{index}"] = result.tolist()
        runs.append(arrays)
        greedy_by_run.append(greedy)
        if repeat == 0:
            source_dtypes = repeat_sources
    reproducibility = {"schema_version": CORE_SCHEMA_VERSION, "repeat_count": args.repeat_count,
                       "arrays": compare_runs(runs), "greedy_continuations": greedy_by_run,
                       "comparison_tolerances": {"bf16": None, "f16": None},
                       "tolerances_selected_automatically": False}
    metadata = {"schema_version": CORE_SCHEMA_VERSION, "kind": "longcat-next-core-reference",
                "precision": args.precision, "serialized_activation_dtype": "float32 except integer arrays",
                "source_dtypes": source_dtypes, "checkpoint": inspection,
                "software_versions": {"transformers": EXPECTED_TRANSFORMERS,
                                      "torch": torch.__version__},
                "seeds": {"python": 20260725, "torch": 20260725},
                "selected_logit_token_ids": selected_logit_ids,
                "module_anchors": {
                    "base_embedding": "model.model.embed_tokens forward output",
                    "ngram_projection_raw": "model.model.ngram_embeddings.post_projs[0..11] outputs",
                    "ngram_contributions": "raw projection outputs with official conditional /13 scaling applied",
                    "fused_pre_trunk": "input to model.model.layers[0]",
                    "physical_blocks_0_2": "inputs to logical layers 0/1 input_layernorm[1]",
                    "physical_blocks_1_27": "outputs of logical layers 0/13",
                    "final_norm": "model.model.norm output",
                    "logits": "LongcatNextForCausalLM logits_to_keep=1 output"}}
    stem = f"longcat-next-core-{args.precision}"
    return write_core_outputs(args.output_dir, stem, runs[0], metadata,
                              reproducibility, args.max_output_bytes)
