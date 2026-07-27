#!/usr/bin/env python3
"""Local-only LongCat-Next checkpoint inspection and core fixture helpers."""

import hashlib
import importlib
import importlib.metadata
import io
import inspect
import json
import math
import os
import random
import re
import sys
import zipfile
import importlib.util
from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote, urlparse

EXPECTED_TENSORS = 13450
EXPECTED_PAYLOAD = 150825367872
EXPECTED_SHARDS = 15
EXPECTED_SHARD_BYTES = 150827115056
EXPECTED_TRANSFORMERS = "4.57.6"
OFFICIAL_FLASH_ATTN = "2.7.4.post1"
CORE_SCHEMA_VERSION = 2

_V2_SPEC = importlib.util.spec_from_file_location(
    "longcat_next_core_reference_v2", Path(__file__).with_name("core_reference_v2.py"))
v2 = importlib.util.module_from_spec(_V2_SPEC)
_V2_SPEC.loader.exec_module(v2)
REQUIRED_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "generation_config.json",
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
    "generation_config.json": "a253caa8a57fbf3782ca2db5ecbc02e1f208b2ebdda056ef501f98a35b1d02cf",
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
EXPECTED_DEPENDENCIES = {
    "torch": ("torch", "2.6.0"),
    "torchvision": ("torchvision", "0.21.0"),
    "torchaudio": ("torchaudio", "2.6.0"),
    "accelerate": ("accelerate", "1.10.0"),
    "transformers": ("transformers", "4.57.6"),
    "librosa": ("librosa", "0.11.0"),
    "diffusers": ("diffusers", "0.34.0"),
    "einops": ("einops", None),
    "flash_attn": ("flash-attn", OFFICIAL_FLASH_ATTN),
    "safetensors": ("safetensors", None),
    "numpy": ("numpy", None),
}
RUNTIME_PROFILES = ("official-pinned", "blackwell-compatible")

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
    generation = read_json(root / "generation_config.json", "generation config")
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
    for field, expected in (("bos_token_id", 1), ("eos_token_id", 2),
                            ("pad_token_id", 3), ("transformers_version", "4.57.6")):
        require(generation.get(field) == expected,
                f"generation_config {field} must be {expected!r}, got {generation.get(field)!r}")
    visual = generation.get("visual_generation_config")
    audio = generation.get("audio_generation_config")
    require(isinstance(visual, dict), "generation_config must contain visual_generation_config")
    require(isinstance(audio, dict), "generation_config must contain audio_generation_config")
    visual_custom = visual.get("custom_params")
    require(isinstance(visual_custom, dict),
            "visual_generation_config must contain custom_params")
    require(visual_custom.get("token_h") == 37 and visual_custom.get("token_w") == 37,
            "visual token_h and token_w must both be 37")
    require(visual_custom.get("anyres_prefix") ==
            "<longcat_img_token_size>{h} {w}</longcat_img_token_size>",
            "visual anyres_prefix does not match the pinned official value")
    require(audio.get("audio_parallel_decoding") is False,
            "audio_parallel_decoding must be false")
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
            "generation_config_sha256": identities["generation_config.json"],
            "tokenizer_config_sha256": file_sha256(root / "tokenizer_config.json"),
            "tokenizer_sha256": file_sha256(root / "tokenizer.json")}


def version_pair(value):
    match = __import__("re").match(r"^(\d+)\.(\d+)", value or "")
    return tuple(map(int, match.groups())) if match else None


def runtime_probe(packages, runtime_profile, placement):
    report = {"ok": True, "operating_system": os.name,
              "platform": __import__("platform").platform(),
              "python_version": __import__("platform").python_version(),
              "python_implementation": __import__("platform").python_implementation(),
              "python_executable": sys.executable,
              "cuda_selected": placement in ("auto", "cuda")}
    try:
        torch = importlib.import_module("torch")
        report.update({"torch_version": getattr(torch, "__version__", None),
                       "torch_cuda_build": getattr(torch.version, "cuda", None),
                       "torch_cxx11_abi": (torch.compiled_with_cxx11_abi()
                                           if hasattr(torch, "compiled_with_cxx11_abi")
                                           else None)})
        if runtime_profile == "official-pinned":
            report["ok"] &= packages["torch"]["version_ok"]
        if report["cuda_selected"]:
            require(torch.cuda.is_available(), "CUDA placement was selected but torch.cuda.is_available() is false")
            capability = tuple(torch.cuda.get_device_capability(0))
            arch_list = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
            report.update({"gpu_name": torch.cuda.get_device_name(0),
                           "gpu_compute_capability": list(capability),
                           "torch_cuda_arch_list": arch_list,
                           "sm_120_listed": "sm_120" in arch_list or "compute_120" in arch_list,
                           "bf16_supported": bool(torch.cuda.is_bf16_supported())})
            if capability == (12, 0) and runtime_profile == "official-pinned":
                raise CoreFixtureError(
                    "sm_120 Blackwell cannot use the official-pinned torch 2.6.0 runtime; use blackwell-compatible")
            if runtime_profile == "blackwell-compatible":
                require(capability == (12, 0),
                        f"blackwell-compatible requires an sm_120 GPU, got compute capability {capability}")
                require(version_pair(report["torch_version"]) >= (2, 7),
                        f"sm_120 requires torch 2.7 or newer, got {report['torch_version']}")
                require(version_pair(report["torch_cuda_build"]) >= (12, 8),
                        f"sm_120 requires a CUDA 12.8+ torch build, got {report['torch_cuda_build']}")
                require(report["sm_120_listed"],
                        f"installed torch architecture list does not include sm_120: {arch_list}")
            require(report["bf16_supported"], "selected CUDA device does not report BF16 support")
            left = torch.tensor([1.0, 2.0], device="cuda")
            right = (left * left + 1).sum()
            torch.cuda.synchronize()
            require(float(right.cpu()) == 7.0, "small CUDA tensor operation returned an unexpected value")
            report["cuda_tensor_operation"] = "passed"
        if runtime_profile == "blackwell-compatible":
            torch_pair = version_pair(packages["torch"]["installed_version"])
            audio_pair = version_pair(packages["torchaudio"]["installed_version"])
            vision_pair = version_pair(packages["torchvision"]["installed_version"])
            require(audio_pair == torch_pair,
                    f"torchaudio {audio_pair} does not match torch {torch_pair}")
            require(vision_pair is not None and torch_pair is not None and
                    vision_pair[0] == 0 and vision_pair[1] == torch_pair[1] + 15,
                    f"torchvision {vision_pair} does not match torch {torch_pair}")
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def import_local_custom_classes(model_dir):
    report = {"ok": False, "classes": {}, "error": None}
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        for reference in ("configuration_longcat_next.LongcatNextConfig",
                          "modeling_longcat_next.LongcatNextModel",
                          "modeling_longcat_next.LongcatNextForCausalLM"):
            cls = get_class_from_dynamic_module(
                reference, str(model_dir), local_files_only=True, trust_remote_code=True)
            report["classes"][reference] = f"{cls.__module__}.{cls.__name__}"
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def wheel_filename_from_origin(distribution, explicit_wheel_path=None):
    """Return the original wheel identity, preferring PEP 610 direct_url.json."""
    direct_text = distribution.read_text("direct_url.json")
    direct = None
    if direct_text:
        try:
            direct = json.loads(direct_text)
        except json.JSONDecodeError as exc:
            raise CoreFixtureError(f"invalid flash-attn direct_url.json: {exc}") from exc
        require(isinstance(direct, dict) and isinstance(direct.get("url"), str),
                "flash-attn direct_url.json is missing its URL")
        parsed = urlparse(direct["url"])
        filename = Path(unquote(parsed.path)).name
        archive_hash = (direct.get("archive_info") or {}).get("hash")
        source = "pep610-direct-url"
        url = direct["url"]
    elif explicit_wheel_path is not None:
        value = str(explicit_wheel_path)
        filename = Path(value).name
        if "\\" in value:
            filename = value.rsplit("\\", 1)[-1]
        archive_hash = None
        source = "explicit-wheel-path"
        url = None
    else:
        raise CoreFixtureError(
            "wheel origin identity unavailable: flash-attn direct_url.json is absent; "
            "provide --flash-wheel-path pointing to the original wheel for identity validation")
    require(filename.lower().endswith(".whl"),
            f"FlashAttention origin is not a wheel filename: {filename!r}")
    known_source = False
    if url:
        parsed = urlparse(url)
        known_source = (parsed.hostname or "").lower() == "huggingface.co" and \
            unquote(parsed.path).startswith("/ussoewwin/Flash-Attention-2_for_Windows/")
    return {"identity_source": source, "direct_url": url,
            "original_wheel_filename": filename, "archive_hash": archive_hash,
            "known_community_windows_source": known_source}


def windows_flash_abi(wheel_filename, wheel_tags, compatible_tags, torch_version,
                      cuda_version, capability, torch_cxx11_abi=None):
    """Validate the metadata encoded by a native Windows community wheel."""
    value = wheel_filename.lower()
    cuda_match = re.search(r"cu(\d{3})", value)
    torch_match = re.search(r"torch(\d+\.\d+(?:\.\d+)?)", value)
    cxx_match = re.search(r"cxx11abi(true|false)", value)
    wheel_cuda = (f"{cuda_match.group(1)[:-1]}.{cuda_match.group(1)[-1]}"
                  if cuda_match else None)
    wheel_torch = torch_match.group(1) if torch_match else None
    normalized_torch = (torch_version or "").split("+")[0]
    wheel_cxx11_abi = None if cxx_match is None else cxx_match.group(1) == "true"
    matching_tags = sorted(set(wheel_tags) & set(compatible_tags))
    checks = {
        "sm_120_device": tuple(capability or ()) == (12, 0),
        "blackwell_kernel_build": ".blackwell" in value,
        "cuda_build_matches": wheel_cuda is not None and wheel_cuda == cuda_version,
        "torch_build_matches": wheel_torch is not None and wheel_torch == normalized_torch,
        "cxx11_abi_matches": (wheel_cxx11_abi is not None and
                              torch_cxx11_abi is not None and
                              wheel_cxx11_abi == bool(torch_cxx11_abi)),
        "python_abi_platform_tag_matches": bool(matching_tags),
    }
    return {"ok": all(checks.values()), "checks": checks,
            "wheel_cuda": wheel_cuda, "wheel_torch": wheel_torch,
            "wheel_cxx11_abi": wheel_cxx11_abi,
            "wheel_tags": sorted(wheel_tags), "executing_compatible_tags": matching_tags,
            "executing_python": sys.version.split()[0]}


def windows_flash_distribution_report(distribution, runtime, explicit_wheel_path=None):
    from packaging.tags import sys_tags
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    wheel_text = distribution.read_text("WHEEL") or ""
    origin = wheel_filename_from_origin(distribution, explicit_wheel_path)
    metadata_tags = [line.split(":", 1)[1].strip() for line in wheel_text.splitlines()
                     if line.startswith("Tag:")]
    try:
        _, _, _, filename_tags = parse_wheel_filename(origin["original_wheel_filename"])
    except InvalidWheelFilename as exc:
        raise CoreFixtureError(
            f"cannot parse original FlashAttention wheel filename: {exc}") from exc
    wheel_tags = [str(tag) for tag in filename_tags]
    require(set(wheel_tags) == set(metadata_tags),
            "original FlashAttention wheel Python/ABI/platform tags do not match installed WHEEL metadata")
    report = windows_flash_abi(
        origin["original_wheel_filename"], wheel_tags, [str(tag) for tag in sys_tags()],
        runtime.get("torch_version"), runtime.get("torch_cuda_build"),
        runtime.get("gpu_compute_capability"), runtime.get("torch_cxx11_abi"))
    report.update({"distribution_name": distribution.metadata.get("Name", "flash-attn"),
                   "distribution_version": distribution.version,
                   "distribution_path": str(distribution.locate_file("")),
                   "installed_wheel_metadata_tags": sorted(metadata_tags),
                   **origin,
                   "community_unofficial_windows_build": True,
                   "community_source":
                       "https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows"})
    return report


def perform_flash_attention_smoke(torch, flash_attn):
    function = getattr(flash_attn, "flash_attn_func")
    query = torch.randn((1, 4, 2, 16), device="cuda", dtype=torch.bfloat16)
    output = function(query, query, query, dropout_p=0.0, causal=True)
    require(tuple(output.shape) == tuple(query.shape),
            f"FlashAttention output shape {tuple(output.shape)} does not match input {tuple(query.shape)}")
    require(bool(torch.isfinite(output).all()), "FlashAttention tiny forward produced non-finite values")
    torch.cuda.synchronize()
    return {"operation": "passed", "input_shape": list(query.shape),
            "output_shape": list(output.shape), "dtype": "bfloat16",
            "causal": True, "finite_values": True, "cuda_synchronize": "passed"}


def flash_attention_probe(runtime, placement, runtime_profile, flash_wheel_path=None):
    platform_module = __import__("platform")
    report = {"ok": True, "platform": platform_module.platform(),
              "operating_system": platform_module.system(), "operation": "not requested",
              "official_pinned_version": OFFICIAL_FLASH_ATTN}
    if placement not in ("auto", "cuda"):
        return report
    try:
        torch = importlib.import_module("torch")
        flash_attn = importlib.import_module("flash_attn")
        distribution = importlib.metadata.distribution("flash-attn")
        installed = distribution.version
        report.update({"installed_distribution_version": installed,
                       "module_path": getattr(flash_attn, "__file__", None),
                       "version_departure_from_official": installed != OFFICIAL_FLASH_ATTN})
        if runtime_profile == "official-pinned":
            require(installed == OFFICIAL_FLASH_ATTN,
                    f"official-pinned requires flash-attn {OFFICIAL_FLASH_ATTN}, got {installed}")
            report["provenance"] = "official LongCat requirement"
        elif platform_module.system() == "Windows":
            abi = windows_flash_distribution_report(distribution, runtime, flash_wheel_path)
            report["windows_wheel_abi"] = abi
            require(abi["ok"], "native Windows FlashAttention wheel ABI does not match " +
                    "the executing Python, PyTorch, CUDA, platform, and sm_120 device: " +
                    json.dumps(abi, sort_keys=True))
            report.update({"provenance": "community/unofficial native Windows build",
                           "wsl_required": False})
        report.update(perform_flash_attention_smoke(torch, flash_attn))
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def packages_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_preflight(runtime_profile="official-pinned", placement="cpu", model_dir=None,
                         flash_wheel_path=None):
    require(runtime_profile in RUNTIME_PROFILES,
            f"runtime profile must be one of {RUNTIME_PROFILES}")
    packages = {}
    for module_name, (distribution, expected) in EXPECTED_DEPENDENCIES.items():
        row = {"distribution": distribution, "required_version": expected,
               "installed_version": None, "import_ok": False, "error": None}
        try:
            row["installed_version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            row["error"] = "distribution is not installed"
        try:
            importlib.import_module(module_name)
            row["import_ok"] = True
        except Exception as exc:
            row["error"] = f"import failed: {type(exc).__name__}: {exc}"
        exact_runtime = not (runtime_profile == "blackwell-compatible" and
                             module_name in ("torch", "torchvision", "torchaudio", "flash_attn"))
        row["version_ok"] = expected is None or not exact_runtime or row["installed_version"] == expected
        row["official_pinned_version"] = expected
        row["runtime_profile_allows_departure"] = not exact_runtime
        if not row["version_ok"] and row["error"] is None:
            row["error"] = (f"expected {expected}, installed {row['installed_version']}")
        packages[module_name] = row
    packages_ok = all(row["import_ok"] and row["version_ok"] for row in packages.values())
    runtime = runtime_probe(packages, runtime_profile, placement)
    if packages_ok and runtime["ok"]:
        flash = flash_attention_probe(runtime, placement, runtime_profile, flash_wheel_path)
    else:
        flash = {"ok": False, "skipped": True,
                 "reason": "package imports/versions or runtime probe failed"}
    # The dynamic loader is intentionally gated behind every earlier stage.
    if model_dir is None:
        custom_code = None
    elif packages_ok and runtime["ok"] and flash["ok"]:
        custom_code = import_local_custom_classes(model_dir)
    else:
        custom_code = {"ok": False, "skipped": True,
                       "reason": "runtime and FlashAttention preflight must pass first"}
    ok = (packages_ok and runtime["ok"] and flash["ok"] and
          (custom_code is None or custom_code["ok"]))
    return {"ok": ok, "runtime_profile": runtime_profile, "packages": packages,
            "runtime": runtime, "flash_attention": flash, "custom_code": custom_code}


def require_dependency_preflight(runtime_profile="official-pinned", placement="cpu", model_dir=None,
                                 flash_wheel_path=None):
    report = dependency_preflight(runtime_profile, placement, model_dir, flash_wheel_path)
    require(report["ok"], "dependency preflight failed; do not load model weights:\n" +
            json.dumps(report, indent=2, sort_keys=True))
    return report


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
              "use_safetensors": True, "low_cpu_mem_usage": True, "dtype": dtype}
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


ROUTER_GPU_HEADROOM_BYTES = 128 * 1024 * 1024
ROUTER_PLACEMENT_POLICY = "longcat-router-preload-v1"
ROUTER_PRELOAD_CLASS = "LongcatFlashTopkRouter"


def reserve_router_gpu_headroom(requested_gpu_bytes):
    require(requested_gpu_bytes > ROUTER_GPU_HEADROOM_BYTES,
            "GPU memory limit is smaller than reserved LongCat router headroom")
    return requested_gpu_bytes - ROUTER_GPU_HEADROOM_BYTES


def router_no_split_classes(existing):
    return sorted(set(existing) | {ROUTER_PRELOAD_CLASS})


def write_placement_failure(output_dir, phase, placement, policy, error):
    v2.atomic_json(Path(output_dir) / "placement-audit.json", {
        "schema_version": 2, "placement": placement, "placement_policy": policy,
        "phase": phase, "all_router_direct_weight_access_safe": False,
        "router_count": None, "exception_type": type(error).__name__,
        "exception_message": str(error),
        "conflicting_key_pair": getattr(error, "conflicting_key_pair", None),
    })


def resolve_router_prefixes(named_modules):
    import re
    found = []
    for name, _ in named_modules:
        match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.router", name)
        if match:
            found.append((int(match.group(1)), name))
    found.sort()
    require([index for index, _ in found] == list(range(14)),
            f"explicit placement must resolve exactly routers 0..13, got {found}")
    return [name for _, name in found]


class DeviceMapValidationError(CoreFixtureError):
    def __init__(self, message, conflicting_key_pair=None):
        super().__init__(message)
        self.conflicting_key_pair = conflicting_key_pair


def validate_ordinary_device_map(device_map, model, router_prefixes):
    keys = list(device_map)
    if "" in device_map and len(keys) != 1:
        raise DeviceMapValidationError(
            "empty root device-map key is only valid as the sole assignment", ["", keys[1]])
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if ((left and right.startswith(left + ".")) or
                    (right and left.startswith(right + "."))):
                pair = [left, right]
                raise DeviceMapValidationError(
                    f"overlapping device-map assignments: {left!r} and {right!r}", pair)
    parameters = list(model.named_parameters())
    for name, _ in parameters:
        governing = [key for key in keys if key == "" or name == key or name.startswith(key + ".")]
        if len(governing) != 1:
            raise DeviceMapValidationError(
                f"parameter {name} has {len(governing)} governing device-map assignments")
    router_governing = {}
    for prefix in router_prefixes:
        direct_descendants = [key for key in keys if key.startswith(prefix + ".")]
        if direct_descendants:
            raise DeviceMapValidationError(
                f"router {prefix} has an independent descendant assignment",
                [prefix, direct_descendants[0]])
        governing = [key for key in keys if key == "" or prefix == key or prefix.startswith(key + ".")]
        if len(governing) != 1:
            raise DeviceMapValidationError(
                f"router {prefix} has {len(governing)} governing assignments")
        router_governing[prefix] = {
            "entry": governing[0], "device": device_map[governing[0]]}
    return {"non_overlapping_device_map": True,
            "parameter_count": len(parameters), "router_governing_assignments": router_governing}


def router_cuda_footprint(empty_model, router_prefixes, expected_dtype):
    modules = dict(empty_model.named_modules())
    dtype_sizes = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}
    require(str(expected_dtype) in dtype_sizes,
            f"unsupported router placement dtype {expected_dtype}")
    element_bytes = dtype_sizes[str(expected_dtype)]
    classifier_bytes = 0
    correction_bias_bytes = 0
    for prefix in router_prefixes:
        router = modules[prefix]
        weight = router.classifier.weight
        bias = router.e_score_correction_bias
        require(tuple(weight.shape) == (384, 3072),
                f"{prefix}.classifier.weight has unexpected shape {tuple(weight.shape)}")
        require(tuple(bias.shape) == (384,),
                f"{prefix}.e_score_correction_bias has unexpected shape {tuple(bias.shape)}")
        classifier_bytes += int(weight.numel()) * element_bytes
        correction_bias_bytes += int(bias.numel()) * element_bytes
    return {"router_parameter_dtype": str(expected_dtype),
            "router_parameter_element_size_bytes": element_bytes,
            "router_classifier_bytes": classifier_bytes,
            "router_correction_bias_bytes": correction_bias_bytes,
            "router_cuda_bytes": classifier_bytes + correction_bias_bytes}


def infer_ordinary_device_map(infer, model, max_memory, dtype, no_split):
    """Invoke Accelerate without adding any descendant placement overrides."""
    return infer(
        model, max_memory=max_memory, dtype=dtype,
        no_split_module_classes=no_split, offload_buffers=True)


def build_auto_dispatch_plan(model_dir, dtype, cpu_memory=None, gpu_memory=None):
    """Build the pinned empty model and an ordinary non-overlapping Accelerate map."""
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import convert_file_size_to_int, find_tied_parameters
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True, dtype=dtype)
    config.torch_dtype = dtype
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    tie_weights = getattr(empty_model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    tied_parameter_groups = [list(group) for group in find_tied_parameters(empty_model)]
    router_prefixes = resolve_router_prefixes(empty_model.named_modules())
    router_classes = {type(dict(empty_model.named_modules())[prefix]).__name__
                      for prefix in router_prefixes}
    require(router_classes == {ROUTER_PRELOAD_CLASS},
            f"unexpected runtime router classes: {sorted(router_classes)}")
    footprint = router_cuda_footprint(empty_model, router_prefixes, dtype)
    requested_gpu = (convert_file_size_to_int(gpu_memory) if gpu_memory else
                     int(torch.cuda.mem_get_info(0)[0]))
    max_memory = {0: reserve_router_gpu_headroom(requested_gpu)}
    if cpu_memory:
        max_memory["cpu"] = convert_file_size_to_int(cpu_memory)
    no_split = router_no_split_classes(empty_model._get_no_split_modules("auto"))
    ordinary = infer_ordinary_device_map(
        infer_auto_device_map, empty_model, max_memory, dtype, no_split)
    validation = validate_ordinary_device_map(ordinary, empty_model, router_prefixes)
    canonical = json.dumps(ordinary, sort_keys=True, separators=(",", ":"), default=str)
    policy = {
        "placement_policy": ROUTER_PLACEMENT_POLICY,
        "router_prefixes": router_prefixes,
        "preload_module_classes": [ROUTER_PRELOAD_CLASS],
        "router_no_split_class": ROUTER_PRELOAD_CLASS,
        "no_split_module_classes": no_split,
        "offload_buffers": True,
        "tied_parameter_groups": tied_parameter_groups,
        "router_inventory_classifier_bytes": footprint["router_classifier_bytes"],
        "router_inventory_correction_bias_bytes": footprint["router_correction_bias_bytes"],
        "router_inventory_total_bytes": footprint["router_cuda_bytes"],
        "single_router_preload_bytes": footprint["router_cuda_bytes"] // 14,
        "router_parameter_dtype": footprint["router_parameter_dtype"],
        "router_parameter_element_size_bytes": footprint["router_parameter_element_size_bytes"],
        "reserved_gpu_headroom_bytes": ROUTER_GPU_HEADROOM_BYTES,
        "automatic_map_gpu_budget_bytes": max_memory[0],
        "ordinary_device_map_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "ordinary_device_map": ordinary,
        "device_map_validation": validation,
    }
    return empty_model, ordinary, policy


def _audit_router_placement(model, placement, output_path, policy=None, expected_weight_dtype=None):
    modules = dict(model.named_modules())
    prefixes = resolve_router_prefixes(modules.items())
    rows = []
    failures = []
    device_map = getattr(model, "hf_device_map", {}) or {}
    map_validation = validate_ordinary_device_map(device_map, model, prefixes)
    policy_map_matches = policy is None or policy.get("ordinary_device_map") == device_map
    preload_verified = (placement != "auto" or
                        (policy is not None and
                         policy.get("preload_module_classes") == [ROUTER_PRELOAD_CLASS] and
                         policy.get("router_no_split_class") == ROUTER_PRELOAD_CLASS))
    for logical, prefix in enumerate(prefixes):
        router = modules[prefix]
        weight = router.classifier.weight
        correction = router.e_score_correction_bias
        weight_device = getattr(weight.device, "type", str(weight.device))
        correction_device = getattr(correction.device, "type", str(correction.device))
        weight_bytes = int(weight.numel()) * int(weight.element_size())
        correction_bytes = int(correction.numel()) * int(correction.element_size())
        governing = map_validation["router_governing_assignments"][prefix]
        governing_device = governing["device"]
        parent_hook = getattr(router, "_hf_hook", None)
        child_hook = getattr(router.classifier, "_hf_hook", None)
        hook_execution = getattr(parent_hook, "execution_device", None)
        hook_offload = getattr(parent_hook, "offload", None)
        hook_place_submodules = getattr(parent_hook, "place_submodules", None)
        resident_cuda = str(governing_device) in ("0", "cuda", "cuda:0")
        resident_cpu = placement == "cpu" and str(governing_device) == "cpu"
        offloaded = str(governing_device) in ("cpu", "disk")
        resident_safe = (((resident_cuda and weight_device == "cuda" and correction_device == "cuda") or
                          (resident_cpu and weight_device == "cpu" and correction_device == "cpu")) and
                         not bool(getattr(weight, "is_meta", False)) and
                         not bool(getattr(correction, "is_meta", False)))
        offloaded_safe = (offloaded and parent_hook is not None and
                          str(hook_execution) in ("0", "cuda", "cuda:0") and
                          hook_place_submodules is True and preload_verified)
        direct_safe = resident_safe or offloaded_safe
        row = {
            "logical_layer": logical, "physical_block": 2 * logical,
            "router_module": prefix,
            "governing_device_map_entry": governing["entry"],
            "governing_device": governing_device,
            "router_class": type(router).__name__,
            "classifier_weight_device": str(weight.device),
            "classifier_weight_is_meta": bool(getattr(weight, "is_meta", False)),
            "classifier_weight_shape": list(weight.shape), "classifier_weight_dtype": str(weight.dtype),
            "classifier_weight_bytes": weight_bytes,
            "correction_bias_device": str(correction.device),
            "correction_bias_is_meta": bool(getattr(correction, "is_meta", False)),
            "correction_bias_dtype": str(correction.dtype),
            "correction_bias_bytes": correction_bytes,
            "router_total_bytes": weight_bytes + correction_bytes,
            "router_hf_hook": None if parent_hook is None else type(parent_hook).__qualname__,
            "router_execution_device": str(hook_execution),
            "router_hook_offload": hook_offload,
            "router_hook_place_submodules": hook_place_submodules,
            "classifier_hf_hook": None if child_hook is None else type(child_hook).__qualname__,
            "classifier_execution_device": str(getattr(child_hook, "execution_device", None)),
            "classifier_hook_offload": getattr(child_hook, "offload", None),
            "classifier_hook_place_submodules": getattr(child_hook, "place_submodules", None),
            "direct_weight_access_safe": direct_safe,
        }
        rows.append(row)
        if tuple(weight.shape) != (384, 3072) or tuple(correction.shape) != (384,):
            failures.append(f"{prefix}: unexpected router parameter shape")
        if expected_weight_dtype is not None and str(weight.dtype) != str(expected_weight_dtype):
            failures.append(f"{prefix}: classifier expected checkpoint dtype {expected_weight_dtype}")
        if expected_weight_dtype is not None and str(correction.dtype) != str(expected_weight_dtype):
            failures.append(f"{prefix}: correction bias expected checkpoint dtype {expected_weight_dtype}")
        if not direct_safe:
            failures.append(f"{prefix}: direct classifier-weight access is unsafe")
    if not preload_verified:
        failures.append("LongcatFlashTopkRouter preload policy is not verified")
    if not policy_map_matches:
        failures.append("dispatched device map differs from placement policy")
    report = {"schema_version": 2, "placement": placement,
              "placement_policy": None if policy is None else policy.get("placement_policy"),
              "all_router_direct_weight_access_safe": not failures,
              "preload_policy_verified": preload_verified,
              "non_overlapping_device_map": map_validation["non_overlapping_device_map"],
              "router_count": len(rows), "routers": rows, "failures": failures,
              "classifier_dtype_verified": not any("classifier expected" in row for row in failures),
              "correction_bias_dtype_verified": not any("correction bias expected" in row for row in failures)}
    v2.atomic_json(output_path, report)
    require(not failures, "invalid LongCat router placement: " + "; ".join(failures))
    return report


def audit_router_placement(model, placement, output_path, policy=None, expected_weight_dtype=None):
    output_path = Path(output_path)
    try:
        return _audit_router_placement(
            model, placement, output_path, policy, expected_weight_dtype)
    except Exception as error:
        if not output_path.exists():
            v2.atomic_json(output_path, {
                "schema_version": 2, "placement": placement,
                "placement_policy": None if policy is None else policy.get("placement_policy"),
                "all_router_direct_weight_access_safe": False, "router_count": None,
                "routers": [], "failures": [f"{type(error).__name__}: {error}"],
            })
        raise


def load_model_with_placement_audit(loader, model_dir, load_kwargs, output_dir,
                                    placement, placement_policy):
    try:
        return loader.from_pretrained(str(model_dir), **load_kwargs)
    except Exception as exc:
        write_placement_failure(
            output_dir, "checkpoint_load", placement, placement_policy, exc)
        raise CoreFixtureError(
            f"local official model loading failed without network fallback: {exc}") from exc


def dispatch_auto_model_with_preload(model, model_dir, device_map, offload_dir, dtype,
                                     output_dir, placement_policy, dispatcher=None,
                                     generation_config_class=None):
    if dispatcher is None:
        from accelerate import load_checkpoint_and_dispatch as dispatcher
    if generation_config_class is None:
        from transformers import GenerationConfig as generation_config_class
    required = {"checkpoint", "device_map", "offload_folder", "offload_state_dict",
                "offload_buffers", "dtype", "preload_module_classes"}
    signature = inspect.signature(dispatcher)
    require(required <= set(signature.parameters),
            "Accelerate load_checkpoint_and_dispatch lacks required pinned parameters: " +
            ", ".join(sorted(required - set(signature.parameters))))
    try:
        model = dispatcher(
            model, checkpoint=str(model_dir), device_map=device_map,
            offload_folder=str(offload_dir), offload_state_dict=True,
            offload_buffers=True, dtype=dtype,
            preload_module_classes=[ROUTER_PRELOAD_CLASS])
        model.generation_config = generation_config_class.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True)
        return model
    except Exception as exc:
        write_placement_failure(
            output_dir, "checkpoint_load", "auto", placement_policy, exc)
        raise CoreFixtureError(
            f"local official model loading failed without network fallback: {exc}") from exc


def tokenizer_loading_kwargs(fix_mistral_regex=False):
    require(fix_mistral_regex is False,
            "LongCat fixture generation forbids fix_mistral_regex=True; preserve tokenizer.json")
    return {"local_files_only": True, "trust_remote_code": True,
            "fix_mistral_regex": False}


def tokenizer_backend_pretokenizer_sha256(tokenizer):
    try:
        state = tokenizer.backend_tokenizer.pre_tokenizer.__getstate__()
    except (AttributeError, TypeError) as exc:
        raise CoreFixtureError(f"cannot serialize tokenizer backend pre-tokenizer state: {exc}") from exc
    if isinstance(state, bytes):
        payload = state
    elif isinstance(state, str):
        payload = state.encode("utf-8")
    else:
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tokenizer_provenance(tokenizer, model_dir):
    root = Path(model_dir)
    config = read_json(root / "tokenizer_config.json", "tokenizer config")
    declared_class = config.get("tokenizer_class")
    require(declared_class == "BloomTokenizer",
            f"pinned declared tokenizer class must be BloomTokenizer, got {declared_class!r}")
    runtime_class = tokenizer.__class__.__name__
    runtime_is_fast = getattr(tokenizer, "is_fast", None)
    require(runtime_class == "BloomTokenizerFast" and runtime_is_fast is True,
            "pinned LongCat tokenizer must execute as BloomTokenizerFast with is_fast=true; "
            f"got {runtime_class} with is_fast={runtime_is_fast!r}")
    slow_class = getattr(tokenizer, "slow_tokenizer_class", None)
    slow_class_name = getattr(slow_class, "__name__", None) if slow_class is not None else None
    if slow_class_name is not None:
        require(slow_class_name == "BloomTokenizer",
                f"BloomTokenizerFast slow_tokenizer_class must be BloomTokenizer, got {slow_class_name}")
    return {"declared_tokenizer_class": declared_class,
            "runtime_tokenizer_class": runtime_class,
            "runtime_tokenizer_is_fast": runtime_is_fast,
            "runtime_slow_tokenizer_class": slow_class_name,
            "tokenizer_source_directory": str(Path(model_dir).resolve()),
            "fix_mistral_regex": False,
            "tokenizer_config_json_sha256": file_sha256(root / "tokenizer_config.json"),
            "tokenizer_json_sha256": file_sha256(root / "tokenizer.json"),
            "backend_pre_tokenizer_state_sha256":
                tokenizer_backend_pretokenizer_sha256(tokenizer)}


def effective_dtype_provenance(model, precision, torch):
    requested = torch.bfloat16 if precision == "bf16" else torch.float16
    modules = resolve_capture_modules(model)
    embedding_dtype = modules["base_embedding"].weight.dtype
    effective = getattr(model, "dtype", embedding_dtype)
    require(embedding_dtype == requested,
            f"requested {requested} core dtype but base embedding weight uses {embedding_dtype}")
    require(effective == requested,
            f"requested {requested} core dtype but model reports effective dtype {effective}")
    return {"requested_precision": precision, "requested_torch_dtype": str(requested),
            "effective_model_dtype": str(effective),
            "base_embedding_weight_dtype": str(embedding_dtype)}


def fixture_greedy_generation_config(original, max_new_tokens):
    require(isinstance(max_new_tokens, int) and max_new_tokens >= 1,
            "greedy max_new_tokens must be at least 1")
    config = deepcopy(original)
    fields = ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens",
              "use_cache", "return_dict_in_generate")
    base = {name: getattr(original, name, None) for name in fields}
    overrides = {"use_model_defaults": False, "do_sample": False,
                 "temperature": None, "top_p": None, "top_k": None,
                 "max_new_tokens": max_new_tokens, "use_cache": True,
                 "return_dict_in_generate": True}
    for name, value in overrides.items():
        if name == "use_model_defaults":
            continue
        setattr(config, name, value)
    return config, {"copied_base_generation_configuration": base,
                    "explicit_call_overrides": overrides}


def resolve_effective_greedy_policy(model, generation_config, overrides):
    prepare = getattr(model, "_prepare_generation_config", None)
    require(callable(prepare),
            "official model does not expose _prepare_generation_config for greedy-policy validation")
    effective, _ = prepare(deepcopy(generation_config), **dict(overrides))
    fields = ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens",
              "use_cache", "return_dict_in_generate")
    resolved = {name: getattr(effective, name, None) for name in fields}
    resolved["use_model_defaults"] = overrides.get("use_model_defaults")
    require(resolved == overrides,
            "effective generation policy is not the requested greedy policy: " +
            json.dumps(resolved, sort_keys=True))
    return resolved


def generate_with_greedy_policy(model, encoded, generation_config, policy,
                                prompt_name, repeat_index, finite_checker=None,
                                attention_backend="default"):
    overrides = policy["explicit_call_overrides"]
    effective = resolve_effective_greedy_policy(model, generation_config, overrides)
    mode = "sampling" if effective["do_sample"] else "greedy"
    require(mode == "greedy", f"effective decoding method for {prompt_name} is {mode}, not greedy")
    handles = []
    try:
        if finite_checker is not None:
            import torch
            handles = v2.install_trunk_finite_hooks(
                model, finite_checker, serialize_blocks=(), capture={})
            handles.extend(v2.install_output_finite_hooks(
                model, finite_checker, "generation", prompt_name))
            sdpa_context = v2.instrument_sdpa(torch, finite_checker, attention_backend)
        else:
            import contextlib
            sdpa_context = contextlib.nullcontext()
        with sdpa_context:
            generated = model.generate(
                **encoded, generation_config=generation_config,
                output_scores=True, **overrides)
        if finite_checker is not None:
            scores = getattr(generated, "scores", None)
            require(scores is not None, "greedy generation did not return requested scores")
            for step, score in enumerate(scores):
                finite_checker.check(score, module_name="generation.scores",
                                     operation="generation", role="score",
                                     prompt=prompt_name, generation_step=step)
    except Exception as exc:
        raise CoreFixtureError(
            "generation failed; no fixture was written; "
            f"prompt={prompt_name}; repeat_index={repeat_index}; "
            f"requested_policy={json.dumps(overrides, sort_keys=True)}; "
            f"effective_decoding_mode={mode}; {type(exc).__name__}: {exc}") from exc
    finally:
        for handle in handles:
            handle.remove()
    return generated, effective


def require_prompt_ids_match(prompt_name, direct_ids, greedy_ids):
    direct = list(direct_ids)
    greedy = list(greedy_ids)
    require(greedy == direct,
            f"direct-forward and greedy tokenization differ for {prompt_name}")
    return greedy


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


def build_text_generation_context(model, model_dir):
    try:
        from transformers import GenerationConfig
        generation = GenerationConfig.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True)
        visual = GenerationConfig(**generation.visual_generation_config)
        audio = GenerationConfig(**generation.audio_generation_config)
        dynamic_module = importlib.import_module(model.__class__.__module__)
        status_class = getattr(dynamic_module, "LongcatNextForCausalLMGenerationStatus")
        status = status_class(visual, audio)
        status.switch_to("text")
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise CoreFixtureError(
            f"cannot construct official text-mode multimodal generation status: {exc}") from exc
    require(getattr(status, "mode", None) == "text",
            "official multimodal generation status did not remain in text mode")
    return status, visual, audio


def extract_greedy_sequences(result):
    sequences = None
    if hasattr(result, "sequences"):
        sequences = result.sequences
    elif isinstance(result, tuple) and len(result) == 4:
        sequences = result[0]
    else:
        raise CoreFixtureError(
            "official generate returned neither a return-dictionary with .sequences nor a four-item tuple")
    require(hasattr(sequences, "detach") and hasattr(sequences, "ndim"),
            "official greedy sequences are not a tensor")
    require(sequences.ndim == 2,
            f"official greedy sequences must have shape [batch, sequence], got rank {sequences.ndim}")
    return sequences


def call_text_forward(model, forward_kwargs, generation_context):
    status, visual_generation_config, audio_generation_config = generation_context
    require(getattr(status, "mode", None) == "text",
            "direct forward requires official multimodal generation status in text mode")
    return model(**forward_kwargs,
                 multimodal_generation_status=status,
                 visual_generation_config=visual_generation_config,
                 audio_generation_config=audio_generation_config)


def summarize_forward_logits(output, selected_logit_ids, include_complete=False, top_k=10):
    import numpy as np
    require(hasattr(output, "logits") and output.logits is not None,
            "official direct forward did not return logits")
    logits, source_dtype = tensor_array(output.logits, "final_logits")
    require(logits.ndim == 3,
            f"official logits must have shape [batch, sequence, vocabulary], got rank {logits.ndim}")
    require(logits.shape[-1] == 131125,
            f"official final logits must have 131125 entries, got {logits.shape[-1]}")
    final = logits[:, -1, :]
    order = np.argsort(final, axis=-1)[:, ::-1][:, :top_k]
    values = np.take_along_axis(final, order, axis=-1)
    arrays = {"selected_logits": final[:, selected_logit_ids],
              "topk_token_ids": order.astype(np.int64),
              "topk_values": values, "argmax_token_id": order[:, :1].astype(np.int64)}
    if include_complete:
        arrays["complete_final_position_logits"] = final
    dtypes = {name: ("int64" if name in ("topk_token_ids", "argmax_token_id") else source_dtype)
              for name in arrays}
    return arrays, dtypes


def deterministic_npz_bytes(arrays):
    import numpy as np
    v2.validate_array_inventory(arrays, error_type=CoreFixtureError)
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
        finite = v2.validate_numpy_array(name, value, CoreFixtureError)
        data = value.tobytes(order="C")
        result[name] = {"shape": list(value.shape), "serialized_dtype": str(value.dtype),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "finite": finite}
    return result


def compare_runs(runs):
    import numpy as np
    require(len(runs) >= 1, "at least one official run is required")
    baseline = runs[0]
    for index, run in enumerate(runs):
        v2.validate_array_inventory(run, error_type=CoreFixtureError)
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
    validation = v2.validate_array_inventory(arrays, error_type=CoreFixtureError)
    if metadata.get("kind") == "longcat-next-core-reference":
        contract = v2.load_accepted_contract(
            Path(__file__).with_name("core-accepted-contract-v2.json"))
        validation.update(v2.validate_accepted_arrays(
            arrays, metadata["precision"], contract, CoreFixtureError))
    npz = deterministic_npz_bytes(arrays)
    metadata = dict(metadata)
    metadata["arrays"] = array_metadata(arrays)
    metadata["whole_candidate_validation"] = validation
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

def tensor_array(tensor, name="captured_tensor"):
    import numpy as np
    if isinstance(tensor, np.ndarray):
        result = np.ascontiguousarray(tensor)
        v2.validate_numpy_array(name, result, CoreFixtureError)
        return result, str(tensor.dtype)
    value = tensor.detach().cpu()
    source_dtype = str(value.dtype).replace("torch.", "")
    if source_dtype == "bfloat16":
        value = value.float()
    result = value.contiguous().numpy()
    v2.validate_numpy_array(name, result, CoreFixtureError)
    return result, source_dtype


def load_case_ids(weight_free_fixture, tokenizer):
    corpus = read_json(weight_free_fixture, "weight-free fixture")
    cases = []
    for case in corpus.get("cases", []):
        if "input_ids" in case and case["name"] != "all_ignored_ids":
            cases.append((case["name"], list(case["input_ids"])))
    prompts = ["The capital of France is", "Write one short sentence about a cat."]
    prompt_input_ids = {}
    for index, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)
        ids = list(encoded["input_ids"])
        prompt_input_ids[f"prompt_{index}"] = ids
        cases.append((f"tokenizer_prompt_{index}", ids))
    require(cases, "no explicit token-ID cases were found")
    return cases, prompts, prompt_input_ids


def prepare_case_inputs(case_name, input_ids):
    import numpy as np
    ids = np.asarray([input_ids], dtype=np.int64)
    attention = np.ones_like(ids)
    if case_name == "bos_left_zero":
        for index, token in enumerate(input_ids):
            if token != 0:
                break
            attention[0, index] = 0
    positions = attention.cumsum(axis=-1) - 1
    positions[attention == 0] = 1
    cache_position = np.arange(ids.shape[1], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": attention,
            "position_ids": positions, "cache_position": cache_position,
            "position_id_provenance":
                "Transformers 4.57.6 GenerationMixin.prepare_inputs_for_generation cumsum-minus-one; masked positions set to 1"}


def analytical_ngram_decomposition(base, raw_projections, ignored_mask, official_fused):
    import numpy as np
    base = np.asarray(base, dtype=np.float32)
    raw = [np.asarray(value, dtype=np.float32) for value in raw_projections]
    ignored = np.asarray(ignored_mask, dtype=bool)
    reconstructed = base.copy()
    for value in raw:
        reconstructed = reconstructed + value
    reconstructed[:, ~ignored, :] /= 13.0
    contributions = []
    for value in raw:
        analytical = value.copy()
        analytical[:, ~ignored, :] /= 13.0
        contributions.append(analytical)
    official = np.asarray(official_fused, dtype=np.float32)
    error = reconstructed - official
    absolute = np.abs(error)
    denominator = np.maximum(np.abs(official), np.finfo(np.float32).tiny)
    report = {"max_absolute_error": float(absolute.max(initial=0.0)),
              "max_relative_error": float((absolute / denominator).max(initial=0.0)),
              "authority": "official fused_pre_trunk_embedding",
              "analytical_dtype": "float32",
              "is_official_captured_intermediate": False}
    return contributions, reconstructed, error, report


def capture_forward(model, input_ids, selected_logit_ids, case_name, generation_context,
                    finite_checker=None, attention_backend="default"):
    import numpy as np
    import torch
    modules = resolve_capture_modules(model)
    require(finite_checker is not None, "schema-v2 capture requires an on-device finite checker")
    captured = {}
    source_dtypes = {}
    handles = []

    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        array, dtype = tensor_array(value, name)
        captured[name] = array
        source_dtypes[name] = dtype

    # Register finite hooks before capture hooks so final norm/lm_head/model
    # outputs are rejected on-device before any CPU or NumPy conversion.
    handles.extend(v2.install_output_finite_hooks(model, finite_checker))
    handles.extend(v2.install_trunk_finite_hooks(
        model, finite_checker, serialize_blocks=(), capture={}))
    handles.append(modules["base_embedding"].register_forward_hook(
        lambda module, args, output: finite_checker.check(
            output, module_name="model.model.embed_tokens", operation="forward", role="output")))
    handles.append(modules["base_embedding"].register_forward_hook(
        lambda module, args, output: save("base_embedding", output)))
    for index, projection in enumerate(modules["ngram_projections"]):
        handles.append(projection.register_forward_hook(
            lambda module, args, output, index=index: finite_checker.check(
                output, module_name=f"model.model.ngram_embeddings.post_projs.{index}",
                operation="forward", role="output")))
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
    prepared = prepare_case_inputs(case_name, input_ids)
    device = modules["base_embedding"].weight.device
    ids = torch.tensor(prepared["input_ids"], dtype=torch.long, device=device)
    attention = torch.tensor(prepared["attention_mask"], dtype=torch.long, device=device)
    positions = torch.tensor(prepared["position_ids"], dtype=torch.long, device=device)
    cache_position = torch.tensor(prepared["cache_position"], dtype=torch.long, device=device)
    status, visual_generation_config, audio_generation_config = generation_context
    require(getattr(status, "mode", None) == "text",
            "direct forward requires official multimodal generation status in text mode")
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False), \
             v2.instrument_router_linear(torch) as router_session, \
             v2.instrument_sdpa(torch, finite_checker, attention_backend) as sdpa_observation:
            output = call_text_forward(model, {
                    "input_ids": ids.clone(), "attention_mask": attention,
                    "position_ids": positions, "cache_position": cache_position, "use_cache": False,
                    "logits_to_keep": 1, "return_dict": True}, generation_context)
        if finite_checker is not None:
            finite_checker.sdpa_observation = sdpa_observation
            finite_checker.router_materialization = v2.router_materialization_summary(router_session)
            require(finite_checker.router_materialization["router_invocation_count"] == 14 and
                    finite_checker.router_materialization["logical_layers_observed"] == list(range(14)) and
                    finite_checker.router_materialization["all_live_router_weights_materialized"] and
                    finite_checker.router_materialization["all_live_correction_biases_materialized"],
                    "complete direct forward did not materialize all fourteen LongCat routers")
        if attention_backend == "eager":
            require(not sdpa_observation["calls"],
                    "eager attention requested but the text trunk invoked SDPA")
        if attention_backend in ("sdpa-math", "sdpa-f32"):
            require(sdpa_observation["calls"],
                    f"{attention_backend} requested but the text trunk did not invoke SDPA")
    finally:
        for handle in handles:
            handle.remove()
    require(len([name for name in captured if name.startswith("ngram_projection_raw_")]) == 12,
            f"official module hooks did not capture all twelve n-gram contributions for {case_name}")
    ignored = np.asarray([(131072 <= token < 131125) for token in input_ids], dtype=bool)
    raw = [captured[f"ngram_projection_raw_{index:02d}"] for index in range(12)]
    analytical, reconstructed, error, reconstruction_report = analytical_ngram_decomposition(
        captured["base_embedding"], raw, ignored, captured["fused_pre_trunk_embedding"])
    for index, value in enumerate(analytical):
        name = f"ngram_analytical_f32_contribution_{index:02d}"
        captured[name] = value
        source_dtypes[name] = "analytical_float32_not_official_intermediate"
    captured["ngram_analytical_f32_reconstruction"] = reconstructed
    captured["ngram_analytical_f32_reconstruction_error"] = error
    captured["ngram_analytical_f32_max_absolute_error"] = np.asarray(
        [reconstruction_report["max_absolute_error"]], dtype=np.float32)
    captured["ngram_analytical_f32_max_relative_error"] = np.asarray(
        [reconstruction_report["max_relative_error"]], dtype=np.float32)
    source_dtypes.update({
        "ngram_analytical_f32_reconstruction": "analytical_float32_not_official_intermediate",
        "ngram_analytical_f32_reconstruction_error": "analytical_float32_not_official_intermediate",
        "ngram_analytical_f32_max_absolute_error": "analytical_float32_not_official_intermediate",
        "ngram_analytical_f32_max_relative_error": "analytical_float32_not_official_intermediate"})
    for block in (0, 1, 2, 27):
        require(f"physical_block_{block:02d}" in captured,
                f"official module hook for physical block {block} did not fire")
    logit_arrays, logit_dtypes = summarize_forward_logits(
        output, selected_logit_ids, case_name.startswith("tokenizer_prompt_0"))
    captured.update(logit_arrays)
    source_dtypes.update(logit_dtypes)
    captured["input_ids"] = ids.cpu().numpy()
    captured["attention_mask"] = attention.cpu().numpy()
    captured["position_ids"] = positions.cpu().numpy()
    captured["cache_position"] = cache_position.cpu().numpy()
    source_dtypes.update({"input_ids": "int64", "attention_mask": "int64",
                          "position_ids": "int64", "cache_position": "int64"})
    if finite_checker is not None:
        finite_checker.write_trace()
    return captured, source_dtypes


def run_core_worker_generation(args, weight_free_fixture, diagnostic=False):
    worker_started_utc = v2.utc_now()
    validate_core_options(args.precision, args.placement, args.repeat_count, args.max_output_bytes)
    inspection = validate_checkpoint(args.model_dir, args.hash_shards)
    enforce_offline_environment()
    dependency_report = require_dependency_preflight(
        args.runtime_profile, args.placement, args.model_dir, args.flash_wheel_path)
    if not getattr(args, "skip_source_scan", False):
        source_validation = v2.verify_and_scan_source(
            args.model_dir, args.shard_manifest, args.output_dir)
    else:
        require(diagnostic, "source scanning may only be bypassed by core-diagnose")
        source_validation = {"all_finite": None, "bypassed_for_diagnosis": True}
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
    router_element_bytes = 2
    router_classifier_bytes = 14 * 384 * 3072 * router_element_bytes
    router_correction_bias_bytes = 14 * 384 * router_element_bytes
    placement_policy = {
        "placement_policy": (ROUTER_PLACEMENT_POLICY if args.placement == "auto" else
                             f"longcat-router-{args.placement}-whole-model-v1"),
        "router_parameter_dtype": str(load["dtype"]),
        "router_parameter_element_size_bytes": router_element_bytes,
        "router_classifier_bytes": router_classifier_bytes,
        "router_correction_bias_bytes": router_correction_bias_bytes,
        "router_cuda_bytes": router_classifier_bytes + router_correction_bias_bytes,
        "reserved_gpu_headroom_bytes": 0,
        "explicit_device_map_sha256": None,
    }
    auto_model = None
    ordinary_map = None
    if args.placement == "auto":
        try:
            auto_model, ordinary_map, placement_policy = build_auto_dispatch_plan(
                args.model_dir, load["dtype"], args.cpu_memory, args.gpu_memory)
        except Exception as error:
            phase = ("device_map_validation" if isinstance(error, DeviceMapValidationError)
                     else "device_map_construction")
            v2.atomic_json(Path(args.output_dir) / "placement-preload-audit.json", {
                "schema_version": 2,
                "phase": phase,
                "placement": args.placement, "valid": False,
                "failures": [f"{type(error).__name__}: {error}"],
                "conflicting_key_pair": getattr(error, "conflicting_key_pair", None),
            })
            write_placement_failure(
                args.output_dir, phase, args.placement, placement_policy, error)
            raise CoreFixtureError(f"LongCat router preload placement failed: {error}") from error
    v2.atomic_json(Path(args.output_dir) / "placement-policy.json", placement_policy)
    v2.atomic_json(Path(args.output_dir) / "placement-preload-audit.json", {
        "schema_version": 2, "phase": "before_checkpoint_load",
        "placement": args.placement, **placement_policy,
        "router_prefix_count": len(placement_policy.get("router_prefixes", []))
        if args.placement == "auto" else 14,
    })
    if args.attention_backend != "default":
        load["attn_implementation"] = (
            "eager" if args.attention_backend == "eager" else "sdpa")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), **tokenizer_loading_kwargs(False))
    except (OSError, RuntimeError, ValueError) as exc:
        raise CoreFixtureError(f"local official tokenizer loading failed without network fallback: {exc}") from exc
    if args.placement == "auto":
        require(args.offload_dir is not None,
                "automatic LongCat placement requires --offload-dir")
        model = dispatch_auto_model_with_preload(
            auto_model, args.model_dir, ordinary_map, args.offload_dir, load["dtype"],
            args.output_dir, placement_policy)
    else:
        model = load_model_with_placement_audit(
            AutoModelForCausalLM, args.model_dir, load, args.output_dir,
            args.placement, placement_policy)
    model.eval()
    placement_audit = audit_router_placement(
        model, args.placement, Path(args.output_dir) / "placement-audit.json",
        placement_policy, load["dtype"])
    attention_provenance = v2.configure_attention_backend(model, args.attention_backend)
    dtype_provenance = effective_dtype_provenance(model, args.precision, torch)
    tokenizer_metadata = tokenizer_provenance(tokenizer, args.model_dir)
    initial_pretokenizer_sha256 = tokenizer_metadata["backend_pre_tokenizer_state_sha256"]
    cases, prompts, prompt_input_ids = load_case_ids(weight_free_fixture, tokenizer)
    requested_cases = list(getattr(args, "case", []) or [])
    if requested_cases:
        known = {name for name, _ in cases}
        require(set(requested_cases) <= known,
                f"unknown requested cases: {sorted(set(requested_cases) - known)}")
        cases = [row for row in cases if row[0] in requested_cases]
    tokenizer_metadata["prompts"] = [
        {"name": f"prompt_{index}", "text": prompt,
         "input_ids": prompt_input_ids[f"prompt_{index}"]}
        for index, prompt in enumerate(prompts)]
    greedy_generation_config, greedy_policy = fixture_greedy_generation_config(
        model.generation_config, args.max_new_tokens)
    selected_logit_ids = sorted({0, 1, 2, 131071, *range(131072, 131125)})
    runs = []
    source_dtypes = {}
    greedy_by_run = []
    sdpa_observations = []
    router_materialization_observations = []
    for repeat in range(args.repeat_count):
        arrays = {}
        repeat_sources = {}
        for case_name, ids in cases:
            generation_context = build_text_generation_context(model, args.model_dir)
            checker = v2.TorchFiniteChecker(
                Path(args.output_dir) / "diagnostics" / case_name,
                case_name, args.attention_backend, getattr(args, "run_index", 0))
            case_arrays, case_sources = capture_forward(
                model, ids, selected_logit_ids, case_name, generation_context,
                checker, args.attention_backend)
            sdpa_observations.append({"case": case_name, "run_index": repeat,
                                      **checker.sdpa_observation})
            router_materialization_observations.append({
                "case": case_name, "run_index": repeat,
                **checker.router_materialization})
            for name, value in case_arrays.items():
                arrays[f"{case_name}/{name}"] = value
                repeat_sources[f"{case_name}/{name}"] = case_sources[name]
        greedy = {}
        for index, prompt in ([] if diagnostic else enumerate(prompts)):
            encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
            greedy_input_ids = encoded["input_ids"].detach().cpu().tolist()[0]
            require_prompt_ids_match(
                f"prompt_{index}", prompt_input_ids[f"prompt_{index}"], greedy_input_ids)
            device = resolve_capture_modules(model)["base_embedding"].weight.device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generation_checker = v2.TorchFiniteChecker(
                Path(args.output_dir) / "diagnostics" / f"generation-prompt-{index}",
                f"generation_prompt_{index}", args.attention_backend,
                getattr(args, "run_index", 0))
            with torch.inference_mode():
                generated, effective_policy = generate_with_greedy_policy(
                    model, encoded, greedy_generation_config, greedy_policy,
                    f"prompt_{index}", repeat, generation_checker,
                    args.attention_backend)
            generation_checker.write_trace()
            if "effective_resolved_greedy_policy" not in greedy_policy:
                greedy_policy["effective_resolved_greedy_policy"] = effective_policy
            else:
                require(greedy_policy["effective_resolved_greedy_policy"] == effective_policy,
                        "effective greedy policy changed between prompts or repeats")
            result = extract_greedy_sequences(generated).detach().cpu().numpy()
            arrays[f"greedy_ids/prompt_{index}"] = result
            repeat_sources[f"greedy_ids/prompt_{index}"] = "int64"
            greedy[f"prompt_{index}"] = result.tolist()
        runs.append(arrays)
        greedy_by_run.append(greedy)
        if repeat == 0:
            source_dtypes = repeat_sources
    require(tokenizer_backend_pretokenizer_sha256(tokenizer) == initial_pretokenizer_sha256,
            "tokenizer backend pre-tokenizer changed during fixture generation")
    reproducibility = {"schema_version": CORE_SCHEMA_VERSION, "repeat_count": args.repeat_count,
                       "arrays": compare_runs(runs), "greedy_continuations": greedy_by_run,
                       "comparison_tolerances": {"bf16": None, "f16": None},
                       "tolerances_selected_automatically": False,
                       "runtime_profile": args.runtime_profile,
                       "runtime_departures_from_official": [
                           {"package": name, "official": row["official_pinned_version"],
                            "installed": row["installed_version"]}
                           for name, row in dependency_report["packages"].items()
                           if row["official_pinned_version"] is not None and
                           row["installed_version"] != row["official_pinned_version"]]}
    reconstruction_reports = {}
    for name, value in runs[0].items():
        suffix = "/ngram_analytical_f32_max_absolute_error"
        if name.endswith(suffix):
            case = name[:-len(suffix)]
            relative = runs[0][case + "/ngram_analytical_f32_max_relative_error"]
            reconstruction_reports[case] = {
                "max_absolute_error": float(value.reshape(-1)[0]),
                "max_relative_error": float(relative.reshape(-1)[0]),
                "authority": case + "/fused_pre_trunk_embedding",
                "analytical_dtype": "float32",
                "is_official_captured_intermediate": False}
    provenance = v2.collect_provenance(
        Path(__file__).resolve().parents[2], args.model_dir, args.precision,
        args.attention_backend, getattr(args, "run_index", 0),
        [Path(__file__), Path(__file__).with_name("core_reference_v2.py"),
         Path(__file__).with_name("make-reference-fixtures.py"),
         Path(__file__).with_name("checkpoint-shards-v2.json"),
         Path(__file__).with_name("core-accepted-contract-v2.json")], model,
        {"effective_precision": dtype_provenance["effective_model_dtype"],
         "effective_attention_backend": attention_provenance,
         "shard_manifest_sha256": v2.sha256_file(args.shard_manifest),
         "flash_attention_wheel_identity": dependency_report["flash_attention"],
         "autocast_during_direct_forward": False,
         "local_monkey_patches": ["scoped SDPA finite instrumentation",
                                  "scoped router F.linear finite instrumentation"],
         "placement_policy": placement_policy,
         "start_utc": worker_started_utc, "end_utc": None})
    metadata = {"schema_version": CORE_SCHEMA_VERSION,
                "kind": ("longcat-next-core-diagnostic" if diagnostic else
                         "longcat-next-core-reference"),
                "precision": args.precision, "serialized_activation_dtype": "float32 except integer arrays",
                "runtime_profile": args.runtime_profile,
                "source_dtypes": source_dtypes,
                "checkpoint": {**inspection, "repository": v2.PINNED_MODEL_REPOSITORY,
                               "revision": v2.PINNED_MODEL_REVISION},
                "software_versions": {"transformers": EXPECTED_TRANSFORMERS,
                                      "torch": torch.__version__},
                "dependency_preflight": dependency_report,
                "placement_policy": placement_policy,
                "placement_audit": placement_audit,
                "source_weight_validation": source_validation,
                "attention_backend": attention_provenance,
                "sdpa_observations": sdpa_observations,
                "router_live_materialization": router_materialization_observations,
                "provenance": provenance,
                "tokenizer": tokenizer_metadata,
                "dtype_provenance": dtype_provenance,
                "fixture_generation_settings": greedy_policy,
                "seeds": {"python": 20260725, "torch": 20260725},
                "selected_logit_token_ids": selected_logit_ids,
                "ngram_analytical_reconstruction_reports": reconstruction_reports,
                "module_anchors": {
                    "base_embedding": "model.model.embed_tokens forward output",
                    "ngram_projection_raw": "model.model.ngram_embeddings.post_projs[0..11] directly captured outputs",
                    "ngram_analytical_f32_contributions": "float32 analytical decomposition derived from raw projections; not an official captured intermediate",
                    "fused_pre_trunk_embedding": "input to model.model.layers[0]; directly captured parity authority",
                    "physical_blocks_0_2": "inputs to logical layers 0/1 input_layernorm[1]",
                    "physical_blocks_1_27": "outputs of logical layers 0/13",
                    "final_norm": "model.model.norm output",
                    "logits": "LongcatNextForCausalLM logits_to_keep=1 output"}}
    stem = (f"longcat-next-core-diagnostic-{args.precision}" if diagnostic else
            f"longcat-next-core-{args.precision}")
    return write_core_outputs(args.output_dir, stem, runs[0], metadata,
                              reproducibility, args.max_output_bytes)


def run_core_worker(args, weight_free_fixture):
    """Produce one finite staged run. This function never promotes a candidate."""
    require(args.repeat_count == 1, "core-worker executes exactly one run")
    require(args.attention_backend != "sdpa-f32",
            "diagnostic sdpa-f32 is forbidden for accepted workers")
    paths = run_core_worker_generation(args, weight_free_fixture)
    target = Path(args.output_dir) / "arrays.npz"
    paths["npz"].replace(target)
    paths["metadata"].replace(Path(args.output_dir) / "metadata.json")
    paths["reproducibility"].replace(Path(args.output_dir) / "worker-reproducibility.json")
    metadata_path = Path(args.output_dir) / "metadata.json"
    contract = v2.load_accepted_contract(Path(__file__).with_name("core-accepted-contract-v2.json"))
    arrays, validation = v2.load_worker_arrays(
        args.output_dir, precision=args.precision, contract=contract)
    v2.atomic_json(Path(args.output_dir) / "validation.json", validation)
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    metadata["provenance"]["end_utc"] = v2.utc_now()
    v2.atomic_json(metadata_path, metadata)
    return {"arrays": target, "metadata": Path(args.output_dir) / "metadata.json"}


def run_core_diagnose(args, weight_free_fixture):
    """Run selected cases once, skip generation, and always retain a report."""
    if not args.case:
        args.case = list(v2.DEFAULT_DIAGNOSTIC_CASES)
    args.repeat_count = 1
    started = v2.utc_now()
    try:
        paths = run_core_worker_generation(args, weight_free_fixture, diagnostic=True)
        report = {"schema_version": 2, "workflow": "core-diagnose", "accepted": False,
                  "cases": args.case, "attention_backend": args.attention_backend,
                  "start_utc": started, "end_utc": v2.utc_now(), "all_finite": True}
        return paths
    except Exception as exc:
        report = {"schema_version": 2, "workflow": "core-diagnose", "accepted": False,
                  "cases": args.case, "attention_backend": args.attention_backend,
                  "start_utc": started, "end_utc": v2.utc_now(), "all_finite": False,
                  "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        v2.atomic_json(Path(args.output_dir) / "diagnostic-report.json", report)


def run_core_generation(args, weight_free_fixture):
    """Acceptance parent: start independent worker processes then atomically promote."""
    require(args.repeat_count >= 2, "core acceptance requires at least two independent processes")
    require(args.attention_backend != "sdpa-f32",
            "diagnostic sdpa-f32 is forbidden for accepted candidates")
    require(Path(args.output_dir).name != "bf16-candidate-15b7fe8c",
            "refusing to overwrite the rejected schema-v1 candidate")
    expected_name = re.compile(
        rf"{args.precision}-candidate-[0-9a-f]{{7,40}}-{re.escape(args.attention_backend)}-v2")
    require(expected_name.fullmatch(Path(args.output_dir).name) is not None,
            "schema-v2 candidate directory must be named "
            f"{args.precision}-candidate-<generator-commit>-{args.attention_backend}-v2")
    script = Path(__file__).with_name("make-reference-fixtures.py")
    commands = []
    for index in range(args.repeat_count):
        command = [sys.executable, str(script), "--mode", "core-worker",
                   "--model-dir", str(args.model_dir), "--output-dir", "{run_dir}",
                   "--precision", args.precision, "--placement", args.placement,
                   "--runtime-profile", args.runtime_profile, "--repeat-count", "1",
                   "--run-index", str(index), "--attention-backend", args.attention_backend,
                   "--shard-manifest", str(args.shard_manifest),
                   "--max-output-bytes", str(args.max_output_bytes),
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--cpu-memory", args.cpu_memory, "--gpu-memory", args.gpu_memory]
        if args.offload_dir:
            command += ["--offload-dir", str(args.offload_dir)]
        if args.flash_wheel_path:
            command += ["--flash-wheel-path", str(args.flash_wheel_path)]
        commands.append(command)
    implementations = [Path(__file__), Path(__file__).with_name("core_reference_v2.py"),
                       script, Path(__file__).with_name("checkpoint-shards-v2.json"),
                       Path(__file__).with_name("core-accepted-contract-v2.json")]
    return v2.orchestrate_workers(
        args.output_dir, commands, args.precision, args.attention_backend,
        Path(__file__).with_name("core-accepted-contract-v2.json"), implementations)
