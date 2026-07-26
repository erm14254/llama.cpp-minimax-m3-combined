#!/usr/bin/env python3
"""Schema-v2 safety, diagnosis, and independent-worker helpers.

This module is deliberately importable without Torch. Runtime-only facilities
import Torch/Safetensors lazily so all orchestration and validation tests remain
checkpoint-free.
"""

import contextlib
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import itertools
import contextvars
from pathlib import Path

SCHEMA_VERSION = 2
EXPECTED_ACCEPTED_ARRAYS = 433
PINNED_MODEL_REPOSITORY = "meituan-longcat/LongCat-Next"
PINNED_MODEL_REVISION = "0cf0631862402ff36366e513e4023d22e7e5c84c"
ATTENTION_BACKENDS = ("default", "eager", "sdpa-math", "sdpa-f32")
DEFAULT_DIAGNOSTIC_CASES = (
    "eos_window_position_0",
    "prompt_at_once_vs_token_at_a_time",
)
CURRENT_ATTENTION_CONTEXT = contextvars.ContextVar("longcat_attention_context", default=None)
PROVENANCE_REQUIRED_FIELDS = frozenset({
    "generator_schema_version", "script_sha256", "llama_cpp", "checkpoint",
    "python", "operating_system", "torch", "cuda", "packages", "environment",
    "requested_precision", "requested_attention_backend", "model_hf_device_map",
    "observed_devices", "observed_dtypes", "autocast_enabled",
    "deterministic_algorithms", "process_id", "run_index", "start_utc", "end_utc",
    "effective_precision", "effective_attention_backend", "shard_manifest_sha256",
    "autocast_during_direct_forward", "local_monkey_patches",
    "sdpa_backend_controls",
})


class V2Error(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise V2Error(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    temporary.replace(path)


PINNED_TRANSFORMERS_VERSION = "4.57.6"
PINNED_ROUTER_REPLAY_VARIANT = "transformers-4.57.6-unconditional-scale"


def router_runtime_identity(module):
    """Pin router replay to the installed Transformers implementation."""
    runtime_module = type(module).__module__
    runtime_class = type(module).__qualname__
    version = importlib.metadata.version("transformers")
    require(version == PINNED_TRANSFORMERS_VERSION,
            f"router replay requires Transformers {PINNED_TRANSFORMERS_VERSION}, got {version}")
    require(runtime_module == "transformers.models.longcat_flash.modeling_longcat_flash",
            f"unexpected LongCat router module: {runtime_module}")
    source_module = importlib.import_module(runtime_module)
    source_path = Path(source_module.__file__).resolve()
    require(source_path.name == "modeling_longcat_flash.py",
            f"unexpected LongCat router source: {source_path}")
    return {
        "runtime_router_class": runtime_class,
        "runtime_router_module": runtime_module,
        "transformers_version": version,
        "runtime_router_source_path": str(source_path),
        "runtime_router_source_sha256": sha256_file(source_path),
        "replay_variant": PINNED_ROUTER_REPLAY_VARIANT,
    }


def numpy_finite_report(name, value):
    import numpy as np
    array = np.asarray(value)
    report = {"name": name, "shape": list(array.shape), "dtype": str(array.dtype),
              "total_elements": int(array.size), "finite_count": int(array.size),
              "nan_count": 0, "positive_infinity_count": 0,
              "negative_infinity_count": 0, "first_affected_indices": []}
    if array.dtype.kind != "f":
        return report
    finite = np.isfinite(array)
    report.update({
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(array).sum()),
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "negative_infinity_count": int(np.isneginf(array).sum()),
        "first_affected_indices": np.argwhere(~finite)[:8].tolist(),
    })
    if finite.any():
        report["finite_minimum"] = float(array[finite].min())
        report["finite_maximum"] = float(array[finite].max())
    else:
        report["finite_minimum"] = None
        report["finite_maximum"] = None
    return report


def torch_finite_report(torch, name, tensor):
    """Describe a Torch tensor without narrowing BF16 or retaining a copy."""
    total = int(tensor.numel())
    report = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
              "total_elements": total, "finite_count": total, "nan_count": 0,
              "positive_infinity_count": 0, "negative_infinity_count": 0,
              "first_affected_indices": []}
    if not tensor.is_floating_point():
        if total:
            report["finite_minimum"] = float(tensor.min().item())
            report["finite_maximum"] = float(tensor.max().item())
            report["finite_absolute_maximum"] = float(tensor.abs().max().item())
        else:
            report["finite_minimum"] = report["finite_maximum"] = None
            report["finite_absolute_maximum"] = None
        return report
    finite = torch.isfinite(tensor)
    report.update({
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "positive_infinity_count": int(torch.isposinf(tensor).sum().item()),
        "negative_infinity_count": int(torch.isneginf(tensor).sum().item()),
        "first_affected_indices": torch.nonzero(~finite, as_tuple=False)[:8].cpu().tolist(),
    })
    if report["finite_count"]:
        values = tensor[finite]
        report["finite_minimum"] = float(values.min().item())
        report["finite_maximum"] = float(values.max().item())
        report["finite_absolute_maximum"] = float(values.abs().max().item())
    else:
        report["finite_minimum"] = report["finite_maximum"] = None
        report["finite_absolute_maximum"] = None
    return report


def validate_numpy_array(name, value, error_type=V2Error):
    report = numpy_finite_report(name, value)
    if report["finite_count"] != report["total_elements"]:
        raise error_type(
            f"non-finite floating array {name}: nan={report['nan_count']} "
            f"+inf={report['positive_infinity_count']} "
            f"-inf={report['negative_infinity_count']}")
    return report


def validate_array_inventory(arrays, expected=None, error_type=V2Error):
    reports = {name: validate_numpy_array(name, value, error_type)
               for name, value in sorted(arrays.items())}
    if expected is not None and len(arrays) != expected:
        raise error_type(f"accepted inventory must contain exactly {expected} arrays, got {len(arrays)}")
    return {"array_count": len(arrays), "expected_array_count": expected,
            "all_finite": True, "arrays": reports}


def load_accepted_contract(path):
    contract = json.loads(Path(path).read_text(encoding="ascii"))
    require(contract.get("schema_version") == 2, "accepted contract schema must be 2")
    require(contract.get("expected_array_count") == EXPECTED_ACCEPTED_ARRAYS,
            "accepted contract count must remain 433")
    require(len(contract.get("direct_cases", [])) == 10, "accepted contract needs ten direct cases")
    return contract


def expected_array_contract(precision, contract):
    require(precision in ("bf16", "f16"), "contract precision must be bf16 or f16")
    activation_dtype = "float32" if precision == "bf16" else "float16"
    expected = {}
    for case in contract["direct_cases"]:
        prefix = case + "/"
        for name in ("input_ids", "attention_mask", "position_ids"):
            expected[prefix + name] = {"dtype": "int64", "shape": (1, "tokens")}
        expected[prefix + "cache_position"] = {"dtype": "int64", "shape": ("tokens",)}
        for name in contract["hidden_surfaces"]:
            expected[prefix + name] = {"dtype": activation_dtype, "shape": (1, "tokens", 3072)}
        for index in range(12):
            expected[prefix + f"ngram_projection_raw_{index:02d}"] = {
                "dtype": activation_dtype, "shape": (1, "tokens", 3072)}
            expected[prefix + f"ngram_analytical_f32_contribution_{index:02d}"] = {
                "dtype": "float32", "shape": (1, "tokens", 3072)}
        expected[prefix + "ngram_analytical_f32_reconstruction"] = {
            "dtype": "float32", "shape": (1, "tokens", 3072)}
        expected[prefix + "ngram_analytical_f32_reconstruction_error"] = {
            "dtype": "float32", "shape": (1, "tokens", 3072)}
        for name in ("ngram_analytical_f32_max_absolute_error",
                     "ngram_analytical_f32_max_relative_error"):
            expected[prefix + name] = {"dtype": "float32", "shape": (1,)}
        expected[prefix + "selected_logits"] = {
            "dtype": activation_dtype, "shape": (1, contract["selected_logit_count"])}
        expected[prefix + "topk_token_ids"] = {"dtype": "int64", "shape": (1, contract["top_k"])}
        expected[prefix + "topk_values"] = {"dtype": activation_dtype, "shape": (1, contract["top_k"])}
        expected[prefix + "argmax_token_id"] = {"dtype": "int64", "shape": (1, 1)}
        if case == contract["complete_logits_case"]:
            expected[prefix + "complete_final_position_logits"] = {
                "dtype": activation_dtype, "shape": (1, contract["vocabulary_size"])}
    for name in contract["greedy_arrays"]:
        expected[name] = {"dtype": "int64", "shape": (1, "prompt_plus_generated")}
    require(len(expected) == EXPECTED_ACCEPTED_ARRAYS,
            f"internal accepted contract generated {len(expected)} names, not 433")
    return expected


def validate_accepted_arrays(arrays, precision, contract, error_type=V2Error):
    expected = expected_array_contract(precision, contract)
    require(set(arrays) == set(expected),
            f"accepted array names differ: missing={sorted(set(expected)-set(arrays))} "
            f"unexpected={sorted(set(arrays)-set(expected))}")
    token_counts = {}
    for case in contract["direct_cases"]:
        token_counts[case] = int(arrays[f"{case}/input_ids"].shape[1])
    for name, rule in expected.items():
        value = arrays[name]
        require(str(value.dtype) == rule["dtype"],
                f"{name}: dtype {value.dtype} != {rule['dtype']}")
        case = name.split("/", 1)[0]
        resolved = []
        for dim in rule["shape"]:
            if dim == "tokens":
                dim = token_counts[case]
            elif dim == "prompt_plus_generated":
                prompt = "tokenizer_prompt_" + name.rsplit("_", 1)[-1]
                dim = token_counts[prompt] + contract["generated_tokens"]
            resolved.append(dim)
        require(tuple(value.shape) == tuple(resolved),
                f"{name}: shape {tuple(value.shape)} != {tuple(resolved)}")
        validate_numpy_array(name, value, error_type)
    return {"all_finite": True, "array_count": len(arrays), "contract_valid": True}


class TorchFiniteChecker:
    """Recursively validates Torch values and atomically records first failure."""

    def __init__(self, report_dir, case, backend, run_index=0, torch_module=None):
        self.report_dir = Path(report_dir)
        self.case = case
        self.backend = backend
        self.run_index = run_index
        self.sequence = 0
        self.checks = []
        self.first_nonfinite = None
        self.torch_module = torch_module

    def _trace_payload(self):
        return {"schema_version": SCHEMA_VERSION, "case": self.case,
                "attention_backend": self.backend, "run_index": self.run_index,
                "checks": self.checks, "first_nonfinite": self.first_nonfinite}

    def _persist_failure(self, report):
        self.first_nonfinite = report
        atomic_json(self.report_dir / "first-nonfinite.json", report)
        atomic_json(self.report_dir / "finite-trace.json", self._trace_payload())

    def _tensor_report(self, tensor, context, abort=True):
        if self.torch_module is None:
            import torch
        else:
            torch = self.torch_module
        self.sequence += 1
        counts = torch_finite_report(torch, context.get("module_name") or "tensor", tensor)
        total = counts["total_elements"]
        report = {
            "schema_version": SCHEMA_VERSION, "check_sequence": self.sequence,
            "case": self.case, "logical_layer": context.get("logical_layer"),
            "physical_block": context.get("physical_block"),
            "module_name": context.get("module_name"),
            "operation": context.get("operation"), "backend": self.backend,
            "role": context.get("role"), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "shape": list(tensor.shape),
            "total_elements": total, "finite_count": total, "nan_count": 0,
            "positive_infinity_count": 0, "negative_infinity_count": 0,
            "first_affected_indices": [], "active_attention_backend": self.backend,
            "run_index": self.run_index, "process_id": os.getpid(),
            "prompt": context.get("prompt"), "generation_step": context.get("generation_step"),
        }
        report.update({key: value for key, value in counts.items() if key not in ("name", "shape", "dtype")})
        self.checks.append(report)
        if abort and report["finite_count"] != total:
            self._persist_failure(report)
            raise V2Error(
                f"first non-finite tensor: {report['module_name']} "
                f"physical_block={report['physical_block']} role={report['role']}")
        return report

    def inspect_tensor(self, tensor, **context):
        return self._tensor_report(tensor, context, abort=False)

    def fail_report(self, report, message=None):
        self._persist_failure(report)
        raise V2Error(message or
                      f"first non-finite tensor: {report['module_name']} "
                      f"physical_block={report['physical_block']} role={report['role']}")

    def check(self, value, **context):
        if self.torch_module is None:
            import torch
        else:
            torch = self.torch_module
        if torch.is_tensor(value):
            self._tensor_report(value, context)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                self.check(child, **{**context, "role": f"{context.get('role')}.{key}"})
            return
        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                self.check(child, **{**context, "role": f"{context.get('role')}[{index}]"})
            return
        # Transformers ModelOutput behaves like a mapping, but custom outputs
        # may expose only to_tuple(). Avoid traversing arbitrary module state.
        to_tuple = getattr(value, "to_tuple", None)
        if callable(to_tuple):
            self.check(to_tuple(), **context)

    def write_trace(self):
        atomic_json(self.report_dir / "finite-trace.json", self._trace_payload())


def install_trunk_finite_hooks(model, checker, serialize_blocks=(0, 1, 2, 27), capture=None):
    """Install ordered module hooks and all 28 physical-block checks."""
    capture = capture if capture is not None else {}
    trunk = model.model
    layers = list(trunk.layers)
    require(len(layers) == 14, f"expected 14 logical layers, got {len(layers)}")
    handles = []

    attention_tokens = {}
    router_inputs = {}

    def router_context(name):
        import re
        match = re.fullmatch(r"layers\.(\d+)\.mlp\.router", name)
        if not match:
            return None
        logical = int(match.group(1))
        return logical, 2 * logical, "model.model." + name

    def router_pre(name, logical, physical):
        def hook(module, args):
            require(args, f"{name} router forward-pre has no input")
            router_inputs.setdefault(id(module), []).append(args[0])
            checker.check(args[0], module_name=name, logical_layer=logical,
                          physical_block=physical, operation="router_input", role="router_input")
        return hook

    def router_post(name, logical, physical):
        def hook(module, args, output):
            hidden_states = router_inputs[id(module)].pop()
            if output is None:
                return
            require(isinstance(output, (tuple, list)) and len(output) == 2,
                    f"{name} must return (topk_indices, topk_weights)")
            topk_indices, original_weights = output
            checker.check(topk_indices, module_name=name, logical_layer=logical,
                          physical_block=physical, operation="router_output", role="topk_indices")
            original_report = checker.inspect_tensor(
                original_weights, module_name=name, logical_layer=logical,
                physical_block=physical, operation="router_output", role="topk_weights")
            if original_report["finite_count"] == original_report["total_elements"]:
                return

            if checker.torch_module is None:
                import torch
            else:
                torch = checker.torch_module
            common = {"module_name": name, "logical_layer": logical,
                      "physical_block": physical}
            replay = []
            reasons = []
            try:
                identity = router_runtime_identity(module)
                settings = {
                    "schema_version": SCHEMA_VERSION, "case": checker.case,
                    "logical_layer": logical, "physical_block": physical,
                    "module_name": name, "operation": "router_configuration",
                    "role": "configuration", "top_k": int(module.top_k),
                    "n_routed_experts": int(module.n_routed_experts),
                    "routed_scaling_factor": float(module.routed_scaling_factor),
                    "router_bias": bool(module.router_bias),
                    "classifier_bias_present": getattr(module.classifier, "bias", None) is not None,
                    "config_hidden_size": int(module.config.hidden_size),
                    "norm_topk_prob_attribute_present": hasattr(module, "norm_topk_prob"),
                    "process_id": os.getpid(), "run_index": checker.run_index,
                    **identity,
                }
                require(not settings["norm_topk_prob_attribute_present"],
                        f"{name}: pinned router unexpectedly defines norm_topk_prob")
                checker.checks.append(settings)

                def inspect(value, operation, role=None):
                    report = checker.inspect_tensor(
                        value, **common, operation=operation, role=role or operation)
                    replay.append(report)
                    return value

                router_input = inspect(hidden_states, "router_input")
                weight = inspect(module.classifier.weight, "classifier_weight")
                bias = inspect(module.e_score_correction_bias, "e_score_correction_bias")
                flattened = router_input.view(-1, module.config.hidden_size)
                # Transformers 4.57.6 omits classifier.bias even when router_bias is true.
                router_logits = inspect(torch.nn.functional.linear(
                    flattened.type(torch.float32), weight.type(torch.float32)), "router_logits")
                scores = inspect(router_logits.softmax(dim=-1), "softmax_scores")
                scores_for_choice = inspect(
                    scores.view(-1, module.n_routed_experts) + bias.unsqueeze(0),
                    "scores_for_choice")
                require(int(module.top_k) == int(topk_indices.shape[-1]),
                        f"{name}: configured top_k differs from original output")
                replay_indices = inspect(torch.topk(
                    scores_for_choice, k=module.top_k, dim=-1, sorted=False)[1],
                    "topk_indices")
                gathered = inspect(scores.gather(1, replay_indices), "topk_weights_gathered")
                replay_final = inspect(
                    gathered * module.routed_scaling_factor, "topk_weights_scaled")

                if not torch.equal(replay_indices, topk_indices):
                    reasons.append("indices_order")
                if tuple(replay_final.shape) != tuple(original_weights.shape):
                    reasons.append("shape")
                if replay_final.dtype != original_weights.dtype:
                    reasons.append("dtype")
                if not reasons:
                    original_finite = torch.isfinite(original_weights)
                    replay_finite = torch.isfinite(replay_final)
                    if not torch.equal(original_finite, replay_finite):
                        reasons.append("finite_mask")
                    classification_equal = (
                        torch.equal(torch.isnan(original_weights), torch.isnan(replay_final)) and
                        torch.equal(torch.isposinf(original_weights), torch.isposinf(replay_final)) and
                        torch.equal(torch.isneginf(original_weights), torch.isneginf(replay_final)))
                    if not classification_equal:
                        reasons.append("nonfinite_classification")
                    if torch.equal(original_finite, replay_finite):
                        if bool(original_finite.all().item()):
                            left_bytes = original_weights.detach().contiguous().view(torch.uint8)
                            right_bytes = replay_final.detach().contiguous().view(torch.uint8)
                            if not torch.equal(left_bytes, right_bytes):
                                reasons.append("finite_values")
                        elif not torch.equal(original_weights[original_finite],
                                             replay_final[replay_finite]):
                            reasons.append("finite_values")
            except Exception as error:
                checker.checks.append({
                    "schema_version": SCHEMA_VERSION, "case": checker.case,
                    "logical_layer": logical, "physical_block": physical,
                    "module_name": name, "operation": "router_instrumentation_error",
                    "role": "instrumentation", "attribution_status": "replay_failed",
                    "exception_type": type(error).__name__, "exception_message": str(error),
                    "process_id": os.getpid(), "run_index": checker.run_index})
                checker.fail_report(
                    original_report,
                    f"{name}: original router output was non-finite but internal attribution failed")
            if reasons:
                checker.checks.append({
                    "schema_version": SCHEMA_VERSION, "case": checker.case,
                    "logical_layer": logical, "physical_block": physical,
                    "module_name": name, "operation": "router_replay_mismatch",
                    "role": "replay_mismatch", "reasons": reasons,
                    "process_id": os.getpid(), "run_index": checker.run_index})
            first_invalid = next((row for row in replay
                                  if row["finite_count"] != row["total_elements"]), None)
            if first_invalid is None or reasons:
                checker.fail_report(
                    first_invalid or original_report,
                    f"{name}: router diagnostic replay mismatch; refusing false attribution")
            checker.fail_report(first_invalid)
        return hook

    def derive_context(name):
        import re
        shortcut = re.match(r"layers\.(\d+)\.mlp(?:\.|$)", name)
        if shortcut:
            logical = int(shortcut.group(1))
            return logical, 2 * logical
        match = re.match(
            r"layers\.(\d+)(?:\.(?:self_attn|mlps|input_layernorm)\.(\d+))?", name)
        if not match:
            return None, None
        logical = int(match.group(1))
        physical = 2 * logical + int(match.group(2)) if match.group(2) is not None else None
        return logical, physical

    def pre(name, logical=None, physical=None, attention=False):
        def hook(module, args):
            if attention:
                token = CURRENT_ATTENTION_CONTEXT.set({
                    "logical_layer": logical, "physical_block": physical,
                    "module_name": name, "sdpa_call_index": 0})
                attention_tokens.setdefault(id(module), []).append(token)
            try:
                checker.check(args, module_name=name, logical_layer=logical,
                              physical_block=physical, operation="forward-pre", role="input")
            except Exception:
                if attention:
                    CURRENT_ATTENTION_CONTEXT.reset(attention_tokens[id(module)].pop())
                raise
        return hook

    def post(name, logical=None, physical=None, attention=False):
        def hook(module, args, output):
            try:
                checker.check(output, module_name=name, logical_layer=logical,
                              physical_block=physical, operation="forward", role="output")
                if physical in serialize_blocks:
                    value = output[0] if isinstance(output, tuple) else output
                    capture[f"physical_block_{physical:02d}"] = value
            finally:
                if attention and attention_tokens.get(id(module)):
                    CURRENT_ATTENTION_CONTEXT.reset(attention_tokens[id(module)].pop())
        return hook

    def physical_pre(name, logical, physical):
        def hook(module, args):
            checker.check(args, module_name=name, logical_layer=logical,
                          physical_block=physical, operation="physical-block", role="input")
            if physical in serialize_blocks:
                capture[f"physical_block_{physical:02d}"] = args[0]
        return hook

    for logical, layer in enumerate(layers):
        even = 2 * logical
        odd = even + 1
        norms = list(layer.input_layernorm)
        require(len(norms) == 2, f"logical layer {logical} does not expose two physical norms")
        # Input to the second physical norm is the even physical block output.
        handles.append(norms[1].register_forward_pre_hook(
            physical_pre(f"model.model.layers.{logical}.physical_block_{even:02d}", logical, even)))
        handles.append(layer.register_forward_hook(
            post(f"model.model.layers.{logical}.physical_block_{odd:02d}", logical, odd)))

    # Complete ordered text-trunk module coverage. Physical boundaries above
    # provide stable block identities; these hooks localize inside each block.
    for name, module in trunk.named_modules():
        if not name or not name.startswith("layers."):
            continue
        fq_name = "model.model." + name
        router = router_context(name)
        if router is not None:
            logical, physical, fq_name = router
            handles.append(module.register_forward_pre_hook(
                router_pre(fq_name, logical, physical)))
            hook = router_post(fq_name, logical, physical)
            try:
                handles.append(module.register_forward_hook(hook, always_call=True))
            except TypeError:
                handles.append(module.register_forward_hook(hook))
            continue
        logical, physical = derive_context(name)
        is_attention = physical is not None and name.endswith(f"self_attn.{physical % 2}")
        handles.append(module.register_forward_pre_hook(
            pre(fq_name, logical, physical, is_attention)))
        hook = post(fq_name, logical, physical, is_attention)
        try:
            handles.append(module.register_forward_hook(hook, always_call=is_attention))
        except TypeError:  # compatibility with older Torch hook APIs
            handles.append(module.register_forward_hook(hook))
    return handles


def install_output_finite_hooks(model, checker, operation="forward", prompt=None):
    handles = []
    common = {"operation": operation, "prompt": prompt}
    handles.append(model.model.norm.register_forward_hook(
        lambda module, args, output: checker.check(
            output, module_name="model.model.norm", role="output", **common)))
    handles.append(model.lm_head.register_forward_hook(
        lambda module, args, output: checker.check(
            output, module_name="model.lm_head", role="output", **common)))
    handles.append(model.register_forward_hook(
        lambda module, args, output: checker.check(
            output, module_name="LongcatNextForCausalLM", role="output", **common)))
    return handles


def manual_sdpa_f32(torch, query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False):
    require(float(dropout_p) == 0.0, "sdpa-f32 requires dropout_p=0 during inference")
    q = query.float()
    k = key.float()
    v = value.float()
    if enable_gqa:
        require(q.shape[-3] % k.shape[-3] == 0, "GQA query heads must be divisible by KV heads")
        factor = q.shape[-3] // k.shape[-3]
        k = k.repeat_interleave(factor, dim=-3)
        v = v.repeat_interleave(factor, dim=-3)
    score = torch.matmul(q, k.transpose(-2, -1))
    score = score * (float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    if is_causal:
        q_len, k_len = score.shape[-2:]
        causal = torch.ones((q_len, k_len), dtype=torch.bool, device=score.device).tril()
        score = score.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            score = score.masked_fill(~attn_mask, float("-inf"))
        else:
            score = score + attn_mask.float()
    probabilities = torch.softmax(score, dim=-1)
    return torch.matmul(probabilities, v).to(query.dtype)


def manual_sdpa_f32_numpy(query, key, value, attn_mask=None, is_causal=False,
                          scale=None, enable_gqa=False):
    """Checkpoint-free oracle for the diagnostic SDPA implementation."""
    import numpy as np
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    if enable_gqa:
        require(q.shape[-3] % k.shape[-3] == 0, "GQA query heads must be divisible by KV heads")
        factor = q.shape[-3] // k.shape[-3]
        k = np.repeat(k, factor, axis=-3); v = np.repeat(v, factor, axis=-3)
    score = np.matmul(q, np.swapaxes(k, -2, -1))
    score *= float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if is_causal:
        score = np.where(np.tri(score.shape[-2], score.shape[-1], dtype=bool), score, -np.inf)
    if attn_mask is not None:
        mask = np.asarray(attn_mask)
        score = np.where(mask, score, -np.inf) if mask.dtype.kind == "b" else score + mask
    maximum = np.max(score, axis=-1, keepdims=True)
    weights = np.exp(score - maximum)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v)


@contextlib.contextmanager
def instrument_sdpa(torch, checker, backend):
    require(backend in ATTENTION_BACKENDS, f"unknown attention backend {backend}")
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    calls = []
    global_sequence = 0
    backend_context = contextlib.nullcontext()
    controls = {"requested": backend, "api": None}
    if backend == "sdpa-math":
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend_context = sdpa_kernel(backends=[SDPBackend.MATH])
            controls["api"] = "torch.nn.attention.sdpa_kernel(MATH)"
        except (ImportError, AttributeError):
            backend_context = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False,
                enable_cudnn=False)
            controls["api"] = "torch.backends.cuda.sdp_kernel(math-only)"

    def wrapped(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False):
        nonlocal global_sequence
        global_sequence += 1
        active = CURRENT_ATTENTION_CONTEXT.get() or {}
        module_call = int(active.get("sdpa_call_index", 0))
        if active:
            active["sdpa_call_index"] = module_call + 1
        call = {"is_causal": bool(is_causal), "scale": scale,
                "dropout_p": float(dropout_p), "enable_gqa": bool(enable_gqa),
                "backend": backend, "logical_layer": active.get("logical_layer"),
                "physical_block": active.get("physical_block"),
                "module_name": active.get("module_name"),
                "module_call_index": module_call, "global_call_sequence": global_sequence}
        calls.append(call)
        common = {"module_name": active.get("module_name") or
                  "torch.nn.functional.scaled_dot_product_attention",
                  "logical_layer": active.get("logical_layer"),
                  "physical_block": active.get("physical_block"),
                  "operation": "scaled_dot_product_attention"}
        for role, tensor in (("query", query), ("key", key), ("value", value)):
            checker.check(tensor, **common, role=role)
        if attn_mask is not None and getattr(attn_mask, "is_floating_point", lambda: False)():
            checker.check(attn_mask, **common, role="attention_mask")
        output = (manual_sdpa_f32(torch, query, key, value, attn_mask, dropout_p,
                                  is_causal, scale, enable_gqa)
                  if backend == "sdpa-f32" else
                  original(query, key, value, attn_mask=attn_mask,
                           dropout_p=dropout_p, is_causal=is_causal,
                           scale=scale, enable_gqa=enable_gqa))
        checker.check(output, **common, role="output")
        return output

    functional.scaled_dot_product_attention = wrapped
    try:
        with backend_context:
            yield {"calls": calls, "controls": controls,
                   "monkey_patch": "scoped SDPA finite instrumentation"}
    finally:
        functional.scaled_dot_product_attention = original


def configure_attention_backend(model_or_config, backend):
    require(backend in ATTENTION_BACKENDS, f"unknown attention backend {backend}")
    requested = None if backend == "default" else ("eager" if backend == "eager" else "sdpa")
    config = getattr(model_or_config, "config", model_or_config)
    if requested is not None:
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = requested
        elif hasattr(config, "attn_implementation"):
            config.attn_implementation = requested
        else:
            raise V2Error(f"custom model configuration cannot request {requested} attention")
    effective = getattr(config, "_attn_implementation",
                        getattr(config, "attn_implementation", None))
    if backend == "eager" and effective != "eager":
        raise V2Error(f"eager attention requested but effective implementation is {effective!r}")
    return {"requested_backend": backend, "requested_implementation": requested,
            "effective_implementation": effective}


def load_shard_manifest(path):
    data = json.loads(Path(path).read_text(encoding="ascii"))
    require(data.get("schema_version") == 1, "shard manifest schema must be 1")
    require(data.get("repository") == PINNED_MODEL_REPOSITORY, "wrong shard repository")
    require(data.get("revision") == PINNED_MODEL_REVISION, "wrong shard revision")
    shards = data.get("shards")
    require(isinstance(shards, list) and len(shards) == 15, "shard manifest must contain 15 entries")
    names = set()
    for row in shards:
        require(set(row) == {"filename", "bytes", "lfs_sha256"}, "invalid shard manifest entry fields")
        require(row["filename"] not in names, "duplicate shard manifest filename")
        require(row["filename"].startswith("model-") and row["filename"].endswith(".safetensors"),
                "invalid shard filename")
        require(isinstance(row["bytes"], int) and row["bytes"] > 0, "invalid shard byte count")
        require(len(row["lfs_sha256"]) == 64 and
                all(c in "0123456789abcdef" for c in row["lfs_sha256"]), "invalid LFS SHA-256 OID")
        names.add(row["filename"])
    expected_names = {f"model-{index:05d}-of-00015.safetensors" for index in range(1, 16)}
    require(names == expected_names, "shard manifest filenames do not match the official 15-shard set")
    return data


def verify_and_scan_source(model_dir, manifest_path, report_dir, safe_open_fn=None, torch_module=None):
    scan_started = utc_now()
    if torch_module is None:
        import torch
    else:
        torch = torch_module
    if safe_open_fn is None:
        from safetensors import safe_open
        safe_open_fn = safe_open
    manifest = load_shard_manifest(manifest_path)
    root = Path(model_dir)
    expected_names = {row["filename"] for row in manifest["shards"]}
    actual_names = {path.name for path in root.glob("model-*-of-00015.safetensors")}
    require(actual_names == expected_names,
            f"checkpoint shard filenames differ: missing={sorted(expected_names-actual_names)} "
            f"unexpected={sorted(actual_names-expected_names)}")
    dtype_counts = {}
    tensor_count = 0
    for row in manifest["shards"]:
        path = root / row["filename"]
        require(path.is_file(), f"missing checkpoint shard {row['filename']}")
        require(path.stat().st_size == row["bytes"], f"wrong byte count for {row['filename']}")
        require(sha256_file(path) == row["lfs_sha256"], f"wrong LFS SHA-256 for {row['filename']}")
        reports = []
        with safe_open_fn(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                tensor = shard.get_tensor(name)
                tensor_count += 1
                dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1
                if tensor.is_floating_point():
                    report = torch_finite_report(torch, f"{row['filename']}:{name}", tensor)
                    reports.append(report)
                    if report["finite_count"] != report["total_elements"]:
                        atomic_json(Path(report_dir) / "source-scan" /
                                    "first-nonfinite-source.json", report)
                        raise V2Error(f"non-finite source tensor {row['filename']}:{name}")
                del tensor
        atomic_json(Path(report_dir) / "source-scan" / (row["filename"] + ".json"), {
            "schema_version": SCHEMA_VERSION, "shard": row["filename"],
            "all_finite": True, "floating_tensors": reports})
    summary = {"schema_version": SCHEMA_VERSION, "all_finite": True,
               "verified_shards": [dict(row) for row in manifest["shards"]],
               "tensor_count": tensor_count, "dtype_counts": dtype_counts,
               "start_utc": scan_started, "end_utc": utc_now(),
               "manifest_sha256": sha256_file(manifest_path)}
    atomic_json(Path(report_dir) / "source-scan-summary.json", summary)
    return summary


def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def sdpa_backend_controls(torch):
    controls = {"effective_kernel_identity": "not determinable before/after an individual SDPA call"}
    cuda = getattr(getattr(torch, "backends", None), "cuda", None)
    for name in ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled",
                 "cudnn_sdp_enabled"):
        function = getattr(cuda, name, None)
        try:
            controls[name] = bool(function()) if callable(function) else None
        except Exception:
            controls[name] = None
    return controls


def collect_provenance(repo_root, model_dir, precision, backend, run_index,
                       script_paths, model=None, extra=None):
    import torch
    root = Path(repo_root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    dirty = bool(git("status", "--porcelain"))
    devices, dtypes = set(), set()
    device_map = None
    if model is not None:
        device_map = getattr(model, "hf_device_map", None)
        for value in itertools.chain(model.parameters(), model.buffers()):
            devices.add(str(value.device)); dtypes.add(str(value.dtype))
    cuda = {"available": bool(torch.cuda.is_available()), "build": torch.version.cuda,
            "runtime": None, "driver": None, "gpu": None, "capability": None}
    if cuda["available"]:
        cuda.update({"gpu": torch.cuda.get_device_name(0),
                     "capability": list(torch.cuda.get_device_capability(0))})
        try: cuda["runtime"] = torch._C._cuda_getRuntimeVersion()
        except AttributeError: pass
        try: cuda["driver"] = torch._C._cuda_getDriverVersion()
        except AttributeError: pass
    env_names = ("CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING", "HF_HUB_OFFLINE",
                 "TRANSFORMERS_OFFLINE", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_LOGS",
                 "TORCH_CUDNN_V8_API_DISABLED", "PYTORCH_NO_CUDA_MEMORY_CACHING",
                 "NVIDIA_TF32_OVERRIDE")
    report = {
        "generator_schema_version": SCHEMA_VERSION,
        "script_sha256": {Path(p).name: sha256_file(p) for p in script_paths},
        "llama_cpp": {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
                      "dirty": dirty},
        "checkpoint": {"repository": PINNED_MODEL_REPOSITORY,
                       "revision": PINNED_MODEL_REVISION, "path": str(Path(model_dir).resolve())},
        "python": {"executable": sys.executable, "version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "operating_system": platform.platform(), "torch": torch.__version__, "cuda": cuda,
        "packages": package_versions(("transformers", "accelerate", "safetensors", "numpy", "flash-attn")),
        "environment": {name: os.environ.get(name) for name in env_names},
        "requested_precision": precision, "requested_attention_backend": backend,
        "model_hf_device_map": device_map, "observed_devices": sorted(devices),
        "observed_dtypes": sorted(dtypes), "autocast_enabled": bool(torch.is_autocast_enabled()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "sdpa_backend_controls": sdpa_backend_controls(torch),
        "process_id": os.getpid(), "run_index": run_index, "start_utc": utc_now(),
    }
    if extra: report.update(extra)
    require(PROVENANCE_REQUIRED_FIELDS <= set(report),
            f"provenance is missing fields: {sorted(PROVENANCE_REQUIRED_FIELDS - set(report))}")
    return report


def load_worker_arrays(run_dir, expected=EXPECTED_ACCEPTED_ARRAYS, precision=None, contract=None):
    import numpy as np
    path = Path(run_dir) / "arrays.npz"
    require(path.is_file(), f"worker result is missing {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validation = validate_array_inventory(arrays, expected)
    if contract is not None:
        validation.update(validate_accepted_arrays(arrays, precision, contract))
    return arrays, validation


def array_metadata_for(value):
    data = value.tobytes(order="C")
    return {"shape": list(value.shape), "serialized_dtype": str(value.dtype),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def validate_worker_artifacts(run_dir, precision, backend, run_index, contract,
                              expected_script_hashes):
    root = Path(run_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="ascii"))
    reproducibility = json.loads((root / "worker-reproducibility.json").read_text(encoding="ascii"))
    validation_file = json.loads((root / "validation.json").read_text(encoding="ascii"))
    require(metadata.get("schema_version") == 2, "worker metadata schema must be 2")
    require(metadata.get("kind") == "longcat-next-core-reference", "worker kind is not reference")
    require(metadata.get("precision") == precision, "worker precision differs")
    require(metadata.get("checkpoint", {}).get("revision") == PINNED_MODEL_REVISION,
            "worker checkpoint revision differs")
    require(metadata.get("source_weight_validation", {}).get("all_finite") is True,
            "worker source weights were not validated finite")
    require(metadata.get("whole_candidate_validation", {}).get("all_finite") is True,
            "worker arrays were not validated finite")
    provenance = metadata.get("provenance", {})
    require(PROVENANCE_REQUIRED_FIELDS <= set(provenance), "worker provenance is incomplete")
    require(provenance.get("run_index") == run_index and isinstance(provenance.get("process_id"), int),
            "worker process/run provenance differs")
    require(provenance.get("requested_attention_backend") == backend,
            "worker requested attention backend differs")
    require(provenance.get("requested_precision") == precision and
            isinstance(provenance.get("effective_precision"), str),
            "worker requested/effective precision differs")
    effective = provenance.get("effective_attention_backend", {})
    require(effective.get("requested_backend") == backend, "worker effective backend record differs")
    require("effective_implementation" in effective,
            "worker effective attention implementation is missing")
    require(backend != "sdpa-f32", "sdpa-f32 workers cannot be accepted")
    require(provenance.get("script_sha256") == expected_script_hashes,
            "worker generator script identities differ")
    require(provenance.get("shard_manifest_sha256") ==
            expected_script_hashes["checkpoint-shards-v2.json"], "worker shard manifest hash differs")
    require(isinstance(provenance.get("start_utc"), str) and isinstance(provenance.get("end_utc"), str),
            "worker timestamps are incomplete")
    require(isinstance(provenance.get("sdpa_backend_controls"), dict) and
            "effective_kernel_identity" in provenance["sdpa_backend_controls"],
            "worker SDPA backend controls are incomplete")
    require(reproducibility.get("schema_version") == 2 and
            reproducibility.get("repeat_count") == 1, "worker reproducibility document is invalid")
    require(validation_file.get("all_finite") is True and
            validation_file.get("contract_valid") is True, "worker validation document is invalid")
    arrays, validation = load_worker_arrays(root, precision=precision, contract=contract)
    recorded = metadata.get("arrays", {})
    require(set(recorded) == set(arrays), "worker array metadata names differ from NPZ")
    for name, value in arrays.items():
        actual = array_metadata_for(value)
        for field in ("shape", "serialized_dtype", "sha256", "bytes"):
            require(recorded[name].get(field) == actual[field],
                    f"worker metadata {name}/{field} differs from NPZ")
    return arrays, metadata, validation


def compare_independent_runs(runs):
    import numpy as np
    require(len(runs) >= 2, "at least two independent worker runs are required")
    for index, run in enumerate(runs):
        validate_array_inventory(run, EXPECTED_ACCEPTED_ARRAYS)
        require(set(run) == set(runs[0]), f"run {index} inventory differs from run 0")
    report = {}
    for name, baseline in sorted(runs[0].items()):
        validate_numpy_array(name, baseline)
        maximum_abs = maximum_rel = 0.0
        byte_identical = True
        for run in runs[1:]:
            candidate = run[name]
            validate_numpy_array(name, candidate)
            require(candidate.shape == baseline.shape, f"shape mismatch for {name}")
            byte_identical &= candidate.tobytes() == baseline.tobytes()
            if baseline.dtype.kind in "biu":
                require(np.array_equal(candidate, baseline), f"exact output differs for {name}")
            else:
                left, right = baseline.astype(np.float64), candidate.astype(np.float64)
                diff = np.abs(left - right)
                maximum_abs = max(maximum_abs, float(diff.max(initial=0.0)))
                maximum_rel = max(maximum_rel, float((diff / np.maximum(
                    np.abs(left), np.finfo(np.float64).tiny)).max(initial=0.0)))
        report[name] = {"shape": list(baseline.shape), "dtype": str(baseline.dtype),
                        "byte_identical": byte_identical,
                        "max_absolute_difference": maximum_abs,
                        "max_relative_difference": maximum_rel,
                        "comparison_tolerance": None}
    return report


def orchestrate_workers(candidate_root, worker_commands, precision, backend, contract_path,
                        implementation_paths, expected=EXPECTED_ACCEPTED_ARRAYS,
                        runner=subprocess.run):
    require(len(worker_commands) >= 2, "core acceptance requires at least two worker processes")
    final = Path(candidate_root)
    require(not final.exists(), f"refusing to overwrite candidate directory {final}")
    staging = final.with_name("." + final.name + f".staging-{os.getpid()}")
    require(not staging.exists(), f"staging directory already exists: {staging}")
    runs_root = staging / "runs"
    runs_root.mkdir(parents=True)
    arrays_by_run = []
    worker_process_ids = []
    contract = load_accepted_contract(contract_path)
    expected_script_hashes = {Path(path).name: sha256_file(path) for path in implementation_paths}
    try:
        for index, command in enumerate(worker_commands):
            run_dir = runs_root / f"run-{index:02d}"
            run_dir.mkdir()
            command = [str(run_dir) if item == "{run_dir}" else item for item in command]
            with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, \
                 (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                result = runner(command, stdout=stdout, stderr=stderr, text=True)
            require(result.returncode == 0, f"independent worker {index} failed with exit code {result.returncode}")
            arrays, metadata, validation = validate_worker_artifacts(
                run_dir, precision, backend, index, contract, expected_script_hashes)
            arrays_by_run.append(arrays)
            worker_process_ids.append(metadata["provenance"]["process_id"])
        require(len(set(worker_process_ids)) == len(worker_process_ids),
                "core workers did not run in independent processes")
        reproducibility = compare_independent_runs(arrays_by_run)
        all_byte_identical = all(row["byte_identical"] for row in reproducibility.values())
        require(all_byte_identical,
                "independent worker arrays are not byte-identical; no tolerance is selected")
        per_run_hashes = {f"run-{i:02d}": sha256_file(runs_root / f"run-{i:02d}" / "arrays.npz")
                          for i in range(len(arrays_by_run))}
        parent_reproducibility = {
            "schema_version": SCHEMA_VERSION, "independent_processes": len(arrays_by_run),
            "repeat_count": len(arrays_by_run),
            "arrays": reproducibility, "comparison_tolerances": {"bf16": None, "f16": None},
            "byte_identical": all_byte_identical,
            "greedy_continuations": [
                {name: run[name].tolist() for name in contract["greedy_arrays"]}
                for run in arrays_by_run],
            "tolerances_selected_automatically": False}
        atomic_json(staging / "reproducibility.json", parent_reproducibility)
        atomic_json(staging / "candidate-validation.json", {
            "schema_version": SCHEMA_VERSION, "exact_inventory_count": len(arrays_by_run[0]),
            "expected_inventory_count": expected, "whole_candidate_finite": True,
            "per_run_npz_sha256": per_run_hashes, "accepted": True,
            "cross_run_byte_identical": all_byte_identical,
            "promotion_utc": utc_now()})
        run_metadata_hashes = {
            f"run-{i:02d}": sha256_file(runs_root / f"run-{i:02d}" / "metadata.json")
            for i in range(len(arrays_by_run))}
        atomic_json(staging / "candidate-metadata.json", {
            "schema_version": SCHEMA_VERSION, "kind": "longcat-next-core-reference-candidate",
            "accepted_inventory_count": expected, "run_count": len(arrays_by_run),
            "precision": precision, "attention_backend": backend,
            "per_run_metadata_sha256": run_metadata_hashes,
            "acceptance_utc": utc_now()})
        stem = f"longcat-next-core-{precision}"
        canonical_npz = staging / f"{stem}.npz"
        canonical_metadata = staging / f"{stem}.json"
        canonical_reproducibility = staging / "longcat-next-core-reproducibility.json"
        shutil.copy2(runs_root / "run-00" / "arrays.npz", canonical_npz)
        shutil.copy2(runs_root / "run-00" / "metadata.json", canonical_metadata)
        atomic_json(canonical_reproducibility, parent_reproducibility)
        require(sha256_file(canonical_npz) == per_run_hashes["run-00"],
                "canonical root NPZ differs from run 00")
        staging.replace(final)
    except Exception:
        # Retain staging worker evidence, but never expose it as an accepted candidate.
        raise
    return final


def validate_candidate(candidate_dir, contract_path, implementation_paths):
    root = Path(candidate_dir)
    candidate_metadata = json.loads((root / "candidate-metadata.json").read_text(encoding="ascii"))
    candidate_validation = json.loads((root / "candidate-validation.json").read_text(encoding="ascii"))
    parent_repro = json.loads((root / "longcat-next-core-reproducibility.json").read_text(encoding="ascii"))
    require(candidate_metadata.get("schema_version") == 2, "candidate schema must be 2")
    precision = candidate_metadata.get("precision")
    backend = candidate_metadata.get("attention_backend")
    require(backend != "sdpa-f32", "sdpa-f32 candidate is forbidden")
    run_dirs = sorted((root / "runs").glob("run-*"))
    require(len(run_dirs) >= 2 and len(run_dirs) == candidate_metadata.get("run_count"),
            "candidate independent process count differs")
    contract = load_accepted_contract(contract_path)
    identities = {Path(path).name: sha256_file(path) for path in implementation_paths}
    arrays_by_run = []
    process_ids = []
    for index, run_dir in enumerate(run_dirs):
        arrays, metadata, validation = validate_worker_artifacts(
            run_dir, precision, backend, index, contract, identities)
        arrays_by_run.append(arrays)
        process_ids.append(metadata["provenance"]["process_id"])
        require(candidate_validation.get("per_run_npz_sha256", {}).get(f"run-{index:02d}") ==
                sha256_file(run_dir / "arrays.npz"), "candidate per-run NPZ hash differs")
        require(candidate_metadata.get("per_run_metadata_sha256", {}).get(f"run-{index:02d}") ==
                sha256_file(run_dir / "metadata.json"), "candidate per-run metadata hash differs")
    comparison = compare_independent_runs(arrays_by_run)
    require(len(set(process_ids)) == len(process_ids), "candidate workers are not independent processes")
    require(all(row["byte_identical"] for row in comparison.values()),
            "candidate runs are not byte-identical")
    stem = f"longcat-next-core-{precision}"
    canonical_npz = root / f"{stem}.npz"
    canonical_metadata = root / f"{stem}.json"
    require(canonical_npz.is_file() and canonical_metadata.is_file(),
            "candidate canonical root artifacts are missing")
    require(sha256_file(canonical_npz) == sha256_file(run_dirs[0] / "arrays.npz"),
            "canonical NPZ differs from run 00")
    require(sha256_file(canonical_metadata) == sha256_file(run_dirs[0] / "metadata.json"),
            "canonical metadata differs from run 00")
    require(parent_repro.get("independent_processes") == len(run_dirs) and
            parent_repro.get("byte_identical") is True, "parent reproducibility is invalid")
    require(candidate_validation.get("whole_candidate_finite") is True and
            candidate_validation.get("exact_inventory_count") == EXPECTED_ACCEPTED_ARRAYS,
            "candidate validation gate is invalid")
    return {"schema_version": 2, "candidate_dir": str(root.resolve()), "valid": True,
            "precision": precision, "attention_backend": backend,
            "independent_processes": len(run_dirs), "array_count": EXPECTED_ACCEPTED_ARRAYS,
            "canonical_npz_sha256": sha256_file(canonical_npz)}
