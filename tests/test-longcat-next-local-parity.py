#!/usr/bin/env python3
"""Opt-in LongCat-Next C++ capture/comparison with frozen Stage-1 policy."""
import argparse
import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROUTED_EXPERT_COUNT = 384
REAL_EXPERT_COUNT = 256

CPP_TO_REFERENCE = {
    "inp_embd": "base_embedding", "inp_embd_ngram": "fused_pre_trunk_embedding",
    "l_out-0": "physical_block_00", "l_out-1": "physical_block_01",
    "l_out-2": "physical_block_02", "l_out-27": "physical_block_27",
    "h_nextn": "final_normalized_hidden_state",
    "final_logits": "complete_final_position_logits",
}
for i in range(12):
    CPP_TO_REFERENCE[f"ngram_proj-{i}"] = f"ngram_projection_raw_{i:02d}"
DIRECT_NAMES = set(CPP_TO_REFERENCE)
ALL_BLOCK_CPP_NAMES = {f"l_out-{block}" for block in range(28)}
ALL_BLOCK_REFERENCE_NAMES = {f"physical_block_{block:02d}" for block in range(28)}
COMPONENT_SUFFIXES = ("block_input", "attention_norm", "attention_output",
                      "post_attention_residual", "ffn_norm", "dense_output", "block_output")
COMPONENT_MOE_SUFFIXES = ("router_logits", "router_probabilities", "router_selection_scores",
                          "router_topk_indices", "router_topk_weights", "identity_weight_sum",
                          "identity_residual", "moe_shortcut")
OBSERVED_FLOAT32_ROUTER_SUFFIXES = {
    "identity_residual", "identity_weight_sum", "router_logits", "router_probabilities",
    "router_selection_scores", "router_topk_weights"}


def observed_bfloat16_policy_names():
    return {name for name in component_names()
            if name.split("__", 1)[1] not in OBSERVED_FLOAT32_ROUTER_SUFFIXES |
               {"router_topk_indices"}}


def observed_float32_policy_names():
    return {name for name in component_names()
            if name.split("__", 1)[1] in OBSERVED_FLOAT32_ROUTER_SUFFIXES}


def boundary_rounding_policy_names():
    names = {f"physical_block_{block:02d}__{suffix}" for block in range(10)
             for suffix in ("post_attention_residual", "block_output")}
    names.update(f"physical_block_{block:02d}__moe_shortcut" for block in range(0, 10, 2))
    names.update(f"physical_block_{block:02d}__block_input" for block in range(1, 10))
    return names


def hidden_surface_rounding_additions():
    return observed_bfloat16_policy_names() - boundary_rounding_policy_names()
COMPONENT_CPP_BASE = {
    "block_in": "block_input", "attn_norm": "attention_norm", "attn_out": "attention_output",
    "ffn_inp": "post_attention_residual", "ffn_norm": "ffn_norm", "ffn_out": "dense_output",
    "l_out": "block_output", "ffn_moe_logits": "router_logits",
    "ffn_moe_probs": "router_probabilities", "ffn_moe_probs_biased": "router_selection_scores",
    "ffn_moe_topk": "router_topk_indices", "ffn_moe_weights_scaled": "router_topk_weights",
    "identity_weight_sum": "identity_weight_sum", "identity_residual": "identity_residual",
    "moe_shortcut": "moe_shortcut"}
INTEGER_SUFFIXES = ("input_ids", "attention_mask", "position_ids", "cache_position")


def validate_reference_finiteness(npz, output_dir):
    invalid = {}
    scanned = 0
    for key in npz.files:
        value = np.asarray(npz[key])
        if value.dtype.kind != "f":
            continue
        scanned += 1
        bad = np.argwhere(~np.isfinite(value))
        if bad.size:
            invalid[key] = {"non_finite_count": int(bad.shape[0]),
                            "nan_count": int(np.isnan(value).sum()),
                            "positive_infinity_count": int(np.isposinf(value).sum()),
                            "negative_infinity_count": int(np.isneginf(value).sum()),
                            "first_affected_indices": bad[:8].tolist()}
    report = {"scanned_floating_arrays": scanned, "invalid_arrays": invalid,
              "passed": not invalid}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reference-validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def run_capture_after_validation(validation, command, runner=subprocess.run):
    if not validation["passed"]:
        keys = ", ".join(sorted(validation["invalid_arrays"]))
        raise ValueError(f"invalid parity reference contains non-finite arrays: {keys}")
    return runner(command, check=True)


def read_manifest(path):
    rows = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"malformed capture manifest line {number}")
        name, dtype, shape, filename = fields
        if name in rows:
            raise ValueError(f"duplicate C++ capture name: {name}")
        try:
            dims = tuple(map(int, shape.split(",")))
        except ValueError as exc:
            raise ValueError(f"malformed shape for {name}: {shape}") from exc
        if dtype not in {"f32", "f16", "bf16"} or len(dims) != 4 or min(dims) < 1:
            raise ValueError(f"malformed capture metadata for {name}")
        rows[name] = (dtype, dims, path.parent / filename)
    unexpected = sorted(set(rows) - set(CPP_TO_REFERENCE))
    missing = sorted(DIRECT_NAMES - set(rows))
    if unexpected or missing:
        raise ValueError(f"capture-name mismatch: missing={missing}, unexpected={unexpected}")
    return rows


def read_all_blocks_manifest(path):
    rows = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"malformed all-block manifest line {number}")
        name, dtype, shape, filename = fields
        if name in rows:
            raise ValueError(f"duplicate all-block C++ capture name: {name}")
        try:
            dims = tuple(map(int, shape.split(",")))
        except ValueError as exc:
            raise ValueError(f"malformed all-block shape for {name}: {shape}") from exc
        if dtype not in {"f32", "f16", "bf16"} or len(dims) != 4 or min(dims) < 1:
            raise ValueError(f"malformed all-block capture metadata for {name}")
        rows[name] = (dtype, dims, path.parent / filename)
    missing = sorted(ALL_BLOCK_CPP_NAMES - set(rows))
    unexpected = sorted(set(rows) - ALL_BLOCK_CPP_NAMES)
    if missing or unexpected:
        raise ValueError(f"all-block capture-name mismatch: missing={missing}, unexpected={unexpected}")
    return rows


def component_names():
    names = []
    for block in range(10):
        names.extend(f"physical_block_{block:02d}__{suffix}" for suffix in COMPONENT_SUFFIXES)
        if block % 2 == 0:
            names.extend(f"physical_block_{block:02d}__{suffix}" for suffix in COMPONENT_MOE_SUFFIXES)
    if len(names) != 110 or len(set(names)) != 110:
        raise AssertionError("component inventory must contain 110 unique names")
    return names


def component_window_names(start=10, count=4):
    if start < 0 or start % 2 or count <= 0 or count % 2 or start + count > 28:
        raise ValueError("component window requires even start/count within physical blocks 0-27")
    names = []
    for block in range(start, start + count):
        names.extend(f"physical_block_{block:02d}__{suffix}" for suffix in COMPONENT_SUFFIXES)
        if block % 2 == 0:
            names.extend(f"physical_block_{block:02d}__{suffix}" for suffix in COMPONENT_MOE_SUFFIXES)
    return names


def read_component_window_manifest(path, start=10, count=4):
    expected = set(component_window_names(start, count)); rows = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 4: raise ValueError(f"malformed component-window manifest line {number}")
        cpp_name, dtype, shape, filename = fields
        dash = cpp_name.rfind("-")
        if dash < 1 or not cpp_name[dash + 1:].isdigit(): raise ValueError("malformed component-window callback")
        block = int(cpp_name[dash + 1:]); suffix = COMPONENT_CPP_BASE.get(cpp_name[:dash])
        canonical = f"physical_block_{block:02d}__{suffix}" if suffix else None
        if canonical not in expected or canonical in rows: raise ValueError("unexpected or duplicate component-window row")
        dims = tuple(map(int, shape.split(",")))
        if dtype not in {"f32", "f16", "bf16", "i32"} or len(dims) != 4:
            raise ValueError("malformed component-window metadata")
        rows[canonical] = (dtype, dims, path.parent / filename)
    if set(rows) != expected: raise ValueError("component-window inventory mismatch")
    return rows


def read_component_manifest(path):
    rows = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"malformed component manifest line {number}")
        cpp_name, dtype, shape, filename = fields
        if cpp_name in rows:
            raise ValueError(f"duplicate component C++ capture name: {cpp_name}")
        dash = cpp_name.rfind("-")
        if dash < 1 or not cpp_name[dash + 1:].isdigit():
            raise ValueError(f"malformed component callback name: {cpp_name}")
        block = int(cpp_name[dash + 1:]); base = cpp_name[:dash]
        suffix = COMPONENT_CPP_BASE.get(base)
        canonical = f"physical_block_{block:02d}__{suffix}" if suffix is not None else None
        try:
            dims = tuple(map(int, shape.split(",")))
        except ValueError as exc:
            raise ValueError(f"malformed component shape for {cpp_name}") from exc
        if canonical is None or canonical not in set(component_names()) or dtype not in {"f32", "f16", "bf16", "i32"} or len(dims) != 4:
            raise ValueError(f"unexpected component capture: {cpp_name}")
        rows[cpp_name] = (canonical, dtype, dims, path.parent / filename)
    canonical_names = [row[0] for row in rows.values()]
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("duplicate canonical component capture")
    expected = set(component_names())
    if set(canonical_names) != expected:
        raise ValueError(f"component capture inventory mismatch: missing={sorted(expected-set(canonical_names))}, unexpected={sorted(set(canonical_names)-expected)}")
    return {canonical: (dtype, dims, raw) for canonical, dtype, dims, raw in rows.values()}


def decode_raw(dtype_name, dims, path, kind):
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16}[dtype_name]
    raw = np.fromfile(path, dtype=dtype)
    if raw.size != np.prod(dims):
        raise ValueError(f"{path}: raw element count {raw.size} != shape product {np.prod(dims)}")
    if dtype_name == "bf16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        raw = raw.astype(np.float32)
    if kind == "logits":
        return raw.reshape((1, dims[0]))
    # GGML [hidden,tokens,1,1], contiguous with hidden fastest, maps to
    # Python [batch=1,tokens,hidden]. Never squeeze singleton dimensions.
    return raw.reshape((dims[1], dims[0]))[None, :, :]


def decode_component_raw(canonical_name, dtype_name, dims, path, layout_provenance=None):
    """Decode the validated GGML layout for one canonical component surface."""
    suffix = canonical_name.split("__", 1)[1]
    expected_width = (384 if suffix in {
        "router_logits", "router_probabilities", "router_selection_scores"}
        else 12 if suffix in {"router_topk_indices", "router_topk_weights"}
        else 1 if suffix == "identity_weight_sum" else 3072)
    if suffix == "router_topk_weights":
        if len(dims) != 4 or dims[0] != 1 or dims[1] != 12 or dims[2] <= 0 or dims[3] != 1:
            raise ValueError(
                f"{canonical_name}: router top-k weights require GGML layout "
                f"[1,12,tokens,1], got {dims}")
        if dtype_name == "i32":
            raise ValueError(f"{canonical_name}: router top-k weights must be floating point")
        dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16}[dtype_name]
        raw = np.fromfile(path, dtype=dtype)
        if raw.size != np.prod(dims):
            raise ValueError(f"{path}: raw component element count differs from shape")
        if dtype_name == "bf16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            raw = raw.astype(np.float32)
        # GGML ne[0] is contiguous: [1, top_k, tokens, 1] becomes
        # the canonical [batch=1, tokens, top_k] surface.
        return raw.reshape((dims[2], dims[1], dims[0]))[:, :, 0][None, :, :]

    if (len(dims) != 4 or dims[0] != expected_width or dims[1] <= 0 or
            dims[2] != 1 or dims[3] != 1):
        raise ValueError(
            f"{canonical_name}: expected GGML layout [{expected_width},tokens,1,1], got {dims}")
    if suffix == "router_topk_indices":
        if dtype_name != "i32":
            raise ValueError(f"{canonical_name}: router top-k indices must be i32")
        raw = np.fromfile(path, dtype=np.int32)
        compact_count = int(np.prod(dims))
        legacy_count = (dims[1] - 1) * ROUTED_EXPERT_COUNT + dims[0]
        if raw.size == compact_count:
            result = raw.reshape((dims[1], dims[0]))[None, :, :]
            representation = "compact_logical"
        elif raw.size == legacy_count:
            result = np.stack([
                raw[token * ROUTED_EXPERT_COUNT:token * ROUTED_EXPERT_COUNT + dims[0]]
                for token in range(dims[1])], axis=0)[None, :, :]
            representation = "legacy_strided_argsort_view"
        else:
            raise ValueError(
                f"{path}: router top-k index raw count {raw.size} is neither compact "
                f"{compact_count} nor pinned legacy strided {legacy_count}")
        if layout_provenance is not None:
            layout_provenance.append({"name": canonical_name, "representation": representation})
        return result
    if dtype_name == "i32":
        raise ValueError(f"{canonical_name}: floating component unexpectedly uses i32")
    return decode_raw(dtype_name, dims, path, "hidden")


def policy_for(policy, precision, suffix):
    key = "ngram_projection_raw" if suffix.startswith("ngram_projection_raw_") else suffix
    if key not in policy[precision]:
        raise ValueError(f"no frozen tolerance for {precision}/{suffix}")
    return policy[precision][key]


def floating_result(reference, candidate, tolerance):
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: C++ {candidate.shape} != reference {reference.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("all floating comparison values must be finite")
    diff = np.abs(candidate - reference)
    limit = tolerance["atol"] + tolerance["rtol"] * np.abs(reference)
    violation = np.zeros_like(diff, dtype=np.float64)
    np.divide(diff, limit, out=violation, where=limit != 0)
    violation[(limit == 0) & (diff != 0)] = np.inf
    denom = np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    return {**tolerance, "max_absolute_error": float(diff.max(initial=0)),
            "max_relative_error": float((diff / denom).max(initial=0)),
            "max_normalized_tolerance_violation": float(violation.max(initial=0)),
            "passed": bool(np.all(diff <= limit))}


def all_block_result(reference, candidate, criterion):
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: C++ {candidate.shape} != reference {reference.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("all all-block comparison values must be finite")
    diff = np.abs(candidate - reference)
    limit = criterion["atol"] + criterion["rtol"] * np.abs(reference)
    violation = np.zeros_like(diff, dtype=np.float64)
    np.divide(diff, limit, out=violation, where=limit != 0)
    violation[(limit == 0) & (diff != 0)] = np.inf
    denominator = np.maximum(np.abs(reference), np.finfo(np.float32).tiny)
    left = reference.reshape(-1).astype(np.float64)
    right = candidate.reshape(-1).astype(np.float64)
    norm_product = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / norm_product) if norm_product else float(np.array_equal(left, right))
    return {
        "maximum_absolute_error": float(diff.max(initial=0)),
        "mean_absolute_error": float(diff.mean()),
        "rms_error": float(np.sqrt(np.mean(np.square(diff, dtype=np.float64)))),
        "maximum_relative_error": float((diff / denominator).max(initial=0)),
        "cosine_similarity": cosine,
        "maximum_normalized_violation_under_diagnostic_criterion": float(
            violation.max(initial=0)),
        "within_diagnostic_criterion": bool(np.all(diff <= limit)),
    }


def compare_all_blocks(reference_npz, capture_dir, attention_mask, criterion):
    reference_names = set(reference_npz.files)
    if reference_names != ALL_BLOCK_REFERENCE_NAMES:
        raise ValueError("official all-block inventory mismatch: "
                         f"missing={sorted(ALL_BLOCK_REFERENCE_NAMES-reference_names)}, "
                         f"unexpected={sorted(reference_names-ALL_BLOCK_REFERENCE_NAMES)}")
    rows = read_all_blocks_manifest(capture_dir / "all-blocks-diagnostics.tsv")
    mask = np.asarray(attention_mask, dtype=bool)
    attended_tokens = np.flatnonzero(mask).tolist()
    results = []
    for block in range(28):
        cpp_name = f"l_out-{block}"
        reference_name = f"physical_block_{block:02d}"
        dtype, dims, raw_path = rows[cpp_name]
        candidate = decode_raw(dtype, dims, raw_path, "hidden")
        reference = np.asarray(reference_npz[reference_name], dtype=np.float32)
        if reference.shape[1] != mask.size or candidate.shape[1] != mask.size:
            raise ValueError(f"{reference_name}: attention mask length differs from token count")
        result = all_block_result(reference[:, mask, :], candidate[:, mask, :], criterion)
        results.append({"physical_block": block, "reference_name": reference_name,
                        "cpp_name": cpp_name, **result})
    first = next((row["physical_block"] for row in results
                  if not row["within_diagnostic_criterion"]), None)
    last = None if first in (None, 0) else max(
        row["physical_block"] for row in results
        if row["physical_block"] < first and row["within_diagnostic_criterion"])
    return {"accepted": False,
            "diagnostic_criterion_source": "bf16.physical_block_02",
            "diagnostic_criterion": dict(criterion),
            "first_block_exceeding_diagnostic_criterion": first,
            "last_block_within_diagnostic_criterion_before_failure": last,
            "blocks": results}


def router_index_report(cpp, default, math):
    arrays = {"cpp": np.asarray(cpp), "python_default": np.asarray(default),
              "python_math": np.asarray(math)}
    if len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("router index shapes differ")
    ordered = np.all((arrays["cpp"] == arrays["python_default"]) &
                     (arrays["cpp"] == arrays["python_math"]), axis=-1)
    selected = np.ones(ordered.shape, dtype=bool)
    for index in np.ndindex(ordered.shape):
        selected[index] = len({tuple(sorted(arrays[name][index].tolist())) for name in arrays}) == 1
    ordered_bad = np.argwhere(~ordered)
    selected_bad = np.argwhere(~selected)
    token = int(ordered_bad[0, -1]) if ordered_bad.size else (
        int(selected_bad[0, -1]) if selected_bad.size else None)
    lists = None if token is None else {
        name: value.reshape(-1, value.shape[-1])[token].tolist() for name, value in arrays.items()}
    return {"exact_ordered_equality": bool(ordered.all()),
            "per_token_selected_set_equality": bool(selected.all()),
            "first_token_with_ordered_mismatch": None if not ordered_bad.size else int(ordered_bad[0, -1]),
            "first_token_with_selected_set_mismatch": None if not selected_bad.size else int(selected_bad[0, -1]),
            "ordered_index_lists_at_first_mismatch": lists}


def align_router_weights(indices, weights, expert_count=384):
    indices = np.asarray(indices); weights = np.asarray(weights)
    if indices.shape != weights.shape:
        raise ValueError("router index and weight shapes differ")
    aligned = np.zeros(indices.shape[:-1] + (expert_count,), dtype=np.float32)
    np.put_along_axis(aligned, indices.astype(np.int64), weights.astype(np.float32), axis=-1)
    return aligned


def compare_block_components(default_npz, math_npz, capture_dir, attention_mask, criterion):
    expected = set(component_names())
    for label, archive in (("default", default_npz), ("math", math_npz)):
        if set(archive.files) != expected:
            raise ValueError(f"{label} component reference inventory differs from exact 110 names")
    manifest = read_component_manifest(capture_dir / "block-components-diagnostics.tsv")
    mask = np.asarray(attention_mask, dtype=bool)
    results = []
    first_router_set = None
    first_material = None
    classification = None
    indices_by_block = {}
    decode_provenance = []
    for name in component_names():
        block = int(name[15:17]); suffix = name.split("__", 1)[1]
        dtype, dims, raw = manifest[name]
        cpp = decode_component_raw(name, dtype, dims, raw, decode_provenance)
        default = np.asarray(default_npz[name]); math = np.asarray(math_npz[name])
        expected_width = (384 if suffix in {"router_logits", "router_probabilities", "router_selection_scores"}
                          else 12 if suffix in {"router_topk_indices", "router_topk_weights"}
                          else 1 if suffix == "identity_weight_sum" else 3072)
        expected_shape = (1, mask.size, expected_width)
        if cpp.shape != expected_shape or default.shape != expected_shape or math.shape != expected_shape:
            raise ValueError(f"{name}: expected exact component shape {expected_shape}, got "
                             f"cpp={cpp.shape}, default={default.shape}, math={math.shape}")
        if cpp.shape[1] != mask.size or default.shape[1] != mask.size or math.shape[1] != mask.size:
            raise ValueError(f"{name}: component token count differs from attention mask")
        cpp, default, math = cpp[:, mask, :], default[:, mask, :], math[:, mask, :]
        if suffix == "router_topk_indices":
            report = router_index_report(cpp, default, math)
            indices_by_block[block] = (cpp, default, math)
            if not report["per_token_selected_set_equality"] and first_router_set is None:
                first_router_set = block
            results.append({"name": name, "physical_block": block, "kind": "router_indices", **report})
            continue
        comparisons = {
            "cpp_vs_python_default": all_block_result(default, cpp, criterion),
            "cpp_vs_python_math": all_block_result(math, cpp, criterion),
            "python_default_vs_python_math": all_block_result(default, math, criterion)}
        hidden = cpp.shape[-1] == 3072
        if not hidden:
            for comparison in comparisons.values():
                comparison.pop("maximum_normalized_violation_under_diagnostic_criterion")
                comparison.pop("within_diagnostic_criterion")
        if suffix == "router_topk_weights":
            require_indices = indices_by_block.get(block)
            if require_indices is None:
                raise ValueError(f"{name}: router indices must precede weights")
            cpp_i, default_i, math_i = require_indices
            aligned = (align_router_weights(cpp_i, cpp), align_router_weights(default_i, default),
                       align_router_weights(math_i, math))
            aligned_comparisons = {
                "cpp_vs_python_default": all_block_result(aligned[1], aligned[0], criterion),
                "cpp_vs_python_math": all_block_result(aligned[2], aligned[0], criterion),
                "python_default_vs_python_math": all_block_result(aligned[1], aligned[2], criterion)}
            for comparison in aligned_comparisons.values():
                comparison.pop("maximum_normalized_violation_under_diagnostic_criterion")
                comparison.pop("within_diagnostic_criterion")
            result = {"name": name, "physical_block": block, "kind": "router_weights",
                      "returned_topk_order": comparisons, "expert_id_aligned": aligned_comparisons}
        else:
            result = {"name": name, "physical_block": block, "kind": "floating",
                      "hidden_size_diagnostic_criterion": hidden, "comparisons": comparisons}
        results.append(result)
        if (hidden and first_material is None and
                not comparisons["cpp_vs_python_default"]["within_diagnostic_criterion"] and
                not comparisons["cpp_vs_python_math"]["within_diagnostic_criterion"]):
            first_material = name
            classification = {
                "attention_output": "attention output", "post_attention_residual": "post-attention residual",
                "dense_output": "dense output", "identity_residual": "identity residual",
                "moe_shortcut": "complete MoE shortcut", "block_output": "final block addition"}.get(
                    suffix, suffix)
    return {"accepted": False, "array_count": 110,
            "diagnostic_criterion_source": "bf16.physical_block_02",
            "component_decode_provenance": decode_provenance,
            "first_physical_block_with_router_selected_set_difference": first_router_set,
            "first_raw_router_difference": first_router_set,
            "first_component_outside_diagnostic_criterion_vs_both_backends": first_material,
            "first_large_discrepancy_classification": classification,
            "first_large_discrepancy_classification_semantics":
                "local-component evidence only; routing causality is deferred to numerical attribution",
            "raw_router_classification_deprecated": True,
            "components": results}


def bf16_round_to_float32(values):
    """IEEE float32 -> BF16 round-to-nearest-even -> float32, including specials."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    exponent = bits & np.uint32(0x7f800000)
    mantissa = bits & np.uint32(0x007fffff)
    special = exponent == np.uint32(0x7f800000)
    rounded = bits + np.uint32(0x00007fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    result = np.where(special, bits, rounded) & np.uint32(0xffff0000)
    # Preserve NaN classification even when its payload exists only in the
    # discarded low sixteen bits.
    result = np.where(special & (mantissa != 0), result | np.uint32(0x00010000), result)
    return np.ascontiguousarray(result).view(np.float32).reshape(array.shape)


def exactly_bf16_representable(values):
    values = np.asarray(values, dtype=np.float32)
    return bool(np.array_equal(values, bf16_round_to_float32(values), equal_nan=True))


def material_metric_change(baseline, rounded):
    """Classify RMS change using a 1% relative floor and 1e-12 absolute floor."""
    delta = float(rounded - baseline)
    threshold = max(1e-12, 0.01 * max(abs(float(baseline)), abs(float(rounded))))
    return "unchanged" if abs(delta) <= threshold else ("improved" if delta < 0 else "worsened")


def metric_delta(baseline, rounded, cosine=False):
    absolute = float(rounded - baseline)
    percentage = None if baseline == 0 else float(100.0 * absolute / abs(baseline))
    direction = material_metric_change(-baseline if cosine else baseline,
                                       -rounded if cosine else rounded)
    return {"absolute_delta": absolute, "percentage_delta": percentage,
            "material_classification": direction}


def load_source_dtype_evidence(path):
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    result = {}
    def visit(value):
        if isinstance(value, dict):
            name = value.get("name")
            dtype = value.get("source_torch_dtype", value.get("source_dtype"))
            if name in set(component_names()) and dtype is not None:
                result[name] = str(dtype)
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(payload)
    return result


def _profile_capture_dir(root, case):
    root = Path(root)
    candidate = root / case
    return candidate if (candidate / "block-components-diagnostics.tsv").is_file() else root


def validate_profile_capture(root, case, expect_rounding, baseline_identity=None):
    root = Path(root); capture = _profile_capture_dir(root, case)
    direct = read_manifest(capture / "captures.tsv")
    if set(direct) != DIRECT_NAMES or len(direct) != 20:
        raise ValueError(f"{root}: standard manifest must contain exact 20-name inventory")
    components = read_component_manifest(capture / "block-components-diagnostics.tsv")
    if len(components) != 110:
        raise ValueError(f"{root}: component manifest must contain exactly 110 rows")
    metadata = {}
    if expect_rounding:
        metadata_path = root / "capture-run-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        if metadata.get("longcat_bf16_boundary_rounding") is not True:
            raise ValueError("rounded profile metadata must explicitly enable boundary rounding")
    elif baseline_identity != "legacy-default-off":
        raise ValueError("baseline profile must be explicitly identified as legacy-default-off")
    inputs = json.loads((capture / "inputs.json").read_text(encoding="ascii"))
    provenance = []
    for name, (dtype, dims, raw) in components.items():
        array = decode_component_raw(name, dtype, dims, raw, provenance)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name}: profile contains non-finite values")
    index_representations = {row["representation"] for row in provenance}
    expected_representation = "compact_logical" if expect_rounding else "legacy_strided_argsort_view"
    if index_representations != {expected_representation}:
        raise ValueError(f"{root}: router indices must use {expected_representation}")
    return capture, components, inputs, metadata


def _persistent_failure(rows):
    failures = [row["physical_block"] for row in rows
                if row["suffix"] == "block_output" and
                not row["baseline_or_rounded_vs_default"]["within_diagnostic_criterion"] and
                not row["baseline_or_rounded_vs_math"]["within_diagnostic_criterion"]]
    for block in failures:
        later = [row for row in rows if row["suffix"] == "block_output" and
                 row["physical_block"] >= block]
        if later and all(not row["baseline_or_rounded_vs_default"]["within_diagnostic_criterion"] and
                         not row["baseline_or_rounded_vs_math"]["within_diagnostic_criterion"]
                         for row in later):
            return block
    return None


def compare_component_profiles(default_npz, math_npz, baseline_root, rounded_root, case,
                               attention_mask, criterion, baseline_identity,
                               execution_context, source_dtype_evidence=None,
                               reference_argmax=None):
    if execution_context != "cpu-flash-disabled-threads-0-bf16":
        raise ValueError("profile replay requires explicit CPU/Flash-disabled/threads=0/BF16 context")
    expected = set(component_names())
    if set(default_npz.files) != expected or set(math_npz.files) != expected:
        raise ValueError("profile reference archives must contain the exact 110-name inventory")
    baseline_dir, baseline_manifest, baseline_inputs, _ = validate_profile_capture(
        baseline_root, case, False, baseline_identity)
    rounded_dir, rounded_manifest, rounded_inputs, rounded_metadata = validate_profile_capture(
        rounded_root, case, True)
    if baseline_inputs != rounded_inputs:
        raise ValueError("profile captures differ in case inputs or attention mask")
    mask = np.asarray(attention_mask, dtype=bool)
    source_dtypes = load_source_dtype_evidence(source_dtype_evidence)
    rows = []; aggregates = {}; first_improvement = None; first_regression = None
    grid = {side: [] for side in ("cpp_baseline", "cpp_rounded", "python_default", "python_math")}
    arrays = {side: {} for side in grid}
    for name in component_names():
        block = int(name[15:17]); suffix = name.split("__", 1)[1]
        b = decode_component_raw(name, *baseline_manifest[name])[:, mask, :]
        r = decode_component_raw(name, *rounded_manifest[name])[:, mask, :]
        d = np.asarray(default_npz[name])[:, mask, :]
        m = np.asarray(math_npz[name])[:, mask, :]
        for side, value in (("cpp_baseline", b), ("cpp_rounded", r),
                            ("python_default", d), ("python_math", m)):
            arrays[side][name] = value
            if np.issubdtype(value.dtype, np.floating):
                if not np.isfinite(value).all(): raise ValueError(f"{name}: non-finite {side}")
                if exactly_bf16_representable(value): grid[side].append(name)
        if suffix == "router_topk_indices":
            continue
        bd = all_block_result(d, b, criterion); bm = all_block_result(m, b, criterion)
        rd = all_block_result(d, r, criterion); rm = all_block_result(m, r, criterion)
        rb = all_block_result(b.astype(np.float32), r.astype(np.float32), criterion)
        hidden = b.shape[-1] == 3072
        metrics = ("maximum_absolute_error", "mean_absolute_error", "rms_error", "cosine_similarity")
        delta_default = {key: metric_delta(bd[key], rd[key], key == "cosine_similarity") for key in metrics}
        delta_math = {key: metric_delta(bm[key], rm[key], key == "cosine_similarity") for key in metrics}
        if hidden:
            key = "maximum_normalized_violation_under_diagnostic_criterion"
            delta_default[key] = metric_delta(bd[key], rd[key]); delta_math[key] = metric_delta(bm[key], rm[key])
        state_default = material_metric_change(bd["rms_error"], rd["rms_error"])
        state_math = material_metric_change(bm["rms_error"], rm["rms_error"])
        transition_default = ("pass -> fail" if bd["within_diagnostic_criterion"] and not rd["within_diagnostic_criterion"]
                              else "fail -> pass" if not bd["within_diagnostic_criterion"] and rd["within_diagnostic_criterion"]
                              else "unchanged") if hidden else "not_applicable"
        transition_math = ("pass -> fail" if bm["within_diagnostic_criterion"] and not rm["within_diagnostic_criterion"]
                           else "fail -> pass" if not bm["within_diagnostic_criterion"] and rm["within_diagnostic_criterion"]
                           else "unchanged") if hidden else "not_applicable"
        row = {"name": name, "physical_block": block, "suffix": suffix,
               "baseline_cpp_vs_python_default": bd, "baseline_cpp_vs_python_math": bm,
               "rounded_cpp_vs_python_default": rd, "rounded_cpp_vs_python_math": rm,
               "rounded_cpp_vs_baseline_cpp": rb,
               "deltas_vs_python_default": delta_default, "deltas_vs_python_math": delta_math,
               "classification_vs_python_default": state_default,
               "classification_vs_python_math": state_math,
               "pass_state_transition_vs_python_default": transition_default,
               "pass_state_transition_vs_python_math": transition_math,
               "exactly_bf16_representable": {side: name in grid[side] for side in grid},
               "observed_python_source_dtype": source_dtypes.get(name)}
        rows.append(row)
        if first_improvement is None and (state_default == "improved" or state_math == "improved"):
            first_improvement = name
        if first_regression is None and (state_default == "worsened" or state_math == "worsened"):
            first_regression = name
        for backend, state, transition in (("python_default", state_default, transition_default),
                                           ("python_math", state_math, transition_math)):
            bucket = aggregates.setdefault(backend, {}).setdefault(suffix,
                {"improved": 0, "worsened": 0, "unchanged": 0, "pass -> fail": 0, "fail -> pass": 0})
            bucket[state] += 1
            if transition in ("pass -> fail", "fail -> pass"): bucket[transition] += 1

    routing = []
    for block in range(0, 10, 2):
        p = f"physical_block_{block:02d}__"
        names = {side: arrays[side][p + "router_topk_indices"] for side in arrays}
        shortcut_row = next(row for row in rows if row["name"] == p + "moe_shortcut")
        for token in range(names["cpp_baseline"].shape[1]):
            sets = {side: set(value[0, token].tolist()) for side, value in names.items()}
            real = {side: {x for x in values if x < REAL_EXPERT_COUNT} for side, values in sets.items()}
            changed = real["cpp_baseline"] != real["cpp_rounded"]
            if not changed: movement = "unchanged"
            else:
                before = (real["cpp_baseline"] == real["python_default"], real["cpp_baseline"] == real["python_math"])
                after = (real["cpp_rounded"] == real["python_default"], real["cpp_rounded"] == real["python_math"])
                movement = ("toward default" if after[0] and not before[0] else
                            "toward math" if after[1] and not before[1] else
                            "away from both" if any(before) and not any(after) else "toward neither")
            routing.append({"physical_block": block, "token": token,
                "baseline_vs_rounded_returned_order_equal": bool(np.array_equal(
                    names["cpp_baseline"][0, token], names["cpp_rounded"][0, token])),
                "baseline_vs_rounded_raw_set_equal": sets["cpp_baseline"] == sets["cpp_rounded"],
                "baseline_vs_rounded_real_expert_set_equal": not changed,
                "baseline_vs_rounded_identity_presence_equal":
                    any(x >= REAL_EXPERT_COUNT for x in sets["cpp_baseline"]) ==
                    any(x >= REAL_EXPERT_COUNT for x in sets["cpp_rounded"]),
                "pairwise_against_python": {
                    f"{profile}_vs_{backend}": {
                        "returned_order_equal": bool(np.array_equal(
                            names[profile][0, token], names[backend][0, token])),
                        "raw_set_equal": sets[profile] == sets[backend],
                        "real_expert_set_equal": real[profile] == real[backend],
                        "identity_presence_equal":
                            any(x >= REAL_EXPERT_COUNT for x in sets[profile]) ==
                            any(x >= REAL_EXPERT_COUNT for x in sets[backend]),
                    }
                    for profile in ("cpp_baseline", "cpp_rounded")
                    for backend in ("python_default", "python_math")},
                "boundary_rounding_changed_real_expert": changed, "movement": movement,
                "shortcut_remains_within_criterion_vs_both":
                    shortcut_row["rounded_cpp_vs_python_default"]["within_diagnostic_criterion"] and
                    shortcut_row["rounded_cpp_vs_python_math"]["within_diagnostic_criterion"],
                "shortcut_rms_delta_vs_python_default":
                    shortcut_row["deltas_vs_python_default"]["rms_error"],
                "shortcut_rms_delta_vs_python_math":
                    shortcut_row["deltas_vs_python_math"]["rms_error"]})

    baseline_view = [{"physical_block": row["physical_block"], "suffix": row["suffix"],
        "baseline_or_rounded_vs_default": row["baseline_cpp_vs_python_default"],
        "baseline_or_rounded_vs_math": row["baseline_cpp_vs_python_math"]} for row in rows]
    rounded_view = [{"physical_block": row["physical_block"], "suffix": row["suffix"],
        "baseline_or_rounded_vs_default": row["rounded_cpp_vs_python_default"],
        "baseline_or_rounded_vs_math": row["rounded_cpp_vs_python_math"]} for row in rows]
    hidden = [row for row in rows if row["suffix"] in {"block_input", "attention_norm", "attention_output",
              "post_attention_residual", "ffn_norm", "dense_output", "block_output", "identity_residual", "moe_shortcut"}]
    totals = {}
    for backend in ("default", "math"):
        totals[backend] = {profile: sum(row[f"{profile}_cpp_vs_python_{backend}"]["rms_error"] for row in hidden)
                           for profile in ("baseline", "rounded")}
    baseline_argmax = json.loads((baseline_dir / "decoding.json").read_text())["argmax_id"]
    rounded_argmax = json.loads((rounded_dir / "decoding.json").read_text())["argmax_id"]
    observed_bf16 = sorted(name for name, dtype in source_dtypes.items() if "bfloat16" in dtype)
    observed_f32 = sorted(name for name, dtype in source_dtypes.items() if "float32" in dtype)
    hidden_policy_enabled = rounded_metadata.get("longcat_bf16_hidden_surface_rounding") is True
    enforced_policy = set(observed_bf16) if hidden_policy_enabled else set()
    observed_bf16_on_grid = sorted(enforced_policy & set(grid["cpp_rounded"]))
    return {"accepted": False, "kind": "longcat-next-component-profile-diff",
        "material_change_definition": "RMS delta exceeds max(1e-12, 1% of larger RMS)",
        "diagnostic_criterion_source": "bf16.physical_block_02", "components": rows,
        "aggregate_totals_by_backend_and_suffix": aggregates,
        "first_material_improvement": first_improvement, "first_material_regression": first_regression,
        "profile_with_lower_total_hidden_rms_vs_python_default": min(totals["default"], key=totals["default"].get),
        "profile_with_lower_total_hidden_rms_vs_python_math": min(totals["math"], key=totals["math"].get),
        "profile_with_later_persistent_failure_vs_both":
            "baseline" if (_persistent_failure(baseline_view) or 999) > (_persistent_failure(rounded_view) or 999)
            else "rounded" if (_persistent_failure(rounded_view) or 999) > (_persistent_failure(baseline_view) or 999)
            else "tie",
        "profile_with_matching_reference_argmax":
            "both" if baseline_argmax == rounded_argmax == reference_argmax else
            "baseline" if baseline_argmax == reference_argmax else
            "rounded" if rounded_argmax == reference_argmax else "neither",
        "boundary_rounding_overall_verdict":
            "mixed tradeoff; boundary rounding matches the Python residual dtype contract but does not restore numerical or decoding parity",
        "python_components_observed_bfloat16": observed_bf16,
        "python_components_observed_float32": observed_f32,
        "cpp_baseline_components_exactly_bf16_grid": sorted(grid["cpp_baseline"]),
        "cpp_rounded_components_exactly_bf16_grid": sorted(grid["cpp_rounded"]),
        "observed_bfloat16_policy_surface_count": len(observed_bf16),
        "observed_bfloat16_policy_surfaces_on_bf16_grid": observed_bf16_on_grid,
        "observed_bfloat16_policy_coverage_complete":
            bool(observed_bf16) and set(observed_bf16_on_grid) == set(observed_bf16),
        "observed_float32_policy_surfaces_rounded": sorted(enforced_policy & set(observed_f32)),
        "representability_is_not_execution_dtype": True,
        "routing_profile_comparison": routing}


def metric_change(raw, rounded, lower_is_better=True):
    if raw == rounded:
        return "unchanged"
    improved = rounded < raw if lower_is_better else rounded > raw
    return "improved" if improved else "worsened"


def rounding_comparison(reference, raw_cpp, criterion, hidden):
    rounded_cpp = bf16_round_to_float32(raw_cpp)
    raw = all_block_result(reference, raw_cpp, criterion)
    rounded = all_block_result(reference, rounded_cpp, criterion)
    if not hidden:
        for report in (raw, rounded):
            report.pop("maximum_normalized_violation_under_diagnostic_criterion")
            report.pop("within_diagnostic_criterion")
    changes = {
        "maximum_absolute_error": metric_change(raw["maximum_absolute_error"], rounded["maximum_absolute_error"]),
        "mean_absolute_error": metric_change(raw["mean_absolute_error"], rounded["mean_absolute_error"]),
        "rms_error": metric_change(raw["rms_error"], rounded["rms_error"]),
        "cosine_similarity": metric_change(raw["cosine_similarity"], rounded["cosine_similarity"], False),
    }
    return {"raw_cpp_f32": raw, "bf16_rounded_cpp": rounded,
            "metric_changes_after_bf16_rounding": changes,
            "pass_state_changed": bool(hidden and
                raw["within_diagnostic_criterion"] != rounded["within_diagnostic_criterion"])}


def semantic_router_report(cpp_indices, default_indices, math_indices,
                           cpp_identity_sum, default_identity_sum, math_identity_sum):
    arrays = {"cpp": np.asarray(cpp_indices), "python_default": np.asarray(default_indices),
              "python_math": np.asarray(math_indices)}
    flat = {name: value.reshape(-1, value.shape[-1]) for name, value in arrays.items()}
    token_rows = []
    first_real_mismatch = None
    first_real_lists = None
    for token in range(next(iter(flat.values())).shape[0]):
        ordered = {name: row[token].tolist() for name, row in flat.items()}
        raw_sets = {name: sorted(set(values)) for name, values in ordered.items()}
        real_sets = {name: sorted(value for value in values if value < REAL_EXPERT_COUNT)
                     for name, values in raw_sets.items()}
        identity_counts = {name: sum(value >= REAL_EXPERT_COUNT for value in values)
                           for name, values in ordered.items()}
        exact_order = len({tuple(values) for values in ordered.values()}) == 1
        raw_equal = len({tuple(values) for values in raw_sets.values()}) == 1
        real_equal = len({tuple(values) for values in real_sets.values()}) == 1
        identity_count_equal = len(set(identity_counts.values())) == 1
        identity_presence = {name: count > 0 for name, count in identity_counts.items()}
        identity_presence_equal = len(set(identity_presence.values())) == 1
        identity_only = (not raw_equal and real_equal and identity_count_equal and
                         identity_presence_equal)
        pairs = (("cpp_vs_python_default", "cpp", "python_default"),
                 ("cpp_vs_python_math", "cpp", "python_math"),
                 ("python_default_vs_python_math", "python_default", "python_math"))
        if not real_equal and first_real_mismatch is None:
            first_real_mismatch = token
            first_real_lists = real_sets
        token_rows.append({
            "token": token, "exact_returned_index_order_equality": exact_order,
            "raw_expert_id_selected_set_equality": raw_equal,
            "selected_real_expert_set_equality": real_equal,
            "pairwise_exact_returned_index_order_equality": {
                label: ordered[left] == ordered[right] for label, left, right in pairs},
            "pairwise_raw_expert_id_selected_set_equality": {
                label: raw_sets[left] == raw_sets[right] for label, left, right in pairs},
            "pairwise_selected_real_expert_set_equality": {
                label: real_sets[left] == real_sets[right] for label, left, right in pairs},
            "identity_expert_counts": identity_counts,
            "identity_expert_presence_equivalent": identity_presence_equal,
            "pairwise_identity_expert_presence_equivalent": {
                label: identity_presence[left] == identity_presence[right]
                for label, left, right in pairs},
            "raw_set_mismatch_is_identity_only_id_substitution": identity_only,
        })
    identity_metrics = {
        "cpp_vs_python_default": all_block_result(default_identity_sum, cpp_identity_sum,
                                                    {"atol": 0.125, "rtol": 0.03125}),
        "cpp_vs_python_math": all_block_result(math_identity_sum, cpp_identity_sum,
                                                {"atol": 0.125, "rtol": 0.03125}),
        "python_default_vs_python_math": all_block_result(default_identity_sum, math_identity_sum,
                                                           {"atol": 0.125, "rtol": 0.03125}),
    }
    for report in identity_metrics.values():
        report.pop("maximum_normalized_violation_under_diagnostic_criterion")
        report.pop("within_diagnostic_criterion")
    return {
        "tokens": token_rows, "identity_weight_sum_comparisons": identity_metrics,
        "first_token_with_real_expert_set_mismatch": first_real_mismatch,
        "real_expert_lists_at_first_mismatch": first_real_lists,
        "has_real_expert_set_difference": first_real_mismatch is not None,
        "has_identity_presence_difference": any(not row["identity_expert_presence_equivalent"] for row in token_rows),
        "has_identity_only_id_substitution": any(row["raw_set_mismatch_is_identity_only_id_substitution"] for row in token_rows),
        "has_cpp_vs_python_default_real_expert_set_difference": any(
            not row["pairwise_selected_real_expert_set_equality"]["cpp_vs_python_default"] for row in token_rows),
        "has_cpp_vs_python_math_real_expert_set_difference": any(
            not row["pairwise_selected_real_expert_set_equality"]["cpp_vs_python_math"] for row in token_rows),
        "has_cpp_vs_python_default_identity_presence_difference": any(
            not row["pairwise_identity_expert_presence_equivalent"]["cpp_vs_python_default"] for row in token_rows),
        "has_cpp_vs_python_math_identity_presence_difference": any(
            not row["pairwise_identity_expert_presence_equivalent"]["cpp_vs_python_math"] for row in token_rows),
    }


def _vector_metrics(left, right):
    left = np.asarray(left, np.float32); right = np.asarray(right, np.float32)
    difference = left - right
    left64 = left.reshape(-1).astype(np.float64)
    right64 = right.reshape(-1).astype(np.float64)
    norm_product = np.linalg.norm(left64) * np.linalg.norm(right64)
    cosine = (float(np.dot(left64, right64) / norm_product) if norm_product else
              float(np.array_equal(left64, right64)))
    return {"maximum_absolute_difference": float(np.abs(difference).max(initial=0)),
            "rms_difference": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
            "cosine_similarity": cosine, "exact_equality": bool(np.array_equal(left, right))}


def reconstruct_router_correction_bias(probabilities, selection_scores):
    """Audit the captured float32 correction bias without assuming exact subtraction."""
    probabilities = np.asarray(probabilities, np.float32).reshape(-1, ROUTED_EXPERT_COUNT)
    selection_scores = np.asarray(selection_scores, np.float32).reshape(-1, ROUTED_EXPERT_COUNT)
    if probabilities.shape != selection_scores.shape or probabilities.shape[0] == 0:
        raise ValueError("router probability/selection-score shape mismatch")
    reconstructed = selection_scores - probabilities
    first = reconstructed[0]
    ranges = reconstructed.max(axis=0) - reconstructed.min(axis=0)
    byte_exact = all(row.tobytes() == first.tobytes() for row in reconstructed)
    return reconstructed, {
        "maximum_per_expert_range_across_tokens": float(ranges.max(initial=0)),
        "maximum_absolute_deviation_from_first_token":
            float(np.abs(reconstructed - first).max(initial=0)),
        "token_invariant_byte_exact": bool(byte_exact),
        "mean_reconstructed_bias_per_expert": reconstructed.mean(axis=0).tolist(),
    }


def router_token_cutoff(selection_scores, topk_indices, attended_token):
    """Rank the complete score vector; returned top-k order is membership only."""
    scores = np.asarray(selection_scores, np.float32).reshape(-1)
    selected_order = np.asarray(topk_indices).reshape(-1).astype(np.int64)
    if scores.size != ROUTED_EXPERT_COUNT or selected_order.size != 12:
        raise ValueError("router cutoff analysis requires 384 scores and 12 selected experts")
    if len(set(selected_order.tolist())) != 12 or np.any(selected_order < 0) or np.any(selected_order >= scores.size):
        raise ValueError("router cutoff analysis received invalid selected expert IDs")
    # Stable expert-ID tie break is diagnostic only; ranking never uses returned top-k order.
    ranked = np.lexsort((np.arange(scores.size), -scores))
    ranks = np.empty(scores.size, dtype=np.int64); ranks[ranked] = np.arange(1, scores.size + 1)
    selected = set(int(value) for value in selected_order)
    unselected = [expert for expert in range(scores.size) if expert not in selected]
    lowest = min(float(scores[expert]) for expert in selected)
    highest = max(float(scores[expert]) for expert in unselected)
    def expert_row(expert):
        return {"expert_id": int(expert), "rank": int(ranks[expert]),
                "selection_score": float(scores[expert]), "selected": expert in selected,
                "expert_class": "real" if expert < REAL_EXPERT_COUNT else "identity"}
    return {"attended_token": int(attended_token),
            "ranking_source": "complete_384_wide_selection_scores",
            "returned_topk_order_used_for_ranking": False,
            "selected_expert_set": sorted(selected),
            "selected_real_expert_set": sorted(expert for expert in selected if expert < REAL_EXPERT_COUNT),
            "selected_identity_expert_set": sorted(expert for expert in selected if expert >= REAL_EXPERT_COUNT),
            "lowest_selected_score": lowest, "highest_unselected_score": highest,
            "topk_cutoff_margin": lowest - highest,
            "ranked_experts_around_cutoff": [expert_row(ranked[rank - 1]) for rank in range(8, 17)],
            "expert_ranks": ranks, "scores": scores}


def disputed_router_token(left, right, left_name, right_name):
    left_set = set(left["selected_expert_set"]); right_set = set(right["selected_expert_set"])
    only_left = sorted(left_set - right_set); only_right = sorted(right_set - left_set)
    if not only_left and not only_right:
        return None
    def detail(expert):
        return {"expert_id": expert,
                "left_score": float(left["scores"][expert]),
                "right_score": float(right["scores"][expert]),
                "left_rank": int(left["expert_ranks"][expert]),
                "right_rank": int(right["expert_ranks"][expert]),
                "left_selected": expert in left_set, "right_selected": expert in right_set,
                "left_rank_distance_from_top12_boundary": int(left["expert_ranks"][expert]) - 12,
                "right_rank_distance_from_top12_boundary": int(right["expert_ranks"][expert]) - 12,
                "expert_class": "real" if expert < REAL_EXPERT_COUNT else "identity"}
    inversions = []
    for left_expert in only_left:
        for right_expert in only_right:
            left_gap = float(left["scores"][left_expert] - left["scores"][right_expert])
            right_gap = float(right["scores"][left_expert] - right["scores"][right_expert])
            inversions.append({"left_only_expert": left_expert, "right_only_expert": right_expert,
                "left_implementation_score_gap": left_gap,
                "right_implementation_score_gap": right_gap,
                "gap_sign_flips": bool((left_gap > 0 > right_gap) or (left_gap < 0 < right_gap))})
    delta = np.abs(left["scores"] - right["scores"])
    displaced_gaps = [abs(row[key]) for row in inversions for key in (
        "left_implementation_score_gap", "right_implementation_score_gap")]
    scale = {"maximum_absolute_selection_score_difference": float(delta.max(initial=0)),
             "rms_selection_score_difference": float(np.sqrt(np.mean(np.square(delta, dtype=np.float64)))),
             "left_cutoff_margin": left["topk_cutoff_margin"],
             "right_cutoff_margin": right["topk_cutoff_margin"],
             "minimum_absolute_displaced_expert_score_gap": min(displaced_gaps),
             "maximum_absolute_displaced_expert_score_gap": max(displaced_gaps)}
    if left["topk_cutoff_margin"] != 0:
        scale["maximum_score_delta_over_left_cutoff_margin"] = (
            scale["maximum_absolute_selection_score_difference"] / left["topk_cutoff_margin"])
    if right["topk_cutoff_margin"] != 0:
        scale["maximum_score_delta_over_right_cutoff_margin"] = (
            scale["maximum_absolute_selection_score_difference"] / right["topk_cutoff_margin"])
    return {"attended_token": left["attended_token"], "left_implementation": left_name,
            "right_implementation": right_name, "experts_selected_only_by_left": only_left,
            "experts_selected_only_by_right": only_right,
            "disputed_experts": [detail(expert) for expert in sorted(set(only_left + only_right))],
            "ordering_inversions": inversions,
            "has_ordering_inversion": any(row["gap_sign_flips"] for row in inversions),
            "numerical_scale": scale}


def router_cutoff_analysis(sides, prefix, attended_tokens):
    implementations = ("cpp", "default", "math")
    correction = {}; reconstructed = {}
    cutoffs = {}
    for implementation in implementations:
        reconstructed[implementation], correction[implementation] = reconstruct_router_correction_bias(
            sides[implementation][prefix + "router_probabilities"],
            sides[implementation][prefix + "router_selection_scores"])
        scores = sides[implementation][prefix + "router_selection_scores"].reshape(-1, ROUTED_EXPERT_COUNT)
        indices = sides[implementation][prefix + "router_topk_indices"].reshape(-1, 12)
        cutoffs[implementation] = [router_token_cutoff(scores[token], indices[token], attended_tokens[token])
                                   for token in range(scores.shape[0])]
    pair_specs = (("cpp_vs_python_default", "cpp", "default"),
                  ("cpp_vs_python_math", "cpp", "math"),
                  ("python_default_vs_python_math", "default", "math"))
    bias_pairs = {label: _vector_metrics(reconstructed[left].mean(axis=0),
                                         reconstructed[right].mean(axis=0))
                  for label, left, right in pair_specs}
    disputes = {}
    for label, left, right in pair_specs:
        disputes[label] = [report for token in range(len(attended_tokens))
            if (report := disputed_router_token(cutoffs[left][token], cutoffs[right][token], left, right))]
    # Remove large private rank/score vectors after all pairwise reports have been derived.
    public_cutoffs = {implementation: [{key: value for key, value in row.items()
        if key not in {"expert_ranks", "scores"}} for row in rows]
        for implementation, rows in cutoffs.items()}
    return {"reconstructed_correction_bias": {"implementations": correction,
                                               "pairwise_metrics": bias_pairs},
            "token_cutoffs": public_cutoffs, "pairwise_disputed_experts": disputes}


def _score_topk_indices(scores):
    scores = np.asarray(scores, np.float32).reshape(-1)
    if scores.size != ROUTED_EXPERT_COUNT:
        raise ValueError("router selection decomposition requires 384 scores")
    return np.lexsort((np.arange(scores.size), -scores))[:12].astype(np.int64)


def constant_bias_reconstruction(probabilities, selection_scores, captured_indices, attended_tokens):
    tokenwise_bias, bias_report = reconstruct_router_correction_bias(probabilities, selection_scores)
    constant_bias = tokenwise_bias.mean(axis=0, dtype=np.float32)
    probabilities = np.asarray(probabilities, np.float32).reshape(-1, ROUTED_EXPERT_COUNT)
    selection_scores = np.asarray(selection_scores, np.float32).reshape(-1, ROUTED_EXPERT_COUNT)
    captured_indices = np.asarray(captured_indices).reshape(-1, 12)
    reconstructed = np.asarray(probabilities + constant_bias[None, :], np.float32)
    difference = reconstructed - selection_scores
    membership = []
    for token in range(reconstructed.shape[0]):
        reconstructed_set = sorted(_score_topk_indices(reconstructed[token]).tolist())
        captured_set = sorted(captured_indices[token].astype(np.int64).tolist())
        membership.append({"attended_token": int(attended_tokens[token]),
                           "selected_set_equality": reconstructed_set == captured_set,
                           "reconstructed_selected_expert_set": reconstructed_set,
                           "captured_selected_expert_set": captured_set})
    report = {**bias_report,
        "maximum_absolute_score_reconstruction_error": float(np.abs(difference).max(initial=0)),
        "rms_score_reconstruction_error":
            float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
        "score_reconstruction_exact_equality": bool(np.array_equal(reconstructed, selection_scores)),
        "per_token_selected_set_equality": membership,
        "every_token_reproduces_captured_membership": all(row["selected_set_equality"] for row in membership)}
    return constant_bias, reconstructed, report


def _public_cutoff(scores, attended_token):
    cutoff = router_token_cutoff(scores, _score_topk_indices(scores), attended_token)
    return {key: value for key, value in cutoff.items() if key not in {"expert_ranks", "scores"}}


def correction_bias_dtype_grid_audit(left_bias, right_bias):
    left_bias = np.asarray(left_bias, np.float32); right_bias = np.asarray(right_bias, np.float32)
    candidates = {"raw_float32": left_bias,
                  "bf16_rne_then_float32": bf16_round_to_float32(left_bias),
                  "f16_rne_then_float32": left_bias.astype(np.float16).astype(np.float32)}
    reports = {}
    for name, candidate in candidates.items():
        metrics = _vector_metrics(candidate, right_bias)
        reports[name] = {**metrics,
            "exact_match_count_out_of_384": int(np.count_nonzero(candidate == right_bias))}
    return {"candidates": reports,
            "closest_candidate_by_rms": min(reports, key=lambda name: reports[name]["rms_difference"]),
            "diagnostic_only_not_a_runtime_dtype_verdict": True}


def probability_bias_pair_decomposition(sides, prefix, attended_tokens,
                                        left_name, right_name):
    constants = {}; native_scores = {}; reconstruction = {}
    for implementation in (left_name, right_name):
        constants[implementation], native_scores[implementation], reconstruction[implementation] = (
            constant_bias_reconstruction(
                sides[implementation][prefix + "router_probabilities"],
                sides[implementation][prefix + "router_selection_scores"],
                sides[implementation][prefix + "router_topk_indices"], attended_tokens))
    probabilities = {name: np.asarray(sides[name][prefix + "router_probabilities"], np.float32).reshape(
        -1, ROUTED_EXPERT_COUNT) for name in (left_name, right_name)}
    captured_scores = {name: np.asarray(sides[name][prefix + "router_selection_scores"], np.float32).reshape(
        -1, ROUTED_EXPERT_COUNT) for name in (left_name, right_name)}
    captured_sets = {name: [set(np.asarray(sides[name][prefix + "router_topk_indices"]).reshape(-1, 12)[token].tolist())
        for token in range(len(attended_tokens))] for name in (left_name, right_name)}
    coalition_specs = {
        "native_left": (left_name, left_name),
        "right_probabilities_only": (right_name, left_name),
        "right_bias_only": (left_name, right_name),
        "native_right": (right_name, right_name),
    }
    rows = []
    for token, attended_token in enumerate(attended_tokens):
        coalitions = {}
        coalition_scores = {}
        for label, (probability_source, bias_source) in coalition_specs.items():
            scores = np.asarray(probabilities[probability_source][token] + constants[bias_source], np.float32)
            coalition_scores[label] = scores
            cutoff = _public_cutoff(scores, attended_token)
            selected = set(cutoff["selected_expert_set"])
            coalitions[label] = {**cutoff,
                "selected_set_equality_vs_native_left": selected == captured_sets[left_name][token],
                "selected_set_equality_vs_native_right": selected == captured_sets[right_name][token]}
        native_equal = captured_sets[left_name][token] == captured_sets[right_name][token]
        probability_restores = coalitions["right_probabilities_only"]["selected_set_equality_vs_native_right"]
        bias_restores = coalitions["right_bias_only"]["selected_set_equality_vs_native_right"]
        reconstruction_valid = (reconstruction[left_name]["per_token_selected_set_equality"][token]["selected_set_equality"] and
                                reconstruction[right_name]["per_token_selected_set_equality"][token]["selected_set_equality"])
        both_restores = coalitions["native_right"]["selected_set_equality_vs_native_right"]
        if native_equal:
            classification = "native memberships already equal"
        elif not reconstruction_valid:
            classification = "neither hybrid reproduces right membership"
        elif probability_restores and bias_restores:
            classification = "both components independently sufficient"
        elif probability_restores:
            classification = "probability component sufficient"
        elif bias_restores:
            classification = "bias component sufficient"
        elif both_restores and reconstruction_valid:
            classification = "requires both components"
        else:
            classification = "neither hybrid reproduces right membership"
        only_left = sorted(captured_sets[left_name][token] - captured_sets[right_name][token])
        only_right = sorted(captured_sets[right_name][token] - captured_sets[left_name][token])
        disputed = []
        for expert in sorted(set(only_left + only_right)):
            total = float(captured_scores[left_name][token, expert] - captured_scores[right_name][token, expert])
            probability_delta = float(probabilities[left_name][token, expert] - probabilities[right_name][token, expert])
            bias_delta = float(constants[left_name][expert] - constants[right_name][expert])
            disputed.append({"expert_id": expert, "orientation": "left_minus_right",
                "total_selection_score_delta": total, "probability_delta": probability_delta,
                "constant_bias_delta": bias_delta,
                "float32_reconstruction_residual": total - probability_delta - bias_delta})
        inversions = []
        for left_expert in only_left:
            for right_expert in only_right:
                def gap(values): return float(values[left_expert] - values[right_expert])
                left_gap = gap(captured_scores[left_name][token]); right_gap = gap(captured_scores[right_name][token])
                probability_change = gap(probabilities[left_name][token]) - gap(probabilities[right_name][token])
                bias_change = float((constants[left_name][left_expert] - constants[left_name][right_expert]) -
                                    (constants[right_name][left_expert] - constants[right_name][right_expert]))
                gap_change = left_gap - right_gap
                probability_hybrid_gap = gap(coalition_scores["right_probabilities_only"])
                bias_hybrid_gap = gap(coalition_scores["right_bias_only"])
                combination_gap = gap(coalition_scores["native_right"])
                sign_flip = lambda first, second: bool((first > 0 > second) or (first < 0 < second))
                probability_flips = sign_flip(left_gap, probability_hybrid_gap)
                bias_flips = sign_flip(left_gap, bias_hybrid_gap)
                inversions.append({"left_only_expert": left_expert, "right_only_expert": right_expert,
                    "left_native_score_gap": left_gap, "right_native_score_gap": right_gap,
                    "score_gap_change_left_minus_right": gap_change,
                    "probability_gap_change": probability_change, "bias_gap_change": bias_change,
                    "decomposition_residual": gap_change - probability_change - bias_change,
                    "probability_component_alone_reverses_ordering": probability_flips,
                    "bias_component_alone_reverses_ordering": bias_flips,
                    "only_combination_reproduces_native_inversion":
                        bool(reconstruction_valid and sign_flip(left_gap, right_gap) and
                             sign_flip(left_gap, combination_gap) and
                             not probability_flips and not bias_flips)})
        rows.append({"attended_token": int(attended_token), "coalitions": coalitions,
            "native_left_membership_reconstruction_valid":
                reconstruction[left_name]["per_token_selected_set_equality"][token]["selected_set_equality"],
            "native_right_membership_reconstruction_valid":
                reconstruction[right_name]["per_token_selected_set_equality"][token]["selected_set_equality"],
            "right_probabilities_only_restores_right_membership": bool(
                not native_equal and reconstruction_valid and probability_restores),
            "right_bias_only_restores_right_membership": bool(
                not native_equal and reconstruction_valid and bias_restores),
            "both_required_for_right_membership": bool(not native_equal and not probability_restores and
                                                        not bias_restores and both_restores and reconstruction_valid),
            "neither_hybrid_restores_right_membership": bool(not native_equal and (
                not reconstruction_valid or (not probability_restores and not bias_restores and not both_restores))),
            "membership_classification": classification,
            "membership_classification_decisive": bool(reconstruction_valid),
            "disputed_expert_delta_decomposition": disputed,
            "ordering_inversion_decomposition": inversions})
    return {"left_implementation": left_name, "right_implementation": right_name,
        "native_reconstruction": reconstruction, "tokens": rows,
        "correction_bias_dtype_grid_audit": correction_bias_dtype_grid_audit(
            constants[left_name], constants[right_name])}


def router_probability_bias_decomposition(sides, prefix, attended_tokens):
    pairs = (("cpp_vs_python_default", "cpp", "default"),
             ("cpp_vs_python_math", "cpp", "math"),
             ("python_default_vs_python_math", "default", "math"))
    return {label: probability_bias_pair_decomposition(
        sides, prefix, attended_tokens, left, right) for label, left, right in pairs}


def diagnostic_softmax_stable_float32(logits):
    values = np.asarray(logits, np.float32)
    shifted = np.asarray(values - np.max(values, axis=-1, keepdims=True), np.float32)
    exponential = np.asarray(np.exp(shifted), np.float32)
    denominator = np.sum(exponential, axis=-1, keepdims=True, dtype=np.float32)
    result = np.asarray(exponential / denominator, np.float32)
    if not np.isfinite(result).all():
        raise ValueError("stable_float32 diagnostic softmax produced non-finite values")
    return result


def diagnostic_softmax_stable_float64_then_float32(logits):
    values = np.asarray(logits, np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    result = np.asarray(np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True), np.float32)
    if not np.isfinite(result).all():
        raise ValueError("stable_float64_then_float32 diagnostic softmax produced non-finite values")
    return result


DIAGNOSTIC_SOFTMAX_REFERENCES = {
    "stable_float32": diagnostic_softmax_stable_float32,
    "stable_float64_then_float32": diagnostic_softmax_stable_float64_then_float32,
}


def diagnostic_softmax_residual(logits, captured_probabilities, reference):
    captured = np.asarray(captured_probabilities, np.float32).reshape(-1, ROUTED_EXPERT_COUNT)
    softmax = reference(np.asarray(logits, np.float32).reshape(-1, ROUTED_EXPERT_COUNT))
    residual = np.asarray(captured - softmax, np.float32)
    reconstructed = np.asarray(softmax + residual, np.float32)
    difference = reconstructed - captured
    probability_difference = softmax - captured
    metrics = _vector_metrics(softmax, captured)
    return softmax, residual, {"maximum_absolute_difference_vs_captured_probabilities":
            metrics["maximum_absolute_difference"],
        "rms_difference_vs_captured_probabilities": metrics["rms_difference"],
        "cosine_similarity_vs_captured_probabilities": metrics["cosine_similarity"],
        "exact_equality_vs_captured_probabilities": metrics["exact_equality"],
        "maximum_absolute_probability_sum_error":
            float(np.abs(np.sum(softmax, axis=-1, dtype=np.float32) - np.float32(1)).max(initial=0)),
        "minimum_probability": float(softmax.min(initial=np.inf)),
        "maximum_probability": float(softmax.max(initial=-np.inf)), "finite": bool(np.isfinite(softmax).all()),
        "maximum_absolute_residual": float(np.abs(residual).max(initial=0)),
        "rms_residual": float(np.sqrt(np.mean(np.square(residual, dtype=np.float64)))),
        "residual_sum": float(np.sum(residual, dtype=np.float64)),
        "minimum_residual": float(residual.min(initial=np.inf)),
        "maximum_residual": float(residual.max(initial=-np.inf)),
        "residual_addback_exactly_reconstructs_captured_probabilities": bool(np.array_equal(reconstructed, captured)),
        "reconstruction_maximum_absolute_error": float(np.abs(difference).max(initial=0)),
        "reconstruction_rms_error": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
        "captured_minus_softmax_maximum_absolute": float(np.abs(probability_difference).max(initial=0))}


def centered_logit_comparison(left_logits, right_logits):
    left = np.asarray(left_logits, np.float32); right = np.asarray(right_logits, np.float32)
    left_max = float(left.max()); right_max = float(right.max())
    left_mean = float(np.mean(left, dtype=np.float32)); right_mean = float(np.mean(right, dtype=np.float32))
    return {"raw_logits": _vector_metrics(left, right),
        "max_centered_logits": _vector_metrics(
            np.asarray(left - left_max, np.float32), np.asarray(right - right_max, np.float32)),
        "mean_centered_logits": _vector_metrics(
            np.asarray(left - left_mean, np.float32), np.asarray(right - right_mean, np.float32)),
        "left_tokenwise_maximum": left_max, "right_tokenwise_maximum": right_max,
        "maximum_difference_left_minus_right": left_max - right_max,
        "left_tokenwise_mean": left_mean, "right_tokenwise_mean": right_mean,
        "mean_difference_left_minus_right": left_mean - right_mean}


def logit_softmax_pair_variant(sides, prefix, attended_tokens, left_name, right_name,
                               reference_name, reference):
    logits = {name: np.asarray(sides[name][prefix + "router_logits"], np.float32).reshape(
        -1, ROUTED_EXPERT_COUNT) for name in (left_name, right_name)}
    probabilities = {name: np.asarray(sides[name][prefix + "router_probabilities"], np.float32).reshape(
        -1, ROUTED_EXPERT_COUNT) for name in (left_name, right_name)}
    captured_indices = {name: np.asarray(sides[name][prefix + "router_topk_indices"]).reshape(-1, 12)
                        for name in (left_name, right_name)}
    left_bias, _, _ = constant_bias_reconstruction(
        sides[left_name][prefix + "router_probabilities"],
        sides[left_name][prefix + "router_selection_scores"],
        sides[left_name][prefix + "router_topk_indices"], attended_tokens)
    softmax = {}; residual = {}; reconstruction = {}
    for name in (left_name, right_name):
        softmax[name], residual[name], reconstruction[name] = diagnostic_softmax_residual(
            logits[name], probabilities[name], reference)
    rows = []
    for token, attended_token in enumerate(attended_tokens):
        captured_outcomes = {name: set(_score_topk_indices(np.asarray(
            probabilities[name][token] + left_bias, np.float32)).tolist()) for name in (left_name, right_name)}
        intended = {name: set(captured_indices[name][token].tolist()) for name in (left_name, right_name)}
        left_valid = captured_outcomes[left_name] == intended[left_name]
        right_valid = captured_outcomes[right_name] == intended[right_name]
        specs = {"native_left_probability": (left_name, left_name),
                 "right_logits_only": (right_name, left_name),
                 "right_softmax_residual_only": (left_name, right_name),
                 "native_right_probability": (right_name, right_name)}
        coalitions = {}; coalition_probabilities = {}
        all_finite = True
        for label, (logit_source, residual_source) in specs.items():
            probability = np.asarray(softmax[logit_source][token] + residual[residual_source][token], np.float32)
            coalition_probabilities[label] = probability
            all_finite &= bool(np.isfinite(probability).all())
            cutoff = _public_cutoff(np.asarray(probability + left_bias, np.float32), attended_token)
            selected = set(cutoff["selected_expert_set"])
            coalitions[label] = {**cutoff,
                "selected_set_equality_vs_left_probability_outcome": selected == captured_outcomes[left_name],
                "selected_set_equality_vs_right_probability_outcome": selected == captured_outcomes[right_name],
                "probability_minimum": float(probability.min(initial=np.inf)),
                "probability_maximum": float(probability.max(initial=-np.inf)),
                "probability_sum": float(np.sum(probability, dtype=np.float32)),
                "all_probability_values_finite": bool(np.isfinite(probability).all())}
        native_residual_valid = all(reconstruction[name][
            "residual_addback_exactly_reconstructs_captured_probabilities"] for name in (left_name, right_name))
        decisive = bool(left_valid and right_valid and native_residual_valid and all_finite)
        outcomes_equal = captured_outcomes[left_name] == captured_outcomes[right_name]
        logit_restores = coalitions["right_logits_only"]["selected_set_equality_vs_right_probability_outcome"]
        residual_restores = coalitions["right_softmax_residual_only"]["selected_set_equality_vs_right_probability_outcome"]
        both_restores = coalitions["native_right_probability"]["selected_set_equality_vs_right_probability_outcome"]
        if outcomes_equal: classification = "native probability outcomes already equal"
        elif not decisive: classification = "reconstruction not decisive"
        elif logit_restores and residual_restores: classification = "both components independently sufficient"
        elif logit_restores: classification = "logit component sufficient"
        elif residual_restores: classification = "softmax-reconstruction residual sufficient"
        elif both_restores: classification = "requires both components"
        else: classification = "neither hybrid reproduces right probability outcome"
        only_left = sorted(captured_outcomes[left_name] - captured_outcomes[right_name])
        only_right = sorted(captured_outcomes[right_name] - captured_outcomes[left_name])
        left_max = float(logits[left_name][token].max()); right_max = float(logits[right_name][token].max())
        left_mean = float(np.mean(logits[left_name][token], dtype=np.float32))
        right_mean = float(np.mean(logits[right_name][token], dtype=np.float32))
        disputed = []
        for expert in sorted(set(only_left + only_right)):
            captured_delta = float(probabilities[left_name][token, expert] - probabilities[right_name][token, expert])
            softmax_delta = float(softmax[left_name][token, expert] - softmax[right_name][token, expert])
            residual_delta = float(residual[left_name][token, expert] - residual[right_name][token, expert])
            disputed.append({"expert_id": expert, "orientation": "left_minus_right",
                "captured_probability_delta": captured_delta,
                "diagnostic_softmax_probability_delta": softmax_delta,
                "softmax_reconstruction_residual_delta": residual_delta,
                "decomposition_residual": captured_delta - softmax_delta - residual_delta,
                "left_logit": float(logits[left_name][token, expert]),
                "right_logit": float(logits[right_name][token, expert]),
                "raw_logit_delta": float(logits[left_name][token, expert] - logits[right_name][token, expert]),
                "max_centered_logit_delta": float((logits[left_name][token, expert] - left_max) -
                                                   (logits[right_name][token, expert] - right_max)),
                "mean_centered_logit_delta": float((logits[left_name][token, expert] - left_mean) -
                                                    (logits[right_name][token, expert] - right_mean))})
        inversions = []
        for left_expert in only_left:
            for right_expert in only_right:
                gap = lambda values: float(values[left_expert] - values[right_expert])
                left_gap = gap(probabilities[left_name][token]); right_gap = gap(probabilities[right_name][token])
                softmax_change = gap(softmax[left_name][token]) - gap(softmax[right_name][token])
                residual_change = gap(residual[left_name][token]) - gap(residual[right_name][token])
                captured_change = left_gap - right_gap
                logit_gap = gap(logits[left_name][token]); right_logit_gap = gap(logits[right_name][token])
                sign_flip = lambda first, second: bool((first > 0 > second) or (first < 0 < second))
                logits_flip = sign_flip(left_gap, gap(coalition_probabilities["right_logits_only"]))
                residual_flip = sign_flip(left_gap, gap(coalition_probabilities["right_softmax_residual_only"]))
                inversions.append({"left_only_expert": left_expert, "right_only_expert": right_expert,
                    "left_captured_probability_gap": left_gap, "right_captured_probability_gap": right_gap,
                    "captured_probability_gap_change": captured_change,
                    "diagnostic_softmax_probability_gap_change": softmax_change,
                    "softmax_residual_gap_change": residual_change,
                    "decomposition_residual": captured_change - softmax_change - residual_change,
                    "left_raw_logit_gap": logit_gap, "right_raw_logit_gap": right_logit_gap,
                    "raw_logit_gap_change": logit_gap - right_logit_gap,
                    "replacing_logits_alone_reverses_probability_ordering": logits_flip,
                    "replacing_residual_alone_reverses_probability_ordering": residual_flip,
                    "only_combination_reproduces_right_probability_ordering": bool(
                        decisive and sign_flip(left_gap, right_gap) and not logits_flip and not residual_flip)})
        rows.append({"attended_token": int(attended_token), "coalitions": coalitions,
            "left_probability_outcome_matches_captured_native_left_membership": left_valid,
            "right_probability_outcome_matches_captured_native_right_membership": right_valid,
            "left_and_right_probability_outcomes_differ": not outcomes_equal,
            "right_logits_only_restores_right_probability_outcome": bool(decisive and not outcomes_equal and logit_restores),
            "right_softmax_residual_only_restores_right_probability_outcome": bool(
                decisive and not outcomes_equal and residual_restores),
            "both_required_for_right_probability_outcome": bool(
                decisive and not outcomes_equal and not logit_restores and not residual_restores and both_restores),
            "neither_hybrid_restores_right_probability_outcome": bool(
                not outcomes_equal and (not decisive or (not logit_restores and not residual_restores and not both_restores))),
            "membership_classification": classification,
            "membership_classification_decisive": decisive,
            "centered_logit_metrics": centered_logit_comparison(logits[left_name][token], logits[right_name][token]),
            "disputed_expert_probability_decomposition": disputed,
            "ordering_gap_decomposition": inversions})
    return {"reference_variant": reference_name, "left_implementation": left_name,
        "right_implementation": right_name, "softmax_reconstruction_metrics": reconstruction,
        "tokens": rows, "diagnostic_softmax_is_not_backend_kernel_identity": True}


def router_logit_softmax_decomposition(sides, prefix, attended_tokens):
    pairs = (("cpp_vs_python_default", "cpp", "default"),
             ("cpp_vs_python_math", "cpp", "math"),
             ("python_default_vs_python_math", "default", "math"))
    result = {}
    for pair_label, left, right in pairs:
        variants = {name: logit_softmax_pair_variant(
            sides, prefix, attended_tokens, left, right, name, reference)
            for name, reference in DIAGNOSTIC_SOFTMAX_REFERENCES.items()}
        robustness = []
        for token, attended_token in enumerate(attended_tokens):
            classifications = {name: report["tokens"][token]["membership_classification"]
                               for name, report in variants.items()}
            robustness.append({"attended_token": int(attended_token),
                "classifications": classifications,
                "classification_agreement": len(set(classifications.values())) == 1})
        affected = [row for row in robustness if variants["stable_float32"]["tokens"][
            attended_tokens.index(row["attended_token"])]["left_and_right_probability_outcomes_differ"]]
        result[pair_label] = {"variants": variants, "cross_variant_robustness": robustness,
            "all_affected_tokens_have_reference_variant_agreement":
                all(row["classification_agreement"] for row in affected)}
    return result


DIAGNOSTIC_LINEAR_VARIANTS = ("float64_matmul_then_float32", "float32_matmul",
                              "bf16_grid_inputs_and_weights_then_float32_matmul")


def diagnostic_router_linear(router_input, weight, variant):
    x = np.asarray(router_input, np.float32); w = np.asarray(weight, np.float32)
    if w.ndim != 2 or w.shape[0] != ROUTED_EXPERT_COUNT or x.shape[-1] != w.shape[1]:
        raise ValueError(f"canonical router linear orientation must be [384, hidden], got {w.shape}")
    if variant == "float64_matmul_then_float32":
        result = np.asarray(x.astype(np.float64) @ w.astype(np.float64).T, np.float32)
    elif variant == "float32_matmul":
        result = np.asarray(x @ w.T, np.float32)
    elif variant == "bf16_grid_inputs_and_weights_then_float32_matmul":
        result = np.asarray(bf16_round_to_float32(x) @ bf16_round_to_float32(w).T, np.float32)
    else:
        raise ValueError(f"unknown diagnostic linear variant: {variant}")
    if not np.isfinite(result).all(): raise ValueError("diagnostic router linear produced non-finite logits")
    return result


def router_input_lineage_audit(sides, prefix, attended_tokens):
    """Establish the norm-output router input; block_input is deliberately audited as a rejected alias."""
    reports = {}
    established = True; block_alias_exact = True
    for side in ("cpp", "default", "math"):
        identity = np.asarray(sides[side][prefix + "identity_residual"], np.float32).reshape(
            -1, sides[side][prefix + "identity_residual"].shape[-1])
        weights = np.asarray(sides[side][prefix + "identity_weight_sum"], np.float32).reshape(-1)
        ffn_norm = np.asarray(sides[side][prefix + "ffn_norm"], np.float32).reshape(identity.shape)
        block_input = np.asarray(sides[side][prefix + "block_input"], np.float32).reshape(identity.shape)
        valid = np.flatnonzero(weights != 0)
        if valid.size:
            reconstructed = identity[valid] / weights[valid, None]
            norm_metrics = _vector_metrics(reconstructed, ffn_norm[valid])
            block_metrics = _vector_metrics(reconstructed, block_input[valid])
            side_established = bool(norm_metrics["maximum_absolute_difference"] <= 2e-6)
            block_alias_exact &= bool(block_metrics["exact_equality"])
        else:
            norm_metrics = block_metrics = None; side_established = False; block_alias_exact = False
        established &= side_established
        reports[side] = {"tokens_with_nonzero_identity_weight_sum": [int(attended_tokens[i]) for i in valid],
            "identity_residual_divided_by_weight_vs_ffn_norm": norm_metrics,
            "identity_residual_divided_by_weight_vs_block_input": block_metrics,
            "numerical_lineage_consistent_with_ffn_norm": side_established}
    return {"status": "established" if established else "not established",
        "router_input_canonical_surface": "ffn_norm" if established else None,
        "proposed_block_input_alias_is_exact": block_alias_exact,
        "python_source_lineage": "post_attention_layernorm output -> router F.linear input",
        "cpp_source_lineage": "build_norm(ffn_inp, ffn_norm) output cur -> ggml_mul_mat(ffn_gate_inp, cur)",
        "numerical_consistency_audit": reports,
        "linear_decomposition_permitted": established}


def router_weight_equivalence_audit(python_weight, gguf_weight):
    py = np.asarray(python_weight, np.float32); gg = np.asarray(gguf_weight, np.float32)
    if py.ndim != 2 or gg.ndim != 2 or py.shape != gg.shape or py.shape[0] != ROUTED_EXPERT_COUNT:
        raise ValueError(f"router weights must share canonical [384, hidden] orientation: {py.shape}, {gg.shape}")
    def compare(a, b):
        metrics = _vector_metrics(a, b)
        return {"exact_element_count": int(a.size), "exact_byte_equality": bool(a.tobytes() == b.tobytes()),
            "exact_match_count": int(np.count_nonzero(a == b)), **metrics}
    rounded = bf16_round_to_float32(py)
    raw = compare(py, gg); bf16 = compare(rounded, gg)
    return {"orientation_validated": True, "canonical_shape": list(py.shape),
        "python_raw_tensor_vs_gguf_decoded_float32": raw,
        "python_bf16_rounded_to_float32_vs_gguf_decoded_float32": bf16,
        "gguf_equals_bf16_rounded_python_exactly": bool(np.array_equal(rounded, gg)),
        "weights_equivalent_for_shared_weight_analysis": bool(np.array_equal(rounded, gg))}


def _linear_projection_metrics(captured_delta, projected_delta):
    captured = np.asarray(captured_delta, np.float32); projected = np.asarray(projected_delta, np.float32)
    residual = np.asarray(captured - projected, np.float32)
    captured_rms = float(np.sqrt(np.mean(np.square(captured, dtype=np.float64))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    report = _vector_metrics(captured, projected)
    report.update({"captured_delta_rms": captured_rms, "projected_delta_rms":
        float(np.sqrt(np.mean(np.square(projected, dtype=np.float64)))), "residual_rms": residual_rms,
        "explained_rms_fraction": (1.0 - residual_rms / captured_rms) if captured_rms else None})
    return report


def classify_router_linear_membership(native_equal, decisive, input_restores, residual_restores,
                                      both_restores=True):
    if native_equal: return "native outcomes already equal"
    if not decisive: return "analysis not decisive"
    if input_restores and residual_restores: return "both components independently sufficient"
    if input_restores: return "router-input component sufficient"
    if residual_restores: return "diagnostic-linear residual sufficient"
    if both_restores: return "requires both components"
    return "neither hybrid reproduces Python outcome"


def router_linear_decomposition(sides, prefix, attended_tokens, python_weight, gguf_weight):
    lineage = router_input_lineage_audit(sides, prefix, attended_tokens)
    weight_audit = router_weight_equivalence_audit(python_weight, gguf_weight)
    result = {"router_input_lineage": lineage, "weight_equivalence": weight_audit,
        "diagnostic_matmul_is_not_cpu_cuda_blas_or_ggml_kernel_identity": True,
        "variants": {}, "weight_sensitive_fallback": None}
    if not lineage["linear_decomposition_permitted"]:
        result["status"] = "analysis not decisive"; return result
    inputs = {name: np.asarray(sides[name][prefix + "ffn_norm"], np.float32).reshape(
        -1, sides[name][prefix + "ffn_norm"].shape[-1]) for name in ("cpp", "default", "math")}
    logits = {name: np.asarray(sides[name][prefix + "router_logits"], np.float32).reshape(
        -1, ROUTED_EXPERT_COUNT) for name in ("cpp", "default", "math")}
    if not weight_audit["weights_equivalent_for_shared_weight_analysis"]:
        result["status"] = "weight representation differs"
        result["weight_sensitive_fallback"] = {variant: {
            label: {"finite": bool(np.isfinite(diagnostic_router_linear(inp, weight, variant)).all()),
                    "shape": list(diagnostic_router_linear(inp, weight, variant).shape)}
            for label, inp, weight in (("cpp_input_gguf_weight", inputs["cpp"], gguf_weight),
                ("python_input_gguf_weight", inputs["default"], gguf_weight),
                ("cpp_input_python_weight", inputs["cpp"], python_weight),
                ("python_input_python_weight", inputs["default"], python_weight))}
            for variant in DIAGNOSTIC_LINEAR_VARIANTS}
        return result
    result["status"] = "complete"
    shared = np.asarray(gguf_weight, np.float32)
    probabilities = {name: np.asarray(sides[name][prefix + "router_probabilities"], np.float32).reshape(
        -1, ROUTED_EXPERT_COUNT) for name in ("cpp", "default")}
    captured_indices = {name: np.asarray(sides[name][prefix + "router_topk_indices"]).reshape(-1, 12)
                        for name in ("cpp", "default")}
    cpp_bias, _, _ = constant_bias_reconstruction(sides["cpp"][prefix + "router_probabilities"],
        sides["cpp"][prefix + "router_selection_scores"], sides["cpp"][prefix + "router_topk_indices"], attended_tokens)
    for variant in DIAGNOSTIC_LINEAR_VARIANTS:
        diagnostic = {name: diagnostic_router_linear(inputs[name], shared, variant)
                      for name in ("cpp", "default", "math")}
        residual = {name: np.asarray(logits[name] - diagnostic[name], np.float32)
                    for name in ("cpp", "default", "math")}
        implementation_metrics = {name: {**_vector_metrics(logits[name], diagnostic[name]),
            "finite": bool(np.isfinite(diagnostic[name]).all()),
            "residual_maximum_absolute": float(np.abs(residual[name]).max(initial=0)),
            "residual_rms": float(np.sqrt(np.mean(np.square(residual[name], dtype=np.float64))))}
            for name in ("cpp", "default", "math")}
        token_rows = []
        for token, attended in enumerate(attended_tokens):
            intended = {name: set(captured_indices[name][token]) for name in ("cpp", "default")}
            native_outcomes = {name: set(_score_topk_indices(np.asarray(probabilities[name][token] + cpp_bias,
                np.float32))) for name in ("cpp", "default")}
            by_softmax = {}
            for softmax_name, softmax in DIAGNOSTIC_SOFTMAX_REFERENCES.items():
                specs = {"native_cpp": ("cpp", "cpp"), "python_input_only": ("default", "cpp"),
                    "python_linear_residual_only": ("cpp", "default"), "native_python": ("default", "default")}
                coalitions = {}
                for label, (input_side, residual_side) in specs.items():
                    candidate_logits = np.asarray(diagnostic[input_side][token] + residual[residual_side][token], np.float32)
                    candidate_probability = softmax(candidate_logits)
                    cutoff = _public_cutoff(np.asarray(candidate_probability + cpp_bias, np.float32), int(attended))
                    selected = set(cutoff["selected_expert_set"])
                    coalitions[label] = {**cutoff, "finite": bool(np.isfinite(candidate_probability).all()),
                        "equals_native_cpp_probability_outcome": selected == native_outcomes["cpp"],
                        "equals_native_python_probability_outcome": selected == native_outcomes["default"]}
                input_restores = coalitions["python_input_only"]["equals_native_python_probability_outcome"]
                residual_restores = coalitions["python_linear_residual_only"]["equals_native_python_probability_outcome"]
                native_valid = (native_outcomes["cpp"] == intended["cpp"] and
                                native_outcomes["default"] == intended["default"] and
                                coalitions["native_cpp"]["equals_native_cpp_probability_outcome"] and
                                coalitions["native_python"]["equals_native_python_probability_outcome"] and
                                all(row["finite"] for row in coalitions.values()))
                classification = classify_router_linear_membership(
                    native_outcomes["cpp"] == native_outcomes["default"], native_valid,
                    input_restores, residual_restores,
                    coalitions["native_python"]["equals_native_python_probability_outcome"])
                by_softmax[softmax_name] = {"coalitions": coalitions, "classification": classification,
                    "classification_decisive": native_valid}
            classifications = [row["classification"] for row in by_softmax.values()]
            disputed = sorted(native_outcomes["cpp"] ^ native_outcomes["default"])
            captured_delta = logits["cpp"][token] - logits["default"][token]
            projected = diagnostic_router_linear(
                inputs["cpp"][token:token+1] - inputs["default"][token:token+1], shared, variant)[0]
            direct = {"raw": _linear_projection_metrics(captured_delta, projected),
                "maximum_centered": _linear_projection_metrics(
                    captured_delta - captured_delta.max(), projected - projected.max()),
                "mean_centered": _linear_projection_metrics(
                    captured_delta - captured_delta.mean(), projected - projected.mean()),
                "disputed_experts": [{"expert_id": int(expert),
                    "captured_logit_delta": float(captured_delta[expert]),
                    "projected_input_delta_contribution": float(projected[expert]),
                    "remaining_linear_residual": float(captured_delta[expert] - projected[expert])}
                    for expert in disputed]}
            token_rows.append({"attended_token": int(attended), "softmax_references": by_softmax,
                "cross_softmax_classification_agreement": len(set(classifications)) == 1,
                "direct_delta_projection": direct})
        result["variants"][variant] = {"implementation_metrics": implementation_metrics, "tokens": token_rows}
    return result


def primary_router_linear_summary(linear_reports, first_affected_block):
    selected = next((row for row in linear_reports if row["physical_block"] == first_affected_block), None)
    if selected is None: return None
    analysis = selected["analysis"]
    summary = {"physical_block": selected["physical_block"],
        "router_input_lineage_status": analysis["router_input_lineage"]["status"],
        "router_input_canonical_surface": analysis["router_input_lineage"]["router_input_canonical_surface"],
        "weight_equivalence_status": analysis["weight_equivalence"][
            "weights_equivalent_for_shared_weight_analysis"],
        "diagnostic_matmul_is_not_cpu_cuda_blas_or_ggml_kernel_identity": True,
        "status": analysis["status"]}
    if analysis["status"] != "complete": return summary
    per_variant = {}; affected = set(); input_tokens = set(); residual_tokens = set(); both_tokens = set()
    token_classifications = {}
    for variant, report in analysis["variants"].items():
        per_variant[variant] = []
        for row in report["tokens"]:
            classifications = {name: value["classification"] for name, value in row["softmax_references"].items()}
            per_variant[variant].append({"attended_token": row["attended_token"],
                "classifications": classifications,
                "cross_softmax_classification_agreement": row["cross_softmax_classification_agreement"]})
            token_classifications.setdefault(row["attended_token"], set()).update(classifications.values())
            for classification in classifications.values():
                if classification != "native outcomes already equal": affected.add(row["attended_token"])
                if classification == "router-input component sufficient": input_tokens.add(row["attended_token"])
                if classification == "diagnostic-linear residual sufficient": residual_tokens.add(row["attended_token"])
                if classification == "requires both components": both_tokens.add(row["attended_token"])
    summary.update({"affected_attended_tokens": sorted(affected), "per_variant_classifications": per_variant,
        "cross_matmul_and_cross_softmax_agreement": all(len(values) == 1 for values in token_classifications.values()),
        "tokens_explained_by_router_input": sorted(input_tokens),
        "tokens_explained_by_linear_residual": sorted(residual_tokens),
        "tokens_requiring_both": sorted(both_tokens),
        "direct_delta_projection_metrics": {variant: [row["direct_delta_projection"]
            for row in report["tokens"] if row["attended_token"] in affected]
            for variant, report in analysis["variants"].items()},
        "block_12_remains_available_as_aligned_comparison": any(
            row["physical_block"] == 12 for row in linear_reports)})
    return summary


def primary_router_cutoff_summary(router_cutoffs, component_rows, even_analysis):
    primary = next((row for row in router_cutoffs
        if row["pairwise_disputed_experts"]["cpp_vs_python_default"]), None)
    if primary is None:
        return None
    block = primary["physical_block"]
    component_by_suffix = {row["suffix"]: row for row in component_rows
                           if row["physical_block"] == block}
    shortcut_row = next(row for row in even_analysis if row["physical_block"] == block)
    disputes = primary["pairwise_disputed_experts"]["cpp_vs_python_default"]
    def primary_pass(suffix):
        return component_by_suffix[suffix]["cpp_vs_python_default"]["within_diagnostic_criterion"]
    aligned_pass = shortcut_row["expert_id_aligned_topk_weights"][
        "cpp_vs_python_default"]["within_diagnostic_criterion"]
    continuous_passes = all(primary_pass(suffix) for suffix in (
        "router_logits", "router_probabilities", "router_selection_scores"))
    return {"physical_block": block, "affected_attended_token_count": len(disputes),
        "affected_attended_tokens": [row["attended_token"] for row in disputes],
        "first_affected_attended_token": disputes[0]["attended_token"],
        "continuous_router_logits_within_criterion": primary_pass("router_logits"),
        "continuous_router_probabilities_within_criterion": primary_pass("router_probabilities"),
        "continuous_router_selection_scores_within_criterion": primary_pass("router_selection_scores"),
        "returned_order_topk_weights_within_criterion": primary_pass("router_topk_weights"),
        "expert_id_aligned_topk_weights_within_criterion": aligned_pass,
        "identity_weight_sum_within_criterion": primary_pass("identity_weight_sum"),
        "identity_residual_within_criterion": primary_pass("identity_residual"),
        "complete_moe_shortcut_within_criterion": primary_pass("moe_shortcut"),
        "reconstructed_correction_bias_pairwise_metrics":
            primary["reconstructed_correction_bias"]["pairwise_metrics"],
        "all_selected_set_differences_have_ordering_inversions":
            all(row["has_ordering_inversion"] for row in disputes),
        "all_disputed_experts_are_reported_with_cutoff_ranks": all(
            len(row["disputed_experts"]) ==
                len(row["experts_selected_only_by_left"]) + len(row["experts_selected_only_by_right"]) and
            all("left_rank" in expert and "right_rank" in expert for expert in row["disputed_experts"])
            for row in disputes),
        "descriptive_evidence": (
            "continuous scores remain within criterion while discrete membership differs"
            if continuous_passes else
            "discrete membership differs and at least one continuous score surface is outside criterion"),
        "returned_order_topk_weight_failure_not_treated_as_weight_math_failure":
            bool(not primary_pass("router_topk_weights") and aligned_pass),
        "returned_order_note":
            "returned-order top-k weight failure is not treated as a weight-math failure when expert-ID-aligned weights pass"}


def primary_probability_bias_summary(block_reports):
    primary = None
    affected = []
    for block_report in block_reports:
        pair = block_report["pairwise"]["cpp_vs_python_default"]
        affected = [row for row in pair["tokens"]
                    if row["membership_classification"] != "native memberships already equal"]
        if affected:
            primary = (block_report, pair)
            break
    if primary is None:
        return None
    block_report, pair = primary
    probability_tokens = [row["attended_token"] for row in affected
                          if row["right_probabilities_only_restores_right_membership"]]
    bias_tokens = [row["attended_token"] for row in affected
                   if row["right_bias_only_restores_right_membership"]]
    both_tokens = [row["attended_token"] for row in affected
                   if row["both_required_for_right_membership"]]
    neither_tokens = [row["attended_token"] for row in affected
                      if row["neither_hybrid_restores_right_membership"]]
    descriptions = []
    if len(probability_tokens) == len(affected):
        descriptions.append("probability component is sufficient for all affected tokens")
    if bias_tokens:
        descriptions.append("bias component is sufficient for at least one affected token")
    if both_tokens:
        descriptions.append("probability and bias changes interact at the cutoff")
    return {"physical_block": block_report["physical_block"],
        "affected_attended_tokens": [row["attended_token"] for row in affected],
        "native_left_membership_reconstruction_valid": all(
            row["native_left_membership_reconstruction_valid"] for row in affected),
        "native_right_membership_reconstruction_valid": all(
            row["native_right_membership_reconstruction_valid"] for row in affected),
        "tokens_where_probability_swap_alone_restores_default": probability_tokens,
        "tokens_where_bias_swap_alone_restores_default": bias_tokens,
        "tokens_where_both_are_required": both_tokens,
        "tokens_where_neither_hybrid_restores_default": neither_tokens,
        "per_token_membership_classification": [
            {"attended_token": row["attended_token"],
             "classification": row["membership_classification"],
             "decisive": row["membership_classification_decisive"]} for row in affected],
        "disputed_expert_delta_decomposition": [
            {"attended_token": row["attended_token"], "experts": row["disputed_expert_delta_decomposition"]}
            for row in affected],
        "ordering_inversion_decomposition": [
            {"attended_token": row["attended_token"], "pairs": row["ordering_inversion_decomposition"]}
            for row in affected],
        "correction_bias_dtype_grid_audit": pair["correction_bias_dtype_grid_audit"],
        "descriptive_evidence": descriptions,
        "categorical_membership_results_are_not_continuous_causality_verdicts": True}


def primary_logit_softmax_summary(block_reports):
    primary = None
    affected_positions = []
    for block_report in block_reports:
        pair = block_report["pairwise"]["cpp_vs_python_default"]
        f32_tokens = pair["variants"]["stable_float32"]["tokens"]
        affected_positions = [index for index, row in enumerate(f32_tokens)
                              if row["left_and_right_probability_outcomes_differ"]]
        if affected_positions:
            primary = (block_report, pair)
            break
    if primary is None:
        return None
    block_report, pair = primary
    variants = pair["variants"]
    token_rows = []
    logits_tokens = set(); residual_tokens = set(); both_tokens = set(); indecisive_tokens = set()
    for index in affected_positions:
        attended_token = variants["stable_float32"]["tokens"][index]["attended_token"]
        classifications = {name: report["tokens"][index]["membership_classification"]
                           for name, report in variants.items()}
        agreement = len(set(classifications.values())) == 1
        token_rows.append({"attended_token": attended_token, "classifications": classifications,
                           "classification_agreement": agreement})
        for report in variants.values():
            row = report["tokens"][index]
            if row["right_logits_only_restores_right_probability_outcome"]: logits_tokens.add(attended_token)
            if row["right_softmax_residual_only_restores_right_probability_outcome"]: residual_tokens.add(attended_token)
            if row["both_required_for_right_probability_outcome"]: both_tokens.add(attended_token)
            if not row["membership_classification_decisive"]: indecisive_tokens.add(attended_token)
    descriptions = []
    affected_tokens = [row["attended_token"] for row in token_rows]
    if all(all(variants[name]["tokens"][index]["right_logits_only_restores_right_probability_outcome"]
               for name in variants) for index in affected_positions):
        descriptions.append("captured-logit differences are sufficient under both diagnostic references")
    if residual_tokens:
        descriptions.append("softmax-reconstruction residual is sufficient under at least one reference")
    if both_tokens:
        descriptions.append("logit and residual components interact")
    if not all(row["classification_agreement"] for row in token_rows):
        descriptions.append("diagnostic reference choice changes the categorical result")
    return {"physical_block": block_report["physical_block"],
        "affected_attended_tokens": affected_tokens,
        "left_probability_outcome_valid": all(variants[name]["tokens"][index][
            "left_probability_outcome_matches_captured_native_left_membership"]
            for name in variants for index in affected_positions),
        "right_probability_outcome_valid": all(variants[name]["tokens"][index][
            "right_probability_outcome_matches_captured_native_right_membership"]
            for name in variants for index in affected_positions),
        "per_token_classifications": token_rows,
        "tokens_where_logits_alone_are_sufficient": sorted(logits_tokens),
        "tokens_where_softmax_residual_alone_is_sufficient": sorted(residual_tokens),
        "tokens_requiring_both": sorted(both_tokens),
        "tokens_where_analysis_is_not_decisive": sorted(indecisive_tokens),
        "centered_logit_metrics": {name: [report["tokens"][index]["centered_logit_metrics"]
            for index in affected_positions] for name, report in variants.items()},
        "disputed_expert_decompositions": {name: [report["tokens"][index][
            "disputed_expert_probability_decomposition"] for index in affected_positions]
            for name, report in variants.items()},
        "ordering_gap_decompositions": {name: [report["tokens"][index]["ordering_gap_decomposition"]
            for index in affected_positions] for name, report in variants.items()},
        "softmax_reconstruction_metrics": {name: report["softmax_reconstruction_metrics"]
                                            for name, report in variants.items()},
        "all_affected_tokens_have_reference_variant_agreement":
            all(row["classification_agreement"] for row in token_rows),
        "descriptive_evidence": descriptions,
        "diagnostic_softmax_decomposition_is_not_a_backend_kernel_identity_verdict": True}


def reconstruct_odd_coalition(post_attention_residual, dense_output, previous_even_moe_shortcut):
    """Apply the official odd-block association with both observed BF16 boundaries."""
    trunk = bf16_round_to_float32(post_attention_residual + dense_output)
    return bf16_round_to_float32(trunk + previous_even_moe_shortcut)


def odd_cross_substitution(cpp, python, odd_block, criterion):
    p = lambda block, suffix: f"physical_block_{block:02d}__{suffix}"
    component_keys = {
        "post_attention_residual": p(odd_block, "post_attention_residual"),
        "dense_output": p(odd_block, "dense_output"),
        "previous_even_moe_shortcut": p(odd_block - 1, "moe_shortcut"),
    }
    actual = python[p(odd_block, "block_output")]
    players = tuple(component_keys)
    coalition_labels = {
        frozenset(): "cpp_P + cpp_D + cpp_S",
        frozenset({players[0]}): "py_P + cpp_D + cpp_S",
        frozenset({players[1]}): "cpp_P + py_D + cpp_S",
        frozenset({players[2]}): "cpp_P + cpp_D + py_S",
        frozenset({players[0], players[1]}): "py_P + py_D + cpp_S",
        frozenset({players[0], players[2]}): "py_P + cpp_D + py_S",
        frozenset({players[1], players[2]}): "cpp_P + py_D + py_S",
        frozenset(players): "py_P + py_D + py_S",
    }
    coalitions = {}
    rms_by_subset = {}
    for subset, label in coalition_labels.items():
        selected = {player: (python if player in subset else cpp)[key]
                    for player, key in component_keys.items()}
        candidate = reconstruct_odd_coalition(
            selected["post_attention_residual"], selected["dense_output"],
            selected["previous_even_moe_shortcut"])
        metrics = all_block_result(actual, candidate, criterion)
        coalitions[label] = {**metrics,
            "passed": metrics["within_diagnostic_criterion"],
            "maximum_normalized_criterion_violation":
                metrics["maximum_normalized_violation_under_diagnostic_criterion"],
            "exact_reconstruction": bool(np.array_equal(actual, candidate)),
            "python_components": sorted(subset)}
        rms_by_subset[subset] = metrics["rms_error"]

    empty = frozenset(); full = frozenset(players)
    if not coalitions[coalition_labels[full]]["exact_reconstruction"]:
        raise ValueError(
            f"physical block {odd_block}: all-Python coalition does not exactly reconstruct output")
    baseline = rms_by_subset[empty]
    shapley = {}
    for player in players:
        others = [candidate for candidate in players if candidate != player]
        contribution = 0.0
        for size in range(3):
            for members in itertools.combinations(others, size):
                subset = frozenset(members)
                weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
                contribution += weight * (rms_by_subset[subset] - rms_by_subset[subset | {player}])
        shapley[player] = contribution
    total_reduction = baseline - rms_by_subset[full]
    shapley_sum = sum(shapley.values())

    single_labels = {player: coalition_labels[frozenset({player})] for player in players}
    pair_labels = {"post_attention_residual_and_dense_output": coalition_labels[frozenset(players[:2])],
                   "post_attention_residual_and_previous_even_moe_shortcut":
                       coalition_labels[frozenset((players[0], players[2]))],
                   "dense_output_and_previous_even_moe_shortcut":
                       coalition_labels[frozenset(players[1:])]}
    baseline_passes = coalitions[coalition_labels[empty]]["passed"]
    single_restores = {player: (not baseline_passes and coalitions[label]["passed"])
                       for player, label in single_labels.items()}
    pair_restores = {name: (not baseline_passes and coalitions[label]["passed"])
                     for name, label in pair_labels.items()}
    sufficient = [player for player, restores in single_restores.items() if restores]
    display = {players[0]: "post-attention residual", players[1]: "dense output",
               players[2]: "previous-even MoE shortcut"}
    if baseline_passes:
        threshold = "already passing"
    elif len(sufficient) > 1:
        threshold = "multiple single components independently sufficient"
    elif len(sufficient) == 1:
        threshold = f"single component sufficient: {display[sufficient[0]]}"
    elif any(pair_restores.values()) or coalitions[coalition_labels[full]]["passed"]:
        threshold = "requires multiple components"
    else:
        threshold = "no tested substitution restores pass"

    # Backward-compatible branch-level views are aliases of the corresponding coalitions.
    reports = {
        "all_cpp": coalitions[coalition_labels[empty]],
        "python_shortcut_only": coalitions[single_labels[players[2]]],
        "python_trunk_only": coalitions[pair_labels["post_attention_residual_and_dense_output"]],
        "all_python_reconstruction": coalitions[coalition_labels[full]],
    }
    shortcut_reduction = baseline - reports["python_shortcut_only"]["rms_error"]
    trunk_reduction = baseline - reports["python_trunk_only"]["rms_error"]
    if shortcut_reduction <= 0 and trunk_reduction <= 0: dominant = "indeterminate"
    elif shortcut_reduction > 1.25 * max(trunk_reduction, 0): dominant = "previous-even MoE shortcut"
    elif trunk_reduction > 1.25 * max(shortcut_reduction, 0): dominant = "odd attention/dense trunk"
    else: dominant = "mixed"
    return {"alternatives": reports, "coalitions": coalitions,
            "coalition_count": len(coalitions),
            "shapley_rms_reduction_post_attention_residual": shapley[players[0]],
            "shapley_rms_reduction_dense_output": shapley[players[1]],
            "shapley_rms_reduction_previous_even_moe_shortcut": shapley[players[2]],
            "shapley_sum": shapley_sum,
            "total_all_cpp_to_all_python_rms_reduction": total_reduction,
            "shapley_additivity_error": shapley_sum - total_reduction,
            "post_attention_residual_only_restores_pass": single_restores[players[0]],
            "dense_output_only_restores_pass": single_restores[players[1]],
            "previous_even_moe_shortcut_only_restores_pass": single_restores[players[2]],
            **{name + "_restores_pass": restores for name, restores in pair_restores.items()},
            "threshold_crossing_attribution": threshold,
            "rms_reduction_shortcut_only": shortcut_reduction,
            "rms_reduction_trunk_only": trunk_reduction, "dominant_branch": dominant,
            "dominant_branch_is_exclusive_causality": False}


def compare_component_window(default_npz, math_npz, capture_dir, attention_mask,
                             criterion, start=10, count=4, router_weights=None):
    expected = set(component_window_names(start, count))
    if set(default_npz.files) != expected or set(math_npz.files) != expected:
        raise ValueError("default/math component-window inventory mismatch")
    manifest = read_component_window_manifest(
        Path(capture_dir) / "block-components-window-diagnostics.tsv", start, count)
    mask = np.asarray(attention_mask, dtype=bool)
    attended_tokens = np.flatnonzero(mask).tolist()
    sides = {"cpp": {}, "default": {}, "math": {}}
    results = []; first_default = first_math = first_both = None
    for name in component_window_names(start, count):
        cpp = decode_component_raw(name, *manifest[name])[:, mask, :]
        default = np.asarray(default_npz[name])[:, mask, :]; math = np.asarray(math_npz[name])[:, mask, :]
        if not all(np.isfinite(x).all() for x in (cpp, default, math) if np.issubdtype(x.dtype, np.floating)):
            raise ValueError(f"{name}: non-finite component-window value")
        sides["cpp"][name] = cpp; sides["default"][name] = default; sides["math"][name] = math
        if name.endswith("__router_topk_indices"): continue
        d = all_block_result(default, cpp, criterion); m = all_block_result(math, cpp, criterion)
        hidden = cpp.shape[-1] == 3072
        row = {"name": name, "physical_block": int(name[15:17]),
               "suffix": name.split("__", 1)[1], "cpp_vs_python_default": d,
               "cpp_vs_python_math": m, "primary_oracle": "python_default",
               "sensitivity_control": "python_math"}
        results.append(row)
        if hidden and not d["within_diagnostic_criterion"] and first_default is None: first_default = name
        if hidden and not m["within_diagnostic_criterion"] and first_math is None: first_math = name
        if hidden and not d["within_diagnostic_criterion"] and not m["within_diagnostic_criterion"] and first_both is None:
            first_both = name
    routers = []; first_real_default = first_real_math = None; even_analysis = []
    router_cutoffs = []; probability_bias_reports = []; logit_softmax_reports = []; linear_reports = []
    for block in range(start, start + count, 2):
        p = f"physical_block_{block:02d}__"
        semantic = semantic_router_report(sides["cpp"][p + "router_topk_indices"],
            sides["default"][p + "router_topk_indices"], sides["math"][p + "router_topk_indices"],
            sides["cpp"][p + "identity_weight_sum"], sides["default"][p + "identity_weight_sum"],
            sides["math"][p + "identity_weight_sum"])
        semantic["physical_block"] = block; routers.append(semantic)
        cutoff = router_cutoff_analysis(sides, p, attended_tokens)
        cutoff["physical_block"] = block
        router_cutoffs.append(cutoff)
        probability_bias_reports.append({"physical_block": block,
            "pairwise": router_probability_bias_decomposition(sides, p, attended_tokens)})
        logit_softmax_reports.append({"physical_block": block,
            "pairwise": router_logit_softmax_decomposition(sides, p, attended_tokens)})
        if router_weights is not None:
            linear_reports.append({"physical_block": block, "analysis": router_linear_decomposition(
                sides, p, attended_tokens, router_weights[f"physical_block_{block:02d}__python_weight"],
                router_weights[f"physical_block_{block:02d}__gguf_weight"])})
        if semantic["has_cpp_vs_python_default_real_expert_set_difference"] and first_real_default is None:
            first_real_default = block
        if semantic["has_cpp_vs_python_math_real_expert_set_difference"] and first_real_math is None:
            first_real_math = block
        shortcut = next(row for row in results if row["name"] == p + "moe_shortcut")
        block_output = next(row for row in results if row["name"] == p + "block_output")
        ci, di, mi = (sides[x][p + "router_topk_indices"] for x in ("cpp", "default", "math"))
        cw, dw, mw = (sides[x][p + "router_topk_weights"] for x in ("cpp", "default", "math"))
        even_analysis.append({"physical_block": block,
            "moe_shortcut": {"cpp_vs_python_default": shortcut["cpp_vs_python_default"],
                             "cpp_vs_python_math": shortcut["cpp_vs_python_math"]},
            "router_semantics": semantic,
            "identity_weight_sum": {
                "cpp_vs_python_default": all_block_result(sides["default"][p + "identity_weight_sum"], sides["cpp"][p + "identity_weight_sum"], criterion),
                "cpp_vs_python_math": all_block_result(sides["math"][p + "identity_weight_sum"], sides["cpp"][p + "identity_weight_sum"], criterion)},
            "expert_id_aligned_topk_weights": {
                "cpp_vs_python_default": all_block_result(align_router_weights(di, dw), align_router_weights(ci, cw), criterion),
                "cpp_vs_python_math": all_block_result(align_router_weights(mi, mw), align_router_weights(ci, cw), criterion)},
            "shortcut_discrepancy_present_while_even_output_passes_default":
                (not shortcut["cpp_vs_python_default"]["within_diagnostic_criterion"] and
                 block_output["cpp_vs_python_default"]["within_diagnostic_criterion"])})
    attribution = {}
    for block in range(start + 1, start + count, 2):
        for label in ("default", "math"):
            attribution[f"block_{block}_vs_{label}"] = odd_cross_substitution(
                sides["cpp"], sides[label], block, criterion)
    first_identity_default = next((row["physical_block"] for row in routers
        if row["has_cpp_vs_python_default_identity_presence_difference"]), None)
    first_shortcut_while_even_passes = next((
        f"physical_block_{row['physical_block']:02d}__moe_shortcut"
        for row in even_analysis
        if row["shortcut_discrepancy_present_while_even_output_passes_default"]), None)
    primary_summary = primary_router_cutoff_summary(router_cutoffs, results, even_analysis)
    primary_probability_bias = primary_probability_bias_summary(probability_bias_reports)
    primary_logit_softmax = primary_logit_softmax_summary(logit_softmax_reports)
    primary_linear = primary_router_linear_summary(linear_reports, first_real_default)
    return {"accepted": False, "array_count": len(expected), "physical_block_start": start,
        "physical_block_count": count, "primary_oracle": "python_default",
        "sensitivity_control": "python_math",
        "first_component_outside_criterion_vs_python_default": first_default,
        "first_component_outside_criterion_vs_python_math": first_math,
        "first_component_outside_criterion_vs_both": first_both,
        "first_physical_block_with_cpp_vs_python_default_real_expert_difference": first_real_default,
        "first_physical_block_with_cpp_vs_python_math_real_expert_difference": first_real_math,
        "first_primary_oracle_component_failure": first_default,
        "first_primary_oracle_real_expert_difference": first_real_default,
        "first_primary_oracle_identity_presence_difference": first_identity_default,
        "first_primary_oracle_shortcut_failure_while_even_output_passes":
            first_shortcut_while_even_passes,
        "first_both_backend_component_failure": first_both,
        "primary_oracle_router_cutoff_summary": primary_summary,
        "primary_oracle_router_probability_bias_decomposition": primary_probability_bias,
        "primary_oracle_router_logit_softmax_decomposition": primary_logit_softmax,
        "primary_oracle_router_linear_decomposition": primary_linear,
        **{f"block_{block}_dominant_branch_vs_{label}": attribution[f"block_{block}_vs_{label}"]["dominant_branch"]
           for block in range(start + 1, start + count, 2) for label in ("default", "math")},
        "components": results, "router_semantics": routers,
        "router_cutoff_analysis": router_cutoffs,
        "router_probability_bias_decomposition": probability_bias_reports,
        "router_logit_softmax_decomposition": logit_softmax_reports,
        "router_linear_decomposition": linear_reports,
        "even_block_shortcut_analysis": even_analysis, "odd_block_cross_substitution": attribution}
def reconstruction_metrics(target, alternatives):
    reports = {}
    for name, candidate in alternatives.items():
        difference = np.abs(np.asarray(target, np.float32) - np.asarray(candidate, np.float32))
        reports[name] = {
            "byte_exact": (np.asarray(target).dtype == np.asarray(candidate).dtype and
                           np.asarray(target).shape == np.asarray(candidate).shape and
                           np.ascontiguousarray(target).tobytes() == np.ascontiguousarray(candidate).tobytes()),
            "maximum_absolute_error": float(difference.max(initial=0)),
            "mean_absolute_error": float(difference.mean()),
            "rms_error": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))),
            "cosine_similarity": all_block_result(
                np.asarray(target), np.asarray(candidate), {"atol": 0.125, "rtol": 0.03125})[
                    "cosine_similarity"],
        }
    closest = min(reports, key=lambda name: reports[name]["rms_error"])
    exact = [name for name, report in reports.items() if report["byte_exact"]]
    return {"alternatives": reports, "closest_by_rms": closest, "byte_exact_alternatives": exact}


def reconstruct_component_boundaries(arrays):
    output = []
    for block in range(10):
        prefix = f"physical_block_{block:02d}__"
        block_input = arrays[prefix + "block_input"]
        attention_output = arrays[prefix + "attention_output"]
        post = arrays[prefix + "post_attention_residual"]
        attention_f32 = block_input + attention_output
        row = {"physical_block": block, "attention_residual": reconstruction_metrics(post, {
            "pure_f32_addition": attention_f32,
            "bf16_round_after_addition": bf16_round_to_float32(attention_f32),
        })}
        dense = arrays[prefix + "dense_output"]
        target = arrays[prefix + "block_output"]
        first = post + dense
        if block % 2 == 0:
            alternatives = {"pure_f32_addition": first,
                            "bf16_round_after_addition": bf16_round_to_float32(first)}
        else:
            shortcut = arrays[f"physical_block_{block - 1:02d}__moe_shortcut"]
            alternatives = {
                "all_f32_official_association": first + shortcut,
                "bf16_after_each_official_addition": bf16_round_to_float32(
                    bf16_round_to_float32(first) + shortcut),
                "bf16_only_at_final_output": bf16_round_to_float32(first + shortcut),
                "previous_grouping_all_f32": (dense + shortcut) + post,
                "previous_grouping_bf16_after_each_addition": bf16_round_to_float32(
                    bf16_round_to_float32(dense + shortcut) + post),
            }
        row["block_output"] = reconstruction_metrics(target, alternatives)
        output.append(row)
    return output


def reconstruction_proves_bf16_boundary_contract(reconstruction):
    expected = {
        "cpp": ("pure_f32_addition", "pure_f32_addition", "all_f32_official_association"),
        "python_default": ("bf16_round_after_addition", "bf16_round_after_addition",
                           "bf16_after_each_official_addition"),
        "python_math": ("bf16_round_after_addition", "bf16_round_after_addition",
                        "bf16_after_each_official_addition"),
    }
    for side, (attention_name, even_name, odd_name) in expected.items():
        rows = reconstruction.get(side, [])
        if len(rows) != 10:
            return False
        for row in rows:
            if row["attention_residual"]["byte_exact_alternatives"] != [attention_name]:
                return False
            output_name = even_name if row["physical_block"] % 2 == 0 else odd_name
            if row["block_output"]["byte_exact_alternatives"] != [output_name]:
                return False
    return True


def classify_numerical_attribution(router_reports, floating_reports, first_persistent_block,
                                   reconstruction=None):
    for router in router_reports:
        semantic_difference = (
            (router.get("has_cpp_vs_python_default_real_expert_set_difference", False) or
             router.get("has_cpp_vs_python_default_identity_presence_difference", False)) and
            (router.get("has_cpp_vs_python_math_real_expert_set_difference", False) or
             router.get("has_cpp_vs_python_math_identity_presence_difference", False)))
        if (semantic_difference and not router["shortcut_within_criterion_vs_both"] and
                (first_persistent_block is None or router["physical_block"] < first_persistent_block)):
            return "real router-selection divergence"
    local_suffixes = (("attention_output", "attention output divergence"),
                      ("dense_output", "dense output divergence"),
                      ("moe_shortcut", "complete shortcut divergence"))
    for suffix, label in local_suffixes:
        if any(row["suffix"] == suffix and row["outside_vs_both"] for row in floating_reports):
            return label
    if any(row["suffix"] == "post_attention_residual" and row["outside_vs_both"] and
           not row["python_backends_outside_criterion"] for row in floating_reports):
        return "residual/addition precision divergence"
    if any(row["suffix"] in {"attention_norm", "ffn_norm"} and row["outside_vs_both"] and
           not row["python_backends_outside_criterion"] for row in floating_reports):
        return "norm sensitivity"
    if reconstruction is not None and reconstruction_proves_bf16_boundary_contract(reconstruction):
        return "BF16 residual-boundary precision contract mismatch"
    return "cumulative numerical drift with no discrete local operator failure"


def numerical_attribution_report(default_npz, math_npz, capture_dir, attention_mask, criterion):
    manifest = read_component_manifest(capture_dir / "block-components-diagnostics.tsv")
    mask = np.asarray(attention_mask, dtype=bool)
    sides = {"cpp": {}, "python_default": {}, "python_math": {}}
    floating = []
    first_default_change = None
    first_math_change = None
    largest_improvement = None
    largest_improvement_value = float("-inf")
    for name in component_names():
        dtype, dims, raw = manifest[name]
        cpp = decode_component_raw(name, dtype, dims, raw)
        default = np.asarray(default_npz[name])
        math = np.asarray(math_npz[name])
        cpp, default, math = cpp[:, mask, :], default[:, mask, :], math[:, mask, :]
        sides["cpp"][name] = cpp
        sides["python_default"][name] = default
        sides["python_math"][name] = math
        if name.endswith("__router_topk_indices"):
            continue
        suffix = name.split("__", 1)[1]
        hidden = cpp.shape[-1] == 3072
        default_report = rounding_comparison(default, cpp, criterion, hidden)
        math_report = rounding_comparison(math, cpp, criterion, hidden)
        if default_report["pass_state_changed"] and first_default_change is None:
            first_default_change = name
        if math_report["pass_state_changed"] and first_math_change is None:
            first_math_change = name
        for backend, report in (("default", default_report), ("math", math_report)):
            improvement = (report["raw_cpp_f32"]["rms_error"] -
                           report["bf16_rounded_cpp"]["rms_error"])
            if improvement > largest_improvement_value:
                largest_improvement_value = improvement
                largest_improvement = {"name": name, "backend": backend,
                                       "rms_improvement": improvement}
        raw_default = all_block_result(default, cpp, criterion)
        raw_math = all_block_result(math, cpp, criterion)
        py_pair = all_block_result(default, math, criterion)
        floating.append({"name": name, "physical_block": int(name[15:17]), "suffix": suffix,
                         "hidden_size_surface": hidden,
                         "outside_vs_both": bool(hidden and
                             not raw_default["within_diagnostic_criterion"] and
                             not raw_math["within_diagnostic_criterion"]),
                         "python_backends_outside_criterion": bool(hidden and
                             not py_pair["within_diagnostic_criterion"]),
                         "cpp_vs_python_default": default_report,
                         "cpp_vs_python_math": math_report})

    routers = []
    first_real = None
    first_cpp_default_real = None
    first_cpp_math_real = None
    first_identity_only = None
    for block in range(0, 10, 2):
        prefix = f"physical_block_{block:02d}__"
        semantic = semantic_router_report(
            sides["cpp"][prefix + "router_topk_indices"],
            sides["python_default"][prefix + "router_topk_indices"],
            sides["python_math"][prefix + "router_topk_indices"],
            sides["cpp"][prefix + "identity_weight_sum"],
            sides["python_default"][prefix + "identity_weight_sum"],
            sides["python_math"][prefix + "identity_weight_sum"])
        shortcut = next(row for row in floating if row["name"] == prefix + "moe_shortcut")
        semantic.update({"physical_block": block,
                         "shortcut_within_criterion_vs_both": not shortcut["outside_vs_both"]})
        routers.append(semantic)
        if semantic["has_real_expert_set_difference"] and first_real is None:
            first_real = block
        if semantic["has_cpp_vs_python_default_real_expert_set_difference"] and first_cpp_default_real is None:
            first_cpp_default_real = block
        if semantic["has_cpp_vs_python_math_real_expert_set_difference"] and first_cpp_math_real is None:
            first_cpp_math_real = block
        if semantic["has_identity_only_id_substitution"] and first_identity_only is None:
            first_identity_only = block

    block_outputs = [row for row in floating if row["suffix"] == "block_output"]
    failures = [row["physical_block"] for row in block_outputs if row["outside_vs_both"]]
    first_persistent = None
    for block in failures:
        later = [row for row in block_outputs if row["physical_block"] >= block]
        if later and all(row["outside_vs_both"] for row in later):
            first_persistent = block
            break
    reconstruction = {name: reconstruct_component_boundaries(values) for name, values in sides.items()}
    classification = classify_numerical_attribution(
        routers, floating, first_persistent, reconstruction)
    return {
        "accepted": False, "kind": "longcat-next-block-component-numerical-attribution",
        "diagnostic_criterion_source": "bf16.physical_block_02",
        "diagnostic_criterion": dict(criterion),
        "first_physical_block_with_real_expert_set_difference": first_real,
        "first_physical_block_with_cpp_vs_python_default_real_expert_set_difference": first_cpp_default_real,
        "first_physical_block_with_cpp_vs_python_math_real_expert_set_difference": first_cpp_math_real,
        "first_physical_block_with_identity_only_id_substitution": first_identity_only,
        "first_persistent_block_output_failure_vs_both_backends": first_persistent,
        "first_component_where_cpp_bf16_rounding_changes_default_pass_state": first_default_change,
        "first_component_where_cpp_bf16_rounding_changes_math_pass_state": first_math_change,
        "component_with_largest_rms_improvement_after_cpp_bf16_rounding": largest_improvement,
        "revised_attribution_classification": classification,
        "router_semantics": routers, "floating_components": floating,
        "residual_reconstruction": reconstruction,
    }


def exact_result(reference, candidate):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    return {"passed": bool(np.array_equal(reference, candidate)),
            "reference": reference.tolist(), "candidate": candidate.tolist()}


def require_finite_logits(logits, case):
    if not np.isfinite(logits).all():
        raise ValueError(f"{case}: complete C++ direct logits contain NaN or infinity")


def reference_key(npz, case, suffix, required=True):
    matches = [name for name in npz.files if name.startswith(case + "/") and name.endswith("/" + suffix)]
    if len(matches) != 1:
        if not required and not matches:
            return None
        raise ValueError(f"{case}/{suffix}: expected one reference array, got {matches}")
    return matches[0]


def make_case_manifest(npz, output):
    cases = []
    prefixes = sorted(name[:-len("/input_ids")] for name in npz.files if name.endswith("/input_ids"))
    for case in prefixes:
        row = {"name": case.replace("/", "_")}
        for suffix in INTEGER_SUFFIXES:
            row[suffix] = npz[reference_key(npz, case, suffix)].reshape(-1).astype(np.int64).tolist()
        row["reference_prefix"] = case
        greedy_key = None
        if case.startswith("tokenizer_prompt_"):
            greedy_key = "greedy_ids/prompt_" + case.rsplit("_", 1)[-1]
        row["greedy_eight_tokens"] = greedy_key in npz.files if greedy_key else False
        if row["greedy_eight_tokens"] and npz[greedy_key].size != len(row["input_ids"]) + 8:
            raise ValueError(f"{greedy_key}: expected prompt plus exactly eight generated tokens")
        cases.append(row)
    output.write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="ascii")
    return cases


def compare_case(npz, metadata, case, capture_dir, precision, policy):
    inputs = json.loads((capture_dir / "inputs.json").read_text(encoding="ascii"))
    prefix = case["reference_prefix"]
    exact = {}
    for suffix in INTEGER_SUFFIXES:
        ref = npz[reference_key(npz, prefix, suffix)].reshape(-1).astype(np.int64)
        got = np.asarray(inputs[suffix], dtype=np.int64)
        exact[suffix] = exact_result(ref, got)
    mask = np.asarray(case["attention_mask"], dtype=bool)
    arrays = {}
    captures = read_manifest(capture_dir / "captures.tsv")
    for cpp_name, (dtype, dims, raw_path) in captures.items():
        suffix = CPP_TO_REFERENCE[cpp_name]
        key = reference_key(npz, prefix, suffix, required=cpp_name != "final_logits")
        if key is None:
            continue
        candidate = decode_raw(dtype, dims, raw_path, "logits" if cpp_name == "final_logits" else "hidden")
        reference = npz[key].astype(np.float32)
        # Embedding surfaces include every position. Attention-dependent
        # surfaces compare only attended rows on both sides.
        if suffix.startswith("physical_block_") or suffix == "final_normalized_hidden_state":
            candidate, reference = candidate[:, mask, :], reference[:, mask, :]
        tolerance = policy_for(policy, precision, suffix)
        arrays[suffix] = floating_result(reference, candidate, tolerance)

    # Every case derives decoding summaries from the direct-forward logits,
    # using the reference's reversed argsort tie rule.
    raw_logits = np.fromfile(capture_dir / "final_logits.f32.raw", dtype=np.float32).reshape(1, -1)
    require_finite_logits(raw_logits, prefix)
    order = np.argsort(raw_logits, axis=-1)[:, ::-1]
    top_ids = order[:, :10].astype(np.int64)
    top_values = np.take_along_axis(raw_logits, top_ids, axis=-1)
    selected_ids = np.asarray(metadata["selected_logit_token_ids"], dtype=np.int64)
    derived = {"selected_logits": raw_logits[:, selected_ids], "topk_values": top_values}
    for suffix, candidate in derived.items():
        reference = npz[reference_key(npz, prefix, suffix)].astype(np.float32)
        arrays[suffix] = floating_result(reference, candidate, policy_for(policy, precision, suffix))
    integer_results = {}
    for suffix, candidate in {"topk_token_ids": top_ids, "argmax_token_id": order[:, :1]}.items():
        reference = npz[reference_key(npz, prefix, suffix)].astype(np.int64)
        integer_results[suffix] = exact_result(reference, candidate)
    decoding = json.loads((capture_dir / "decoding.json").read_text(encoding="ascii"))
    arrays["decoding_topk_values"] = floating_result(
        top_values, np.asarray(decoding["top_k_values"], dtype=np.float32).reshape(1, -1),
        policy_for(policy, precision, "topk_values"))
    integer_results["decoding_top_k_ids"] = {
        "passed": decoding["top_k_ids"] == top_ids.reshape(-1).tolist(),
        "reference": top_ids.reshape(-1).tolist(), "candidate": decoding["top_k_ids"]}
    integer_results["decoding_argmax_id"] = {
        "passed": decoding["argmax_id"] == int(order[0, 0]),
        "reference": int(order[0, 0]), "candidate": decoding["argmax_id"]}
    if case["greedy_eight_tokens"]:
        prompt_index = case["name"].rsplit("_", 1)[-1]
        greedy_ref = npz[f"greedy_ids/prompt_{prompt_index}"].astype(np.int64).reshape(-1)
        greedy_got = np.asarray(decoding["prompt_plus_continuation_ids"], dtype=np.int64)
        integer_results["greedy_ids"] = exact_result(greedy_ref, greedy_got)
        if len(decoding["greedy_continuation_ids"]) != 8:
            integer_results["greedy_ids"]["passed"] = False
    passed = all(x["passed"] for x in arrays.values()) and all(x["passed"] for x in exact.values()) and all(x["passed"] for x in integer_results.values())
    return {"inputs": exact, "floating_arrays": arrays, "exact_decoding": integer_results, "passed": passed}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--precision", choices=("bf16", "f16"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--capture-exe", type=Path)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--flash-attn", choices=("auto", "disabled", "enabled"), default="auto")
    p.add_argument("--layer0-diagnostic", type=int, choices=(0, 1), default=0)
    p.add_argument("--all-blocks-diagnostic", type=int, choices=(0, 1), default=0)
    p.add_argument("--all-blocks-reference-npz", type=Path)
    p.add_argument("--block-components-diagnostic", type=int, choices=(0, 1), default=0)
    p.add_argument("--block-components-window-diagnostic", type=int, choices=(0, 1), default=0)
    p.add_argument("--block-components-window-start", type=int, default=10)
    p.add_argument("--block-components-window-count", type=int, default=4)
    p.add_argument("--block-components-window-default-npz", type=Path)
    p.add_argument("--block-components-window-math-npz", type=Path)
    p.add_argument("--component-window-replay-only", type=int, choices=(0, 1), default=0)
    p.add_argument("--router-linear-diagnostic-npz", type=Path)
    p.add_argument("--router-linear-diagnostic-json", type=Path)
    p.add_argument("--longcat-bf16-boundary-rounding", type=int, choices=(0, 1), default=0)
    p.add_argument("--longcat-bf16-hidden-surface-rounding", type=int, choices=(0, 1), default=0)
    p.add_argument("--block-components-default-npz", type=Path)
    p.add_argument("--block-components-math-npz", type=Path)
    p.add_argument("--component-replay-only", type=int, choices=(0, 1), default=0)
    p.add_argument("--component-attribution-replay-only", type=int, choices=(0, 1), default=0)
    p.add_argument("--component-profile-diff-replay-only", type=int, choices=(0, 1), default=0)
    p.add_argument("--baseline-capture-dir", type=Path)
    p.add_argument("--rounded-capture-dir", type=Path)
    p.add_argument("--baseline-profile-identity", choices=("legacy-default-off",))
    p.add_argument("--profile-execution-context", choices=("cpu-flash-disabled-threads-0-bf16",))
    p.add_argument("--component-source-dtype-json", type=Path)
    p.add_argument("--case", action="append", default=[], help="capture only this reference case (repeatable)")
    p.add_argument("--tolerance-policy", type=Path, default=Path(__file__).parent / "fixtures/longcat-next/stage1-tolerances.json")
    return p


def validate_all_blocks_options(args):
    if args.longcat_bf16_hidden_surface_rounding and not args.longcat_bf16_boundary_rounding:
        raise ValueError("--longcat-bf16-hidden-surface-rounding requires boundary rounding")
    replay_flags = (args.component_replay_only, args.component_attribution_replay_only,
                    args.component_profile_diff_replay_only, args.component_window_replay_only)
    if sum(replay_flags) > 1:
        raise ValueError("component replay modes are mutually exclusive")
    replay_only = any(replay_flags)
    if replay_only:
        if args.model is not None or args.capture_exe is not None:
            raise ValueError("--component-replay-only must not receive --model or --capture-exe")
        if args.all_blocks_diagnostic or args.all_blocks_reference_npz is not None:
            raise ValueError("--component-replay-only cannot run all-block capture")
        if args.component_window_replay_only:
            if not args.block_components_window_diagnostic or args.block_components_diagnostic:
                raise ValueError("window replay requires only --block-components-window-diagnostic 1")
        elif not args.block_components_diagnostic or args.block_components_window_diagnostic:
            raise ValueError("canonical replay requires only --block-components-diagnostic 1")
        if args.component_profile_diff_replay_only:
            required = (args.baseline_capture_dir, args.rounded_capture_dir,
                        args.baseline_profile_identity, args.profile_execution_context)
            if any(value is None for value in required):
                raise ValueError("profile diff replay requires both captures, baseline identity, and execution context")
        elif args.baseline_capture_dir is not None or args.rounded_capture_dir is not None:
            raise ValueError("profile capture directories require --component-profile-diff-replay-only 1")
    elif args.model is None or args.capture_exe is None:
        raise ValueError("normal parity mode requires --model and --capture-exe")
    if args.all_blocks_diagnostic:
        if len(args.case) != 1:
            raise ValueError("--all-blocks-diagnostic requires exactly one --case")
        if args.all_blocks_reference_npz is None:
            raise ValueError("--all-blocks-diagnostic requires --all-blocks-reference-npz")
    elif args.all_blocks_reference_npz is not None:
        raise ValueError("--all-blocks-reference-npz requires --all-blocks-diagnostic 1")
    component_paths = (args.block_components_default_npz, args.block_components_math_npz)
    if args.block_components_diagnostic and args.block_components_window_diagnostic:
        raise ValueError("canonical and window component diagnostics are mutually exclusive")
    if args.block_components_diagnostic:
        if len(args.case) != 1:
            raise ValueError("--block-components-diagnostic requires exactly one --case")
        if any(path is None for path in component_paths):
            raise ValueError("--block-components-diagnostic requires both component reference NPZs")
    elif any(path is not None for path in component_paths):
        raise ValueError("component reference NPZs require --block-components-diagnostic 1")
    window_paths = (args.block_components_window_default_npz, args.block_components_window_math_npz)
    if args.block_components_window_diagnostic:
        component_window_names(args.block_components_window_start, args.block_components_window_count)
        if len(args.case) != 1: raise ValueError("component-window diagnostic requires exactly one --case")
        if any(path is None for path in window_paths): raise ValueError("component-window diagnostic requires both window NPZs")
    elif any(path is not None for path in window_paths):
        raise ValueError("component-window NPZs require window diagnostic mode")
    linear_paths = (args.router_linear_diagnostic_npz, args.router_linear_diagnostic_json)
    if any(path is not None for path in linear_paths):
        if not args.component_window_replay_only or any(path is None for path in linear_paths):
            raise ValueError("router-linear diagnostics require both artifacts in component-window replay-only mode")


def validate_component_window_capture_metadata(root, start, count):
    """Bind replay to the exact diagnostic window recorded by the capture."""
    path = Path(root) / "capture-run-metadata.json"
    if not path.is_file():
        raise ValueError("component-window replay requires capture-run-metadata.json")
    metadata = json.loads(path.read_text(encoding="ascii"))
    expected = (True, int(start), int(count))
    observed = (metadata.get("block_components_window_diagnostic"),
                metadata.get("block_components_window_start"),
                metadata.get("block_components_window_count"))
    if observed != expected:
        raise ValueError(f"component-window metadata mismatch: expected {expected}, got {observed}")
    if metadata.get("block_components_diagnostic") is not False:
        raise ValueError("component-window metadata must disable canonical component capture")
    return metadata


def main():
    args = build_parser().parse_args()
    try:
        validate_all_blocks_options(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    metadata = json.loads((args.reference_dir / f"longcat-next-core-{args.precision}.json").read_text(encoding="ascii"))
    npz = np.load(args.reference_dir / f"longcat-next-core-{args.precision}.npz", allow_pickle=False)
    policy = json.loads(args.tolerance_policy.read_text(encoding="ascii"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_reference_finiteness(npz, args.output_dir)
    replay_only = (args.component_replay_only or args.component_attribution_replay_only or
                   args.component_profile_diff_replay_only or args.component_window_replay_only)
    if replay_only:
        with tempfile.TemporaryDirectory() as temporary:
            cases = make_case_manifest(npz, Path(temporary) / "case-manifest.json")
    else:
        manifest_path = args.output_dir / "case-manifest.json"
        cases = make_case_manifest(npz, manifest_path)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["reference_prefix"] in selected]
        missing = selected - {case["reference_prefix"] for case in cases}
        if missing:
            raise ValueError(f"unknown requested cases: {sorted(missing)}")
        if not replay_only:
            manifest_path.write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="ascii")
    greedy_cases = {case["reference_prefix"] for case in cases if case["greedy_eight_tokens"]}
    if not args.case and greedy_cases != {"tokenizer_prompt_0", "tokenizer_prompt_1"}:
        raise ValueError(f"expected eight-token greedy references for both tokenizer prompts, got {greedy_cases}")
    if replay_only:
        if args.component_window_replay_only:
            validate_component_window_capture_metadata(
                args.output_dir, args.block_components_window_start,
                args.block_components_window_count)
            criterion = policy["bf16"]["physical_block_02"]
            capture_dir = args.output_dir / cases[0]["name"]
            if not capture_dir.is_dir(): capture_dir = args.output_dir
            with np.load(args.block_components_window_default_npz, allow_pickle=False) as default_window, \
                 np.load(args.block_components_window_math_npz, allow_pickle=False) as math_window:
                router_weights = None
                if args.router_linear_diagnostic_npz is not None:
                    linear_metadata = json.loads(args.router_linear_diagnostic_json.read_text(encoding="ascii"))
                    if (linear_metadata.get("bounded_physical_blocks") != [10, 12] or
                            linear_metadata.get("model_instantiated") is not False or
                            linear_metadata.get("inference_executed") is not False):
                        raise ValueError("invalid bounded router-linear diagnostic metadata")
                    router_weights = np.load(args.router_linear_diagnostic_npz, allow_pickle=False)
                try:
                    report = compare_component_window(default_window, math_window, capture_dir,
                        cases[0]["attention_mask"], criterion, args.block_components_window_start,
                        args.block_components_window_count, router_weights)
                finally:
                    if router_weights is not None: router_weights.close()
            (args.output_dir / "block-components-window-three-way.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="ascii")
            return
        if args.component_profile_diff_replay_only:
            criterion = policy["bf16"]["physical_block_02"]
            reference_argmax = int(npz[reference_key(
                npz, cases[0]["reference_prefix"], "argmax_token_id")].reshape(-1)[0])
            with np.load(args.block_components_default_npz, allow_pickle=False) as default_components, \
                 np.load(args.block_components_math_npz, allow_pickle=False) as math_components:
                report = compare_component_profiles(
                    default_components, math_components, args.baseline_capture_dir,
                    args.rounded_capture_dir, cases[0]["name"], cases[0]["attention_mask"],
                    criterion, args.baseline_profile_identity, args.profile_execution_context,
                    args.component_source_dtype_json, reference_argmax)
            (args.output_dir / "block-components-profile-diff.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="ascii")
            return
        read_manifest(args.output_dir / cases[0]["name"] / "captures.tsv")
        criterion = policy["bf16"]["physical_block_02"]
        with np.load(args.block_components_default_npz, allow_pickle=False) as default_components, \
             np.load(args.block_components_math_npz, allow_pickle=False) as math_components:
            component_report = compare_block_components(
                default_components, math_components, args.output_dir / cases[0]["name"],
                cases[0]["attention_mask"], criterion)
        (args.output_dir / "block-components-three-way.json").write_text(
            json.dumps(component_report, indent=2) + "\n", encoding="ascii")
        if args.component_attribution_replay_only:
            with np.load(args.block_components_default_npz, allow_pickle=False) as default_components, \
                 np.load(args.block_components_math_npz, allow_pickle=False) as math_components:
                attribution = numerical_attribution_report(
                    default_components, math_components, args.output_dir / cases[0]["name"],
                    cases[0]["attention_mask"], criterion)
            (args.output_dir / "block-components-numerical-attribution.json").write_text(
                json.dumps(attribution, indent=2) + "\n", encoding="ascii")
        return
    command = [str(args.capture_exe), "--model", str(args.model), "--case-manifest", str(manifest_path),
                    "--output-dir", str(args.output_dir), "--n-gpu-layers", str(args.n_gpu_layers),
                    "--threads", str(args.threads), "--flash-attn", args.flash_attn,
                    "--layer0-diagnostic", str(args.layer0_diagnostic),
                    "--all-blocks-diagnostic", str(args.all_blocks_diagnostic),
                    "--block-components-diagnostic", str(args.block_components_diagnostic),
                    "--block-components-window-diagnostic", str(args.block_components_window_diagnostic),
                    "--block-components-window-start", str(args.block_components_window_start),
                    "--block-components-window-count", str(args.block_components_window_count),
                    "--longcat-bf16-boundary-rounding", str(args.longcat_bf16_boundary_rounding),
                    "--longcat-bf16-hidden-surface-rounding", str(args.longcat_bf16_hidden_surface_rounding)]
    try:
        run_capture_after_validation(validation, command)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    reports = {case["name"]: compare_case(npz, metadata, case, args.output_dir / case["name"], args.precision, policy) for case in cases}
    if args.all_blocks_diagnostic:
        criterion = policy["bf16"]["physical_block_02"]
        with np.load(args.all_blocks_reference_npz, allow_pickle=False) as all_blocks_reference:
            all_blocks = compare_all_blocks(
                all_blocks_reference, args.output_dir / cases[0]["name"],
                cases[0]["attention_mask"], criterion)
        (args.output_dir / "all-blocks-comparison.json").write_text(
            json.dumps(all_blocks, indent=2) + "\n", encoding="ascii")
    if args.block_components_diagnostic:
        criterion = policy["bf16"]["physical_block_02"]
        with np.load(args.block_components_default_npz, allow_pickle=False) as default_components, \
             np.load(args.block_components_math_npz, allow_pickle=False) as math_components:
            component_report = compare_block_components(
                default_components, math_components, args.output_dir / cases[0]["name"],
                cases[0]["attention_mask"], criterion)
        (args.output_dir / "block-components-three-way.json").write_text(
            json.dumps(component_report, indent=2) + "\n", encoding="ascii")
    if args.block_components_window_diagnostic:
        criterion = policy["bf16"]["physical_block_02"]
        with np.load(args.block_components_window_default_npz, allow_pickle=False) as default_window, \
             np.load(args.block_components_window_math_npz, allow_pickle=False) as math_window:
            window_report = compare_component_window(default_window, math_window,
                args.output_dir / cases[0]["name"], cases[0]["attention_mask"], criterion,
                args.block_components_window_start, args.block_components_window_count)
        (args.output_dir / "block-components-window-three-way.json").write_text(
            json.dumps(window_report, indent=2) + "\n", encoding="ascii")
    overall = all(row["passed"] for row in reports.values())
    report = {"precision": args.precision, "tolerance_policy": str(args.tolerance_policy), "cases": reports, "passed": overall}
    (args.output_dir / "comparison-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if not overall:
        raise SystemExit("LongCat-Next C++ parity comparison failed; diagnose the report without widening tolerances")


if __name__ == "__main__":
    main()
