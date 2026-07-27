#!/usr/bin/env python3
"""Opt-in LongCat-Next C++ capture/comparison with frozen Stage-1 policy."""
import argparse
import json
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
    if first_router_set is not None and (first_material is None or
            first_router_set <= int(first_material[15:17])):
        classification = "router indices"
    return {"accepted": False, "array_count": 110,
            "diagnostic_criterion_source": "bf16.physical_block_02",
            "component_decode_provenance": decode_provenance,
            "first_physical_block_with_router_selected_set_difference": first_router_set,
            "first_component_outside_diagnostic_criterion_vs_both_backends": first_material,
            "first_large_discrepancy_classification": classification,
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
        "tokens": token_rows,
        "identity_weight_sum_comparisons": identity_metrics,
        "first_token_with_real_expert_set_mismatch": first_real_mismatch,
        "real_expert_lists_at_first_mismatch": first_real_lists,
        "has_real_expert_set_difference": first_real_mismatch is not None,
        "has_identity_presence_difference": any(
            not row["identity_expert_presence_equivalent"] for row in token_rows),
        "has_identity_only_id_substitution": any(
            row["raw_set_mismatch_is_identity_only_id_substitution"] for row in token_rows),
    }


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


def classify_numerical_attribution(router_reports, floating_reports, first_persistent_block):
    for router in router_reports:
        semantic_difference = (router["has_real_expert_set_difference"] or
                               router["has_identity_presence_difference"])
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
    classification = classify_numerical_attribution(routers, floating, first_persistent)
    return {
        "accepted": False, "kind": "longcat-next-block-component-numerical-attribution",
        "diagnostic_criterion_source": "bf16.physical_block_02",
        "diagnostic_criterion": dict(criterion),
        "first_physical_block_with_real_expert_set_difference": first_real,
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
    p.add_argument("--block-components-default-npz", type=Path)
    p.add_argument("--block-components-math-npz", type=Path)
    p.add_argument("--component-replay-only", type=int, choices=(0, 1), default=0)
    p.add_argument("--component-attribution-replay-only", type=int, choices=(0, 1), default=0)
    p.add_argument("--case", action="append", default=[], help="capture only this reference case (repeatable)")
    p.add_argument("--tolerance-policy", type=Path, default=Path(__file__).parent / "fixtures/longcat-next/stage1-tolerances.json")
    return p


def validate_all_blocks_options(args):
    replay_only = args.component_replay_only or args.component_attribution_replay_only
    if replay_only:
        if not args.block_components_diagnostic:
            raise ValueError("--component-replay-only requires --block-components-diagnostic 1")
        if args.model is not None or args.capture_exe is not None:
            raise ValueError("--component-replay-only must not receive --model or --capture-exe")
        if args.all_blocks_diagnostic or args.all_blocks_reference_npz is not None:
            raise ValueError("--component-replay-only cannot run all-block capture")
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
    if args.block_components_diagnostic:
        if len(args.case) != 1:
            raise ValueError("--block-components-diagnostic requires exactly one --case")
        if any(path is None for path in component_paths):
            raise ValueError("--block-components-diagnostic requires both component reference NPZs")
    elif any(path is not None for path in component_paths):
        raise ValueError("component reference NPZs require --block-components-diagnostic 1")


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
    replay_only = args.component_replay_only or args.component_attribution_replay_only
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
                    "--block-components-diagnostic", str(args.block_components_diagnostic)]
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
    overall = all(row["passed"] for row in reports.values())
    report = {"precision": args.precision, "tolerance_policy": str(args.tolerance_policy), "cases": reports, "passed": overall}
    (args.output_dir / "comparison-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if not overall:
        raise SystemExit("LongCat-Next C++ parity comparison failed; diagnose the report without widening tolerances")


if __name__ == "__main__":
    main()
