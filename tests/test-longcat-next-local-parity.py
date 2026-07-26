#!/usr/bin/env python3
"""Opt-in LongCat-Next C++ capture/comparison with frozen Stage-1 policy."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

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
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--precision", choices=("bf16", "f16"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--capture-exe", type=Path, required=True)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--flash-attn", choices=("auto", "disabled", "enabled"), default="auto")
    p.add_argument("--layer0-diagnostic", type=int, choices=(0, 1), default=0)
    p.add_argument("--case", action="append", default=[], help="capture only this reference case (repeatable)")
    p.add_argument("--tolerance-policy", type=Path, default=Path(__file__).parent / "fixtures/longcat-next/stage1-tolerances.json")
    return p


def main():
    args = build_parser().parse_args()
    metadata = json.loads((args.reference_dir / f"longcat-next-core-{args.precision}.json").read_text(encoding="ascii"))
    npz = np.load(args.reference_dir / f"longcat-next-core-{args.precision}.npz", allow_pickle=False)
    policy = json.loads(args.tolerance_policy.read_text(encoding="ascii"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_reference_finiteness(npz, args.output_dir)
    manifest_path = args.output_dir / "case-manifest.json"
    cases = make_case_manifest(npz, manifest_path)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["reference_prefix"] in selected]
        missing = selected - {case["reference_prefix"] for case in cases}
        if missing:
            raise ValueError(f"unknown requested cases: {sorted(missing)}")
        manifest_path.write_text(json.dumps({"schema_version": 1, "cases": cases}, indent=2) + "\n", encoding="ascii")
    greedy_cases = {case["reference_prefix"] for case in cases if case["greedy_eight_tokens"]}
    if not args.case and greedy_cases != {"tokenizer_prompt_0", "tokenizer_prompt_1"}:
        raise ValueError(f"expected eight-token greedy references for both tokenizer prompts, got {greedy_cases}")
    command = [str(args.capture_exe), "--model", str(args.model), "--case-manifest", str(manifest_path),
                    "--output-dir", str(args.output_dir), "--n-gpu-layers", str(args.n_gpu_layers),
                    "--threads", str(args.threads), "--flash-attn", args.flash_attn,
                    "--layer0-diagnostic", str(args.layer0_diagnostic)]
    try:
        run_capture_after_validation(validation, command)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    reports = {case["name"]: compare_case(npz, metadata, case, args.output_dir / case["name"], args.precision, policy) for case in cases}
    overall = all(row["passed"] for row in reports.values())
    report = {"precision": args.precision, "tolerance_policy": str(args.tolerance_policy), "cases": reports, "passed": overall}
    (args.output_dir / "comparison-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if not overall:
        raise SystemExit("LongCat-Next C++ parity comparison failed; diagnose the report without widening tolerances")


if __name__ == "__main__":
    main()
