#!/usr/bin/env python3
"""Opt-in LongCat-Next llama.cpp capture/comparison runner.

Floating errors are reported, never accepted against an invented tolerance.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


CPP_TO_REFERENCE = {
    "inp_embd": "base_embedding",
    "inp_embd_ngram": "fused_pre_trunk_embedding",
    "l_out-0": "physical_block_00",
    "l_out-1": "physical_block_01",
    "l_out-2": "physical_block_02",
    "l_out-27": "physical_block_27",
    "result_norm": "final_normalized_hidden_state",
    "final_logits": "complete_final_position_logits",
}
for i in range(12):
    CPP_TO_REFERENCE[f"ngram_proj-{i}"] = f"ngram_projection_raw_{i:02d}"


def _read_manifest(path: Path):
    rows = {}
    for line in path.read_text(encoding="ascii").splitlines():
        name, dtype, shape, filename = line.split("\t")
        if name in rows:
            raise ValueError(f"duplicate C++ capture name: {name}")
        rows[name] = (dtype, tuple(map(int, shape.split(","))), path.parent / filename)
    return rows


def compare(reference_npz: Path, capture_dir: Path, report_path: Path, case_name: str):
    reference = np.load(reference_npz, allow_pickle=False)
    captures = _read_manifest(capture_dir / "captures.tsv")
    unexpected = sorted(set(captures) - set(CPP_TO_REFERENCE))
    required = set(CPP_TO_REFERENCE) - {"final_logits"}
    missing = sorted(required - set(captures))
    if unexpected or missing:
        raise ValueError(f"capture-name mismatch: missing={missing}, unexpected={unexpected}")
    report = {"comparison_tolerance": None, "arrays": {}}
    dtype_map = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16}
    for cpp_name, (_, shape, raw_path) in captures.items():
        suffix = CPP_TO_REFERENCE[cpp_name]
        matches = [name for name in reference.files
                   if name.startswith(case_name + "/") and name.endswith("/" + suffix)]
        if cpp_name == "final_logits" and not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"reference capture {suffix}: expected one match, got {matches}")
        ref = reference[matches[0]].astype(np.float32)
        dtype_name = captures[cpp_name][0]
        raw = np.fromfile(raw_path, dtype=dtype_map[dtype_name])
        if dtype_name == "bf16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        raw = raw.astype(np.float32).reshape(tuple(reversed(shape))).squeeze()
        if raw.shape != ref.shape:
            raise ValueError(f"{cpp_name}: shape {raw.shape} != {ref.shape}")
        diff = np.abs(raw - ref)
        denom = np.maximum(np.abs(ref), np.finfo(np.float32).tiny)
        report["arrays"][cpp_name] = {
            "reference": matches[0], "shape": list(ref.shape),
            "max_absolute_error": float(diff.max(initial=0)),
            "max_relative_error": float((diff / denom).max(initial=0)),
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument("--precision", choices=("bf16", "f16"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--capture-exe", type=Path, required=True)
    args = p.parse_args()
    metadata_path = args.reference_dir / f"longcat-next-core-{args.precision}.json"
    npz_path = args.reference_dir / f"longcat-next-core-{args.precision}.npz"
    json.loads(metadata_path.read_text(encoding="ascii"))
    reference = np.load(npz_path, allow_pickle=False)
    prompts = {name[:-len("/input_ids")]: reference[name].reshape(-1).tolist()
               for name in reference.files if name.endswith("/input_ids")}
    if not prompts:
        raise ValueError("reference NPZ contains no input_ids cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, input_ids in prompts.items():
        case_dir = args.output_dir / name.replace("/", "_")
        command = [str(args.capture_exe), "--model", str(args.model), "--tokens",
                   ",".join(map(str, input_ids)), "--output-dir", str(case_dir)]
        subprocess.run(command, check=True)
        reports[name] = compare(npz_path, case_dir, case_dir / "comparison.json", name)
    (args.output_dir / "comparison-report.json").write_text(
        json.dumps({"precision": args.precision, "tolerances": None, "cases": reports}, indent=2) + "\n",
        encoding="ascii")


if __name__ == "__main__":
    main()
