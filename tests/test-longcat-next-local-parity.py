#!/usr/bin/env python3
"""Opt-in local LongCat-Next reference/candidate inventory gate.

This intentionally does not select numeric tolerances. It validates the accepted
Python reference contract and the required C++ capture families before a reviewer
freezes cross-implementation tolerances.
"""

import argparse
import json
from pathlib import Path


REQUIRED_SUFFIXES = (
    "/base_embedding",
    "/ngram_projection_raw_00",
    "/ngram_projection_raw_11",
    "/fused_pre_trunk_embedding",
    "/physical_block_00",
    "/physical_block_01",
    "/physical_block_02",
    "/physical_block_27",
    "/final_normalized_hidden_state",
    "/selected_logits",
    "greedy_ids/prompt_0",
)


def validate_reference(directory: Path) -> None:
    metadata_files = sorted(directory.glob("longcat-next-core-*.json"))
    reproducibility = directory / "longcat-next-core-reproducibility.json"
    if len(metadata_files) != 1 or not reproducibility.is_file():
        raise ValueError(f"{directory}: expected one precision metadata file and reproducibility report")
    metadata = json.loads(metadata_files[0].read_text(encoding="ascii"))
    report = json.loads(reproducibility.read_text(encoding="ascii"))
    arrays = metadata.get("arrays", {})
    if len(arrays) != 433:
        raise ValueError(f"{directory}: expected 433 arrays, got {len(arrays)}")
    if report.get("repeat_count") != 2:
        raise ValueError(f"{directory}: expected two official repeats")
    if report.get("comparison_tolerances") != {"bf16": None, "f16": None}:
        raise ValueError(f"{directory}: comparison tolerances are not pending/null")
    repeats = report.get("arrays", {})
    if len(repeats) != 433 or not all(row.get("byte_identical") for row in repeats.values()):
        raise ValueError(f"{directory}: official repeats are not 433/433 byte-identical")
    names = tuple(arrays)
    for suffix in REQUIRED_SUFFIXES:
        if not any(name.endswith(suffix) for name in names):
            raise ValueError(f"{directory}: missing required capture family {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-reference", type=Path, required=True)
    parser.add_argument("--f16-reference", type=Path, required=True)
    args = parser.parse_args()
    validate_reference(args.bf16_reference)
    validate_reference(args.f16_reference)
    print("Reference inventories accepted; C++ numeric comparison remains blocked until tolerances are frozen.")


if __name__ == "__main__":
    main()
