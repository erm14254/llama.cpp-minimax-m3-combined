#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

ORIGINAL_CAPTURE = Path(
    r"D:\llama.cpp-longcat-mtp\capture_longcat_sparse_hf_gate3_logits.py"
)

EXPECTED_ORIGINAL_SHA256 = (
    "bb82bcb6c3bc1d21685221a884dac3b39dc7af06f54fea6187f606dddf4213cb"
)
EXPECTED_TOKEN_SHA256 = (
    "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
)
EXPECTED_TOKEN_COUNT = 512
VOCAB_SIZE = 131072


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-bin", required=True)
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-json", required=True)
    ns = ap.parse_args()

    if not ORIGINAL_CAPTURE.is_file():
        stop(f"original validated capture script missing: {ORIGINAL_CAPTURE}")

    original_sha = sha256_file(ORIGINAL_CAPTURE)
    print(f"original_capture_sha256={original_sha}")

    if original_sha != EXPECTED_ORIGINAL_SHA256:
        stop(
            "validated Gate-3 capture SHA mismatch; expected "
            f"{EXPECTED_ORIGINAL_SHA256}, got {original_sha}"
        )

    tokens_bin = Path(ns.tokens_bin).resolve()

    if not tokens_bin.is_file():
        stop(f"token file missing: {tokens_bin}")

    token_sha = sha256_file(tokens_bin)
    print(f"tokens_bin_sha256={token_sha}")

    if token_sha != EXPECTED_TOKEN_SHA256:
        stop(
            "authoritative 512-token SHA mismatch; expected "
            f"{EXPECTED_TOKEN_SHA256}, got {token_sha}"
        )

    raw = tokens_bin.read_bytes()

    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop(
            f"unexpected token file length: {len(raw)} bytes; "
            f"expected {EXPECTED_TOKEN_COUNT * 4}"
        )

    input_ids = list(struct.unpack(f"<{EXPECTED_TOKEN_COUNT}i", raw))

    if len(input_ids) != EXPECTED_TOKEN_COUNT:
        stop(
            f"unexpected token count: {len(input_ids)} "
            f"!= {EXPECTED_TOKEN_COUNT}"
        )

    for i, token_id in enumerate(input_ids):
        if not 0 <= token_id < VOCAB_SIZE:
            stop(f"token {i} out of range: {token_id}")

    print(f"sequence_length={len(input_ids)}")
    print(f"first_8_ids={input_ids[:8]}")
    print(f"last_8_ids={input_ids[-8:]}")

    spec = importlib.util.spec_from_file_location(
        "validated_gate3_capture",
        ORIGINAL_CAPTURE,
    )
    if spec is None or spec.loader is None:
        stop("could not construct import spec for validated Gate-3 capture")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Replace only the validated script's hard-coded four-token oracle input.
    module.INPUT_IDS = input_ids

    sys.argv = [
        str(ORIGINAL_CAPTURE),
        "--model-dir",
        ns.model_dir,
        "--out-bin",
        ns.out_bin,
        "--out-json",
        ns.out_json,
    ]

    rc = module.main()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
