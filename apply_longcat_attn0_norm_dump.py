#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    r"D:\llama.cpp-longcat-pre-gate4\common\debug.cpp"
)

EXPECTED_BEFORE = (
    "70636687ab319e7a66d9a4c16f15fcb7376f5afecaf9892148b8d9ecf1119bbf"
)

MARKER = "LONGCAT_LOGICAL0_ATTN0_NORM_VECTOR_DUMP"

OLD = """    if (tensor_name == "ffn_inp-1") {
        filename = "logical0_attn1_resid.bin";
        return true;
    }

    for (int logical = 0; logical < 13; ++logical) {
"""

NEW = """    if (tensor_name == "ffn_inp-1") {
        filename = "logical0_attn1_resid.bin";
        return true;
    }

    // LONGCAT_LOGICAL0_ATTN0_NORM_VECTOR_DUMP:
    // First normalized activation entering physical attention block 0.
    if (tensor_name == "attn_norm-0") {
        filename = "logical0_attn0_norm.bin";
        return true;
    }

    for (int logical = 0; logical < 13; ++logical) {
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


raw = TARGET.read_bytes()
before = sha256(raw)

print("before_sha256=" + before)

if before != EXPECTED_BEFORE:
    raise SystemExit(
        "STOP: common/debug.cpp is not at the expected logical-0 dump state"
    )

try:
    decoded = raw.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit(
        "STOP: common/debug.cpp is not UTF-8: %s"
        % exc
    )

newline = "\r\n" if "\r\n" in decoded else "\n"
text = decoded.replace("\r\n", "\n")

if MARKER in text:
    raise SystemExit(
        "STOP: attn0 norm dump marker already exists"
    )

count = text.count(OLD)

print("anchor_count=%d" % count)

if count != 1:
    raise SystemExit(
        "STOP: attn0 norm anchor count is %d, expected exactly 1"
        % count
    )

text = text.replace(OLD, NEW, 1)

if text.count(MARKER) != 1:
    raise SystemExit(
        "STOP: attn0 norm marker count is not exactly 1 after patch"
    )

output = text.replace("\n", newline).encode("utf-8")

TARGET.write_bytes(output)

after = sha256(output)

if after == before:
    raise SystemExit(
        "STOP: common/debug.cpp SHA did not change"
    )

print("after_sha256=" + after)
print("newline=" + ("CRLF" if newline == "\r\n" else "LF"))
print("LONGCAT ATTN0 NORM DUMP: APPLIED")