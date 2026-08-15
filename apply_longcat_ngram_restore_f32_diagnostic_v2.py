#!/usr/bin/env python3

from pathlib import Path
import hashlib


TARGET = Path(
    r"D:\llama.cpp-longcat-pre-gate4\src\models\longcat-flash-ngram.cpp"
)

EXPECTED_BEFORE = (
    "c1e7e2cb082072e86ca042c6ae286d37c2af9246b9251bc82c128d426f88d0dc"
)

MARKER = "LONGCAT_NGRAM_BF16_RESTORE_F32_DIAGNOSTIC"

OLD = """        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);
        cb(inpL, "inp_embd_ngram", -1);

        res->add_input(std::move(inp));
"""

NEW = """        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);
        cb(inpL, "inp_embd_ngram", -1);

        // LONGCAT_NGRAM_BF16_RESTORE_F32_DIAGNOSTIC:
        // BF16 values are exactly representable in F32. Restore the graph's
        // expected activation type without changing the HF-exact values.
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_F32);

        res->add_input(std::move(inp));
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


raw = TARGET.read_bytes()
before = sha256(raw)

print("before_sha256=" + before)

if before != EXPECTED_BEFORE:
    raise SystemExit(
        "STOP: source SHA is not the expected BF16 diagnostic state"
    )

try:
    decoded = raw.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit("STOP: source is not UTF-8: %s" % exc)

newline = "\r\n" if "\r\n" in decoded else "\n"

# Normalize solely for deterministic multiline matching.
text = decoded.replace("\r\n", "\n")

if MARKER in text:
    raise SystemExit(
        "STOP: restore-F32 diagnostic marker already exists"
    )

count = text.count(OLD)

print("restore_anchor_count=%d" % count)

if count != 1:
    raise SystemExit(
        "STOP: restore-F32 anchor count is %d, expected exactly 1"
        % count
    )

text = text.replace(OLD, NEW, 1)

if text.count(MARKER) != 1:
    raise SystemExit(
        "STOP: restore-F32 marker count is not exactly 1 after patch"
    )

output = text.replace("\n", newline).encode("utf-8")

TARGET.write_bytes(output)

after = sha256(output)

if after == before:
    raise SystemExit(
        "STOP: source SHA did not change after patch"
    )

print("after_sha256=" + after)
print("newline=" + ("CRLF" if newline == "\r\n" else "LF"))
print("LONGCAT NGRAM RESTORE-F32 DIAGNOSTIC: APPLIED")