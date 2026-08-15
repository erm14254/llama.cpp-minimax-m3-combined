#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    r"D:\llama.cpp-longcat-pre-gate4\src\models\longcat-flash-ngram.cpp"
)

EXPECTED_BEFORE = (
    "75f8bbf0426a3eae4af0319dc93f62b0d6230e71aa518fb62b8c80cafaa5496f"
)

MARKER = "LONGCAT_ATTN0_HF_RMSNORM_DIAGNOSTIC"

OLD = """        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
"""

NEW = """        // norm
        if (il == 0) {
            // LONGCAT_ATTN0_HF_RMSNORM_DIAGNOSTIC:
            //
            // Transformers LongCat RMSNorm semantics for BF16 activations:
            //
            //   1. RMS normalization in F32
            //   2. round normalized activation to BF16
            //   3. multiply by the BF16 norm weight
            //   4. round output to BF16
            //
            // The GGUF norm weight is F32 but was proven to be the exact
            // numerical expansion of the HF BF16 weight, so multiplying the
            // BF16-rounded activation by that F32 value is numerically the
            // same pre-output-rounding product.
            //
            // Restore F32 afterward because the existing llama.cpp LongCat
            // trunk expects F32 activations.
            cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);

            cur = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);

            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);

            cur = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
            cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        } else {
            cur = build_norm(
                inpL,
                model.layers[il].attn_norm,
                NULL,
                LLM_NORM_RMS,
                il);
        }

        cb(cur, "attn_norm", il);
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


raw = TARGET.read_bytes()
before = sha256(raw)

print("before_sha256=" + before)

if before != EXPECTED_BEFORE:
    raise SystemExit(
        "STOP: LongCat source is not at the expected diagnostic state"
    )

try:
    decoded = raw.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit(
        "STOP: LongCat source is not UTF-8: %s"
        % exc
    )

newline = "\r\n" if "\r\n" in decoded else "\n"

text = decoded.replace("\r\n", "\n")

if MARKER in text:
    raise SystemExit(
        "STOP: attn0 HF RMSNorm diagnostic marker already exists"
    )

count = text.count(OLD)

print("anchor_count=%d" % count)

if count != 1:
    raise SystemExit(
        "STOP: attention-norm anchor count is %d, expected exactly 1"
        % count
    )

text = text.replace(OLD, NEW, 1)

if text.count(MARKER) != 1:
    raise SystemExit(
        "STOP: diagnostic marker count is not exactly 1 after patch"
    )

output = text.replace("\n", newline).encode("utf-8")

TARGET.write_bytes(output)

after = sha256(output)

if after == before:
    raise SystemExit(
        "STOP: source SHA did not change"
    )

print("after_sha256=" + after)
print("newline=" + ("CRLF" if newline == "\r\n" else "LF"))
print("LONGCAT ATTN0 HF RMSNORM DIAGNOSTIC: APPLIED")