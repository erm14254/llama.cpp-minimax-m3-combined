#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    r"D:\llama.cpp-longcat-pre-gate4\src\models\longcat-flash-ngram.cpp"
)

EXPECTED_BEFORE = (
    "fd0b2e0d150e04fbdb3c0ba38ae7b051d21b999ef836a06207aa38e7d20ee1dd"
)

MARKER_Q = "LONGCAT_ATTN0_QA_NORM_EPS_DIAGNOSTIC"
MARKER_KV = "LONGCAT_ATTN0_KVA_NORM_EPS_DIAGNOSTIC"

OLD_Q = """                q_lora = build_norm(q_lora, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q_lora, "q", il);
"""

NEW_Q = """                if (il == 0) {
                    // LONGCAT_ATTN0_QA_NORM_EPS_DIAGNOSTIC:
                    // HF LongCat q_a_layernorm uses its default epsilon 1e-6.
                    q_lora = ggml_rms_norm(ctx0, q_lora, 1.0e-6f);
                    q_lora = ggml_mul(
                        ctx0,
                        q_lora,
                        model.layers[il].attn_q_a_norm);
                } else {
                    q_lora = build_norm(
                        q_lora,
                        model.layers[il].attn_q_a_norm,
                        nullptr,
                        LLM_NORM_RMS,
                        il);
                }
                cb(q_lora, "q", il);
"""

OLD_KV = """            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);
"""

NEW_KV = """            if (il == 0) {
                // LONGCAT_ATTN0_KVA_NORM_EPS_DIAGNOSTIC:
                // HF LongCat kv_a_layernorm uses its default epsilon 1e-6.
                kv_cmpr = ggml_rms_norm(ctx0, kv_cmpr, 1.0e-6f);
                kv_cmpr = ggml_mul(
                    ctx0,
                    kv_cmpr,
                    model.layers[il].attn_kv_a_norm);
            } else {
                kv_cmpr = build_norm(
                    kv_cmpr,
                    model.layers[il].attn_kv_a_norm,
                    nullptr,
                    LLM_NORM_RMS,
                    il);
            }
            cb(kv_cmpr, "kv_cmpr", il);
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


raw = TARGET.read_bytes()
before = sha256(raw)

print("before_sha256=" + before)

if before != EXPECTED_BEFORE:
    raise SystemExit(
        "STOP: LongCat source is not at the expected attn0-RMS diagnostic state"
    )

decoded = raw.decode("utf-8")
newline = "\r\n" if "\r\n" in decoded else "\n"
text = decoded.replace("\r\n", "\n")

for marker in (MARKER_Q, MARKER_KV):
    if marker in text:
        raise SystemExit(
            "STOP: MLA epsilon diagnostic marker already exists: " + marker
        )

q_count = text.count(OLD_Q)
kv_count = text.count(OLD_KV)

print("q_anchor_count=%d" % q_count)
print("kv_anchor_count=%d" % kv_count)

if q_count != 1:
    raise SystemExit(
        "STOP: q-A norm anchor count is %d, expected 1"
        % q_count
    )

if kv_count != 1:
    raise SystemExit(
        "STOP: KV-A norm anchor count is %d, expected 1"
        % kv_count
    )

text = text.replace(OLD_Q, NEW_Q, 1)
text = text.replace(OLD_KV, NEW_KV, 1)

if text.count(MARKER_Q) != 1:
    raise SystemExit(
        "STOP: q-A epsilon marker count is not 1"
    )

if text.count(MARKER_KV) != 1:
    raise SystemExit(
        "STOP: KV-A epsilon marker count is not 1"
    )

output = text.replace("\n", newline).encode("utf-8")
TARGET.write_bytes(output)

after = sha256(output)

if after == before:
    raise SystemExit("STOP: source SHA did not change")

print("after_sha256=" + after)
print("newline=" + ("CRLF" if newline == "\r\n" else "LF"))
print("LONGCAT ATTN0 MLA NORM EPS DIAGNOSTIC: APPLIED")