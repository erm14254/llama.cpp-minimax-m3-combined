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

OLD_Q = """                // LoRA Q path
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
"""

NEW_Q = """                // LoRA Q path
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                if (il == 0) {
                    // LONGCAT_ATTN0_QA_NORM_EPS_DIAGNOSTIC:
                    // HF LongCat q_a_layernorm uses the RMSNorm default eps=1e-6.
                    q = ggml_rms_norm(ctx0, q, 1.0e-6f);
                    q = ggml_mul(
                        ctx0,
                        q,
                        model.layers[il].attn_q_a_norm);
                } else {
                    q = build_norm(
                        q,
                        model.layers[il].attn_q_a_norm,
                        nullptr,
                        LLM_NORM_RMS,
                        il);
                }
                cb(q, "q", il);

                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
"""

OLD_KV = """            // normalize compressed KV
            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // MLA LoRA scaling: kv_cmpr *= sqrt(hidden_size / kv_lora_rank)
"""

NEW_KV = """            // normalize compressed KV
            if (il == 0) {
                // LONGCAT_ATTN0_KVA_NORM_EPS_DIAGNOSTIC:
                // HF LongCat kv_a_layernorm uses the RMSNorm default eps=1e-6.
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

            // MLA LoRA scaling: kv_cmpr *= sqrt(hidden_size / kv_lora_rank)
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

try:
    decoded = raw.decode("utf-8")
except UnicodeDecodeError as exc:
    raise SystemExit(
        "STOP: LongCat source is not UTF-8: %s"
        % exc
    )

newline = "\r\n" if "\r\n" in decoded else "\n"
text = decoded.replace("\r\n", "\n")

for marker in (MARKER_Q, MARKER_KV):
    if marker in text:
        raise SystemExit(
            "STOP: MLA epsilon marker already exists: " + marker
        )

q_count = text.count(OLD_Q)
kv_count = text.count(OLD_KV)

print("q_anchor_count=%d" % q_count)
print("kv_anchor_count=%d" % kv_count)

if q_count != 1:
    raise SystemExit(
        "STOP: exact main-trunk q-A anchor count is %d, expected 1"
        % q_count
    )

if kv_count != 1:
    raise SystemExit(
        "STOP: exact main-trunk KV-A anchor count is %d, expected 1"
        % kv_count
    )

text = text.replace(OLD_Q, NEW_Q, 1)
text = text.replace(OLD_KV, NEW_KV, 1)

if text.count(MARKER_Q) != 1:
    raise SystemExit(
        "STOP: q-A epsilon marker count is not exactly 1"
    )

if text.count(MARKER_KV) != 1:
    raise SystemExit(
        "STOP: KV-A epsilon marker count is not exactly 1"
    )

# Ensure the already-proven main attention RMSNorm diagnostic remains present.
main_norm_marker = "LONGCAT_ATTN0_HF_RMSNORM_DIAGNOSTIC"

if text.count(main_norm_marker) != 1:
    raise SystemExit(
        "STOP: proven attn0 HF RMSNorm diagnostic marker was lost or duplicated"
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
print("LONGCAT ATTN0 MLA NORM EPS DIAGNOSTIC V2: APPLIED")