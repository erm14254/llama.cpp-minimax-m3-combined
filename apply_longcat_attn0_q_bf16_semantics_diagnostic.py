#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    r"D:\llama.cpp-longcat-pre-gate4\src\models\longcat-flash-ngram.cpp"
)

EXPECTED_BEFORE = (
    "9899ba54ce377f4ef3ec7e21eb6e1f4a4c7ff765bab66c54509e7c23b1609188"
)

MARKER = "LONGCAT_ATTN0_Q_BF16_SEMANTICS_DIAGNOSTIC"

OLD = """            if (model.layers[il].wq_a) {
                // LoRA Q path
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
                cb(q, "q", il);

                // MLA LoRA scaling: q *= sqrt(hidden_size / q_lora_rank)
                q = ggml_scale(ctx0, q, mla_scale_q);
                cb(q, "q_scaled", il);
            } else {
"""

NEW = """            if (model.layers[il].wq_a) {
                // LoRA Q path
                if (il == 0) {
                    // LONGCAT_ATTN0_Q_BF16_SEMANTICS_DIAGNOSTIC:
                    //
                    // HF block-0 Q path is BF16 at the Linear/RMSNorm
                    // boundaries. Widen only where ggml RMSNorm / downstream
                    // graph operations require F32.
                    ggml_tensor * q_in_bf16 =
                        ggml_cast(ctx0, cur, GGML_TYPE_BF16);

                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_a,
                        q_in_bf16);

                    // q_a_proj output is BF16 in the HF BF16 model.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    cb(q, "q", il);

                    // LONGCAT_ATTN0_QA_NORM_EPS_DIAGNOSTIC:
                    // HF q_a_layernorm uses eps=1e-6 and computes the RMS
                    // reduction in F32.
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    q = ggml_rms_norm(ctx0, q, 1.0e-6f);

                    // LongcatFlashRMSNorm casts the normalized activation
                    // back to BF16 before multiplying by its BF16 weight.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);

                    q = ggml_mul(
                        ctx0,
                        q,
                        model.layers[il].attn_q_a_norm);

                    // RMSNorm output is BF16.
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    cb(q, "q", il);

                    // q_b_proj consumes BF16 and produces BF16 in HF.
                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_b,
                        q);

                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    cb(q, "q", il);

                    // HF scales the BF16 q_pass/q_rot tensors and therefore
                    // returns to the BF16 lattice here as well. Widen the
                    // rounded result for the existing llama.cpp RoPE path.
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    q = ggml_scale(ctx0, q, mla_scale_q);
                    q = ggml_cast(ctx0, q, GGML_TYPE_BF16);
                    q = ggml_cast(ctx0, q, GGML_TYPE_F32);
                    cb(q, "q_scaled", il);
                } else {
                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_a,
                        cur);
                    cb(q, "q", il);

                    q = build_norm(
                        q,
                        model.layers[il].attn_q_a_norm,
                        nullptr,
                        LLM_NORM_RMS,
                        il);
                    cb(q, "q", il);

                    q = ggml_mul_mat(
                        ctx0,
                        model.layers[il].wq_b,
                        q);
                    cb(q, "q", il);

                    // MLA LoRA scaling: q *= sqrt(hidden_size / q_lora_rank)
                    q = ggml_scale(ctx0, q, mla_scale_q);
                    cb(q, "q_scaled", il);
                }
            } else {
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


raw = TARGET.read_bytes()
before = sha256(raw)

print("before_sha256=" + before)

if before != EXPECTED_BEFORE:
    raise SystemExit(
        "STOP: LongCat source is not at the expected MLA-eps diagnostic state"
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
        "STOP: Q BF16 diagnostic marker already exists"
    )

count = text.count(OLD)

print("q_anchor_count=%d" % count)

if count != 1:
    raise SystemExit(
        "STOP: exact block-0 Q-path anchor count is %d, expected 1"
        % count
    )

text = text.replace(OLD, NEW, 1)

required_markers = (
    "LONGCAT_ATTN0_HF_RMSNORM_DIAGNOSTIC",
    "LONGCAT_ATTN0_QA_NORM_EPS_DIAGNOSTIC",
    "LONGCAT_ATTN0_KVA_NORM_EPS_DIAGNOSTIC",
    MARKER,
)

for marker in required_markers:
    count = text.count(marker)
    print("%s_count=%d" % (marker, count))

    if count != 1:
        raise SystemExit(
            "STOP: marker %s count is %d, expected 1"
            % (marker, count)
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
print("LONGCAT ATTN0 Q BF16 SEMANTICS DIAGNOSTIC: APPLIED")