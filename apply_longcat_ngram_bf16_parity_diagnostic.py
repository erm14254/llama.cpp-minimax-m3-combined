#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


MARKER = "LONGCAT_NGRAM_BF16_PARITY_DIAGNOSTIC"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ns = ap.parse_args()

    repo = Path(ns.repo)
    target = repo / "src" / "models" / "longcat-flash-ngram.cpp"

    if not target.is_file():
        stop(f"target missing: {target}")

    raw = target.read_bytes()
    before_sha = sha256_bytes(raw)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        stop(f"target is not UTF-8: {exc}")

    newline = "\r\n" if "\r\n" in text else "\n"
    text = text.replace("\r\n", "\n")

    print(f"target={target}")
    print(f"before_sha256={before_sha}")

    if MARKER in text:
        print("LONGCAT NGRAM BF16 PARITY DIAGNOSTIC: ALREADY APPLIED")
        return 0

    old1 = """        // For each embedder: lookup embedding table, project to hidden_size, accumulate
        for (uint32_t j = 0; j < n_ngram && j < (uint32_t)llama_model::NGRAM_MAX; j++) {
"""

    new1 = """        // LONGCAT_NGRAM_BF16_PARITY_DIAGNOSTIC:
        // HF keeps the running N-gram embedding state on the BF16 lattice.
        // Round the base embedding before beginning the sequential accumulation.
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);

        // For each embedder: lookup embedding table, project to hidden_size, accumulate
        for (uint32_t j = 0; j < n_ngram && j < (uint32_t)llama_model::NGRAM_MAX; j++) {
"""

    old2 = """            ggml_tensor * proj = ggml_mul_mat(ctx0, model.ngram_proj[j], emb);
            cb(proj, "ngram_proj", j);

            inpL = ggml_add(ctx0, inpL, proj);
"""

    new2 = """            ggml_tensor * proj = ggml_mul_mat(ctx0, model.ngram_proj[j], emb);
            cb(proj, "ngram_proj", j);

            // ggml_mul_mat returns F32. HF parity only requires that each
            // projection be rounded to BF16 before the sequential BF16 add.
            proj = ggml_cast(ctx0, proj, GGML_TYPE_BF16);

            // Perform the arithmetic in F32 using already-BF16-rounded operands,
            // then round the running accumulator back to BF16 after every add.
            // This avoids depending on native BF16 ADD kernel semantics while
            // reproducing the validated standalone HF arithmetic.
            ggml_tensor * inpL_f32 = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
            ggml_tensor * proj_f32 = ggml_cast(ctx0, proj, GGML_TYPE_F32);

            inpL = ggml_add(ctx0, inpL_f32, proj_f32);
            inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);
"""

    old3 = """        // Normalize: x = (base + sum_of_projections) / (1 + n_ngram)
        inpL = ggml_scale(ctx0, inpL, 1.0f / (1.0f + (float)n_ngram));
        cb(inpL, "inp_embd_ngram", -1);
"""

    new3 = """        // Normalize: x = (base + sum_of_projections) / (1 + n_ngram)
        // Compute from the BF16-rounded accumulator and round the result back
        // to BF16, matching frozen HF NgramEmbedding.forward().
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
        inpL = ggml_scale(ctx0, inpL, 1.0f / (1.0f + (float)n_ngram));
        inpL = ggml_cast(ctx0, inpL, GGML_TYPE_BF16);
        cb(inpL, "inp_embd_ngram", -1);
"""

    anchors = [
        ("base BF16 anchor", old1, new1),
        ("projection/add anchor", old2, new2),
        ("normalization anchor", old3, new3),
    ]

    for label, old, _new in anchors:
        count = text.count(old)
        if count != 1:
            stop(f"{label} count is {count}, expected exactly 1")

    for _label, old, new in anchors:
        text = text.replace(old, new, 1)

    if text.count(MARKER) != 1:
        stop("diagnostic marker count is not exactly 1 after patch")

    output = text.replace("\n", newline).encode("utf-8")
    target.write_bytes(output)

    after_sha = sha256_bytes(output)

    print(f"after_sha256={after_sha}")
    print("LONGCAT NGRAM BF16 PARITY DIAGNOSTIC: APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())