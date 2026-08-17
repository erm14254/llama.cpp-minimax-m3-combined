#!/usr/bin/env python3

"""Generate the canonical block-0 RoPE targets for experiments R0/R1.

Ground truth is produced by executing the ACTUAL installed
transformers.models.longcat_flash.modeling_longcat_flash
.apply_rotary_pos_emb_interleave on the captured Blackwell oracles, in BF16 on
CUDA -- HF execution, not a model of it -- then canonicalized through the
proven permutation P into llama.cpp's in-place interleaved layout
(CPP[2j] = HF[j], CPP[2j+1] = HF[32+j]).

The outputs are byte-frozen: this script STOPS unless they match the
pre-registered SHA256 values recorded at design time. Regenerating on the same
environment must be a no-op.

  q_pe_rope_target.bin  [512, 2048] F32  c8b9b6bfd8759f839c333e2b74f3775f...
  k_pe_rope_target.bin  [512,   64] F32  3ed6f4e731227d49952fc687aefb2ede...

Inputs (SHA-verified): rope_cos / rope_sin / q_b_proj / kv_a_proj_with_mqa
from the authoritative HF attention capture directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_INPUTS = {
    "rope_cos.bin": "8771da1ea77d102e07bbc08064e6da6226ab4c2cb2a195c25d197c35487d9bb2",
    "rope_sin.bin": "5c5dede92d05f23dfab1a27285685f61e5596df888dd429fae6d8b6591b7ff0a",
    "q_b_proj.bin": "4f3b647b62c60475fc03f023ce46a5c01951c45847ced2557b5692b2ed3e79b1",
    "kv_a_proj_with_mqa.bin": "513390418c9877fa46286d397db7c9c9fb6408852836fb7827106acd183ceecc",
}

EXPECTED_TARGETS = {
    "q_pe_rope_target.bin": (
        "c8b9b6bfd8759f839c333e2b74f3775fe0b89bf82dc296497ee17990669dfc95",
        (512, 2048),
    ),
    "k_pe_rope_target.bin": (
        "3ed6f4e731227d49952fc687aefb2ede9067eceec7eed39096d861634158bc1d",
        (512, 64),
    ),
}

MLA_SCALE_Q = 1.4142135623730951  # sqrt(3072/1536), summary.json mla_scale_q_lora


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(root: Path, name: str, rows: int, width: int) -> np.ndarray:
    path = root / name
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != EXPECTED_INPUTS[name]:
        stop("input SHA mismatch for %s: %s" % (name, got))
    values = np.frombuffer(raw, dtype="<f4").reshape(rows, width)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args()

    hf_dir = Path(ns.hf_dir).resolve()
    out_dir = Path(ns.out_dir).resolve()

    import torch

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    from transformers.models.longcat_flash.modeling_longcat_flash import (
        apply_rotary_pos_emb_interleave,
    )

    dev = torch.device("cuda:0")

    cos = load(hf_dir, "rope_cos.bin", 512, 64)
    sin = load(hf_dir, "rope_sin.bin", 512, 64)
    q_b = load(hf_dir, "q_b_proj.bin", 512, 6144).reshape(512, 32, 192)
    kv = load(hf_dir, "kv_a_proj_with_mqa.bin", 512, 576)

    def t(x: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

    # Inputs exactly as the HF forward holds them (modeling:417-433):
    #   q_rot = bf16(q_b_proj[..., 128:] * mla_scale_q_lora)   [1, 32, 512, 64]
    #   k_rot = bf16 kv_a_proj[..., 512:], unscaled            [1,  1, 512, 64]
    q_rot = (t(q_b[:, :, 128:]).to(torch.bfloat16) * MLA_SCALE_Q)
    q_hf = q_rot.permute(1, 0, 2).unsqueeze(0).contiguous()
    k_hf = t(kv[:, 512:]).to(torch.bfloat16).reshape(1, 1, 512, 64).contiguous()
    cos_hf = t(cos).to(torch.bfloat16).unsqueeze(0)
    sin_hf = t(sin).to(torch.bfloat16).unsqueeze(0)

    q_gt, k_gt = apply_rotary_pos_emb_interleave(q_hf, k_hf, cos_hf, sin_hf)

    gt_q = q_gt[0].permute(1, 0, 2).float().cpu().numpy()  # [512, 32, 64] HF layout
    gt_k = k_gt[0, 0].float().cpu().numpy()                # [512, 64] HF layout

    def unpermute_P(hf_arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(hf_arr)
        out[..., 0::2] = hf_arr[..., :32]
        out[..., 1::2] = hf_arr[..., 32:]
        return out

    targets = {
        "q_pe_rope_target.bin": unpermute_P(gt_q).reshape(512, 2048).astype("<f4"),
        "k_pe_rope_target.bin": unpermute_P(gt_k).astype("<f4"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name, values in targets.items():
        expect_sha, expect_shape = EXPECTED_TARGETS[name]
        if values.shape != expect_shape:
            stop("%s shape %s != %s" % (name, values.shape, expect_shape))
        data = values.tobytes()
        got = sha256_bytes(data)
        if got != expect_sha:
            stop(
                "%s does not match the pre-registered canonical SHA\n"
                "  expected %s\n  got      %s" % (name, expect_sha, got)
            )
        (out_dir / name).write_bytes(data)
        meta = {
            "name": name,
            "shape": list(expect_shape),
            "order": "token-major",
            "layout": "cpp-interleaved (permutation P applied)",
            "dtype": "float32-le",
            "bytes": len(data),
            "sha256": got,
        }
        (out_dir / (name.replace(".bin", ".json"))).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest[name] = meta
        print("%-24s %s  PASS (pre-registered)" % (name, got))

    (out_dir / "targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("CANONICAL ROPE TARGETS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
