#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import types
from pathlib import Path


EXPECTED_RUNTIME_SHA256 = (
    "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
)

EXPECTED_TOKEN_SHA256 = (
    "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
)

EXPECTED_INPUT_SHA256 = (
    "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f"
)

EXPECTED_LOGICAL0_SHA256 = (
    "5292e88a34a9c6625668309f6b06a352efe6b6254c383fdc32eea5a2018fa2ff"
)

EXPECTED_TOKEN_COUNT = 512
EXPECTED_HIDDEN = 3072
EXPECTED_HIDDEN_STATES = 15
VOCAB_SIZE = 131072


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-bin", required=True)
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--out-dir", required=True)

    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    tokens_bin = Path(ns.tokens_bin).resolve()
    oracle_dir = Path(ns.oracle_dir).resolve()
    out_dir = Path(ns.out_dir).resolve()

    if not model_dir.is_dir():
        stop("model directory missing: %s" % model_dir)

    if not tokens_bin.is_file():
        stop("token file missing: %s" % tokens_bin)

    if not oracle_dir.is_dir():
        stop("oracle directory missing: %s" % oracle_dir)

    runtime = model_dir / "modeling_longcat_flash_sparse.py"

    if not runtime.is_file():
        stop("runtime missing: %s" % runtime)

    runtime_sha = sha256_file(runtime)
    token_sha = sha256_file(tokens_bin)

    print("runtime_sha256=" + runtime_sha)
    print("tokens_bin_sha256=" + token_sha)

    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop("frozen runtime SHA mismatch")

    if token_sha != EXPECTED_TOKEN_SHA256:
        stop("authoritative token SHA mismatch")

    raw = tokens_bin.read_bytes()

    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop(
            "unexpected token file size: %d"
            % len(raw)
        )

    ids = list(
        struct.unpack(
            "<%di" % EXPECTED_TOKEN_COUNT,
            raw,
        )
    )

    for i, token_id in enumerate(ids):
        if not 0 <= token_id < VOCAB_SIZE:
            stop(
                "token %d out of range: %d"
                % (i, token_id)
            )

    try:
        import numpy as np
        import torch
        import transformers

        from transformers import AutoModelForCausalLM

    except Exception as exc:
        stop("import failure: %s" % exc)

    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    print("torch=" + torch.__version__)
    print("transformers=" + transformers.__version__)
    print("sequence_length=%d" % len(ids))

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )

    model.eval()

    torch.cuda.synchronize()

    print(
        "load_seconds=%.3f"
        % (time.perf_counter() - t0)
    )

    if not hasattr(model, "model"):
        stop("CausalLM has no .model trunk")

    trunk = model.model

    if not hasattr(trunk, "layers"):
        stop("model trunk has no layers")

    if len(trunk.layers) != 14:
        stop(
            "unexpected logical layer count: %d"
            % len(trunk.layers)
        )

    layer0 = trunk.layers[0]

    captured: dict[str, torch.Tensor] = {}

    def snap(name: str, tensor: torch.Tensor) -> None:
        if name in captured:
            stop("duplicate stage capture: " + name)

        if tuple(tensor.shape) != (
            1,
            EXPECTED_TOKEN_COUNT,
            EXPECTED_HIDDEN,
        ):
            stop(
                "%s unexpected shape %s"
                % (name, tuple(tensor.shape))
            )

        # Keep only the final prompt-token vector on GPU.
        captured[name] = (
            tensor[0, -1]
            .detach()
            .clone()
        )

    def capture_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        sequence_attention_mask=None,
        **kwargs,
    ):
        snap("input", hidden_states)

        residual = hidden_states

        hidden_states = self.input_layernorm[0](
            hidden_states
        )

        hidden_states, _, topk_indices = self.self_attn[0](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            sequence_attention_mask=sequence_attention_mask,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        snap("attn0_resid", hidden_states)

        residual = hidden_states

        hidden_states = self.post_attention_layernorm[0](
            hidden_states
        )

        shortcut_mlp_output = self.mlp(
            hidden_states
        )

        hidden_states = self.mlps[0](
            hidden_states
        )

        hidden_states = residual + hidden_states

        snap("mlp0_resid", hidden_states)

        residual = hidden_states

        hidden_states = self.input_layernorm[1](
            hidden_states
        )

        hidden_states, _, _ = self.self_attn[1](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            sequence_attention_mask=sequence_attention_mask,
            topk_indices=topk_indices,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        snap("attn1_resid", hidden_states)

        residual = hidden_states

        hidden_states = self.post_attention_layernorm[1](
            hidden_states
        )

        hidden_states = self.mlps[1](
            hidden_states
        )

        hidden_states = (
            residual
            + hidden_states
            + shortcut_mlp_output
        )

        snap("logical0_out", hidden_states)

        return hidden_states

    original_forward = layer0.forward

    layer0.forward = types.MethodType(
        capture_forward,
        layer0,
    )

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device="cuda:0",
    )

    try:
        t1 = time.perf_counter()

        with torch.inference_mode():
            out = trunk(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        torch.cuda.synchronize()

        print(
            "forward_seconds=%.3f"
            % (time.perf_counter() - t1)
        )

    finally:
        layer0.forward = original_forward

    hidden_states = out.hidden_states

    if hidden_states is None:
        stop("HF did not return hidden_states")

    if len(hidden_states) != EXPECTED_HIDDEN_STATES:
        stop(
            "unexpected hidden-state count: %d != %d"
            % (
                len(hidden_states),
                EXPECTED_HIDDEN_STATES,
            )
        )

    expected_capture_names = {
        "input",
        "attn0_resid",
        "mlp0_resid",
        "attn1_resid",
        "logical0_out",
    }

    if set(captured) != expected_capture_names:
        stop(
            "captured stage set mismatch: %s"
            % sorted(captured)
        )

    def to_f32_bytes(
        tensor: torch.Tensor,
    ) -> tuple[bytes, "np.ndarray"]:
        v = (
            tensor
            .float()
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )

        if v.shape != (EXPECTED_HIDDEN,):
            stop(
                "unexpected captured vector shape: %s"
                % (v.shape,)
            )

        if not np.isfinite(v).all():
            stop("captured vector contains nonfinite values")

        return v.tobytes(), v

    canonical_input_bytes, _ = to_f32_bytes(
        hidden_states[0][0, -1]
    )

    canonical_logical0_bytes, _ = to_f32_bytes(
        hidden_states[1][0, -1]
    )

    captured_input_bytes, _ = to_f32_bytes(
        captured["input"]
    )

    captured_logical0_bytes, _ = to_f32_bytes(
        captured["logical0_out"]
    )

    if canonical_input_bytes != captured_input_bytes:
        stop(
            "wrapper input differs from canonical hidden_states[0]"
        )

    if canonical_logical0_bytes != captured_logical0_bytes:
        stop(
            "wrapper logical0 output differs from canonical hidden_states[1]"
        )

    oracle_input = (
        oracle_dir
        / "inp_embd_ngram.bin"
    )

    oracle_logical0 = (
        oracle_dir
        / "logical_00.bin"
    )

    if not oracle_input.is_file():
        stop("input oracle missing")

    if not oracle_logical0.is_file():
        stop("logical_00 oracle missing")

    oracle_input_sha = sha256_file(
        oracle_input
    )

    oracle_logical0_sha = sha256_file(
        oracle_logical0
    )

    print(
        "oracle_input_sha256="
        + oracle_input_sha
    )

    print(
        "oracle_logical0_sha256="
        + oracle_logical0_sha
    )

    if oracle_input_sha != EXPECTED_INPUT_SHA256:
        stop("frozen HF input oracle changed")

    if oracle_logical0_sha != EXPECTED_LOGICAL0_SHA256:
        stop("frozen HF logical_00 oracle changed")

    if sha256_bytes(captured_input_bytes) != EXPECTED_INPUT_SHA256:
        stop(
            "derivative capture input is not byte-exact to frozen oracle"
        )

    if sha256_bytes(captured_logical0_bytes) != EXPECTED_LOGICAL0_SHA256:
        stop(
            "derivative capture logical0 output is not byte-exact to frozen oracle"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered = [
        "input",
        "attn0_resid",
        "mlp0_resid",
        "attn1_resid",
        "logical0_out",
    ]

    summary = {
        "runtime_sha256": runtime_sha,
        "tokens_bin_sha256": token_sha,
        "input_oracle_sha256": oracle_input_sha,
        "logical0_oracle_sha256": oracle_logical0_sha,
        "surfaces": {},
    }

    print()
    print("===== LOGICAL-0 HF STAGES =====")

    for name in ordered:
        data, values = to_f32_bytes(
            captured[name]
        )

        path = out_dir / (name + ".bin")

        path.write_bytes(data)

        digest = sha256_bytes(data)

        summary["surfaces"][name] = {
            "sha256": digest,
            "bytes": len(data),
            "min": float(values.min()),
            "max": float(values.max()),
        }

        print(
            "%-14s bytes=%d sha256=%s min=%g max=%g"
            % (
                name,
                len(data),
                digest,
                float(values.min()),
                float(values.max()),
            )
        )

    summary_path = out_dir / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "input self-check = BYTE-EXACT"
    )

    print(
        "logical0 self-check = BYTE-EXACT"
    )

    print(
        "HF LOGICAL-0 STAGE CAPTURE: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())