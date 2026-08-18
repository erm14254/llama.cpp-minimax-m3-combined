#!/usr/bin/env python3
"""HF logical-layer-1 post_attention_layernorm capture (full-sequence).

Extends the proven block-1 stages monkey-patch (statement-for-statement
replica of LongcatFlashSparseDecoderLayer.forward with snap() inserts) to
capture the two post-attention (ffn) norm outputs:

  ffn0_norm   post_attention_layernorm[0] output  (THIS experiment's
              comparandum: the C++ ffn_norm-2 role, judged under the exact
              ffn_inp-2 predecessor reset)
  ffn1_norm   post_attention_layernorm[1] output  (recorded free-of-charge
              for the future block-3 twin; unused this round)

Fail-closed gates:
  - runtime / tokens / attn0_resid-oracle SHAs;
  - DETERMINISM: all 8 previously committed layer-1 surfaces re-emitted
    hash-identical to the committed manifest;
  - SAME-PASS IDENTITY: the tensor fed to post_attention_layernorm[0] is
    bytewise-equal to the attn0_resid snap (and that snap reproduces the
    committed 4718460b... oracle - the exact value the C++ injector lands);
  - RUNTIME EPS GATE: post_attention_layernorm[0]/[1].variance_epsilon read
    from the INSTANTIATED modules and recorded (expected 1e-5 - verified,
    not assumed; a differing value is a recorded discovery, not a capture
    failure).

Measurement-only; no model or runtime modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import types
from pathlib import Path

EXPECTED_RUNTIME_SHA256 = "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
EXPECTED_TOKEN_SHA256 = "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
EXPECTED_ATTN0_RESID_SHA256 = "4718460be4d2bb0243c4b9bcf76e20ca4b8d5a0f35ec3717ca6b8dd5cb5f73c3"
EXPECTED_TOKEN_COUNT = 512
EXPECTED_HIDDEN = 3072
EXPECTED_HIDDEN_STATES = 15
VOCAB_SIZE = 131072

PRIOR_SHA = {
    "input": "d810f93c50ea42c5909ab289ebf62a0c5629f40530d2e5fc706dde67f0eaf763",
    "attn0_norm": "afa16c6c3324387e9261c708cae044b8fcb08acda8c8f6315d2ba8d39a8f0fd7",
    "attn0_out": "c90c8e0669b9261f3bfa21abc1cc7f4f7fae48ee4393755ad99ed9e7c1a5e2e9",
    "attn0_resid": "4718460be4d2bb0243c4b9bcf76e20ca4b8d5a0f35ec3717ca6b8dd5cb5f73c3",
    "mlp0_resid": "16b9283f2cccec060e6a78004774020d730394fbfa7c314d68a9ad959ac336fc",
    "attn1_norm": "6a6280625b6cdf05f40d84b807f581043badd318140e306d3260f368a2d3ef1e",
    "attn1_resid": "40e19bfabd731936d695746876ad1101cb5ea95ef31e18aa5b169fe3d95e56e9",
    "layer_out": "85097c18565f04c0e0676146ae7ee3f5ffc674789db6f17c028606595d6d16e2",
}


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
    ap.add_argument("--attn0-resid-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    tokens_bin = Path(ns.tokens_bin).resolve()
    resid_bin = Path(ns.attn0_resid_bin).resolve()
    out_dir = Path(ns.out_dir).resolve()

    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set - refusing to run")

    for p, what in ((model_dir, "model dir"), (tokens_bin, "token file"), (resid_bin, "attn0_resid oracle")):
        if not p.exists():
            stop(f"{what} missing: {p}")

    runtime = model_dir / "modeling_longcat_flash_sparse.py"
    if not runtime.is_file():
        stop(f"runtime missing: {runtime}")

    runtime_sha = sha256_file(runtime)
    token_sha = sha256_file(tokens_bin)
    resid_sha = sha256_file(resid_bin)
    print(f"runtime_sha256={runtime_sha}")
    print(f"tokens_bin_sha256={token_sha}")
    print(f"attn0_resid_sha256={resid_sha}")
    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop("frozen runtime SHA mismatch")
    if token_sha != EXPECTED_TOKEN_SHA256:
        stop("authoritative token SHA mismatch")
    if resid_sha != EXPECTED_ATTN0_RESID_SHA256:
        stop("attn0_resid oracle SHA mismatch")

    raw = tokens_bin.read_bytes()
    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop(f"unexpected token file size: {len(raw)}")
    ids = list(struct.unpack(f"<{EXPECTED_TOKEN_COUNT}i", raw))
    for i, token_id in enumerate(ids):
        if not 0 <= token_id < VOCAB_SIZE:
            stop(f"token {i} out of range: {token_id}")

    try:
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM
    except Exception as exc:
        stop(f"import failure: {exc}")

    if not torch.cuda.is_available():
        stop("CUDA unavailable")
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")

    resid_oracle = np.frombuffer(resid_bin.read_bytes(), dtype="<f4").reshape(
        EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN
    )

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
    print(f"load_seconds={time.perf_counter() - t0:.3f}")

    trunk = model.model
    if len(trunk.layers) != 14:
        stop(f"unexpected logical layer count: {len(trunk.layers)}")
    layer1 = trunk.layers[1]

    # RUNTIME EPS GATE: read from the instantiated modules (verified, not
    # assumed). A non-1e-5 value is a recorded discovery, not a failure.
    eps0 = float(layer1.post_attention_layernorm[0].variance_epsilon)
    eps1 = float(layer1.post_attention_layernorm[1].variance_epsilon)
    cfg_eps = float(model.config.rms_norm_eps)
    print(f"runtime_eps: post_attention_layernorm[0]={eps0!r} [1]={eps1!r} config.rms_norm_eps={cfg_eps!r}")
    if eps0 != eps1:
        stop(f"post_attention_layernorm eps mismatch between sublayers: {eps0!r} vs {eps1!r}")

    captured: dict[str, torch.Tensor] = {}

    def snap(name: str, tensor: torch.Tensor) -> None:
        if name in captured:
            stop("duplicate stage capture: " + name)
        if tuple(tensor.shape) != (1, EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN):
            stop(f"{name} unexpected shape {tuple(tensor.shape)}")
        captured[name] = tensor.detach().clone()

    # Statement-for-statement replica of LongcatFlashSparseDecoderLayer.forward
    # (modeling_longcat_flash_sparse.py:980-1032) with snap() inserts only;
    # identical to the committed block-1 stages replica plus the two ffn-norm
    # snaps and the same-pass norm-input assertion.
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
        hidden_states = self.input_layernorm[0](hidden_states)
        snap("attn0_norm", hidden_states)

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
        snap("attn0_out", hidden_states)
        hidden_states = residual + hidden_states
        snap("attn0_resid", hidden_states)

        residual = hidden_states
        # SAME-PASS IDENTITY: the tensor entering post_attention_layernorm[0]
        # must be bytewise the attn0_resid snap.
        if not torch.equal(hidden_states, captured["attn0_resid"]):
            stop("norm input != attn0_resid snap (same-pass identity FAIL)")
        hidden_states = self.post_attention_layernorm[0](hidden_states)
        snap("ffn0_norm", hidden_states)
        shortcut_mlp_output = self.mlp(hidden_states)
        hidden_states = self.mlps[0](hidden_states)
        hidden_states = residual + hidden_states
        snap("mlp0_resid", hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm[1](hidden_states)
        snap("attn1_norm", hidden_states)

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
        hidden_states = self.post_attention_layernorm[1](hidden_states)
        snap("ffn1_norm", hidden_states)
        hidden_states = self.mlps[1](hidden_states)
        hidden_states = residual + hidden_states + shortcut_mlp_output
        snap("layer_out", hidden_states)
        return hidden_states

    original_forward = layer1.forward
    layer1.forward = types.MethodType(capture_forward, layer1)

    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
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
        print(f"forward_seconds={time.perf_counter() - t1:.3f}")
    finally:
        layer1.forward = original_forward

    hidden_states = out.hidden_states
    if hidden_states is None or len(hidden_states) != EXPECTED_HIDDEN_STATES:
        stop("unexpected hidden_states")

    expected_names = {
        "input", "attn0_norm", "attn0_out", "attn0_resid", "ffn0_norm",
        "mlp0_resid", "attn1_norm", "attn1_resid", "ffn1_norm", "layer_out",
    }
    if set(captured) != expected_names:
        stop(f"captured stage set mismatch: {sorted(captured)}")

    if not torch.equal(captured["input"], hidden_states[1]):
        stop("input snap != hidden_states[1]")
    if not torch.equal(captured["layer_out"], hidden_states[2]):
        stop("layer_out snap != hidden_states[2] - patched forward not faithful")
    resid_f32 = (
        captured["attn0_resid"][0].float().detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    )
    if not np.array_equal(resid_f32, resid_oracle):
        stop("attn0_resid snap != committed oracle bytes")
    print("identity gates: input==hidden_states[1], layer_out==hidden_states[2], "
          "attn0_resid==oracle, norm-input==attn0_resid (same pass) PASS")

    out_dir.mkdir(parents=True, exist_ok=True)
    order = ["input", "attn0_norm", "attn0_out", "attn0_resid", "ffn0_norm",
             "mlp0_resid", "attn1_norm", "attn1_resid", "ffn1_norm", "layer_out"]

    summary = {
        "description": "HF logical-layer-1 post_attention_layernorm capture (full-sequence)",
        "runtime_sha256": runtime_sha,
        "tokens_bin_sha256": token_sha,
        "attn0_resid_oracle_sha256": resid_sha,
        "sequence_length": EXPECTED_TOKEN_COUNT,
        "hidden_size": EXPECTED_HIDDEN,
        "layout": "token-major [512, 3072] float32-le",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tf32_disabled": True,
        "runtime_eps_gate": {
            "post_attention_layernorm_0_variance_epsilon": eps0,
            "post_attention_layernorm_1_variance_epsilon": eps1,
            "config_rms_norm_eps": cfg_eps,
            "note": "read from the instantiated modules; verified, not assumed",
        },
        "identity_gates": (
            "input==hidden_states[1]; layer_out==hidden_states[2]; "
            "attn0_resid==committed oracle; norm input==attn0_resid same-pass"
        ),
        "surfaces": {},
    }
    sums_lines = []
    for name in order:
        v = (
            captured[name][0].float().detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        )
        if not np.isfinite(v).all():
            stop(f"{name}: non-finite values")
        path = out_dir / f"{name}.bin"
        path.write_bytes(v.tobytes())
        full_sha = sha256_file(path)
        if name in PRIOR_SHA and full_sha != PRIOR_SHA[name]:
            stop(f"{name}: determinism gate FAIL: {full_sha} != committed {PRIOR_SHA[name]}")
        summary["surfaces"][name] = {
            "sha256": full_sha,
            "determinism_gate": "PASS" if name in PRIOR_SHA else "recorded-fresh",
            "min": float(v.min()),
            "max": float(v.max()),
            "rms": float(np.sqrt(np.mean(v.astype(np.float64) ** 2))),
        }
        sums_lines.append(f"{full_sha}  {name}.bin")
        print(f"{name}: sha256={full_sha} ({summary['surfaces'][name]['determinism_gate']})")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums_lines.append(f"{sha256_file(out_dir / 'summary.json')}  summary.json")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    print(f"determinism gates: 8/8 committed surfaces reproduced; out_dir={out_dir}")
    print("HF FFN-NORM CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
