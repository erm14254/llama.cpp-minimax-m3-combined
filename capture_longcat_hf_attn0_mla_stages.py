#!/usr/bin/env python3

"""Capture HF full-sequence 512-token physical block-0 MLA stage surfaces.

Standalone block-0 harness: the full 149.76 GiB runtime is not loaded. Only the
weights physical attention block 0 actually needs are materialized, and the
58.5 GiB of N-gram tables are read by row slice (6144 rows total).

Three hard gates prove the harness is numerically identical to the full-model
run that produced the frozen oracles. They are never widened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path


EXPECTED_RUNTIME_SHA256 = (
    "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
)

EXPECTED_TOKEN_SHA256 = (
    "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
)

# Frozen gates -- never widen, never regenerate.
EXPECTED_INPUT_SHA256 = (
    "d0e9edc8503e388d62b27f39c6ad24560021b1f4533410bf4f86a28fef4ea45f"
)

EXPECTED_ATTN_NORM_SHA256 = (
    "a1c4c20cf346cbc5d13a0ea3d1d142f1b54f8e024ff3f04fdb31e9120e75b0af"
)

EXPECTED_ATTN0_RESID_SHA256 = (
    "2e9b65ebbdf015899af9f18add5587dbc78735c121ac009fccbd0a0fb4cbd177"
)

EXPECTED_TOKEN_COUNT = 512
EXPECTED_HIDDEN = 3072
VOCAB_SIZE = 131072

ATTN_NORM_KEY = "model.layers.0.input_layernorm.0.weight"
ATTN_PREFIX = "model.layers.0.self_attn.0."

# Module attribute -> expected output width, in pipeline order.
SURFACES = [
    ("q_a_proj", 1536),
    ("q_a_layernorm", 1536),
    ("q_b_proj", 6144),
    ("kv_a_proj_with_mqa", 576),
    ("kv_a_layernorm", 512),
    ("o_proj", 3072),
]

# Attention-path localization extras (2026-08-17): non-module-output surfaces.
#   rope_cos / rope_sin -- the rotary embedding actually consumed by the MLA
#       (BF16 per modeling_longcat_flash.py:124, yarn attention_scaling baked
#       in); captured from the position_embeddings tuple itself.
#   attn_o_input -- the o_proj INPUT (the 4096-wide attention context), taken
#       from the o_proj forward hook's `args`, i.e. a module boundary.
EXTRA_SURFACES = [
    ("rope_cos", 64),
    ("rope_sin", 64),
    ("attn_o_input", 4096),
]


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


def load_tokens(tokens_bin: Path) -> list[int]:
    raw = tokens_bin.read_bytes()

    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop("unexpected token file size: %d" % len(raw))

    ids = list(struct.unpack("<%di" % EXPECTED_TOKEN_COUNT, raw))

    for i, token_id in enumerate(ids):
        if not 0 <= token_id < VOCAB_SIZE:
            stop("token %d out of range: %d" % (i, token_id))

    return ids


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scratch-dir", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument(
        "--oracle-resid",
        default=None,
        help="frozen attn0_resid.bin; on gate-3 failure report metrics",
    )

    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    tokens_bin = Path(ns.tokens_bin).resolve()
    out_dir = Path(ns.out_dir).resolve()
    scratch_dir = Path(ns.scratch_dir).resolve()

    if not model_dir.is_dir():
        stop("model directory missing: %s" % model_dir)

    if not tokens_bin.is_file():
        stop("token file missing: %s" % tokens_bin)

    runtime = model_dir / "modeling_longcat_flash_sparse.py"

    if not runtime.is_file():
        stop("frozen runtime missing: %s" % runtime)

    runtime_sha = sha256_file(runtime)
    token_sha = sha256_file(tokens_bin)

    print("runtime_sha256   =", runtime_sha)
    print("tokens_bin_sha256=", token_sha)

    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop("frozen runtime SHA mismatch")

    if token_sha != EXPECTED_TOKEN_SHA256:
        stop("authoritative token SHA mismatch")

    ids = load_tokens(tokens_bin)

    try:
        import numpy as np
        import torch
        import transformers

        from torch import nn
        from safetensors import safe_open

        from transformers.masking_utils import create_causal_mask
        from transformers.models.longcat_flash.modeling_longcat_flash import (
            LongcatFlashRMSNorm,
            LongcatFlashRotaryEmbedding,
        )

    except Exception as exc:
        stop("import failure: %s" % exc)

    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    print("torch            =", torch.__version__)
    print("transformers     =", transformers.__version__)
    print("attn_impl        =", ns.attn_impl)
    print("sequence_length  = %d" % len(ids))

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # The model directory name is long enough that transformers' dynamic-module
    # cache path exceeds Windows MAX_PATH. Import the frozen runtime from a
    # short-path package copy instead; the copies are SHA-verified below, and
    # the canonical model directory is never written to.
    runtime_pkg = scratch_dir / "lcsparse"
    runtime_pkg.mkdir(parents=True, exist_ok=True)

    (runtime_pkg / "__init__.py").write_text("", encoding="utf-8")

    for fname, expect in (
        ("configuration_longcat_flash_sparse.py", None),
        ("modeling_longcat_flash_sparse.py", EXPECTED_RUNTIME_SHA256),
    ):
        src = model_dir / fname

        if not src.is_file():
            stop("frozen runtime component missing: %s" % src)

        dst = runtime_pkg / fname
        dst.write_bytes(src.read_bytes())

        if expect is not None and sha256_file(dst) != expect:
            stop("frozen runtime copy SHA mismatch: %s" % fname)

    sys.path.insert(0, str(scratch_dir))

    try:
        import importlib

        cfg_mod = importlib.import_module(
            "lcsparse.configuration_longcat_flash_sparse"
        )
        run_mod = importlib.import_module(
            "lcsparse.modeling_longcat_flash_sparse"
        )

    except Exception as exc:
        stop("frozen runtime import failure: %s" % exc)

    config = cfg_mod.LongcatFlashSparseConfig.from_pretrained(str(model_dir))

    config._attn_implementation = ns.attn_impl

    # LongcatFlashSparseModel.__init__ performs this mutation; replicate it so
    # rope/mask construction sees the same config the full model would.
    config.num_hidden_layers = 2 * config.num_layers

    if int(config.hidden_size) != EXPECTED_HIDDEN:
        stop("unexpected hidden size: %d" % config.hidden_size)

    index_path = model_dir / "model.safetensors.index.json"

    if not index_path.is_file():
        stop("Safetensors index missing")

    weight_map = json.loads(
        index_path.read_text(encoding="utf-8")
    ).get("weight_map", {})

    if not weight_map:
        stop("empty Safetensors weight map")

    def fetch(name: str) -> "torch.Tensor":
        shard = weight_map.get(name)

        if shard is None:
            stop("checkpoint tensor absent from index: " + name)

        shard_path = model_dir / shard

        if not shard_path.is_file():
            stop("shard missing: %s" % shard_path)

        with safe_open(str(shard_path), framework="pt", device="cpu") as h:
            if name not in h.keys():
                stop("indexed tensor absent from shard: " + name)

            return h.get_tensor(name).contiguous()

    class SlicedEmbedding(nn.Module):
        """Row-sliced stand-in for a 4.9 GiB N-gram nn.Embedding table.

        A gather is a pure memory read, so restricting it to the rows the
        sequence actually touches is numerically exact. Gate 1 proves it.
        """

        def __init__(self, key: str, num_embeddings: int, dim: int):
            super().__init__()
            self.key = key
            self.num_embeddings = num_embeddings
            self.dim = dim
            self.rows_read = 0

            # NgramEmbedding.forward reads `.weight.device` to place the ids.
            # An empty placeholder carries the device/dtype without holding
            # the 4.9 GiB table.
            self.register_buffer(
                "weight",
                torch.empty(0, dim, device=device, dtype=dtype),
                persistent=False,
            )

        def forward(self, ids: "torch.Tensor") -> "torch.Tensor":
            flat = ids.reshape(-1)

            if int(flat.min()) < 0 or int(flat.max()) >= self.num_embeddings:
                stop("n-gram id out of range for " + self.key)

            uniq, inverse = torch.unique(flat, return_inverse=True)
            wanted = uniq.tolist()

            shard = weight_map.get(self.key)

            if shard is None:
                stop("n-gram table absent from index: " + self.key)

            with safe_open(
                str(model_dir / shard), framework="pt", device="cpu"
            ) as h:
                sl = h.get_slice(self.key)
                shape = sl.get_shape()

                if list(shape) != [self.num_embeddings, self.dim]:
                    stop(
                        "%s unexpected table shape %s"
                        % (self.key, shape)
                    )

                compact = torch.empty(
                    (len(wanted), self.dim), dtype=dtype
                )

                for pos, row in enumerate(wanted):
                    compact[pos] = sl[row : row + 1].to(dtype)[0]

            self.rows_read += len(wanted)

            gathered = compact.to(device=device)[inverse]

            return gathered.view(*ids.shape, self.dim)

    print()
    print("loading block-0 weights ...")
    t_load = time.perf_counter()

    ngram_cls = run_mod.NgramEmbedding
    mla_cls = run_mod.LongcatFlashSparseMLA

    # embed_tokens is real (805 MB BF16); the 12 N-gram tables would be
    # 58.5 GiB, so NgramEmbedding is constructed on meta and its embedders are
    # replaced with row-sliced readers.
    embed_tokens = nn.Embedding(
        int(config.vocab_size),
        EXPECTED_HIDDEN,
        config.pad_token_id,
    )

    embed_tokens.weight.data = (
        fetch("model.embed_tokens.weight").to(device=device, dtype=dtype)
    )

    with torch.device("meta"):
        ngram = ngram_cls(config, embed_tokens)

    ngram.word_embeddings = embed_tokens

    num_embedders = int(config.emb_split_num) * (
        int(config.emb_neighbor_num) - 1
    )

    if num_embedders != 12:
        stop("unexpected N-gram embedder count: %d" % num_embedders)

    for i in range(num_embedders):
        table_key = "model.oe_embed_tokens%d.weight" % i
        proj_key = "model.oe_embed_proj%d.weight" % i

        expected_vocab = int(ngram.m + i * 2 + 1)

        ngram.embedders[i] = SlicedEmbedding(
            table_key, expected_vocab, EXPECTED_HIDDEN // num_embedders
        )

        proj = nn.Linear(
            EXPECTED_HIDDEN // num_embedders, EXPECTED_HIDDEN, bias=False
        )
        proj.weight.data = fetch(proj_key).to(device=device, dtype=dtype)
        ngram.post_projs[i] = proj

    ngram.eval()

    attn_norm = LongcatFlashRMSNorm(
        EXPECTED_HIDDEN, eps=float(config.rms_norm_eps)
    ).to(device=device, dtype=dtype)

    attn_norm.weight.data = (
        fetch(ATTN_NORM_KEY).to(device=device, dtype=dtype)
    )
    attn_norm.eval()

    mla = mla_cls(config, 0, indexer_owner=True)

    mla_state = {}
    loaded_keys = []

    for full_key, shard in weight_map.items():
        if not full_key.startswith(ATTN_PREFIX):
            continue

        local = full_key[len(ATTN_PREFIX) :]
        mla_state[local] = fetch(full_key)
        loaded_keys.append(full_key)

    if not mla_state:
        stop("no block-0 self_attn tensors found")

    missing, unexpected = mla.load_state_dict(mla_state, strict=False)

    real_missing = [k for k in missing if not k.endswith("_buffer")]

    if real_missing:
        stop("MLA missing weights: %s" % sorted(real_missing))

    if unexpected:
        stop("MLA unexpected weights: %s" % sorted(unexpected))

    mla = mla.to(device=device, dtype=dtype)
    mla.eval()

    rotary = LongcatFlashRotaryEmbedding(config=config).to(device=device)
    rotary.eval()

    print(
        "block-0 tensors loaded = %d (%.3fs)"
        % (len(loaded_keys), time.perf_counter() - t_load)
    )

    # ---------------------------------------------------------------- inputs
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    def row_bytes(tensor: "torch.Tensor", index: int) -> bytes:
        v = (
            tensor[0, index]
            .float()
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )

        if not np.isfinite(v).all():
            stop("nonfinite values in captured row %d" % index)

        return v.tobytes()

    with torch.inference_mode():
        hidden = ngram(input_ids, ngram_context=None)

    if tuple(hidden.shape) != (1, EXPECTED_TOKEN_COUNT, EXPECTED_HIDDEN):
        stop("unexpected N-gram output shape %s" % (tuple(hidden.shape),))

    total_rows = sum(m.rows_read for m in ngram.embedders)
    print("n-gram rows read       =", total_rows)

    input_final = row_bytes(hidden, -1)
    input_sha = sha256_bytes(input_final)

    print()
    print("GATE 1 input row 511   =", input_sha)

    if input_sha != EXPECTED_INPUT_SHA256:
        stop(
            "GATE 1 FAILED: reconstructed block-0 input is not byte-exact "
            "to the frozen HF oracle"
        )

    print("GATE 1 PASS")

    with torch.inference_mode():
        normed = attn_norm(hidden)

    norm_final = row_bytes(normed, -1)
    norm_sha = sha256_bytes(norm_final)

    print("GATE 2 attn_norm r511  =", norm_sha)

    if norm_sha != EXPECTED_ATTN_NORM_SHA256:
        stop(
            "GATE 2 FAILED: block-0 input RMSNorm is not byte-exact to the "
            "frozen HF oracle"
        )

    print("GATE 2 PASS")

    # ---------------------------------------------------------------- capture
    captured: dict[str, "torch.Tensor"] = {}

    def make_hook(name: str, width: int):
        def hook(module, args, output):
            if name in captured:
                stop("surface fired twice: " + name)

            tensor = output[0] if isinstance(output, tuple) else output

            if tuple(tensor.shape) != (
                1,
                EXPECTED_TOKEN_COUNT,
                width,
            ):
                stop(
                    "%s unexpected shape %s (expected [1, %d, %d])"
                    % (name, tuple(tensor.shape), EXPECTED_TOKEN_COUNT, width)
                )

            captured[name] = tensor.detach().clone()

        return hook

    handles = []

    for name, width in SURFACES:
        module = getattr(mla, name, None)

        if module is None:
            stop("MLA has no submodule " + name)

        handles.append(module.register_forward_hook(make_hook(name, width)))

    # attn_o_input: the o_proj forward hook's `args[0]` is the module INPUT --
    # the 4096-wide attention context entering the output projection.
    def o_input_hook(module, args, output):
        if "attn_o_input" in captured:
            stop("surface fired twice: attn_o_input")

        tensor = args[0]

        if tuple(tensor.shape) != (1, EXPECTED_TOKEN_COUNT, 4096):
            stop(
                "attn_o_input unexpected shape %s (expected [1, %d, 4096])"
                % (tuple(tensor.shape), EXPECTED_TOKEN_COUNT)
            )

        captured["attn_o_input"] = tensor.detach().clone()

    handles.append(mla.o_proj.register_forward_hook(o_input_hook))

    cache_position = torch.arange(EXPECTED_TOKEN_COUNT, device=device)
    position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    with torch.inference_mode():
        position_embeddings = rotary(hidden, position_ids)

        # rope_cos / rope_sin: capture the exact tensors handed to the MLA.
        rope_cos, rope_sin = position_embeddings

        for rope_name, rope_tensor in (
            ("rope_cos", rope_cos),
            ("rope_sin", rope_sin),
        ):
            if tuple(rope_tensor.shape) != (1, EXPECTED_TOKEN_COUNT, 64):
                stop(
                    "%s unexpected shape %s (expected [1, %d, 64])"
                    % (rope_name, tuple(rope_tensor.shape), EXPECTED_TOKEN_COUNT)
                )

            captured[rope_name] = rope_tensor.detach().clone()

        t_fwd = time.perf_counter()

        out, _, topk_indices = mla(
            hidden_states=normed,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_values=None,
            sequence_attention_mask=None,
        )

        torch.cuda.synchronize()

    for handle in handles:
        handle.remove()

    print("forward_seconds        = %.3f" % (time.perf_counter() - t_fwd))
    print("lsa_mode               =", mla.last_lsa_mode)

    if mla.last_lsa_mode != "full-owner":
        stop(
            "expected the exact full-attention path at 512 tokens, got %s"
            % mla.last_lsa_mode
        )

    if topk_indices is not None:
        stop("full-owner path unexpectedly returned top-K indices")

    missing_surfaces = [
        n for n, _ in SURFACES + EXTRA_SURFACES if n not in captured
    ]

    if missing_surfaces:
        stop("surfaces never fired: %s" % missing_surfaces)

    with torch.inference_mode():
        resid = hidden + out

    resid_final = row_bytes(resid, -1)
    resid_sha = sha256_bytes(resid_final)

    print()
    print("GATE 3 attn0_resid r511=", resid_sha)

    if resid_sha != EXPECTED_ATTN0_RESID_SHA256:
        if ns.oracle_resid:
            oracle_path = Path(ns.oracle_resid).resolve()

            if not oracle_path.is_file():
                stop("oracle residual missing: %s" % oracle_path)

            if sha256_file(oracle_path) != EXPECTED_ATTN0_RESID_SHA256:
                stop("supplied oracle residual is not the frozen vector")

            ref = np.fromfile(oracle_path, dtype="<f4")
            got = np.frombuffer(resid_final, dtype="<f4")

            diff = got.astype(np.float64) - ref.astype(np.float64)
            rms = float(np.sqrt((diff ** 2).mean()))
            ref_rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))

            cos = float(
                (got.astype(np.float64) @ ref.astype(np.float64))
                / (np.linalg.norm(got) * np.linalg.norm(ref))
            )

            print()
            print("--- gate 3 deviation vs frozen oracle ---")
            print("max_abs   = %.9g" % float(np.abs(diff).max()))
            print("RMSE      = %.9g" % rms)
            print("rel_RMSE  = %.9g" % (rms / ref_rms))
            print("cosine    = %.12f" % cos)
            print("exact_elems = %d / %d" % (int((diff == 0).sum()), ref.size))

        stop(
            "GATE 3 FAILED: block-0 attention residual is not byte-exact to "
            "the frozen HF oracle -- do not proceed on this environment"
        )

    print("GATE 3 PASS")

    # ----------------------------------------------------------------- output
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "runtime_sha256": runtime_sha,
        "tokens_bin_sha256": token_sha,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "attn_implementation": ns.attn_impl,
        "lsa_mode": mla.last_lsa_mode,
        # sdpa softmax scaling actually used (qk_head_dim**-0.5 adjusted by
        # yarn_apply_mscale, modeling_longcat_flash.py:380).
        "attn_scaling": float(mla.scaling),
        "mla_scale_q_lora": float(mla.mla_scale_q_lora),
        "mla_scale_kv_lora": float(mla.mla_scale_kv_lora),
        "gates": {
            "input": input_sha,
            "attn_norm": norm_sha,
            "attn0_resid": resid_sha,
        },
        "surfaces": {},
    }

    print()
    print("===== HF BLOCK-0 MLA STAGES (full sequence) =====")

    for name, width in SURFACES + EXTRA_SURFACES:
        tensor = captured[name]

        values = (
            tensor[0]
            .float()
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )

        if values.shape != (EXPECTED_TOKEN_COUNT, width):
            stop("%s canonical shape wrong: %s" % (name, values.shape))

        if not np.isfinite(values).all():
            stop("%s contains nonfinite values" % name)

        data = values.tobytes()
        digest = sha256_bytes(data)
        final_digest = sha256_bytes(values[-1].tobytes())

        (out_dir / (name + ".bin")).write_bytes(data)

        meta = {
            "name": name,
            "shape": [EXPECTED_TOKEN_COUNT, width],
            "order": "token-major",
            "dtype": "float32-le",
            "source_dtype": str(tensor.dtype),
            "bytes": len(data),
            "sha256": digest,
            "final_row_sha256": final_digest,
            "min": float(values.min()),
            "max": float(values.max()),
        }

        (out_dir / (name + ".json")).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summary["surfaces"][name] = meta

        print(
            "%-20s [%d, %5d] sha=%s min=%g max=%g"
            % (
                name,
                EXPECTED_TOKEN_COUNT,
                width,
                digest,
                meta["min"],
                meta["max"],
            )
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("HF ATTN0 MLA STAGE CAPTURE: PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
