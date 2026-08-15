#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(r"D:\llama.cpp-longcat-pre-gate4")

HF_DIR = Path(
    r"D:\LongCat-Flash-Lite-Sparse-Uncensored-Heretic-Native-MTP-And-LSA-Preserved"
)

RUNTIME = HF_DIR / "modeling_longcat_flash_sparse.py"

HF_ORACLE = ROOT / "hf_hidden_512_v4" / "inp_embd_ngram.bin"
CPP_ORACLE = (
    ROOT
    / "cpp_hidden_512_fa_off_f32"
    / "inp_embd_ngram.bin"
)

TOKEN_DIR = ROOT / "sparse_512_fa_off"

OUT_DIR = ROOT / "ngram_arithmetic_recon_512"

EXPECTED_RUNTIME_SHA = (
    "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
)

EXPECTED_TOKEN_SHA = (
    "4893d78751e8577f817da21a7e00718c7c14e0d80732e1b7e4da36da6677821c"
)

EXPECTED_FINAL_IDS = [
    1966557,
    1966545,
    1966533,
    1966521,
    6458256,
    4834970,
    3211684,
    1588398,
    7348131,
    4074420,
    883953,
    8000369,
]

NTOK = 512
HIDDEN = 3072
NGRAM_DIM = 256
VOCAB = 131072
NEIGHBOR = 4
SPLIT = 4
RATIO = 78
EOS = 2
NGRAM_COUNT = 12


def stop(message: str) -> None:
    raise SystemExit("STOP: " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shift_right_ignore_eos(
    tokens: np.ndarray,
    shift: int,
) -> np.ndarray:
    result = np.zeros_like(tokens, dtype=np.int64)

    eos_positions = np.flatnonzero(tokens == EOS)
    prev_idx = 0

    for eos_idx_np in eos_positions:
        eos_idx = int(eos_idx_np)
        end_idx = eos_idx + 1

        if end_idx - prev_idx > shift:
            result[prev_idx + shift:end_idx] = (
                tokens[prev_idx:end_idx - shift]
            )

        prev_idx = end_idx

    if prev_idx < tokens.size and tokens.size - prev_idx > shift:
        result[prev_idx + shift:] = (
            tokens[prev_idx:tokens.size - shift]
        )

    return result


def compute_hash_ids(tokens: np.ndarray) -> list[np.ndarray]:
    m = RATIO * VOCAB

    shifted = {
        shift: shift_right_ignore_eos(tokens, shift)
        for shift in range(1, NEIGHBOR)
    }

    result: list[np.ndarray] = []

    for ng in range(2, NEIGHBOR + 1):
        for split in range(SPLIT):
            index = (ng - 2) * SPLIT + split
            emb_vocab_dim = m + index * 2 + 1

            power_mods = []
            power_mod = 1

            for _ in range(ng - 1):
                power_mod = (
                    power_mod * VOCAB
                ) % emb_vocab_dim
                power_mods.append(power_mod)

            values = tokens.copy()

            for p in range(ng - 1):
                values = (
                    values
                    + shifted[p + 1] * power_mods[p]
                )

            result.append(values % emb_vocab_dim)

    return result


def load_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        stop("missing oracle: %s" % path)

    if path.stat().st_size != HIDDEN * 4:
        stop(
            "%s has %d bytes, expected %d"
            % (path, path.stat().st_size, HIDDEN * 4)
        )

    x = np.fromfile(path, dtype="<f4")

    if x.size != HIDDEN:
        stop("wrong vector length: %s" % path)

    if not np.isfinite(x).all():
        stop("nonfinite oracle: %s" % path)

    return x


def metric(candidate: np.ndarray, reference: np.ndarray):
    c = candidate.astype(np.float64)
    r = reference.astype(np.float64)

    delta = c - r
    ad = np.abs(delta)

    rmse = float(np.sqrt(np.mean(delta * delta)))
    ref_rms = float(np.sqrt(np.mean(r * r)))
    rel = rmse / ref_rms if ref_rms else float("nan")

    denom = float(np.linalg.norm(c) * np.linalg.norm(r))
    cosine = (
        float(np.dot(c, r) / denom)
        if denom
        else float("nan")
    )

    return {
        "max_abs": float(ad.max()),
        "mean_abs": float(ad.mean()),
        "rmse": rmse,
        "rel": rel,
        "cosine": cosine,
        "exact": int(np.count_nonzero(candidate == reference)),
    }


def get_index() -> dict[str, str]:
    path = HF_DIR / "model.safetensors.index.json"

    if not path.is_file():
        stop("missing Safetensors index")

    obj = json.loads(path.read_text(encoding="utf-8"))
    weight_map = obj.get("weight_map")

    if not isinstance(weight_map, dict):
        stop("invalid Safetensors weight_map")

    return weight_map


def load_slice(
    weight_map: dict[str, str],
    name: str,
    row: int,
) -> torch.Tensor:
    shard_name = weight_map.get(name)

    if shard_name is None:
        stop("missing source tensor: %s" % name)

    shard = HF_DIR / shard_name

    with safe_open(
        shard,
        framework="pt",
        device="cpu",
    ) as handle:
        sl = handle.get_slice(name)
        shape = tuple(int(v) for v in sl.get_shape())

        if row < 0 or row >= shape[0]:
            stop(
                "%s row %d outside shape %s"
                % (name, row, shape)
            )

        value = sl[row:row + 1].contiguous()

    if value.dtype != torch.bfloat16:
        stop("%s row is not BF16" % name)

    return value


def load_full(
    weight_map: dict[str, str],
    name: str,
) -> torch.Tensor:
    shard_name = weight_map.get(name)

    if shard_name is None:
        stop("missing source tensor: %s" % name)

    shard = HF_DIR / shard_name

    with safe_open(
        shard,
        framework="pt",
        device="cpu",
    ) as handle:
        value = handle.get_tensor(name).contiguous()

    if value.dtype != torch.bfloat16:
        stop("%s is not BF16" % name)

    return value


def main() -> int:
    if not torch.cuda.is_available():
        stop("CUDA is unavailable")

    runtime_sha = sha256_file(RUNTIME)

    print("runtime SHA256 =", runtime_sha)

    if runtime_sha != EXPECTED_RUNTIME_SHA:
        stop("frozen v4 runtime SHA mismatch")

    candidates = sorted(TOKEN_DIR.glob("*-tokens.bin"))

    if len(candidates) != 1:
        stop(
            "expected one *-tokens.bin, found %d"
            % len(candidates)
        )

    token_path = candidates[0]
    token_sha = sha256_file(token_path)

    print("token SHA256   =", token_sha)

    if token_sha != EXPECTED_TOKEN_SHA:
        stop("authoritative 512-token SHA mismatch")

    tokens = np.fromfile(
        token_path,
        dtype="<i4",
    ).astype(np.int64)

    if tokens.size != NTOK:
        stop(
            "token count %d != %d"
            % (tokens.size, NTOK)
        )

    ids = compute_hash_ids(tokens)

    if len(ids) != NGRAM_COUNT:
        stop("wrong number of n-gram ID arrays")

    final_ids = [int(x[-1]) for x in ids]

    print("final IDs      =", final_ids)

    if final_ids != EXPECTED_FINAL_IDS:
        stop("final n-gram IDs changed")

    hf_oracle = load_vector(HF_ORACLE)
    cpp_oracle = load_vector(CPP_ORACLE)

    weight_map = get_index()

    base_row = load_slice(
        weight_map,
        "model.embed_tokens.weight",
        483,
    )

    if tuple(base_row.shape) != (1, HIDDEN):
        stop(
            "base row shape %s != (1, %d)"
            % (tuple(base_row.shape), HIDDEN)
        )

    # Preserve HF's [batch, sequence, hidden] shape.
    base_cpu = (
        base_row
        .view(1, 1, HIDDEN)
        .expand(1, NTOK, HIDDEN)
        .contiguous()
    )

    # Build each complete [1, 512, 256] n-gram embedding input
    # while reading only the very small set of rows actually used.
    emb_inputs_cpu = []

    print()
    print("===== SELECTED ROW COUNTS =====")

    for index in range(NGRAM_COUNT):
        source_name = (
            "model.oe_embed_tokens%d.weight" % index
        )

        unique_ids = sorted(
            int(v)
            for v in np.unique(ids[index])
        )

        row_map = {}

        for row_id in unique_ids:
            row_map[row_id] = load_slice(
                weight_map,
                source_name,
                row_id,
            )

        rows = torch.cat(
            [
                row_map[int(row_id)]
                for row_id in ids[index]
            ],
            dim=0,
        )

        if tuple(rows.shape) != (NTOK, NGRAM_DIM):
            stop(
                "%s assembled shape %s"
                % (source_name, tuple(rows.shape))
            )

        emb_inputs_cpu.append(
            rows.view(1, NTOK, NGRAM_DIM).contiguous()
        )

        print(
            "ngram %2d: unique_rows=%d ids=%s"
            % (
                index,
                len(unique_ids),
                unique_ids,
            )
        )

    device = torch.device("cuda:0")

    print()
    print("===== CUDA =====")
    print("device =", torch.cuda.get_device_name(device))
    print(
        "torch.backends.cuda.matmul.allow_tf32 =",
        torch.backends.cuda.matmul.allow_tf32,
    )

    # Match the frozen HF capture configuration.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(
        "TF32 after disable =",
        torch.backends.cuda.matmul.allow_tf32,
    )

    with torch.inference_mode():
        base_bf = base_cpu.to(
            device=device,
            dtype=torch.bfloat16,
        )

        # Variant A: literal frozen-HF arithmetic:
        # BF16 linear output, BF16 sequential adds, BF16 division.
        x_hf_bf16 = base_bf.clone()

        # Variant B: BF16 matmul outputs, but F32 add/divide.
        x_bf16proj_f32acc = base_bf.float()

        # Variant C: F32 matmul result rounded to BF16 before the
        # same BF16 sequential add/divide chain.
        x_f32proj_bf16seq = base_bf.clone()

        # Variant D: exact BF16-stored inputs promoted to F32 before
        # matmul, then F32 add/divide throughout.
        x_f32_all = base_bf.float()

        projection_shard_name = weight_map.get(
            "model.oe_embed_proj0.weight"
        )

        if projection_shard_name is None:
            stop("projection 0 missing from index")

        # All 12 are known to live in the same final shard, but fail
        # closed if that invariant changed.
        for index in range(NGRAM_COUNT):
            name = "model.oe_embed_proj%d.weight" % index

            if weight_map.get(name) != projection_shard_name:
                stop(
                    "projection tensors are no longer in one shard"
                )

        projection_shard = HF_DIR / projection_shard_name

        with safe_open(
            projection_shard,
            framework="pt",
            device="cpu",
        ) as handle:
            for index in range(NGRAM_COUNT):
                name = (
                    "model.oe_embed_proj%d.weight" % index
                )

                w_cpu = handle.get_tensor(name).contiguous()

                if w_cpu.dtype != torch.bfloat16:
                    stop("%s is not BF16" % name)

                if tuple(w_cpu.shape) != (HIDDEN, NGRAM_DIM):
                    stop(
                        "%s shape %s != (%d, %d)"
                        % (
                            name,
                            tuple(w_cpu.shape),
                            HIDDEN,
                            NGRAM_DIM,
                        )
                    )

                emb_bf = emb_inputs_cpu[index].to(
                    device=device,
                    dtype=torch.bfloat16,
                )

                w_bf = w_cpu.to(
                    device=device,
                    dtype=torch.bfloat16,
                )

                proj_bf = F.linear(emb_bf, w_bf)

                if proj_bf.dtype != torch.bfloat16:
                    stop(
                        "BF16 F.linear unexpectedly returned %s"
                        % proj_bf.dtype
                    )

                proj_f32 = F.linear(
                    emb_bf.float(),
                    w_bf.float(),
                )

                if proj_f32.dtype != torch.float32:
                    stop(
                        "F32 F.linear unexpectedly returned %s"
                        % proj_f32.dtype
                    )

                x_hf_bf16 = x_hf_bf16 + proj_bf

                x_bf16proj_f32acc = (
                    x_bf16proj_f32acc
                    + proj_bf.float()
                )

                x_f32proj_bf16seq = (
                    x_f32proj_bf16seq
                    + proj_f32.to(torch.bfloat16)
                )

                x_f32_all = x_f32_all + proj_f32

                del emb_bf
                del w_bf
                del proj_bf
                del proj_f32
                del w_cpu

        x_hf_bf16 = x_hf_bf16 / 13
        x_f32proj_bf16seq = x_f32proj_bf16seq / 13

        x_bf16proj_f32acc = x_bf16proj_f32acc / 13.0
        x_f32_all = x_f32_all / 13.0

        if x_hf_bf16.dtype != torch.bfloat16:
            stop("HF reconstruction did not remain BF16")

        if x_f32proj_bf16seq.dtype != torch.bfloat16:
            stop("F32-proj/BF16-seq did not remain BF16")

        if x_bf16proj_f32acc.dtype != torch.float32:
            stop("BF16-proj/F32-acc did not remain F32")

        if x_f32_all.dtype != torch.float32:
            stop("F32-all did not remain F32")

        torch.cuda.synchronize()

        variants = {
            "hf_cuda_bf16_seq":
                x_hf_bf16[0, -1].float().cpu().numpy(),
            "f32proj_bf16_seq":
                x_f32proj_bf16seq[0, -1].float().cpu().numpy(),
            "bf16proj_f32_acc":
                x_bf16proj_f32acc[0, -1].cpu().numpy(),
            "f32_all":
                x_f32_all[0, -1].cpu().numpy(),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("===== VARIANT SHA256 =====")

    for name, values in variants.items():
        values = np.asarray(
            values,
            dtype="<f4",
        )

        if values.size != HIDDEN:
            stop("%s wrong output size" % name)

        if not np.isfinite(values).all():
            stop("%s contains nonfinite values" % name)

        out_path = OUT_DIR / (name + ".bin")
        values.tofile(out_path)

        print(
            "%-22s %s"
            % (
                name,
                sha256_file(out_path),
            )
        )

        variants[name] = values

    print()
    print("===== AGAINST HF inp_embd_ngram =====")
    print(
        "{:<22} {:>11} {:>11} {:>11} {:>11} {:>14} {:>11}".format(
            "variant",
            "max_abs",
            "mean_abs",
            "rmse",
            "rel_rmse",
            "cosine",
            "exact",
        )
    )
    print("-" * 104)

    for name, values in variants.items():
        m = metric(values, hf_oracle)

        print(
            "{:<22} {:11.6g} {:11.6g} {:11.6g} {:11.6g} "
            "{:14.10f} {:4d}/3072".format(
                name,
                m["max_abs"],
                m["mean_abs"],
                m["rmse"],
                m["rel"],
                m["cosine"],
                m["exact"],
            )
        )

    print()
    print("===== AGAINST C++ inp_embd_ngram =====")
    print(
        "{:<22} {:>11} {:>11} {:>11} {:>11} {:>14} {:>11}".format(
            "variant",
            "max_abs",
            "mean_abs",
            "rmse",
            "rel_rmse",
            "cosine",
            "exact",
        )
    )
    print("-" * 104)

    for name, values in variants.items():
        m = metric(values, cpp_oracle)

        print(
            "{:<22} {:11.6g} {:11.6g} {:11.6g} {:11.6g} "
            "{:14.10f} {:4d}/3072".format(
                name,
                m["max_abs"],
                m["mean_abs"],
                m["rmse"],
                m["rel"],
                m["cosine"],
                m["exact"],
            )
        )

    hf_recon_exact = np.array_equal(
        variants["hf_cuda_bf16_seq"],
        hf_oracle,
    )

    print()
    print(
        "hf_cuda_bf16_seq raw exact to HF oracle =",
        hf_recon_exact,
    )

    print(
        "hf_cuda_bf16_seq exact elements =",
        int(
            np.count_nonzero(
                variants["hf_cuda_bf16_seq"]
                == hf_oracle
            )
        ),
        "/ 3072",
    )

    print()
    print("STANDALONE N-GRAM ARITHMETIC RECONSTRUCTION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())