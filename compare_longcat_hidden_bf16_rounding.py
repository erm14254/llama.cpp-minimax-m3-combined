#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import torch

HF_DIR = Path(r"D:\llama.cpp-longcat-pre-gate4\hf_hidden_512_v4")
CPP_DIR = Path(r"D:\llama.cpp-longcat-pre-gate4\cpp_hidden_512_fa_off_f32")

NAMES = (
    ["inp_embd_ngram"]
    + ["logical_%02d" % i for i in range(13)]
    + ["result_norm"]
)


def bf16_round(x):
    t = torch.from_numpy(x.copy())
    return t.to(torch.bfloat16).to(torch.float32).numpy()


print(
    "{:<17} {:>12} {:>12} {:>12} {:>12} {:>10}".format(
        "surface",
        "raw_rmse",
        "bf16_rmse",
        "raw_rel",
        "bf16_rel",
        "exact",
    )
)
print("-" * 82)

for name in NAMES:
    h = np.fromfile(HF_DIR / (name + ".bin"), dtype="<f4")
    x = np.fromfile(CPP_DIR / (name + ".bin"), dtype="<f4")

    if h.size != 3072 or x.size != 3072:
        raise SystemExit(
            "STOP: wrong size for %s: HF=%d C++=%d"
            % (name, h.size, x.size)
        )

    if not np.isfinite(h).all() or not np.isfinite(x).all():
        raise SystemExit("STOP: nonfinite values at %s" % name)

    # Confirm the HF capture itself lies exactly on the BF16 lattice.
    h_round = bf16_round(h)
    if not np.array_equal(h, h_round):
        raise SystemExit(
            "STOP: HF surface is not exactly BF16-representable: %s"
            % name
        )

    xb = bf16_round(x)

    h64 = h.astype(np.float64)
    x64 = x.astype(np.float64)
    xb64 = xb.astype(np.float64)

    raw_delta = x64 - h64
    bf16_delta = xb64 - h64

    raw_rmse = float(np.sqrt(np.mean(raw_delta * raw_delta)))
    bf16_rmse = float(np.sqrt(np.mean(bf16_delta * bf16_delta)))

    hf_rms = float(np.sqrt(np.mean(h64 * h64)))

    raw_rel = raw_rmse / hf_rms
    bf16_rel = bf16_rmse / hf_rms

    exact = int(np.count_nonzero(xb == h))

    print(
        "{:<17} {:12.6g} {:12.6g} {:12.6g} {:12.6g} {:4d}/3072".format(
            name,
            raw_rmse,
            bf16_rmse,
            raw_rel,
            bf16_rel,
            exact,
        )
    )

print()
print("BF16-ROUNDING DIAGNOSTIC: PASS")