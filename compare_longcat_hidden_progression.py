#!/usr/bin/env python3

from pathlib import Path
import numpy as np

HF_DIR = Path(r"D:\llama.cpp-longcat-pre-gate4\hf_hidden_512_v4")
CPP_DIR = Path(r"D:\llama.cpp-longcat-pre-gate4\cpp_hidden_512_fa_off_f32")

NAMES = (
    ["inp_embd_ngram"]
    + ["logical_%02d" % i for i in range(13)]
    + ["result_norm"]
)

print(
    "{:<17} {:>11} {:>11} {:>11} {:>11} {:>14}".format(
        "surface",
        "max_abs",
        "mean_abs",
        "rmse",
        "rel_rmse",
        "cosine",
    )
)
print("-" * 82)

for name in NAMES:
    hf_path = HF_DIR / (name + ".bin")
    cpp_path = CPP_DIR / (name + ".bin")

    if not hf_path.is_file():
        raise SystemExit("STOP: missing HF file: %s" % hf_path)

    if not cpp_path.is_file():
        raise SystemExit("STOP: missing C++ file: %s" % cpp_path)

    h = np.fromfile(hf_path, dtype="<f4")
    x = np.fromfile(cpp_path, dtype="<f4")

    if h.size != 3072 or x.size != 3072:
        raise SystemExit(
            "STOP: wrong size for %s: HF=%d C++=%d"
            % (name, h.size, x.size)
        )

    if not np.isfinite(h).all():
        raise SystemExit("STOP: HF nonfinite: %s" % name)

    if not np.isfinite(x).all():
        raise SystemExit("STOP: C++ nonfinite: %s" % name)

    h64 = h.astype(np.float64)
    x64 = x.astype(np.float64)

    delta = x64 - h64
    abs_delta = np.abs(delta)

    rmse = float(np.sqrt(np.mean(delta * delta)))
    hf_rms = float(np.sqrt(np.mean(h64 * h64)))
    rel_rmse = rmse / hf_rms if hf_rms != 0.0 else float("nan")

    denom = float(np.linalg.norm(h64) * np.linalg.norm(x64))
    cosine = (
        float(np.dot(h64, x64) / denom)
        if denom != 0.0
        else float("nan")
    )

    print(
        "{:<17} {:11.6g} {:11.6g} {:11.6g} {:11.6g} {:14.10f}".format(
            name,
            float(abs_delta.max()),
            float(abs_delta.mean()),
            rmse,
            rel_rmse,
            cosine,
        )
    )

print()
print("HIDDEN PROGRESSION COMPARISON: PASS")