#!/usr/bin/env python3
"""HF 2050 first-owner LSA indexer surface capture (Run A).

One full-CausalLM forward at the frozen 2050x483 raw-token stream,
use_cache=False, capturing the first-owner indexer surfaces that mirror
the byte-identical C++ S-family bank, plus a secondary final-row logits
copy that MUST byte-equal the Run B canonical logits (the
instrumentation-inertness gate).

Instrumentation policy (observation-inertness rules, pre-registered):
  * Pure module forward hooks wherever a module boundary exists
    (template: capture_longcat_hf_block2_mla_stages_512.py).
  * Call-once observation wrappers for indexer.project_key and
    indexer._rope_q: each calls the ORIGINAL bound method exactly once,
    snaps the returned tensor, and returns the EXACT original tensor
    object unchanged. No replica execution anywhere.
  * Indexer gates via an in-situ F-proxy: the sparse module's namespace
    name `F` is rebound to a forwarding proxy whose `linear` calls the
    real torch.nn.functional.linear, snaps the result when uniquely
    identified (layer-0 owner window AND fp32 input AND weight shape
    [16, 3072]), and returns the exact original result. Exactly one
    match is a hard gate. The snapped tensor is the PRE-SCALE projection
    w_raw (the 16**-0.5 scale at S:552 is a separate op); the exact
    identity `C++ lsa_indexer_weights = w_raw/sqrt(2048)` is recorded.
  * select()/torch.topk execute exactly once, in the production forward;
    the owner top-K artifacts come ONLY from the production self_attn
    output tuple element [2]. No second top-k exists on any path.

Fail-closed sparse-engagement gates (mirroring the C++ validity stack)
run on every sublayer's mode/range and on all 14 owner top-K tensors.
Gate 4 remains NOT RUN; no Gate-4 criterion; no C++ comparison here
(the offline comparator judges blockers 1-3 separately).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path

EXPECTED_RUNTIME_SHA256 = (
    "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
)
EXPECTED_CONFIG_SHA256 = (
    "116a80c97e4215bf26668d93b4efd6043b2990e26f9157f2697cffeac17027d5"
)
EXPECTED_BASE_MODELING_SHA256 = (
    "bf7aa6387cf5bdf6c80b4a0f1b7bdd4878809fe33763323247c5fb73c4018659"
)
EXPECTED_ROPE_UTILS_SHA256 = (
    "a8bf3f6a53760366fb5fa51cecc06a8707d3cded36fd8f3ac51e140c0718af21"
)
EXPECTED_TOKEN_SHA256 = (
    "eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed"
)
EXPECTED_TOKEN_COUNT = 2050
EXPECTED_TOKEN_ID = 483
VOCAB_SIZE = 131072

EXPECTED_TORCH_VERSION = "2.13.0+cu132"
EXPECTED_TRANSFORMERS_VERSION = "5.15.0"

EXPECTED_NUM_LAYERS = 14
EXPECTED_INDEX_TOPK = 2048
EXPECTED_INDEX_N_HEADS = 16
EXPECTED_INDEX_HEAD_DIM = 128
EXPECTED_ROPE_HEAD_DIM = 64
INIT_TOKENS = 16
LOCAL_TOKENS = 1024
OWNER_MODE = "sparse-owner"
REUSE_MODE = "sparse-reuse"
EXPECTED_VALID_TOPK_RANGE = [1, 2048]

EXPECTED_ROPE_PARAMETERS = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 120.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 8192.0,
    "rope_theta": 1000000.0,
    "rope_type": "yarn",
}

# Full-sequence activation surfaces: name -> trailing width.
SURFACE_WIDTHS = {
    "hf_attn_norm0": 3072,
    "hf_q_a_norm0": 1536,
    "hf_indexer_k_proj": 128,
    "hf_indexer_k_norm": 128,
    "hf_indexer_k": 128,
    "hf_indexer_q_proj": 2048,
    "hf_indexer_q": 2048,          # flattened head-major h*128+d
    "hf_indexer_weights_prescale": 16,
    "hf_rope_cos": 64,
    "hf_rope_sin": 64,
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
    ap.add_argument("--canonical-logits-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args()

    model_dir = Path(ns.model_dir).resolve()
    tokens_bin = Path(ns.tokens_bin).resolve()
    canonical_bin = Path(ns.canonical_logits_bin).resolve()
    out_dir = Path(ns.out_dir).resolve()

    # ---- environment fail-closes ----
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set - refusing to run")
    if importlib.util.find_spec("kernels") is not None:
        stop("'kernels' package importable - RMSNorm kernel-swap hazard")
    if out_dir.exists():
        stop(f"out dir already exists (fresh-dir contract): {out_dir}")

    runtime = model_dir / "modeling_longcat_flash_sparse.py"
    config_py = model_dir / "configuration_longcat_flash_sparse.py"
    for p, what in (
        (model_dir, "model dir"),
        (runtime, "frozen runtime"),
        (config_py, "frozen configuration"),
        (tokens_bin, "tokens bin"),
        (canonical_bin, "canonical Run B logits bin"),
    ):
        if not p.exists():
            stop(f"{what} missing: {p}")

    runtime_sha = sha256_file(runtime)
    config_sha = sha256_file(config_py)
    token_sha = sha256_file(tokens_bin)
    canonical_sha = sha256_file(canonical_bin)
    print(f"runtime_sha256={runtime_sha}")
    print(f"config_sha256={config_sha}")
    print(f"tokens_bin_sha256={token_sha}")
    print(f"canonical_logits_sha256={canonical_sha}")
    if runtime_sha != EXPECTED_RUNTIME_SHA256:
        stop("frozen runtime SHA mismatch")
    if config_sha != EXPECTED_CONFIG_SHA256:
        stop("frozen configuration SHA mismatch")
    if token_sha != EXPECTED_TOKEN_SHA256:
        stop("authoritative 2050-token SHA mismatch")

    canonical_bytes = canonical_bin.read_bytes()
    if len(canonical_bytes) != VOCAB_SIZE * 4:
        stop(
            f"canonical logits size {len(canonical_bytes)} != "
            f"{VOCAB_SIZE * 4}"
        )

    raw = tokens_bin.read_bytes()
    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop("token file size mismatch")
    ids = list(struct.unpack(f"<{EXPECTED_TOKEN_COUNT}i", raw))
    for i, tid in enumerate(ids):
        if not 0 <= tid < VOCAB_SIZE:
            stop(f"token {i} out of range: {tid}")
        if tid != EXPECTED_TOKEN_ID:
            stop(f"token {i} != frozen id {EXPECTED_TOKEN_ID}: {tid}")
    print(f"sequence_length={len(ids)}")
    print(f"sys_executable={sys.executable}")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if torch.__version__ != EXPECTED_TORCH_VERSION:
        stop(f"torch {torch.__version__} != frozen {EXPECTED_TORCH_VERSION}")
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        stop(
            f"transformers {transformers.__version__} != frozen "
            f"{EXPECTED_TRANSFORMERS_VERSION}"
        )
    if not torch.cuda.is_available():
        stop("CUDA unavailable")

    tf_root = Path(transformers.__file__).resolve().parent
    base_modeling = (
        tf_root / "models" / "longcat_flash" / "modeling_longcat_flash.py"
    )
    rope_utils = tf_root / "modeling_rope_utils.py"
    for p, expected, what in (
        (base_modeling, EXPECTED_BASE_MODELING_SHA256, "base modeling"),
        (rope_utils, EXPECTED_ROPE_UTILS_SHA256, "rope utils"),
    ):
        if not p.is_file():
            stop(f"{what} missing: {p}")
        got = sha256_file(p)
        print(f"{p.name}_sha256={got}")
        if got != expected:
            stop(f"{what} SHA mismatch: {got}")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        stop("TF32 read-back failed")
    print(f"float32_matmul_precision={torch.get_float32_matmul_precision()}")

    # ---- model load (proven 512-family path) ----
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

    # ---- post-load config / runtime-instantiated module gates ----
    cfg = model.config
    trunk = model.model
    if len(trunk.layers) != EXPECTED_NUM_LAYERS:
        stop(f"layer count {len(trunk.layers)} != {EXPECTED_NUM_LAYERS}")
    for name, want in (
        ("index_topk", EXPECTED_INDEX_TOPK),
        ("index_n_heads", EXPECTED_INDEX_N_HEADS),
        ("index_head_dim", EXPECTED_INDEX_HEAD_DIM),
        ("cli_factor", 2),
        ("num_hidden_layers", 2 * EXPECTED_NUM_LAYERS),
        ("vocab_size", VOCAB_SIZE),
    ):
        got = getattr(cfg, name, None)
        if int(got) != int(want):
            stop(f"config.{name}={got} != {want}")
    if abs(float(cfg.rms_norm_eps) - 1e-5) > 1e-12:
        stop(f"config.rms_norm_eps={cfg.rms_norm_eps} != 1e-5")

    rope_params = dict(cfg.rope_parameters)
    if set(rope_params.keys()) != set(EXPECTED_ROPE_PARAMETERS.keys()):
        stop(f"rope_parameters keys {sorted(rope_params)} unexpected")
    for k, want in EXPECTED_ROPE_PARAMETERS.items():
        got = rope_params[k]
        if k == "rope_type":
            if got != want:
                stop(f"rope_parameters.{k}={got!r} != {want!r}")
        elif abs(float(got) - float(want)) > 1e-9:
            stop(f"rope_parameters.{k}={got} != {want}")

    attn0 = trunk.layers[0].self_attn[0]
    idx = attn0.indexer
    if idx is None or not getattr(attn0, "indexer_owner", False):
        stop("layers[0].self_attn[0] is not the indexer owner")
    if trunk.layers[0].self_attn[1].indexer is not None:
        stop("layers[0].self_attn[1] unexpectedly owns an indexer")
    for got, want, what in (
        (float(idx.k_norm.variance_epsilon), 1e-6, "indexer.k_norm eps"),
        (float(attn0.q_a_layernorm.variance_epsilon), 1e-6, "q_a_layernorm eps"),
        (
            float(trunk.layers[0].input_layernorm[0].variance_epsilon),
            1e-5,
            "input_layernorm[0] eps",
        ),
        (float(trunk.rotary_emb.attention_scaling), 1.0, "attention_scaling"),
    ):
        if abs(got - want) > 1e-15:
            stop(f"{what}={got} != {want}")
    for got, want, what in (
        (int(idx.n_heads), EXPECTED_INDEX_N_HEADS, "indexer.n_heads"),
        (int(idx.head_dim), EXPECTED_INDEX_HEAD_DIM, "indexer.head_dim"),
        (int(idx.rope_head_dim), EXPECTED_ROPE_HEAD_DIM, "indexer.rope_head_dim"),
        (int(idx.topk), EXPECTED_INDEX_TOPK, "indexer.topk"),
        (int(idx.num_init_tokens), INIT_TOKENS, "indexer.num_init_tokens"),
        (int(idx.num_local_tokens), LOCAL_TOKENS, "indexer.num_local_tokens"),
    ):
        if got != want:
            stop(f"{what}={got} != {want}")
    for w, dt, shape, what in (
        (idx.wk.weight, torch.bfloat16, (128, 3072), "indexer.wk.weight"),
        (idx.wq_b.weight, torch.bfloat16, (2048, 1536), "indexer.wq_b.weight"),
        (
            idx.weights_proj.weight,
            torch.float32,
            (16, 3072),
            "indexer.weights_proj.weight",
        ),
        (idx.k_norm.weight, torch.bfloat16, (128,), "indexer.k_norm.weight"),
    ):
        if w.dtype != dt or tuple(w.shape) != shape:
            stop(f"{what}: dtype {w.dtype} shape {tuple(w.shape)}")

    out_dir.mkdir(parents=True, exist_ok=False)

    summary: dict = {
        "description": (
            "HF 2050 first-owner LSA indexer capture (Run A). "
            "Observation-only instrumentation; select()/torch.topk "
            "executed exactly once in the production forward."
        ),
        "runtime_sha256": runtime_sha,
        "config_sha256": config_sha,
        "base_modeling_sha256": EXPECTED_BASE_MODELING_SHA256,
        "rope_utils_sha256": EXPECTED_ROPE_UTILS_SHA256,
        "tokens_bin_sha256": token_sha,
        "canonical_logits_sha256": canonical_sha,
        "sequence_length": EXPECTED_TOKEN_COUNT,
        "layout": "token-major [2050, width] float32-le",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sys_executable": sys.executable,
        "tf32_matmul_allow": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allow": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "surfaces": {},
        "weights": {},
        "gates": {},
    }
    sums_lines: list[str] = []

    def save_f32(name: str, arr, source_dtype: str, extra: dict | None = None):
        arr = np.ascontiguousarray(arr, dtype="<f4")
        if not np.isfinite(arr).all():
            stop(f"{name}: non-finite values")
        path = out_dir / f"{name}.bin"
        path.write_bytes(arr.tobytes())
        sha = sha256_file(path)
        entry = {
            "shape": list(arr.shape),
            "order": "token-major",
            "dtype": "float32-le",
            "source_dtype": source_dtype,
            "bytes": arr.nbytes,
            "sha256": sha,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        if extra:
            entry.update(extra)
        sidecar = dict(entry)
        sidecar["name"] = name
        (out_dir / f"{name}.json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
        )
        summary["surfaces"][name] = entry
        sums_lines.append(f"{sha}  {name}.bin")
        print(f"{name}: sha256={sha} shape={list(arr.shape)}")
        return sha

    # ---- weight bank (pre-forward module reads) ----
    for wname, w in (
        ("hf_weight_k_norm", idx.k_norm.weight),
        ("hf_weight_wk", idx.wk.weight),
        ("hf_weight_wq_b", idx.wq_b.weight),
        ("hf_weight_weights_proj", idx.weights_proj.weight),
    ):
        v = w.detach().float().cpu().contiguous().numpy()
        sha = save_f32(wname, v, str(w.dtype))
        summary["weights"][wname] = sha

    # ---- instrumentation ----
    caps: dict = {}
    counters = {"project_key": 0, "_rope_q": 0, "f_linear_matches": 0}
    window = {"active": False}
    sublayer_records: list[dict] = []
    topk_caps: dict[int, object] = {}

    def once(name: str, t):
        if name in caps:
            stop(f"capture fired twice: {name}")
        caps[name] = t.detach().clone()

    def out_hook(name: str):
        def h(_m, _i, output):
            once(name, output)
        return h

    def rope_hook(_m, _i, output):
        once("hf_rope_cos", output[0])
        once("hf_rope_sin", output[1])

    def attn_pre_hook(_m, _args, _kwargs):
        window["active"] = True
        return None

    def record_hook(layer_idx: int, sub: int):
        def h(mod, args, kwargs, output):
            if layer_idx == 0 and sub == 0:
                window["active"] = False
            rec: dict = {"layer": layer_idx, "sublayer": sub}
            try:
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                rec["seq_len"] = (
                    int(hidden.shape[1]) if hidden is not None else None
                )
                rec["mode"] = getattr(mod, "last_lsa_mode", None)
                rng = getattr(mod, "last_lsa_valid_topk_range", None)
                rec["valid_topk_range"] = (
                    [int(rng[0]), int(rng[1])] if rng is not None else None
                )
                topk = (
                    output[2]
                    if isinstance(output, tuple) and len(output) >= 3
                    else None
                )
                rec["topk_is_none"] = topk is None
                if topk is not None:
                    rec["topk_shape"] = [int(x) for x in topk.shape]
                    rec["topk_dtype"] = str(topk.dtype)
                    if sub == 0:
                        if layer_idx in topk_caps:
                            rec["error"] = "owner topk captured twice"
                        else:
                            topk_caps[layer_idx] = topk.detach().to(
                                "cpu", copy=True
                            )
            except Exception as exc:  # noqa: BLE001 - recorded, gated below
                rec["error"] = f"{type(exc).__name__}: {exc}"
            sublayer_records.append(rec)
            return None

        return h

    # Call-once observation wrappers (original called exactly once; the
    # exact original tensor object is returned to the model).
    orig_project_key = idx.project_key
    orig_rope_q = idx._rope_q

    def observed_project_key(hidden_states, position_embeddings):
        out = orig_project_key(hidden_states, position_embeddings)
        counters["project_key"] += 1
        once("hf_indexer_k", out)
        return out

    def observed_rope_q(query, position_embeddings):
        out = orig_rope_q(query, position_embeddings)
        counters["_rope_q"] += 1
        once("hf_indexer_q_4d", out)
        return out

    # In-situ F-proxy for the gates F.linear call inside select().
    sparse_mod = sys.modules[type(idx).__module__]
    if not hasattr(sparse_mod, "F"):
        stop(
            "sparse module has no top-level name 'F' - the primary gates "
            "observation seam is unavailable (review before any fallback)"
        )
    real_F = sparse_mod.F

    class _FProxy:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def linear(self, *args, **kwargs):
            out = self._real.linear(*args, **kwargs)
            try:
                inp = args[0] if args else kwargs.get("input")
                weight = args[1] if len(args) > 1 else kwargs.get("weight")
                if (
                    window["active"]
                    and inp is not None
                    and weight is not None
                    and inp.dtype == torch.float32
                    and weight.dtype == torch.float32
                    and tuple(weight.shape) == (16, 3072)
                ):
                    counters["f_linear_matches"] += 1
                    once("hf_indexer_weights_prescale", out)
            except Exception as exc:  # noqa: BLE001 - fail loudly, not silently
                stop(f"F-proxy observer error: {type(exc).__name__}: {exc}")
            return out

    handles = [
        trunk.layers[0].input_layernorm[0].register_forward_hook(
            out_hook("hf_attn_norm0")
        ),
        attn0.q_a_layernorm.register_forward_hook(out_hook("hf_q_a_norm0")),
        idx.wk.register_forward_hook(out_hook("hf_indexer_k_proj")),
        idx.k_norm.register_forward_hook(out_hook("hf_indexer_k_norm")),
        idx.wq_b.register_forward_hook(out_hook("hf_indexer_q_proj")),
        trunk.rotary_emb.register_forward_hook(rope_hook),
        attn0.register_forward_pre_hook(attn_pre_hook, with_kwargs=True),
    ]
    for i, layer in enumerate(trunk.layers):
        for sub in (0, 1):
            handles.append(
                layer.self_attn[sub].register_forward_hook(
                    record_hook(i, sub), with_kwargs=True
                )
            )

    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
    try:
        idx.project_key = observed_project_key
        idx._rope_q = observed_rope_q
        sparse_mod.F = _FProxy(real_F)
        t1 = time.perf_counter()
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
        torch.cuda.synchronize()
        print(f"forward_seconds={time.perf_counter() - t1:.3f}")
    finally:
        sparse_mod.F = real_F
        for attr in ("project_key", "_rope_q"):
            if attr in idx.__dict__:
                del idx.__dict__[attr]
        for h in handles:
            h.remove()

    # ---- post-forward gates ----
    if counters["project_key"] != 1:
        stop(f"project_key wrapper fired {counters['project_key']}x != 1")
    if counters["_rope_q"] != 1:
        stop(f"_rope_q wrapper fired {counters['_rope_q']}x != 1")
    if counters["f_linear_matches"] != 1:
        stop(
            "F-proxy gates signature matched "
            f"{counters['f_linear_matches']}x != exactly 1 inside the window"
        )
    summary["gates"]["observation_counters"] = dict(counters)

    expected_caps = {
        "hf_attn_norm0",
        "hf_q_a_norm0",
        "hf_indexer_k_proj",
        "hf_indexer_k_norm",
        "hf_indexer_q_proj",
        "hf_indexer_k",
        "hf_indexer_q_4d",
        "hf_indexer_weights_prescale",
        "hf_rope_cos",
        "hf_rope_sin",
    }
    missing = expected_caps - set(caps)
    if missing:
        stop(f"captures did not fire: {sorted(missing)}")

    if len(sublayer_records) != 2 * EXPECTED_NUM_LAYERS:
        stop(
            f"sublayer records {len(sublayer_records)} != "
            f"{2 * EXPECTED_NUM_LAYERS}"
        )
    seen = set()
    for rec in sublayer_records:
        key = (rec["layer"], rec["sublayer"])
        if key in seen:
            stop(f"duplicate sublayer record {key}")
        seen.add(key)
        tag = f"layer {rec['layer']} sublayer {rec['sublayer']}"
        if "error" in rec:
            stop(f"{tag}: record error {rec['error']}")
        if rec["seq_len"] != EXPECTED_TOKEN_COUNT:
            stop(f"{tag}: seq_len {rec['seq_len']} != 2050")
        expected_mode = OWNER_MODE if rec["sublayer"] == 0 else REUSE_MODE
        if rec["mode"] != expected_mode:
            stop(f"{tag}: mode {rec['mode']!r} != {expected_mode!r}")
        if rec["valid_topk_range"] != EXPECTED_VALID_TOPK_RANGE:
            stop(
                f"{tag}: valid_topk_range {rec['valid_topk_range']} != "
                f"{EXPECTED_VALID_TOPK_RANGE}"
            )
        if rec["topk_is_none"]:
            stop(f"{tag}: top-K is None (dense signature)")
        if rec["topk_shape"] != [1, EXPECTED_TOKEN_COUNT, EXPECTED_INDEX_TOPK]:
            stop(f"{tag}: topk_shape {rec['topk_shape']}")
        if rec["topk_dtype"] != "torch.int64":
            stop(f"{tag}: topk_dtype {rec['topk_dtype']}")
    if len(topk_caps) != EXPECTED_NUM_LAYERS:
        stop(f"owner topk captures {len(topk_caps)} != {EXPECTED_NUM_LAYERS}")
    summary["gates"]["sublayer_records"] = sublayer_records

    # Owner top-K structural battery (C++ validity-stack mirror), all owners.
    positions = np.arange(EXPECTED_TOKEN_COUNT, dtype=np.int64)
    expected_neg1 = np.maximum(0, EXPECTED_INDEX_TOPK - (positions + 1))
    for layer_idx in range(EXPECTED_NUM_LAYERS):
        t = topk_caps[layer_idx][0].numpy()
        tag = f"owner{2 * layer_idx:02d}"
        if t.shape != (EXPECTED_TOKEN_COUNT, EXPECTED_INDEX_TOPK):
            stop(f"{tag}: shape {t.shape}")
        if t.min() < -1 or t.max() >= EXPECTED_TOKEN_COUNT:
            stop(f"{tag}: index range [{t.min()}, {t.max()}]")
        neg1 = (t == -1).sum(axis=1)
        if not np.array_equal(neg1, expected_neg1):
            bad = int(np.nonzero(neg1 != expected_neg1)[0][0])
            stop(
                f"{tag}: -1 filler count row {bad}: {int(neg1[bad])} != "
                f"{int(expected_neg1[bad])}"
            )
        s = np.sort(t, axis=1)
        dup = (s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)
        if dup.any():
            stop(f"{tag}: duplicate non-negative index in row "
                 f"{int(np.nonzero(dup.any(axis=1))[0][0])}")
        causal_bad = (t > positions[:, None]) & (t >= 0)
        if causal_bad.any():
            stop(f"{tag}: non-causal index in row "
                 f"{int(np.nonzero(causal_bad.any(axis=1))[0][0])}")
        for p in (2048, 2049):
            row = set(int(x) for x in t[p] if x >= 0)
            needed = set(range(INIT_TOKENS)) | set(
                range(p - LOCAL_TOKENS + 1, p + 1)
            )
            if not needed.issubset(row):
                stop(f"{tag}: forced containment failed at row {p}")
    print("owner top-K structural battery: PASS (14/14 owners)")
    summary["gates"]["topk_structural_battery"] = "PASS 14/14"

    # ---- logits: shape, finiteness, A==B byte gate ----
    logits = output.logits
    if tuple(logits.shape) != (1, EXPECTED_TOKEN_COUNT, VOCAB_SIZE):
        stop(f"unexpected logits shape {tuple(logits.shape)}")
    last = logits[0, -1].float().detach().cpu().contiguous().numpy()
    last = np.ascontiguousarray(last, dtype="<f4")
    if not np.isfinite(last).all():
        stop("final-row logits contain non-finite values")
    ours = last.tobytes()
    if ours != canonical_bytes:
        stop(
            "Run A final-row logits != Run B canonical logits "
            "(instrumentation-inertness gate FAILED)"
        )
    print("logits identity gate: Run A final row == Run B canonical "
          "(byte-exact) PASS")
    summary["gates"]["runA_logits_byte_equal_runB"] = True
    top1 = int(last.argmax())
    summary["gates"]["runA_top1"] = top1
    print(f"runA_top1={top1}")

    # ---- saves ----
    def widened(name: str):
        t = caps[name]
        if t.dim() != 3 or tuple(t.shape[:2]) != (1, EXPECTED_TOKEN_COUNT):
            stop(f"{name}: unexpected shape {tuple(t.shape)}")
        return (
            t[0].float().detach().cpu().contiguous().numpy(),
            str(t.dtype),
        )

    for name in (
        "hf_attn_norm0",
        "hf_q_a_norm0",
        "hf_indexer_k_proj",
        "hf_indexer_k_norm",
        "hf_indexer_k",
        "hf_indexer_q_proj",
        "hf_indexer_weights_prescale",
        "hf_rope_cos",
        "hf_rope_sin",
    ):
        v, src = widened(name)
        if v.shape[1] != SURFACE_WIDTHS[name]:
            stop(f"{name}: width {v.shape[1]} != {SURFACE_WIDTHS[name]}")
        extra = None
        if name == "hf_indexer_weights_prescale":
            extra = {
                "scale_identity": (
                    "C++ lsa_indexer_weights == w_raw / sqrt(2048); this "
                    "surface is the pre-scale F.linear output w_raw "
                    "(in-situ production call, F-proxy seam)"
                )
            }
        save_f32(name, v, src, extra)

    q4 = caps["hf_indexer_q_4d"]
    if tuple(q4.shape) != (1, EXPECTED_TOKEN_COUNT, 16, 128):
        stop(f"hf_indexer_q_4d: unexpected shape {tuple(q4.shape)}")
    qv = q4[0].float().detach().cpu().contiguous().numpy().reshape(
        EXPECTED_TOKEN_COUNT, 16 * 128
    )
    save_f32(
        "hf_indexer_q",
        qv,
        str(q4.dtype),
        {"head_layout": "head-major h*128+d (matches C++ lsa_indexer_q_2d)"},
    )

    for layer_idx in range(EXPECTED_NUM_LAYERS):
        t = topk_caps[layer_idx][0].numpy()
        v32 = t.astype("<f4")
        if not np.array_equal(v32.astype(np.int64), t):
            stop(f"owner{2 * layer_idx:02d}: int->f32 conversion not exact")
        save_f32(
            f"hf_top_k_owner{2 * layer_idx:02d}",
            v32,
            "torch.int64",
            {"int_to_f32_exact": True, "fillers": "-1 (HF class)"},
        )

    logits_path = out_dir / "hf_logits_2050_runA.bin"
    logits_path.write_bytes(ours)
    logits_sha = sha256_file(logits_path)
    summary["surfaces"]["hf_logits_2050_runA"] = {
        "shape": [VOCAB_SIZE],
        "dtype": "float32-le",
        "bytes": len(ours),
        "sha256": logits_sha,
        "byte_equal_canonical": True,
        "top1": top1,
    }
    sums_lines.append(f"{logits_sha}  hf_logits_2050_runA.bin")
    print(f"hf_logits_2050_runA: sha256={logits_sha}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    sums_lines.append(f"{sha256_file(out_dir / 'summary.json')}  summary.json")
    (out_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sums_lines) + "\n", encoding="utf-8"
    )

    print("HF 2050 FIRST-OWNER LSA CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
