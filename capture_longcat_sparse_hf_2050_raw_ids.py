#!/usr/bin/env python3
"""2050 sibling of capture_longcat_sparse_hf_512_raw_ids.py (Run B).

Canonical HF 2050 final-row logits bank with an INDEPENDENT fail-closed
sparse-engagement proof. The frozen Gate-3 core
(D:\\llama.cpp-longcat-mtp\\capture_longcat_sparse_hf_gate3_logits.py,
SHA bb82bcb6...) is reused byte-frozen exactly as in the proven 512
wrapper: SHA-gate before import, exec_module, rebind INPUT_IDS, rewrite
sys.argv, call main() in-process.

Deltas vs the 512 sibling (all wrapper-side; the core is never edited):
  * EXPECTED_TOKEN_SHA256 / EXPECTED_TOKEN_COUNT pinned to the frozen
    2050 stream (2050 x i32 483, eb04e101...).
  * Pre-import environment gates: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
    fail-close, torch/transformers version asserts, transformers
    base-module SHA gates (the LongCat base classes carry half the
    load-bearing numerics), kernels-package absence, cudnn.allow_tf32
    hardening (the core sets matmul.allow_tf32 itself).
  * Sparse-engagement observation shim with a PROVEN, EXCLUSIVE
    interception seam: after exec_module the wrapper proves, from
    module.main's OWN executable bytecode (dis over main.__code__;
    nested code objects, comments, docstrings and unrelated scopes
    cannot satisfy the detector), that the name reaching
    `.from_pretrained()` is bound EXCLUSIVELY by an approved seam.
    The deferred seam requires the IMPORT_NAME "transformers" ->
    IMPORT_FROM "AutoModelForCausalLM" sequence with each import's own
    adjacent store, and REJECTS competing definitions/rebindings:
    IMPORT_FROM of the name from any non-transformers module, any
    additional STORE_FAST/STORE_NAME/STORE_GLOBAL/STORE_DEREF of the
    name beyond the approved import stores, any DELETE of the name,
    IMPORT_STAR, closure (cell/free) capture, or broken import->store
    adjacency. The module-global seam requires a pure LOAD_GLOBAL/
    LOAD_NAME read with ZERO binding-capable instructions on the name
    (identity-gated against the real transformers class, with
    main.__globals__ verified to BE the core module namespace, before
    that exact binding is patched; the deferred seam patches the
    transformers module attribute the import resolves at call time).
    The proof further connects the binding to the ACTUAL CALL SITES:
    every LOAD_ATTR/LOAD_METHOD "from_pretrained" in main's code
    object must be immediately fed by a load of the proven binding
    (LOAD_FAST/LOAD_FAST_CHECK of the name for the deferred-local
    seam; LOAD_GLOBAL/LOAD_NAME for the module-global seam; only
    EXTENDED_ARG/NOP/CACHE bookkeeping is skipped) -- zero sites, any
    foreign-receiver site (e.g. other.AutoModelForCausalLM
    .from_pretrained), or mixed receiver kinds fail closed BEFORE
    module.main(). Every patched binding is restored to its exact
    original object in finally, and the full bytecode-scan evidence
    (site counts, per-site receiver opcode/name, rejected sites, the
    call-site-proof flags) plus the resolved seam list are recorded in
    the engagement proof. The pass-through class delegates from_pretrained
    to the real class, registers OBSERVATION-ONLY forward hooks on all
    28 attention sublayers of the returned (unchanged) model, and
    returns it. Hooks record python primitives only (mode strings,
    sequence length, valid-topk ranges, top-K structure, and the FULL
    owner-0 structural battery: range [-1,2050), exact per-row filler
    counts, per-row unique nonnegative entries, causal indices only,
    zero fillers at rows 2047/2048/2049, forced containment at rows
    2048/2049) and return None -- no tensor is altered or replaced, no
    arithmetic touched.
  * Post-main() collector validation: even when the core prints PASS,
    this wrapper independently validates 14x "sparse-owner" +
    14x "sparse-reuse", seq-len 2050 everywhere, valid ranges (1, 2048),
    non-None int64 [1, 2050, 2048] top-K everywhere, and the owner-0
    structural class. Any failure -- including an empty or partial
    collector -- exits nonzero and the canonical artifact is REJECTED.

Gate 4 remains NOT RUN. This banks a future oracle only; it is NOT a
Gate-4 acceptance test and performs no C++ comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

ORIGINAL_CAPTURE = Path(
    r"D:\llama.cpp-longcat-mtp\capture_longcat_sparse_hf_gate3_logits.py"
)

EXPECTED_ORIGINAL_SHA256 = (
    "bb82bcb6c3bc1d21685221a884dac3b39dc7af06f54fea6187f606dddf4213cb"
)
EXPECTED_TOKEN_SHA256 = (
    "eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed"
)
EXPECTED_TOKEN_COUNT = 2050
EXPECTED_TOKEN_ID = 483
VOCAB_SIZE = 131072

EXPECTED_TORCH_VERSION = "2.13.0+cu132"
EXPECTED_TRANSFORMERS_VERSION = "5.15.0"
EXPECTED_BASE_MODELING_SHA256 = (
    "bf7aa6387cf5bdf6c80b4a0f1b7bdd4878809fe33763323247c5fb73c4018659"
)
EXPECTED_ROPE_UTILS_SHA256 = (
    "a8bf3f6a53760366fb5fa51cecc06a8707d3cded36fd8f3ac51e140c0718af21"
)

EXPECTED_NUM_LAYERS = 14
EXPECTED_INDEX_TOPK = 2048
OWNER_MODE = "sparse-owner"
REUSE_MODE = "sparse-reuse"
EXPECTED_VALID_TOPK_RANGE = (1, 2048)
INIT_TOKENS = 16
LOCAL_TOKENS = 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def _scan_main_bindings(code) -> dict:
    """Prove, from a function's OWN executable bytecode (its code object
    only; nested code objects are deliberately excluded), that the name
    ``AutoModelForCausalLM`` reaching ``.from_pretrained()`` can be
    bound EXCLUSIVELY by an approved seam:

    - ``exclusive_deferred_binding``: at least one ``IMPORT_NAME
      "transformers"`` -> ``IMPORT_FROM "AutoModelForCausalLM"``
      sequence (the deferred ``from transformers import ...``, which
      resolves the *transformers module attribute* at call time), each
      immediately followed by its own store of the name, AND no
      competing binding path exists: no ``IMPORT_FROM`` of the name
      from any other module, no additional ``STORE_FAST``/``STORE_NAME``
      /``STORE_GLOBAL``/``STORE_DEREF`` of the name beyond the approved
      import stores, no ``DELETE_*`` of the name, no ``IMPORT_STAR``,
      and the name is not a cell/free variable of this code object;
    - ``exclusive_global_read``: the function only READS the name
      (``LOAD_GLOBAL``/``LOAD_NAME``) -- the *module-global binding* --
      with zero binding-capable instructions on the name.

    Any other combination (a foreign import, a rebinding store, a
    delete, ``import *``, closure capture, or a broken import->store
    adjacency) makes the receiver provenance ambiguous and BOTH
    exclusivity flags stay False, so the caller fails closed.

    The scan additionally proves the CALL-SITE USE: every direct
    ``LOAD_ATTR``/``LOAD_METHOD "from_pretrained"`` in this code object
    is located and the instruction immediately feeding it (only the
    harmless EXTENDED_ARG/NOP/CACHE interpreter bookkeeping is skipped)
    must be a load of the proven binding -- ``LOAD_FAST``/
    ``LOAD_FAST_CHECK "AutoModelForCausalLM"`` for the deferred-local
    seam (LOAD_FAST_CHECK is the same local read plus an unboundedness
    check; no broader receiver pattern is accepted), or ``LOAD_GLOBAL``
    /``LOAD_NAME`` of the name for the module-global seam.
    ``callsite_proof_deferred`` / ``callsite_proof_global`` are True
    only when the corresponding exclusive binding is proven AND at
    least one from_pretrained site exists AND every site's receiver is
    that binding. A site with any other receiver (e.g.
    ``other.AutoModelForCausalLM.from_pretrained``), a mix of receiver
    kinds, or zero sites leaves both flags False, so the caller fails
    closed BEFORE module.main(). Comments, docstrings, helper
    functions and unrelated scopes emit none of these instructions in
    this code object.
    """
    import dis

    name = "AutoModelForCausalLM"
    store_ops = {"STORE_FAST", "STORE_NAME", "STORE_GLOBAL", "STORE_DEREF"}
    delete_ops = {"DELETE_FAST", "DELETE_NAME", "DELETE_GLOBAL", "DELETE_DEREF"}
    skip_ops = {"EXTENDED_ARG", "NOP", "CACHE"}
    local_load_ops = {"LOAD_FAST", "LOAD_FAST_CHECK"}
    global_load_ops = {"LOAD_GLOBAL", "LOAD_NAME"}

    real_ins = [
        ins for ins in dis.get_instructions(code) if ins.opname not in skip_ops
    ]

    approved_imports = 0
    foreign_imports = 0
    approved_stores = 0
    extra_stores = 0
    deletes = 0
    import_star = False
    loads_global = False
    current_import = None
    pending_approved_store = False
    adjacency_broken = False

    for ins in real_ins:
        op = ins.opname
        if pending_approved_store:
            pending_approved_store = False
            if op in store_ops and ins.argval == name:
                approved_stores += 1
                continue  # consumed as the approved import's own store
            adjacency_broken = True
        if op == "IMPORT_NAME":
            current_import = ins.argval
        elif op == "IMPORT_STAR":
            import_star = True
        elif op == "IMPORT_FROM" and ins.argval == name:
            if current_import == "transformers":
                approved_imports += 1
                pending_approved_store = True
            else:
                foreign_imports += 1
        elif op in store_ops and ins.argval == name:
            extra_stores += 1
        elif op in delete_ops and ins.argval == name:
            deletes += 1
        elif op in global_load_ops and ins.argval == name:
            loads_global = True
    if pending_approved_store:
        adjacency_broken = True

    # ---- call-site receiver proof for from_pretrained ----
    fp_receivers: list[dict] = []
    fp_approved_local = 0
    fp_approved_global = 0
    fp_rejected: list[dict] = []
    for i, ins in enumerate(real_ins):
        if ins.opname in ("LOAD_ATTR", "LOAD_METHOD") and ins.argval == "from_pretrained":
            recv = real_ins[i - 1] if i > 0 else None
            entry = {
                "receiver_opname": recv.opname if recv is not None else None,
                "receiver_name": str(recv.argval) if recv is not None else None,
            }
            fp_receivers.append(entry)
            if (
                recv is not None
                and recv.opname in local_load_ops
                and recv.argval == name
            ):
                fp_approved_local += 1
            elif (
                recv is not None
                and recv.opname in global_load_ops
                and recv.argval == name
            ):
                fp_approved_global += 1
            else:
                fp_rejected.append(entry)
    fp_total = len(fp_receivers)

    cell_or_free = name in (tuple(code.co_cellvars) + tuple(code.co_freevars))
    no_ambiguity = (
        foreign_imports == 0
        and extra_stores == 0
        and deletes == 0
        and not import_star
        and not cell_or_free
        and not adjacency_broken
    )
    exclusive_deferred = (
        approved_imports >= 1
        and approved_stores == approved_imports
        and no_ambiguity
    )
    exclusive_global_read = (
        loads_global
        and approved_imports == 0
        and approved_stores == 0
        and no_ambiguity
    )
    callsite_proof_deferred = (
        exclusive_deferred
        and fp_total >= 1
        and not fp_rejected
        and fp_approved_global == 0
        and fp_approved_local == fp_total
    )
    callsite_proof_global = (
        exclusive_global_read
        and fp_total >= 1
        and not fp_rejected
        and fp_approved_local == 0
        and fp_approved_global == fp_total
    )
    return {
        "deferred_from_transformers_import": approved_imports >= 1,
        "loads_global": loads_global,
        "approved_transformers_imports": approved_imports,
        "approved_import_stores": approved_stores,
        "foreign_imports_of_name": foreign_imports,
        "extra_stores_of_name": extra_stores,
        "deletes_of_name": deletes,
        "import_star_present": import_star,
        "name_is_cell_or_free": cell_or_free,
        "import_store_adjacency_broken": adjacency_broken,
        "exclusive_deferred_binding": exclusive_deferred,
        "exclusive_global_read": exclusive_global_read,
        "from_pretrained_sites_total": fp_total,
        "from_pretrained_sites_approved_local": fp_approved_local,
        "from_pretrained_sites_approved_global": fp_approved_global,
        "from_pretrained_receivers": fp_receivers,
        "from_pretrained_rejected_sites": fp_rejected,
        "callsite_proof_deferred": callsite_proof_deferred,
        "callsite_proof_global": callsite_proof_global,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-bin", required=True)
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--proof-json", required=True)
    ns = ap.parse_args()

    # ---- environment fail-closes BEFORE any heavy work ----
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        stop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set - refusing to run")
    if importlib.util.find_spec("kernels") is not None:
        stop("'kernels' package importable - RMSNorm kernel-swap hazard")

    if not ORIGINAL_CAPTURE.is_file():
        stop(f"original validated capture script missing: {ORIGINAL_CAPTURE}")

    original_sha = sha256_file(ORIGINAL_CAPTURE)
    print(f"original_capture_sha256={original_sha}")
    if original_sha != EXPECTED_ORIGINAL_SHA256:
        stop(
            "validated Gate-3 capture SHA mismatch; expected "
            f"{EXPECTED_ORIGINAL_SHA256}, got {original_sha}"
        )

    tokens_bin = Path(ns.tokens_bin).resolve()
    if not tokens_bin.is_file():
        stop(f"token file missing: {tokens_bin}")

    token_sha = sha256_file(tokens_bin)
    print(f"tokens_bin_sha256={token_sha}")
    if token_sha != EXPECTED_TOKEN_SHA256:
        stop(
            "authoritative 2050-token SHA mismatch; expected "
            f"{EXPECTED_TOKEN_SHA256}, got {token_sha}"
        )

    raw = tokens_bin.read_bytes()
    if len(raw) != EXPECTED_TOKEN_COUNT * 4:
        stop(
            f"unexpected token file length: {len(raw)} bytes; "
            f"expected {EXPECTED_TOKEN_COUNT * 4}"
        )

    input_ids = list(struct.unpack(f"<{EXPECTED_TOKEN_COUNT}i", raw))
    if len(input_ids) != EXPECTED_TOKEN_COUNT:
        stop(
            f"unexpected token count: {len(input_ids)} "
            f"!= {EXPECTED_TOKEN_COUNT}"
        )
    for i, token_id in enumerate(input_ids):
        if not 0 <= token_id < VOCAB_SIZE:
            stop(f"token {i} out of range: {token_id}")
        if token_id != EXPECTED_TOKEN_ID:
            stop(
                f"token {i} != frozen id {EXPECTED_TOKEN_ID}: {token_id} "
                "(this wrapper is pinned to the 2050x483 stream)"
            )

    print(f"sequence_length={len(input_ids)}")
    print(f"first_8_ids={input_ids[:8]}")
    print(f"last_8_ids={input_ids[-8:]}")
    print(f"sys_executable={sys.executable}")

    # ---- version + base-module gates (same sys.modules the core uses) ----
    try:
        import torch
        import transformers
    except Exception as exc:  # noqa: BLE001 - fail-closed report
        stop(f"failed to import torch/transformers: {exc}")

    if torch.__version__ != EXPECTED_TORCH_VERSION:
        stop(
            f"torch version {torch.__version__} != frozen "
            f"{EXPECTED_TORCH_VERSION}"
        )
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        stop(
            f"transformers version {transformers.__version__} != frozen "
            f"{EXPECTED_TRANSFORMERS_VERSION}"
        )

    tf_root = Path(transformers.__file__).resolve().parent
    base_modeling = tf_root / "models" / "longcat_flash" / "modeling_longcat_flash.py"
    rope_utils = tf_root / "modeling_rope_utils.py"
    for p, expected, what in (
        (base_modeling, EXPECTED_BASE_MODELING_SHA256, "base modeling_longcat_flash.py"),
        (rope_utils, EXPECTED_ROPE_UTILS_SHA256, "modeling_rope_utils.py"),
    ):
        if not p.is_file():
            stop(f"{what} missing: {p}")
        got = sha256_file(p)
        print(f"{p.name}_sha256={got}")
        if got != expected:
            stop(f"{what} SHA mismatch; expected {expected}, got {got}")

    # Wrapper-side TF32 hardening. The frozen core sets matmul.allow_tf32
    # itself; cudnn is set here (non-arithmetic configuration, both knobs
    # required by the round's environment contract).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ---- exec the frozen core ----
    spec = importlib.util.spec_from_file_location(
        "validated_gate3_capture",
        ORIGINAL_CAPTURE,
    )
    if spec is None or spec.loader is None:
        stop("could not construct import spec for validated Gate-3 capture")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Replace only the validated script's hard-coded four-token oracle input.
    module.INPUT_IDS = input_ids

    # ---- sparse-engagement observation shim (observation-only) ----
    collector: list[dict] = []
    meta: dict = {"installed": False, "install_error": None}

    def _owner0_battery(topk) -> dict:
        """Full approved structural battery, observation-only: read-only
        torch reductions on the production top-K tensor; primitives out."""
        import torch as _t

        t = topk[0]  # [2050, 2048] int64, production tensor (never mutated)
        out: dict = {}
        pos = _t.arange(t.shape[0], device=t.device, dtype=t.dtype)
        out["range_min"] = int(t.min().item())
        out["range_max"] = int(t.max().item())
        out["range_ok"] = bool(
            out["range_min"] >= -1 and out["range_max"] < EXPECTED_TOKEN_COUNT
        )
        neg1 = (t == -1).sum(dim=1)
        expected = _t.clamp(EXPECTED_INDEX_TOPK - (pos + 1), min=0)
        out["filler_counts_exact"] = bool(_t.equal(neg1, expected))
        if not out["filler_counts_exact"]:
            bad = (neg1 != expected).nonzero()
            out["filler_first_bad_row"] = int(bad[0, 0].item())
        out["fillers_2047_2049_zero"] = bool(
            int(neg1[2047].item()) == 0
            and int(neg1[2048].item()) == 0
            and int(neg1[2049].item()) == 0
        )
        s, _ = _t.sort(t, dim=1)
        dup = (s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)
        out["unique_nonneg"] = not bool(dup.any().item())
        causal_bad = (t > pos[:, None]) & (t >= 0)
        out["causal_only"] = not bool(causal_bad.any().item())
        forced_ok = {}
        for p in (2048, 2049):
            row = t[p]
            needed = _t.cat(
                (
                    _t.arange(0, INIT_TOKENS, device=t.device, dtype=t.dtype),
                    _t.arange(
                        p - LOCAL_TOKENS + 1, p + 1, device=t.device, dtype=t.dtype
                    ),
                )
            )
            forced_ok[str(p)] = bool(_t.isin(needed, row).all().item())
        out["forced_containment"] = forced_ok
        return out

    def _record_hook(layer_idx: int, sub: int):
        def hook(mod, args, kwargs, output):
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
                topk = output[2] if isinstance(output, tuple) and len(output) >= 3 else None
                rec["topk_is_none"] = topk is None
                if topk is not None:
                    rec["topk_shape"] = [int(x) for x in topk.shape]
                    rec["topk_dtype"] = str(topk.dtype)
                    if layer_idx == 0 and sub == 0:
                        rec["owner0_battery"] = _owner0_battery(topk)
            except Exception as exc:  # noqa: BLE001 - recorded, validated later
                rec["error"] = f"{type(exc).__name__}: {exc}"
            collector.append(rec)
            return None

        return hook

    def _install_observers(model) -> None:
        try:
            trunk = model.model
            layers = list(trunk.layers)
            meta["n_layers"] = len(layers)
            for i, layer in enumerate(layers):
                for sub in (0, 1):
                    layer.self_attn[sub].register_forward_hook(
                        _record_hook(i, sub), with_kwargs=True
                    )
            meta["installed"] = True
        except Exception as exc:  # noqa: BLE001 - recorded, validated later
            meta["install_error"] = f"{type(exc).__name__}: {exc}"

    # ---- interception-seam proof (fail-closed BEFORE any GPU work) ----
    # Prove, from module.main's OWN executable bytecode, how main() can
    # bind AutoModelForCausalLM; identity-gate and patch exactly the
    # proven binding(s); restore every patched binding in finally. A raw
    # source-substring is NOT accepted as proof (a comment, docstring,
    # helper or unrelated scope could satisfy it).
    real_auto = transformers.AutoModelForCausalLM
    if not hasattr(real_auto, "from_pretrained"):
        stop("transformers.AutoModelForCausalLM lacks from_pretrained")

    main_fn = getattr(module, "main", None)
    if not callable(main_fn) or not hasattr(main_fn, "__code__"):
        stop("core module.main is not an inspectable python function")
    if main_fn.__globals__ is not module.__dict__:
        stop(
            "core main().__globals__ is not the core module namespace - "
            "cannot reason about its global bindings; refusing to run"
        )

    scan = _scan_main_bindings(main_fn.__code__)
    module_global = module.__dict__.get("AutoModelForCausalLM", None)

    seams: list[str] = []
    if scan["callsite_proof_global"]:
        if module_global is None:
            stop(
                "core main() bytecode reads a global AutoModelForCausalLM "
                "but no such module-global exists after exec_module - "
                "the seam cannot be proven; refusing to run"
            )
        if module_global is not real_auto:
            stop(
                "core module-global AutoModelForCausalLM is not identical "
                "to transformers.AutoModelForCausalLM - refusing to patch "
                "an unproven binding"
            )
        seams.append("core_module_global")
    if scan["callsite_proof_deferred"]:
        seams.append("transformers_module_attribute")
    if not seams:
        stop(
            "the binding actually used by the from_pretrained call site(s) "
            "in module.main cannot be proven to be an EXCLUSIVE approved "
            "AutoModelForCausalLM seam (no approved seam; a competing "
            "import/rebinding store/deletion/import-star/closure-capture/"
            "broken import-store adjacency; zero from_pretrained sites; or "
            "a from_pretrained site whose receiver is not the proven "
            f"binding; scan={scan}) - refusing to run"
        )
    meta["interception_seams"] = seams
    meta["seam_bytecode_scan"] = scan
    meta["core_module_global_present"] = module_global is not None
    meta["main_globals_is_module_dict"] = True
    meta["from_pretrained_calls"] = 0
    print(f"interception_seams={seams}")
    print(f"seam_bytecode_scan={scan}")

    class _ObservingAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            meta["from_pretrained_calls"] += 1
            model = real_auto.from_pretrained(*args, **kwargs)
            _install_observers(model)
            return model

    sys.argv = [
        str(ORIGINAL_CAPTURE),
        "--model-dir",
        ns.model_dir,
        "--out-bin",
        ns.out_bin,
        "--out-json",
        ns.out_json,
    ]

    try:
        if "core_module_global" in seams:
            module.AutoModelForCausalLM = _ObservingAutoModelForCausalLM
        if "transformers_module_attribute" in seams:
            transformers.AutoModelForCausalLM = _ObservingAutoModelForCausalLM
        rc = int(module.main())
    finally:
        if "transformers_module_attribute" in seams:
            transformers.AutoModelForCausalLM = real_auto
        if "core_module_global" in seams:
            module.AutoModelForCausalLM = real_auto

    print(f"core_rc={rc}")

    # ---- post-main fail-closed engagement validation ----
    failures: list[str] = []

    if not meta["installed"]:
        failures.append(
            "observation shim never installed "
            f"(install_error={meta['install_error']!r})"
        )
    if meta.get("from_pretrained_calls") != 1:
        failures.append(
            "observing from_pretrained fired "
            f"{meta.get('from_pretrained_calls')}x != exactly 1 "
            "(interception seam not exercised as proven)"
        )
    if meta.get("n_layers") != EXPECTED_NUM_LAYERS:
        failures.append(f"unexpected layer count: {meta.get('n_layers')}")
    if len(collector) != 2 * EXPECTED_NUM_LAYERS:
        failures.append(
            f"collector has {len(collector)} records, expected "
            f"{2 * EXPECTED_NUM_LAYERS} (empty/partial collector is a "
            "hard failure)"
        )

    seen = set()
    for rec in collector:
        key = (rec.get("layer"), rec.get("sublayer"))
        if key in seen:
            failures.append(f"duplicate observation record for {key}")
        seen.add(key)
        tag = f"layer {rec.get('layer')} sublayer {rec.get('sublayer')}"
        if "error" in rec:
            failures.append(f"{tag}: hook error {rec['error']}")
            continue
        if rec.get("seq_len") != EXPECTED_TOKEN_COUNT:
            failures.append(f"{tag}: seq_len {rec.get('seq_len')} != 2050")
        expected_mode = OWNER_MODE if rec.get("sublayer") == 0 else REUSE_MODE
        if rec.get("mode") != expected_mode:
            failures.append(
                f"{tag}: mode {rec.get('mode')!r} != {expected_mode!r}"
            )
        if rec.get("valid_topk_range") != list(EXPECTED_VALID_TOPK_RANGE):
            failures.append(
                f"{tag}: valid_topk_range {rec.get('valid_topk_range')} "
                f"!= {list(EXPECTED_VALID_TOPK_RANGE)}"
            )
        if rec.get("topk_is_none"):
            failures.append(f"{tag}: top-K tensor is None (dense signature)")
            continue
        if rec.get("topk_shape") != [1, EXPECTED_TOKEN_COUNT, EXPECTED_INDEX_TOPK]:
            failures.append(f"{tag}: topk_shape {rec.get('topk_shape')}")
        if rec.get("topk_dtype") != "torch.int64":
            failures.append(f"{tag}: topk_dtype {rec.get('topk_dtype')}")
        if rec.get("layer") == 0 and rec.get("sublayer") == 0:
            bat = rec.get("owner0_battery") or {}
            if not bat:
                failures.append("owner0: structural battery missing")
            for key in (
                "range_ok",
                "filler_counts_exact",
                "fillers_2047_2049_zero",
                "unique_nonneg",
                "causal_only",
            ):
                if bat.get(key) is not True:
                    failures.append(
                        f"owner0 battery {key}={bat.get(key)!r} != True"
                    )
            forced = bat.get("forced_containment") or {}
            for p in ("2048", "2049"):
                if forced.get(p) is not True:
                    failures.append(
                        f"owner0 battery forced containment row {p}: "
                        f"{forced.get(p)}"
                    )

    for key in sorted(
        {(i, s) for i in range(EXPECTED_NUM_LAYERS) for s in (0, 1)} - seen
    ):
        failures.append(f"missing observation record for layer/sublayer {key}")

    if rc != 0:
        failures.append(f"frozen core returned rc={rc}")

    proof = {
        "purpose": (
            "Run B independent sparse-engagement proof (observation-only "
            "shim around the byte-frozen Gate-3 core). NOT a Gate-4 "
            "criterion; no C++ comparison."
        ),
        "original_capture_sha256": original_sha,
        "tokens_bin_sha256": token_sha,
        "sequence_length": EXPECTED_TOKEN_COUNT,
        "expected_owner_mode": OWNER_MODE,
        "expected_reuse_mode": REUSE_MODE,
        "expected_valid_topk_range": list(EXPECTED_VALID_TOPK_RANGE),
        "collector": collector,
        "meta": meta,
        "core_rc": rc,
        "failures": failures,
        "engagement_proof": "PASS" if not failures else "FAIL",
    }
    proof_path = Path(ns.proof_json).resolve()
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(f"proof_json={proof_path}")

    if failures:
        for f in failures:
            print(f"ENGAGEMENT-FAIL: {f}")
        print(
            "HF 2050 RAW-IDS CAPTURE: REJECTED "
            "(sparse-engagement proof failed; canonical artifact NOT banked)"
        )
        return 86

    print("sparse_engagement_proof=PASS (14 owners + 14 reuse, all gates)")
    print("HF 2050 RAW-IDS CAPTURE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
