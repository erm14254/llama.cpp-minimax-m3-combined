"""P-CF1 v1 -- Captured-Surface Offline Counterfactual Membership Sensitivity.

Implements the accepted, frozen P-CF1 protocol (design Revision 4; Stage-P
design freeze). Owner00 ONLY; rows 2048 and 2049 ONLY; frozen 2050-token
capture state ONLY.

Scientific question (frozen): does substituting (i) the indexer
scoring-weight surface alone, (ii) the layout-mapped post-RoPE Q surface
alone, or (iii) both together, from one side's captured tensors into the
other side's otherwise-unchanged captured scoring inputs, change the
TOPK=2048 membership SET computed by the registered offline reconstruction,
relative to the banked receiving-side owner00 set at that row -- given that
the same reconstruction first reproduces that banked baseline exactly?

AUTHORITY BOUNDARY (frozen): every result of this protocol is a statement
about the registered offline reconstruction ONLY. Whether an offline
counterfactual effect (or null) would reproduce the corresponding
production-engine counterfactual is explicitly UNRESOLVED and out of scope.
No production-level causal membership claim is licensed by any outcome.
Baseline reproduction is a NECESSARY validity gate; it is NOT sufficient
for production-oracle authority on synthetic off-baseline branches.

Counterfactual matrix (K is NEVER substituted; receiving-frame K held fixed):
  Direction C (receiving frame = C++ captured state; uses registered pi^-1):
    CF-C0 baseline: Q_cpp,          W_cpp_eff, K_cpp   (gate branch)
    CF-C1 weights:  Q_cpp,          W_hf_eff,  K_cpp
    CF-C2 Q:        pi^-1(Q_hf),    W_cpp_eff, K_cpp
    CF-C3 both:     pi^-1(Q_hf),    W_hf_eff,  K_cpp
  Direction H (receiving frame = HF captured state; uses registered pi):
    CF-H0 baseline: Q_hf,           W_hf_eff,  K_hf    (gate branch)
    CF-H1 weights:  Q_hf,           W_cpp_eff, K_hf
    CF-H2 Q:        pi(Q_cpp),      W_hf_eff,  K_hf
    CF-H3 both:     pi(Q_cpp),      W_cpp_eff, K_hf
Effective weights only: W_cpp_eff = captured post-scale tensor unchanged;
W_hf_eff = hf prescale * f32(1/sqrtf(2048)) (the frozen surface-5 mapping).
The prescale<->effective mapping is never inverted.

Registered full-Q lane mappings (assignment/permutation only; per head
[128]; no arithmetic; no rounding):
  pi   (C++ interleaved -> HF half-split):
    dst[0:32]=src[0:64:2]; dst[32:64]=src[1:64:2]; dst[64:128]=src[64:128]
  pi^-1 (HF half-split -> C++ interleaved):
    dst[0:64:2]=src[0:32]; dst[1:64:2]=src[32:64]; dst[64:128]=src[64:128]

Frozen offline scorer (verbatim from the Amendment-1 comparator; trig-free;
consumes captured post-RoPE tensors as opaque inputs; introduces NO trig,
NO angle decoding, NO threshold derived from observed CUDA discrepancies):
  score[j] = sum_h w[h] * relu(dot128(q[h,:], k[j,:])), float64, j in [0, p];
  forced {0..15} u {p-1023..p} -> +inf; stable argsort; top-2048 as a SET.
Primary verdict is determined by membership-SET identity ONLY:
  MEMBERSHIP-INVARIANT / MEMBERSHIP-AFFECTING (n positions).
Boundary margin, MARGIN_FLOOR_REL-derived floor, below-floor boolean and
the COMPLETE entering/dropped/symmetric-difference lists are OBSERVATIONAL
ONLY and never modify a verdict. No BORDERLINE verdict exists in P-CF1.
The only aggregate descriptor is JOINT-ONLY MEMBERSHIP EFFECT (direction D)
for exactly: CF-D1 and CF-D2 MEMBERSHIP-INVARIANT at all adjudicated rows
while CF-D3 is MEMBERSHIP-AFFECTING at any row, with a VALID baseline and
no tie-unadjudicated branch/row in the pattern.

Failure taxonomy (frozen; four disjoint classes; precedence):
  argument parsing (missing required args -> non-scientific exit 1,
  nothing created, no verdict)
  -> CLASS R  REFUSED TO START (exit 3): resolved --out-dir already exists;
     checked BEFORE output creation and before reading or hash-checking any
     scientific input; the existing directory is left completely untouched;
     no verdict is written.
  -> registered Stage-E absolute path contract (mismatch -> non-scientific
     exit 1, nothing created, no verdict)
  -> create fresh output directory
  -> CLASS G  GLOBAL ABORT (exit 1): evidence/container/implementation
     integrity failure (SHA/manifest/provenance mismatch; wrong
     size/shape/carrier; banked top-K carrier not exactly integral;
     BF16-lattice violation on ANY of the four captured post-RoPE Q/K
     surfaces (C++ Q, C++ K, HF Q, HF K); non-finite loaded scoring value
     or non-finite pre-mask reconstructed score in ANY branch (enforced by
     the registered post-scorer finite_max gate before any tie or verdict
     interpretation); environment/backend binding failure; mapping
     self-test failure;
     unexpected exception). verdict.json is preserved with reasons/anomaly;
     under the frozen late-abort policy NO substituted result from an
     aborted run retains an adjudicated verdict (already-computed entries
     become NOT ADJUDICATED (GLOBAL ABORT) with raw measurements retained
     as diagnostic_only data).
  -> CLASS D  DIRECTION INVALIDATION: begins only after all CLASS-G gates
     pass; direction-local semantic validity of the otherwise well-formed
     banked/reconstructed owner00 sets (reconstructed != banked; set size
     != 2048; duplicates; index negative where prohibited / > p; forced-set
     non-containment; baseline TIE-AT-CUT) => BASELINE RECONSTRUCTION
     INVALID (direction D); CF-D1/2/3 all NOT ADJUDICATED; the other
     direction proceeds only on its own gate.
  -> CLASS B  BRANCH/ROW NON-ADJUDICATION: after a VALID baseline, an exact
     float64 score tie spanning the TOPK cut in a substituted branch makes
     only that branch/row NOT ADJUDICATED (TIE-AT-CUT). The stable-argsort
     order keeps the computation deterministic but never grants causal
     authority at a tie. Tie rows carry the complete frozen observational
     schema (margins and the complete entering/dropped/symmetric-difference
     lists) under diagnostic_only, strictly subordinate to the
     non-adjudicating verdict.

Environment/backend contract (frozen; single-thread for repeatability, not
a relaxation of any gate): the five registered thread variables are set to
"1" BEFORE numpy is imported (see the module top); NumPy must be exactly
2.5.2 with the bundled scipy-openblas64 backend; the registered OpenBLAS
DLL must exist and hash to its frozen SHA256; the registered interpreter is
the repo venv Python 3.12.10. torch/transformers/CUDA are never imported.

CLI (frozen):
  analyze_longcat_lsa_cf1_owner00_v1.py
    --cpp-s-dir <path> --hf-dir <path> --hf-run-dir <path> --out-dir <path>
    [--self-test]
--self-test is completely synthetic/helper-only and never accesses the real
capture bank. For the future authorized Stage-E run the four paths must
resolve exactly to the registered absolute paths recorded below.

Stage-E execution requires a separate, explicit, exactly-once human
authorization binding the authoring commit SHA, a clean tracked tree, the
reviewed checkout-form script SHA256, the frozen input/evidence chain and
this environment contract. No rerun is implicitly authorized by an abort or
by successful completion.

Nothing here adjudicates blocker-1 membership impact or blocker-3
angle-swap membership impact (both remain NOT ADJUDICATED); owners 02..26
remain observational only; the prohibited oracle functions (ggml_angles,
hf_angles, rotate_cpp_class) and pair_decode/pair_norm_ratio are absent.
This script makes NO Gate-4 statement; Gate 4 remains NOT RUN.
"""
from __future__ import annotations

# =====================================================================
# REGISTERED ENVIRONMENT CONTRACT -- the five thread-control variables
# are assigned BEFORE `import numpy` so the bundled scipy-openblas64
# backend is single-threaded for the whole process lifetime. This
# ordering is load-bearing and is verified statically at Stage A.
# =====================================================================
import os

THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _var in THREAD_ENV_VARS:
    os.environ[_var] = "1"

import argparse
import hashlib
import json
import platform
import re
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------- frozen scorer constants (verbatim values) ----------------

N_TOKENS = 2050
TOPK = 2048
HEAD_DIM = 128
ROPE_DIM = 64
N_HEADS = 16
INIT_TOKENS = 16
LOCAL_TOKENS = 1024
MARGIN_FLOOR_REL = 1e-4

ROWS = (2048, 2049)

# ---------------- registered Stage-E absolute path contract ----------------

REGISTERED_CPP_S_DIR = r"D:\llama.cpp-longcat-claude\cpp_lsa_2050_S1"
REGISTERED_HF_DIR = r"D:\llama.cpp-longcat-claude\hf_lsa_2050_capture"
REGISTERED_HF_RUN_DIR = r"D:\llama.cpp-longcat-claude\hf_logits_2050_v1"
REGISTERED_OUT_DIR = r"D:\llama.cpp-longcat-claude\lsa_cf1_owner00_v1"

# ---------------- frozen environment/backend binding ----------------

REGISTERED_PYTHON = r"D:\llama.cpp-longcat-claude\.venv\Scripts\python.exe"
EXPECTED_PY_VERSION = "3.12.10"
EXPECTED_NUMPY_VERSION = "2.5.2"
REGISTERED_OPENBLAS_DLL = (
    r"D:\llama.cpp-longcat-claude\.venv\Lib\site-packages\numpy.libs"
    r"\libscipy_openblas64_-327b2e0bcffce2882e0dc04cdeb4eaa6.dll"
)
OPENBLAS_DLL_BYTES = 20495360
OPENBLAS_DLL_SHA256 = (
    "327b2e0bcffce2882e0dc04cdeb4eaa6382ed4dba29871b6176fa157782f484a"
)
EXPECTED_BACKEND_SUBSTRING = "scipy-openblas"

# ---------------- frozen evidence-chain constants ----------------

EXPECTED_CPP_GIT_HEAD = "2dd49d39c11a4378ebd3abed2a51aea3f575accb"
EXPECTED_TOKEN_SHA256 = (
    "eb04e101e452e3bb60911f02b5b5ac538e8a183efcb564d7c9859d7e03266bed"
)
EXPECTED_RUNTIME_SHA256 = (
    "a3bc31616c1f0ddff9f195cbc78f4561a40187c50fe3bf29e2e98d9228947428"
)
EXPECTED_CORE_SHA256 = (
    "bb82bcb6c3bc1d21685221a884dac3b39dc7af06f54fea6187f606dddf4213cb"
)
EXPECTED_RUNB_SCRIPT_SHA256 = (
    "3d5b93316d82aab7d19237b5c98de1251bcfd7fbf1c3bfaa915475762affdea9"
)
EXPECTED_RUNA_SCRIPT_SHA256 = (
    "18fcc5e191e39bf23489e4848ad6ec7659c638c341cd8a919b96c36bd9b9e18f"
)
PARENT_CMP_SHA256 = (
    "0b9206426182ea3810136afda938b5c689a1a28cba780a59fbe7ec5bd4bd4e45"
)
ORIGINAL_FAILED_VERDICT_SHA256 = (
    "6b1b296a6bc2fd8dba77b08db3411010793959fe11676ef95b749970ccf41295"
)
DIAG_V1_SHA256 = (
    "73189745853ceeb3dfa14f3ea4fefd5335a437c04b64ad9f61c489094eda781c"
)
DIAG_V2_SHA256 = (
    "242da66e05b44119e47c6c8e601ea015b7de6b5d1dcdc90ac9680027b9a8b553"
)
DIAG_V3_SHA256 = (
    "6b90ef0870139715a5ede06d703523d48bac78ac32a85f816c86a2d8d70a0743"
)
AMEND1_CMP_SHA256 = (
    "f36bb95d828f4bb3437bdcf4a533547bc1b1ff75c381fd5f00093bdfb12fefec"
)
AMEND1_VERDICT_SHA256 = (
    "4cfffea98ecaa3389a3cf40e0f06b0aeee7b5cd2cbe03273970097e3ca044c8d"
)

CPP_MANIFEST_SHA256 = (
    "edec84b40287fd960373864c4f1088f65d36052308e05f0cf9687eb2d75705e1"
)
HF_MANIFEST_SHA256 = (
    "3806a4de7613b4e1a007c4d8cdd010348561d315f5c3d46d2d0e0ba834383322"
)

# Frozen byte-stable C++ S-family surfaces (verbatim constant table from
# the Amendment-1 comparator; byte-identical across S1/S2/S3).
CPP_STABLE_SHA256 = {
    "lsa_anchor_attn_norm0_full.bin": "28f15cb7fb59b64d6ee565155e6b29cbbb66e297f362f9a66b4c8d1245d3c046",
    "lsa_anchor_q_a_norm0_full.bin": "49d3d02d70705cad4ca371e535933cd2cea5651d8b02230ae2e303f68ac01d2e",
    "lsa_indexer_k_proj_full.bin": "5ddd67d8c0085aa63443b6f75b837004914097464a3815192d1961e11d4b0e96",
    "lsa_indexer_k_norm_full.bin": "57deb53cbb34ef3a14f36d4672ef3c269f9ae93138fe7cca55d12bf8186880e6",
    "lsa_indexer_k_full.bin": "2f57bc0f5d39534e3e55c17a7c59581476fd394dcb3a26a03e0723b99be1cf75",
    "lsa_indexer_q_proj_full.bin": "a75dfb8070dba5682cb31031dde3e6d29e02fac516946daff633117dce9dd7f3",
    "lsa_indexer_q_full.bin": "cc8ecd1a56a53f57c365d3cde552712d9b5c2b88cd013593e6f470458c6ccfd6",
    "lsa_indexer_weights_full.bin": "321a15e612641287a1ad5456ea5d5b9af2ee5134e2155ffba17a4593270d741e",
}

CPP_WIDTHS = {
    "lsa_anchor_attn_norm0_full.bin": 3072,
    "lsa_anchor_q_a_norm0_full.bin": 1536,
    "lsa_indexer_k_proj_full.bin": 128,
    "lsa_indexer_k_norm_full.bin": 128,
    "lsa_indexer_k_full.bin": 128,
    "lsa_indexer_q_proj_full.bin": 2048,
    "lsa_indexer_q_full.bin": 2048,
    "lsa_indexer_weights_full.bin": 16,
}

HF_WIDTHS = {
    "hf_attn_norm0.bin": 3072,
    "hf_q_a_norm0.bin": 1536,
    "hf_indexer_k_proj.bin": 128,
    "hf_indexer_k_norm.bin": 128,
    "hf_indexer_k.bin": 128,
    "hf_indexer_q_proj.bin": 2048,
    "hf_indexer_q.bin": 2048,
    "hf_indexer_weights_prescale.bin": 16,
    "hf_rope_cos.bin": 64,
    "hf_rope_sin.bin": 64,
}

HF_WEIGHT_BINS = {
    "hf_weight_k_norm.bin": 128,
    "hf_weight_wk.bin": 128 * 3072,
    "hf_weight_wq_b.bin": 2048 * 1536,
    "hf_weight_weights_proj.bin": 16 * 3072,
}

# P-CF1 SS2 scoring-input constants: the exact frozen SHA of every artifact
# this protocol loads as a tensor (defense in depth on top of the manifest
# rehash; the three C++ stable surfaces repeat CPP_STABLE_SHA256 values).
CPP_TOPK_OWNER00_SHA256 = (
    "5f8835a2a39b5b003269635f68d4db80376a89fd708dddefd1f7762610f19448"
)
HF_INPUT_SHA256 = {
    "hf_indexer_q.bin": "fd9d2e6afdb829d5784d32f84985ca276df4b7e42e4add06d97acdefb6c3e6aa",
    "hf_indexer_k.bin": "86428e54191c431381450a7d1fcc8b01facd2af1ffdd85ecad0a4acade8590a6",
    "hf_indexer_weights_prescale.bin": "78f3327dcc2f251538886d56215ab9b84c81caf76aabd0262211d6e4c7d3f1a2",
    "hf_top_k_owner00.bin": "c8b3b8edbee7a3f56c5d1b5b935ff081b1079085140b7f8bc6382291527989a7",
}

CPP_LOADED = (
    "lsa_indexer_q_full.bin",
    "lsa_indexer_k_full.bin",
    "lsa_indexer_weights_full.bin",
)
HF_LOADED = (
    "hf_indexer_q.bin",
    "hf_indexer_k.bin",
    "hf_indexer_weights_prescale.bin",
)

REASONS: list[str] = []


def stop(msg: str) -> None:
    REASONS.append(msg)
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- bf16 / bitwise helpers (verbatim) ----------------

def _u32(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="<f4")
    return x.view(np.uint32)


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise equality of the widened-F32 carriers (signed zero is NOT
    normalized: -0.0 != +0.0 here)."""
    ua, ub = _u32(a), _u32(b)
    return ua.shape == ub.shape and bool(np.array_equal(ua, ub))


def bits_diff_count(a: np.ndarray, b: np.ndarray) -> int:
    return int((_u32(a) != _u32(b)).sum())


def rne_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> bf16 bit pattern (finite inputs)."""
    u = _u32(x).astype(np.uint64)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return r.astype(np.uint16)


def widen_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view("<f4")


def rne_bf16(x: np.ndarray) -> np.ndarray:
    return widen_bits(rne_bf16_bits(x))


def load_f32(path: Path, width: int) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype="<f4")
    if data.size != N_TOKENS * width:
        stop(f"{path.name}: size {data.size} != {N_TOKENS}x{width}")
    return data.reshape(N_TOKENS, width)


def parse_sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        stop(f"manifest missing: {path}")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        m = re.match(r"^([0-9a-f]{64})\s+(.+)$", line.strip())
        if not m:
            stop(f"malformed manifest line in {path.name}: {line!r}")
        name = m.group(2)
        if name in entries:
            stop(f"duplicate manifest entry in {path.name}: {name}")
        entries[name] = m.group(1)
    return entries


def read_json(path: Path) -> dict:
    """BOM-tolerant JSON read (PowerShell 5.1 Out-File -Encoding utf8
    writes a BOM; utf-8-sig also reads BOM-less files)."""
    if not path.is_file():
        stop(f"json missing: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def f32(x) -> np.float32:
    return np.float32(x)


# ---------------- frozen 64-lane RoPE-block mappings (verbatim) ----------------

def pi_map(cpp_roped: np.ndarray) -> np.ndarray:
    """Candidate permutation pi: HF[i] = C++[2i], HF[32+i] = C++[2i+1]."""
    out = np.empty_like(cpp_roped)
    out[:, : ROPE_DIM // 2] = cpp_roped[:, 0:ROPE_DIM:2]
    out[:, ROPE_DIM // 2:] = cpp_roped[:, 1:ROPE_DIM:2]
    return out


def pi_map_blocks(block: np.ndarray) -> np.ndarray:
    """Re-lift of the Amendment-1 nested helper (body verbatim; hoisted to
    module level). Operates on 64-wide RoPE blocks only: either one [.,64]
    block or the [N_TOKENS, N_HEADS*ROPE_DIM] RoPE-lane submatrix. It does
    NOT accept the full 2048-wide Q tensor."""
    if block.shape[1] == ROPE_DIM:
        return pi_map(block)
    b3d = block.reshape(N_TOKENS, N_HEADS, ROPE_DIM)
    out = np.empty_like(b3d)
    out[:, :, : ROPE_DIM // 2] = b3d[:, :, 0:ROPE_DIM:2]
    out[:, :, ROPE_DIM // 2:] = b3d[:, :, 1:ROPE_DIM:2]
    return out.reshape(N_TOKENS, N_HEADS * ROPE_DIM)


# ---------------- membership re-scoring (verbatim) ----------------

def forced_positions(p: int) -> np.ndarray:
    return np.concatenate(
        (
            np.arange(0, INIT_TOKENS, dtype=np.int64),
            np.arange(p - LOCAL_TOKENS + 1, p + 1, dtype=np.int64),
        )
    )


def rescore_row(
    p: int,
    q_row_f64: np.ndarray,        # [16, 128]
    weights_row_f64: np.ndarray,  # [16] (C++ post-scale weights)
    k_all_f32: np.ndarray,        # [2050, 128] variant post-rope K
) -> tuple[set[int], float, float]:
    """Returns (membership set, boundary margin, max abs finite score)."""
    k = k_all_f32[: p + 1].astype(np.float64)          # causal candidates
    dots = q_row_f64 @ k.T                             # [16, p+1]
    scores = weights_row_f64 @ np.maximum(dots, 0.0)   # [p+1]
    finite_max = float(np.max(np.abs(scores)))
    forced = forced_positions(p)
    masked = scores.copy()
    masked[forced] = np.inf
    order = np.argsort(-masked, kind="stable")
    selected = order[:TOPK]
    boundary_in = masked[order[TOPK - 1]]
    boundary_out = masked[order[TOPK]] if order.size > TOPK else -np.inf
    if np.isinf(boundary_in):
        margin = float("inf")
    else:
        margin = float(boundary_in - boundary_out)
    return set(int(x) for x in selected), margin, finite_max


# =====================================================================
# NEW REGISTERED P-CF1 COMPONENTS (design Revision 4). No new scientific
# arithmetic beyond the frozen scorer and the frozen f32 weight mapping.
# =====================================================================

# ---- registered full-Q lane mappings (assignment/permutation only) ----

def pi_full_q(q2d: np.ndarray) -> np.ndarray:
    """pi: C++ interleaved -> HF half-split, full Q [N_TOKENS, 16*128].
    Per head: dst[0:32]=src[0:64:2]; dst[32:64]=src[1:64:2];
    dst[64:128]=src[64:128]. Pure permutation; no arithmetic."""
    q3 = q2d.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
    out = np.empty_like(q3)
    out[:, :, 0 : ROPE_DIM // 2] = q3[:, :, 0:ROPE_DIM:2]
    out[:, :, ROPE_DIM // 2 : ROPE_DIM] = q3[:, :, 1:ROPE_DIM:2]
    out[:, :, ROPE_DIM:] = q3[:, :, ROPE_DIM:]
    return out.reshape(N_TOKENS, N_HEADS * HEAD_DIM)


def pi_inv_full_q(q2d: np.ndarray) -> np.ndarray:
    """pi^-1: HF half-split -> C++ interleaved, full Q [N_TOKENS, 16*128].
    Per head: dst[0:64:2]=src[0:32]; dst[1:64:2]=src[32:64];
    dst[64:128]=src[64:128]. Pure permutation; no arithmetic."""
    q3 = q2d.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
    out = np.empty_like(q3)
    out[:, :, 0:ROPE_DIM:2] = q3[:, :, 0 : ROPE_DIM // 2]
    out[:, :, 1:ROPE_DIM:2] = q3[:, :, ROPE_DIM // 2 : ROPE_DIM]
    out[:, :, ROPE_DIM:] = q3[:, :, ROPE_DIM:]
    return out.reshape(N_TOKENS, N_HEADS * HEAD_DIM)


# ---- P-CF1 reporting wrapper (set identity is the ONLY verdict source) ----

def tie_at_cut(margin: float) -> bool:
    """Exact float64 tie spanning the TOPK cut: rank-2047 score equals
    rank-2048 score, i.e. boundary margin exactly 0.0."""
    return margin == 0.0


def report_membership(
    banked: set[int], variant: set[int], margin: float, finite_max: float
) -> dict:
    """P-CF1 primary verdict from membership-SET identity ONLY, against the
    banked receiving-side owner00 set. Margin/floor/below-floor and the
    complete entering/dropped/symmetric-difference lists are OBSERVATIONAL
    ONLY and never modify the verdict. No BORDERLINE verdict exists in
    P-CF1 and no list is truncated."""
    diff = banked.symmetric_difference(variant)
    if not diff:
        v = "MEMBERSHIP-INVARIANT"
    else:
        v = f"MEMBERSHIP-AFFECTING ({len(diff)} positions)"
    floor = MARGIN_FLOOR_REL * finite_max
    return {
        "verdict": v,
        "sym_diff_size": len(diff),
        "sym_diff": sorted(int(x) for x in diff),
        "entering": sorted(int(x) for x in (variant - banked)),
        "dropped": sorted(int(x) for x in (banked - variant)),
        "observational": {
            "boundary_margin": margin,
            "margin_floor": floor,
            "below_floor": bool(margin < floor),
        },
    }


def tie_entry(
    banked: set[int], variant: set[int], margin: float, finite_max: float
) -> dict:
    """NOT ADJUDICATED (TIE-AT-CUT) branch/row entry carrying the COMPLETE
    frozen observational schema, derived from the deterministic registered
    offline reconstruction and clearly subordinate to the non-adjudicating
    verdict: the stable-argsort order grants no scientific or causal
    authority, and no diagnostic field modifies any verdict. No list is
    truncated."""
    rep = report_membership(banked, variant, margin, finite_max)
    return {
        "verdict": "NOT ADJUDICATED (TIE-AT-CUT)",
        "diagnostic_only": {
            "sym_diff_size": rep["sym_diff_size"],
            "sym_diff": rep["sym_diff"],
            "entering": rep["entering"],
            "dropped": rep["dropped"],
            "observational": rep["observational"],
        },
    }


# ---- CLASS-D semantic validation of an owner00 index row ----

def banked_row_failures(side: str, row_vals: np.ndarray, p: int) -> list[str]:
    """Direction-local SEMANTIC checks (CLASS D) on an otherwise well-formed
    integer index row at sparse-active row p. `side` is 'cpp', 'hf' or
    'recon' (reconstructed); container-level integrity (carrier integrality,
    shapes, SHAs) is CLASS G and happens earlier."""
    fails: list[str] = []
    vals = np.asarray(row_vals, dtype=np.int64)
    if int(vals.size) != TOPK:
        fails.append(f"row length {int(vals.size)} != TOPK {TOPK}")
    if side == "hf" and bool((vals == -1).any()):
        fails.append("HF -1 filler present at a sparse-active row")
    if bool((vals < 0).any()):
        fails.append(f"{int((vals < 0).sum())} negative indices present")
    if bool((vals > p).any()):
        fails.append(
            f"{int((vals > p).sum())} indices exceed causal candidate "
            f"range [0, {p}]"
        )
    s = set(int(x) for x in vals)
    if len(s) != int(vals.size):
        fails.append(
            f"duplicate indices: {int(vals.size) - len(s)} duplicates"
        )
    forced = set(int(x) for x in forced_positions(p))
    if not forced.issubset(s):
        fails.append(
            f"forced set not contained ({len(forced - s)} forced "
            f"positions missing)"
        )
    return fails


# ---- environment/backend gate (pure comparison helper + collector) ----

def environment_failures(obs: dict) -> list[str]:
    """Pure comparison of observed environment facts against the frozen
    binding. Testable with synthetic observations (no I/O here)."""
    fails: list[str] = []
    if os.path.normcase(os.path.normpath(obs.get("python_executable", ""))) != (
        os.path.normcase(os.path.normpath(REGISTERED_PYTHON))
    ):
        fails.append(
            f"interpreter {obs.get('python_executable')!r} != registered "
            f"{REGISTERED_PYTHON!r}"
        )
    if obs.get("python_version") != EXPECTED_PY_VERSION:
        fails.append(
            f"python version {obs.get('python_version')!r} != "
            f"{EXPECTED_PY_VERSION!r}"
        )
    if obs.get("numpy_version") != EXPECTED_NUMPY_VERSION:
        fails.append(
            f"numpy version {obs.get('numpy_version')!r} != "
            f"{EXPECTED_NUMPY_VERSION!r}"
        )
    cfg = obs.get("numpy_config")
    if cfg is None:
        fails.append("numpy.show_config(mode='dicts') unavailable")
    elif EXPECTED_BACKEND_SUBSTRING not in json.dumps(cfg).lower():
        fails.append(
            f"numpy config does not identify the "
            f"{EXPECTED_BACKEND_SUBSTRING!r} backend"
        )
    if obs.get("openblas_dll_exists") is not True:
        fails.append("registered OpenBLAS DLL missing")
    else:
        if obs.get("openblas_dll_bytes") != OPENBLAS_DLL_BYTES:
            fails.append(
                f"OpenBLAS DLL bytes {obs.get('openblas_dll_bytes')} != "
                f"{OPENBLAS_DLL_BYTES}"
            )
        if obs.get("openblas_dll_sha256") != OPENBLAS_DLL_SHA256:
            fails.append("OpenBLAS DLL SHA256 != frozen binding")
    for var in THREAD_ENV_VARS:
        if (obs.get("thread_env") or {}).get(var) != "1":
            fails.append(f"thread variable {var} != '1'")
    return fails


def collect_environment() -> dict:
    obs = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
        "thread_env_set_before_numpy_import": True,  # module-top assignment
    }
    try:
        obs["numpy_config"] = np.show_config(mode="dicts")
    except TypeError:
        obs["numpy_config"] = None
    dllp = Path(REGISTERED_OPENBLAS_DLL)
    obs["openblas_dll_path"] = str(dllp)
    obs["openblas_dll_exists"] = bool(dllp.is_file())
    if obs["openblas_dll_exists"]:
        obs["openblas_dll_bytes"] = int(dllp.stat().st_size)
        obs["openblas_dll_sha256"] = sha256_file(dllp)
    return obs


# ---- frozen late-abort result policy (CLASS G) ----

def apply_global_abort_policy(verdict: dict) -> int:
    """After CLASS G, no substituted result retains an adjudicated verdict:
    every already-computed substituted branch/row entry carrying a
    MEMBERSHIP-* verdict is rewritten to NOT ADJUDICATED (GLOBAL ABORT),
    with its raw measurements retained under diagnostic_only (no
    adjudicating authority). Returns the number of rewritten entries."""
    rewritten = 0
    for dres in (verdict.get("directions") or {}).values():
        for branch in (dres.get("branches") or {}).values():
            rows = branch.get("rows") or {}
            for key, entry in list(rows.items()):
                if isinstance(entry, dict) and str(
                    entry.get("verdict", "")
                ).startswith("MEMBERSHIP"):
                    rows[key] = {
                        "verdict": "NOT ADJUDICATED (GLOBAL ABORT)",
                        "diagnostic_only": entry,
                    }
                    rewritten += 1
    if rewritten:
        verdict["class_g_late_abort_rewritten_entries"] = rewritten
    return rewritten


# ---- mapping gate (CLASS G item: registered self-tests) ----

def _synthetic_unique_q() -> np.ndarray:
    """[N_TOKENS, N_HEADS*HEAD_DIM] f32 carrier whose every element is a
    distinct 32-bit pattern (values are irrelevant; identity is bitwise)."""
    n = N_TOKENS * N_HEADS * HEAD_DIM
    return (
        np.arange(n, dtype=np.uint32)
        .view("<f4")
        .reshape(N_TOKENS, N_HEADS * HEAD_DIM)
    )


def mapping_synthetic_failures() -> list[str]:
    """Synthetic structural self-tests for pi/pi^-1 (Stage-A safe: no file
    access, no captured data)."""
    fails: list[str] = []
    syn = _synthetic_unique_q()
    if not bits_equal(pi_inv_full_q(pi_full_q(syn)), syn):
        fails.append("synthetic pi^-1(pi(x)) != x")
    if not bits_equal(pi_full_q(pi_inv_full_q(syn)), syn):
        fails.append("synthetic pi(pi^-1(y)) != y")
    syn3 = syn.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
    for name, fn in (("pi", pi_full_q), ("pi_inv", pi_inv_full_q)):
        out3 = fn(syn).reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        if not bits_equal(out3[:, :, ROPE_DIM:], syn3[:, :, ROPE_DIM:]):
            fails.append(f"synthetic {name}: no-RoPE lanes 64..127 changed")
    rope_sub = syn3[:, :, :ROPE_DIM].reshape(N_TOKENS, N_HEADS * ROPE_DIM)
    full_rope = (
        pi_full_q(syn)
        .reshape(N_TOKENS, N_HEADS, HEAD_DIM)[:, :, :ROPE_DIM]
        .reshape(N_TOKENS, N_HEADS * ROPE_DIM)
    )
    if not bits_equal(pi_map_blocks(rope_sub), full_rope):
        fails.append("synthetic 64-lane core disagrees with pi_map_blocks")
    head0 = syn3[:, 0, :ROPE_DIM]
    out_head0 = pi_full_q(syn).reshape(N_TOKENS, N_HEADS, HEAD_DIM)[
        :, 0, :ROPE_DIM
    ]
    if not bits_equal(pi_map(head0), out_head0):
        fails.append("synthetic head-0 64-lane core disagrees with pi_map")
    return fails


def mapping_gate_captured(cpp_q2d: np.ndarray, hf_q2d: np.ndarray) -> dict:
    """Stage-E-only captured-tensor mapping self-tests (CLASS G on
    failure). Never called by --self-test."""
    res: dict = {}
    syn_fails = mapping_synthetic_failures()
    if syn_fails:
        stop(f"mapping synthetic self-test failed: {syn_fails}")
    res["synthetic"] = "PASS"
    if not bits_equal(pi_inv_full_q(pi_full_q(cpp_q2d)), cpp_q2d):
        stop("captured C++ Q: pi^-1(pi(x)) != x")
    if not bits_equal(pi_full_q(pi_inv_full_q(hf_q2d)), hf_q2d):
        stop("captured HF Q: pi(pi^-1(y)) != y")
    res["captured_round_trips"] = "PASS"
    for name, arr, fn in (
        ("cpp_pi", cpp_q2d, pi_full_q),
        ("hf_pi_inv", hf_q2d, pi_inv_full_q),
    ):
        a3 = arr.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        o3 = fn(arr).reshape(N_TOKENS, N_HEADS, HEAD_DIM)
        if not bits_equal(o3[:, :, ROPE_DIM:], a3[:, :, ROPE_DIM:]):
            stop(f"captured {name}: no-RoPE lanes 64..127 changed")
    res["captured_no_rope_invariance"] = "PASS"
    c3 = cpp_q2d.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
    rope_sub = c3[:, :, :ROPE_DIM].reshape(N_TOKENS, N_HEADS * ROPE_DIM)
    full_rope = (
        pi_full_q(cpp_q2d)
        .reshape(N_TOKENS, N_HEADS, HEAD_DIM)[:, :, :ROPE_DIM]
        .reshape(N_TOKENS, N_HEADS * ROPE_DIM)
    )
    if not bits_equal(pi_map_blocks(rope_sub), full_rope):
        stop("captured C++ Q: 64-lane core disagrees with pi_map_blocks")
    res["captured_64lane_core_vs_frozen_helper"] = "PASS"
    return res


# ---- scoring orchestration ----

def assert_finite_scores(context: str, finite_max: float) -> None:
    """Registered CLASS-G gate: `finite_max` is max(abs(scores)) over the
    PRE-mask score vector, so a NaN or +/-inf anywhere in the scores makes
    it non-finite. A non-finite value means the run is scientifically
    untrustworthy: CLASS G via stop(), never a branch- or direction-level
    result, and never an adjudicated membership verdict. No tolerance."""
    if not np.isfinite(finite_max):
        stop(
            f"non-finite pre-mask score vector (CLASS G) at {context}: "
            f"max|score| = {finite_max!r}"
        )


def score_row(
    context: str,
    q2d: np.ndarray,
    w2d: np.ndarray,
    k2d: np.ndarray,
    p: int,
) -> tuple[set[int], float, float]:
    """The ONLY scientific entry point to the frozen verbatim rescore_row:
    every baseline (CF-D0) and substituted (D1/D2/D3) score passes the
    non-finite CLASS-G gate BEFORE any tie classification or membership
    reporting. rescore_row itself is never modified."""
    q3 = q2d.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
    rset, margin, fmax = rescore_row(
        p, q3[p].astype(np.float64), w2d[p].astype(np.float64), k2d
    )
    assert_finite_scores(context, fmax)
    return rset, margin, fmax


def evaluate_direction(
    dname: str,
    side: str,
    q2d: np.ndarray,
    w2d: np.ndarray,
    k2d: np.ndarray,
    banked_ints: np.ndarray,
    branches: dict[str, tuple[np.ndarray, np.ndarray]],
    res: dict,
) -> None:
    """Runs CF-D0 (CLASS-D gate) and, if VALID, the three substituted
    branches of one receiving direction. All inputs already passed CLASS G.

    Late-abort retention contract: `res` MUST already be attached to
    verdict["directions"][dname] BEFORE this call; every branch container
    is attached to `res` BEFORE its row loop begins, and every completed
    row entry is inserted immediately after it is safely formed -- so a
    later CLASS-G failure finds all safely completed substituted results
    in place for the frozen late-abort policy. Nothing here catches and
    continues past a CLASS-G stop()."""
    res["receiving_frame"] = side
    res["baseline"] = {"rows": {}}
    res["invalid_clauses"] = []
    res["branches"] = {}
    banked_sets: dict[int, set[int]] = {}
    valid = True
    tie_in_baseline = False
    for p in ROWS:
        brow = banked_ints[p]
        bfails = banked_row_failures(side, brow, p)
        bset = set(int(x) for x in brow)
        banked_sets[p] = bset
        rset, margin, fmax = score_row(
            f"direction {dname} baseline CF-{dname}0 row {p}",
            q2d,
            w2d,
            k2d,
            p,
        )
        rarr = np.fromiter(sorted(rset), dtype=np.int64, count=len(rset))
        rfails = banked_row_failures("recon", rarr, p)
        tie = tie_at_cut(margin)
        equal = rset == bset
        floor = MARGIN_FLOOR_REL * fmax
        res["baseline"]["rows"][str(p)] = {
            "banked_failures": bfails,
            "reconstructed_failures": rfails,
            "tie_at_cut": bool(tie),
            "reproduces_banked": bool(equal),
            "sym_diff_size": len(rset.symmetric_difference(bset)),
            "observational": {
                "boundary_margin": margin,
                "margin_floor": floor,
                "below_floor": bool(margin < floor),
            },
        }
        for f in bfails:
            res["invalid_clauses"].append(f"row {p}: banked: {f}")
        for f in rfails:
            res["invalid_clauses"].append(f"row {p}: reconstructed: {f}")
        if tie:
            res["invalid_clauses"].append(f"row {p}: TIE-AT-CUT")
            tie_in_baseline = True
        if not equal:
            res["invalid_clauses"].append(
                f"row {p}: reconstructed != banked "
                f"(sym diff {len(rset.symmetric_difference(bset))})"
            )
        if bfails or rfails or tie or not equal:
            valid = False
    if valid:
        res["status"] = f"BASELINE RECONSTRUCTION VALID (direction {dname})"
    else:
        suffix = " (TIE-AT-CUT)" if tie_in_baseline else ""
        res["status"] = (
            f"BASELINE RECONSTRUCTION INVALID (direction {dname}){suffix}"
        )
    for bname, (bq, bw) in branches.items():
        rows: dict = {}
        res["branches"][bname] = {"rows": rows}
        for p in ROWS:
            if not valid:
                rows[str(p)] = {
                    "verdict": "NOT ADJUDICATED",
                    "reason": res["status"],
                }
                continue
            rset, margin, fmax = score_row(
                f"direction {dname} branch {bname} row {p}", bq, bw, k2d, p
            )
            if tie_at_cut(margin):
                entry = tie_entry(banked_sets[p], rset, margin, fmax)
            else:
                entry = report_membership(banked_sets[p], rset, margin, fmax)
            rows[str(p)] = entry


def joint_only_effect(dname: str, dres: dict) -> dict:
    """The ONLY aggregate descriptor: JOINT-ONLY MEMBERSHIP EFFECT
    (direction D) for exactly D1+D2 MEMBERSHIP-INVARIANT at all adjudicated
    rows while D3 is MEMBERSHIP-AFFECTING at any row, with a VALID baseline
    and no tie-unadjudicated branch/row in the pattern. Every other pattern
    is reported per-branch only, with no aggregate label."""
    out = {"assigned": False, "label": None}
    if not str(dres.get("status", "")).startswith(
        "BASELINE RECONSTRUCTION VALID"
    ):
        out["reason"] = "baseline not VALID"
        return out
    verdicts: dict[str, list[str]] = {}
    for bname, branch in (dres.get("branches") or {}).items():
        verdicts[bname] = [
            str(e.get("verdict", ""))
            for e in (branch.get("rows") or {}).values()
        ]
    names = sorted(verdicts)
    if len(names) != 3:
        out["reason"] = "branch set incomplete"
        return out
    b1, b2, b3 = names[0], names[1], names[2]
    all_adjudicated = all(
        v.startswith("MEMBERSHIP")
        for name in names
        for v in verdicts[name]
    )
    if not all_adjudicated:
        out["reason"] = "tie-unadjudicated branch/row in the pattern"
        return out
    inv1 = all(v == "MEMBERSHIP-INVARIANT" for v in verdicts[b1])
    inv2 = all(v == "MEMBERSHIP-INVARIANT" for v in verdicts[b2])
    aff3 = any(v.startswith("MEMBERSHIP-AFFECTING") for v in verdicts[b3])
    if inv1 and inv2 and aff3:
        out["assigned"] = True
        out["label"] = f"JOINT-ONLY MEMBERSHIP EFFECT (direction {dname})"
    else:
        out["reason"] = "outcome pattern is not the joint-only pattern"
    return out


# =====================================================================
# --self-test: completely synthetic/helper-only. Never touches the real
# capture bank, never hashes the real DLL, never reads any file.
# =====================================================================

def run_self_test() -> int:
    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, bool(ok)))
        print(f"SELF-TEST {'PASS' if ok else 'FAIL'}: {name}")

    # 1-2: pi/pi^-1 synthetic structural battery (round trips both
    # directions, no-RoPE invariance, 64-lane core vs frozen helpers).
    syn_fails = mapping_synthetic_failures()
    check("mapping synthetic battery", not syn_fails)
    if syn_fails:
        for f in syn_fails:
            print(f"  detail: {f}")

    # 3: bf16-lattice helper sanity.
    on_lat = widen_bits(np.array([0x3F80, 0x0001, 0x8000], dtype=np.uint16))
    off_lat = np.array([1.0000001], dtype="<f4")
    check(
        "bf16 lattice membership helper",
        bits_equal(rne_bf16(on_lat), on_lat)
        and not bits_equal(rne_bf16(off_lat), off_lat),
    )

    # 4: rescore_row structural semantics on synthetic data.
    rng = np.random.RandomState(20260819)
    q = rng.standard_normal((N_TOKENS, N_HEADS * HEAD_DIM)).astype("<f4")
    k = rng.standard_normal((N_TOKENS, HEAD_DIM)).astype("<f4")
    w = (np.abs(rng.standard_normal((N_TOKENS, N_HEADS))) + 0.1).astype(
        "<f4"
    )
    p = ROWS[0]
    base_set, base_margin, base_fmax = score_row(
        "self-test synthetic baseline", q, w, k, p
    )
    forced = set(int(x) for x in forced_positions(p))
    check(
        "baseline structural (size/range/forced)",
        len(base_set) == TOPK
        and min(base_set) >= 0
        and max(base_set) <= p
        and forced.issubset(base_set),
    )
    excluded = sorted(set(range(p + 1)) - base_set)
    check("exactly one excluded candidate at row 2048", len(excluded) == 1)

    # 5: reporting wrapper -- invariant case.
    inv = report_membership(base_set, set(base_set), base_margin, base_fmax)
    check(
        "report_membership invariant",
        inv["verdict"] == "MEMBERSHIP-INVARIANT"
        and inv["sym_diff"] == []
        and inv["entering"] == []
        and inv["dropped"] == []
        and isinstance(inv["observational"]["below_floor"], bool),
    )

    # 6: reporting wrapper -- affecting case via a deterministic K boost.
    ok_target = False
    if len(excluded) == 1:
        target = excluded[0]
        unforced_sel = sorted(base_set - forced)
        scores_ok = False
        if unforced_sel:
            q3 = q.reshape(N_TOKENS, N_HEADS, HEAD_DIM)
            qi = q3[p].astype(np.float64)
            wi = w[p].astype(np.float64)

            def one_score(j: int) -> float:
                d = qi @ k[j].astype(np.float64)
                return float(wi @ np.maximum(d, 0.0))

            top_unforced = max(unforced_sel, key=one_score)
            scores_ok = one_score(top_unforced) > 0.0
        check("synthetic boost precondition", scores_ok)
        if scores_ok:
            k2 = k.copy()
            k2[target] = (k[top_unforced].astype(np.float64) * 2.0).astype(
                "<f4"
            )
            var_set, var_margin, var_fmax = score_row(
                "self-test synthetic boost", q, w, k2, p
            )
            rep = report_membership(base_set, var_set, var_margin, var_fmax)
            ok_target = (
                rep["verdict"] == "MEMBERSHIP-AFFECTING (2 positions)"
                and target in rep["entering"]
                and len(rep["dropped"]) == 1
                and rep["sym_diff"]
                == sorted(rep["entering"] + rep["dropped"])
            )
    check("report_membership affecting (complete lists)", ok_target)

    # 7: tie classification helper.
    check(
        "tie_at_cut helper",
        tie_at_cut(0.0)
        and not tie_at_cut(5e-324)
        and not tie_at_cut(float("inf")),
    )

    # 8: banked-row semantic validator.
    good = np.array(sorted(set(range(p + 1)) - {16}), dtype=np.int64)
    bad_filler = good.copy()
    bad_filler[0] = -1
    bad_dup = good.copy()
    bad_dup[0] = int(bad_dup[1])
    bad_range = good.copy()
    bad_range[0] = p + 1000
    bad_forced = np.array(
        sorted(set(range(p + 1)) - {0}), dtype=np.int64
    )
    check(
        "banked_row_failures semantics",
        not banked_row_failures("cpp", good, p)
        and any(
            "filler" in f
            for f in banked_row_failures("hf", bad_filler, p)
        )
        and any(
            "duplicate" in f
            for f in banked_row_failures("cpp", bad_dup, p)
        )
        and any(
            "exceed" in f
            for f in banked_row_failures("cpp", bad_range, p)
        )
        and any(
            "forced" in f
            for f in banked_row_failures("cpp", bad_forced, p)
        ),
    )

    # 9: environment-gate helper on synthetic observations (no I/O).
    good_obs = {
        "python_executable": REGISTERED_PYTHON,
        "python_version": EXPECTED_PY_VERSION,
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "numpy_config": {
            "Build Dependencies": {"blas": {"name": "scipy-openblas"}}
        },
        "openblas_dll_exists": True,
        "openblas_dll_bytes": OPENBLAS_DLL_BYTES,
        "openblas_dll_sha256": OPENBLAS_DLL_SHA256,
        "thread_env": {v: "1" for v in THREAD_ENV_VARS},
    }
    bad_obs = dict(good_obs)
    bad_obs["numpy_version"] = "2.5.1"
    check(
        "environment_failures helper",
        not environment_failures(good_obs)
        and any(
            "numpy version" in f for f in environment_failures(bad_obs)
        ),
    )

    # 10: frozen late-abort policy on a synthetic verdict.
    synth_verdict = {
        "directions": {
            "C": {
                "branches": {
                    "1_weights_only": {
                        "rows": {
                            "2048": {
                                "verdict": "MEMBERSHIP-INVARIANT",
                                "sym_diff_size": 0,
                            }
                        }
                    }
                }
            }
        }
    }
    n_rw = apply_global_abort_policy(synth_verdict)
    entry = synth_verdict["directions"]["C"]["branches"]["1_weights_only"][
        "rows"
    ]["2048"]
    check(
        "apply_global_abort_policy",
        n_rw == 1
        and entry["verdict"] == "NOT ADJUDICATED (GLOBAL ABORT)"
        and entry["diagnostic_only"]["verdict"] == "MEMBERSHIP-INVARIANT",
    )

    # 11: joint-only pattern helper on synthetic direction results.
    def synth_dir(v1: str, v2: str, v3: str) -> dict:
        return {
            "status": "BASELINE RECONSTRUCTION VALID (direction X)",
            "branches": {
                "1_weights_only": {"rows": {"2048": {"verdict": v1}}},
                "2_q_only": {"rows": {"2048": {"verdict": v2}}},
                "3_both": {"rows": {"2048": {"verdict": v3}}},
            },
        }

    jo_yes = joint_only_effect(
        "X",
        synth_dir(
            "MEMBERSHIP-INVARIANT",
            "MEMBERSHIP-INVARIANT",
            "MEMBERSHIP-AFFECTING (2 positions)",
        ),
    )
    jo_no = joint_only_effect(
        "X",
        synth_dir(
            "MEMBERSHIP-AFFECTING (2 positions)",
            "MEMBERSHIP-INVARIANT",
            "MEMBERSHIP-INVARIANT",
        ),
    )
    jo_tie = joint_only_effect(
        "X",
        synth_dir(
            "MEMBERSHIP-INVARIANT",
            "NOT ADJUDICATED (TIE-AT-CUT)",
            "MEMBERSHIP-AFFECTING (2 positions)",
        ),
    )
    check(
        "joint_only_effect pattern logic",
        jo_yes["assigned"]
        and not jo_no["assigned"]
        and not jo_tie["assigned"],
    )

    # 12: non-finite pre-mask-score CLASS-G gate (helper level, synthetic).
    n_reasons0 = len(REASONS)
    gate_finite_ok = True
    try:
        assert_finite_scores("self-test finite", 1.0)
    except SystemExit:
        gate_finite_ok = False
    gate_trips = 0
    for badval in (float("nan"), float("inf"), float("-inf")):
        try:
            assert_finite_scores("self-test non-finite", badval)
        except SystemExit:
            gate_trips += 1
    context_ok = all(
        "non-finite pre-mask score vector" in r
        for r in REASONS[n_reasons0:]
    )
    del REASONS[n_reasons0:]
    check(
        "assert_finite_scores CLASS-G gate",
        gate_finite_ok and gate_trips == 3 and context_ok,
    )

    # 13: complete tie-entry observational schema, subordinate to the
    # NOT ADJUDICATED verdict.
    tset = set(range(TOPK))
    tvar = (tset - {5}) | {9999}
    te = tie_entry(tset, tvar, 0.0, 10.0)
    dgo = te.get("diagnostic_only") or {}
    check(
        "tie_entry complete schema (subordinate to NOT ADJUDICATED)",
        te["verdict"] == "NOT ADJUDICATED (TIE-AT-CUT)"
        and dgo.get("sym_diff_size") == 2
        and dgo.get("sym_diff") == [5, 9999]
        and dgo.get("entering") == [9999]
        and dgo.get("dropped") == [5]
        and dgo.get("observational", {}).get("boundary_margin") == 0.0
        and dgo.get("observational", {}).get("below_floor") is True
        and "verdict" not in dgo,
    )

    # 14: pre-attached orchestration retention under a simulated late
    # CLASS-G abort (fully synthetic direction evaluation; no file access).
    q_s = rng.standard_normal((N_TOKENS, N_HEADS * HEAD_DIM)).astype("<f4")
    k_s = rng.standard_normal((N_TOKENS, HEAD_DIM)).astype("<f4")
    w_s = (np.abs(rng.standard_normal((N_TOKENS, N_HEADS))) + 0.1).astype(
        "<f4"
    )
    bank_s = np.zeros((N_TOKENS, TOPK), dtype=np.int64)
    for p_s in ROWS:
        sset, _sm, _sf = score_row(
            "self-test synthetic bank", q_s, w_s, k_s, p_s
        )
        bank_s[p_s] = np.fromiter(sorted(sset), dtype=np.int64, count=TOPK)
    verdict_s: dict = {"directions": {}}
    res_s: dict = {}
    verdict_s["directions"]["X"] = res_s
    evaluate_direction(
        "X",
        "cpp",
        q_s,
        w_s,
        k_s,
        bank_s,
        {
            "1_weights_only": (q_s, w_s),
            "2_q_only": (q_s, w_s),
            "3_both": (q_s, w_s),
        },
        res_s,
    )
    attached = verdict_s["directions"]["X"] is res_s
    base_valid = str(res_s.get("status", "")).startswith(
        "BASELINE RECONSTRUCTION VALID"
    )
    pre_entry = (
        res_s.get("branches", {})
        .get("1_weights_only", {})
        .get("rows", {})
        .get("2048", {})
    )
    pre_ok = pre_entry.get("verdict") == "MEMBERSHIP-INVARIANT"
    n_rw2 = apply_global_abort_policy(verdict_s)
    post_entry = verdict_s["directions"]["X"]["branches"][
        "1_weights_only"
    ]["rows"]["2048"]
    check(
        "pre-attached rows survive simulated late CLASS-G abort",
        attached
        and base_valid
        and pre_ok
        and n_rw2 == 6
        and post_entry["verdict"] == "NOT ADJUDICATED (GLOBAL ABORT)"
        and post_entry["diagnostic_only"]["verdict"]
        == "MEMBERSHIP-INVARIANT",
    )

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"SELF-TEST SUMMARY: {len(results) - n_fail}/{len(results)} PASS")
    return 0 if n_fail == 0 else 1


# =====================================================================
# main
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-s-dir")
    ap.add_argument("--hf-dir")
    ap.add_argument("--hf-run-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--self-test", action="store_true")
    ns = ap.parse_args()

    if ns.self_test:
        return run_self_test()
    if not (ns.cpp_s_dir and ns.hf_dir and ns.hf_run_dir and ns.out_dir):
        print(
            "STOP: --cpp-s-dir, --hf-dir, --hf-run-dir and --out-dir are "
            "required"
        )
        return 1

    # CLASS R -- checked after argument parsing and --out-dir resolution,
    # BEFORE reading or hash-checking any scientific input and before the
    # registered path contract (so a sentinel refusal test never touches
    # scientific inputs). The existing directory is left untouched.
    out_dir = Path(ns.out_dir).resolve()
    if out_dir.exists():
        print(f"REFUSED-TO-START: out dir already exists: {out_dir}")
        print("(an earlier result is never overwritten; exit 3)")
        return 3

    # Registered Stage-E absolute path contract (non-scientific stop;
    # nothing is created, no verdict is written).
    cpp_dir = Path(ns.cpp_s_dir).resolve()
    hf_dir = Path(ns.hf_dir).resolve()
    hf_run_dir = Path(ns.hf_run_dir).resolve()
    contract = {
        "--cpp-s-dir": (cpp_dir, Path(REGISTERED_CPP_S_DIR)),
        "--hf-dir": (hf_dir, Path(REGISTERED_HF_DIR)),
        "--hf-run-dir": (hf_run_dir, Path(REGISTERED_HF_RUN_DIR)),
        "--out-dir": (out_dir, Path(REGISTERED_OUT_DIR)),
    }
    bad_paths = [
        f"{flag}: {got} != registered {want}"
        for flag, (got, want) in contract.items()
        if os.path.normcase(str(got)) != os.path.normcase(str(want))
    ]
    if bad_paths:
        print("STOP: registered Stage-E path contract violated:")
        for b in bad_paths:
            print(f"  {b}")
        return 1

    out_dir.mkdir(parents=False, exist_ok=False)

    verdict: dict = {
        "protocol": (
            "P-CF1 v1 -- Captured-Surface Offline Counterfactual "
            "Membership Sensitivity (owner00; rows 2048/2049; frozen "
            "design Revision 4). Offline-reconstruction authority ONLY: "
            "no production-level causal claim; baseline reproduction is "
            "necessary, not sufficient, for production authority. "
            "NOT a Gate-4 criterion; Gate 4 remains NOT RUN."
        ),
        "paths": {
            "cpp_s_dir": str(cpp_dir),
            "hf_dir": str(hf_dir),
            "hf_run_dir": str(hf_run_dir),
            "out_dir": str(out_dir),
        },
        "reasons": REASONS,
    }

    def finish(code: int) -> int:
        (out_dir / "verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"verdict_json={out_dir / 'verdict.json'}")
        print(f"exit={code} reasons={REASONS}")
        return code

    try:
        # ========= CLASS G -- environment/backend binding =========
        env = collect_environment()
        verdict["environment"] = env
        env_fails = environment_failures(env)
        if env_fails:
            stop(f"environment/backend binding failure: {env_fails}")

        # ========= CLASS G -- frozen evidence-chain bindings =========
        integrity: dict = {}
        verdict["integrity"] = integrity
        repo = Path(__file__).resolve().parent
        self_sha = sha256_file(Path(__file__).resolve())
        integrity["cf1_self_sha256"] = self_sha
        evidence_bindings = {
            "parent_comparator": (
                repo / "analyze_longcat_lsa_hf_cpp_blockers_2050.py",
                PARENT_CMP_SHA256,
            ),
            "original_failed_verdict": (
                repo / "lsa_hf_blockers_2050" / "verdict.json",
                ORIGINAL_FAILED_VERDICT_SHA256,
            ),
            "diag_v1": (
                repo / "lsa_hf_blockers_2050_rope_diag_v1" / "diagnostic.json",
                DIAG_V1_SHA256,
            ),
            "diag_v2": (
                repo / "lsa_hf_blockers_2050_rope_diag_v2" / "diagnostic.json",
                DIAG_V2_SHA256,
            ),
            "diag_v3": (
                repo / "lsa_hf_blockers_2050_rope_diag_v3" / "diagnostic.json",
                DIAG_V3_SHA256,
            ),
            "amend1_comparator": (
                repo / "analyze_longcat_lsa_hf_cpp_blockers_2050_amend1.py",
                AMEND1_CMP_SHA256,
            ),
            "amend1_verdict": (
                repo / "lsa_hf_blockers_2050_amend1" / "verdict.json",
                AMEND1_VERDICT_SHA256,
            ),
        }
        for what, (pth, want) in evidence_bindings.items():
            if not pth.is_file():
                stop(f"evidence binding missing: {what}: {pth}")
            got = sha256_file(pth)
            if got != want:
                stop(
                    f"evidence binding SHA mismatch: {what}: {got} != {want}"
                )
        integrity["evidence_chain"] = {
            what: want for what, (_p, want) in evidence_bindings.items()
        }

        # ---- C++ manifest + provenance binding ----
        cpp_sums_path = cpp_dir / "SHA256SUMS.txt"
        cpp_sums = parse_sums(cpp_sums_path)
        integrity["cpp_manifest_sha256"] = sha256_file(cpp_sums_path)
        if integrity["cpp_manifest_sha256"] != CPP_MANIFEST_SHA256:
            stop("C++ manifest SHA != frozen constant")
        cpp_needed = list(CPP_STABLE_SHA256) + [
            f"lsa_top_k_owner{2 * li:02d}_full.bin" for li in range(14)
        ]
        for name in cpp_needed:
            pth = cpp_dir / name
            if not pth.is_file():
                stop(f"C++ artifact missing: {pth}")
            if name not in cpp_sums:
                stop(f"C++ manifest lacks entry for {name}")
            got = sha256_file(pth)
            if got != cpp_sums[name]:
                stop(f"C++ manifest rehash mismatch: {name}")
            if name in CPP_STABLE_SHA256 and got != CPP_STABLE_SHA256[name]:
                stop(
                    f"C++ surface {name} SHA {got} != frozen byte-stable "
                    f"{CPP_STABLE_SHA256[name]}"
                )
            if (
                name == "lsa_top_k_owner00_full.bin"
                and got != CPP_TOPK_OWNER00_SHA256
            ):
                stop("C++ owner00 bank SHA != frozen P-CF1 constant")
        cpp_prov = read_json(cpp_dir / "run_provenance.json")
        if cpp_prov.get("git_head") != EXPECTED_CPP_GIT_HEAD:
            stop(
                "C++ provenance git_head "
                f"{cpp_prov.get('git_head')} != {EXPECTED_CPP_GIT_HEAD}"
            )
        if (
            cpp_prov.get("token_stream_sha256_reconstructed")
            != EXPECTED_TOKEN_SHA256
        ):
            stop("C++ provenance token stream SHA mismatch")
        integrity["cpp_provenance_git_head"] = cpp_prov.get("git_head")

        # ---- HF runner-provenance chain (exactly as Amendment 1 gates it;
        # the capture stays bound to the ORIGINAL parent comparator and is
        # never rewritten to point at P-CF1) ----
        hf_prov_path = hf_run_dir / "run_provenance.json"
        hf_prov = read_json(hf_prov_path)
        integrity["hf_run_provenance_sha256"] = sha256_file(hf_prov_path)

        if hf_prov.get("a_equals_b") is not True:
            stop("HF provenance a_equals_b is not True")
        if hf_prov.get("engagement_proof") != "PASS":
            stop("HF provenance engagement_proof != PASS")
        if hf_prov.get("tokens_bin_sha256") != EXPECTED_TOKEN_SHA256:
            stop("HF provenance token stream SHA mismatch")
        if hf_prov.get("core_script_sha256") != EXPECTED_CORE_SHA256:
            stop("HF provenance frozen-core SHA mismatch")
        if hf_prov.get("runB_script_sha256") != EXPECTED_RUNB_SCRIPT_SHA256:
            stop("HF provenance Run-B script SHA != protocol constant")
        if hf_prov.get("runA_script_sha256") != EXPECTED_RUNA_SCRIPT_SHA256:
            stop("HF provenance Run-A script SHA != protocol constant")
        integrity["cmp_parent_sha256"] = PARENT_CMP_SHA256
        if hf_prov.get("cmp_script_sha256") != PARENT_CMP_SHA256:
            stop(
                "HF provenance cmp_script_sha256 != frozen PARENT "
                "comparator SHA (the capture was bound to the parent "
                "comparator)"
            )
        runner_path = repo / "run_longcat_hf_lsa_2050_capture.ps1"
        if not runner_path.is_file():
            stop(
                "runner script missing for provenance reverification: "
                f"{runner_path}"
            )
        runner_sha = sha256_file(runner_path)
        integrity["runner_script_sha256"] = runner_sha
        if hf_prov.get("runner_script_sha256") != runner_sha:
            stop(
                "HF provenance runner_script_sha256 != actual runner "
                "script hash (protocol drift between capture and analysis)"
            )

        canon_bin = hf_run_dir / "hf_sparse_2050_v1.bin"
        canon_json_path = hf_run_dir / "hf_sparse_2050_v1.json"
        proof_path = hf_run_dir / "sparse_engagement_proof.json"
        for pth in (canon_bin, canon_json_path, proof_path):
            if not pth.is_file():
                stop(f"HF run artifact missing: {pth}")
        canon_sha = sha256_file(canon_bin)
        if hf_prov.get("canonical_logits_sha256") != canon_sha:
            stop("HF provenance canonical logits SHA != actual Run-B bin")
        if hf_prov.get("runB_core_json_sha256") != sha256_file(
            canon_json_path
        ):
            stop("HF provenance runB_core_json_sha256 != actual core json")
        if hf_prov.get("runB_engagement_proof_sha256") != sha256_file(
            proof_path
        ):
            stop("HF provenance runB_engagement_proof_sha256 != actual proof")
        runa_logits_path = hf_dir / "hf_logits_2050_runA.bin"
        if not runa_logits_path.is_file():
            stop(f"HF Run-A logits missing: {runa_logits_path}")
        runa_logits_sha = sha256_file(runa_logits_path)
        if hf_prov.get("runA_logits_sha256") != runa_logits_sha:
            stop("HF provenance runA_logits_sha256 != actual Run-A bin")
        if runa_logits_sha != canon_sha:
            stop(
                "Run-A logits != canonical Run-B logits at analysis time "
                "(A==B violated)"
            )
        integrity["canonical_logits_sha256"] = canon_sha

        core_json = read_json(canon_json_path)
        if core_json.get("logits_bin_sha256") != canon_sha:
            stop("Run-B core json logits_bin_sha256 != actual canonical bin")
        if core_json.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256:
            stop("Run-B core json frozen runtime SHA mismatch")

        proof = read_json(proof_path)
        if proof.get("engagement_proof") != "PASS":
            stop("Run-B engagement proof contents != PASS")
        if proof.get("core_rc") != 0:
            stop(f"Run-B engagement proof core_rc {proof.get('core_rc')} != 0")
        if proof.get("original_capture_sha256") != EXPECTED_CORE_SHA256:
            stop("Run-B engagement proof frozen-core SHA mismatch")
        if proof.get("tokens_bin_sha256") != EXPECTED_TOKEN_SHA256:
            stop("Run-B engagement proof token SHA mismatch")
        if proof.get("failures"):
            stop(f"Run-B engagement proof lists failures: {proof['failures']}")
        coll = proof.get("collector") or []
        if len(coll) != 28:
            stop(f"Run-B engagement collector has {len(coll)} records != 28")
        seen_ls: set = set()
        for rec in coll:
            if "error" in rec:
                stop(f"Run-B engagement record error: {rec['error']}")
            key = (rec.get("layer"), rec.get("sublayer"))
            if key in seen_ls:
                stop(f"Run-B engagement duplicate record {key}")
            seen_ls.add(key)
            want_mode = (
                "sparse-owner" if rec.get("sublayer") == 0 else "sparse-reuse"
            )
            if rec.get("mode") != want_mode:
                stop(f"Run-B engagement record {key} mode {rec.get('mode')!r}")
        if seen_ls != {(i, s) for i in range(14) for s in (0, 1)}:
            stop("Run-B engagement collector does not cover all 28 sublayers")
        bat = next(
            (
                r.get("owner0_battery")
                for r in coll
                if r.get("layer") == 0 and r.get("sublayer") == 0
            ),
            None,
        )
        if not bat:
            stop("Run-B engagement proof lacks the owner0 battery")
        for key in (
            "range_ok",
            "filler_counts_exact",
            "fillers_2047_2049_zero",
            "unique_nonneg",
            "causal_only",
        ):
            if bat.get(key) is not True:
                stop(f"Run-B engagement owner0 battery {key} is not True")
        for pkey in ("2048", "2049"):
            if (bat.get("forced_containment") or {}).get(pkey) is not True:
                stop(f"Run-B engagement forced containment row {pkey} failed")

        for run_key in ("runB", "runA"):
            info = hf_prov.get(run_key) or {}
            for stream in ("stdout", "stderr"):
                lp = info.get(f"{stream}_log")
                lsha = info.get(f"{stream}_log_sha256")
                lbytes = info.get(f"{stream}_log_bytes")
                if not lp or not lsha or lbytes is None:
                    stop(f"HF provenance {run_key} {stream} log binding missing")
                pth = Path(lp)
                if not pth.is_file():
                    stop(f"HF provenance-bound log missing: {pth}")
                if pth.stat().st_size != int(lbytes):
                    stop(f"HF provenance-bound log size mismatch: {pth}")
                if sha256_file(pth) != lsha:
                    stop(f"HF provenance-bound log SHA mismatch: {pth}")
        integrity["hf_log_bindings"] = "4/4 rehash OK"

        # ---- HF manifest: exact expected inventory, every entry rehashed ----
        hf_sums_path = hf_dir / "SHA256SUMS.txt"
        hf_sums = parse_sums(hf_sums_path)
        integrity["hf_manifest_sha256"] = sha256_file(hf_sums_path)
        if integrity["hf_manifest_sha256"] != HF_MANIFEST_SHA256:
            stop("HF manifest SHA != frozen constant")
        hf_expected_names = (
            list(HF_WIDTHS)
            + [f"hf_top_k_owner{2 * li:02d}.bin" for li in range(14)]
            + list(HF_WEIGHT_BINS)
            + ["hf_logits_2050_runA.bin", "summary.json"]
        )
        if set(hf_sums) != set(hf_expected_names):
            missing = sorted(set(hf_expected_names) - set(hf_sums))
            extra = sorted(set(hf_sums) - set(hf_expected_names))
            stop(
                f"HF manifest inventory mismatch: missing={missing} "
                f"extra={extra}"
            )
        for name, sha in hf_sums.items():
            pth = hf_dir / name
            if not pth.is_file():
                stop(f"HF artifact missing: {pth}")
            if sha256_file(pth) != sha:
                stop(f"HF manifest rehash mismatch: {name}")
        for name, want in HF_INPUT_SHA256.items():
            if hf_sums.get(name) != want:
                stop(f"HF input {name} SHA != frozen P-CF1 constant")
        if hf_prov.get("runA_manifest_sha256") != integrity[
            "hf_manifest_sha256"
        ]:
            stop("HF provenance runA_manifest_sha256 != actual SHA256SUMS.txt")
        if hf_prov.get("runA_summary_sha256") != hf_sums["summary.json"]:
            stop("HF provenance runA_summary_sha256 != actual summary.json")

        hf_summary = read_json(hf_dir / "summary.json")
        integrity["hf_summary_sha256"] = hf_sums["summary.json"]
        gates = hf_summary.get("gates", {})
        if gates.get("runA_logits_byte_equal_runB") is not True:
            stop("HF summary gate runA_logits_byte_equal_runB is not True")
        if hf_summary.get("tokens_bin_sha256") != EXPECTED_TOKEN_SHA256:
            stop("HF summary token stream SHA mismatch")
        if hf_summary.get("runtime_sha256") != EXPECTED_RUNTIME_SHA256:
            stop("HF summary frozen runtime SHA mismatch")
        verdict["integrity"] = integrity

        # ========= CLASS G -- load the eight SS2 scoring inputs =========
        cpp: dict[str, np.ndarray] = {
            name: load_f32(cpp_dir / name, CPP_WIDTHS[name])
            for name in CPP_LOADED
        }
        hf: dict[str, np.ndarray] = {
            name: load_f32(hf_dir / name, HF_WIDTHS[name])
            for name in HF_LOADED
        }
        cpp_bank_arr = load_f32(
            cpp_dir / "lsa_top_k_owner00_full.bin", TOPK
        )
        cpp_bank = cpp_bank_arr.astype(np.int64)
        if not np.array_equal(cpp_bank.astype("<f4"), cpp_bank_arr):
            stop("lsa_top_k_owner00_full.bin: values not exactly integral")
        hf_bank_arr = load_f32(hf_dir / "hf_top_k_owner00.bin", TOPK)
        hf_bank = hf_bank_arr.astype(np.int64)
        if not np.array_equal(hf_bank.astype("<f4"), hf_bank_arr):
            stop("hf_top_k_owner00.bin: values not exactly integral")

        # ---- CLASS G container/structural gates on the loaded surfaces ----
        lattice_gated = {
            "cpp lsa_indexer_q_full.bin": cpp["lsa_indexer_q_full.bin"],
            "cpp lsa_indexer_k_full.bin": cpp["lsa_indexer_k_full.bin"],
            "hf hf_indexer_q.bin": hf["hf_indexer_q.bin"],
            "hf hf_indexer_k.bin": hf["hf_indexer_k.bin"],
        }
        for name, arr in lattice_gated.items():
            bad = bits_diff_count(rne_bf16(arr), arr)
            if bad:
                stop(f"{name}: {bad} values off the BF16 lattice")
        finite_gated = dict(lattice_gated)
        finite_gated["cpp lsa_indexer_weights_full.bin"] = cpp[
            "lsa_indexer_weights_full.bin"
        ]
        finite_gated["hf hf_indexer_weights_prescale.bin"] = hf[
            "hf_indexer_weights_prescale.bin"
        ]
        for name, arr in finite_gated.items():
            if not bool(np.isfinite(arr).all()):
                stop(f"{name}: non-finite values present")

        # ---- CLASS G registered mapping self-tests (incl. captured) ----
        verdict["mapping_gate"] = mapping_gate_captured(
            cpp["lsa_indexer_q_full.bin"], hf["hf_indexer_q.bin"]
        )

        # ========= effective weights (frozen surface-5 mapping) =========
        scale_c = f32(1.0) / np.sqrt(f32(2048.0), dtype=np.float32)
        w_cpp_eff = cpp["lsa_indexer_weights_full.bin"]
        w_hf_eff = (
            hf["hf_indexer_weights_prescale.bin"] * scale_c
        ).astype(np.float32)

        q_cpp = cpp["lsa_indexer_q_full.bin"]
        q_hf = hf["hf_indexer_q.bin"]
        k_cpp = cpp["lsa_indexer_k_full.bin"]
        k_hf = hf["hf_indexer_k.bin"]
        q_hf_in_cpp_layout = pi_inv_full_q(q_hf)
        q_cpp_in_hf_layout = pi_full_q(q_cpp)

        # ========= CLASS D gates + branches, per direction =========
        # Late-abort retention: each direction container is attached to
        # the verdict BEFORE its evaluation begins, so a later CLASS-G
        # abort preserves every safely completed entry for the frozen
        # late-abort policy.
        verdict["directions"] = {}
        res_c: dict = {}
        verdict["directions"]["C"] = res_c
        evaluate_direction(
            "C",
            "cpp",
            q_cpp,
            w_cpp_eff,
            k_cpp,
            cpp_bank,
            {
                "1_weights_only": (q_cpp, w_hf_eff),
                "2_q_only": (q_hf_in_cpp_layout, w_cpp_eff),
                "3_both": (q_hf_in_cpp_layout, w_hf_eff),
            },
            res_c,
        )
        res_h: dict = {}
        verdict["directions"]["H"] = res_h
        evaluate_direction(
            "H",
            "hf",
            q_hf,
            w_hf_eff,
            k_hf,
            hf_bank,
            {
                "1_weights_only": (q_hf, w_cpp_eff),
                "2_q_only": (q_cpp_in_hf_layout, w_hf_eff),
                "3_both": (q_cpp_in_hf_layout, w_cpp_eff),
            },
            res_h,
        )
        verdict["joint_only_membership_effect"] = {
            dname: joint_only_effect(dname, dres)
            for dname, dres in verdict["directions"].items()
        }

        verdict["anomaly"] = None
        return finish(0)
    except SystemExit as exc:
        verdict["anomaly"] = str(exc)
        apply_global_abort_policy(verdict)
        return finish(1)
    except Exception:
        REASONS.append("unexpected exception (see anomaly)")
        verdict["anomaly"] = (
            "UNEXPECTED EXCEPTION:\n" + traceback.format_exc()
        )
        apply_global_abort_policy(verdict)
        return finish(1)


if __name__ == "__main__":
    raise SystemExit(main())
